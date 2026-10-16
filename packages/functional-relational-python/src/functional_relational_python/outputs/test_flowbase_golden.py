"""Golden tests that rebuild two of Flowbase's hydrated views with this library.

The Entities are minimal versions of Flowbase's `*Db` models: their IDs and
References, plus a few plain fields. Flowbase's shared fields (`created_at`,
`is_deleted`, ...) are left out, because they add nothing a View treats differently.

- `CompanyMembershipHydrated` (`apps/backend/djank/hydrated_models/
  company_membership.py`) resolves a membership's user and company role, and keeps
  `company_id` as an ID. It is a single-level View.
- `ShopOrderHydrated` (`apps/backend/djank/hydrated_models/shop_order.py`) resolves
  an order's department, and its company membership hydrated as above. It is a
  nested View. Flowbase's version also holds the order's line items, which are
  to-many, and their total amount; both are left out.
"""

import datetime
import enum

import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler
from functional_relational_python.domain.entity_graph import EntityGraph, FieldName
from functional_relational_python.domain.id import Id
from functional_relational_python.language import ir, steps
from functional_relational_python.language.views import View, ViewName
from functional_relational_python.outputs import engine
from functional_relational_python.outputs.python.plan import PythonLayout, PythonPlan
from functional_relational_python.outputs.python.render import PythonRenderer
from functional_relational_python.outputs.python.symbols import ModulePath
from functional_relational_python.support.filesystem import FileText


class Text(pydantic.RootModel[str], frozen=True):
    pass


class LastUsedAt(pydantic.RootModel[datetime.datetime], frozen=True):
    @staticmethod
    def fake() -> LastUsedAt:
        return LastUsedAt(datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC))


class PlacedAt(pydantic.RootModel[datetime.datetime], frozen=True):
    @staticmethod
    def fake() -> PlacedAt:
        return PlacedAt(datetime.datetime(2026, 3, 4, tzinfo=datetime.UTC))


class CompanyId(Id[int]):
    pass


class Company(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text

    @staticmethod
    def fake() -> Company:
        return Company(id=CompanyId(1), name=Text("Acme"))


class UserId(Id[int]):
    pass


class User(pydantic.BaseModel, frozen=True):
    id: UserId
    email: Text
    first_name: Text
    last_name: Text

    @staticmethod
    def fake() -> User:
        return User(
            id=UserId(1),
            email=Text("ada@example.com"),
            first_name=Text("Ada"),
            last_name=Text("Lovelace"),
        )


class CompanyRoleId(Id[int]):
    pass


class CompanyRole(pydantic.BaseModel, frozen=True):
    id: CompanyRoleId
    company_id: CompanyId
    name: Text

    @staticmethod
    def fake(company_id: CompanyId) -> CompanyRole:
        return CompanyRole(
            id=CompanyRoleId(1), company_id=company_id, name=Text("Admin")
        )


class CompanyMembershipId(Id[int]):
    pass


class CompanyMembership(pydantic.BaseModel, frozen=True):
    id: CompanyMembershipId
    user_id: UserId
    company_id: CompanyId
    company_role_id: CompanyRoleId
    name: Text
    last_used_at: LastUsedAt

    @staticmethod
    def fake(role: CompanyRole, user_id: UserId) -> CompanyMembership:
        return CompanyMembership(
            id=CompanyMembershipId(1),
            user_id=user_id,
            company_id=role.company_id,
            company_role_id=role.id,
            name=Text("Ada at Acme"),
            last_used_at=LastUsedAt.fake(),
        )


class DepartmentId(Id[int]):
    pass


class Department(pydantic.BaseModel, frozen=True):
    id: DepartmentId
    name: Text
    company_id: CompanyId

    @staticmethod
    def fake(company_id: CompanyId) -> Department:
        return Department(
            id=DepartmentId(1), name=Text("Copenhagen"), company_id=company_id
        )


class ShopOrderStatus(enum.StrEnum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ShopOrderId(Id[int]):
    pass


class ShopOrder(pydantic.BaseModel, frozen=True):
    id: ShopOrderId
    order_number: Text
    company_id: CompanyId
    department_id: DepartmentId
    company_membership_id: CompanyMembershipId
    status: ShopOrderStatus
    placed_at: PlacedAt

    @staticmethod
    def fake(department: Department, membership_id: CompanyMembershipId) -> ShopOrder:
        return ShopOrder(
            id=ShopOrderId(1),
            order_number=Text("SO-0001"),
            company_id=department.company_id,
            department_id=department.id,
            company_membership_id=membership_id,
            status=ShopOrderStatus.NEW,
            placed_at=PlacedAt.fake(),
        )


def named(*fields: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in fields)


def hydrate_membership() -> steps.Step:
    """Flowbase leaves `company_id` an ID, so only these two are resolved."""
    return steps.resolve(named(FieldName("user_id"), FieldName("company_role_id")))


def flowbase_views() -> list[View]:
    return [
        View(
            name=ViewName("CompanyMembershipHydrated"),
            source=CompanyMembership,
            step=hydrate_membership(),
        ),
        View(
            name=ViewName("ShopOrderHydrated"),
            source=ShopOrder,
            step=steps.pipeline(
                steps.resolve(named(FieldName("department_id"))),
                steps.resolve(
                    named(FieldName("company_membership_id")), hydrate_membership()
                ),
            ),
        ),
    ]


def entity_graph() -> EntityGraph:
    return EntityGraph.build(
        [Company, User, CompanyRole, CompanyMembership, Department, ShopOrder]
    ).unwrap()


def compiled_views() -> compiler.CompiledViews:
    match compiler.compile_views(entity_graph(), flowbase_views()):
        case Ok(compiled):
            return compiled
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def test_golden_flowbase_view_models() -> None:
    match PythonPlan.of(
        compiled_views(),
        PythonLayout.models_only(lambda _: ModulePath("flowbase.views")),
    ):
        case Ok(plan):
            (file,) = PythonRenderer.render(plan).files
        case Err(error):
            pytest.fail(f"expected View models, got {error}")

    assert file.text == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.

from functional_relational_python.outputs.test_flowbase_golden import (
    CompanyId,
    CompanyMembershipId,
    CompanyRole,
    Department,
    LastUsedAt,
    PlacedAt,
    ShopOrderId,
    ShopOrderStatus,
    Text,
    User,
)
from pydantic import BaseModel


class CompanyMembershipHydrated(BaseModel, frozen=True):
    id: CompanyMembershipId
    user: User
    company_id: CompanyId
    company_role: CompanyRole
    name: Text
    last_used_at: LastUsedAt


class ShopOrderHydratedCompanyMembership(BaseModel, frozen=True):
    id: CompanyMembershipId
    user: User
    company_id: CompanyId
    company_role: CompanyRole
    name: Text
    last_used_at: LastUsedAt


class ShopOrderHydrated(BaseModel, frozen=True):
    id: ShopOrderId
    order_number: Text
    company_id: CompanyId
    department: Department
    company_membership: ShopOrderHydratedCompanyMembership
    status: ShopOrderStatus
    placed_at: PlacedAt
"""
    )


def test_golden_flowbase_view_results() -> None:
    company = Company.fake()
    user = User.fake()
    role = CompanyRole.fake(company.id)
    membership = CompanyMembership.fake(role, user.id)
    department = Department.fake(company.id)
    order = ShopOrder.fake(department, membership.id)
    view_engine = engine.InMemoryViewEngine(entity_graph(), compiled_views())
    for entity in (company, user, role, membership, department, order):
        view_engine.put(entity).unwrap()
    hydrated_membership = engine.ViewRow(
        {
            FieldName("id"): membership.id,
            FieldName("user"): user,
            FieldName("company_id"): company.id,
            FieldName("company_role"): role,
            FieldName("name"): membership.name,
            FieldName("last_used_at"): membership.last_used_at,
        }
    )

    assert [view_engine.read(view.name).unwrap() for view in flowbase_views()] == [
        (hydrated_membership,),
        (
            engine.ViewRow(
                {
                    FieldName("id"): order.id,
                    FieldName("order_number"): order.order_number,
                    FieldName("company_id"): company.id,
                    FieldName("department"): department,
                    FieldName("company_membership"): hydrated_membership,
                    FieldName("status"): order.status,
                    FieldName("placed_at"): order.placed_at,
                }
            ),
        ),
    ]
