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

A third golden test reads Flowbase's multi-table campaigns: a base `ChannelCampaign`
row, plus at most one detail row per channel (`MetaCampaign`, `LinkedInCampaign`),
each holding a one-to-one Reference to its `ChannelCampaign`. A View over
`ChannelCampaign` reverses those References to read each campaign's detail rows.

A fourth rebuilds Flowbase's Meta campaign classes (`djank/db_models/
meta_campaign.py`) as the Union `Template | (Draft | Submitted)`: each Variant is
a View over the base table that filters by the campaign's `kind` and
`submit_state`, and flattens the Meta detail row into itself. The Meta campaign
Union is itself a Variant of a cross-channel Union, discriminated by `channel`.

A fifth rebuilds Flowbase's Meta campaign update payloads (`services/channels/
meta/entities/campaign.py`) as a Union of a draft and a submitted update Payload,
each tagged by a `submit_state` field it adds.
"""

import datetime
import enum
import types
from typing import Annotated, Literal

import pydantic
import pytest
from pydantic import MISSING
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler
from functional_relational_python.compiler.model_names import PascalCase
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
)
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    OneToOne,
    OneToOneMarker,
)
from functional_relational_python.outputs import engine
from functional_relational_python.outputs.python.plan import PythonLayout, PythonPlan
from functional_relational_python.outputs.python.render import PythonRenderer
from functional_relational_python.outputs.python.symbols import FirstParty, ModulePath
from functional_relational_python.pipeline import conditions, ir, steps
from functional_relational_python.pipeline import selectors as sel
from functional_relational_python.pipeline.payloads import Payload, PayloadName
from functional_relational_python.pipeline.unions import Union, UnionName
from functional_relational_python.pipeline.views import View, ViewName
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
    return sel.matching(lambda field: field.name in fields)


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
        [Company, User, CompanyRole, CompanyMembership, Department, ShopOrder],
        IdBase.fake(),
        OneToOneMarker.fake(),
    ).unwrap()


def compiled_views() -> compiler.CompiledViews:
    match compiler.compile_views(entity_graph(), flowbase_views()):
        case Ok(compiled):
            return compiled
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def test_golden_flowbase_view_models() -> None:
    match PythonPlan.of(
        compiler.CompiledCatalog(
            views=compiled_views().views,
            payloads=(),
            unions=(),
            payload_unions=(),
            one_to_one_marker=OneToOneMarker.fake(),
        ),
        PythonLayout.models_only(lambda _: ModulePath("flowbase.views")),
    ):
        case Ok(plan):
            (file,) = PythonRenderer.render(plan, first_party=FirstParty.fake()).files
        case Err(error):
            pytest.fail(f"expected View models, got {error}")

    assert file.text == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.

from pydantic import BaseModel

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


class ChannelCampaignId(Id[int]):
    pass


class ChannelCampaign(pydantic.BaseModel, frozen=True):
    id: ChannelCampaignId
    company_id: CompanyId
    name: Text


class MetaCampaignId(Id[int]):
    pass


class MetaCampaign(pydantic.BaseModel, frozen=True):
    id: MetaCampaignId
    channel_campaign_id: Annotated[ChannelCampaignId, OneToOne()]
    ad_account_id: Text


class LinkedInCampaignId(Id[int]):
    pass


class LinkedInCampaign(pydantic.BaseModel, frozen=True):
    id: LinkedInCampaignId
    channel_campaign_id: Annotated[ChannelCampaignId, OneToOne()]
    campaign_group: Text


def campaign_view() -> View:
    without_base = steps.omit("channel_campaign_id")
    return View(
        name=ViewName("ChannelCampaignDetails"),
        source=ChannelCampaign,
        step=steps.pipeline(
            steps.reverse(MetaCampaign, without_base),
            steps.reverse(LinkedInCampaign, without_base),
        ),
    )


def campaign_graph() -> EntityGraph:
    return EntityGraph.build(
        [Company, ChannelCampaign, MetaCampaign, LinkedInCampaign],
        IdBase.fake(),
        OneToOneMarker.fake(),
    ).unwrap()


def compiled_campaign_views() -> compiler.CompiledViews:
    match compiler.compile_views(campaign_graph(), [campaign_view()]):
        case Ok(compiled):
            return compiled
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def test_golden_flowbase_campaign_view_model() -> None:
    match PythonPlan.of(
        compiler.CompiledCatalog(
            views=compiled_campaign_views().views,
            payloads=(),
            unions=(),
            payload_unions=(),
            one_to_one_marker=OneToOneMarker.fake(),
        ),
        PythonLayout.models_only(lambda _: ModulePath("flowbase.views")),
    ):
        case Ok(plan):
            (file,) = PythonRenderer.render(plan, first_party=FirstParty.fake()).files
        case Err(error):
            pytest.fail(f"expected View models, got {error}")

    assert file.text == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.

from pydantic import BaseModel

from functional_relational_python.outputs.test_flowbase_golden import (
    ChannelCampaignId,
    CompanyId,
    LinkedInCampaignId,
    MetaCampaignId,
    Text,
)


class ChannelCampaignDetailsMetaCampaign(BaseModel, frozen=True):
    id: MetaCampaignId
    ad_account_id: Text


class ChannelCampaignDetailsLinkedInCampaign(BaseModel, frozen=True):
    id: LinkedInCampaignId
    campaign_group: Text


class ChannelCampaignDetails(BaseModel, frozen=True):
    id: ChannelCampaignId
    company_id: CompanyId
    name: Text
    meta_campaign: ChannelCampaignDetailsMetaCampaign | None
    linked_in_campaign: ChannelCampaignDetailsLinkedInCampaign | None
"""
    )


def test_golden_flowbase_campaign_view_results() -> None:
    company = Company.fake()
    on_meta = ChannelCampaign(
        id=ChannelCampaignId(1), company_id=company.id, name=Text("Spring sale")
    )
    on_linkedin = ChannelCampaign(
        id=ChannelCampaignId(2), company_id=company.id, name=Text("Hiring")
    )
    meta = MetaCampaign(
        id=MetaCampaignId(1),
        channel_campaign_id=on_meta.id,
        ad_account_id=Text("act_1"),
    )
    linkedin = LinkedInCampaign(
        id=LinkedInCampaignId(1),
        channel_campaign_id=on_linkedin.id,
        campaign_group=Text("Talent"),
    )
    view_engine = engine.InMemoryViewEngine(campaign_graph(), compiled_campaign_views())
    for entity in (company, on_meta, on_linkedin, meta, linkedin):
        view_engine.put(entity).unwrap()

    assert view_engine.read(campaign_view().name).unwrap() == (
        engine.ViewRow(
            {
                FieldName("id"): on_meta.id,
                FieldName("company_id"): company.id,
                FieldName("name"): on_meta.name,
                FieldName("meta_campaign"): engine.ViewRow(
                    {
                        FieldName("id"): meta.id,
                        FieldName("ad_account_id"): meta.ad_account_id,
                    }
                ),
                FieldName("linked_in_campaign"): None,
            }
        ),
        engine.ViewRow(
            {
                FieldName("id"): on_linkedin.id,
                FieldName("company_id"): company.id,
                FieldName("name"): on_linkedin.name,
                FieldName("meta_campaign"): None,
                FieldName("linked_in_campaign"): engine.ViewRow(
                    {
                        FieldName("id"): linkedin.id,
                        FieldName("campaign_group"): linkedin.campaign_group,
                    }
                ),
            }
        ),
    )


class Channel(enum.StrEnum):
    META = "meta"
    LINKEDIN = "linkedin"


class CampaignKind(enum.StrEnum):
    TEMPLATE = "template"
    INSTANCE = "instance"


class SubmitState(enum.StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"


class SubmittedAt(pydantic.RootModel[datetime.datetime], frozen=True):
    @staticmethod
    def fake() -> SubmittedAt:
        return SubmittedAt(datetime.datetime(2026, 5, 6, tzinfo=datetime.UTC))


class CampaignName(pydantic.RootModel[str], frozen=True):
    pass


class LifetimeBudget(pydantic.RootModel[int], frozen=True):
    pass


class ChannelCampaignEntityId(Id[int]):
    pass


class ChannelCampaignEntity(pydantic.BaseModel, frozen=True):
    """The base table every channel's campaigns share."""

    id: ChannelCampaignEntityId
    company_id: CompanyId
    channel: Channel
    kind: CampaignKind
    submit_state: SubmitState
    submitted_at: SubmittedAt | None


class MetaCampaignEntityId(Id[int]):
    pass


class MetaCampaignEntity(pydantic.BaseModel, frozen=True):
    """The Meta detail table: at most one row per base row."""

    id: MetaCampaignEntityId
    channel_campaign_id: Annotated[ChannelCampaignEntityId, OneToOne()]
    name: CampaignName | None
    lifetime_budget: LifetimeBudget | None


class LinkedInCampaignEntityId(Id[int]):
    pass


class LinkedInCampaignEntity(pydantic.BaseModel, frozen=True):
    """The LinkedIn detail table: at most one row per base row."""

    id: LinkedInCampaignEntityId
    channel_campaign_id: Annotated[ChannelCampaignEntityId, OneToOne()]
    campaign_group: Text


def has(field: FieldName, value: enum.Enum) -> ir.Comparison:
    return conditions.eq(conditions.field(field), value)


def with_detail(entity: type[pydantic.BaseModel], *then: steps.Step) -> steps.Step:
    """Reads the campaign's detail row, which it must have, into its own fields;
    the base table's `id` stays the campaign's."""
    field = FieldName(PascalCase(entity.__name__).snake_case().root)
    return steps.pipeline(
        steps.reverse(
            entity,
            steps.omit("id"),
            steps.omit("channel_campaign_id"),
            *then,
        ),
        steps.non_nullable(field.root),
        steps.flatten(field.root),
    )


def meta_campaign_variants() -> tuple[View, View, View]:
    """Flowbase's `TemplateMetaCampaignDb`, `DraftMetaCampaignDb` and
    `SubmittedMetaCampaignDb`."""
    instance = (
        has(FieldName("channel"), Channel.META),
        has(FieldName("kind"), CampaignKind.INSTANCE),
    )
    return (
        View(
            name=ViewName("TemplateMetaCampaign"),
            source=ChannelCampaignEntity,
            step=steps.pipeline(
                steps.where(
                    conditions.and_(
                        has(FieldName("channel"), Channel.META),
                        has(FieldName("kind"), CampaignKind.TEMPLATE),
                        conditions.is_null(conditions.field(FieldName("submitted_at"))),
                    )
                ),
                steps.omit("submit_state"),
                with_detail(MetaCampaignEntity),
            ),
        ),
        View(
            name=ViewName("DraftMetaCampaign"),
            source=ChannelCampaignEntity,
            step=steps.pipeline(
                steps.where(
                    conditions.and_(
                        *instance, has(FieldName("submit_state"), SubmitState.DRAFT)
                    )
                ),
                with_detail(MetaCampaignEntity),
            ),
        ),
        View(
            name=ViewName("SubmittedMetaCampaign"),
            source=ChannelCampaignEntity,
            step=steps.pipeline(
                steps.where(
                    conditions.and_(
                        *instance, has(FieldName("submit_state"), SubmitState.SUBMITTED)
                    )
                ),
                steps.non_nullable("submitted_at"),
                with_detail(MetaCampaignEntity, steps.non_nullable("name")),
            ),
        ),
    )


def linkedin_campaign() -> View:
    return View(
        name=ViewName("LinkedInCampaign"),
        source=ChannelCampaignEntity,
        step=steps.pipeline(
            steps.where(has(FieldName("channel"), Channel.LINKEDIN)),
            with_detail(LinkedInCampaignEntity),
        ),
    )


def channel_campaign_union() -> Union:
    """`Template | (Draft | Submitted)` for Meta, itself a Variant of the Union of
    every channel's campaigns."""
    template, draft, submitted = meta_campaign_variants()
    meta = Union(
        name=UnionName("MetaCampaign"),
        discriminator=FieldName("kind"),
        variants=(
            template,
            Union(
                name=UnionName("InstanceMetaCampaign"),
                discriminator=FieldName("submit_state"),
                variants=(draft, submitted),
            ),
        ),
    )
    return Union(
        name=UnionName("ChannelCampaignUnion"),
        discriminator=FieldName("channel"),
        variants=(meta, linkedin_campaign()),
    )


def channel_campaign_graph() -> EntityGraph:
    return EntityGraph.build(
        [Company, ChannelCampaignEntity, MetaCampaignEntity, LinkedInCampaignEntity],
        IdBase.fake(),
        OneToOneMarker.fake(),
    ).unwrap()


def compiled_channel_campaigns() -> compiler.CompiledViews:
    match compiler.compile_views(
        channel_campaign_graph(), [], [channel_campaign_union()]
    ):
        case Ok(compiled):
            return compiled
        case Err(error):
            pytest.fail(f"expected a compiled Union, got {error}")


def test_golden_flowbase_meta_campaign_union_models() -> None:
    compiled = compiled_channel_campaigns()
    match PythonPlan.of(
        compiler.CompiledCatalog(
            views=compiled.views,
            payloads=(),
            unions=compiled.unions,
            payload_unions=(),
            one_to_one_marker=OneToOneMarker.fake(),
        ),
        PythonLayout.models_only(lambda _: ModulePath("flowbase.views")),
    ):
        case Ok(plan):
            (file,) = PythonRenderer.render(plan, first_party=FirstParty.fake()).files
        case Err(error):
            pytest.fail(f"expected Union models, got {error}")

    assert file.text == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator

from functional_relational_python.outputs.test_flowbase_golden import (
    CampaignKind,
    CampaignName,
    Channel,
    ChannelCampaignEntityId,
    CompanyId,
    LifetimeBudget,
    SubmitState,
    SubmittedAt,
    Text,
)


class TemplateMetaCampaign(BaseModel, frozen=True):
    id: ChannelCampaignEntityId
    company_id: CompanyId
    channel: Literal[Channel.META]
    kind: Literal[CampaignKind.TEMPLATE]
    submitted_at: Literal[None]
    name: CampaignName | None
    lifetime_budget: LifetimeBudget | None


class DraftMetaCampaign(BaseModel, frozen=True):
    id: ChannelCampaignEntityId
    company_id: CompanyId
    channel: Literal[Channel.META]
    kind: Literal[CampaignKind.INSTANCE]
    submit_state: Literal[SubmitState.DRAFT]
    submitted_at: SubmittedAt | None
    name: CampaignName | None
    lifetime_budget: LifetimeBudget | None


class SubmittedMetaCampaign(BaseModel, frozen=True):
    id: ChannelCampaignEntityId
    company_id: CompanyId
    channel: Literal[Channel.META]
    kind: Literal[CampaignKind.INSTANCE]
    submit_state: Literal[SubmitState.SUBMITTED]
    submitted_at: SubmittedAt
    name: CampaignName
    lifetime_budget: LifetimeBudget | None


class LinkedInCampaign(BaseModel, frozen=True):
    id: ChannelCampaignEntityId
    company_id: CompanyId
    channel: Literal[Channel.LINKEDIN]
    kind: CampaignKind
    submit_state: SubmitState
    submitted_at: SubmittedAt | None
    campaign_group: Text


type InstanceMetaCampaign = Annotated[
    DraftMetaCampaign | SubmittedMetaCampaign,
    Discriminator("submit_state"),
]
type MetaCampaign = Annotated[
    TemplateMetaCampaign | InstanceMetaCampaign,
    Discriminator("kind"),
]
type ChannelCampaignUnion = Annotated[
    MetaCampaign | LinkedInCampaign,
    Discriminator("channel"),
]
"""
    )


def test_golden_flowbase_meta_campaign_union_results() -> None:
    company = Company.fake()

    def campaign(
        id_value: ChannelCampaignEntityId,
        channel: Channel,
        kind: CampaignKind,
        submit_state: SubmitState,
        submitted_at: SubmittedAt | None,
    ) -> ChannelCampaignEntity:
        return ChannelCampaignEntity(
            id=id_value,
            company_id=company.id,
            channel=channel,
            kind=kind,
            submit_state=submit_state,
            submitted_at=submitted_at,
        )

    template = campaign(
        ChannelCampaignEntityId(1),
        Channel.META,
        CampaignKind.TEMPLATE,
        SubmitState.DRAFT,
        None,
    )
    draft = campaign(
        ChannelCampaignEntityId(2),
        Channel.META,
        CampaignKind.INSTANCE,
        SubmitState.DRAFT,
        None,
    )
    submitted = campaign(
        ChannelCampaignEntityId(3),
        Channel.META,
        CampaignKind.INSTANCE,
        SubmitState.SUBMITTED,
        SubmittedAt.fake(),
    )
    on_linkedin = campaign(
        ChannelCampaignEntityId(4),
        Channel.LINKEDIN,
        CampaignKind.INSTANCE,
        SubmitState.DRAFT,
        None,
    )
    # A submitted template matches no Variant, so it is no result.
    submitted_template = campaign(
        ChannelCampaignEntityId(5),
        Channel.META,
        CampaignKind.TEMPLATE,
        SubmitState.SUBMITTED,
        SubmittedAt.fake(),
    )
    meta_details = {
        meta.id: MetaCampaignEntity(
            id=MetaCampaignEntityId(meta.id.root),
            channel_campaign_id=meta.id,
            name=CampaignName(f"Campaign {meta.id.root}"),
            lifetime_budget=None,
        )
        for meta in (template, draft, submitted, submitted_template)
    }
    linkedin = LinkedInCampaignEntity(
        id=LinkedInCampaignEntityId(1),
        channel_campaign_id=on_linkedin.id,
        campaign_group=Text("Talent"),
    )
    view_engine = engine.InMemoryViewEngine(
        channel_campaign_graph(), compiled_channel_campaigns()
    )
    for entity in (
        company,
        template,
        draft,
        submitted,
        on_linkedin,
        submitted_template,
        *meta_details.values(),
        linkedin,
    ):
        view_engine.put(entity).unwrap()

    def with_meta(row: ChannelCampaignEntity, *without: FieldName) -> engine.ViewRow:
        """The campaign's own fields but `without`, then its Meta detail's."""
        detail = meta_details[row.id]
        return engine.ViewRow(
            {
                **{
                    FieldName(name): value
                    for name, value in vars(row).items()
                    if FieldName(name) not in without
                },
                FieldName("name"): detail.name,
                FieldName("lifetime_budget"): detail.lifetime_budget,
            }
        )

    assert view_engine.read_union(UnionName("ChannelCampaignUnion")).unwrap() == (
        engine.VariantRow(
            view=ViewName("TemplateMetaCampaign"),
            row=with_meta(template, FieldName("submit_state")),
        ),
        engine.VariantRow(
            view=ViewName("DraftMetaCampaign"),
            row=with_meta(draft),
        ),
        engine.VariantRow(
            view=ViewName("SubmittedMetaCampaign"),
            row=with_meta(submitted),
        ),
        engine.VariantRow(
            view=ViewName("LinkedInCampaign"),
            row=engine.ViewRow(
                {
                    **{
                        FieldName(name): value
                        for name, value in vars(on_linkedin).items()
                    },
                    FieldName("campaign_group"): linkedin.campaign_group,
                }
            ),
        ),
    )


def update_meta_campaign_payload() -> Union:
    """Flowbase's `UpdateDraftMetaCampaignPayload` and
    `UpdateSubmittedMetaCampaignPayload`, as one Union tagged by `submit_state`.
    An update may leave any field out; a submitted campaign keeps its name."""
    every_field = sel.all_fields()

    def update(name: PayloadName, tag: ir.AddedType, *retypes: steps.Step) -> Payload:
        return Payload(
            name=name,
            source=MetaCampaignEntity,
            step=steps.pipeline(
                steps.pick(["name", "lifetime_budget"]),
                *retypes,
                steps.default(every_field, MISSING),
                steps.add("submit_state", tag),
            ),
        )

    return Union(
        name=UnionName("UpdateMetaCampaignPayload"),
        discriminator=FieldName("submit_state"),
        variants=(
            update(
                PayloadName("UpdateDraftMetaCampaignPayload"),
                Literal[SubmitState.DRAFT],
                steps.widen(every_field, MISSING),
            ),
            update(
                PayloadName("UpdateSubmittedMetaCampaignPayload"),
                Literal[SubmitState.SUBMITTED],
                steps.set_type("name", CampaignName | MISSING),
                steps.widen("lifetime_budget", MISSING),
            ),
        ),
    )


def update_meta_campaign_payload_module() -> FileText:
    compiled = compiler.compile_catalog(
        channel_campaign_graph(), [], [], [update_meta_campaign_payload()]
    ).unwrap()
    match PythonPlan.of(
        compiled,
        PythonLayout.models_only(lambda _: ModulePath("flowbase.payloads")),
    ):
        case Ok(plan):
            (file,) = PythonRenderer.render(plan, first_party=FirstParty.fake()).files
            return file.text
        case Err(error):
            pytest.fail(f"expected a Payload Union model, got {error}")


def test_golden_flowbase_update_meta_campaign_payload_union_model() -> None:
    assert update_meta_campaign_payload_module() == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.

from typing import Annotated, Literal

from pydantic import MISSING, BaseModel, Discriminator

from functional_relational_python.outputs.test_flowbase_golden import (
    CampaignName,
    LifetimeBudget,
    SubmitState,
)


class UpdateDraftMetaCampaignPayload(BaseModel, frozen=True):
    name: CampaignName | MISSING | None = MISSING
    lifetime_budget: LifetimeBudget | MISSING | None = MISSING
    submit_state: Literal[SubmitState.DRAFT]


class UpdateSubmittedMetaCampaignPayload(BaseModel, frozen=True):
    name: CampaignName | MISSING = MISSING
    lifetime_budget: LifetimeBudget | MISSING | None = MISSING
    submit_state: Literal[SubmitState.SUBMITTED]


type UpdateMetaCampaignPayload = Annotated[
    UpdateDraftMetaCampaignPayload | UpdateSubmittedMetaCampaignPayload,
    Discriminator("submit_state"),
]
"""
    )


def test_the_update_meta_campaign_payload_union_reads_each_payload_by_its_tag() -> None:
    module = types.ModuleType("flowbase_payloads")
    exec(update_meta_campaign_payload_module().root, module.__dict__)
    adapter = pydantic.TypeAdapter(module.UpdateMetaCampaignPayload)

    assert [
        type(adapter.validate_python({"submit_state": state})).__name__
        for state in (SubmitState.SUBMITTED, SubmitState.DRAFT)
    ] == ["UpdateSubmittedMetaCampaignPayload", "UpdateDraftMetaCampaignPayload"]
