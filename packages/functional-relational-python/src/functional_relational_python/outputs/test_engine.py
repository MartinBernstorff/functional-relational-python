import enum
from collections.abc import Callable
from typing import Annotated

import pydantic
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler, test_compiler
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
)
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    OneToOne,
    OneToOneMarker,
    ParentLink,
)
from functional_relational_python.outputs import engine
from functional_relational_python.pipeline import conditions, ir, steps
from functional_relational_python.pipeline import selectors as sel
from functional_relational_python.pipeline.unions import Union, UnionName
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.support.diagnostic import DiagnosticsError


class Text(pydantic.RootModel[str], frozen=True):
    pass


class CompanyId(Id[int]):
    @staticmethod
    def fake() -> CompanyId:
        return CompanyId(1)


class Company(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text
    secret: Text

    @staticmethod
    def fake(company_id: CompanyId) -> Company:
        return Company(id=company_id, name=Text("Acme"), secret=Text("hunter2"))


class Memo(pydantic.BaseModel, frozen=True):
    """A Source that is not an Entity: it has no ID."""

    body: Text


def make_engine(*views: View) -> engine.InMemoryViewEngine:
    graph = EntityGraph.build([Company], IdBase.fake(), OneToOneMarker.fake()).unwrap()
    match compiler.compile_views(graph, views):
        case Ok(compiled):
            return engine.InMemoryViewEngine(graph, compiled)
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def view_of(step: steps.Step) -> View:
    return View(name=ViewName.fake(), source=Company, step=step)


def read(
    view_engine: engine.InMemoryViewEngine, view: ViewName
) -> tuple[engine.ViewRow, ...]:
    match view_engine.read(view):
        case Ok(rows):
            return rows
        case Err(error):
            pytest.fail(f"expected rows, got {error}")


def row_of(company: Company, *omitted: FieldName) -> engine.ViewRow:
    return engine.ViewRow(
        {
            FieldName(name): getattr(company, name)
            for name in Company.model_fields
            if FieldName(name) not in omitted
        }
    )


@pytest.mark.parametrize(
    "omitted",
    [(), (FieldName("secret"),), (FieldName("secret"), FieldName("name"))],
    ids=["none", "one", "two"],
)
def test_a_view_returns_each_entity_without_the_omitted_fields(
    omitted: tuple[FieldName, ...],
) -> None:
    view = view_of(steps.pipeline(*(steps.omit(field.root) for field in omitted)))
    view_engine = make_engine(view)
    companies = [Company.fake(CompanyId(1)), Company.fake(CompanyId(2))]
    for company in companies:
        view_engine.put(company).unwrap()

    assert read(view_engine, view.name) == tuple(
        row_of(company, *omitted) for company in companies
    )


def test_a_renamed_field_keeps_its_value_and_position_under_the_new_name() -> None:
    new = FieldName("title")
    view = view_of(steps.rename("name", new.root))
    view_engine = make_engine(view)
    company = Company.fake(CompanyId.fake())
    view_engine.put(company).unwrap()

    assert [list(row.root.items()) for row in read(view_engine, view.name)] == [
        [
            (FieldName("id"), company.id),
            (new, company.name),
            (FieldName("secret"), company.secret),
        ]
    ]


def test_omitting_the_selected_primary_key_drops_it_from_each_row() -> None:
    view = view_of(steps.omit(sel.matching(steps.is_primary_key)))
    view_engine = make_engine(view)
    company = Company.fake(CompanyId.fake())
    view_engine.put(company).unwrap()

    assert read(view_engine, view.name) == (row_of(company, FieldName("id")),)


def test_selected_renamed_fields_keep_their_values_and_positions() -> None:
    view = view_of(
        steps.rename(
            named(FieldName("id"), FieldName("secret")),
            lambda field: FieldName(f"company_{field.root}"),
        )
    )
    view_engine = make_engine(view)
    company = Company.fake(CompanyId.fake())
    view_engine.put(company).unwrap()

    assert [list(row.root.items()) for row in read(view_engine, view.name)] == [
        [
            (FieldName("company_id"), company.id),
            (FieldName("name"), company.name),
            (FieldName("company_secret"), company.secret),
        ]
    ]


def test_a_picked_view_returns_each_entity_with_only_the_listed_fields() -> None:
    view = view_of(steps.pick(["name"]))
    view_engine = make_engine(view)
    company = Company.fake(CompanyId.fake())
    view_engine.put(company).unwrap()

    assert read(view_engine, view.name) == (
        engine.ViewRow({FieldName("name"): company.name}),
    )


class Kind(enum.Enum):
    COMPANY = "company"


@pytest.mark.parametrize(
    "default",
    ["company", None, pydantic.MISSING, [], {}, Kind.COMPANY],
    ids=["str", "none", "missing", "list", "dict", "enum"],
)
def test_an_added_field_holds_its_default_in_each_row(
    default: ir.UncheckedDefault,
) -> None:
    added = FieldName("kind")
    view = view_of(
        steps.pipeline(steps.pick(["name"]), steps.add(added.root, object, default))
    )
    view_engine = make_engine(view)
    company = Company.fake(CompanyId.fake())
    view_engine.put(company).unwrap()

    assert read(view_engine, view.name) == (
        engine.ViewRow({FieldName("name"): company.name, added: default}),
    )


def test_put_replaces_the_entity_with_the_same_id() -> None:
    view = view_of(steps.pipeline())
    view_engine = make_engine(view)
    original = Company.fake(CompanyId.fake())
    replacement = Company(
        id=original.id, name=Text(f"{original.name.root} 2"), secret=original.secret
    )
    view_engine.put(original).unwrap()
    view_engine.put(replacement).unwrap()

    assert read(view_engine, view.name) == (row_of(replacement),)


def test_a_view_over_an_empty_engine_has_no_rows() -> None:
    view = view_of(steps.pipeline())

    assert read(make_engine(view), view.name) == ()


def test_views_over_the_same_entities_are_independent() -> None:
    full = View(name=ViewName("Full"), source=Company, step=steps.pipeline())
    public = View(name=ViewName("Public"), source=Company, step=steps.omit("secret"))
    view_engine = make_engine(full, public)
    company = Company.fake(CompanyId.fake())
    view_engine.put(company).unwrap()

    assert (read(view_engine, full.name), read(view_engine, public.name)) == (
        (row_of(company),),
        (row_of(company, FieldName("secret")),),
    )


def test_put_rejects_a_model_that_is_not_an_entity() -> None:
    match make_engine().put(Memo(body=Text("note"))):
        case Ok():
            pytest.fail("expected NotAnEntityError")
        case Err(engine.NotAnEntityError() as error):
            assert error.model is Memo
        case Err(error):
            pytest.fail(f"expected NotAnEntityError, got {error!r}")


def test_reading_a_view_over_a_source_that_is_not_an_entity_is_an_error() -> None:
    view = View(name=ViewName.fake(), source=Memo, step=steps.pipeline())

    match make_engine(view).read(view.name):
        case Err(engine.NonEntitySourceError() as error):
            assert (error.view, error.source) == (view.name, Memo)
        case result:
            pytest.fail(f"expected NonEntitySourceError, got {result}")


def test_reading_an_unknown_view_names_it_and_the_known_views() -> None:
    known = view_of(steps.pipeline())
    unknown = ViewName(f"{known.name.root}Typo")

    match make_engine(known).read(unknown):
        case Ok(rows):
            pytest.fail(f"expected UnknownViewError, got {rows}")
        case Err(engine.UnknownViewError() as error):
            assert (error.view, error.known) == (unknown, (known.name,))
        case Err(error):
            pytest.fail(f"expected UnknownViewError, got {error}")


# Resolving References.


class DepartmentId(Id[int]):
    pass


class Department(pydantic.BaseModel, frozen=True):
    id: DepartmentId
    name: Text
    company_id: CompanyId

    def itself(self) -> engine.FieldValue:
        return self

    def name_of(self) -> engine.FieldValue:
        return self.name

    @staticmethod
    def fake(department_id: DepartmentId, company_id: CompanyId) -> Department:
        return Department(id=department_id, name=Text("Sales"), company_id=company_id)


class EmployeeId(Id[int]):
    pass


class Employee(pydantic.BaseModel, frozen=True):
    id: EmployeeId
    department_id: DepartmentId
    mentor_id: EmployeeId | None

    @staticmethod
    def fake(
        employee_id: EmployeeId,
        department_id: DepartmentId,
        mentor_id: EmployeeId | None,
    ) -> Employee:
        return Employee(
            id=employee_id, department_id=department_id, mentor_id=mentor_id
        )


def hr_engine(*views: View) -> engine.InMemoryViewEngine:
    graph = EntityGraph.build(
        [Company, Department, Employee], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    match compiler.compile_views(graph, views):
        case Ok(compiled):
            return engine.InMemoryViewEngine(graph, compiled)
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def employee_view(step: steps.Step) -> View:
    return View(name=ViewName.fake(), source=Employee, step=step)


def named(*fields: FieldName) -> ir.Select:
    return sel.matching(lambda field: field.name in fields)


def test_hydrate_replaces_each_reference_with_its_entity() -> None:
    view = employee_view(steps.hydrate())
    view_engine = hr_engine(view)
    department = Department.fake(DepartmentId(1), CompanyId.fake())
    mentor = Employee.fake(EmployeeId(1), department.id, None)
    mentee = Employee.fake(EmployeeId(2), department.id, mentor.id)
    for entity in (department, mentor, mentee):
        view_engine.put(entity).unwrap()

    assert read(view_engine, view.name) == tuple(
        engine.ViewRow(
            {
                FieldName("id"): employee.id,
                FieldName("department"): department,
                FieldName("mentor"): expected_mentor,
            }
        )
        for employee, expected_mentor in ((mentor, None), (mentee, mentor))
    )


def test_a_nested_pipeline_runs_on_the_referenced_entity() -> None:
    view = employee_view(
        steps.resolve(
            named(FieldName("department_id")),
            steps.hydrate(),
            steps.omit("id"),
        )
    )
    view_engine = hr_engine(view)
    company = Company.fake(CompanyId.fake())
    department = Department.fake(DepartmentId(1), company.id)
    employee = Employee.fake(EmployeeId(1), department.id, None)
    for entity in (company, department, employee):
        view_engine.put(entity).unwrap()

    assert read(view_engine, view.name) == (
        engine.ViewRow(
            {
                FieldName("id"): employee.id,
                FieldName("department"): engine.ViewRow(
                    {FieldName("name"): department.name, FieldName("company"): company}
                ),
                FieldName("mentor_id"): None,
            }
        ),
    )


def test_every_dangling_reference_is_reported_with_every_missing_id() -> None:
    view = employee_view(
        steps.resolve(
            named(FieldName("department_id"), FieldName("mentor_id")),
            steps.hydrate(),
        )
    )
    view_engine = hr_engine(view)
    missing_company = CompanyId(7)
    stored = Department.fake(DepartmentId(1), missing_company)
    missing_departments = (DepartmentId(2), DepartmentId(3))
    missing_mentor = EmployeeId(9)
    employees = (
        Employee.fake(EmployeeId(1), missing_departments[0], missing_mentor),
        Employee.fake(EmployeeId(2), missing_departments[1], None),
        Employee.fake(EmployeeId(3), missing_departments[1], None),
        Employee.fake(EmployeeId(4), stored.id, None),
    )
    for entity in (stored, *employees):
        view_engine.put(entity).unwrap()

    match view_engine.read(view.name):
        case Err(DiagnosticsError() as error):
            assert error.diagnostics == (
                engine.DanglingReference(
                    entity=Employee,
                    field=FieldName("department_id"),
                    missing=missing_departments,
                ),
                engine.DanglingReference(
                    entity=Employee,
                    field=FieldName("mentor_id"),
                    missing=(missing_mentor,),
                ),
                engine.DanglingReference(
                    entity=Department,
                    field=FieldName("company_id"),
                    missing=(missing_company,),
                ),
            )
        case result:
            pytest.fail(f"expected dangling References, got {result}")


def test_a_dangling_reference_names_the_entity_the_field_and_the_ids() -> None:
    dangling = engine.DanglingReference(
        entity=Employee,
        field=FieldName("department_id"),
        missing=(DepartmentId(2), DepartmentId(3)),
    )

    assert dangling.describe().root == (
        "Employee.department_id references missing IDs: 2, 3"
    )


class ProfileId(Id[int]):
    pass


class Profile(pydantic.BaseModel, frozen=True):
    """An Employee's detail row: at most one per Employee."""

    id: ProfileId
    employee_id: Annotated[EmployeeId, OneToOne()]
    bio: Text

    @staticmethod
    def fake(profile_id: ProfileId, employee_id: EmployeeId) -> Profile:
        return Profile(id=profile_id, employee_id=employee_id, bio=Text("Hi"))


def profile_engine(*views: View) -> engine.InMemoryViewEngine:
    graph = EntityGraph.build(
        [Company, Department, Employee, Profile], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    match compiler.compile_views(graph, views):
        case Ok(compiled):
            return engine.InMemoryViewEngine(graph, compiled)
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def test_reverse_gives_the_entity_referencing_each_row_or_none() -> None:
    view = employee_view(steps.pipeline(steps.pick(["id"]), steps.reverse(Profile)))
    view_engine = profile_engine(view)
    department = DepartmentId(1)
    with_profile = Employee.fake(EmployeeId(1), department, None)
    without_profile = Employee.fake(EmployeeId(2), department, None)
    profile = Profile.fake(ProfileId(1), with_profile.id)
    for entity in (with_profile, without_profile, profile):
        view_engine.put(entity).unwrap()

    assert read(view_engine, view.name) == tuple(
        engine.ViewRow({FieldName("id"): employee.id, FieldName("profile"): found})
        for employee, found in ((with_profile, profile), (without_profile, None))
    )


def test_flattening_a_reversed_entity_lifts_its_values_or_none() -> None:
    view = employee_view(
        steps.pipeline(
            steps.pick(["id"]),
            steps.reverse(Profile, steps.pick(["bio"])),
            steps.flatten("profile"),
        )
    )
    view_engine = profile_engine(view)
    department = DepartmentId(1)
    with_profile = Employee.fake(EmployeeId(1), department, None)
    without_profile = Employee.fake(EmployeeId(2), department, None)
    profile = Profile.fake(ProfileId(1), with_profile.id)
    for entity in (with_profile, without_profile, profile):
        view_engine.put(entity).unwrap()

    assert read(view_engine, view.name) == tuple(
        engine.ViewRow({FieldName("id"): employee.id, FieldName("bio"): bio})
        for employee, bio in ((with_profile, profile.bio), (without_profile, None))
    )


def test_a_reverse_sees_the_entities_put_since_the_last_read() -> None:
    view = employee_view(steps.pipeline(steps.pick(["id"]), steps.reverse(Profile)))
    view_engine = profile_engine(view)
    employee = Employee.fake(EmployeeId(1), DepartmentId(1), None)
    view_engine.put(employee).unwrap()
    before = read(view_engine, view.name)
    profile = Profile.fake(ProfileId(1), employee.id)
    view_engine.put(profile).unwrap()

    assert (before, read(view_engine, view.name)) == (
        (engine.ViewRow({FieldName("id"): employee.id, FieldName("profile"): None}),),
        (
            engine.ViewRow(
                {FieldName("id"): employee.id, FieldName("profile"): profile}
            ),
        ),
    )


def test_a_nested_pipeline_runs_on_the_reversed_entity() -> None:
    view = employee_view(
        steps.pipeline(
            steps.pick(["id"]),
            steps.reverse(Profile, steps.pick(["bio"])),
        )
    )
    view_engine = profile_engine(view)
    employee = Employee.fake(EmployeeId(1), DepartmentId(1), None)
    profile = Profile.fake(ProfileId(1), employee.id)
    for entity in (employee, profile):
        view_engine.put(entity).unwrap()

    assert read(view_engine, view.name) == (
        engine.ViewRow(
            {
                FieldName("id"): employee.id,
                FieldName("profile"): engine.ViewRow({FieldName("bio"): profile.bio}),
            }
        ),
    )


def test_every_id_several_entities_share_in_a_one_to_one_reference_is_reported() -> (
    None
):
    view = employee_view(steps.reverse(Profile))
    view_engine = profile_engine(view)
    employees = tuple(
        Employee.fake(EmployeeId(index), DepartmentId(1), None) for index in (1, 2, 3)
    )
    profiles = (
        Profile.fake(ProfileId(1), employees[0].id),
        Profile.fake(ProfileId(2), employees[0].id),
        Profile.fake(ProfileId(3), employees[1].id),
        Profile.fake(ProfileId(4), employees[1].id),
        Profile.fake(ProfileId(5), employees[2].id),
    )
    for entity in (*employees, *profiles):
        view_engine.put(entity).unwrap()

    match view_engine.read(view.name):
        case Err(DiagnosticsError() as error):
            assert error.diagnostics == (
                engine.SharedOneToOneReference(
                    entity=Profile,
                    field=FieldName("employee_id"),
                    shared=(employees[0].id, employees[1].id),
                ),
            )
        case result:
            pytest.fail(f"expected shared one-to-one References, got {result}")


def test_a_shared_one_to_one_reference_names_the_entity_the_field_and_the_ids() -> None:
    shared = engine.SharedOneToOneReference(
        entity=Profile,
        field=FieldName("employee_id"),
        shared=(EmployeeId(1), EmployeeId(2)),
    )

    assert shared.describe().root == (
        "Profile.employee_id is one-to-one, but several Profile entities reference "
        "each of: 1, 2"
    )


def test_flatten_lifts_the_referenced_entitys_values_into_each_row() -> None:
    view = employee_view(
        steps.pipeline(
            steps.resolve(
                named(FieldName("department_id")),
                steps.omit("id"),
                steps.hydrate(),
            ),
            steps.flatten("department"),
        )
    )
    view_engine = hr_engine(view)
    company = Company.fake(CompanyId.fake())
    department = Department.fake(DepartmentId(1), company.id)
    employee = Employee.fake(EmployeeId(1), department.id, None)
    for entity in (company, department, employee):
        view_engine.put(entity).unwrap()

    assert read(view_engine, view.name) == (
        engine.ViewRow(
            {
                FieldName("id"): employee.id,
                FieldName("name"): department.name,
                FieldName("company"): company,
                FieldName("mentor_id"): None,
            }
        ),
    )


def test_fields_lifted_out_of_an_absent_reference_are_none() -> None:
    view = employee_view(
        steps.pipeline(
            steps.resolve(
                named(FieldName("mentor_id")),
                steps.rename(
                    sel.all_fields(),
                    lambda field: FieldName(f"mentor_{field.root}"),
                ),
            ),
            steps.flatten("mentor"),
        )
    )
    view_engine = hr_engine(view)
    department = Department.fake(DepartmentId(1), CompanyId.fake())
    mentor = Employee.fake(EmployeeId(1), department.id, None)
    mentee = Employee.fake(EmployeeId(2), department.id, mentor.id)
    for entity in (department, mentor, mentee):
        view_engine.put(entity).unwrap()

    assert read(view_engine, view.name) == tuple(
        engine.ViewRow(
            {
                FieldName("id"): employee.id,
                FieldName("department_id"): department.id,
                FieldName("mentor_id"): None if lifted is None else lifted.id,
                FieldName("mentor_department_id"): (
                    None if lifted is None else lifted.department_id
                ),
                FieldName("mentor_mentor_id"): (
                    None if lifted is None else lifted.mentor_id
                ),
            }
        )
        for employee, lifted in ((mentor, None), (mentee, mentor))
    )


def mentor_flattened(*then: steps.Step) -> steps.Step:
    """Lifts the mentor's fields, prefixed `mentor_`; `then` runs on the mentor."""
    return steps.pipeline(
        steps.pick(["id", "mentor_id"]),
        steps.resolve(
            named(FieldName("mentor_id")),
            steps.rename(
                sel.all_fields(),
                lambda field: FieldName(f"mentor_{field.root}"),
            ),
            *then,
        ),
        steps.flatten("mentor"),
    )


@pytest.mark.parametrize(
    ("step", "field", "of_department"),
    [
        (
            steps.pipeline(
                mentor_flattened(),
                steps.resolve(named(FieldName("mentor_department_id"))),
            ),
            FieldName("mentor_department"),
            Department.itself,
        ),
        (
            steps.pipeline(
                mentor_flattened(
                    steps.resolve(
                        named(FieldName("mentor_department_id")),
                        steps.pick(["name"]),
                    )
                ),
                steps.flatten("mentor_department"),
            ),
            FieldName("name"),
            Department.name_of,
        ),
    ],
    ids=["a lifted Reference resolved", "a lifted resolved field flattened"],
)
def test_an_entity_lifted_out_is_found_through_the_flattened_one(
    step: steps.Step,
    field: FieldName,
    of_department: Callable[[Department], engine.FieldValue],
) -> None:
    view = employee_view(step)
    view_engine = hr_engine(view)
    department = Department.fake(DepartmentId(1), CompanyId.fake())
    mentor = Employee.fake(EmployeeId(1), department.id, None)
    mentee = Employee.fake(EmployeeId(2), department.id, mentor.id)
    for entity in (department, mentor, mentee):
        view_engine.put(entity).unwrap()

    rows = read(view_engine, view.name)

    # The mentor has no mentor, so nothing is found for them.
    assert [(row.root[FieldName("id")], row.root[field]) for row in rows] == [
        (mentor.id, None),
        (mentee.id, of_department(department)),
    ]


def test_a_dangling_reference_lifted_out_is_reported_on_its_entity() -> None:
    view = employee_view(
        steps.pipeline(
            steps.resolve(
                named(FieldName("department_id")),
                steps.omit("id"),
                steps.hydrate(),
            ),
            steps.flatten("department"),
        )
    )
    view_engine = hr_engine(view)
    missing_company = CompanyId(7)
    department = Department.fake(DepartmentId(1), missing_company)
    for entity in (department, Employee.fake(EmployeeId(1), department.id, None)):
        view_engine.put(entity).unwrap()

    match view_engine.read(view.name):
        case Err(DiagnosticsError() as error):
            assert error.diagnostics == (
                engine.DanglingReference(
                    entity=Department,
                    field=FieldName("company_id"),
                    missing=(missing_company,),
                ),
            )
        case result:
            pytest.fail(f"expected a dangling Reference, got {result}")


# Property-based tests: the engine runs every View the compiler compiles, over
# EntityGraphs generated per example (see `test_compiler`).


class TotalityCase(pydantic.BaseModel, frozen=True):
    views: test_compiler.ViewCase
    stored: tuple[pydantic.BaseModel, ...]


@st.composite
def totality_cases(draw: st.DrawFn) -> TotalityCase:
    views = draw(test_compiler.view_cases())
    return TotalityCase(
        views=views, stored=draw(test_compiler.stored_entities(views.models))
    )


def has_unique_field_names(relation: ir.Relation) -> bool:
    """Whether no two fields share a name, in nested relations too."""
    names = relation.field_names()
    return len(set(names)) == len(names) and all(
        has_unique_field_names(field.relation)
        for field in relation.fields
        if isinstance(field, ir.NestedField)
    )


@given(totality_cases())
def test_every_view_either_gives_results_or_reports_diagnostics(
    case: TotalityCase,
) -> None:
    graph = case.views.models.graph()
    checked = compiler.check(graph, case.views.views)
    compiled = tuple(v for v in checked if isinstance(v, compiler.CompiledView))
    view_engine = engine.InMemoryViewEngine(
        graph, compiler.CompiledViews(views=compiled, unions=())
    )
    for entity in case.stored:
        view_engine.put(entity).unwrap()

    assert [view.name for view in checked] == [view.name for view in case.views.views]
    assert case.views.valid <= {view.name for view in compiled}
    for view in checked:
        event(f"valid by construction: {view.name in case.views.valid}")
        match view:
            case compiler.CompiledView() if view.source not in graph.entities:
                assert has_unique_field_names(view.relation)
                match view_engine.read(view.name):
                    case Err(engine.NonEntitySourceError()):
                        pass
                    case result:
                        pytest.fail(f"expected NonEntitySourceError, got {result}")
            case compiler.CompiledView():
                assert has_unique_field_names(view.relation)
                rows = view_engine.read(view.name).unwrap()
                event(f"compiled view has rows: {bool(rows)}")
                assert all(
                    tuple(row.root) == view.relation.field_names() for row in rows
                )
                assert len(rows) == sum(
                    type(entity) is view.source for entity in case.stored
                )
            case compiler.PoisonedView(diagnostics=reported):
                for diagnostic in reported:
                    event(f"diagnostic: {type(diagnostic).__name__}")
                    assert test_compiler.owner_of(diagnostic) == view.name
                    assert diagnostic.describe().root
    match compiler.compile_views(graph, case.views.views):
        case Ok(all_compiled):
            assert test_compiler.shapes(all_compiled.views) == test_compiler.shapes(
                compiled
            )
        case Err(error):
            assert error.diagnostics == tuple(
                diagnostic
                for view in checked
                if isinstance(view, compiler.PoisonedView)
                for diagnostic in view.diagnostics
            )


class RoundTripCase(pydantic.BaseModel, frozen=True):
    models: test_compiler.Models
    source: type[pydantic.BaseModel]
    resolved: tuple[FieldName, ...]
    stored: tuple[pydantic.BaseModel, ...]


@st.composite
def round_trip_cases(draw: st.DrawFn) -> RoundTripCase:
    drawn_models = draw(test_compiler.models())
    graph = drawn_models.graph()
    # Only a graph without References gives a source that resolves nothing.
    referencing = [
        model for model in drawn_models.entities if graph.entities[model].references
    ]
    source = draw(st.sampled_from(referencing or drawn_models.entities))
    references = [reference.field for reference in graph.entities[source].references]
    resolved = (
        draw(st.lists(st.sampled_from(references), unique=True, min_size=1))
        if references
        else []
    )
    return RoundTripCase(
        models=drawn_models,
        source=source,
        resolved=tuple(resolved),
        stored=draw(test_compiler.stored_entities(drawn_models)),
    )


def with_ids_again(
    source: type[pydantic.BaseModel], relation: ir.Relation, row: engine.ViewRow
) -> pydantic.BaseModel:
    """The `source` instance `row` was built from: each resolved Entity is replaced
    by its ID, under the Reference's own name."""
    values: dict[str, object] = {}
    for field in relation.fields:
        value = row.root[field.name]
        match field.value:
            case ir.SourceField(field=name) | ir.RequiredField(field=name):
                values[name.root] = value
            case ir.Lookup(reference=reference):
                values[reference.field.root] = (
                    vars(value)["id"] if isinstance(value, pydantic.BaseModel) else None
                )
            case ir.Added() | ir.ReverseLookup():
                # Neither holds a value of the Source.
                pass
            case ir.ChainedLookup() | ir.Lifted():
                pytest.fail(f"a View that only resolves lifted {field.name.root}")
    return source(**values)


@given(round_trip_cases())
def test_resolving_then_replacing_entities_by_their_ids_gives_the_original(
    case: RoundTripCase,
) -> None:
    graph = case.models.graph()
    view = View(
        name=ViewName.fake(),
        source=case.source,
        step=steps.resolve(named(*case.resolved)),
    )
    compiled = compiler.compile_views(graph, [view]).unwrap()
    view_engine = engine.InMemoryViewEngine(graph, compiled)
    for entity in case.stored:
        view_engine.put(entity).unwrap()

    rows = view_engine.read(view.name).unwrap()

    (compiled_view,) = compiled.views
    absent = [
        field.name
        for field in compiled_view.relation.fields
        for row in rows
        if isinstance(field.value, ir.Lookup) and row.root[field.name] is None
    ]
    event(f"references resolved: {len(case.resolved)}")
    event(f"an optional reference is absent: {bool(absent)}")
    assert [
        with_ids_again(case.source, compiled_view.relation, row) for row in rows
    ] == [entity for entity in case.stored if type(entity) is case.source]


# where.


def mentees() -> tuple[Employee, Employee, Employee]:
    """A mentor-less employee, their mentee, and that mentee's mentee."""
    department_id = DepartmentId(1)
    first = Employee.fake(EmployeeId(1), department_id, None)
    second = Employee.fake(EmployeeId(2), department_id, first.id)
    third = Employee.fake(EmployeeId(3), department_id, second.id)
    return first, second, third


def mentor() -> ir.FieldRef:
    return conditions.field(FieldName("mentor_id"))


@pytest.mark.parametrize(
    ("condition", "kept"),
    [
        (conditions.eq(mentor(), EmployeeId(1)), [1]),
        (conditions.ne(mentor(), EmployeeId(1)), [0, 2]),
        (conditions.eq(mentor(), None), [0]),
        (conditions.ne(mentor(), None), [1, 2]),
        (conditions.is_null(mentor()), [0]),
        (conditions.not_(conditions.is_null(mentor())), [1, 2]),
        (
            conditions.or_(
                conditions.eq(mentor(), EmployeeId(1)), conditions.is_null(mentor())
            ),
            [0, 1],
        ),
        (
            conditions.and_(
                conditions.ne(mentor(), EmployeeId(1)),
                conditions.not_(conditions.is_null(mentor())),
            ),
            [2],
        ),
        (conditions.eq(conditions.field(FieldName("id")), mentor()), []),
        (conditions.ne(conditions.field(FieldName("id")), mentor()), [0, 1, 2]),
    ],
    ids=[
        "eq",
        "ne-is-true-for-none",
        "eq-none",
        "ne-none",
        "is-null",
        "not",
        "or",
        "and",
        "eq-field",
        "ne-field",
    ],
)
def test_a_filtered_view_returns_only_the_rows_its_condition_keeps(
    condition: ir.Condition, kept: list[int]
) -> None:
    view = employee_view(steps.where(condition))
    view_engine = hr_engine(view)
    employees = mentees()
    for employee in employees:
        view_engine.put(employee).unwrap()

    assert [row.root[FieldName("id")] for row in read(view_engine, view.name)] == [
        employees[index].id for index in kept
    ]


class PlayerId(Id[int]):
    pass


class Player(pydantic.BaseModel, frozen=True):
    id: PlayerId
    score: test_compiler.Score


@pytest.mark.parametrize(
    ("condition", "kept"),
    [
        (conditions.lt, [0]),
        (conditions.le, [0, 1]),
        (conditions.gt, [2]),
        (conditions.ge, [1, 2]),
    ],
    ids=["lt", "le", "gt", "ge"],
)
def test_an_ordering_condition_keeps_the_rows_on_its_side(
    condition: Callable[[ir.FieldRef, conditions.Operand], ir.Comparison],
    kept: list[int],
) -> None:
    view = View(
        name=ViewName.fake(),
        source=Player,
        step=steps.where(
            condition(conditions.field(FieldName("score")), test_compiler.Score(2))
        ),
    )
    graph = EntityGraph.build([Player], IdBase.fake(), OneToOneMarker.fake()).unwrap()
    view_engine = engine.InMemoryViewEngine(
        graph, compiler.compile_views(graph, [view]).unwrap()
    )
    players = [
        Player(id=PlayerId(index), score=test_compiler.Score(index + 1))
        for index in range(3)
    ]
    for player in players:
        view_engine.put(player).unwrap()

    assert [row.root[FieldName("id")] for row in read(view_engine, view.name)] == [
        players[index].id for index in kept
    ]


def test_an_excluded_row_is_no_view_result_even_with_a_dangling_reference() -> None:
    view = employee_view(
        steps.pipeline(steps.where(conditions.is_null(mentor())), steps.hydrate())
    )
    view_engine = hr_engine(view)
    company = Company.fake(CompanyId.fake())
    department = Department.fake(DepartmentId(1), company.id)
    kept = Employee.fake(EmployeeId(1), department.id, None)
    excluded = Employee.fake(EmployeeId(2), DepartmentId(99), kept.id)
    for entity in (company, department, kept, excluded):
        view_engine.put(entity).unwrap()

    assert [row.root[FieldName("id")] for row in read(view_engine, view.name)] == [
        kept.id
    ]


# non_nullable.


def unmet(
    view_engine: engine.InMemoryViewEngine, view: ViewName
) -> tuple[engine.ReadDiagnostic, ...]:
    match view_engine.read(view):
        case Err(DiagnosticsError() as error):
            return error.diagnostics
        case result:
            pytest.fail(f"expected unmet requirements, got {result}")


def test_every_unmet_requirement_is_reported_with_every_id() -> None:
    view = employee_view(steps.non_nullable("mentor_id"))
    view_engine = hr_engine(view)
    unmentored = (
        Employee.fake(EmployeeId(1), DepartmentId(1), None),
        Employee.fake(EmployeeId(2), DepartmentId(1), None),
    )
    mentored = Employee.fake(EmployeeId(3), DepartmentId(1), unmentored[0].id)
    for entity in (*unmentored, mentored):
        view_engine.put(entity).unwrap()

    assert unmet(view_engine, view.name) == (
        engine.UnmetRequirement(
            entity=Employee,
            field=FieldName("mentor_id"),
            ids=tuple(employee.id for employee in unmentored),
        ),
    )


def test_a_requirement_in_a_nested_pipeline_names_the_referenced_entity() -> None:
    view = employee_view(
        steps.resolve(named(FieldName("mentor_id")), steps.non_nullable("mentor_id"))
    )
    view_engine = hr_engine(view)
    # Only `mentor` is looked up, and it has no mentor of its own.
    mentor = Employee.fake(EmployeeId(1), DepartmentId(1), None)
    mentee = Employee.fake(EmployeeId(2), DepartmentId(1), mentor.id)
    for entity in (mentor, mentee):
        view_engine.put(entity).unwrap()

    assert unmet(view_engine, view.name) == (
        engine.UnmetRequirement(
            entity=Employee, field=FieldName("mentor_id"), ids=(mentor.id,)
        ),
    )


def test_a_required_reference_that_is_absent_is_unmet() -> None:
    view = employee_view(
        steps.pipeline(
            steps.resolve(named(FieldName("mentor_id"))),
            steps.non_nullable("mentor"),
        )
    )
    view_engine = hr_engine(view)
    unmentored = Employee.fake(EmployeeId(1), DepartmentId(1), None)
    view_engine.put(unmentored).unwrap()

    assert unmet(view_engine, view.name) == (
        engine.UnmetRequirement(
            entity=Employee, field=FieldName("mentor_id"), ids=(unmentored.id,)
        ),
    )


def test_a_required_reverse_that_finds_nothing_is_unmet() -> None:
    view = employee_view(
        steps.pipeline(steps.reverse(Profile), steps.non_nullable("profile"))
    )
    view_engine = profile_engine(view)
    with_profile = Employee.fake(EmployeeId(1), DepartmentId(1), None)
    without_profiles = (
        Employee.fake(EmployeeId(2), DepartmentId(1), None),
        Employee.fake(EmployeeId(3), DepartmentId(1), None),
    )
    profile = Profile.fake(ProfileId(1), with_profile.id)
    for entity in (with_profile, *without_profiles, profile):
        view_engine.put(entity).unwrap()

    assert unmet(view_engine, view.name) == (
        engine.UnmetReverseRequirement(
            entity=Profile,
            via=FieldName("employee_id"),
            ids=tuple(employee.id for employee in without_profiles),
        ),
    )


def test_a_met_requirement_gives_the_value() -> None:
    view = employee_view(
        steps.pipeline(
            steps.pick(["id", "mentor_id"]),
            steps.non_nullable("mentor_id"),
        )
    )
    view_engine = hr_engine(view)
    mentor = EmployeeId(2)
    employee = Employee.fake(EmployeeId(1), DepartmentId(1), mentor)
    view_engine.put(employee).unwrap()

    assert read(view_engine, view.name) == (
        engine.ViewRow({FieldName("id"): employee.id, FieldName("mentor_id"): mentor}),
    )


def test_an_excluded_row_is_not_checked_for_requirements() -> None:
    kept = EmployeeId(2)
    view = employee_view(
        steps.pipeline(
            steps.where(conditions.eq(conditions.field(FieldName("id")), kept)),
            steps.non_nullable("mentor_id"),
        )
    )
    view_engine = hr_engine(view)
    for employee_id in (EmployeeId(1), kept):
        view_engine.put(Employee.fake(employee_id, DepartmentId(1), None)).unwrap()

    assert unmet(view_engine, view.name) == (
        engine.UnmetRequirement(
            entity=Employee, field=FieldName("mentor_id"), ids=(kept,)
        ),
    )


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            engine.UnmetRequirement(
                entity=Employee,
                field=FieldName("mentor_id"),
                ids=(EmployeeId(1), EmployeeId(2)),
            ),
            "Employee.mentor_id is required, but is None for IDs: 1, 2",
        ),
        (
            engine.UnmetReverseRequirement(
                entity=Profile, via=FieldName("employee_id"), ids=(EmployeeId(1),)
            ),
            (
                "A Profile referencing each subject by Profile.employee_id is "
                "required, but none references IDs: 1"
            ),
        ),
    ],
    ids=["field", "reverse"],
)
def test_an_unmet_requirement_names_the_entity_the_field_and_the_ids(
    diagnostic: engine.ReadDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


def test_a_required_field_lifted_through_a_missing_entity_is_unmet() -> None:
    view = employee_view(
        steps.pipeline(
            steps.pick(["id"]),
            steps.reverse(Profile, steps.pick(["bio"])),
            steps.flatten("profile"),
            steps.non_nullable("bio"),
        )
    )
    view_engine = profile_engine(view)
    with_profile = Employee.fake(EmployeeId(1), DepartmentId(1), None)
    without_profile = Employee.fake(EmployeeId(2), DepartmentId(1), None)
    profile = Profile.fake(ProfileId(1), with_profile.id)
    for entity in (with_profile, without_profile, profile):
        view_engine.put(entity).unwrap()

    assert unmet(view_engine, view.name) == (
        engine.UnmetReverseRequirement(
            entity=Profile, via=FieldName("employee_id"), ids=(without_profile.id,)
        ),
    )


# Unions.


class TicketStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    ARCHIVED = "archived"


class TicketId(Id[int]):
    pass


class Ticket(pydantic.BaseModel, frozen=True):
    id: TicketId
    status: TicketStatus
    assignee: Text | None

    @staticmethod
    def fake(ticket_id: TicketId, status: TicketStatus) -> Ticket:
        return Ticket(id=ticket_id, status=status, assignee=None)


def ticket_view(name: ViewName, *then: steps.Step) -> View:
    return View(name=name, source=Ticket, step=steps.pipeline(*then))


def has_status(status: TicketStatus) -> steps.Step:
    return steps.where(conditions.eq(conditions.field(FieldName("status")), status))


def open_or_closed() -> Union:
    return Union(
        name=UnionName("Tickets"),
        discriminator=FieldName("status"),
        variants=(
            ticket_view(ViewName("OpenTicket"), has_status(TicketStatus.OPEN)),
            ticket_view(
                ViewName("ClosedTicket"),
                has_status(TicketStatus.CLOSED),
                steps.non_nullable("assignee"),
            ),
        ),
    )


def union_engine(*unions: Union) -> engine.InMemoryViewEngine:
    graph = EntityGraph.build(
        [Ticket, Company], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    match compiler.compile_views(graph, [], unions):
        case Ok(compiled):
            return engine.InMemoryViewEngine(graph, compiled)
        case Err(error):
            pytest.fail(f"expected a compiled Union, got {error}")


def ticket_row(ticket: Ticket) -> engine.ViewRow:
    return engine.ViewRow(
        {FieldName(name): value for name, value in vars(ticket).items()}
    )


def test_each_row_is_the_result_of_the_variant_keeping_it() -> None:
    union = open_or_closed()
    view_engine = union_engine(union)
    closed = Ticket(id=TicketId(1), status=TicketStatus.CLOSED, assignee=Text("Ada"))
    opened = Ticket.fake(TicketId(2), TicketStatus.OPEN)
    for ticket in (closed, opened):
        view_engine.put(ticket).unwrap()

    assert view_engine.read_union(union.name).unwrap() == (
        engine.VariantRow(view=ViewName("ClosedTicket"), row=ticket_row(closed)),
        engine.VariantRow(view=ViewName("OpenTicket"), row=ticket_row(opened)),
    )


def test_a_row_no_variant_keeps_is_no_result() -> None:
    union = open_or_closed()
    view_engine = union_engine(union)
    view_engine.put(Ticket.fake(TicketId(1), TicketStatus.ARCHIVED)).unwrap()

    assert view_engine.read_union(union.name).unwrap() == ()


def test_the_first_variant_keeping_a_row_gives_its_result() -> None:

    union = Union(
        name=UnionName("Tagged"),
        discriminator=FieldName("kind"),
        variants=(
            ticket_view(
                ViewName("Open"),
                has_status(TicketStatus.OPEN),
                test_compiler.added_tag(FieldName("kind"), test_compiler.Tag("open")),
            ),
            ticket_view(
                ViewName("Any"),
                test_compiler.added_tag(FieldName("kind"), test_compiler.Tag("any")),
            ),
        ),
    )
    view_engine = union_engine(union)
    opened = Ticket.fake(TicketId(1), TicketStatus.OPEN)
    closed = Ticket.fake(TicketId(2), TicketStatus.CLOSED)
    for ticket in (opened, closed):
        view_engine.put(ticket).unwrap()

    assert [row.view for row in view_engine.read_union(union.name).unwrap()] == [
        ViewName("Open"),
        ViewName("Any"),
    ]


def test_a_union_reports_the_unmet_requirements_of_the_variants_it_gives() -> None:
    union = open_or_closed()
    view_engine = union_engine(union)
    unassigned_closed = Ticket.fake(TicketId(1), TicketStatus.CLOSED)
    unassigned_open = Ticket.fake(TicketId(2), TicketStatus.OPEN)
    for ticket in (unassigned_closed, unassigned_open):
        view_engine.put(ticket).unwrap()

    match view_engine.read_union(union.name):
        case Err(DiagnosticsError() as error):
            assert error.diagnostics == (
                engine.UnmetRequirement(
                    entity=Ticket,
                    field=FieldName("assignee"),
                    ids=(unassigned_closed.id,),
                ),
            )
        case result:
            pytest.fail(f"expected an unmet requirement, got {result}")


def test_reading_an_unknown_union_names_it_and_the_known_unions() -> None:
    known = open_or_closed()
    unknown = UnionName(f"{known.name.root}Typo")

    match union_engine(known).read_union(unknown):
        case Err(engine.UnknownUnionError() as error):
            assert (error.union, error.known) == (unknown, (known.name,))
        case result:
            pytest.fail(f"expected UnknownUnionError, got {result}")


def test_reading_a_union_over_a_source_that_is_not_an_entity_is_an_error() -> None:
    def memo(name: ViewName, tag: Text) -> View:
        return View(
            name=name,
            source=Memo,
            step=steps.pipeline(
                steps.where(conditions.eq(conditions.field(FieldName("body")), tag)),
                test_compiler.added_tag(FieldName("kind"), test_compiler.Tag(tag.root)),
            ),
        )

    union = Union(
        name=UnionName("Memos"),
        discriminator=FieldName("kind"),
        variants=(
            memo(ViewName("Short"), Text("short")),
            memo(ViewName("Long"), Text("long")),
        ),
    )

    match union_engine(union).read_union(union.name):
        case Err(engine.NonEntitySourceError() as error):
            assert (error.view, error.source) == (union.name, Memo)
        case result:
            pytest.fail(f"expected NonEntitySourceError, got {result}")


class Seniority(enum.StrEnum):
    JUNIOR = "junior"
    SENIOR = "senior"


class StaffId(Id[int]):
    pass


class Staff(pydantic.BaseModel, frozen=True):
    id: StaffId
    department_id: DepartmentId
    seniority: Seniority
    nickname: Text | None


class ManagerId(Id[int]):
    pass


class Manager(pydantic.BaseModel, frozen=True):
    """A child of `Staff`: it inherits every field of `Staff` but `id`."""

    id: ManagerId
    staff_id: Annotated[StaffId, ParentLink()]
    title: Text


def staff_engine(*views: View) -> engine.InMemoryViewEngine:
    graph = EntityGraph.build(
        [Company, Department, Staff, Manager], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    match compiler.compile_views(graph, views):
        case Ok(compiled):
            return engine.InMemoryViewEngine(graph, compiled)
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def manager_view(step: steps.Step) -> View:
    return View(name=ViewName.fake(), source=Manager, step=step)


def managing(staff: Staff, title: Text) -> Manager:
    return Manager(id=ManagerId(staff.id.root), staff_id=staff.id, title=title)


def test_a_view_over_a_child_filters_on_and_reads_its_parents_fields() -> None:
    view = manager_view(
        steps.pipeline(
            steps.where(
                conditions.eq(
                    conditions.field(FieldName("seniority")), Seniority.SENIOR
                )
            ),
            steps.non_nullable("nickname"),
            steps.omit("staff_id"),
        )
    )
    view_engine = staff_engine(view)
    department = Department.fake(DepartmentId(1), CompanyId.fake())
    junior, senior = (
        Staff(
            id=StaffId(index),
            department_id=department.id,
            seniority=seniority,
            nickname=Text(seniority.value),
        )
        for index, seniority in enumerate(Seniority)
    )
    for entity in (
        department,
        senior,
        junior,
        managing(senior, Text("head")),
        managing(junior, Text("lead")),
    ):
        view_engine.put(entity).unwrap()

    assert read(view_engine, view.name) == (
        engine.ViewRow(
            {
                FieldName("id"): ManagerId(senior.id.root),
                FieldName("department_id"): department.id,
                FieldName("seniority"): Seniority.SENIOR,
                FieldName("nickname"): senior.nickname,
                FieldName("title"): Text("head"),
            }
        ),
    )


def test_a_reference_a_child_inherits_resolves_to_its_entity() -> None:
    view = manager_view(steps.resolve("department_id"))
    view_engine = staff_engine(view)
    department = Department.fake(DepartmentId(1), CompanyId.fake())
    staff = Staff(
        id=StaffId(1),
        department_id=department.id,
        seniority=Seniority.SENIOR,
        nickname=None,
    )
    for entity in (department, staff, managing(staff, Text("head"))):
        view_engine.put(entity).unwrap()

    assert [
        row.root[FieldName("department")] for row in read(view_engine, view.name)
    ] == [department]


def test_a_child_without_its_parent_is_a_dangling_parent_link() -> None:
    view = manager_view(steps.pipeline())
    view_engine = staff_engine(view)
    orphan = Manager(id=ManagerId(7), staff_id=StaffId(7), title=Text("head"))
    view_engine.put(orphan).unwrap()

    match view_engine.read(view.name):
        case Err(DiagnosticsError(diagnostics=reported)):
            assert reported == (
                engine.DanglingReference(
                    entity=Manager,
                    field=FieldName("staff_id"),
                    missing=(orphan.staff_id,),
                ),
            )
        case result:
            pytest.fail(f"expected a dangling parent link, got {result}")


def test_put_rejects_a_child_whose_id_is_not_its_parent_links() -> None:
    staff_id = StaffId(1)
    child = Manager(
        id=ManagerId(staff_id.root + 1), staff_id=staff_id, title=Text("head")
    )

    match staff_engine().put(child):
        case Err(engine.ChildIdMismatchError() as error):
            assert (error.entity, error.parent_link) == (child, FieldName("staff_id"))
        case result:
            pytest.fail(f"expected ChildIdMismatchError, got {result}")
