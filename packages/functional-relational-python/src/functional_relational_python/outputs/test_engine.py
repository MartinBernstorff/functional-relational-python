import enum

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
from functional_relational_python.entity_graph.id import Id, IdBase
from functional_relational_python.outputs import engine
from functional_relational_python.pipeline import ir, steps
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
    graph = EntityGraph.build([Company], IdBase.fake()).unwrap()
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
    view = view_of(steps.pipeline(*map(steps.omit, omitted)))
    view_engine = make_engine(view)
    companies = [Company.fake(CompanyId(1)), Company.fake(CompanyId(2))]
    for company in companies:
        view_engine.put(company).unwrap()

    assert read(view_engine, view.name) == tuple(
        row_of(company, *omitted) for company in companies
    )


def test_a_renamed_field_keeps_its_value_and_position_under_the_new_name() -> None:
    new = FieldName("title")
    view = view_of(steps.rename(FieldName("name"), new))
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
    view = view_of(steps.omit(steps.select(steps.is_primary_key)))
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


def test_a_projected_view_returns_each_entity_with_only_the_listed_fields() -> None:
    view = view_of(steps.project([FieldName("name")]))
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
        steps.pipeline(
            steps.project([FieldName("name")]), steps.add(added, object, default)
        )
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
    public = View(
        name=ViewName("Public"), source=Company, step=steps.omit(FieldName("secret"))
    )
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
        case Err(error):
            assert error.model is Memo


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
    graph = EntityGraph.build([Company, Department, Employee], IdBase.fake()).unwrap()
    match compiler.compile_views(graph, views):
        case Ok(compiled):
            return engine.InMemoryViewEngine(graph, compiled)
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def employee_view(step: steps.Step) -> View:
    return View(name=ViewName.fake(), source=Employee, step=step)


def named(*fields: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in fields)


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
            steps.omit(FieldName("id")),
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


def test_flatten_lifts_the_referenced_entitys_values_into_each_row() -> None:
    view = employee_view(
        steps.pipeline(
            steps.resolve(
                named(FieldName("department_id")),
                steps.omit(FieldName("id")),
                steps.hydrate(),
            ),
            steps.flatten(FieldName("department")),
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
                    steps.select(lambda _field: True),
                    lambda field: FieldName(f"mentor_{field.root}"),
                ),
            ),
            steps.flatten(FieldName("mentor")),
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


def test_a_dangling_reference_lifted_out_is_reported_on_its_entity() -> None:
    view = employee_view(
        steps.pipeline(
            steps.resolve(
                named(FieldName("department_id")),
                steps.omit(FieldName("id")),
                steps.hydrate(),
            ),
            steps.flatten(FieldName("department")),
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
        graph, compiler.CompiledViews(views=compiled)
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
                    assert diagnostic.view == view.name
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
            case ir.SourceField(field=name):
                values[name.root] = value
            case ir.Lookup(reference=reference):
                values[reference.field.root] = (
                    vars(value)["id"] if isinstance(value, pydantic.BaseModel) else None
                )
            case ir.Added():
                # An added field holds no value of the Source.
                pass
            case ir.Lifted():
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
