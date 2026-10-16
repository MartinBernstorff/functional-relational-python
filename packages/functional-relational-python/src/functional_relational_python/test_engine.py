import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python import compiler, engine, ir, steps
from functional_relational_python.entity_graph import EntityGraph, FieldName
from functional_relational_python.id import Id
from functional_relational_python.views import View, ViewName


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
    graph = EntityGraph.build([Company]).unwrap()
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


def test_a_projected_view_returns_each_entity_with_only_the_listed_fields() -> None:
    view = view_of(steps.project([FieldName("name")]))
    view_engine = make_engine(view)
    company = Company.fake(CompanyId.fake())
    view_engine.put(company).unwrap()

    assert read(view_engine, view.name) == (
        engine.ViewRow({FieldName("name"): company.name}),
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
    graph = EntityGraph.build([Company, Department, Employee]).unwrap()
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
        case Err(engine.DanglingReferenceError() as error):
            assert error.dangling == (
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
            pytest.fail(f"expected DanglingReferenceError, got {result}")


def test_a_dangling_reference_names_the_entity_the_field_and_the_ids() -> None:
    dangling = engine.DanglingReference(
        entity=Employee,
        field=FieldName("department_id"),
        missing=(DepartmentId(2), DepartmentId(3)),
    )

    assert dangling.describe().root == (
        "Employee.department_id references missing IDs: 2, 3"
    )
