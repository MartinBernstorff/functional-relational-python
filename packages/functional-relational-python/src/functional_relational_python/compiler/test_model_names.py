import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler, diagnostics
from functional_relational_python.compiler.model_names import (
    CandidateName,
    FunctionName,
    InvalidFieldName,
    InvalidModelName,
    InvalidNestedModelName,
    ModelName,
    ModelNames,
    ViewModelNames,
)
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
)
from functional_relational_python.entity_graph.id import Id
from functional_relational_python.pipeline import ir, steps
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.support.diagnostic import DiagnosticMessage


class Text(pydantic.RootModel[str], frozen=True):
    pass


class DepartmentId(Id[int]):
    pass


class Department(pydantic.BaseModel):
    id: DepartmentId
    name: Text


class Employee(pydantic.BaseModel):
    name: Text
    department_id: DepartmentId


class Keyworded(pydantic.BaseModel):
    """`class_id` would resolve to `class`, a keyword."""

    class_id: DepartmentId


# A Reference whose name is no identifier, so its nested model's name is none either.
Hyphenated = pydantic.create_model("Hyphenated", **{"head-office": (DepartmentId, ...)})


def graph() -> EntityGraph:
    return EntityGraph.build([Department]).unwrap()


def compiled_names(
    view: View,
) -> ViewModelNames | tuple[diagnostics.CompileDiagnostic, ...]:
    match compiler.compile_views(graph(), [view]):
        case Ok(compiled):
            return compiled.views[0].names
        case Err(error):
            return error.diagnostics


def names(model: ModelName, function: FunctionName) -> ViewModelNames:
    return ViewModelNames(model=ModelNames(name=model, nested={}), function=function)


def view_named(name: ViewName) -> View:
    return View(name=name, source=Employee, step=steps.pipeline())


def renaming_to(new: FieldName) -> View:
    return View(
        name=ViewName.fake(),
        source=Employee,
        step=steps.pipeline(steps.rename(FieldName("name"), new), steps.omit(new)),
    )


def invalid_field(
    source: type[pydantic.BaseModel], field: FieldName
) -> InvalidFieldName:
    return InvalidFieldName(
        view=ViewName.fake(),
        source=ir.SourceName(source.__name__),
        field=field,
    )


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        (
            view_named(ViewName("CompanyPublic")),
            names(ModelName("CompanyPublic"), FunctionName("company_public")),
        ),
        (
            view_named(ViewName("Company2Public")),
            names(ModelName("Company2Public"), FunctionName("company2_public")),
        ),
        (
            view_named(ViewName("HTTPServer")),
            names(ModelName("HTTPServer"), FunctionName("httpserver")),
        ),
        (
            View(
                name=ViewName("EmployeeOverview"),
                source=Employee,
                step=steps.hydrate(steps.omit(FieldName("name"))),
            ),
            ViewModelNames(
                model=ModelNames(
                    name=ModelName("EmployeeOverview"),
                    nested={
                        FieldName("department"): ModelNames(
                            name=ModelName("EmployeeOverviewDepartment"), nested={}
                        )
                    },
                ),
                function=FunctionName("employee_overview"),
            ),
        ),
        *(
            (view_named(ViewName(name)), (InvalidModelName(view=ViewName(name)),))
            for name in ["None", "True", "False"]
        ),
        (
            view_named(ViewName("Class")),
            names(ModelName("Class"), FunctionName("class_")),
        ),
        (
            view_named(ViewName("not a class")),
            (InvalidModelName(view=ViewName("not a class")),),
        ),
        *(
            (renaming_to(FieldName(new)), (invalid_field(Employee, FieldName(new)),))
            for new in [
                "_private",
                "class",
                "not-an-identifier",
                "json",
                "model_config",
            ]
        ),
        (
            View(name=ViewName.fake(), source=Keyworded, step=steps.hydrate()),
            (invalid_field(Keyworded, FieldName("class")),),
        ),
        (
            View(
                name=ViewName.fake(),
                source=Hyphenated,
                step=steps.hydrate(steps.omit(FieldName("name"))),
            ),
            (
                InvalidNestedModelName(
                    view=ViewName.fake(),
                    field=FieldName("head-office"),
                    name=CandidateName("CompanySummaryHead-office"),
                ),
            ),
        ),
    ],
    ids=[
        "model-and-function",
        "digit-before-capital",
        "acronym",
        "nested-model",
        "view-named-None",
        "view-named-True",
        "view-named-False",
        "function-is-keyword",
        "not-a-class-name",
        "field-private",
        "field-keyword",
        "field-not-an-identifier",
        "field-shadows-basemodel",
        "field-reserved",
        "resolved-field-keyword",
        "nested-model-not-an-identifier",
    ],
)
def test_the_names_a_view_gets_or_why_it_cannot_have_them(
    view: View, expected: ViewModelNames | tuple[diagnostics.CompileDiagnostic, ...]
) -> None:
    assert compiled_names(view) == expected


def test_a_compile_defect_and_an_invalid_view_name_are_reported_together() -> None:
    unknown = FieldName("nmae")
    defective = View(name=ViewName.fake(), source=Employee, step=steps.omit(unknown))
    badly_named = view_named(ViewName("not-a-class"))

    match compiler.compile_views(graph(), [defective, badly_named]):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            assert error.diagnostics == (
                diagnostics.UnknownField(
                    view=defective.name,
                    source=ir.SourceName("Employee"),
                    field=unknown,
                    valid=tuple(map(FieldName, Employee.model_fields)),
                ),
                InvalidModelName(view=badly_named.name),
            )


def test_a_view_that_does_not_compile_still_reports_its_invalid_name() -> None:
    unknown = FieldName("nmae")
    view = View(name=ViewName("None"), source=Employee, step=steps.omit(unknown))

    assert compiled_names(view) == (
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Employee"),
            field=unknown,
            valid=tuple(map(FieldName, Employee.model_fields)),
        ),
        InvalidModelName(view=view.name),
    )


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            InvalidModelName(view=ViewName("not a class")),
            "not a class: the name is not a valid Python class name.",
        ),
        (
            InvalidNestedModelName(
                view=ViewName("CompanySummary"),
                field=FieldName("head-office"),
                name=CandidateName("CompanySummaryHead-office"),
            ),
            (
                "CompanySummary.head-office: the nested model would be named "
                "'CompanySummaryHead-office', which is not a valid Python class name."
            ),
        ),
        (
            InvalidFieldName(
                view=ViewName("KeywordedView"),
                source=ir.SourceName("Keyworded"),
                field=FieldName("class"),
            ),
            (
                "KeywordedView: 'class' cannot be a field name. A field name is a "
                "Python identifier that is not a keyword, does not start with '_', "
                "and is not an attribute of pydantic.BaseModel."
            ),
        ),
    ],
    ids=["model", "nested-model", "field"],
)
def test_a_name_diagnostic_message_names_the_offending_values(
    diagnostic: diagnostics.CompileDiagnostic, message: str
) -> None:
    assert diagnostic.describe() == DiagnosticMessage(message)
