import contextlib
import sys
import types
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import ClassVar

import pydantic
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler
from functional_relational_python.compiler.test_compiler import Depth, pipelines
from functional_relational_python.domain.entity_graph import EntityGraph, FieldName
from functional_relational_python.domain.id import Id
from functional_relational_python.language import ir, steps
from functional_relational_python.language.views import View, ViewName
from functional_relational_python.outputs import codegen, emitter, engine
from functional_relational_python.outputs.codegen import Codegen, ModulePath


class Text(pydantic.RootModel[str], frozen=True):
    pass


class CompanyId(Id[int]):
    pass


class Company(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text
    secret: Text


class WideId(Id[int]):
    pass


class Wide(pydantic.BaseModel, frozen=True):
    id: WideId
    alpha: Text
    beta: Text | None
    gamma: Text
    delta: Text


class DepartmentId(Id[int]):
    pass


class Department(pydantic.BaseModel, frozen=True):
    id: DepartmentId
    name: Text
    company_id: CompanyId


class EmployeeId(Id[int]):
    pass


class Employee(pydantic.BaseModel, frozen=True):
    id: EmployeeId
    name: Text
    department_id: DepartmentId
    mentor_id: EmployeeId | None


class ArchivedDepartmentId(Id[int]):
    pass


class Archive:
    class Department(pydantic.BaseModel, frozen=True):
        """Named like the module-level Department, so both need one parameter."""

        id: ArchivedDepartmentId


class TransferId(Id[int]):
    pass


class Transfer(pydantic.BaseModel, frozen=True):
    id: TransferId
    old_department_id: DepartmentId
    archived_department_id: ArchivedDepartmentId


# The View models the golden module refers to. Codegen (MB-182) generates these;
# here they are written by hand so the golden module type checks.
class CompanyFull(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text
    secret: Text


class CompanyPublic(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text


class CompanyName(pydantic.BaseModel, frozen=True):
    name: Text


class CompanyTitle(pydantic.BaseModel, frozen=True):
    id: CompanyId
    title: Text


class CompanySecret(pydantic.BaseModel, frozen=True):
    secret: Text


class EmployeeHydrated(pydantic.BaseModel, frozen=True):
    id: EmployeeId
    name: Text
    department: Department
    mentor: Employee | None


class EmployeeOverviewDepartment(pydantic.BaseModel, frozen=True):
    id: DepartmentId
    name: Text
    company: Company


class EmployeeOverview(pydantic.BaseModel, frozen=True):
    id: EmployeeId
    name: Text
    department: EmployeeOverviewDepartment
    mentor_id: EmployeeId | None


class EmployeeMentorMentor(pydantic.BaseModel, frozen=True):
    name: Text


class EmployeeMentor(pydantic.BaseModel, frozen=True):
    mentor: EmployeeMentorMentor | None


def entity_graph() -> EntityGraph:
    return EntityGraph.build([Company, Wide, Department, Employee]).unwrap()


def named(*names: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in names)


def compile_all(views: Iterable[View]) -> compiler.CompiledViews:
    match compiler.compile_views(entity_graph(), views):
        case Ok(compiled):
            return compiled
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def this_module(_view: compiler.CompiledView) -> ModulePath:
    return ModulePath(__name__)


def emit(views: Iterable[View]) -> emitter.EmittedModule:
    match emitter.PythonEmitter.emit(compile_all(views), this_module):
        case Ok(emitted):
            return emitted
        case Err(error):
            pytest.fail(f"expected an emitted module, got {error}")


def emit_diagnostics(views: Iterable[View]) -> tuple[emitter.EmitDiagnostic, ...]:
    match emitter.PythonEmitter.emit(compile_all(views), this_module):
        case Ok(emitted):
            pytest.fail(f"expected diagnostics, got {emitted.root}")
        case Err(error):
            return error.diagnostics


def golden_views() -> list[View]:
    return [
        View(name=ViewName("CompanyFull"), source=Company, step=steps.pipeline()),
        View(
            name=ViewName("CompanyPublic"),
            source=Company,
            step=steps.omit(FieldName("secret")),
        ),
        View(
            name=ViewName("CompanyName"),
            source=Company,
            step=steps.pipeline(
                steps.omit(FieldName("secret")), steps.omit(FieldName("id"))
            ),
        ),
        View(
            name=ViewName("CompanyTitle"),
            source=Company,
            step=steps.pipeline(
                steps.omit(FieldName("secret")),
                steps.rename(FieldName("name"), FieldName("title")),
            ),
        ),
        View(
            name=ViewName("CompanySecret"),
            source=Company,
            step=steps.project([FieldName("secret")]),
        ),
        View(name=ViewName("EmployeeHydrated"), source=Employee, step=steps.hydrate()),
        View(
            name=ViewName("EmployeeOverview"),
            source=Employee,
            step=steps.resolve(named(FieldName("department_id")), steps.hydrate()),
        ),
        View(
            name=ViewName("EmployeeMentor"),
            source=Employee,
            step=steps.pipeline(
                steps.resolve(
                    named(FieldName("mentor_id")),
                    steps.project([FieldName("name")]),
                ),
                steps.project([FieldName("mentor")]),
            ),
        ),
    ]


def test_the_emitted_module_matches_the_golden_file() -> None:
    """The golden file is type checked with the rest of the package (`:typecheck`).

    To regenerate it, write `emit(golden_views()).root` to the file.
    """
    golden = Path(__file__).with_name("test_emitter_golden.py")

    assert emit(golden_views()).root == golden.read_text()


@pytest.mark.parametrize(
    ("view", "function"),
    [
        ("CompanyPublic", "company_public"),
        ("Company", "company"),
        ("Company2Public", "company2_public"),
        ("HTTPServer", "httpserver"),
    ],
)
def test_a_function_is_named_after_its_view_in_snake_case(
    view: str, function: str
) -> None:
    assert emitter.PythonEmitter.function_name(ViewName(view)) == (
        emitter.BoundName(function)
    )


def test_a_source_that_cannot_be_imported_is_a_diagnostic() -> None:
    class Local(pydantic.BaseModel, frozen=True):
        name: Text

    view = View(name=ViewName.fake(), source=Local, step=steps.pipeline())

    assert emit_diagnostics([view]) == (
        emitter.UnimportableSource(view=view.name, source=Local),
    )


def test_a_view_name_that_is_no_function_name_is_a_diagnostic() -> None:
    view = View(name=ViewName("Class"), source=Company, step=steps.pipeline())

    assert emit_diagnostics([view]) == (
        emitter.InvalidFunctionName(
            view=view.name, function=emitter.PythonEmitter.function_name(view.name)
        ),
    )


def test_a_view_name_that_is_neither_a_class_nor_a_function_name_reports_both() -> None:
    view = View(name=ViewName("Company Public"), source=Company, step=steps.pipeline())

    assert emit_diagnostics([view]) == (
        codegen.InvalidModelName(view=view.name),
        emitter.InvalidFunctionName(
            view=view.name, function=emitter.PythonEmitter.function_name(view.name)
        ),
    )


@pytest.mark.parametrize("name", ["None", "True", "False"])
def test_a_view_name_that_is_no_class_name_is_a_diagnostic(name: str) -> None:
    view = View(name=ViewName(name), source=Company, step=steps.pipeline())

    assert emit_diagnostics([view]) == (codegen.InvalidModelName(view=view.name),)


def test_views_with_the_same_function_name_are_a_diagnostic() -> None:
    views = [
        View(name=ViewName(name), source=Company, step=steps.pipeline())
        for name in ["CompanyPublic", "companyPublic"]
    ]

    assert emit_diagnostics(views) == (
        emitter.DuplicateFunctionName(
            function=emitter.BoundName("company_public"),
            views=tuple(view.name for view in views),
        ),
    )


def test_a_function_named_like_an_imported_module_is_a_diagnostic() -> None:
    root = __name__.split(".")[0]
    view = View(
        name=ViewName(root.title().replace("_", "")),
        source=Company,
        step=steps.pipeline(),
    )

    assert emit_diagnostics([view]) == (
        emitter.ShadowedModule(view=view.name, name=emitter.BoundName(root)),
    )


def test_a_view_model_module_named_like_the_parameter_is_a_diagnostic() -> None:
    view = View(name=ViewName.fake(), source=Company, step=steps.pipeline())

    match emitter.PythonEmitter.emit(
        compile_all([view]),
        lambda _view: ModulePath(f"{emitter.PythonEmitter.parameter.root}.views"),
    ):
        case Ok(emitted):
            pytest.fail(f"expected diagnostics, got {emitted.root}")
        case Err(error):
            assert error.diagnostics == (
                emitter.ShadowedModule(
                    view=view.name, name=emitter.PythonEmitter.parameter
                ),
            )


def test_a_view_model_module_named_like_a_lookup_parameter_is_a_diagnostic() -> None:
    view = View(name=ViewName.fake(), source=Employee, step=steps.hydrate())
    parameter = emitter.PythonEmitter.lookup_parameter(Department)

    match emitter.PythonEmitter.emit(
        compile_all([view]), lambda _view: ModulePath(f"{parameter.root}.views")
    ):
        case Ok(emitted):
            pytest.fail(f"expected diagnostics, got {emitted.root}")
        case Err(error):
            assert error.diagnostics == (
                emitter.ShadowedModule(view=view.name, name=parameter),
            )


def hydrated_diagnostics(
    entities: Iterable[type[pydantic.BaseModel]], source: type[pydantic.BaseModel]
) -> tuple[emitter.EmitDiagnostic, ...]:
    graph = EntityGraph.build(entities).unwrap()
    view = View(name=ViewName.fake(), source=source, step=steps.hydrate())
    match emitter.PythonEmitter.emit(
        compiler.compile_views(graph, [view]).unwrap(), this_module
    ):
        case Ok(emitted):
            pytest.fail(f"expected diagnostics, got {emitted.root}")
        case Err(error):
            return error.diagnostics


def test_entities_whose_names_give_the_same_lookup_parameter_are_a_diagnostic() -> None:
    assert hydrated_diagnostics(
        [Company, Department, Archive.Department, Transfer], Transfer
    ) == (
        emitter.DuplicateLookupParameter(
            view=ViewName.fake(),
            parameter=emitter.BoundName("department_by_id"),
            entities=(Department, Archive.Department),
        ),
    )


def test_a_looked_up_entity_that_cannot_be_imported_is_a_diagnostic() -> None:
    class TeamId(Id[int]):
        pass

    class Team(pydantic.BaseModel, frozen=True):
        id: TeamId

    class Player(pydantic.BaseModel, frozen=True):
        team_id: TeamId

    assert hydrated_diagnostics([Team], Player) == (
        emitter.UnimportableSource(view=ViewName.fake(), source=Player),
        emitter.UnimportableReferencedModel(view=ViewName.fake(), model=Team),
        emitter.UnimportableReferencedModel(view=ViewName.fake(), model=TeamId),
    )


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            emitter.UnimportableSource(view=ViewName("CompanyPublic"), source=Company),
            (
                f"CompanyPublic: the Source {__name__}.Company cannot be imported "
                "by its module and name. Define it at module level."
            ),
        ),
        (
            emitter.InvalidFunctionName(
                view=ViewName("Company Public"),
                function=emitter.BoundName("company public"),
            ),
            "Company Public: 'company public' is not a valid Python function name.",
        ),
        (
            emitter.DuplicateFunctionName(
                function=emitter.BoundName("company_public"),
                views=(ViewName("CompanyPublic"), ViewName("companyPublic")),
            ),
            (
                "Views CompanyPublic, companyPublic all emit a function named "
                "'company_public'."
            ),
        ),
        (
            emitter.ShadowedModule(view=ViewName("App"), name=emitter.BoundName("app")),
            "App: the emitted name 'app' would shadow the imported module 'app'.",
        ),
        (
            emitter.UnimportableReferencedModel(
                view=ViewName("TransferView"), model=Department
            ),
            (
                f"TransferView: {__name__}.Department cannot be imported by its "
                "module and name. Define it at module level."
            ),
        ),
        (
            emitter.DuplicateLookupParameter(
                view=ViewName("TransferView"),
                parameter=emitter.BoundName("department_by_id"),
                entities=(Department, Archive.Department),
            ),
            (
                f"TransferView: Entities {__name__}.Department, "
                f"{__name__}.Archive.Department all need a parameter named "
                "'department_by_id'."
            ),
        ),
    ],
    ids=[
        "unimportable",
        "invalid",
        "duplicate",
        "shadowed",
        "unimportable-referenced",
        "duplicate-lookup",
    ],
)
def test_a_diagnostic_message_names_the_offending_values(
    diagnostic: emitter.EmitDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


# Differential test: for generated Sources and pipelines, the emitted functions and
# the InMemoryViewEngine give the same View results.

texts = st.builds(Text, st.text(max_size=3))

wides = st.lists(
    st.builds(
        Wide,
        id=st.builds(WideId, st.integers()),
        alpha=texts,
        beta=st.none() | texts,
        gamma=texts,
        delta=texts,
    ),
    unique_by=lambda wide: wide.id,
    max_size=3,
)


@st.composite
def entities(draw: st.DrawFn) -> tuple[pydantic.BaseModel, ...]:
    """Entities of every Source, with every Reference pointing to one of them.

    IDs are distinct per Source, so no Entity replaces another in the engine.
    """
    # At least one of each, so Views that resolve References have rows.
    drawn_companies = draw(
        st.lists(
            st.builds(
                Company,
                id=st.builds(CompanyId, st.integers()),
                name=texts,
                secret=texts,
            ),
            unique_by=lambda company: company.id,
            min_size=1,
            max_size=3,
        )
    )
    departments = draw(
        st.lists(
            st.builds(
                Department,
                id=st.builds(DepartmentId, st.integers()),
                name=texts,
                company_id=st.sampled_from([c.id for c in drawn_companies]),
            ),
            unique_by=lambda department: department.id,
            min_size=1,
            max_size=3,
        )
    )
    employee_ids = draw(
        st.lists(
            st.builds(EmployeeId, st.integers()), unique=True, min_size=1, max_size=3
        )
    )
    employees = [
        Employee(
            id=employee_id,
            name=draw(texts),
            department_id=draw(st.sampled_from([d.id for d in departments])),
            mentor_id=draw(st.none() | st.sampled_from(employee_ids)),
        )
        for employee_id in employee_ids
    ]
    return (*drawn_companies, *draw(wides), *departments, *employees)


class DifferentialCase(pydantic.BaseModel, frozen=True):
    views: tuple[View, ...]
    entities: tuple[pydantic.BaseModel, ...]


@st.composite
def differential_cases(draw: st.DrawFn) -> DifferentialCase:
    """Views over a pool of Sources, each with a random pipeline."""
    views: list[View] = []
    for index in range(draw(st.integers(min_value=1, max_value=4))):
        source = draw(st.sampled_from([Company, Wide, Department, Employee]))
        views.append(
            View(
                name=ViewName(f"View{index}"),
                source=source,
                step=draw(pipelines(entity_graph(), source, Depth(2))),
            )
        )
    return DifferentialCase(views=tuple(views), entities=draw(entities()))


class GeneratedModels:
    """The module codegen's View models are executed into, for the emitted module."""

    # A top-level name, so `import <name>` binds the module itself.
    module: ClassVar[ModulePath] = ModulePath("frp_differential_view_models")

    @staticmethod
    def layout(_view: compiler.CompiledView) -> ModulePath:
        return GeneratedModels.module

    @staticmethod
    def row_of(model: pydantic.BaseModel) -> engine.ViewRow:
        """A View model as a ViewRow, nested View models included."""
        return engine.ViewRow(
            {
                FieldName(name): GeneratedModels.row_of(value)
                if isinstance(value, pydantic.BaseModel)
                and type(value).__module__ == GeneratedModels.module.root
                else value
                for name, value in vars(model).items()
            }
        )


@contextlib.contextmanager
def emitted_module(compiled: compiler.CompiledViews) -> Generator[types.ModuleType]:
    """Runs the emitted module against the View models codegen generates.

    The models forbid extra fields, because pydantic would otherwise ignore a field
    the emitted function wrongly passes.
    """
    match Codegen.generate(compiled, GeneratedModels.layout):
        case Ok(generated):
            (file,) = generated.files
        case Err(error):
            pytest.fail(f"expected View models, got {error}")
    view_models = types.ModuleType(GeneratedModels.module.root)
    exec(
        file.text.root.replace(
            "(BaseModel, frozen=True)", "(BaseModel, frozen=True, extra='forbid')"
        ),
        view_models.__dict__,
    )
    sys.modules[view_models.__name__] = view_models
    try:
        match emitter.PythonEmitter.emit(compiled, GeneratedModels.layout):
            case Ok(emitted):
                module = types.ModuleType(f"{__name__}_emitted")
                exec(emitted.root, module.__dict__)
                yield module
            case Err(error):
                pytest.fail(f"expected an emitted module, got {error}")
    finally:
        del sys.modules[view_models.__name__]


@given(differential_cases())
def test_emitted_functions_give_the_same_view_results_as_the_engine(
    case: DifferentialCase,
) -> None:
    compiled = compile_all(case.views)
    view_engine = engine.InMemoryViewEngine(entity_graph(), compiled)
    for entity in case.entities:
        view_engine.put(entity).unwrap()

    with emitted_module(compiled) as module:
        for view in compiled.views:
            function = getattr(
                module, emitter.PythonEmitter.function_name(view.name).root
            )
            lookups = {
                emitter.PythonEmitter.lookup_parameter(lookup.target).root: {
                    vars(entity)["id"]: entity
                    for entity in case.entities
                    if isinstance(entity, lookup.target)
                }
                for lookup in view.relation.lookups()
            }
            emitted_rows = [
                list(GeneratedModels.row_of(function(entity, **lookups)).root.items())
                for entity in case.entities
                if isinstance(entity, view.source)
            ]
            engine_rows = [
                list(row.root.items()) for row in view_engine.read(view.name).unwrap()
            ]
            # Catches a strategy that stops drawing resolves over stored rows.
            event(f"resolving view has rows: {bool(lookups) and bool(emitted_rows)}")
            assert emitted_rows == engine_rows
