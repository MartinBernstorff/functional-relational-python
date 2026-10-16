import contextlib
import subprocess
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
from functional_relational_python.outputs import engine
from functional_relational_python.outputs.python.mappings import MappingsRenderer
from functional_relational_python.outputs.python.models import ModelsRenderer
from functional_relational_python.outputs.python.plan import (
    ModuleLayout,
    PythonLayout,
    PythonPlan,
)
from functional_relational_python.outputs.python.render import PythonRenderer
from functional_relational_python.outputs.python.symbols import ModulePath
from functional_relational_python.support.filesystem import (
    Directory,
    FileText,
    InMemoryFileSystem,
    LocalFileSystem,
)
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)


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


def into(module: ModulePath) -> ModuleLayout:
    """Every View into `module`."""
    return lambda _view: module


def plan_of(compiled: compiler.CompiledViews, layout: PythonLayout) -> PythonPlan:
    match PythonPlan.of(compiled, layout):
        case Ok(plan):
            return plan
        case Err(error):
            pytest.fail(f"expected a plan, got {error}")


class Generated:
    """Where the tests generate models and mapping functions: two top-level
    modules, so `import <module>` binds the module itself."""

    models: ClassVar[ModulePath] = ModulePath("frp_generated_models")
    mappings: ClassVar[ModulePath] = ModulePath("frp_generated_mappings")

    @staticmethod
    def layout() -> PythonLayout:
        return PythonLayout(
            models=into(Generated.models), mappings=into(Generated.mappings)
        )

    @staticmethod
    def text(files: tuple[GeneratedFile, ...]) -> FileText:
        (file,) = files
        return file.text


def test_golden_mapping_functions() -> None:
    plan = plan_of(compile_all(golden_views()), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Maps each View's Source to its View model.

import collections.abc

import frp_generated_models
import functional_relational_python.outputs.python.test_mappings


def company_full(
    source: functional_relational_python.outputs.python.test_mappings.Company,
) -> frp_generated_models.CompanyFull:
    return frp_generated_models.CompanyFull(
        id=source.id,
        name=source.name,
        secret=source.secret,
    )


def company_public(
    source: functional_relational_python.outputs.python.test_mappings.Company,
) -> frp_generated_models.CompanyPublic:
    return frp_generated_models.CompanyPublic(
        id=source.id,
        name=source.name,
    )


def company_name(
    source: functional_relational_python.outputs.python.test_mappings.Company,
) -> frp_generated_models.CompanyName:
    return frp_generated_models.CompanyName(
        name=source.name,
    )


def company_title(
    source: functional_relational_python.outputs.python.test_mappings.Company,
) -> frp_generated_models.CompanyTitle:
    return frp_generated_models.CompanyTitle(
        id=source.id,
        title=source.name,
    )


def company_secret(
    source: functional_relational_python.outputs.python.test_mappings.Company,
) -> frp_generated_models.CompanySecret:
    return frp_generated_models.CompanySecret(
        secret=source.secret,
    )


def employee_hydrated(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    department_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.DepartmentId,
        functional_relational_python.outputs.python.test_mappings.Department,
    ],
    employee_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Employee,
    ],
) -> frp_generated_models.EmployeeHydrated:
    return frp_generated_models.EmployeeHydrated(
        id=source.id,
        name=source.name,
        department=department_by_id[source.department_id],
        mentor=None if source.mentor_id is None else employee_by_id[source.mentor_id],
    )


def employee_overview(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    department_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.DepartmentId,
        functional_relational_python.outputs.python.test_mappings.Department,
    ],
    company_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.CompanyId,
        functional_relational_python.outputs.python.test_mappings.Company,
    ],
) -> frp_generated_models.EmployeeOverview:
    return frp_generated_models.EmployeeOverview(
        id=source.id,
        name=source.name,
        department=frp_generated_models.EmployeeOverviewDepartment(
            id=department_by_id[source.department_id].id,
            name=department_by_id[source.department_id].name,
            company=company_by_id[department_by_id[source.department_id].company_id],
        ),
        mentor_id=source.mentor_id,
    )


def employee_mentor(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    employee_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Employee,
    ],
) -> frp_generated_models.EmployeeMentor:
    return frp_generated_models.EmployeeMentor(
        mentor=(
            None
            if source.mentor_id is None
            else frp_generated_models.EmployeeMentorMentor(
                name=employee_by_id[source.mentor_id].name,
            )
        ),
    )
"""
    )


def test_writing_the_mapping_functions_twice_succeeds() -> None:
    files = PythonRenderer.render(
        plan_of(compile_all(golden_views()), Generated.layout())
    )
    filesystem = InMemoryFileSystem()
    output = Directory.fake()
    files.write(filesystem, output).unwrap()

    assert isinstance(files.write(filesystem, output), Ok)


def test_written_models_and_mappings_are_both_kept_and_up_to_date() -> None:
    files = PythonRenderer.render(
        plan_of(compile_all(golden_views()), Generated.layout())
    )
    filesystem = InMemoryFileSystem()
    output = Directory.fake()

    files.write(filesystem, output).unwrap()

    assert isinstance(files.check(filesystem, output), Ok)
    assert set(filesystem.files) == {
        output.file(module.file_path())
        for module in (Generated.models, Generated.mappings)
    }


def test_models_and_mappings_both_count_as_generated() -> None:
    files = PythonRenderer.render(
        plan_of(compile_all(golden_views()), Generated.layout())
    )

    assert [GeneratedFiles.is_generated(file.text) for file in files.files] == [
        True,
        True,
    ]


def test_generated_models_and_mappings_pass_the_type_checker(tmp_path: Path) -> None:
    plan = plan_of(compile_all(golden_views()), Generated.layout())
    generated = tmp_path / "generated"
    PythonRenderer.render(plan).write(LocalFileSystem(), Directory(generated)).unwrap()
    source_root = Path(__file__).parents[3]
    _ = (tmp_path / "pyrefly.toml").write_text(
        f'search-path = ["{generated}", "{source_root}"]\n'
        f'python-interpreter-path = "{sys.executable}"\n'
        'project-includes = ["generated"]\n'
        'preset = "all"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "pyrefly", "check", "--config", "pyrefly.toml"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


# Differential test: for generated Sources and pipelines, the mapping functions and
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


class Executed:
    """Generated models and mapping functions, run as modules."""

    @staticmethod
    def row_of(model: pydantic.BaseModel) -> engine.ViewRow:
        """A View model as a ViewRow, nested View models included."""
        return engine.ViewRow(
            {
                FieldName(name): Executed.row_of(value)
                if isinstance(value, pydantic.BaseModel)
                and type(value).__module__ == Generated.models.root
                else value
                for name, value in vars(model).items()
            }
        )

    @staticmethod
    def models(text: FileText) -> types.ModuleType:
        """The models in `text`, made to forbid extra fields, because pydantic
        would otherwise ignore a field a mapping function wrongly passes."""
        module = types.ModuleType(Generated.models.root)
        exec(text.root, module.__dict__)
        # In definition order, so a model is rebuilt after the models it nests.
        for model in vars(module).values():
            if (
                isinstance(model, type)
                and issubclass(model, pydantic.BaseModel)
                and model.__module__ == module.__name__
            ):
                model.model_config["extra"] = "forbid"
                _ = model.model_rebuild(force=True)
        return module


@contextlib.contextmanager
def mapping_functions(plan: PythonPlan) -> Generator[types.ModuleType]:
    """The mapping functions, run against the View models of the same plan."""
    models = Executed.models(Generated.text(ModelsRenderer.files(plan)))
    sys.modules[models.__name__] = models
    try:
        module = types.ModuleType(Generated.mappings.root)
        exec(Generated.text(MappingsRenderer.files(plan)).root, module.__dict__)
        yield module
    finally:
        del sys.modules[models.__name__]


def view_source(
    compiled: compiler.CompiledViews, name: ViewName
) -> type[pydantic.BaseModel]:
    (source,) = (view.source for view in compiled.views if view.name == name)
    return source


@given(differential_cases())
def test_mapping_functions_give_the_same_view_results_as_the_engine(
    case: DifferentialCase,
) -> None:
    compiled = compile_all(case.views)
    view_engine = engine.InMemoryViewEngine(entity_graph(), compiled)
    for entity in case.entities:
        view_engine.put(entity).unwrap()
    plan = plan_of(compiled, Generated.layout())

    with mapping_functions(plan) as module:
        for function in plan.functions[Generated.mappings]:
            lookups = {
                parameter.name.root: {
                    vars(entity)["id"]: entity
                    for entity in case.entities
                    if isinstance(entity, parameter.entity)
                }
                for parameter in function.lookups
            }
            mapped_rows = [
                list(
                    Executed.row_of(
                        getattr(module, function.symbol.bound_name().root)(
                            entity, **lookups
                        )
                    ).root.items()
                )
                for entity in case.entities
                if isinstance(entity, view_source(compiled, function.view))
            ]
            engine_rows = [
                list(row.root.items())
                for row in view_engine.read(function.view).unwrap()
            ]
            # Catches a strategy that stops drawing resolves over stored rows.
            event(f"resolving view has rows: {bool(lookups) and bool(mapped_rows)}")
            assert mapped_rows == engine_rows
