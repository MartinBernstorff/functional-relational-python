import contextlib
import subprocess
import sys
import types
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import pydantic
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler
from functional_relational_python.compiler.test_compiler import (
    Depth,
    Score,
    Stage,
    pipelines,
)
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
)
from functional_relational_python.entity_graph.id import Id
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
from functional_relational_python.pipeline import conditions, ir, steps
from functional_relational_python.pipeline.views import View, ViewName
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
    rank: Score
    stage: Stage | None


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
    match PythonPlan.of(
        compiler.CompiledCatalog(views=compiled.views, payloads=()), layout
    ):
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
# Regenerate with `frp generate`; `frp check` fails when this is stale.
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


def test_golden_mapping_function_for_steps_on_a_select() -> None:
    view = View(
        name=ViewName("CompanyPrefixed"),
        source=Company,
        step=steps.pipeline(
            steps.omit(steps.select(steps.is_primary_key)),
            steps.rename(
                named(FieldName("name"), FieldName("secret")),
                lambda field: FieldName(f"company_{field.root}"),
            ),
        ),
    )
    plan = plan_of(compile_all([view]), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Maps each View's Source to its View model.

import frp_generated_models
import functional_relational_python.outputs.python.test_mappings


def company_prefixed(
    source: functional_relational_python.outputs.python.test_mappings.Company,
) -> frp_generated_models.CompanyPrefixed:
    return frp_generated_models.CompanyPrefixed(
        company_name=source.name,
        company_secret=source.secret,
    )
"""
    )


def test_a_mapping_function_fills_a_defaulted_field_from_the_source() -> None:
    view = View(
        name=ViewName("CompanyDefaulted"),
        source=Company,
        step=steps.default(FieldName("secret"), None),
    )
    plan = plan_of(compile_all([view]), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Maps each View's Source to its View model.

import frp_generated_models
import functional_relational_python.outputs.python.test_mappings


def company_defaulted(
    source: functional_relational_python.outputs.python.test_mappings.Company,
) -> frp_generated_models.CompanyDefaulted:
    return frp_generated_models.CompanyDefaulted(
        id=source.id,
        name=source.name,
        secret=source.secret,
    )
"""
    )


def tagged_company() -> View:
    """A View with an added literal tag, as for a tagged union."""
    return View(
        name=ViewName("CompanyTagged"),
        source=Company,
        step=steps.add(FieldName("type"), Literal["company"], default="company"),
    )


def test_a_mapping_function_leaves_an_added_field_to_its_default() -> None:
    plan = plan_of(compile_all([tagged_company()]), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Maps each View's Source to its View model.

import frp_generated_models
import functional_relational_python.outputs.python.test_mappings


def company_tagged(
    source: functional_relational_python.outputs.python.test_mappings.Company,
) -> frp_generated_models.CompanyTagged:
    return frp_generated_models.CompanyTagged(
        id=source.id,
        name=source.name,
        secret=source.secret,
    )
"""
    )


def filtered_views() -> list[View]:
    """Filtered Views: one short condition, and conditions too long for a line."""
    mentor = conditions.field(FieldName("mentor_id"))
    return [
        View(
            name=ViewName("EmployeeUnmentored"),
            source=Employee,
            step=steps.where(conditions.is_null(mentor)),
        ),
        View(
            name=ViewName("EmployeeAnnMentored"),
            source=Employee,
            step=steps.pipeline(
                steps.rename(FieldName("name"), FieldName("title")),
                steps.where(
                    conditions.and_(
                        conditions.eq(
                            conditions.field(FieldName("title")), Text("Ann")
                        ),
                        conditions.or_(
                            conditions.not_(conditions.is_null(mentor)),
                            conditions.ne(
                                conditions.field(FieldName("id")), EmployeeId(1)
                            ),
                        ),
                    )
                ),
                steps.where(
                    conditions.not_(
                        conditions.eq(mentor, conditions.field(FieldName("id")))
                    )
                ),
                steps.project([FieldName("id"), FieldName("title")]),
            ),
        ),
    ]


def test_golden_mapping_functions_of_filtered_views() -> None:
    plan = plan_of(compile_all(filtered_views()), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Maps each View's Source to its View model.

import frp_generated_models
import functional_relational_python.outputs.python.test_mappings


def employee_unmentored(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
) -> frp_generated_models.EmployeeUnmentored | None:
    if not (source.mentor_id is None):
        return None
    return frp_generated_models.EmployeeUnmentored(
        id=source.id,
        name=source.name,
        department_id=source.department_id,
        mentor_id=source.mentor_id,
    )


def employee_ann_mentored(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
) -> frp_generated_models.EmployeeAnnMentored | None:
    if not (
        source.name
        == functional_relational_python.outputs.python.test_mappings.Text("Ann")
        and (
            source.mentor_id is not None
            or source.id
            != functional_relational_python.outputs.python.test_mappings.EmployeeId(1)
        )
    ):
        return None
    if not (not (source.mentor_id == source.id)):
        return None
    return frp_generated_models.EmployeeAnnMentored(
        id=source.id,
        title=source.name,
    )
"""
    )


def test_a_filtered_mapping_function_returns_nothing_for_an_excluded_row() -> None:
    (unmentored, _) = filtered_views()
    mentee = Employee(
        id=EmployeeId(2),
        name=Text("Bo"),
        department_id=DepartmentId(1),
        mentor_id=EmployeeId(1),
    )

    with mapping_functions(
        plan_of(compile_all([unmentored]), Generated.layout())
    ) as module:
        mapped = module.employee_unmentored(mentee)

    assert mapped is None


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
    plan = plan_of(
        compile_all([*golden_views(), tagged_company(), *filtered_views()]),
        Generated.layout(),
    )
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

# A small alphabet, so conditions comparing texts are true for some rows.
texts = st.builds(Text, st.text(alphabet="ab", max_size=2))

wides = st.lists(
    st.builds(
        Wide,
        id=st.builds(WideId, st.integers()),
        alpha=texts,
        beta=st.none() | texts,
        gamma=texts,
        delta=texts,
        rank=st.builds(Score, st.integers(0, 3)),
        stage=st.none() | st.sampled_from(Stage),
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
def comparisons(
    draw: st.DrawFn,
    source: type[pydantic.BaseModel],
    stored: Sequence[pydantic.BaseModel],
) -> ir.Condition:
    """A comparison of a field of `source` with a field of the same type, or with
    a literal: a value some row in `stored` has, or `None` for an optional field.

    Only fields of an ordered type are ordered.
    """
    annotations = {
        FieldName(name): Optionality.of_annotation(info.annotation)
        for name, info in source.model_fields.items()
    }
    field = draw(st.sampled_from(list(annotations)))
    split = annotations[field]
    same_type = [
        other
        for other, other_split in annotations.items()
        if other_split.required == split.required
    ]
    values = [
        value
        for row in stored
        if isinstance(row, source) and (value := vars(row)[field.root]) is not None
    ]
    operand_choices: list[ir.FieldRef | object] = [
        *map(conditions.field, same_type),
        *values,
        *([None] if split.optionality is Optionality.OPTIONAL else []),
    ]
    operand = draw(st.sampled_from(operand_choices))
    ordered = split.required is Score and all(
        annotations[candidate.name].optionality is Optionality.REQUIRED
        for candidate in (conditions.field(field), operand)
        if isinstance(candidate, ir.FieldRef)
    )
    comparisons_of = [conditions.eq, conditions.ne]
    if ordered and operand is not None:
        comparisons_of.extend(
            [conditions.lt, conditions.le, conditions.gt, conditions.ge]
        )
    compare = draw(st.sampled_from(comparisons_of))
    return compare(conditions.field(field), operand)


@st.composite
def where_conditions(
    draw: st.DrawFn,
    source: type[pydantic.BaseModel],
    stored: Sequence[pydantic.BaseModel],
    depth: Depth,
) -> ir.Condition:
    """Any Condition over the fields of `source`, nested up to `depth` deep."""
    leaves: list[st.SearchStrategy[ir.Condition]] = [
        comparisons(source, stored),
        st.sampled_from(
            [
                conditions.is_null(conditions.field(FieldName(name)))
                for name in source.model_fields
            ]
        ),
    ]
    if depth.root == 0:
        return draw(st.one_of(leaves))
    nested = where_conditions(source, stored, Depth(depth.root - 1))
    return draw(
        st.one_of(
            *leaves,
            st.lists(nested, min_size=1, max_size=3).map(
                lambda terms: conditions.and_(*terms)
            ),
            st.lists(nested, min_size=1, max_size=3).map(
                lambda terms: conditions.or_(*terms)
            ),
            nested.map(conditions.not_),
        )
    )


@st.composite
def differential_cases(draw: st.DrawFn) -> DifferentialCase:
    """Views over a pool of Sources, each with a random pipeline, which may start
    by filtering with a random Condition."""
    stored = draw(entities())
    views: list[View] = []
    for index in range(draw(st.integers(min_value=1, max_value=4))):
        source = draw(st.sampled_from([Company, Wide, Department, Employee]))
        # Half the Views filter; a list drawn freely is mostly empty.
        filters = draw(
            st.just(list[ir.Condition]())
            | st.lists(
                where_conditions(source, stored, Depth(2)), min_size=1, max_size=2
            )
        )
        views.append(
            View(
                name=ViewName(f"View{index}"),
                source=source,
                step=steps.pipeline(
                    *map(steps.where, filters),
                    draw(pipelines(entity_graph(), source, Depth(2))),
                ),
            )
        )
    return DifferentialCase(views=tuple(views), entities=stored)


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


class NamedBase(pydantic.BaseModel, frozen=True):
    name: Text


def test_a_mapping_function_fills_the_fields_its_model_inherits() -> None:
    view = View(
        name=ViewName("CompanyNamed"),
        source=Company,
        step=steps.omit(FieldName("secret")),
        base=NamedBase,
    )
    company = Company(id=CompanyId(1), name=Text("Acme"), secret=Text("hidden"))

    with mapping_functions(plan_of(compile_all([view]), Generated.layout())) as module:
        mapped = module.company_named(company)

    assert isinstance(mapped, NamedBase)
    assert Executed.row_of(mapped) == engine.ViewRow(
        {FieldName("name"): company.name, FieldName("id"): company.id}
    )


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
            source = view_source(compiled, function.view)
            mapping = getattr(module, function.symbol.bound_name().root)
            mapped = [
                mapping(entity, **lookups)
                for entity in case.entities
                if isinstance(entity, source)
            ]
            mapped_rows = [
                list(Executed.row_of(model).root.items())
                for model in mapped
                if model is not None
            ]
            engine_rows = [
                list(row.root.items())
                for row in view_engine.read(function.view).unwrap()
            ]
            # Catches a strategy that stops drawing resolves over stored rows.
            event(f"resolving view has rows: {bool(lookups) and bool(mapped_rows)}")
            # Catches a strategy whose filters keep every row, or none.
            event(f"filter keeps some rows: {0 < len(mapped_rows) < len(mapped)}")
            assert mapped_rows == engine_rows


@st.composite
def filtered_views_over_entities(draw: st.DrawFn) -> View:
    """A View of Wide or Employee that filters with a random Condition."""
    source = draw(st.sampled_from([Wide, Employee]))
    condition = draw(where_conditions(source, draw(entities()), Depth(3)))
    return View(name=ViewName.fake(), source=source, step=steps.where(condition))


@settings(max_examples=25)
@given(filtered_views_over_entities())
def test_a_filtered_mapping_function_is_as_ruff_formats_it(view: View) -> None:
    text = Generated.text(
        MappingsRenderer.files(plan_of(compile_all([view]), Generated.layout()))
    )

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--diff", "-"],
        input=text.root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
