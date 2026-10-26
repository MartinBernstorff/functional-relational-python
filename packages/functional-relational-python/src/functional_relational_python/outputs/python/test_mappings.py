import contextlib
import enum
import re
import subprocess
import sys
import types
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Literal

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
    EntityNode,
    FieldName,
    Optionality,
)
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    OneToOne,
    OneToOneMarker,
)
from functional_relational_python.outputs import engine
from functional_relational_python.outputs.python.mappings import (
    Expression,
    Indent,
    MappingsRenderer,
)
from functional_relational_python.outputs.python.models import ModelsRenderer
from functional_relational_python.outputs.python.plan import (
    ModuleLayout,
    PythonLayout,
    PythonPlan,
)
from functional_relational_python.outputs.python.render import PythonRenderer
from functional_relational_python.outputs.python.symbols import (
    ModulePath,
    ValueExpression,
)
from functional_relational_python.pipeline import conditions, ir, steps
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.support.diagnostic import DiagnosticsError
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


class ProfileId(Id[int]):
    pass


class Profile(pydantic.BaseModel, frozen=True):
    """An Employee's detail row: at most one per Employee."""

    id: ProfileId
    employee_id: Annotated[EmployeeId, OneToOne()]
    bio: Text


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
    return EntityGraph.build(
        [Company, Wide, Department, Employee, Profile],
        IdBase.fake(),
        OneToOneMarker.fake(),
    ).unwrap()


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
        compiler.CompiledCatalog(
            views=compiled.views,
            payloads=(),
            one_to_one_marker=OneToOneMarker.fake(),
        ),
        layout,
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


def reversing_views() -> list[View]:
    return [
        View(
            name=ViewName("EmployeeProfile"),
            source=Employee,
            step=steps.pipeline(
                steps.project([FieldName("id")]), steps.reverse(Profile)
            ),
        ),
        View(
            name=ViewName("EmployeeBio"),
            source=Employee,
            step=steps.pipeline(
                steps.project([FieldName("id")]),
                steps.reverse(Profile, steps.project([FieldName("bio")])),
            ),
        ),
    ]


def flattened_profile() -> View:
    """The fields of a reversed Entity, lifted to the top level."""
    return View(
        name=ViewName("EmployeeWithBio"),
        source=Employee,
        step=steps.pipeline(
            steps.project([FieldName("id")]),
            steps.reverse(Profile, steps.project([FieldName("bio")])),
            steps.flatten(FieldName("profile")),
        ),
    )


def test_golden_mapping_function_for_a_flattened_reversed_reference() -> None:
    plan = plan_of(compile_all([flattened_profile()]), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Maps each View's Source to its View model.

import collections.abc

import frp_generated_models
import functional_relational_python.outputs.python.test_mappings


def employee_with_bio(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    profile_by_employee_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Profile,
    ],
) -> frp_generated_models.EmployeeWithBio:
    return frp_generated_models.EmployeeWithBio(
        id=source.id,
        bio=(
            None
            if source.id not in profile_by_employee_id
            else profile_by_employee_id[source.id].bio
        ),
    )
"""
    )


def test_golden_mapping_functions_for_reversed_references() -> None:
    plan = plan_of(compile_all(reversing_views()), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Maps each View's Source to its View model.

import collections.abc

import frp_generated_models
import functional_relational_python.outputs.python.test_mappings


def employee_profile(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    profile_by_employee_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Profile,
    ],
) -> frp_generated_models.EmployeeProfile:
    return frp_generated_models.EmployeeProfile(
        id=source.id,
        profile=profile_by_employee_id.get(source.id),
    )


def employee_bio(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    profile_by_employee_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Profile,
    ],
) -> frp_generated_models.EmployeeBio:
    return frp_generated_models.EmployeeBio(
        id=source.id,
        profile=(
            None
            if source.id not in profile_by_employee_id
            else frp_generated_models.EmployeeBioProfile(
                bio=profile_by_employee_id[source.id].bio,
            )
        ),
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
# Regenerate with `frp generate`; `frp check` fails when this is stale.
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


def flattened_employee() -> View:
    """A View that lifts fields out of a required and an optional Reference."""
    return View(
        name=ViewName("EmployeeFlat"),
        source=Employee,
        step=steps.pipeline(
            steps.resolve(
                named(FieldName("department_id")),
                steps.omit(FieldName("id")),
                steps.rename(FieldName("name"), FieldName("department_name")),
                steps.hydrate(),
            ),
            steps.flatten(FieldName("department")),
            steps.resolve(
                named(FieldName("mentor_id")),
                steps.project([FieldName("name")]),
                steps.rename(FieldName("name"), FieldName("mentor_name")),
            ),
            steps.flatten(FieldName("mentor")),
        ),
    )


def test_a_mapping_function_reads_a_lifted_field_off_the_looked_up_entity() -> None:
    plan = plan_of(compile_all([flattened_employee()]), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Maps each View's Source to its View model.

import collections.abc

import frp_generated_models
import functional_relational_python.outputs.python.test_mappings


def employee_flat(
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
    employee_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Employee,
    ],
) -> frp_generated_models.EmployeeFlat:
    return frp_generated_models.EmployeeFlat(
        id=source.id,
        name=source.name,
        department_name=department_by_id[source.department_id].name,
        company=company_by_id[department_by_id[source.department_id].company_id],
        mentor_name=(
            None if source.mentor_id is None else employee_by_id[source.mentor_id].name
        ),
    )
"""
    )


def test_mapping_functions_with_chained_conditionals_are_ruff_formatted() -> None:
    """Fields lifted through two optional References check both for `None`."""
    renamed = steps.rename(
        steps.select(lambda _field: True), lambda field: FieldName(f"m_{field.root}")
    )
    view = View(
        name=ViewName("EmployeeGrandMentor"),
        source=Employee,
        step=steps.pipeline(
            steps.resolve(
                named(FieldName("mentor_id")),
                steps.resolve(
                    named(FieldName("mentor_id")),
                    steps.project([FieldName("name")]),
                    renamed,
                ),
                steps.flatten(FieldName("mentor")),
                renamed,
            ),
            steps.flatten(FieldName("mentor")),
        ),
    )
    text = Generated.text(
        MappingsRenderer.files(plan_of(compile_all([view]), Generated.layout()))
    )

    formatted = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--isolated", "-"],
        input=text.root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "else None\n" in text.root
    assert formatted == text.root


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
        compile_all(
            [
                *golden_views(),
                *reversing_views(),
                tagged_company(),
                *filtered_views(),
                *required_views(),
                flattened_employee(),
                flattened_profile(),
            ]
        ),
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

    IDs are distinct per Source, so no Entity replaces another in the engine. Some
    Employees have a Profile, none has several.
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
    profiled = draw(st.lists(st.sampled_from(employee_ids), unique=True))
    profiles = [
        Profile(id=ProfileId(index), employee_id=employee_id, bio=draw(texts))
        for index, employee_id in enumerate(profiled)
    ]
    return (*drawn_companies, *draw(wides), *departments, *employees, *profiles)


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
        source = draw(st.sampled_from([Company, Wide, Department, Employee, Profile]))
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
                    getattr(entity, parameter.keyed_by.root): entity
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
            reversing = any(
                parameter.keyed_by != EntityNode.primary_key
                for parameter in function.lookups
            )
            event(f"reversing view has rows: {reversing and bool(mapped_rows)}")
            # Catches a strategy whose filters keep every row, or none.
            event(f"filter keeps some rows: {0 < len(mapped_rows) < len(mapped)}")
            assert mapped_rows == engine_rows


@st.composite
def filtered_views_over_entities(draw: st.DrawFn) -> View:
    """A View of Wide or Employee that filters with a random Condition."""
    source = draw(st.sampled_from([Wide, Employee]))
    condition = draw(where_conditions(source, draw(entities()), Depth(3)))
    return View(name=ViewName.fake(), source=source, step=steps.where(condition))


def ruff_format_diff(text: FileText) -> FileText:
    """How ruff would reformat `text`: empty when it is formatted already."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--diff", "-"],
        input=text.root,
        capture_output=True,
        text=True,
        check=False,
    )
    return FileText(result.stdout + result.stderr)


def generated_mappings(views: Iterable[View]) -> FileText:
    return Generated.text(
        MappingsRenderer.files(plan_of(compile_all(views), Generated.layout()))
    )


@settings(max_examples=25)
@given(filtered_views_over_entities())
def test_a_filtered_mapping_function_is_as_ruff_formats_it(view: View) -> None:
    assert ruff_format_diff(generated_mappings([view])) == FileText("")


class Mood(enum.Enum):
    ok = "ok"
    # A member named no identifier, which only the class namespace can define.
    vars()["not ok"] = "not ok"


class Feeling(pydantic.BaseModel, frozen=True):
    """A Source with a field of an enum whose member is named no identifier."""

    name: Text
    mood: Mood


@pytest.mark.parametrize(
    ("name", "condition", "written"),
    [
        (
            ViewName("FeelingNamed"),
            conditions.eq(conditions.field(FieldName("name")), Text("it's")),
            ValueExpression('Text("it\'s")'),
        ),
        (
            ViewName("FeelingQuoted"),
            conditions.eq(conditions.field(FieldName("name")), Text('say "hi"')),
            ValueExpression("""Text('say "hi"')"""),
        ),
        (
            ViewName("FeelingBothQuotes"),
            conditions.eq(conditions.field(FieldName("name")), Text("""a'b"c""")),
            ValueExpression("""Text("a'b\\"c")"""),
        ),
        (
            ViewName("FeelingUnwell"),
            conditions.eq(conditions.field(FieldName("mood")), Mood("not ok")),
            ValueExpression('Mood["not ok"]'),
        ),
        (
            ViewName("FeelingWithViewNameLongEnoughToSplitTheReturnAnnotation"),
            conditions.eq(conditions.field(FieldName("mood")), Mood("ok")),
            ValueExpression("Mood.ok"),
        ),
    ],
    ids=["single-quote", "double-quotes", "both-quotes", "enum-name", "long-name"],
)
def test_a_filtered_mapping_function_writes_literals_as_ruff_does(
    name: ViewName, condition: ir.Condition, written: ValueExpression
) -> None:
    text = generated_mappings(
        [View(name=name, source=Feeling, step=steps.where(condition))]
    )

    assert written.root in text.root
    assert ruff_format_diff(text) == FileText("")


class Written:
    """Writes a multi-line expression, recording each indentation it was written
    at."""

    def __init__(self) -> None:
        self.at: list[Indent] = []

    def write(self, indent: Indent) -> Expression:
        self.at.append(indent)
        return Expression(
            f"call(\n{indent.deeper().pad().root}x,\n{indent.pad().root})"
        )


@pytest.mark.parametrize(
    "guards",
    [(), (Expression("source.field_id"),)],
    ids=["required", "optional"],
)
def test_a_looked_up_value_is_written_once(guards: tuple[Expression, ...]) -> None:
    written = Written()

    _ = Indent(4).looked_up(Expression("field="), guards, written.write)

    assert len(written.at) == 1


# Narrowing and `require`.


def required_views() -> list[View]:
    """Views that narrow by a Filter, or require a field, a Reference, a reversed
    Reference, or a field of a looked-up Entity."""
    mentor = FieldName("mentor_id")
    return [
        View(
            name=ViewName("WideWon"),
            source=Wide,
            step=steps.where(
                conditions.and_(
                    conditions.eq(conditions.field(FieldName("stage")), Stage.WON),
                    conditions.is_null(conditions.field(FieldName("beta"))),
                )
            ),
        ),
        View(
            name=ViewName("EmployeeMentored"),
            source=Employee,
            step=steps.require(mentor),
        ),
        View(
            name=ViewName("EmployeeRequiredMentor"),
            source=Employee,
            step=steps.pipeline(
                steps.resolve(named(mentor)),
                steps.require(FieldName("mentor")),
                steps.project([FieldName("id"), FieldName("mentor")]),
            ),
        ),
        View(
            name=ViewName("EmployeeUnreadMentor"),
            source=Employee,
            step=steps.pipeline(
                steps.resolve(named(mentor), steps.project([])),
                steps.require(FieldName("mentor")),
                steps.project([FieldName("mentor")]),
            ),
        ),
        View(
            name=ViewName("EmployeeProfiled"),
            source=Employee,
            step=steps.pipeline(
                steps.project([FieldName("id")]),
                steps.reverse(Profile),
                steps.require(FieldName("profile")),
            ),
        ),
        View(
            name=ViewName("EmployeeMentorsMentor"),
            source=Employee,
            step=steps.pipeline(
                steps.project([FieldName("id"), mentor]),
                steps.resolve(
                    named(mentor), steps.project([mentor]), steps.require(mentor)
                ),
            ),
        ),
        View(
            name=ViewName("EmployeeUnreadMentorsMentor"),
            source=Employee,
            step=steps.pipeline(
                steps.project([FieldName("id"), mentor]),
                steps.resolve(
                    named(mentor),
                    steps.resolve(named(mentor), steps.project([])),
                    steps.require(FieldName("mentor")),
                    steps.project([FieldName("mentor")]),
                ),
            ),
        ),
        View(
            name=ViewName("EmployeeRequiredBio"),
            source=Employee,
            step=steps.pipeline(
                steps.project([FieldName("id")]),
                steps.reverse(Profile, steps.project([FieldName("bio")])),
                steps.flatten(FieldName("profile")),
                steps.require(FieldName("bio")),
            ),
        ),
        View(
            name=ViewName("EmployeeLiftedMentor"),
            source=Employee,
            step=steps.pipeline(
                steps.project([FieldName("id"), mentor]),
                steps.resolve(
                    named(mentor),
                    steps.project([mentor]),
                    steps.rename(mentor, FieldName("mentors_mentor_id")),
                ),
                steps.flatten(FieldName("mentor")),
                steps.require(FieldName("mentors_mentor_id")),
            ),
        ),
    ]


def test_golden_models_of_narrowed_views() -> None:
    plan = plan_of(compile_all(required_views()[:2]), Generated.layout())

    assert Generated.text(ModelsRenderer.files(plan)) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.

from typing import Literal

from functional_relational_python.compiler.test_compiler import Score, Stage
from functional_relational_python.outputs.python.test_mappings import (
    DepartmentId,
    EmployeeId,
    Text,
    WideId,
)
from pydantic import BaseModel


class WideWon(BaseModel, frozen=True):
    id: WideId
    alpha: Text
    beta: Literal[None]
    gamma: Text
    delta: Text
    rank: Score
    stage: Literal[Stage.WON]


class EmployeeMentored(BaseModel, frozen=True):
    id: EmployeeId
    name: Text
    department_id: DepartmentId
    mentor_id: EmployeeId
"""
    )


def test_golden_mapping_functions_of_required_views() -> None:
    plan = plan_of(compile_all(required_views()), Generated.layout())

    assert Generated.text(MappingsRenderer.files(plan)) == FileText(
        '''\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Maps each View's Source to its View model.

import collections.abc

import frp_generated_models
import functional_relational_python.compiler.test_compiler
import functional_relational_python.outputs.python.test_mappings


def _required[T](value: T | None, of: object, what: str) -> T:
    """`value`, which a View requires: `what` of `of`."""
    if value is None:
        key = getattr(of, "id", of)
        msg = f"{what} is required, but there is none for {type(of).__name__} {key!r}"
        raise ValueError(msg)
    return value


def wide_won(
    source: functional_relational_python.outputs.python.test_mappings.Wide,
) -> frp_generated_models.WideWon | None:
    if not (
        source.stage == functional_relational_python.compiler.test_compiler.Stage.WON
        and source.beta is None
    ):
        return None
    return frp_generated_models.WideWon(
        id=source.id,
        alpha=source.alpha,
        beta=source.beta,
        gamma=source.gamma,
        delta=source.delta,
        rank=source.rank,
        stage=source.stage,
    )


def employee_mentored(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
) -> frp_generated_models.EmployeeMentored:
    return frp_generated_models.EmployeeMentored(
        id=source.id,
        name=source.name,
        department_id=source.department_id,
        mentor_id=_required(
            source.mentor_id,
            source,
            "mentor_id",
        ),
    )


def employee_required_mentor(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    employee_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Employee,
    ],
) -> frp_generated_models.EmployeeRequiredMentor:
    return frp_generated_models.EmployeeRequiredMentor(
        id=source.id,
        mentor=employee_by_id[_required(source.mentor_id, source, "mentor_id")],
    )


def employee_unread_mentor(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    employee_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Employee,
    ],
) -> frp_generated_models.EmployeeUnreadMentor:
    _ = employee_by_id[_required(source.mentor_id, source, "mentor_id")]
    return frp_generated_models.EmployeeUnreadMentor(
        mentor=frp_generated_models.EmployeeUnreadMentorMentor(),
    )


def employee_profiled(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    profile_by_employee_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Profile,
    ],
) -> frp_generated_models.EmployeeProfiled:
    return frp_generated_models.EmployeeProfiled(
        id=source.id,
        profile=_required(
            profile_by_employee_id.get(source.id),
            source,
            "a Profile by employee_id",
        ),
    )


def employee_mentors_mentor(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    employee_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Employee,
    ],
) -> frp_generated_models.EmployeeMentorsMentor:
    return frp_generated_models.EmployeeMentorsMentor(
        id=source.id,
        mentor=(
            None
            if source.mentor_id is None
            else frp_generated_models.EmployeeMentorsMentorMentor(
                mentor_id=_required(
                    employee_by_id[source.mentor_id].mentor_id,
                    employee_by_id[source.mentor_id],
                    "mentor_id",
                ),
            )
        ),
    )


def employee_unread_mentors_mentor(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    employee_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Employee,
    ],
) -> frp_generated_models.EmployeeUnreadMentorsMentor:
    if not (source.mentor_id is None):
        _ = employee_by_id[
            _required(
                employee_by_id[source.mentor_id].mentor_id,
                employee_by_id[source.mentor_id],
                "mentor_id",
            )
        ]
    return frp_generated_models.EmployeeUnreadMentorsMentor(
        id=source.id,
        mentor=(
            None
            if source.mentor_id is None
            else frp_generated_models.EmployeeUnreadMentorsMentorMentor(
                mentor=frp_generated_models.EmployeeUnreadMentorsMentorMentorMentor(),
            )
        ),
    )


def employee_required_bio(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    profile_by_employee_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Profile,
    ],
) -> frp_generated_models.EmployeeRequiredBio:
    return frp_generated_models.EmployeeRequiredBio(
        id=source.id,
        bio=_required(
            profile_by_employee_id.get(source.id), source, "a Profile by employee_id"
        ).bio,
    )


def employee_lifted_mentor(
    source: functional_relational_python.outputs.python.test_mappings.Employee,
    *,
    employee_by_id: collections.abc.Mapping[
        functional_relational_python.outputs.python.test_mappings.EmployeeId,
        functional_relational_python.outputs.python.test_mappings.Employee,
    ],
) -> frp_generated_models.EmployeeLiftedMentor:
    return frp_generated_models.EmployeeLiftedMentor(
        id=source.id,
        mentors_mentor_id=_required(
            employee_by_id[_required(source.mentor_id, source, "mentor_id")].mentor_id,
            employee_by_id[_required(source.mentor_id, source, "mentor_id")],
            "mentor_id",
        ),
    )
'''
    )


def test_required_mapping_functions_are_as_ruff_formats_them() -> None:
    assert ruff_format_diff(generated_mappings(required_views())) == FileText("")


@pytest.mark.parametrize(
    ("view", "message"),
    [
        (
            ViewName("EmployeeMentored"),
            "mentor_id is required, but there is none for Employee EmployeeId(root=1)",
        ),
        (
            ViewName("EmployeeRequiredMentor"),
            "mentor_id is required, but there is none for Employee EmployeeId(root=1)",
        ),
        (
            ViewName("EmployeeUnreadMentor"),
            "mentor_id is required, but there is none for Employee EmployeeId(root=1)",
        ),
        (
            ViewName("EmployeeProfiled"),
            (
                "a Profile by employee_id is required, but there is none for "
                "Employee EmployeeId(root=1)"
            ),
        ),
        (
            ViewName("EmployeeRequiredBio"),
            (
                "a Profile by employee_id is required, but there is none for "
                "Employee EmployeeId(root=1)"
            ),
        ),
        (
            ViewName("EmployeeLiftedMentor"),
            "mentor_id is required, but there is none for Employee EmployeeId(root=1)",
        ),
    ],
    ids=["field", "reference", "unread-reference", "reverse", "lifted", "chain"],
)
def test_a_mapping_function_raises_for_a_value_the_view_requires(
    view: ViewName, message: str
) -> None:
    unmentored = Employee(
        id=EmployeeId(1),
        name=Text("Ann"),
        department_id=DepartmentId(1),
        mentor_id=None,
    )
    plan = plan_of(compile_all(required_views()), Generated.layout())
    (function,) = (
        function
        for function in plan.functions[Generated.mappings]
        if function.view == view
    )

    with mapping_functions(plan) as module:
        mapping = getattr(module, function.symbol.bound_name().root)
        lookups = {
            parameter.name.root: dict[object, object]()
            for parameter in function.lookups
        }
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            mapping(unmentored, **lookups)


@pytest.mark.parametrize(
    "view", [ViewName("EmployeeMentorsMentor"), ViewName("EmployeeUnreadMentorsMentor")]
)
def test_a_requirement_of_a_looked_up_entity_names_it(view: ViewName) -> None:
    mentor = Employee(
        id=EmployeeId(1),
        name=Text("Ann"),
        department_id=DepartmentId(1),
        mentor_id=None,
    )
    mentee = mentor.model_copy(update={"id": EmployeeId(2), "mentor_id": mentor.id})
    plan = plan_of(compile_all(required_views()), Generated.layout())

    message = "mentor_id is required, but there is none for Employee EmployeeId(root=1)"

    (function,) = (
        function
        for function in plan.functions[Generated.mappings]
        if function.view == view
    )

    with (
        mapping_functions(plan) as module,
        pytest.raises(ValueError, match=f"^{re.escape(message)}$"),
    ):
        getattr(module, function.symbol.bound_name().root)(
            mentee, employee_by_id={mentor.id: mentor}
        )


@st.composite
def required_differential_cases(draw: st.DrawFn) -> DifferentialCase:
    """Views over a pool of Sources that may filter, then run a random pipeline,
    then require some of the optional fields it keeps."""
    stored = draw(entities())
    views: list[View] = []
    for index in range(draw(st.integers(min_value=1, max_value=3))):
        source = draw(st.sampled_from([Wide, Employee, Department]))
        filters = draw(st.lists(where_conditions(source, stored, Depth(1)), max_size=1))
        pool = [
            FieldName(name)
            for field in source.model_fields
            for name in (field, field.removesuffix("_id"))
        ]
        candidates = [*pool, FieldName("profile")]
        # Half the Views require every optional field they keep.
        chosen = draw(st.just(set(candidates)) | st.sets(st.sampled_from(candidates)))

        def selected(field: ir.RelationField, chosen: set[FieldName] = chosen) -> bool:
            return field.name in chosen and field.optionality() is Optionality.OPTIONAL

        views.append(
            View(
                name=ViewName(f"View{index}"),
                source=source,
                step=steps.pipeline(
                    *map(steps.where, filters),
                    draw(pipelines(entity_graph(), source, Depth(2))),
                    steps.require(steps.select(selected)),
                ),
            )
        )
    return DifferentialCase(views=tuple(views), entities=stored)


class Raised(pydantic.RootModel[str], frozen=True):
    """The message of the error a mapping function raises."""


def expected_messages(diagnostic: engine.ReadDiagnostic) -> set[Raised]:
    """What a mapping function raises for each ID an unmet requirement names."""
    match diagnostic:
        case engine.UnmetRequirement(entity=entity, field=field, ids=ids):
            what = field.root
            subjects = [entity for _ in ids]
        case engine.UnmetReverseRequirement(entity=entity, via=via, ids=ids):
            what = f"a {entity.__name__} by {via.root}"
            subjects = [
                entity_graph().entity_keyed_by(type(id_value)).unwrap()
                for id_value in ids
            ]
        case engine.DanglingReference() | engine.SharedOneToOneReference():
            pytest.fail(f"expected unmet requirements only, got {diagnostic}")
    return {
        Raised(
            f"{what} is required, but there is none for {subject.__name__} {id_value!r}"
        )
        for subject, id_value in zip(subjects, ids, strict=True)
    }


@settings(deadline=None)
@given(required_differential_cases())
def test_mapping_functions_raise_where_the_engine_reports_an_unmet_requirement(
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
                    getattr(entity, parameter.keyed_by.root): entity
                    for entity in case.entities
                    if isinstance(entity, parameter.entity)
                }
                for parameter in function.lookups
            }
            source = view_source(compiled, function.view)
            mapping = getattr(module, function.symbol.bound_name().root)
            raised: set[Raised] = set()
            mapped: list[pydantic.BaseModel | None] = []
            for entity in case.entities:
                if not isinstance(entity, source):
                    continue
                try:
                    mapped.append(mapping(entity, **lookups))
                except ValueError as error:
                    raised.add(Raised(str(error)))
            match view_engine.read(function.view):
                case Ok(rows):
                    assert raised == set()
                    assert [
                        list(Executed.row_of(model).root.items())
                        for model in mapped
                        if model is not None
                    ] == [list(row.root.items()) for row in rows]
                case Err(DiagnosticsError() as error):
                    reported = {
                        message
                        for diagnostic in error.diagnostics
                        for message in expected_messages(diagnostic)
                    }
                    # A row raises for its first unmet requirement only.
                    assert raised
                    assert raised <= reported
                case Err(other):
                    pytest.fail(f"expected rows or diagnostics, got {other}")
            event(f"requirement unmet: {bool(raised)}")


@settings(max_examples=25, deadline=None)
@given(required_differential_cases())
def test_any_required_mapping_function_is_as_ruff_formats_it(
    case: DifferentialCase,
) -> None:
    diff = ruff_format_diff(generated_mappings(case.views))
    assert diff == FileText(""), diff.root
