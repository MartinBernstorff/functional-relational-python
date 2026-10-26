import contextlib
import subprocess
import sys
import types
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import ClassVar

import django
import pydantic
import pytest
from django.apps import apps
from django.conf import settings as django_settings
from django.db import connection, transaction
from django.db.models import Model, QuerySet
from django.test.utils import CaptureQueriesContext
from hypothesis import event, given, settings
from hypothesis import strategies as st
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler
from functional_relational_python.compiler.test_compiler import Depth, pipelines
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
)
from functional_relational_python.entity_graph.id import Id
from functional_relational_python.outputs import engine
from functional_relational_python.outputs.django.models import (
    AppLabel,
    DjangoModels,
    DjangoModelsRenderer,
)
from functional_relational_python.outputs.django.queries import (
    DjangoQueriesRenderer,
)
from functional_relational_python.outputs.python.models import ModelsRenderer
from functional_relational_python.outputs.python.plan import (
    NameClash,
    PythonLayout,
    PythonPlan,
    Scope,
)
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    ModulePath,
)
from functional_relational_python.outputs.python.test_mappings import (
    Executed,
    Generated,
    into,
)
from functional_relational_python.pipeline import ir, steps
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.support.filesystem import (
    Directory,
    FileText,
    LocalFileSystem,
)
from functional_relational_python.support.generated_files import GeneratedFiles
from functional_relational_python.support.identifier import Identifier


class Text(pydantic.RootModel[str], frozen=True):
    pass


class CompanyId(Id[int]):
    pass


class Company(pydantic.BaseModel, frozen=True):
    """`parent` is a Reference without `_id`, so its foreign key is `parent` and
    its ID attribute `parent_id`."""

    id: CompanyId
    name: Text
    parent: CompanyId | None


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
    nickname: Text | None
    department_id: DepartmentId
    mentor_id: EmployeeId | None


class Note(pydantic.BaseModel, frozen=True):
    """A Source that is not an Entity."""

    text: Text


def entity_graph() -> EntityGraph:
    return EntityGraph.build([Company, Department, Employee]).unwrap()


class Modules:
    """Where the tests generate: top-level modules, so `import <module>` binds the
    module itself. View models go to `Generated.models` of the mapping tests."""

    django: ClassVar[ModulePath] = ModulePath("frp_generated_django")
    queries: ClassVar[ModulePath] = ModulePath("frp_generated_queries")

    @staticmethod
    def config() -> DjangoModels:
        return DjangoModels(
            module=Modules.django,
            app_label=AppLabel("frp_generated"),
            field_types={},
            on_delete={},
            queries=Modules.queries,
        )


def compile_all(views: Iterable[View]) -> compiler.CompiledViews:
    match compiler.compile_views(entity_graph(), views):
        case Ok(compiled):
            return compiled
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def queries_of(views: Iterable[View]) -> FileText:
    compiled = compile_all(views)
    files = DjangoQueriesRenderer.render(
        entity_graph(), Modules.config(), compiled.views, into(Generated.models)
    ).unwrap()
    (file,) = files.files
    return file.text


def named(*names: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in names)


def golden_views() -> list[View]:
    return [
        View(name=ViewName("CompanyFull"), source=Company, step=steps.pipeline()),
        View(name=ViewName("EmployeeHydrated"), source=Employee, step=steps.hydrate()),
        View(
            name=ViewName("EmployeeOverview"),
            source=Employee,
            step=steps.pipeline(
                steps.resolve(
                    named(FieldName("department_id")),
                    steps.resolve(
                        named(FieldName("company_id")), steps.omit(FieldName("id"))
                    ),
                ),
                steps.resolve(
                    named(FieldName("mentor_id")),
                    steps.project([FieldName("name"), FieldName("department_id")]),
                ),
            ),
        ),
        View(
            name=ViewName("EmployeeTagged"),
            source=Employee,
            step=steps.pipeline(
                steps.project([FieldName("id")]),
                steps.add(FieldName("kind"), str, default="employee"),
            ),
        ),
        View(name=ViewName("NoteFull"), source=Note, step=steps.pipeline()),
    ]


def test_golden_queries() -> None:
    assert queries_of(golden_views()) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Fills each View over an Entity from the Entity's Django model.

import collections.abc
import typing

import django.db.models
import frp_generated_models

if typing.TYPE_CHECKING:
    import frp_generated_django


def company_full_queryset(
    queryset: django.db.models.QuerySet[frp_generated_django.Company],
) -> django.db.models.QuerySet[frp_generated_django.Company]:
    return queryset


def company_full_from_row(
    row: frp_generated_django.Company,
) -> frp_generated_models.CompanyFull:
    return frp_generated_models.CompanyFull.model_validate(
        {
            "id": row.id,
            "name": row.name,
            "parent": row.parent_id,
        }
    )


def employee_hydrated_queryset(
    queryset: django.db.models.QuerySet[frp_generated_django.Employee],
) -> django.db.models.QuerySet[frp_generated_django.Employee]:
    return queryset.select_related(
        "department",
        "mentor",
    )


def employee_hydrated_from_row(
    row: frp_generated_django.Employee,
) -> frp_generated_models.EmployeeHydrated:
    return frp_generated_models.EmployeeHydrated.model_validate(
        {
            "id": row.id,
            "name": row.name,
            "nickname": row.nickname,
            "department": {
                "id": row.department.id,
                "name": row.department.name,
                "company_id": row.department.company_id,
            },
            "mentor": (
                None
                if row.mentor is None
                else {
                    "id": row.mentor.id,
                    "name": row.mentor.name,
                    "nickname": row.mentor.nickname,
                    "department_id": row.mentor.department_id,
                    "mentor_id": row.mentor.mentor_id,
                }
            ),
        }
    )


def employee_overview_queryset(
    queryset: django.db.models.QuerySet[frp_generated_django.Employee],
) -> django.db.models.QuerySet[frp_generated_django.Employee]:
    return queryset.select_related(
        "department",
        "department__company",
        "mentor",
    )


def employee_overview_from_row(
    row: frp_generated_django.Employee,
) -> frp_generated_models.EmployeeOverview:
    return frp_generated_models.EmployeeOverview.model_validate(
        {
            "id": row.id,
            "name": row.name,
            "nickname": row.nickname,
            "department": {
                "id": row.department.id,
                "name": row.department.name,
                "company": {
                    "name": row.department.company.name,
                    "parent": row.department.company.parent_id,
                },
            },
            "mentor": (
                None
                if row.mentor is None
                else {
                    "name": row.mentor.name,
                    "department_id": row.mentor.department_id,
                }
            ),
        }
    )


def employee_tagged_queryset(
    queryset: django.db.models.QuerySet[frp_generated_django.Employee],
) -> django.db.models.QuerySet[frp_generated_django.Employee]:
    return queryset


def employee_tagged_from_row(
    row: frp_generated_django.Employee,
) -> frp_generated_models.EmployeeTagged:
    return frp_generated_models.EmployeeTagged.model_validate(
        {
            "id": row.id,
        }
    )


class CompanyQuerySet(django.db.models.QuerySet["frp_generated_django.Company"]):
    def as_company_full(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.CompanyFull]:
        return map(
            company_full_from_row,
            company_full_queryset(self),
        )


class DepartmentQuerySet(django.db.models.QuerySet["frp_generated_django.Department"]):
    pass


class EmployeeQuerySet(django.db.models.QuerySet["frp_generated_django.Employee"]):
    def as_employee_hydrated(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.EmployeeHydrated]:
        return map(
            employee_hydrated_from_row,
            employee_hydrated_queryset(self),
        )

    def as_employee_overview(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.EmployeeOverview]:
        return map(
            employee_overview_from_row,
            employee_overview_queryset(self),
        )

    def as_employee_tagged(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.EmployeeTagged]:
        return map(
            employee_tagged_from_row,
            employee_tagged_queryset(self),
        )
"""
    )


def test_a_view_named_like_a_query_set_method_is_a_diagnostic() -> None:
    view = View(name=ViewName("Manager"), source=Company, step=steps.pipeline())
    compiled = compile_all([view])

    match DjangoQueriesRenderer.render(
        entity_graph(), Modules.config(), compiled.views, into(Generated.models)
    ):
        case Ok(files):
            pytest.fail(f"expected diagnostics, got {files}")
        case Err(error):
            assert error.diagnostics == (
                NameClash(
                    scope=Scope(f"{Modules.queries.root}.CompanyQuerySet"),
                    name=Identifier("as_manager"),
                    origins=(
                        AnnotationText("django.db.models.QuerySet.as_manager"),
                        AnnotationText(
                            f"{Modules.queries.root}.CompanyQuerySet.as_manager "
                            "(the method of View Manager)"
                        ),
                    ),
                ),
            )


type Query = Callable[[QuerySet[Model]], QuerySet[Model]]
"""Filters, orders or slices a QuerySet before it is read as a View."""


def every_row(queryset: QuerySet[Model]) -> QuerySet[Model]:
    return queryset.order_by("pk")


class Database:
    """The generated View models, Django models and queries, run in this process
    against an in-memory SQLite database."""

    @staticmethod
    def configure() -> None:
        """Configures Django once per process. No other test runs Django in its
        own process."""
        if not django_settings.configured:
            django_settings.configure(
                DATABASES={
                    "default": {
                        "ENGINE": "django.db.backends.sqlite3",
                        "NAME": ":memory:",
                    }
                },
                INSTALLED_APPS=[],
                USE_TZ=True,
            )
            django.setup()

    @staticmethod
    def files(views: Iterable[View]) -> GeneratedFiles:
        """The View models, Django models and queries of `views`."""
        compiled = compile_all(views)
        config = Modules.config()
        plan = PythonPlan.of(
            compiler.CompiledCatalog(views=compiled.views, payloads=()),
            PythonLayout.models_only(into(Generated.models)),
        ).unwrap()
        return GeneratedFiles.merged(
            [
                GeneratedFiles(files=ModelsRenderer.files(plan)),
                DjangoModelsRenderer.render(entity_graph(), config).unwrap(),
                DjangoQueriesRenderer.render(
                    entity_graph(), config, compiled.views, into(Generated.models)
                ).unwrap(),
            ]
        ).unwrap()

    @staticmethod
    def text(files: GeneratedFiles, module: ModulePath) -> FileText:
        (text,) = (file.text for file in files.files if file.path == module.file_path())
        return text

    @staticmethod
    @contextlib.contextmanager
    def of(views: Iterable[View]) -> Generator[types.ModuleType]:
        """The module of Django models, with a table per model, for `views`.

        The models are unregistered afterwards, so the next call can define them
        again.
        """
        Database.configure()
        files = Database.files(views)
        label = Modules.config().app_label.root
        sys.modules[Generated.models.root] = Executed.models(
            Database.text(files, Generated.models)
        )
        try:
            for module in (Modules.queries, Modules.django):
                executed = types.ModuleType(module.root)
                sys.modules[module.root] = executed
                exec(Database.text(files, module).root, executed.__dict__)
            with connection.schema_editor() as editor:
                for model in apps.all_models[label].values():
                    editor.create_model(model)
            try:
                yield sys.modules[Modules.django.root]
            finally:
                with connection.schema_editor() as editor:
                    for model in apps.all_models[label].values():
                        editor.delete_model(model)
        finally:
            apps.all_models[label].clear()
            apps.clear_cache()
            for module in (Generated.models, Modules.queries, Modules.django):
                _ = sys.modules.pop(module.root, None)

    @staticmethod
    def put(
        django_models: types.ModuleType, entities: Iterable[pydantic.BaseModel]
    ) -> None:
        """Stores each Entity as a row of its Django model, in one transaction, so
        References are checked once every row is stored."""
        with transaction.atomic():
            for entity in entities:
                model = getattr(django_models, type(entity).__name__)
                node = entity_graph().entities[type(entity)]
                model.objects.create(
                    **{
                        DjangoModelsRenderer.attribute_of(
                            node, FieldName(name)
                        ).root: value
                        for name, value in entity.model_dump().items()
                    }
                )

    @staticmethod
    def read(
        django_models: types.ModuleType, view: compiler.CompiledView, query: Query
    ) -> tuple[engine.ViewRow, ...]:
        """The View's results, from `query` of the Source's QuerySet, checking
        they took one database query."""
        model = getattr(django_models, view.source.__name__)
        method = f"as_{view.names.function.root}"
        with CaptureQueriesContext(connection) as captured:
            rows = tuple(
                Executed.row_of(row)
                for row in getattr(query(model.objects.all()), method)()
            )
        assert len(captured.captured_queries) == 1, captured.captured_queries
        return rows


def entities_by_id(
    entities: Iterable[pydantic.BaseModel],
) -> tuple[pydantic.BaseModel, ...]:
    """`entities` in ID order, so the engine reads them in the order the database
    orders them by primary key."""
    return tuple(sorted(entities, key=lambda entity: entity.model_dump()["id"]))


def engine_rows(
    views: Iterable[compiler.CompiledView], entities: Iterable[pydantic.BaseModel]
) -> dict[ViewName, tuple[engine.ViewRow, ...]]:
    compiled = tuple(views)
    view_engine = engine.InMemoryViewEngine(
        entity_graph(), compiler.CompiledViews(views=compiled)
    )
    for entity in entities_by_id(entities):
        view_engine.put(entity).unwrap()
    return {view.name: view_engine.read(view.name).unwrap() for view in compiled}


def organisation() -> tuple[pydantic.BaseModel, ...]:
    """Several Entities of each kind, so a query per row would show. Some optional
    References are absent and some present; a mentor may be stored after the
    Employee they mentor."""
    companies = tuple(
        Company(
            id=CompanyId(index),
            name=Text(f"company {index}"),
            parent=None if index == 1 else CompanyId(1),
        )
        for index in range(1, 4)
    )
    departments = tuple(
        Department(
            id=DepartmentId(index),
            name=Text(f"department {index}"),
            company_id=companies[index % len(companies)].id,
        )
        for index in range(1, 5)
    )
    employees = tuple(
        Employee(
            id=EmployeeId(index),
            name=Text(f"employee {index}"),
            nickname=None if index % 2 else Text(f"nick {index}"),
            department_id=departments[index % len(departments)].id,
            mentor_id=None if index % 3 == 0 else EmployeeId(index % 3 + 4),
        )
        for index in range(1, 7)
    )
    return (*employees, *departments, *companies)


@pytest.mark.parametrize(
    "view",
    [
        View(name=ViewName("Plain"), source=Employee, step=steps.pipeline()),
        View(
            name=ViewName("RequiredEntity"),
            source=Employee,
            step=steps.resolve(named(FieldName("department_id"))),
        ),
        View(
            name=ViewName("OptionalEntity"),
            source=Employee,
            step=steps.resolve(named(FieldName("mentor_id"))),
        ),
        View(
            name=ViewName("ReferenceWithoutIdSuffix"),
            source=Company,
            step=steps.hydrate(),
        ),
        View(
            name=ViewName("Chain"),
            source=Employee,
            step=steps.resolve(
                named(FieldName("department_id")),
                steps.resolve(
                    named(FieldName("company_id")),
                    steps.resolve(named(FieldName("parent"))),
                ),
            ),
        ),
        View(
            name=ViewName("SelfReference"),
            source=Employee,
            step=steps.hydrate(steps.hydrate(steps.omit(FieldName("name")))),
        ),
        View(
            name=ViewName("OnlyAdded"),
            source=Employee,
            step=steps.pipeline(
                steps.project([]), steps.add(FieldName("kind"), str, default="x")
            ),
        ),
    ],
    ids=lambda view: view.name.root,
)
def test_a_view_reads_in_one_query_like_the_engine(view: View) -> None:
    entities = organisation()
    (compiled,) = compile_all([view]).views

    with Database.of([view]) as django_models:
        Database.put(django_models, entities)
        rows = Database.read(django_models, compiled, every_row)

    assert rows == engine_rows([compiled], entities)[view.name]


def with_mentor_from_last(queryset: QuerySet[Model]) -> QuerySet[Model]:
    return queryset.filter(mentor__isnull=False).order_by("-pk")[1:3]


def test_a_view_reads_the_filtered_ordered_and_sliced_query_set() -> None:
    view = View(name=ViewName("Hydrated"), source=Employee, step=steps.hydrate())
    entities = organisation()
    (compiled,) = compile_all([view]).views
    with_mentor = [
        row
        for row in engine_rows([compiled], entities)[view.name]
        if row.root[FieldName("mentor")] is not None
    ]

    with Database.of([view]) as django_models:
        Database.put(django_models, entities)
        rows = Database.read(django_models, compiled, with_mentor_from_last)

    assert rows == tuple(reversed(with_mentor))[1:3]


# Differential test: for generated Entities and pipelines, the generated queries
# and the InMemoryViewEngine give the same View results.

big_integers = st.integers(min_value=-(2**63), max_value=2**63 - 1)
texts = st.builds(Text, st.text(max_size=3))


@st.composite
def entities(draw: st.DrawFn) -> tuple[pydantic.BaseModel, ...]:
    """Entities of every kind, with every Reference pointing to one of them.

    IDs are distinct per Entity, and fit Django's 64-bit integer fields.
    """
    company_ids = draw(
        st.lists(
            st.builds(CompanyId, big_integers), unique=True, min_size=1, max_size=3
        )
    )
    companies = [
        Company(
            id=company_id,
            name=draw(texts),
            parent=draw(st.none() | st.sampled_from(company_ids)),
        )
        for company_id in company_ids
    ]
    departments = draw(
        st.lists(
            st.builds(
                Department,
                id=st.builds(DepartmentId, big_integers),
                name=texts,
                company_id=st.sampled_from(company_ids),
            ),
            unique_by=lambda department: department.id,
            min_size=1,
            max_size=3,
        )
    )
    employee_ids = draw(
        st.lists(
            st.builds(EmployeeId, big_integers), unique=True, min_size=1, max_size=3
        )
    )
    employees = [
        Employee(
            id=employee_id,
            name=draw(texts),
            nickname=draw(st.none() | texts),
            department_id=draw(st.sampled_from([d.id for d in departments])),
            mentor_id=draw(st.none() | st.sampled_from(employee_ids)),
        )
        for employee_id in employee_ids
    ]
    return (*companies, *departments, *employees)


class DifferentialCase(pydantic.BaseModel, frozen=True):
    views: tuple[View, ...]
    entities: tuple[pydantic.BaseModel, ...]


@st.composite
def differential_cases(draw: st.DrawFn) -> DifferentialCase:
    """Views over every Entity, each with a random pipeline."""
    views: list[View] = []
    for index in range(draw(st.integers(min_value=1, max_value=4))):
        source = draw(st.sampled_from([Company, Department, Employee]))
        views.append(
            View(
                name=ViewName(f"View{index}"),
                source=source,
                step=draw(pipelines(entity_graph(), source, Depth(2))),
            )
        )
    return DifferentialCase(views=tuple(views), entities=draw(entities()))


# Each example defines Django models and creates their tables, which can exceed
# the default deadline.
@settings(deadline=None)
@given(differential_cases())
def test_queries_give_the_same_view_results_as_the_engine(
    case: DifferentialCase,
) -> None:
    compiled = compile_all(case.views).views
    expected = engine_rows(compiled, case.entities)

    with Database.of(case.views) as django_models:
        Database.put(django_models, case.entities)
        read = {
            view.name: Database.read(django_models, view, every_row)
            for view in compiled
        }

    # Catches a strategy that stops drawing resolves.
    event(f"looks up an Entity: {any(view.relation.lookups() for view in compiled)}")
    assert read == expected


def test_generated_queries_pass_the_type_checker(tmp_path: Path) -> None:
    Database.files(golden_views()).write(
        LocalFileSystem(), Directory(tmp_path)
    ).unwrap()
    source_root = Path(__file__).parents[3]
    _ = (tmp_path / "pyrefly.toml").write_text(
        'project-includes = ["frp_generated_*.py"]\n'
        f'search-path = ["{tmp_path}", "{source_root}"]\n'
        f'python-interpreter-path = "{sys.executable}"\n'
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
