import contextlib
import enum
import inspect
import subprocess
import sys
import types
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Annotated, ClassVar

import django
import pydantic
import pytest
from django.apps import apps
from django.conf import settings as django_settings
from django.db import NotSupportedError, connection, transaction
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
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    OneToOne,
    OneToOneMarker,
)
from functional_relational_python.outputs import engine
from functional_relational_python.outputs.django.models import (
    AppLabel,
    DjangoField,
    DjangoModels,
    DjangoModelsRenderer,
)
from functional_relational_python.outputs.django.queries import (
    DjangoQueriesDiagnostic,
    DjangoQueriesRenderer,
    QueriesIntoDjangoModelsModule,
    ReverseInDjangoQueries,
    ReverseInQueryError,
    ViewModelIntoDjangoModelsModule,
    ViewModelIntoQueriesModule,
    ViewQueries,
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


class Role(enum.StrEnum):
    """Django stores it as text, and returns a `str`."""

    STAFF = "staff"
    LEAD = "lead"


class Employee(pydantic.BaseModel, frozen=True):
    id: EmployeeId
    name: Text
    nickname: Text | None
    role: Role
    department_id: DepartmentId
    mentor_id: EmployeeId | None


class Note(pydantic.BaseModel, frozen=True):
    """A Source that is not an Entity."""

    text: Text


def entity_graph() -> EntityGraph:
    return EntityGraph.build(
        [Company, Department, Employee], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()


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
            field_types={Role: DjangoField.builtin(Identifier("TextField"), {})},
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


def flattened_employee() -> View:
    """Lifts fields out of a chain of required References, and of a chain of
    optional ones, whose conditionals chain; one of them an Entity."""
    return View(
        name=ViewName("EmployeeFlat"),
        source=Employee,
        step=steps.pipeline(
            steps.resolve(
                named(FieldName("department_id")),
                steps.omit(FieldName("id")),
                steps.rename(FieldName("name"), FieldName("department_name")),
                steps.resolve(
                    named(FieldName("company_id")),
                    steps.project([FieldName("name")]),
                    steps.rename(FieldName("name"), FieldName("company_name")),
                ),
                steps.flatten(FieldName("company")),
            ),
            steps.flatten(FieldName("department")),
            steps.resolve(
                named(FieldName("mentor_id")),
                steps.project(
                    [
                        FieldName("name"),
                        FieldName("department_id"),
                        FieldName("mentor_id"),
                    ]
                ),
                steps.resolve(named(FieldName("department_id"))),
                steps.resolve(
                    named(FieldName("mentor_id")),
                    steps.project([FieldName("name"), FieldName("nickname")]),
                    steps.rename(
                        steps.select(lambda _field: True), prefixed(Text("grand"))
                    ),
                ),
                steps.flatten(FieldName("mentor")),
                steps.rename(
                    steps.select(lambda _field: True), prefixed(Text("mentor"))
                ),
            ),
            steps.flatten(FieldName("mentor")),
        ),
    )


def prefixed(prefix: Text) -> ir.FieldRenaming:
    return lambda field: FieldName(f"{prefix.root}_{field.root}")


def test_golden_queries() -> None:
    assert queries_of(golden_views()) == FileText(
        """\
# Generated by functional-relational-python. Do not edit.
# Regenerate with `frp generate`; `frp check` fails when this is stale.
# Fills each View over an Entity from the Entity's Django model.

import collections.abc
import typing

import django.db.models
import django.db.models.expressions
import frp_generated_models

if typing.TYPE_CHECKING:
    import frp_generated_django


def company_full_queryset(
    queryset: "django.db.models.QuerySet[frp_generated_django.Company]",
) -> "django.db.models.QuerySet[frp_generated_django.Company]":
    return queryset.defer(None)


def company_full_from_row(
    row: "frp_generated_django.Company",
) -> frp_generated_models.CompanyFull:
    return frp_generated_models.CompanyFull.model_validate(
        {
            "id": row.id,
            "name": row.name,
            "parent": row.parent_id,
        },
        strict=False,
    )


def employee_hydrated_queryset(
    queryset: "django.db.models.QuerySet[frp_generated_django.Employee]",
) -> "django.db.models.QuerySet[frp_generated_django.Employee]":
    return queryset.defer(None).select_related(
        "department",
        "mentor",
    )


def employee_hydrated_from_row(
    row: "frp_generated_django.Employee",
) -> frp_generated_models.EmployeeHydrated:
    return frp_generated_models.EmployeeHydrated.model_validate(
        {
            "id": row.id,
            "name": row.name,
            "nickname": row.nickname,
            "role": row.role,
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
                    "role": row.mentor.role,
                    "department_id": row.mentor.department_id,
                    "mentor_id": row.mentor.mentor_id,
                }
            ),
        },
        strict=False,
    )


def employee_overview_queryset(
    queryset: "django.db.models.QuerySet[frp_generated_django.Employee]",
) -> "django.db.models.QuerySet[frp_generated_django.Employee]":
    return queryset.defer(None).select_related(
        "department",
        "department__company",
        "mentor",
    )


def employee_overview_from_row(
    row: "frp_generated_django.Employee",
) -> frp_generated_models.EmployeeOverview:
    return frp_generated_models.EmployeeOverview.model_validate(
        {
            "id": row.id,
            "name": row.name,
            "nickname": row.nickname,
            "role": row.role,
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
        },
        strict=False,
    )


def employee_tagged_queryset(
    queryset: "django.db.models.QuerySet[frp_generated_django.Employee]",
) -> "django.db.models.QuerySet[frp_generated_django.Employee]":
    return queryset.defer(None)


def employee_tagged_from_row(
    row: "frp_generated_django.Employee",
) -> frp_generated_models.EmployeeTagged:
    return frp_generated_models.EmployeeTagged.model_validate(
        {
            "id": row.id,
        },
        strict=False,
    )


class CompanyQuerySet(django.db.models.QuerySet["frp_generated_django.Company"]):
    def as_company_full(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.CompanyFull]:
        return map(
            company_full_from_row,
            company_full_queryset(self),
        )


class CompanyManager(django.db.models.Manager["frp_generated_django.Company"]):
    @typing.override
    def get_queryset(
        self,
    ) -> CompanyQuerySet:
        return CompanyQuerySet(
            self.model,
            using=self._db,
        )

    @typing.override
    def all(
        self,
    ) -> CompanyQuerySet:
        return self.get_queryset()

    @typing.override
    def filter(
        self,
        *args: object,
        **kwargs: object,
    ) -> CompanyQuerySet:
        return self.get_queryset().filter(*args, **kwargs)

    @typing.override
    def exclude(
        self,
        *args: object,
        **kwargs: object,
    ) -> CompanyQuerySet:
        return self.get_queryset().exclude(*args, **kwargs)

    @typing.override
    def order_by(
        self,
        *field_names: str | django.db.models.expressions.Combinable,
    ) -> CompanyQuerySet:
        return self.get_queryset().order_by(*field_names)

    def as_company_full(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.CompanyFull]:
        return self.get_queryset().as_company_full()


class DepartmentQuerySet(django.db.models.QuerySet["frp_generated_django.Department"]):
    pass


class DepartmentManager(django.db.models.Manager["frp_generated_django.Department"]):
    @typing.override
    def get_queryset(
        self,
    ) -> DepartmentQuerySet:
        return DepartmentQuerySet(
            self.model,
            using=self._db,
        )

    @typing.override
    def all(
        self,
    ) -> DepartmentQuerySet:
        return self.get_queryset()

    @typing.override
    def filter(
        self,
        *args: object,
        **kwargs: object,
    ) -> DepartmentQuerySet:
        return self.get_queryset().filter(*args, **kwargs)

    @typing.override
    def exclude(
        self,
        *args: object,
        **kwargs: object,
    ) -> DepartmentQuerySet:
        return self.get_queryset().exclude(*args, **kwargs)

    @typing.override
    def order_by(
        self,
        *field_names: str | django.db.models.expressions.Combinable,
    ) -> DepartmentQuerySet:
        return self.get_queryset().order_by(*field_names)


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


class EmployeeManager(django.db.models.Manager["frp_generated_django.Employee"]):
    @typing.override
    def get_queryset(
        self,
    ) -> EmployeeQuerySet:
        return EmployeeQuerySet(
            self.model,
            using=self._db,
        )

    @typing.override
    def all(
        self,
    ) -> EmployeeQuerySet:
        return self.get_queryset()

    @typing.override
    def filter(
        self,
        *args: object,
        **kwargs: object,
    ) -> EmployeeQuerySet:
        return self.get_queryset().filter(*args, **kwargs)

    @typing.override
    def exclude(
        self,
        *args: object,
        **kwargs: object,
    ) -> EmployeeQuerySet:
        return self.get_queryset().exclude(*args, **kwargs)

    @typing.override
    def order_by(
        self,
        *field_names: str | django.db.models.expressions.Combinable,
    ) -> EmployeeQuerySet:
        return self.get_queryset().order_by(*field_names)

    def as_employee_hydrated(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.EmployeeHydrated]:
        return self.get_queryset().as_employee_hydrated()

    def as_employee_overview(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.EmployeeOverview]:
        return self.get_queryset().as_employee_overview()

    def as_employee_tagged(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.EmployeeTagged]:
        return self.get_queryset().as_employee_tagged()
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
            compiler.CompiledCatalog(
                views=compiled.views,
                payloads=(),
                one_to_one_marker=OneToOneMarker.fake(),
            ),
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
            role=Role.LEAD if index % 2 else Role.STAFF,
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
        flattened_employee(),
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


def hydrated() -> View:
    return View(name=ViewName("Hydrated"), source=Employee, step=steps.hydrate())


def only_the_name(queryset: QuerySet[Model]) -> QuerySet[Model]:
    """Defers the fields the View needs, including the foreign keys it follows."""
    return queryset.only("name").order_by("pk")


def deferring_the_mentor(queryset: QuerySet[Model]) -> QuerySet[Model]:
    return queryset.defer("mentor", "nickname").order_by("pk")


@pytest.mark.parametrize("query", [only_the_name, deferring_the_mentor])
def test_a_view_loads_the_fields_only_or_defer_left_out(query: Query) -> None:
    entities = organisation()
    (compiled,) = compile_all([hydrated()]).views

    with Database.of([hydrated()]) as django_models:
        Database.put(django_models, entities)
        rows = Database.read(django_models, compiled, query)

    assert rows == engine_rows([compiled], entities)[compiled.name]


def test_a_view_cannot_follow_values() -> None:
    (compiled,) = compile_all([hydrated()]).views

    with Database.of([hydrated()]) as django_models:
        queryset = django_models.Employee.objects.values("pk")
        method = getattr(queryset, f"as_{compiled.names.function.root}")

        with pytest.raises(TypeError):
            _ = list(method())


def test_a_view_cannot_follow_a_combinator() -> None:
    (compiled,) = compile_all([hydrated()]).views

    with Database.of([hydrated()]) as django_models:
        employees = django_models.Employee.objects.all()
        method = getattr(
            employees.union(employees), f"as_{compiled.names.function.root}"
        )

        with pytest.raises(NotSupportedError):
            _ = list(method())


class StrictBase(pydantic.BaseModel, strict=True, frozen=True):
    """A base like a strict project's: an enum field rejects the `str` Django
    returns, unless validated non-strictly."""


def test_a_view_with_a_strict_base_reads_an_enum_field() -> None:
    view = View(
        name=ViewName("StrictEmployee"),
        source=Employee,
        step=steps.hydrate(),
        base=StrictBase,
    )
    entities = organisation()
    (compiled,) = compile_all([view]).views

    with Database.of([view]) as django_models:
        Database.put(django_models, entities)
        rows = Database.read(django_models, compiled, every_row)

    assert rows == engine_rows([compiled], entities)[view.name]
    assert all(isinstance(row.root[FieldName("role")], Role) for row in rows)


def test_generated_signatures_read_without_the_django_models() -> None:
    """Annotations naming the Django models, which the module imports only for
    type checking, are quoted, so evaluating its annotations does not fail."""
    with Database.of(golden_views()):
        queries = sys.modules[Modules.queries.root]
        functions = [
            value
            for value in vars(queries).values()
            if inspect.isfunction(value) and value.__module__ == queries.__name__
        ]
        for function in functions:
            _ = inspect.signature(function)

    assert functions


class LongModules:
    """Module paths long enough that ruff format splits what refers to them."""

    django: ClassVar[ModulePath] = ModulePath(
        "a_project_with_a_long_name.and_a_deeply.nested_package.of_django_models"
    )
    views: ClassVar[ModulePath] = ModulePath(
        "a_project_with_a_long_name.and_a_deeply.nested_package.of_view_models"
    )


@pytest.mark.parametrize(
    ("config", "views"),
    [
        (Modules.config(), Generated.models),
        (
            Modules.config().model_copy(update={"module": LongModules.django}),
            LongModules.views,
        ),
    ],
    ids=["short", "long"],
)
def test_generated_queries_are_ruff_formatted(
    config: DjangoModels, views: ModulePath
) -> None:
    compiled = compile_all([*golden_views(), flattened_employee()]).views
    (file,) = (
        DjangoQueriesRenderer.render(entity_graph(), config, compiled, into(views))
        .unwrap()
        .files
    )

    formatted = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--isolated", "-"],
        input=file.text.root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert formatted == file.text.root


def render_diagnostics(
    config: DjangoModels, views: ModulePath
) -> tuple[DjangoQueriesDiagnostic, ...]:
    view = View(name=ViewName("CompanyFull"), source=Company, step=steps.pipeline())
    match DjangoQueriesRenderer.render(
        entity_graph(), config, compile_all([view]).views, into(views)
    ):
        case Ok(files):
            pytest.fail(f"expected diagnostics, got {files}")
        case Err(error):
            return error.diagnostics


def test_queries_into_the_django_models_module_is_a_diagnostic() -> None:
    config = Modules.config().model_copy(update={"queries": Modules.django})

    assert render_diagnostics(config, Generated.models) == (
        QueriesIntoDjangoModelsModule(module=Modules.django),
    )


def test_a_view_model_into_the_queries_module_is_a_diagnostic() -> None:
    assert render_diagnostics(Modules.config(), Modules.queries) == (
        ViewModelIntoQueriesModule(
            view=ViewName("CompanyFull"), module=Modules.queries
        ),
    )


def test_a_view_model_into_the_django_models_module_is_a_diagnostic() -> None:
    assert render_diagnostics(Modules.config(), Modules.django) == (
        ViewModelIntoDjangoModelsModule(
            view=ViewName("CompanyFull"), module=Modules.django
        ),
    )


def another_company() -> type[pydantic.BaseModel]:
    """An Entity named like `Company`, defined elsewhere."""

    class AnotherCompanyId(Id[int]):
        pass

    class Company(pydantic.BaseModel):
        id: AnotherCompanyId

    return Company


class CompanyProfileId(Id[int]):
    pass


class CompanyProfile(pydantic.BaseModel, frozen=True):
    id: CompanyProfileId
    company_id: Annotated[CompanyId, OneToOne()]
    motto: Text


@pytest.mark.parametrize(
    "then", [(), (steps.project([FieldName("motto")]),)], ids=["entity", "nested"]
)
def test_a_view_that_reverses_a_reference_is_a_diagnostic(
    then: tuple[steps.Step, ...],
) -> None:
    graph = EntityGraph.build(
        [Company, CompanyProfile], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    view = View(
        name=ViewName("CompanyWithProfile"),
        source=Company,
        step=steps.reverse(CompanyProfile, *then),
    )

    match DjangoQueriesRenderer.render(
        graph,
        Modules.config(),
        compiler.compile_views(graph, [view]).unwrap().views,
        into(Generated.models),
    ):
        case Ok(files):
            pytest.fail(f"expected diagnostics, got {files}")
        case Err(error):
            assert error.diagnostics == (
                ReverseInDjangoQueries(
                    view=view.name,
                    entity=CompanyProfile,
                    via=FieldName("company_id"),
                ),
            )


@pytest.mark.parametrize(
    "then", [(), (steps.project([FieldName("motto")]),)], ids=["entity", "nested"]
)
def test_selecting_related_for_a_reversed_field_is_an_error(
    then: tuple[steps.Step, ...],
) -> None:
    graph = EntityGraph.build(
        [Company, CompanyProfile], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    view = View(
        name=ViewName("CompanyWithProfile"),
        source=Company,
        step=steps.reverse(CompanyProfile, *then),
    )
    (compiled,) = compiler.compile_views(graph, [view]).unwrap().views
    queries = ViewQueries(
        view=compiled,
        model=Modules.config().model_of(Company),
        source=Modules.config().model_of(Company),
    )

    with pytest.raises(ReverseInQueryError, match="company_profile"):
        _ = queries.related_paths()


def test_the_reverse_in_django_queries_message_names_the_reference() -> None:
    diagnostic = ReverseInDjangoQueries(
        view=ViewName("CompanyWithProfile"),
        entity=CompanyProfile,
        via=FieldName("company_id"),
    )

    assert diagnostic.describe().root == (
        "CompanyWithProfile: cannot generate Django queries for it, because it "
        "reverses CompanyProfile.company_id, and Django queries cannot follow a "
        "Reference backwards yet."
    )


def test_entities_with_one_name_are_left_to_the_django_models_to_report() -> None:
    graph = EntityGraph.build(
        [Company, another_company()], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()

    rendered = DjangoQueriesRenderer.render(
        graph, Modules.config(), (), into(Generated.models)
    )

    assert isinstance(rendered, Ok)


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
            role=draw(st.sampled_from(Role)),
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
    """Including a project's use of them: the manager's QuerySet methods return
    the Entity's QuerySet, so `as_<view>()` type checks after them."""
    Database.files(golden_views()).write(
        LocalFileSystem(), Directory(tmp_path)
    ).unwrap()
    _ = (tmp_path / "frp_generated_use.py").write_text(
        "import frp_generated_django\n"
        "import frp_generated_models\n"
        "\n"
        "\n"
        "def overviews() -> list[frp_generated_models.EmployeeOverview]:\n"
        "    employees = frp_generated_django.Employee.objects\n"
        "    return [\n"
        "        *employees.filter(name='a').order_by('pk')[:2].as_employee_overview(),\n"
        "        *employees.exclude(name='a').as_employee_overview(),\n"
        "        *employees.order_by('-pk').as_employee_overview(),\n"
        "        *employees.all().as_employee_overview(),\n"
        "        *employees.as_employee_overview(),\n"
        "    ]\n",
        encoding="utf-8",
    )
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
