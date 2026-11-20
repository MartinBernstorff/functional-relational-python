import contextlib
import enum
import inspect
import os
import subprocess
import sys
import types
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import Annotated, ClassVar, Self, override

import django
import pydantic
import pytest
from django.apps import AppConfig, apps
from django.conf import settings as django_settings
from django.db import IntegrityError, NotSupportedError, connection, transaction
from django.db.models import Manager, Model, Q, QuerySet
from django.test.utils import CaptureQueriesContext
from hypothesis import event, given, settings
from hypothesis import strategies as st
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler, narrowing, test_compiler
from functional_relational_python.compiler.test_compiler import (
    Collections,
    Depth,
    Start,
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
    IdValue,
    OneToOne,
    OneToOneMarker,
)
from functional_relational_python.outputs import engine
from functional_relational_python.outputs.django.models import (
    AppLabel,
    DjangoField,
    DjangoModelName,
    DjangoModels,
    DjangoModelsRenderer,
    EntityConfig,
    ExistingDjangoModel,
    FieldConfig,
    GeneratedModel,
    ModelGenericBase,
    ModelOptions,
    PlainBase,
    ReferencePrimaryKey,
    RelatedName,
    Unset,
)
from functional_relational_python.outputs.django.queries import (
    DjangoQueriesDiagnostic,
    DjangoQueriesRenderer,
    Prefetched,
    QueriesIntoDjangoModelsModule,
    RelatedPath,
    ReversedUndeclaredAccessor,
    ViewModelIntoDjangoModelsModule,
    ViewModelIntoQueriesModule,
    ViewQueries,
)
from functional_relational_python.outputs.python import test_symbols
from functional_relational_python.outputs.python.models import ModelsRenderer
from functional_relational_python.outputs.python.plan import (
    NameClash,
    PlacedModels,
    PythonLayout,
    PythonPlan,
    Scope,
    Unimportable,
)
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    FirstParty,
    ModulePath,
    Symbol,
    ValueExpression,
)
from functional_relational_python.outputs.python.test_mappings import (
    Executed,
    Generated,
    into,
)
from functional_relational_python.pipeline import conditions, ir, steps
from functional_relational_python.pipeline.constraints import (
    Check,
    Constraint,
    ConstraintName,
    Unique,
)
from functional_relational_python.pipeline.unions import Union, UnionName
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.support.diagnostic import DiagnosticsError
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


class BadgeId(Id[int]):
    pass


class Level(enum.Enum):
    """A plain enum, not a `StrEnum`: pydantic does not take its value for a
    `Literal` of its members. Django stores it as its value."""

    BRONZE = "bronze"
    GOLD = "gold"


class Rank(pydantic.RootModel[int | None], frozen=True):
    """Never `None` itself, so it may be ordered, but its column may be NULL.
    Like SQL, it orders no `None`: every ordering with one is false."""

    def __lt__(self, other: Rank) -> bool:
        return (
            self.root is not None and other.root is not None and self.root < other.root
        )

    def __le__(self, other: Rank) -> bool:
        return (
            self.root is not None and other.root is not None and self.root <= other.root
        )

    def __gt__(self, other: Rank) -> bool:
        return (
            self.root is not None and other.root is not None and self.root > other.root
        )

    def __ge__(self, other: Rank) -> bool:
        return (
            self.root is not None and other.root is not None and self.root >= other.root
        )


class Stars(pydantic.RootModel[Rank], frozen=True, strict=True):
    """Never `None` itself, but its column may be NULL through the Rank it wraps.
    Strict, so only a non-strict validation takes a NULL for it."""


class Badge(pydantic.BaseModel, frozen=True):
    """Followed backwards from the Employee wearing it. `label` and `caption` are
    both optional and of one type, so they can be compared with each other.
    `rank` may be ordered, but its column may be NULL, as may that of `stars`."""

    id: BadgeId
    employee_id: Annotated[EmployeeId, OneToOne()]
    label: Text | None
    caption: Text | None
    level: Level
    rank: Rank
    stars: Stars


class Note(pydantic.BaseModel, frozen=True):
    """A from_model that is not an Entity."""

    text: Text


class NewestFirstQuerySet[T: Model](QuerySet[T]):
    """A QuerySet base with a method of its own."""

    def newest_first(self) -> Self:
        return self.order_by("-pk")


class WithoutFirstManager[T: Model](Manager[T]):
    """Leaves out the row with ID 1, as a soft-delete Manager leaves out deleted
    rows."""

    @override
    def get_queryset(self) -> QuerySet[T]:
        return super().get_queryset().exclude(pk=1)


def entity_graph() -> EntityGraph:
    return EntityGraph.build(
        [Company, Department, Employee, Badge], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()


class Modules:
    """Where the tests generate: top-level modules, so `import <module>` binds the
    module itself. View models go to `Generated.models` of the mapping tests."""

    django: ClassVar[ModulePath] = ModulePath("frp_generated_django")
    queries: ClassVar[ModulePath] = ModulePath("frp_generated_queries")

    @staticmethod
    def with_options(every_model: ModelOptions) -> DjangoModels:
        """Every Entity's model declared by `every_model`, and a Badge the
        `badge` of the Employee wearing it."""
        return Modules.configured(
            {
                Badge: GeneratedModel(
                    fields={
                        FieldName("employee_id"): FieldConfig(
                            related_name=RelatedName("badge")
                        )
                    }
                )
            },
            every_model,
        )

    @staticmethod
    def with_bases() -> DjangoModels:
        """Every Entity's QuerySet and Manager subclass this module's bases."""
        return Modules.with_options(
            ModelOptions(
                query_set_base=ModelGenericBase(origin=NewestFirstQuerySet),
                manager_base=ModelGenericBase(origin=WithoutFirstManager),
            )
        )

    @staticmethod
    def config(
        entity_configs: Mapping[type[pydantic.BaseModel], EntityConfig] = (
            types.MappingProxyType({})
        ),
    ) -> DjangoModels:
        """Every model as `ModelOptions` declare it by default, but
        `entity_configs`."""
        return Modules.configured(entity_configs, ModelOptions())

    @staticmethod
    def configured(
        entity_configs: Mapping[type[pydantic.BaseModel], EntityConfig],
        every_model: ModelOptions,
    ) -> DjangoModels:
        return DjangoModels(
            module=Modules.django,
            queries=Modules.queries,
            app_label=AppLabel("frp_generated"),
            field_types={
                enum_class: DjangoField.builtin(Identifier("TextField"), {})
                for enum_class in (Role, Level)
            },
            every_model=every_model,
            entity_configs=entity_configs,
        )


class ExistingCompany:
    """`Company`'s Django model as the application wrote it, `Firm`, named unlike
    the Entity. It is in the app of the generated models, so the tests register
    and drop it with them."""

    module: ClassVar[ModulePath] = ModulePath("frp_existing_django")
    source: ClassVar[FileText] = FileText(
        "from django.db import models\n"
        "\n"
        "\n"
        "class Firm(models.Model):\n"
        "    id = models.BigIntegerField(primary_key=True)\n"
        "    name = models.TextField()\n"
        "    parent = models.ForeignKey(\n"
        '        "self", on_delete=models.PROTECT, related_name="subsidiaries", null=True\n'
        "    )\n"
        "\n"
        "    class Meta:\n"
        '        app_label = "frp_generated"\n'
    )

    @staticmethod
    def config() -> DjangoModels:
        return Modules.config(
            {
                Company: ExistingDjangoModel(
                    module=ExistingCompany.module,
                    name=Identifier("Firm"),
                    reverse_accessors={FieldName("parent"): FieldName("subsidiaries")},
                )
            }
        )

    @staticmethod
    def sources() -> tuple[tuple[ModulePath, FileText], ...]:
        return ((ExistingCompany.module, ExistingCompany.source),)


def compile_all(
    views: Iterable[View], unions: Iterable[Union] = ()
) -> compiler.CompiledViews:
    """`views` and `unions`; the compiled Views include each Union's Variants."""
    match compiler.compile_views(entity_graph(), views, unions):
        case Ok(compiled):
            return compiled
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def placed_into(module: ModulePath, compiled: compiler.CompiledViews) -> PlacedModels:
    """Every model of `compiled` placed into `module`."""
    return PlacedModels.of(into(module), (*compiled.views, *compiled.unions)).unwrap()


def generated_by(config: DjangoModels, placed: PlacedModels) -> FirstParty:
    """What `Catalog.generate` has first party: the packages of the Django
    modules and of the placed models."""
    return FirstParty.fake().with_generated(
        [config.module, config.queries, *placed.modules.values()]
    )


def queries_of(views: Iterable[View]) -> FileText:
    compiled = compile_all(views)
    files = DjangoQueriesRenderer.render(
        entity_graph(),
        Modules.config(),
        compiled.views,
        placed_into(Generated.models, compiled),
        first_party=generated_by(
            Modules.config(), placed_into(Generated.models, compiled)
        ),
    ).unwrap()
    (file,) = files.files
    return file.text


def named(*names: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in names)


def golden_views() -> list[View]:
    return [
        View(
            name=ViewName("CompanyFull"),
            query=steps.from_(Company),
        ),
        View(
            name=ViewName("EmployeeHydrated"),
            query=steps.from_(Employee, steps.resolve(steps.references())),
        ),
        View(
            name=ViewName("EmployeeOverview"),
            query=steps.from_(
                Employee,
                steps.resolve(
                    named(FieldName("department_id")),
                    steps.resolve(
                        named(FieldName("company_id")), steps.omit(FieldName("id"))
                    ),
                ),
                steps.resolve(
                    named(FieldName("mentor_id")),
                    steps.pick([FieldName("name"), FieldName("department_id")]),
                ),
            ),
        ),
        View(
            name=ViewName("EmployeeTagged"),
            query=steps.from_(
                Employee,
                steps.pick([FieldName("id")]),
                steps.add(FieldName("kind"), str, default="employee"),
            ),
        ),
        View(
            name=ViewName("NoteFull"),
            query=steps.from_(Note),
        ),
    ]


def flattened_employee() -> View:
    """Lifts fields out of a chain of required References, and of a chain of
    optional ones, whose conditionals chain; one of them an Entity."""
    return View(
        name=ViewName("EmployeeFlat"),
        query=steps.from_(
            Employee,
            steps.resolve(
                named(FieldName("department_id")),
                steps.omit(FieldName("id")),
                steps.rename(FieldName("name"), FieldName("department_name")),
                steps.resolve(
                    named(FieldName("company_id")),
                    steps.pick([FieldName("name")]),
                    steps.rename(FieldName("name"), FieldName("company_name")),
                ),
                steps.flatten(FieldName("company")),
            ),
            steps.flatten(FieldName("department")),
            steps.resolve(
                named(FieldName("mentor_id")),
                steps.pick(
                    [
                        FieldName("name"),
                        FieldName("department_id"),
                        FieldName("mentor_id"),
                    ]
                ),
                steps.resolve(named(FieldName("department_id"))),
                steps.resolve(
                    named(FieldName("mentor_id")),
                    steps.pick([FieldName("name"), FieldName("nickname")]),
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


def field_is(field: FieldName, member: enum.Enum) -> ir.Comparison:
    """`field == member`, which narrows `field` to `Literal[member]`."""
    return conditions.eq(conditions.field(field), member)


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


class BadgeQuerySet(django.db.models.QuerySet["frp_generated_django.Badge"]):
    pass


class BadgeManager(django.db.models.Manager["frp_generated_django.Badge"]):
    _queryset_class = BadgeQuerySet

    @typing.override
    def get_queryset(
        self,
    ) -> BadgeQuerySet:
        queryset = super().get_queryset()
        if not isinstance(queryset, BadgeQuerySet):
            message = f"{type(queryset).__name__} is not BadgeQuerySet"
            raise TypeError(message)
        return queryset

    @typing.override
    def all(
        self,
    ) -> BadgeQuerySet:
        return self.get_queryset()

    @typing.override
    def filter(
        self,
        *args: object,
        **kwargs: object,
    ) -> BadgeQuerySet:
        return self.get_queryset().filter(*args, **kwargs)

    @typing.override
    def exclude(
        self,
        *args: object,
        **kwargs: object,
    ) -> BadgeQuerySet:
        return self.get_queryset().exclude(*args, **kwargs)

    @typing.override
    def order_by(
        self,
        *field_names: str | django.db.models.expressions.Combinable,
    ) -> BadgeQuerySet:
        return self.get_queryset().order_by(*field_names)


class CompanyQuerySet(django.db.models.QuerySet["frp_generated_django.Company"]):
    def as_company_full(
        self,
    ) -> collections.abc.Iterator[frp_generated_models.CompanyFull]:
        return map(
            company_full_from_row,
            company_full_queryset(self),
        )


class CompanyManager(django.db.models.Manager["frp_generated_django.Company"]):
    _queryset_class = CompanyQuerySet

    @typing.override
    def get_queryset(
        self,
    ) -> CompanyQuerySet:
        queryset = super().get_queryset()
        if not isinstance(queryset, CompanyQuerySet):
            message = f"{type(queryset).__name__} is not CompanyQuerySet"
            raise TypeError(message)
        return queryset

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
    _queryset_class = DepartmentQuerySet

    @typing.override
    def get_queryset(
        self,
    ) -> DepartmentQuerySet:
        queryset = super().get_queryset()
        if not isinstance(queryset, DepartmentQuerySet):
            message = f"{type(queryset).__name__} is not DepartmentQuerySet"
            raise TypeError(message)
        return queryset

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
    _queryset_class = EmployeeQuerySet

    @typing.override
    def get_queryset(
        self,
    ) -> EmployeeQuerySet:
        queryset = super().get_queryset()
        if not isinstance(queryset, EmployeeQuerySet):
            message = f"{type(queryset).__name__} is not EmployeeQuerySet"
            raise TypeError(message)
        return queryset

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


def queries_text(views: Iterable[View], unions: Iterable[Union]) -> FileText:
    return Database.text(Database.files(views, unions), Modules.queries)


def test_golden_union_queryset() -> None:
    """Annotated with the first Variant keeping each row, and keeping only rows
    annotated with one."""
    text = queries_text([], [any_employee()])

    assert (
        """\
def any_employee_queryset(
    queryset: "django.db.models.QuerySet[frp_generated_django.Employee]",
) -> "django.db.models.QuerySet[frp_generated_django.Employee]":
    kept = queryset.defer(None).select_related(
        "mentor",
        "badge_by_employee",
    )
    kept = kept.annotate(
        _variant=django.db.models.Case(
            django.db.models.When(
                django.db.models.Q(
                    role=functional_relational_python.outputs.django.test_queries.Role.LEAD.value,
                ),
                then=django.db.models.Value("LeadEmployee"),
            ),
            django.db.models.When(
                (
                    django.db.models.Q(
                        nickname__isnull=False,
                    )
                    | django.db.models.Q(
                        id=1,
                    )
                ),
                then=django.db.models.Value("NicknamedEmployee"),
            ),
        ),
    )
    kept.query.add_q(
        django.db.models.Q(
            _variant__isnull=False,
        ),
    )
    return kept
"""
        in text.root
    )


def colleagues() -> View:
    """Reverses a many-to-one Reference inside a resolve's nested pipeline, and
    at the top level."""
    return View(
        name=ViewName("Colleagues"),
        query=steps.from_(
            Employee,
            steps.resolve(
                named(FieldName("department_id")),
                steps.reverse(Employee, name=FieldName("colleagues")),
            ),
            steps.reverse(Employee, name=FieldName("mentees")),
        ),
    )


def organisation_chart() -> View:
    """A collection in a collection, whose elements resolve and reverse one-to-one
    References, beside the same collection with no steps, which the
    one prefetch of its path loads too."""
    return View(
        name=ViewName("OrganisationChart"),
        query=steps.from_(
            Company,
            steps.pick([FieldName("id")]),
            steps.reverse(
                Department,
                steps.pick([FieldName("id"), FieldName("company_id")]),
                steps.resolve(
                    named(FieldName("company_id")), steps.pick([FieldName("name")])
                ),
                steps.reverse(
                    Employee,
                    steps.pick([FieldName("name"), FieldName("mentor_id")]),
                    steps.resolve(
                        named(FieldName("mentor_id")),
                        steps.pick([FieldName("name")]),
                    ),
                    steps.reverse(Badge, steps.pick([FieldName("level")])),
                ),
                via_field=ir.ViaField.KEEP,
            ),
            steps.reverse(Department, name=FieldName("all_departments")),
        ),
    )


def test_golden_queries_of_nested_pipelines_on_sorted_sequences() -> None:
    """A nested pipeline's Entities are loaded by its prefetch's own queryset,
    and a collection in it by a Prefetch nested in that queryset."""
    assert (
        """\
def organisation_chart_queryset(
    queryset: "django.db.models.QuerySet[frp_generated_django.Company]",
) -> "django.db.models.QuerySet[frp_generated_django.Company]":
    import frp_generated_django

    return queryset.defer(None).prefetch_related(
        django.db.models.Prefetch(
            "department_by_company",
            queryset=(
                frp_generated_django.Department._default_manager.order_by("id")
                .select_related(
                    "company",
                )
                .prefetch_related(
                    django.db.models.Prefetch(
                        "employee_by_department",
                        queryset=(
                            frp_generated_django.Employee._default_manager.order_by(
                                "id"
                            ).select_related(
                                "mentor",
                                "badge_by_employee",
                            )
                        ),
                    ),
                )
            ),
        ),
    )


def organisation_chart_from_row(
    row: "frp_generated_django.Company",
) -> frp_generated_models.OrganisationChart:
    return frp_generated_models.OrganisationChart.model_validate(
        {
            "id": row.id,
            "departments": [
                {
                    "id": _department.id,
                    "company": {
                        "name": _department.company.name,
                    },
                    "employees": [
                        {
                            "name": _employee.name,
                            "mentor": (
                                None
                                if _employee.mentor is None
                                else {
                                    "name": _employee.mentor.name,
                                }
                            ),
                            "badge": (
                                None
                                if not hasattr(_employee, "badge_by_employee")
                                else {
                                    "level": typing.cast(
                                        "frp_generated_django.Badge",
                                        getattr(_employee, "badge_by_employee"),
                                    ).level,
                                }
                            ),
                        }
                        for _employee in typing.cast(
                            "django.db.models.Manager[frp_generated_django.Employee]",
                            getattr(_department, "employee_by_department"),
                        ).all()
                    ],
                }
                for _department in typing.cast(
                    "django.db.models.Manager[frp_generated_django.Department]",
                    getattr(row, "department_by_company"),
                ).all()
            ],
            "all_departments": [
                {
                    "id": _department.id,
                    "name": _department.name,
                }
                for _department in typing.cast(
                    "django.db.models.Manager[frp_generated_django.Department]",
                    getattr(row, "department_by_company"),
                ).all()
            ],
        },
        strict=False,
    )
"""
        in queries_of([organisation_chart()]).root
    )


def test_a_collection_in_a_collection_of_one_entity_binds_another_variable() -> None:
    view = View(
        name=ViewName("Mentorships"),
        query=steps.from_(
            Department,
            steps.reverse(
                Employee,
                steps.reverse_via(
                    Employee, FieldName("mentor_id"), name=FieldName("mentees")
                ),
            ),
        ),
    )

    assert (
        """\
                        for _employee_2 in typing.cast(
                            "django.db.models.Manager[frp_generated_django.Employee]",
                            getattr(_employee, "employee_by_mentor"),
                        ).all()
"""
        in queries_of([view]).root
    )


def test_golden_queries_of_sorted_sequences() -> None:
    """Prefetched through the reverse accessor, ordered by primary key, by a
    queryset of the Django model imported when the function runs."""
    assert (
        """\
def colleagues_queryset(
    queryset: "django.db.models.QuerySet[frp_generated_django.Employee]",
) -> "django.db.models.QuerySet[frp_generated_django.Employee]":
    import frp_generated_django

    return (
        queryset.defer(None)
        .select_related(
            "department",
        )
        .prefetch_related(
            django.db.models.Prefetch(
                "department__employee_by_department",
                queryset=frp_generated_django.Employee._default_manager.order_by("id"),
            ),
            django.db.models.Prefetch(
                "employee_by_mentor",
                queryset=frp_generated_django.Employee._default_manager.order_by("id"),
            ),
        )
    )


def colleagues_from_row(
    row: "frp_generated_django.Employee",
) -> frp_generated_models.Colleagues:
    return frp_generated_models.Colleagues.model_validate(
        {
            "id": row.id,
            "name": row.name,
            "nickname": row.nickname,
            "role": row.role,
            "department": {
                "id": row.department.id,
                "name": row.department.name,
                "company_id": row.department.company_id,
                "colleagues": [
                    {
                        "id": _employee.id,
                        "name": _employee.name,
                        "nickname": _employee.nickname,
                        "role": _employee.role,
                        "mentor_id": _employee.mentor_id,
                    }
                    for _employee in typing.cast(
                        "django.db.models.Manager[frp_generated_django.Employee]",
                        getattr(row.department, "employee_by_department"),
                    ).all()
                ],
            },
            "mentor_id": row.mentor_id,
            "mentees": [
                {
                    "id": _employee.id,
                    "name": _employee.name,
                    "nickname": _employee.nickname,
                    "role": _employee.role,
                    "department_id": _employee.department_id,
                }
                for _employee in typing.cast(
                    "django.db.models.Manager[frp_generated_django.Employee]",
                    getattr(row, "employee_by_mentor"),
                ).all()
            ],
        },
        strict=False,
    )
"""
        in queries_of([colleagues()]).root
    )


@pytest.mark.parametrize(
    ("module", "diagnostic"),
    [
        (Modules.queries, ViewModelIntoQueriesModule),
        (Modules.django, ViewModelIntoDjangoModelsModule),
    ],
    ids=["queries", "django-models"],
)
def test_a_union_model_into_a_django_module_is_a_diagnostic(
    module: ModulePath,
    diagnostic: type[ViewModelIntoQueriesModule | ViewModelIntoDjangoModelsModule],
) -> None:
    """Only the Union's model goes there; its Variants' go elsewhere."""
    union = any_employee()
    compiled = compile_all([], [union])
    placed = PlacedModels(
        modules={
            **placed_into(Generated.models, compiled).modules,
            union.name: module,
        }
    )

    match DjangoQueriesRenderer.render(
        entity_graph(),
        Modules.config(),
        (*compiled.views, *compiled.unions),
        placed,
        first_party=generated_by(Modules.config(), placed),
    ):
        case Ok(files):
            pytest.fail(f"expected diagnostics, got {files}")
        case Err(error):
            assert error.diagnostics == (diagnostic(view=union.name, module=module),)


def test_golden_union_from_row() -> None:
    """The last Variant is the default: the QuerySet keeps no row no Variant
    keeps."""
    text = queries_text([], [any_employee(other_employee())])

    assert (
        """\
def any_employee_from_row(
    row: "frp_generated_django.Employee",
) -> frp_generated_models.AnyEmployee:
    match vars(row)["_variant"]:
        case "LeadEmployee":
            return lead_employee_from_row(row)
        case "NicknamedEmployee":
            return nicknamed_employee_from_row(row)
        case _:
            return other_employee_from_row(row)
"""
        in text.root
    )


def test_golden_field_inequality_with_nulls() -> None:
    """`label != caption` is true where exactly one is `None`, and false where
    both are."""
    (view,) = (view for view in read_views() if view.name.root == "LabelIsNotCaption")

    assert (
        """\
    kept.query.add_q(
        (
            (
                ~django.db.models.Q(
                    label=django.db.models.F("caption"),
                )
                & django.db.models.Q(
                    label__isnull=False,
                    caption__isnull=False,
                )
            )
            | django.db.models.Q(
                label__isnull=True,
                caption__isnull=False,
            )
            | django.db.models.Q(
                label__isnull=False,
                caption__isnull=True,
            )
        ),
    )
"""
        in queries_text([view], []).root
    )


def test_golden_literal_enum_field_is_read_through_its_enum() -> None:
    (view,) = (view for view in read_views() if view.name.root == "Gold")

    assert (
        """\
            "level": functional_relational_python.outputs.django.test_queries.Level(
                row.level
            ),
"""
        in queries_text([view], []).root
    )


def test_a_view_named_like_a_query_set_method_is_a_diagnostic() -> None:
    view = View(
        name=ViewName("Manager"),
        query=steps.from_(Company),
    )
    compiled = compile_all([view])

    match DjangoQueriesRenderer.render(
        entity_graph(),
        Modules.config(),
        compiled.views,
        placed_into(Generated.models, compiled),
        first_party=generated_by(
            Modules.config(), placed_into(Generated.models, compiled)
        ),
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


class FirmId(Id[int]):
    pass


class Firm:
    """Holds an Entity named like `Company`."""

    class Company(pydantic.BaseModel, frozen=True):
        id: FirmId


def test_entities_named_alike_are_a_clash_though_their_models_are_not() -> None:
    """Their QuerySets and Managers are named after the Entities."""
    config = Modules.config(
        {Firm.Company: GeneratedModel(name=DjangoModelName("Firm"))}
    )
    graph = EntityGraph.build(
        [Company, Firm.Company], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()

    match DjangoQueriesRenderer.render(
        graph,
        config,
        (),
        PlacedModels(modules={}),
        first_party=generated_by(config, PlacedModels(modules={})),
    ):
        case Ok(files):
            pytest.fail(f"expected diagnostics, got {files}")
        case Err(error):
            assert {
                diagnostic.name
                for diagnostic in error.diagnostics
                if isinstance(diagnostic, NameClash)
            } == {Identifier("CompanyQuerySet"), Identifier("CompanyManager")}


type Query = Callable[[QuerySet[Model]], QuerySet[Model]]
"""Filters, orders or slices a QuerySet before it is read as a View."""


def every_row(queryset: QuerySet[Model]) -> QuerySet[Model]:
    return queryset.order_by("pk")


class Database:
    """The generated View models, Django models and queries, run in this process
    against an in-memory SQLite database."""

    generated: ClassVar[DjangoModels] = Modules.config()
    """The config with every Django model generated."""

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
            Database.install()

    @staticmethod
    def install() -> None:
        """Registers the app of the generated Django models, so that Django finds
        the reverse relations among its models, which it looks for in installed
        apps only."""
        label = Modules.config().app_label.root
        module = types.ModuleType(label)
        module.__path__ = [str(Path(__file__).parent)]
        config = AppConfig(label, module)
        config.apps = apps
        config.models = apps.all_models[label]
        apps.app_configs[label] = config
        apps.clear_cache()

    @staticmethod
    def files(
        views: Iterable[View],
        unions: Iterable[Union] = (),
        config: DjangoModels = generated,
    ) -> GeneratedFiles:
        """The View and Union models, Django models and queries of `views` and
        `unions`."""
        compiled = compile_all(views, unions)
        plan = PythonPlan.of(
            compiler.CompiledCatalog(
                views=compiled.views,
                write_models=(),
                unions=compiled.unions,
                write_model_unions=(),
                one_to_one_marker=OneToOneMarker.fake(),
            ),
            PythonLayout.models_only(into(Generated.models)),
        ).unwrap()
        return GeneratedFiles.merged(
            [
                GeneratedFiles(
                    files=ModelsRenderer.files(
                        plan,
                        first_party=generated_by(
                            config, placed_into(Generated.models, compiled)
                        ),
                    )
                ),
                DjangoModelsRenderer.render(
                    entity_graph(),
                    config,
                    first_party=generated_by(
                        config, placed_into(Generated.models, compiled)
                    ),
                ).unwrap(),
                DjangoQueriesRenderer.render(
                    entity_graph(),
                    config,
                    (*compiled.views, *compiled.unions),
                    placed_into(Generated.models, compiled),
                    first_party=generated_by(
                        config, placed_into(Generated.models, compiled)
                    ),
                ).unwrap(),
            ]
        ).unwrap()

    @staticmethod
    def text(files: GeneratedFiles, module: ModulePath) -> FileText:
        (text,) = (file.text for file in files.files if file.path == module.file_path())
        return text

    @staticmethod
    @contextlib.contextmanager
    def of(
        views: Iterable[View],
        unions: Iterable[Union] = (),
        config: DjangoModels = generated,
        existing: Iterable[tuple[ModulePath, FileText]] = (),
    ) -> Generator[types.ModuleType]:
        """The module of Django models, with a table per model, for `views` and
        `unions`. `existing` are the modules of the Django models `config` names
        as existing, with their source; their models are in the generated app.

        The models are unregistered afterwards, so the next call can define them
        again.
        """
        Database.configure()
        files = Database.files(views, unions, config)
        label = config.app_label.root
        sys.modules[Generated.models.root] = Executed.models(
            Database.text(files, Generated.models)
        )
        sources = (
            *existing,
            *(
                (module, Database.text(files, module))
                for module in (Modules.queries, Modules.django)
            ),
        )
        try:
            for module, source in sources:
                executed = types.ModuleType(module.root)
                sys.modules[module.root] = executed
                exec(source.root, executed.__dict__)
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
            for module in (Generated.models, *(module for module, _ in sources)):
                _ = sys.modules.pop(module.root, None)

    @staticmethod
    def model_of(entity: type[pydantic.BaseModel], config: DjangoModels) -> type[Model]:
        """The Django model of `entity`, existing or generated, from the modules
        `of` executed."""
        symbol = config.model_of(entity)
        match getattr(sys.modules[symbol.module.root], symbol.bound_name().root):
            case type() as model if issubclass(model, Model):
                return model
            case found:
                pytest.fail(f"expected a Django model, got {found!r}")

    @staticmethod
    def put(
        entities: Iterable[pydantic.BaseModel],
        config: DjangoModels = generated,
    ) -> None:
        """Stores each Entity as a row of its Django model, in one transaction, so
        References are checked once every row is stored. An enum is stored as its
        value, as Django stores one it has a built-in rule for."""
        with transaction.atomic():
            for entity in entities:
                Database.create(entity, config)

    @staticmethod
    def create(entity: pydantic.BaseModel, config: DjangoModels) -> None:
        """Stores `entity` as a row of its Django model."""
        model = Database.model_of(type(entity), config)
        node = entity_graph().entities[type(entity)]
        _ = model.objects.create(
            **{
                DjangoModelsRenderer.attribute_of(
                    config, node, FieldName(name)
                ).root: value
                for name, value in entity.model_dump(mode="json").items()
            }
        )

    @staticmethod
    def primary_key(entity: pydantic.BaseModel) -> IdValue:
        match getattr(entity, EntityNode.primary_key.root):
            case pydantic.RootModel() as id_value:
                return id_value
            case other:
                pytest.fail(f"expected an ID, got {other!r}")

    @staticmethod
    def rejected(
        entities: Iterable[pydantic.BaseModel],
        from_model: type[pydantic.BaseModel],
        config: DjangoModels,
    ) -> frozenset[IdValue]:
        """The IDs of the `from_model` Entities the database refuses to store, each
        stored on its own after every other Entity. Nothing is kept, so a refused
        row leaves no Reference to it dangling."""
        refused: set[IdValue] = set()
        with transaction.atomic():
            for entity in entities:
                if type(entity) is not from_model:
                    Database.create(entity, config)
            for entity in entities:
                if type(entity) is from_model:
                    try:
                        with transaction.atomic():
                            Database.create(entity, config)
                    except IntegrityError:
                        refused.add(Database.primary_key(entity))
            transaction.set_rollback(True)
        return frozenset(refused)

    @staticmethod
    def prefetches(
        views: Iterable[compiler.CompiledView], config: DjangoModels
    ) -> frozenset[tuple[RelatedPath, ...]]:
        """The prefetches the query code of `views` makes, each once, nested ones
        included, each by the paths of the prefetches to it. Each takes at most
        one database query more, however many rows there are: none when there is
        no row to prefetch for."""

        def chains(
            found: Iterable[Prefetched], before: tuple[RelatedPath, ...]
        ) -> Iterator[tuple[RelatedPath, ...]]:
            for prefetch in found:
                chain = (*before, prefetch.path)
                yield chain
                yield from chains(prefetch.prefetches, chain)

        return frozenset(
            chain
            for view in views
            for chain in chains(
                ViewQueries.of(view, config, Generated.models).prefetches, ()
            )
        )

    @staticmethod
    def read(
        django_models: types.ModuleType,
        view: compiler.CompiledView,
        query: Query,
        config: DjangoModels = generated,
    ) -> tuple[engine.ViewRow, ...]:
        """The View's results, from `query` of the from_model's QuerySet, checking
        they took one database query, and at most one more per prefetch."""
        model = getattr(django_models, config.model_name_of(view.from_model).root)
        method = f"as_{view.names.function.root}"
        with CaptureQueriesContext(connection) as captured:
            rows = tuple(
                Executed.row_of(row)
                for row in getattr(query(model.objects.all()), method)()
            )
        most = 1 + len(Database.prefetches([view], config))
        assert 1 <= len(captured.captured_queries) <= most, captured.captured_queries
        return rows

    @staticmethod
    def read_union(
        django_models: types.ModuleType, union: compiler.CompiledUnion, query: Query
    ) -> tuple[engine.VariantRow, ...]:
        """The Union's results, from `query` of the from_model's QuerySet, each with
        the Variant View whose model it is, checking they took one database
        query, and at most one more per prefetch of any Variant, each once."""
        model = getattr(django_models, union.from_model.__name__)
        method = f"as_{union.names.function.root}"
        by_model = {leaf.names.model.name.root: leaf.name for leaf in union.leaves()}
        with CaptureQueriesContext(connection) as captured:
            rows = tuple(
                engine.VariantRow(
                    view=by_model[type(row).__name__], row=Executed.row_of(row)
                )
                for row in getattr(query(model.objects.all()), method)()
            )
        most = 1 + len(Database.prefetches(union.leaves(), Database.generated))
        assert 1 <= len(captured.captured_queries) <= most, captured.captured_queries
        return rows


def entities_by_id(
    entities: Iterable[pydantic.BaseModel],
) -> tuple[pydantic.BaseModel, ...]:
    """`entities` in ID order, so the engine reads them in the order the database
    orders them by primary key."""
    return tuple(sorted(entities, key=lambda entity: entity.model_dump()["id"]))


def view_engine(
    compiled: compiler.CompiledViews, entities: Iterable[pydantic.BaseModel]
) -> engine.InMemoryViewEngine:
    """An engine holding `entities`, in the order the database orders them."""
    held = engine.InMemoryViewEngine(entity_graph(), compiled)
    for entity in entities_by_id(entities):
        held.put(entity).unwrap()
    return held


def engine_rows(
    views: Iterable[compiler.CompiledView], entities: Iterable[pydantic.BaseModel]
) -> dict[ViewName, tuple[engine.ViewRow, ...]]:
    compiled = tuple(views)
    held = view_engine(compiler.CompiledViews(views=compiled, unions=()), entities)
    return {view.name: held.read(view.name).unwrap() for view in compiled}


def badges(employees: Sequence[Employee]) -> tuple[Badge, ...]:
    """A Badge for every Employee but the last, so one Employee has none; each
    with or without a label and caption, some equal."""
    return tuple(
        Badge(
            id=BadgeId(index + 10),
            employee_id=employee.id,
            label=None if index % 3 == 0 else Text(f"label {index % 2}"),
            caption=None if index % 2 == 0 else Text("label 1"),
            level=Level.GOLD if index % 2 else Level.BRONZE,
            rank=Rank(None if index % 4 == 0 else index),
            stars=Stars(Rank(None if index % 3 == 1 else index)),
        )
        for index, employee in enumerate(employees[:-1])
    )


def organisation() -> tuple[pydantic.BaseModel, ...]:
    """Several Entities of each kind, so a query per row would show. Some optional
    References are absent and some present; a mentor may be stored after the
    Employee they mentor. Every Employee but one wears a Badge."""
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
    return (*badges(employees), *employees, *departments, *companies)


def read_views() -> list[View]:
    """Views of every kind a query reads: resolving, reversing, flattening,
    filtering and requiring."""
    return [
        View(
            name=ViewName("Plain"),
            query=steps.from_(Employee),
        ),
        View(
            name=ViewName("RequiredEntity"),
            query=steps.from_(
                Employee, steps.resolve(named(FieldName("department_id")))
            ),
        ),
        View(
            name=ViewName("OptionalEntity"),
            query=steps.from_(Employee, steps.resolve(named(FieldName("mentor_id")))),
        ),
        View(
            name=ViewName("ReferenceWithoutIdSuffix"),
            query=steps.from_(Company, steps.resolve(steps.references())),
        ),
        View(
            name=ViewName("Chain"),
            query=steps.from_(
                Employee,
                steps.resolve(
                    named(FieldName("department_id")),
                    steps.resolve(
                        named(FieldName("company_id")),
                        steps.resolve(named(FieldName("parent"))),
                    ),
                ),
            ),
        ),
        View(
            name=ViewName("SelfReference"),
            query=steps.from_(
                Employee,
                steps.resolve(
                    steps.references(),
                    steps.resolve(steps.references(), steps.omit(FieldName("name"))),
                ),
            ),
        ),
        flattened_employee(),
        View(
            name=ViewName("OnlyAdded"),
            query=steps.from_(
                Employee,
                steps.pick([]),
                steps.add(FieldName("kind"), str, default="x"),
            ),
        ),
        View(
            name=ViewName("Reversed"),
            query=steps.from_(Employee, steps.reverse(Badge)),
        ),
        View(
            name=ViewName("ReversedNested"),
            query=steps.from_(
                Employee, steps.reverse(Badge, steps.pick([FieldName("label")]))
            ),
        ),
        View(
            name=ViewName("ReverseInNested"),
            query=steps.from_(
                Badge,
                steps.resolve(named(FieldName("employee_id")), steps.reverse(Badge)),
            ),
        ),
        # Django fills in a way back over a one-to-one Reference with the row it
        # came from, dropping what it selected related past it.
        View(
            name=ViewName("ReversedAndBack"),
            query=steps.from_(
                Employee,
                steps.reverse(
                    Badge,
                    steps.resolve(
                        named(FieldName("employee_id")),
                        steps.resolve(named(FieldName("department_id"))),
                    ),
                ),
            ),
        ),
        View(
            name=ViewName("ResolvedAndBack"),
            query=steps.from_(
                Badge,
                steps.resolve(
                    named(FieldName("employee_id")),
                    steps.reverse(
                        Badge, steps.resolve(named(FieldName("employee_id")))
                    ),
                ),
            ),
        ),
        View(
            name=ViewName("ReversedFlattened"),
            query=steps.from_(
                Employee,
                steps.reverse(Badge, steps.pick([FieldName("level")])),
                steps.flatten(FieldName("badge")),
            ),
        ),
        # A NULL rank of a Badge found is `Rank(None)`; one of no Badge, `None`.
        # Likewise for stars, wrapping a Rank.
        View(
            name=ViewName("FlattenedRank"),
            query=steps.from_(
                Employee,
                steps.reverse(
                    Badge, steps.pick([FieldName("rank"), FieldName("stars")])
                ),
                steps.flatten(FieldName("badge")),
            ),
        ),
        View(
            name=ViewName("Lead"),
            query=steps.from_(
                Employee, steps.where(field_is(FieldName("role"), Role.LEAD))
            ),
        ),
        View(
            name=ViewName("Gold"),
            query=steps.from_(
                Badge, steps.where(field_is(FieldName("level"), Level.GOLD))
            ),
        ),
        View(
            name=ViewName("NotNicknamed"),
            query=steps.from_(
                Employee,
                steps.where(
                    conditions.ne(
                        conditions.field(FieldName("nickname")), Text("nick 2")
                    )
                ),
            ),
        ),
        View(
            name=ViewName("LabelIsCaption"),
            query=steps.from_(
                Badge,
                steps.where(
                    conditions.eq(
                        conditions.field(FieldName("label")),
                        conditions.field(FieldName("caption")),
                    )
                ),
            ),
        ),
        View(
            name=ViewName("LabelIsNotCaption"),
            query=steps.from_(
                Badge,
                steps.where(
                    conditions.not_(
                        conditions.eq(
                            conditions.field(FieldName("label")),
                            conditions.field(FieldName("caption")),
                        )
                    )
                ),
            ),
        ),
        View(
            name=ViewName("NicknameIsNotItself"),
            query=steps.from_(
                Employee,
                steps.where(
                    conditions.ne(
                        conditions.field(FieldName("nickname")),
                        conditions.field(FieldName("nickname")),
                    )
                ),
            ),
        ),
        # Some ranks are NULL: no ordering keeps them, and so its negation does.
        View(
            name=ViewName("NotOutranked"),
            query=steps.from_(
                Badge,
                steps.where(
                    conditions.not_(
                        conditions.lt(conditions.field(FieldName("rank")), Rank(2))
                    )
                ),
            ),
        ),
        View(
            name=ViewName("LeadOrNicknamed"),
            query=steps.from_(
                Employee,
                steps.where(
                    conditions.or_(
                        conditions.lt(conditions.field(FieldName("role")), Role.STAFF),
                        conditions.not_(
                            conditions.is_null(conditions.field(FieldName("nickname")))
                        ),
                    )
                ),
            ),
        ),
        View(
            name=ViewName("Staff"),
            query=steps.from_(Department, steps.reverse(Employee)),
        ),
        colleagues(),
        # The way back over a one-to-one Reference resumes at the row it came
        # from, so the mentees are prefetched from it.
        View(
            name=ViewName("MenteesAndBack"),
            query=steps.from_(
                Employee,
                steps.reverse(
                    Badge,
                    steps.resolve(
                        named(FieldName("employee_id")),
                        steps.reverse(Employee, name=FieldName("mentees")),
                    ),
                ),
            ),
        ),
        View(
            name=ViewName("RequiredBadge"),
            query=steps.from_(
                Employee,
                steps.where(
                    conditions.not_(
                        conditions.eq(conditions.field(FieldName("id")), EmployeeId(6))
                    )
                ),
                steps.reverse(Badge),
                steps.require(FieldName("badge")),
            ),
        ),
        organisation_chart(),
    ]


@pytest.mark.parametrize("view", read_views(), ids=lambda view: view.name.root)
def test_a_view_reads_in_one_query_like_the_engine(view: View) -> None:
    entities = organisation()
    (compiled,) = compile_all([view]).views

    with Database.of([view]) as django_models:
        Database.put(entities)
        rows = Database.read(django_models, compiled, every_row)

    assert rows == engine_rows([compiled], entities)[view.name]


@pytest.mark.parametrize(
    ("view", "queries"),
    [
        (
            View(
                name=ViewName("Staff"),
                query=steps.from_(Department, steps.reverse(Employee)),
            ),
            2,
        ),
        (
            colleagues(),
            3,
        ),
        (
            organisation_chart(),
            3,
        ),
    ],
    ids=["top-level", "nested-and-top-level", "collection-in-collection"],
)
def test_a_list_reads_each_sorted_sequence_in_one_query_more(
    view: View, queries: int
) -> None:
    """However many rows the list has: organisation() has several of each."""
    (compiled,) = compile_all([view]).views

    with Database.of([view]) as django_models:
        Database.put(organisation())
        model = getattr(django_models, compiled.from_model.__name__)
        with CaptureQueriesContext(connection) as captured:
            rows = list(
                getattr(model.objects.all(), f"as_{compiled.names.function.root}")()
            )

    assert (len(rows) > 1, len(captured.captured_queries)) == (True, queries)


@pytest.mark.parametrize(
    "view",
    [view for view in read_views() if view.query.from_model is not Company],
    ids=lambda view: view.name.root,
)
def test_a_view_reads_through_an_existing_model_like_the_engine(view: View) -> None:
    """`Company`'s Django model is the application's own, which the generated
    models reference and the query selects related through."""
    entities = organisation()
    config = ExistingCompany.config()
    (compiled,) = compile_all([view]).views

    with Database.of([view], (), config, ExistingCompany.sources()) as django_models:
        Database.put(entities, config)
        rows = Database.read(django_models, compiled, every_row)

    assert rows == engine_rows([compiled], entities)[view.name]


def test_a_view_over_an_existing_model_reads_through_its_functions() -> None:
    """The existing model has no QuerySet with a method per View, so the
    application calls the View's functions."""
    view = View(
        name=ViewName("Hydrated"),
        query=steps.from_(Company, steps.resolve(steps.references())),
    )
    entities = organisation()
    config = ExistingCompany.config()
    (compiled,) = compile_all([view]).views
    function = compiled.names.function.root

    with Database.of([view], (), config, ExistingCompany.sources()):
        Database.put(entities, config)
        queries = sys.modules[Modules.queries.root]
        companies = Database.model_of(Company, config).objects.order_by("pk")
        with CaptureQueriesContext(connection) as captured:
            rows = tuple(
                Executed.row_of(getattr(queries, f"{function}_from_row")(row))
                for row in getattr(queries, f"{function}_queryset")(companies)
            )

    assert len(captured.captured_queries) == 1, captured.captured_queries
    assert rows == engine_rows([compiled], entities)[view.name]


def test_an_existing_model_gets_no_query_set_and_is_imported_for_type_checking() -> (
    None
):
    compiled = compile_all(golden_views())

    text = DjangoQueriesRenderer.render(
        entity_graph(),
        ExistingCompany.config(),
        compiled.views,
        placed_into(Generated.models, compiled),
        first_party=generated_by(
            ExistingCompany.config(), placed_into(Generated.models, compiled)
        ),
    ).unwrap()

    (file,) = text.files
    assert "CompanyQuerySet" not in file.text.root
    assert "CompanyManager" not in file.text.root
    assert (
        "if typing.TYPE_CHECKING:\n"
        "    import frp_existing_django\n"
        "\n"
        "    import frp_generated_django\n"
    ) in file.text.root
    assert (
        "def company_full_queryset(\n"
        '    queryset: "django.db.models.QuerySet[frp_existing_django.Firm]",\n'
    ) in file.text.root


def test_a_query_follows_the_reverse_accessor_an_existing_model_declares() -> None:
    """`Badge`'s existing model names its one-to-one Reference to the Employee
    `medal`, where a generated model would name it `badge_by_employee`."""
    config = Modules.config(
        {
            Badge: ExistingDjangoModel(
                module=ModulePath("frp_existing_badges"),
                name=Identifier("Medal"),
                reverse_accessors={FieldName("employee_id"): FieldName("medal")},
            )
        }
    )
    view = View(
        name=ViewName("Decorated"),
        query=steps.from_(Employee, steps.reverse(Badge)),
    )
    compiled = compile_all([view])

    (file,) = (
        DjangoQueriesRenderer.render(
            entity_graph(),
            config,
            compiled.views,
            placed_into(Generated.models, compiled),
            first_party=generated_by(config, placed_into(Generated.models, compiled)),
        )
        .unwrap()
        .files
    )

    assert "badge_by_employee" not in file.text.root
    assert '"medal"' in file.text.root


def undeclaring_company() -> DjangoModels:
    """`ExistingCompany`'s config, its model declaring no reverse accessor."""
    return Modules.config(
        {
            Company: ExistingDjangoModel(
                module=ExistingCompany.module, name=Identifier("Firm")
            )
        }
    )


def rendered_with(
    config: DjangoModels, view: View
) -> Result[GeneratedFiles, DiagnosticsError[DjangoQueriesDiagnostic]]:
    compiled = compile_all([view])
    return DjangoQueriesRenderer.render(
        entity_graph(),
        config,
        compiled.views,
        placed_into(Generated.models, compiled),
        first_party=generated_by(config, placed_into(Generated.models, compiled)),
    )


@pytest.mark.parametrize(
    "query",
    [
        steps.from_(Company, steps.reverse(Company)),
        steps.from_(
            Department,
            steps.resolve(named(FieldName("company_id")), steps.reverse(Company)),
        ),
    ],
    ids=["top-level", "nested"],
)
def test_reversing_a_reference_an_existing_model_declares_no_accessor_for_is_a_diagnostic(
    query: steps.Query,
) -> None:
    view = View(name=ViewName("Subsidiaries"), query=query)

    match rendered_with(undeclaring_company(), view):
        case Ok(files):
            pytest.fail(f"expected diagnostics, got {files}")
        case Err(error):
            assert error.diagnostics == (
                ReversedUndeclaredAccessor(
                    view=view.name, entity=Company, field=FieldName("parent")
                ),
            )


def test_a_view_not_reversing_a_reference_needs_no_declared_accessor() -> None:
    view = View(
        name=ViewName("Hydrated"),
        query=steps.from_(Company, steps.resolve(steps.references())),
    )

    assert isinstance(rendered_with(undeclaring_company(), view), Ok)


class ExistingCompanyKeyedApart:
    """`Company`'s Django model as the application wrote it, `Firm`, whose
    primary key `key` is not the Entity's `id`, which a unique column holds. Its
    Reference `parent` holds a parent's `id`, not its key."""

    source: ClassVar[FileText] = FileText(
        "from django.db import models\n"
        "\n"
        "\n"
        "class Firm(models.Model):\n"
        "    key = models.BigAutoField(primary_key=True)\n"
        "    id = models.BigIntegerField(unique=True)\n"
        "    name = models.TextField()\n"
        "    parent = models.ForeignKey(\n"
        '        "self",\n'
        "        on_delete=models.PROTECT,\n"
        '        to_field="id",\n'
        '        related_name="subsidiaries",\n'
        "        null=True,\n"
        "    )\n"
        "\n"
        "    class Meta:\n"
        '        app_label = "frp_generated"\n'
    )


def test_a_sorted_sequence_is_ordered_by_id_where_the_primary_key_is_not_id() -> None:
    """The Companies are stored in descending `id` order, so their keys ascend
    as their `id`s descend."""
    view = View(
        name=ViewName("Subsidiaries"),
        query=steps.from_(Company, steps.reverse(Company)),
    )
    companies = tuple(
        entity for entity in organisation() if isinstance(entity, Company)
    )
    config = ExistingCompany.config()
    (compiled,) = compile_all([view]).views
    function = compiled.names.function.root
    existing = ((ExistingCompany.module, ExistingCompanyKeyedApart.source),)

    with Database.of([view], (), config, existing):
        Database.put(
            sorted(companies, key=lambda company: company.id.root, reverse=True),
            config,
        )
        queries = sys.modules[Modules.queries.root]
        firms = Database.model_of(Company, config).objects.order_by("id")
        rows = tuple(
            Executed.row_of(getattr(queries, f"{function}_from_row")(row))
            for row in getattr(queries, f"{function}_queryset")(firms)
        )

    assert rows == engine_rows([compiled], companies)[view.name]


@pytest.mark.parametrize("view", read_views(), ids=lambda view: view.name.root)
def test_a_view_reads_through_a_related_name_like_the_engine(view: View) -> None:
    """A one-to-one Reference with a related name is followed backwards through
    it."""
    entities = organisation()
    (compiled,) = compile_all([view]).views
    config = Modules.with_options(ModelOptions())

    with Database.of([view], (), config) as django_models:
        Database.put(entities, config)
        rows = Database.read(django_models, compiled, every_row)

    assert rows == engine_rows([compiled], entities)[view.name]


def prefixed_model_name(entity: type[pydantic.BaseModel]) -> DjangoModelName:
    """`Django<Entity>`."""
    return DjangoModelName(f"Django{entity.__name__}")


def entity_model_name(_: type[pydantic.BaseModel]) -> Unset:
    return Unset()


@pytest.mark.parametrize(
    "model_name_of",
    [entity_model_name, prefixed_model_name],
    ids=["entity names", "model names"],
)
@pytest.mark.parametrize(
    "view",
    [
        *read_views(),
        View(
            name=ViewName("NotTheFirstBadge"),
            query=steps.from_(
                Badge,
                steps.where(
                    conditions.not_(
                        conditions.eq(conditions.field(FieldName("id")), BadgeId(1))
                    )
                ),
            ),
        ),
    ],
    ids=lambda view: view.name.root,
)
def test_a_view_reads_a_model_keyed_by_its_reference_like_the_engine(
    view: View,
    model_name_of: Callable[[type[pydantic.BaseModel]], DjangoModelName | Unset],
) -> None:
    """A Badge's model has the Employee it is of as its primary key, so its `id`
    is read from that column. Each Badge's `id` is its Employee's. A model named
    unlike its Entity is referred to, and named after, by its own name."""
    entities = tuple(
        entity.model_copy(update={"id": BadgeId(entity.employee_id.root)})
        if isinstance(entity, Badge)
        else entity
        for entity in organisation()
    )
    (compiled,) = compile_all([view]).views
    config = Modules.config(
        {
            **{
                entity: GeneratedModel(name=model_name_of(entity))
                for entity in entity_graph().entities
            },
            Badge: GeneratedModel(
                name=model_name_of(Badge),
                primary_key=ReferencePrimaryKey(reference=FieldName("employee_id")),
            ),
        }
    )

    with Database.of([view], (), config) as django_models:
        Database.put(entities, config)
        rows = Database.read(django_models, compiled, every_row, config)

    assert rows == engine_rows([compiled], entities)[view.name]


def test_a_view_reads_the_rows_the_manager_base_keeps() -> None:
    view = View(
        name=ViewName("Hydrated"),
        query=steps.from_(Employee, steps.resolve(steps.references())),
    )
    entities = organisation()
    (compiled,) = compile_all([view]).views
    config = Modules.with_bases()

    with Database.of([view], (), config) as django_models:
        Database.put(entities, config)
        rows = Database.read(django_models, compiled, every_row)

    # The Manager base leaves out the Employee with the lowest ID.
    assert rows == engine_rows([compiled], entities)[view.name][1:]


def with_mentor_from_last(queryset: QuerySet[Model]) -> QuerySet[Model]:
    return queryset.filter(mentor__isnull=False).order_by("-pk")[1:3]


def test_a_view_reads_the_filtered_ordered_and_sliced_query_set() -> None:
    view = View(
        name=ViewName("Hydrated"),
        query=steps.from_(Employee, steps.resolve(steps.references())),
    )
    entities = organisation()
    (compiled,) = compile_all([view]).views
    with_mentor = [
        row
        for row in engine_rows([compiled], entities)[view.name]
        if row.root[FieldName("mentor")] is not None
    ]

    with Database.of([view]) as django_models:
        Database.put(entities)
        rows = Database.read(django_models, compiled, with_mentor_from_last)

    assert rows == tuple(reversed(with_mentor))[1:3]


def last_two(queryset: QuerySet[Model]) -> QuerySet[Model]:
    return queryset.order_by("-pk")[:2]


def lead() -> View:
    return View(
        name=ViewName("Lead"),
        query=steps.from_(
            Employee, steps.where(field_is(FieldName("role"), Role.LEAD))
        ),
    )


def test_a_slice_is_of_the_rows_a_filtered_view_keeps() -> None:
    """The last two Employees are not both leads, but the last two leads are
    read."""
    entities = organisation()
    (compiled,) = compile_all([lead()]).views

    with Database.of([lead()]) as django_models:
        Database.put(entities)
        rows = Database.read(django_models, compiled, last_two)

    assert [row.root[FieldName("id")] for row in rows] == [
        EmployeeId(5),
        EmployeeId(3),
    ]


def test_django_adds_a_condition_to_a_sliced_query_before_its_slice() -> None:
    """The query code relies on this: `query.add_q()` on a sliced QuerySet
    filters before the LIMIT, where `filter()` refuses. If a Django upgrade
    changes it, a filtered View's slice is no longer of the rows it keeps."""
    with Database.of([lead()]) as django_models:
        Database.put(organisation())
        sliced = django_models.Employee.objects.order_by("-pk")[:2]
        kept = sliced.all()
        kept.query.add_q(Q(role=Role.LEAD.value))

        with pytest.raises(TypeError, match="slice"):
            _ = sliced.filter(role=Role.LEAD.value)
        assert [employee.pk for employee in kept] == [5, 3], (
            "Django's `query.add_q()` on a sliced QuerySet no longer filters "
            "before the slice; the generated queries rely on it."
        )


def test_the_method_of_a_filtered_view_says_its_filters_apply_before_a_slice() -> None:
    assert (
        "    def as_lead(\n"
        "        self,\n"
        "    ) -> collections.abc.Iterator[frp_generated_models.Lead]:\n"
        '        """The models of the rows of this QuerySet that the Filters keep.\n'
        "\n"
        "        The Filters apply before a slice the QuerySet has taken: `[:10]` "
        "gives\n"
        "        the first ten rows they keep.\n"
        '        """\n'
    ) in queries_of([lead()]).root


def tagged(name: ViewName, tag: test_compiler.Tag, *pipeline: steps.Step) -> View:
    return View(
        name=name,
        query=steps.from_(
            Employee,
            steps.pipeline(*pipeline, test_compiler.added_tag(FieldName("kind"), tag)),
        ),
    )


def any_employee(*rest: View) -> Union:
    """Leads with their mentor, then Employees with a nickname or the first ID
    with their Badge, so the first lead is kept by both and is a lead; then
    `rest`."""
    return Union(
        name=UnionName("AnyEmployee"),
        discriminator=FieldName("kind"),
        variants=(
            tagged(
                ViewName("LeadEmployee"),
                test_compiler.Tag("lead"),
                steps.where(field_is(FieldName("role"), Role.LEAD)),
                steps.resolve(named(FieldName("mentor_id"))),
            ),
            tagged(
                ViewName("NicknamedEmployee"),
                test_compiler.Tag("nicknamed"),
                steps.where(
                    conditions.or_(
                        conditions.not_(
                            conditions.is_null(conditions.field(FieldName("nickname")))
                        ),
                        conditions.eq(conditions.field(FieldName("id")), EmployeeId(1)),
                    )
                ),
                steps.reverse(Badge),
            ),
            *rest,
        ),
    )


def other_employee() -> View:
    return tagged(ViewName("OtherEmployee"), test_compiler.Tag("other"))


def without_last_nickname(entity: pydantic.BaseModel) -> pydantic.BaseModel:
    """The last Employee, a staff member without a Badge, loses their nickname,
    so that no filtered Variant keeps them."""
    if isinstance(entity, Employee) and entity.id == EmployeeId(6):
        return entity.model_copy(update={"nickname": None})
    return entity


@pytest.mark.parametrize(
    ("union", "query"),
    [
        (any_employee(), every_row),
        (any_employee(), last_two),
        (any_employee(other_employee()), every_row),
        (any_employee(other_employee()), last_two),
    ],
    ids=["filtered", "filtered-sliced", "unfiltered-last", "unfiltered-last-sliced"],
)
def test_a_union_reads_in_one_query_like_the_engine(union: Union, query: Query) -> None:
    entities = tuple(map(without_last_nickname, organisation()))
    compiled = compile_all([], [union])
    (compiled_union,) = compiled.unions
    expected = view_engine(compiled, entities).read_union(union.name).unwrap()
    if query is last_two:
        expected = tuple(reversed(expected))[:2]

    with Database.of([], [union]) as django_models:
        Database.put(entities)
        rows = Database.read_union(django_models, compiled_union, query)

    assert rows == expected


def test_each_variant_of_the_union_keeps_rows() -> None:
    """Catches a test Union whose later Variants read nothing."""
    union = any_employee(other_employee())
    compiled = compile_all([], [union])
    entities = tuple(map(without_last_nickname, organisation()))

    rows = view_engine(compiled, entities).read_union(union.name).unwrap()

    assert {row.view for row in rows} == {
        leaf.name for leaf in compiled.unions[0].leaves()
    }


def hydrated() -> View:
    return View(
        name=ViewName("Hydrated"),
        query=steps.from_(Employee, steps.resolve(steps.references())),
    )


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
        Database.put(entities)
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
        query=steps.from_(Employee, steps.resolve(steps.references())),
        base=StrictBase,
    )
    entities = organisation()
    (compiled,) = compile_all([view]).views

    with Database.of([view]) as django_models:
        Database.put(entities)
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


class PlainBases:
    """`Employee`'s QuerySet and Manager bases as an application would write them
    for it alone: not generic, so they are plain bases."""

    module: ClassVar[ModulePath] = ModulePath("frp_plain_bases")
    source: ClassVar[FileText] = FileText(
        "import typing\n"
        "\n"
        "from django.db import models\n"
        "\n"
        "if typing.TYPE_CHECKING:\n"
        "    import frp_generated_django\n"
        "\n"
        "\n"
        "class EmployeeQuerySetBase(\n"
        '    models.QuerySet["frp_generated_django.Employee"],\n'
        "):\n"
        "    def newest_first(self) -> typing.Self:\n"
        '        return self.order_by("-pk")\n'
        "\n"
        "\n"
        'class EmployeeManagerBase(models.Manager["frp_generated_django.Employee"]):\n'
        "    pass\n"
    )

    @staticmethod
    def config() -> DjangoModels:
        return PlainBases.config_in(PlainBases.module)

    @staticmethod
    def long_config() -> DjangoModels:
        """With bases whose names are long enough that ruff format splits the
        class statements."""
        return PlainBases.config_in(LongModules.views)

    @staticmethod
    def config_in(module: ModulePath) -> DjangoModels:
        return Modules.config(
            {
                Employee: GeneratedModel(
                    query_set_base=PlainBase(
                        origin=Symbol(
                            module=module,
                            qualname=(Identifier("EmployeeQuerySetBase"),),
                        )
                    ),
                    manager_base=PlainBase(
                        origin=Symbol(
                            module=module,
                            qualname=(Identifier("EmployeeManagerBase"),),
                        )
                    ),
                )
            }
        )

    @staticmethod
    def sources() -> tuple[tuple[ModulePath, FileText], ...]:
        return ((PlainBases.module, PlainBases.source),)


class DeclaredManager:
    """A model base declaring every model's manager, as older django-stubs
    declare it on `Model`: a type checker types a model's `objects` as declared
    there unless the model annotates it."""

    module: ClassVar[ModulePath] = ModulePath("frp_declared_manager")
    source: ClassVar[FileText] = FileText(
        "import typing\n"
        "\n"
        "from django.db import models\n"
        "\n"
        "\n"
        "class DeclaredManagerModel(models.Model):\n"
        "    objects: typing.ClassVar[models.Manager[typing.Self]]\n"
        "\n"
        "    class Meta:\n"
        "        abstract = True\n"
    )

    @staticmethod
    def config() -> DjangoModels:
        return Modules.with_options(
            ModelOptions(
                bases=(
                    Symbol(
                        module=DeclaredManager.module,
                        qualname=(Identifier("DeclaredManagerModel"),),
                    ),
                )
            )
        )

    @staticmethod
    def sources() -> tuple[tuple[ModulePath, FileText], ...]:
        return ((DeclaredManager.module, DeclaredManager.source),)


def every_definition() -> tuple[list[View], list[Union]]:
    """Views and a Union generating every kind of query code."""
    return (
        [*golden_views(), *read_views()],
        [any_employee(other_employee())],
    )


@pytest.mark.parametrize(
    ("config", "views"),
    [
        (Modules.config(), Generated.models),
        (
            Modules.config().model_copy(update={"module": LongModules.django}),
            LongModules.views,
        ),
        (ExistingCompany.config(), Generated.models),
        (Modules.with_bases(), Generated.models),
        (PlainBases.config(), Generated.models),
        (PlainBases.long_config(), Generated.models),
    ],
    ids=["short", "long", "existing", "generic-bases", "plain-bases", "long-plain"],
)
def test_generated_queries_are_ruff_formatted_and_sorted(
    config: DjangoModels, views: ModulePath
) -> None:
    compiled = compile_all(*every_definition())
    (file,) = (
        DjangoQueriesRenderer.render(
            entity_graph(),
            config,
            (*compiled.views, *compiled.unions),
            placed_into(views, compiled),
            first_party=generated_by(config, placed_into(views, compiled)),
        )
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
    first_party = FirstParty.fake().with_generated(
        [config.module, config.queries, views]
    )
    assert test_symbols.ruff_import_diff(file.text, first_party) == FileText("")


def render_diagnostics(
    config: DjangoModels, views: ModulePath
) -> tuple[DjangoQueriesDiagnostic, ...]:
    view = View(
        name=ViewName("CompanyFull"),
        query=steps.from_(Company),
    )
    compiled = compile_all([view])
    match DjangoQueriesRenderer.render(
        entity_graph(),
        config,
        compiled.views,
        placed_into(views, compiled),
        first_party=generated_by(config, placed_into(views, compiled)),
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


def test_entities_with_one_name_are_left_to_the_django_models_to_report() -> None:
    graph = EntityGraph.build(
        [Company, another_company()], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()

    rendered = DjangoQueriesRenderer.render(
        graph,
        Modules.config(),
        (),
        PlacedModels(modules={}),
        first_party=generated_by(Modules.config(), PlacedModels(modules={})),
    )

    assert isinstance(rendered, Ok)


def plaque() -> type[pydantic.BaseModel]:
    """An Entity on a Company, with a field of a RootModel defined in a function,
    whose column may be NULL."""

    class PlaqueId(Id[int]):
        pass

    class Score(pydantic.RootModel[int | None]):
        pass

    class Plaque(pydantic.BaseModel):
        id: PlaqueId
        company_id: Annotated[CompanyId, OneToOne()]
        score: Score

    return Plaque


def test_an_unimportable_class_a_field_is_read_through_names_the_field() -> None:
    plaque_entity = plaque()
    graph = EntityGraph.build(
        [Company, plaque_entity], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    view = View(
        name=ViewName("CompanyScore"),
        query=steps.from_(
            Company,
            steps.reverse(plaque_entity, steps.pick([FieldName("score")])),
            steps.flatten(FieldName("plaque")),
        ),
    )
    compiled = compiler.compile_views(graph, [view], []).unwrap()

    rendered = DjangoQueriesRenderer.render(
        graph,
        Modules.config(),
        compiled.views,
        placed_into(Generated.models, compiled),
        first_party=generated_by(
            Modules.config(), placed_into(Generated.models, compiled)
        ),
    )

    assert isinstance(rendered, Err)
    assert (
        Unimportable(
            view=view.name,
            field=FieldName("score"),
            annotation=AnnotationText.of(
                plaque_entity.model_fields["score"].annotation
            ),
        )
        in rendered.error.diagnostics
    )


# Differential test: for generated Entities and pipelines, the generated queries
# and the InMemoryViewEngine give the same View results.

big_integers = st.integers(min_value=-(2**63), max_value=2**63 - 1)
# A small alphabet, so conditions comparing texts are true for some rows.
texts = st.builds(Text, st.text(alphabet="ab", max_size=2))


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
    worn_by = draw(st.lists(st.sampled_from(employee_ids), unique=True))
    badges = [
        Badge(
            id=BadgeId(index),
            employee_id=employee_id,
            label=draw(st.none() | texts),
            caption=draw(st.none() | texts),
            level=draw(st.sampled_from(Level)),
            rank=Rank(draw(st.none() | st.integers(min_value=0, max_value=2))),
            stars=Stars(Rank(draw(st.none() | st.integers(min_value=0, max_value=2)))),
        )
        for index, employee_id in enumerate(worn_by)
    ]
    return (*companies, *departments, *employees, *badges)


@st.composite
def field_conditions(
    draw: st.DrawFn,
    from_model: type[pydantic.BaseModel],
    stored: tuple[pydantic.BaseModel, ...],
    depth: Depth,
) -> ir.Condition:
    """A Condition over `from_model`'s fields, comparing them with each other, with
    `None`, or with values `stored` Entities have, so that it keeps some rows."""
    if depth.root > 0 and draw(st.booleans()):
        smaller = Depth(depth.root - 1)
        terms = draw(
            st.lists(
                field_conditions(from_model, stored, smaller), min_size=1, max_size=3
            )
        )
        match draw(st.sampled_from(["and", "or", "not"])):
            case "and":
                return conditions.and_(*terms)
            case "or":
                return conditions.or_(*terms)
            case _:
                return conditions.not_(terms[0])
    name = draw(st.sampled_from(list(from_model.model_fields)))
    field = conditions.field(FieldName(name))
    split = narrowing.split(from_model.model_fields[name].annotation)
    # A literal is never `None`, nor a RootModel of it.
    values = [
        value
        for entity in stored
        if type(entity) is from_model
        and (value := getattr(entity, name)) is not None
        and value not in (Rank(None), Stars(Rank(None)))
    ]
    same_type = [
        other
        for other, info in from_model.model_fields.items()
        if narrowing.split(info.annotation).required == split.required
    ]
    atoms: list[ir.Condition] = [
        operator(field, conditions.field(FieldName(draw(st.sampled_from(same_type)))))
        for operator in (conditions.eq, conditions.ne)
    ]
    if split.optionality is Optionality.OPTIONAL:
        atoms.append(conditions.is_null(field))
    if values:
        value = draw(st.sampled_from(values))
        atoms.extend(
            operator(field, value) for operator in (conditions.eq, conditions.ne)
        )
        # Of the test Entities' field types, only `Role` and `Rank` define `<`.
        if split.required in (Role, Rank):
            atoms.extend(
                operator(field, value)
                for operator in (
                    conditions.lt,
                    conditions.le,
                    conditions.gt,
                    conditions.ge,
                )
            )
    return draw(st.sampled_from(atoms))


@st.composite
def filtered_pipelines(
    draw: st.DrawFn,
    from_model: type[pydantic.BaseModel],
    filters: st.SearchStrategy[list[ir.Condition]],
) -> steps.Step:
    """A random pipeline over `from_model`, after maybe requiring one of its
    optional fields, and then filtering by each of `filters`.

    Filtering after requiring, so that a requirement is never of a field a
    Filter keeps `None`, which the compiler rejects."""
    optional = [
        FieldName(name)
        for name, info in from_model.model_fields.items()
        if narrowing.split(info.annotation).optionality is Optionality.OPTIONAL
    ]
    required = draw(st.lists(st.sampled_from(optional), max_size=1)) if optional else []
    where = draw(filters)
    return steps.pipeline(
        *map(steps.require, required),
        *map(steps.where, where),
        draw(
            pipelines(
                entity_graph(), Start.whole(from_model), Depth(2), Collections.DRAWN
            )
        ),
    )


class DifferentialCase(pydantic.BaseModel, frozen=True):
    views: tuple[View, ...]
    unions: tuple[Union, ...]
    entities: tuple[pydantic.BaseModel, ...]


@st.composite
def differential_cases(draw: st.DrawFn) -> DifferentialCase:
    """Views over every Entity, each with a random pipeline that may require and
    filter, and maybe a Union of such Views, each tagged, all filtered but maybe
    the last."""
    stored = draw(entities())
    from_models = [Company, Department, Employee, Badge]
    views: list[View] = []
    for index in range(draw(st.integers(min_value=1, max_value=3))):
        from_model = draw(st.sampled_from(from_models))
        views.append(
            View(
                name=ViewName(f"View{index}"),
                query=steps.from_(
                    from_model,
                    draw(
                        filtered_pipelines(
                            from_model,
                            st.lists(
                                field_conditions(from_model, stored, Depth(2)),
                                max_size=1,
                            ),
                        )
                    ),
                ),
            )
        )
    unions: list[Union] = []
    if draw(st.booleans()):
        from_model = draw(st.sampled_from(from_models))
        count = draw(st.integers(min_value=2, max_value=3))
        unfiltered_last = draw(st.booleans())
        variants = tuple(
            View(
                name=ViewName(f"Variant{index}"),
                query=steps.from_(
                    from_model,
                    draw(
                        filtered_pipelines(
                            from_model,
                            st.lists(
                                field_conditions(from_model, stored, Depth(1)),
                                min_size=0
                                if unfiltered_last and index == count - 1
                                else 1,
                                max_size=1,
                            ),
                        )
                    ),
                    test_compiler.added_tag(
                        FieldName("variant_tag"), test_compiler.Tag(f"tag{index}")
                    ),
                ),
            )
            for index in range(count)
        )
        unions.append(
            Union(
                name=UnionName("AnyVariant"),
                discriminator=FieldName("variant_tag"),
                variants=variants,
            )
        )
    return DifferentialCase(views=tuple(views), unions=tuple(unions), entities=stored)


class UnmetRequirementRead(pydantic.BaseModel, frozen=True):
    """A read that failed on a requirement a row does not meet: the engine
    reports it, and the query code raises `ValueError`."""


type ReadResult = (
    tuple[engine.ViewRow, ...] | tuple[engine.VariantRow, ...] | UnmetRequirementRead
)


def engine_result(
    read: Result[
        tuple[engine.ViewRow, ...] | tuple[engine.VariantRow, ...],
        engine.UnknownViewError
        | engine.UnknownUnionError
        | engine.NonEntityFromModelError
        | DiagnosticsError[engine.ReadDiagnostic],
    ],
) -> ReadResult:
    match read:
        case Ok(rows):
            return rows
        case Err(DiagnosticsError(diagnostics=found)) if all(
            isinstance(
                diagnostic, engine.UnmetRequirement | engine.UnmetReverseRequirement
            )
            for diagnostic in found
        ):
            return UnmetRequirementRead()
        case Err(error):
            pytest.fail(f"expected rows or an unmet requirement, got {error}")


def database_result(read: Callable[[], ReadResult]) -> ReadResult:
    try:
        return read()
    except ValueError:
        return UnmetRequirementRead()


# Each example defines Django models and creates their tables, which can exceed
# the default deadline.
@settings(deadline=None)
@given(differential_cases())
def test_queries_give_the_same_view_results_as_the_engine(
    case: DifferentialCase,
) -> None:
    compiled = compile_all(case.views, case.unions)
    held = view_engine(compiled, case.entities)
    expected = {
        **{view.name: engine_result(held.read(view.name)) for view in compiled.views},
        **{
            union.name: engine_result(held.read_union(union.name))
            for union in compiled.unions
        },
    }

    with Database.of(case.views, case.unions) as django_models:
        Database.put(case.entities)
        read = {
            **{
                view.name: database_result(
                    lambda view=view: Database.read(django_models, view, every_row)
                )
                for view in compiled.views
            },
            **{
                union.name: database_result(
                    lambda union=union: Database.read_union(
                        django_models, union, every_row
                    )
                )
                for union in compiled.unions
            },
        }

    # Catch a strategy that stops drawing resolves, reverses, Filters or Unions.
    relations = [view.relation for view in compiled.views]
    lookups = [lookup for relation in relations for lookup in relation.lookups()]
    reverses = any(isinstance(lookup, ir.ReverseLookup) for lookup in lookups)
    unmet = any(
        isinstance(result, UnmetRequirementRead) for result in expected.values()
    )
    event(f"looks up an Entity: {bool(lookups)}")
    event(f"reverses: {reverses}")
    event(f"filters: {any(relation.filters for relation in relations)}")
    event(f"reads a Union: {bool(compiled.unions)}")
    event(f"unmet requirement: {unmet}")
    assert read == expected


def constrained(config: DjangoModels) -> DjangoModels:
    """`config` with constraints, one comparing with an enum member."""
    role = conditions.field(FieldName("role"))
    nickname = conditions.field(FieldName("nickname"))
    return with_constraints(
        config,
        {
            Employee: (
                Check(
                    name=ConstraintName("employee_lead_has_nickname"),
                    condition=conditions.or_(
                        conditions.ne(role, Role.LEAD),
                        conditions.not_(conditions.is_null(nickname)),
                    ),
                ),
            ),
            Department: (
                Unique(
                    name=ConstraintName("department_unique_name"),
                    fields=(FieldName("company_id"), FieldName("name")),
                    where=conditions.ne(conditions.field(FieldName("name")), Text("")),
                ),
            ),
        },
    )


def with_constraints(
    config: DjangoModels,
    constraints: Mapping[type[pydantic.BaseModel], Sequence[Constraint]],
) -> DjangoModels:
    """`config` with each Entity's `constraints` added to its generated model."""
    return DjangoModels(
        module=config.module,
        queries=config.queries,
        app_label=config.app_label,
        field_types=config.field_types,
        every_model=config.every_model,
        entity_configs={
            **config.entity_configs,
            **{
                entity: config.generated_config_of(entity).model_copy(
                    update={"constraints": tuple(added)}
                )
                for entity, added in constraints.items()
            },
        },
    )


type Existing = tuple[tuple[ModulePath, FileText], ...]
"""The modules of the existing Django models a config names, with their source."""


newest_first_uses = (
    ValueExpression("employees.filter(name='a').newest_first()"),
    ValueExpression("employees.all().newest_first()[:2]"),
)
"""Calls of the QuerySet base's own method, after which `as_<view>()` type
checks."""


type_checked_configs = pytest.mark.parametrize(
    ("config", "existing", "uses"),
    [
        (Modules.config(), (), ()),
        (ExistingCompany.config(), ExistingCompany.sources(), ()),
        (Modules.with_bases(), (), newest_first_uses),
        (PlainBases.config(), PlainBases.sources(), newest_first_uses),
        (DeclaredManager.config(), DeclaredManager.sources(), ()),
    ],
    ids=["generated", "existing", "generic-bases", "plain-bases", "declared-manager"],
)
"""`existing` are modules the config refers to, with their source, and `uses`
what a project calls on `Employee.objects` with the config."""


def type_checked(
    directory: Directory,
    config: DjangoModels,
    existing: Existing,
    uses: tuple[ValueExpression, ...],
) -> None:
    """Writes the generated code of `every_definition()` into `directory`, the
    `existing` Django models, and a project's use of it: `objects` is the
    Entity's Manager, whose QuerySet methods return the Entity's QuerySet, so
    `as_<view>()` type checks after them."""
    views, unions = every_definition()
    Database.files(views, unions, constrained(config)).write(
        LocalFileSystem(), directory
    ).unwrap()
    for module, source in existing:
        _ = (directory.root / module.file_path().root).write_text(
            source.root, encoding="utf-8"
        )
    _ = (directory.root / "frp_generated_use.py").write_text(
        "import typing\n"
        "\n"
        "import frp_generated_django\n"
        "import frp_generated_models\n"
        "import frp_generated_queries\n"
        "\n"
        "\n"
        "def manager() -> frp_generated_queries.EmployeeManager:\n"
        "    return typing.assert_type(\n"
        "        frp_generated_django.Employee.objects,\n"
        "        frp_generated_queries.EmployeeManager,\n"
        "    )\n"
        "\n"
        "\n"
        "def overviews() -> list[frp_generated_models.EmployeeOverview]:\n"
        "    employees = frp_generated_django.Employee.objects\n"
        "    return [\n"
        "        *employees.filter(name='a')"
        ".order_by('pk')[:2].as_employee_overview(),\n"
        "        *employees.exclude(name='a').as_employee_overview(),\n"
        "        *employees.order_by('-pk').as_employee_overview(),\n"
        "        *employees.all().as_employee_overview(),\n"
        "        *employees.as_employee_overview(),\n"
        + "".join(f"        *{use.root}.as_employee_overview(),\n" for use in uses)
        + "    ]\n"
        "\n"
        "\n"
        "def any_employees() -> list[frp_generated_models.AnyEmployee]:\n"
        "    employees = frp_generated_django.Employee.objects\n"
        "    return [*employees.order_by('pk')[:2].as_any_employee()]\n",
        encoding="utf-8",
    )


@type_checked_configs
def test_generated_queries_pass_mypy_strict(
    tmp_path: Path,
    config: DjangoModels,
    existing: Existing,
    uses: tuple[ValueExpression, ...],
) -> None:
    """With django-stubs' mypy plugin, which types Django models from a project's
    settings: here, an app whose models are the generated and existing ones."""
    type_checked(Directory(tmp_path), config, existing, uses)
    label = config.app_label.root
    (tmp_path / label).mkdir()
    _ = (tmp_path / label / "__init__.py").write_text("", encoding="utf-8")
    _ = (tmp_path / label / "models.py").write_text(
        "".join(
            f"import {module.root}\n"
            for module in (Modules.django, *(module for module, _ in existing))
        ),
        encoding="utf-8",
    )
    _ = (tmp_path / "frp_settings.py").write_text(
        f'SECRET_KEY = "test"\nINSTALLED_APPS = [{label!r}]\nUSE_TZ = True\n',
        encoding="utf-8",
    )
    _ = (tmp_path / "mypy.ini").write_text(
        "[mypy]\n"
        "plugins = mypy_django_plugin.main\n"
        "\n"
        "[mypy.plugins.django-stubs]\n"
        "django_settings_module = frp_settings\n",
        encoding="utf-8",
    )
    source_root = Path(__file__).parents[3]

    # Imported modules are checked only as far as the generated code uses them.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--follow-imports=silent",
            "--no-incremental",
            *(
                f"{module}.py"
                for module in (
                    Modules.queries.root,
                    Modules.django.root,
                    "frp_generated_use",
                )
            ),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "MYPYPATH": str(source_root),
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(source_root))),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@type_checked_configs
def test_generated_queries_pass_the_type_checker(
    tmp_path: Path,
    config: DjangoModels,
    existing: Existing,
    uses: tuple[ValueExpression, ...],
) -> None:
    """Where a model base declares the manager, the manager overriding it is an
    attribute, which `@override` cannot decorate, so that error is off."""
    type_checked(Directory(tmp_path), config, existing, uses)
    source_root = Path(__file__).parents[3]
    undecoratable = (
        "\n[errors]\nmissing-override-decorator = false\n"
        if existing == DeclaredManager.sources()
        else ""
    )
    _ = (tmp_path / "pyrefly.toml").write_text(
        'project-includes = ["frp_generated_*.py"]\n'
        f'search-path = ["{tmp_path}", "{source_root}"]\n'
        f'python-interpreter-path = "{sys.executable}"\n'
        f'preset = "all"\n{undecoratable}',
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


# require.


@pytest.mark.parametrize(
    "step",
    [
        steps.require(FieldName("nickname")),
        steps.resolve(
            named(FieldName("mentor_id")), steps.require(FieldName("nickname"))
        ),
        steps.reverse(Badge, steps.require(FieldName("label"))),
    ],
    ids=["field", "nested", "nested-in-reverse"],
)
def test_an_unmet_required_field_fails_the_view_models_validation(
    step: steps.Step,
) -> None:
    view = View(
        name=ViewName("RequiringEmployee"),
        query=steps.from_(Employee, step),
    )
    (compiled,) = compile_all([view]).views

    with Database.of([view]) as django_models:
        Database.put(organisation())
        with pytest.raises(pydantic.ValidationError, match="RequiringEmployee"):
            _ = Database.read(django_models, compiled, every_row)


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (
            steps.pipeline(
                steps.resolve(named(FieldName("mentor_id"))),
                steps.require(FieldName("mentor")),
            ),
            "mentor_id is required, but there is none for Employee",
        ),
        (
            steps.pipeline(steps.reverse(Badge), steps.require(FieldName("badge"))),
            "a Badge by employee_id is required, but there is none for Employee",
        ),
        # The mentor's own mentor may be `None`, so validation alone would not
        # fail where there is no mentor.
        (
            steps.pipeline(
                steps.require(FieldName("mentor_id")),
                steps.resolve(
                    named(FieldName("mentor_id")),
                    steps.pick([FieldName("mentor_id")]),
                    steps.rename(FieldName("mentor_id"), FieldName("grand_mentor_id")),
                ),
                steps.flatten(FieldName("mentor")),
            ),
            "mentor_id is required, but there is none for Employee",
        ),
        (
            steps.resolve(
                named(FieldName("department_id")),
                steps.reverse(
                    Employee,
                    steps.pick([FieldName("mentor_id")]),
                    steps.resolve(
                        named(FieldName("mentor_id")),
                        steps.pick([FieldName("name")]),
                    ),
                    steps.require(FieldName("mentor")),
                ),
            ),
            "mentor_id is required, but there is none for Employee",
        ),
        (
            steps.resolve(
                named(FieldName("department_id")),
                steps.reverse(
                    Employee,
                    steps.pick([FieldName("mentor_id")]),
                    steps.resolve(named(FieldName("mentor_id")), steps.pick([])),
                    steps.require(FieldName("mentor")),
                ),
            ),
            "mentor_id is required, but there is none for Employee",
        ),
    ],
    ids=["reference", "reverse", "lifted", "in-collection", "unread-in-collection"],
)
def test_an_unmet_required_entity_raises(step: steps.Step, message: str) -> None:
    """As the mapping functions do, through the module's `_required`."""
    view = View(
        name=ViewName("RequiringEmployee"),
        query=steps.from_(Employee, step),
    )
    (compiled,) = compile_all([view]).views

    with Database.of([view]) as django_models:
        Database.put(organisation())
        with pytest.raises(ValueError, match=message):
            _ = Database.read(django_models, compiled, every_row)


def test_a_met_requirement_reads_like_the_engine() -> None:
    view = View(
        name=ViewName("NicknamedEmployee"),
        query=steps.from_(Employee, steps.require(FieldName("nickname"))),
    )
    entities = tuple(
        entity.model_copy(update={"nickname": Text("nick")})
        if isinstance(entity, Employee)
        else entity
        for entity in organisation()
    )
    (compiled,) = compile_all([view]).views

    with Database.of([view]) as django_models:
        Database.put(entities)
        rows = Database.read(django_models, compiled, every_row)

    assert rows == engine_rows([compiled], entities)[view.name]


class ConstraintCase(pydantic.BaseModel, frozen=True):
    from_model: type[pydantic.BaseModel]
    condition: ir.Condition
    entities: tuple[pydantic.BaseModel, ...]


@st.composite
def constraint_cases(draw: st.DrawFn) -> ConstraintCase:
    stored = draw(entities())
    from_model = draw(st.sampled_from([Company, Department, Employee, Badge]))
    return ConstraintCase(
        from_model=from_model,
        condition=draw(field_conditions(from_model, stored, Depth(2))),
        entities=stored,
    )


# Each example defines Django models and creates their tables, which can exceed
# the default deadline.
@settings(deadline=None)
@given(constraint_cases())
def test_the_database_refuses_exactly_the_rows_a_check_is_false_for(
    case: ConstraintCase,
) -> None:
    """Differential: a row violates a Check exactly where `where` of its
    Condition does not keep it in the InMemoryViewEngine."""
    view = View(
        name=ViewName("Kept"),
        query=steps.from_(case.from_model, steps.where(case.condition)),
    )
    compiled = compile_all([view])
    kept = {
        row.root[FieldName("id")]
        for row in view_engine(compiled, case.entities).read(view.name).unwrap()
    }
    check = Check(name=ConstraintName("checked"), condition=case.condition)
    config = Modules.config({case.from_model: GeneratedModel(constraints=(check,))})

    with Database.of([view], (), config):
        refused = Database.rejected(case.entities, case.from_model, config)

    rows = {
        Database.primary_key(entity)
        for entity in case.entities
        if type(entity) is case.from_model
    }
    event(f"refuses some rows: {bool(refused)}")
    event(f"stores some rows: {refused != rows}")
    assert refused == rows - kept


def test_the_database_refuses_a_second_row_a_unique_keeps() -> None:
    """Rows with a NULL in a Unique's fields, or that its Condition does not keep,
    are not compared."""
    company = Company(id=CompanyId(1), name=Text("a"), parent=None)
    named = [
        Department(id=DepartmentId(index), name=Text(name), company_id=company.id)
        for index, name in enumerate(["a", "a", "b", "dup", "dup"])
    ]
    unique = Unique(
        name=ConstraintName("department_unique_name"),
        fields=(FieldName("company_id"), FieldName("name")),
        where=conditions.ne(conditions.field(FieldName("name")), Text("dup")),
    )
    config = Modules.config({Department: GeneratedModel(constraints=(unique,))})

    # A View, so that there is a module of View models.
    view = View(
        name=ViewName("Departments"),
        query=steps.from_(Department),
    )

    with Database.of([view], (), config):
        refused = Database.rejected([company, *named], Department, config)

    assert refused == {DepartmentId(1)}
