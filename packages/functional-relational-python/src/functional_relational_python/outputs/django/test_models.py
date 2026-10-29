import datetime
import enum
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import types
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import annotated_types
import django.db.models
import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.compiler.diagnostics import UnknownField
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
    RuntimeAnnotation,
)
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    OneToOne,
    OneToOneMarker,
    ParentLink,
)
from functional_relational_python.outputs.django.conditions import DjangoColumn
from functional_relational_python.outputs.django.models import (
    AppLabel,
    AutoIncrementOnNonPrimaryKey,
    AutoNow,
    AutoNowAdd,
    BareDecorator,
    CalledDecorator,
    CatalogModelOptions,
    ChildNotKeyedByParentLink,
    ConstraintNameClash,
    ConstraintsForNonEntity,
    ConstraintsOnExistingModel,
    DatabaseDefault,
    DbTable,
    Default,
    DefaultDjangoModelName,
    DefaultForNonField,
    DefaultForReferencePrimaryKey,
    DjangoDiagnostic,
    DjangoField,
    DjangoFieldNameClash,
    DjangoIndex,
    DjangoModelName,
    DjangoModelNameCaseClash,
    DjangoModelNameClash,
    DjangoModels,
    DjangoModelsRenderer,
    EntityModelOptions,
    ExistingDjangoModel,
    ExistingModelInGeneratedModule,
    ExistingModelOfNonEntity,
    FieldDefault,
    FieldTypes,
    ForeignKeyConstraint,
    IndexField,
    IndexName,
    InvalidDjangoFieldName,
    InvalidReverseAccessorName,
    MixedDjangoChoiceTypes,
    ModelGenericBase,
    ModelOptionsForExistingModel,
    ModelOptionsForNonEntity,
    NoDjangoFieldRule,
    OnDelete,
    OnDeleteForNonReference,
    OnDeleteOnExistingModel,
    OptionalPrimaryKeyReference,
    OrderingTerm,
    OwnPrimaryKey,
    PrimaryKeyOfNonOneToOne,
    ReferenceKey,
    ReferencePrimaryKey,
    RelatedName,
    RelatedNameForNonReference,
    ReverseAccessorClash,
    ReverseAccessorOfNonOneToOne,
    SetNullOnRequiredReference,
    SharedDbTable,
    SharedIndexName,
    UndeclaredReverseAccessor,
    UnimportableConstraintEnum,
    UnimportableDjangoChoices,
    UnwritableDjangoChoices,
)
from functional_relational_python.outputs.django.queries import DjangoQueriesRenderer
from functional_relational_python.outputs.python import test_symbols
from functional_relational_python.outputs.python.plan import PlacedModels
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    FirstParty,
    ModulePath,
    Symbol,
    ValueExpression,
)
from functional_relational_python.pipeline import conditions, ir
from functional_relational_python.pipeline.constraints import (
    Check,
    Constraint,
    ConstraintName,
    EveryRow,
    Unique,
)
from functional_relational_python.support.filesystem import (
    Directory,
    FileText,
    LocalFileSystem,
)
from functional_relational_python.support.generated_files import GeneratedFiles
from functional_relational_python.support.identifier import Identifier


class Text(pydantic.RootModel[str], frozen=True):
    pass


class Email(pydantic.RootModel[Text], frozen=True):
    """Wraps another wrapper, so rules apply through every layer."""


class Integer(pydantic.RootModel[int], frozen=True):
    pass


class Boolean(pydantic.RootModel[bool], frozen=True):
    pass


class Real(pydantic.RootModel[float], frozen=True):
    pass


class Raw(pydantic.RootModel[bytes], frozen=True):
    pass


class Moment(pydantic.RootModel[datetime.datetime], frozen=True):
    pass


class Day(pydantic.RootModel[datetime.date], frozen=True):
    pass


class Clock(pydantic.RootModel[datetime.time], frozen=True):
    pass


class Duration(pydantic.RootModel[datetime.timedelta], frozen=True):
    pass


class Uuid(pydantic.RootModel[uuid.UUID], frozen=True):
    pass


class MaybeText(pydantic.RootModel[str | None], frozen=True):
    pass


class CompanyId(Id[int]):
    pass


class Company(pydantic.BaseModel):
    id: CompanyId
    name: Text
    parent: CompanyId | None


class DepartmentId(Id[uuid.UUID]):
    pass


class Department(pydantic.BaseModel):
    id: DepartmentId
    company_id: CompanyId
    founded: Day


class ThingId(Id[int]):
    pass


class Thing(pydantic.BaseModel):
    """An Entity with a field wrapping each type there is a built-in rule for."""

    id: ThingId
    text: Text
    integer: Integer
    boolean: Boolean
    real: Real
    raw: Raw
    moment: Moment
    day: Day
    clock: Clock
    duration: Duration
    uuid: Uuid
    wrapped_twice: Email
    optional: Text | None
    wraps_optional: MaybeText
    tags: tuple[Text, ...]


def config(
    field_types: Mapping[RuntimeAnnotation, DjangoField],
    on_delete: Mapping[ReferenceKey, OnDelete],
    existing: Mapping[
        type[pydantic.BaseModel], ExistingDjangoModel
    ] = types.MappingProxyType({}),
) -> DjangoModels:
    return DjangoModels(
        module=ModulePath("app.models"),
        app_label=AppLabel("app"),
        field_types=field_types,
        on_delete=on_delete,
        queries=ModulePath("app.queries"),
        constraints={},
        existing=existing,
        model_options=CatalogModelOptions.default(),
        entity_model_options={},
    )


def existing_company(on_delete: Mapping[ReferenceKey, OnDelete]) -> DjangoModels:
    """`Company` has an existing Django model, `Company` of `crm.models`."""
    return config({}, on_delete, {Company: ExistingDjangoModel.fake()})


def generated_by(models: DjangoModels) -> FirstParty:
    """What `Catalog.generate` has first party: the Django modules' packages."""
    return FirstParty.fake().with_generated([models.module, models.queries])


def render(
    entities: Iterable[type[pydantic.BaseModel]], models: DjangoModels
) -> FileText:
    match DjangoModelsRenderer.render(
        EntityGraph.build(entities, IdBase.fake(), OneToOneMarker.fake()).unwrap(),
        models,
        first_party=generated_by(models),
    ):
        case Ok(files):
            (file,) = files.files
            return file.text
        case Err(error):
            pytest.fail(f"expected generated files, got {error}")


def diagnostics(
    entities: Iterable[type[pydantic.BaseModel]], models: DjangoModels
) -> tuple[DjangoDiagnostic, ...]:
    match DjangoModelsRenderer.render(
        EntityGraph.build(entities, IdBase.fake(), OneToOneMarker.fake()).unwrap(),
        models,
        first_party=generated_by(models),
    ):
        case Ok(files):
            pytest.fail(f"expected diagnostics, got {files}")
        case Err(error):
            return error.diagnostics


def field_line(
    entity: type[pydantic.BaseModel], field: FieldName, models: DjangoModels
) -> FileText:
    """The statement declaring `field` of `entity`, dedented: a line, or the
    lines it is split over."""
    (statement,) = re.findall(
        rf"^    {field.root} = .*?\n(?=    \w|\n)",
        render([entity], models).root,
        flags=re.MULTILINE | re.DOTALL,
    )
    return FileText(textwrap.dedent(statement).removesuffix("\n"))


def all_rules_for_tags() -> DjangoModels:
    """`Thing.tags` has no built-in rule, so every test needs one."""
    return config({tuple[Text, ...]: DjangoField.fake()}, {})


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("text", "models.TextField()"),
        ("integer", "models.BigIntegerField()"),
        ("boolean", "models.BooleanField()"),
        ("real", "models.FloatField()"),
        ("raw", "models.BinaryField()"),
        ("moment", "models.DateTimeField()"),
        ("day", "models.DateField()"),
        ("clock", "models.TimeField()"),
        ("duration", "models.DurationField()"),
        ("uuid", "models.UUIDField()"),
        ("wrapped_twice", "models.TextField()"),
        ("optional", "models.TextField(null=True)"),
        ("wraps_optional", "models.TextField(null=True)"),
    ],
)
def test_builtin_rule_for_the_wrapped_type(field: str, expected: str) -> None:
    assert field_line(Thing, FieldName(field), all_rules_for_tags()) == FileText(
        f"{field} = {expected}"
    )


def char_field() -> DjangoField:
    return DjangoField.builtin(
        Identifier("CharField"), {Identifier("max_length"): ValueExpression("254")}
    )


@pytest.mark.parametrize(
    ("field", "overridden"),
    [
        # The rule for the field's own type wins over the rule for what it wraps.
        ("wrapped_twice", Email),
        # The rule for a wrapped type applies to every wrapper of it.
        ("wrapped_twice", str),
        ("wrapped_twice", Text),
        # A type without a built-in rule gets one.
        ("tags", tuple[Text, ...]),
    ],
)
def test_the_catalog_overrides_the_rule_for_a_type(
    field: str, overridden: RuntimeAnnotation
) -> None:
    models = config(
        {tuple[Text, ...]: DjangoField.fake(), overridden: char_field()}, {}
    )

    assert field_line(Thing, FieldName(field), models) == FileText(
        f"{field} = models.CharField(max_length=254)"
    )


def test_an_overridden_field_class_outside_django_is_imported() -> None:
    money = DjangoField(
        field_class=Symbol(
            module=ModulePath("app.fields"), qualname=(Identifier("MoneyField"),)
        ),
        arguments={},
    )

    text = render([Thing], config({tuple[Text, ...]: money}, {})).root

    assert (
        "\n\nfrom django.db import models\n\n"
        "from app.fields import MoneyField\n"
        "from app.queries import ThingManager\n\n\n"
    ) in text
    assert "    tags = MoneyField()\n" in text


def test_an_optional_overridden_field_is_nullable() -> None:
    models = config({tuple[Text, ...]: DjangoField.fake(), Text: char_field()}, {})

    assert field_line(Thing, FieldName("optional"), models) == FileText(
        "optional = models.CharField(max_length=254, null=True)"
    )


def test_every_type_without_a_rule_is_a_diagnostic() -> None:
    class Untyped(pydantic.BaseModel):
        id: ThingId
        tags: tuple[Text, ...]
        labels: frozenset[Text]

    assert set(diagnostics([Untyped], config({}, {}))) == {
        NoDjangoFieldRule(
            entity=Untyped,
            field=FieldName("tags"),
            annotation=AnnotationText.of(tuple[Text, ...]),
            unruled=AnnotationText.of(tuple[Text, ...]),
        ),
        NoDjangoFieldRule(
            entity=Untyped,
            field=FieldName("labels"),
            annotation=AnnotationText.of(frozenset[Text]),
            unruled=AnnotationText.of(frozenset[Text]),
        ),
    }


class Name(pydantic.RootModel[Annotated[str, pydantic.Field(min_length=1)]]):
    pass


class Budget(pydantic.RootModel[pydantic.PositiveInt]):
    pass


PastMoment = Annotated[datetime.datetime, pydantic.AwareDatetime, pydantic.PastDatetime]


type AliasedMoment = PastMoment


class Shade(enum.StrEnum):
    LIGHT = "light"
    DARK = "dark"


class Size(enum.IntEnum):
    SMALL = 1
    LARGE = 2


class Channel(django.db.models.TextChoices):
    META = "meta", "Meta"
    GOOGLE = "google", "Google"


class Status(django.db.models.TextChoices):
    """With `__empty__`, its choices hold `None`."""

    ACTIVE = "active", "Active"
    __empty__ = "Unknown"


class Permission(enum.IntFlag):
    READ = 1
    WRITE = 2


class Count(pydantic.RootModel[Annotated[int, {"unit": "count"}]]):
    """Its root's metadata, a dict, cannot be hashed."""


class RuledId(Id[int]):
    pass


class Ruled(pydantic.BaseModel):
    """An Entity with a field of each kind of type a built-in rule looks into."""

    id: RuledId
    name: Name
    budget: Budget
    moment: PastMoment
    maybe_moment: AliasedMoment | None
    unhashable: Count
    unhashables: list[Annotated[Integer, {"unit": "count"}]]
    shade: Shade
    size: Size
    channel: Channel
    status: Status
    permissions: Permission
    kind: Literal["area", "money"]
    maybe_kind: Literal["area", None]
    member: Literal[Shade.DARK]
    shades: list[Shade]
    maybe_texts: list[Text | None] | None


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        # `Annotated` metadata is left out, even when it cannot be hashed.
        ("name", "models.TextField()"),
        ("budget", "models.BigIntegerField()"),
        ("moment", "models.DateTimeField()"),
        ("maybe_moment", "models.DateTimeField(null=True)"),
        ("unhashable", "models.BigIntegerField()"),
        ("unhashables", "ArrayField(base_field=models.BigIntegerField())"),
        # An enum is the field of its values' type, with its values as choices.
        (
            "shade",
            'models.TextField(choices=[("light", "light"), ("dark", "dark")])',
        ),
        ("size", "models.BigIntegerField(choices=[(1, 1), (2, 2)])"),
        # A Django Choices enum's own choices keep its labels.
        ("channel", "models.TextField(choices=Channel.choices)"),
        # Its choices hold `None`, so its column may be NULL.
        ("status", "models.TextField(choices=Status.choices, null=True)"),
        # A Flag's members combine, so they are no choices.
        ("permissions", "models.BigIntegerField()"),
        # A Literal is the field of its values' type, with them as choices.
        (
            "kind",
            'models.TextField(choices=[("area", "area"), ("money", "money")])',
        ),
        (
            "maybe_kind",
            'models.TextField(choices=[("area", "area")], null=True)',
        ),
        ("member", 'models.TextField(choices=[("dark", "dark")])'),
        # A list is an array of the field of its items.
        (
            "shades",
            # Split as ruff format splits it, since it does not fit on a line.
            (
                "ArrayField(\n"
                '    base_field=models.TextField(choices=[("light", "light"), ("dark", "dark")])\n'
                ")"
            ),
        ),
        (
            "maybe_texts",
            "ArrayField(base_field=models.TextField(null=True), null=True)",
        ),
    ],
)
def test_builtin_rule_for_a_type_it_looks_into(field: str, expected: str) -> None:
    assert field_line(Ruled, FieldName(field), config({}, {})) == FileText(
        f"{field} = {expected}"
    )


def test_a_django_choices_enum_and_the_array_field_are_imported() -> None:
    text = render([Ruled], config({}, {})).root

    assert "\nfrom django.contrib.postgres.fields import ArrayField\n" in text
    assert f"\nfrom {Channel.__module__} import Channel, Status\n" in text


@pytest.mark.parametrize(
    ("field", "overridden", "expected"),
    [
        # The rule for an annotated type wins over the rule for what it annotates.
        ("budget", pydantic.PositiveInt, "models.CharField(max_length=254)"),
        # An enum takes the rule for its values' type.
        (
            "channel",
            str,
            "models.CharField(max_length=254, choices=Channel.choices)",
        ),
    ],
)
def test_the_catalog_overrides_the_rule_inside_a_type(
    field: str, overridden: RuntimeAnnotation, expected: str
) -> None:
    models = config({overridden: char_field()}, {})

    assert field_line(Ruled, FieldName(field), models) == FileText(
        f"{field} = {expected}"
    )


def test_override_arguments_import_the_symbols_and_fields_they_name() -> None:
    colour = Symbol(
        module=ModulePath("app.choices"),
        qualname=(Identifier("Colour"), Identifier("choices")),
    )
    fields = ModulePath("app.fields")
    array = DjangoField(
        field_class=Symbol(module=fields, qualname=(Identifier("ListField"),)),
        arguments={
            Identifier("base_field"): DjangoField(
                field_class=Symbol(
                    module=fields, qualname=(Identifier("ColourField"),)
                ),
                arguments={Identifier("choices"): colour},
            ),
            Identifier("size"): ValueExpression("3"),
        },
    )

    text = render([Thing], config({tuple[Text, ...]: array}, {})).root

    assert (
        "\n\nfrom app.choices import Colour\n"
        "from app.fields import ColourField, ListField\n"
    ) in text
    assert (
        "    tags = ListField(base_field=ColourField(choices=Colour.choices), size=3)\n"
    ) in text


class Mixed(enum.Enum):
    TEXT = "text"
    NUMBER = 1


def test_values_of_several_types_are_a_diagnostic() -> None:
    class HasMixed(pydantic.BaseModel):
        id: ThingId
        enum_values: Mixed
        literal_values: Literal["a", 1]

    assert set(diagnostics([HasMixed], config({}, {}))) == {
        MixedDjangoChoiceTypes(
            entity=HasMixed,
            field=FieldName(field),
            values_of=AnnotationText.of(values_of),
            value_types=(int, str),
        )
        for field, values_of in (
            ("enum_values", Mixed),
            ("literal_values", Literal["a", 1]),
        )
    }


def test_a_list_of_a_type_without_a_rule_is_a_diagnostic() -> None:
    class Unruled(pydantic.BaseModel):
        id: ThingId
        items: list[frozenset[Text]]

    assert diagnostics([Unruled], config({}, {})) == (
        NoDjangoFieldRule(
            entity=Unruled,
            field=FieldName("items"),
            annotation=AnnotationText.of(list[frozenset[Text]]),
            unruled=AnnotationText.of(frozenset[Text]),
        ),
    )


class Holiday(enum.Enum):
    NEW_YEAR = datetime.date(2026, 1, 1)


class Ratio(enum.Enum):
    UNBOUNDED = math.inf


@pytest.mark.parametrize(
    ("values_of", "value"),
    [(Holiday, datetime.date(2026, 1, 1)), (Ratio, math.inf)],
)
def test_choices_generated_code_cannot_write_are_a_diagnostic(
    values_of: type[enum.Enum], value: object
) -> None:
    unwritable = pydantic.create_model(
        "HasUnwritable", id=(ThingId, ...), value=(values_of, ...)
    )

    assert diagnostics([unwritable], config({}, {})) == (
        UnwritableDjangoChoices(
            entity=unwritable,
            field=FieldName("value"),
            values_of=AnnotationText.of(values_of),
            values=(value,),
        ),
    )


class Tree(pydantic.RootModel[list["Tree"]]):
    pass


def test_a_recursive_type_has_no_rule() -> None:
    class Forest(pydantic.BaseModel):
        id: ThingId
        tree: Tree

    assert diagnostics([Forest], config({}, {})) == (
        NoDjangoFieldRule(
            entity=Forest,
            field=FieldName("tree"),
            annotation=AnnotationText.of(Tree),
            unruled=AnnotationText.of(Tree),
        ),
    )


def test_an_auto_increment_override_nested_in_a_field_is_a_diagnostic() -> None:
    class Counts(pydantic.BaseModel):
        id: CompanyId
        counts: list[Integer]

    auto = DjangoField.builtin(Identifier("BigAutoField"), {})

    assert diagnostics([Counts], config({int: auto}, {})) == (
        AutoIncrementOnNonPrimaryKey(entity=Counts, field=FieldName("counts")),
    )


def test_a_django_choices_enum_that_cannot_be_imported_is_a_diagnostic() -> None:
    class Local(django.db.models.TextChoices):
        ONE = "one", "One"

    class HasLocal(pydantic.BaseModel):
        id: ThingId
        local: Local

    assert diagnostics([HasLocal], config({}, {})) == (
        UnimportableDjangoChoices(
            entity=HasLocal, field=FieldName("local"), enum_class=Local
        ),
    )


@pytest.mark.parametrize(
    ("on_delete", "expected"),
    [
        ({}, 'company = models.ForeignKey("Company", on_delete=models.PROTECT, related_name="+")'),
        (
            {ReferenceKey(entity=Department, field=FieldName("company_id")): OnDelete.CASCADE},
            'company = models.ForeignKey("Company", on_delete=models.CASCADE, related_name="+")',
        ),
    ],
)  # fmt: skip
def test_a_reference_is_a_foreign_key(
    on_delete: Mapping[ReferenceKey, OnDelete], expected: str
) -> None:
    text = render([Company, Department], config({}, on_delete)).root

    assert f"    {expected}\n" in text


def test_an_optional_reference_is_a_nullable_foreign_key() -> None:
    text = render([Company], config({}, {})).root

    assert (
        "    parent = models.ForeignKey(\n"
        '        "Company", on_delete=models.PROTECT, related_name="+", null=True\n'
        "    )\n"
    ) in text


class CompanyDetailsId(Id[int]):
    pass


class CompanyDetails(pydantic.BaseModel):
    """One-to-one References, required and optional."""

    id: CompanyDetailsId
    company_id: Annotated[CompanyId, OneToOne()]
    successor_id: Annotated[CompanyId | None, OneToOne()]


def test_a_one_to_one_reference_is_a_one_to_one_field_with_a_reverse_accessor() -> None:
    text = render([Company, CompanyDetails], config({}, {})).root

    assert (
        textwrap.indent(
            textwrap.dedent(
                """\
                company = models.OneToOneField(
                    "Company", on_delete=models.PROTECT, related_name="company_details_by_company"
                )
                successor = models.OneToOneField(
                    "Company",
                    on_delete=models.PROTECT,
                    related_name="company_details_by_successor",
                    null=True,
                )
                """
            ),
            "    ",
        )
        in text
    )


def keyed_by(reference: FieldName) -> EntityModelOptions:
    """The Entity's model has `reference` as its primary key."""
    return EntityModelOptions.inherited().model_copy(
        update={"primary_key": ReferencePrimaryKey(reference=reference)}
    )


def named(name: DjangoModelName, options: EntityModelOptions) -> EntityModelOptions:
    """`options`, with the model named `name`."""
    return options.model_copy(update={"model_name": name})


def prefixed(
    *entities: type[pydantic.BaseModel],
) -> Mapping[type[pydantic.BaseModel], EntityModelOptions]:
    """Each Entity's model named `Django<Entity>`."""
    return {
        entity: named(
            DjangoModelName(f"Django{entity.__name__}"),
            EntityModelOptions.inherited(),
        )
        for entity in entities
    }


def test_a_model_named_by_its_options_is_referred_to_by_that_name() -> None:
    """References to it, reverse accessors named after it, and its table."""
    models = config({}, {}).model_copy(
        update={"entity_model_options": prefixed(Company, CompanyDetails)}
    )

    text = render([Company, CompanyDetails], models).root

    assert "class Company(" not in text
    assert "\nclass DjangoCompany(models.Model):\n" in text
    assert (
        "\nclass DjangoCompanyDetails(models.Model):\n"
        "    objects: ClassVar[CompanyDetailsManager] = CompanyDetailsManager()\n"
        "    id = models.BigIntegerField(primary_key=True)\n"
        "    company = models.OneToOneField(\n"
        '        "DjangoCompany",\n'
        "        on_delete=models.PROTECT,\n"
        '        related_name="django_company_details_by_company",\n'
        "    )\n"
    ) in text
    assert models.model_of(Company) == Symbol(
        module=models.module, qualname=(Identifier("DjangoCompany"),)
    )
    assert models.table_of(Company) == DbTable("app_djangocompany")
    assert models.reverse_accessor_of(
        CompanyDetails, FieldName("company_id")
    ) == FieldName("django_company_details_by_company")


def test_models_named_alike_by_their_options_are_a_diagnostic() -> None:
    models = config({}, {}).model_copy(
        update={
            "entity_model_options": {
                Company: named(DjangoModelName("Firm"), EntityModelOptions.inherited()),
                Department: named(
                    DjangoModelName("Firm"), EntityModelOptions.inherited()
                ),
            }
        }
    )

    assert diagnostics([Company, Department], models) == (
        DjangoModelNameClash(name=Identifier("Firm")),
    )


def test_models_named_alike_but_for_case_are_a_diagnostic() -> None:
    """Django tells models apart by their lowercased names."""
    models = config({}, {}).model_copy(
        update={
            "entity_model_options": {
                Department: named(
                    DjangoModelName(Company.__name__.upper()),
                    EntityModelOptions.inherited(),
                )
            }
        }
    )

    assert diagnostics([Company, Department], models) == (
        DjangoModelNameCaseClash(
            names=(
                Identifier(Company.__name__.upper()),
                Identifier(Company.__name__),
            )
        ),
    )


def test_a_one_to_one_reference_can_be_the_primary_key_in_place_of_the_id() -> None:
    models = config({}, {}).model_copy(
        update={
            "entity_model_options": {CompanyDetails: keyed_by(FieldName("company_id"))}
        }
    )

    text = render([Company, CompanyDetails], models).root

    assert (
        "class CompanyDetails(models.Model):\n"
        "    objects: ClassVar[CompanyDetailsManager] = CompanyDetailsManager()\n"
        "    company = models.OneToOneField(\n"
        '        "Company",\n'
        "        on_delete=models.PROTECT,\n"
        '        related_name="company_details_by_company",\n'
        "        primary_key=True,\n"
        "    )\n"
        "    successor = "
    ) in text


@pytest.mark.parametrize(
    ("entity", "options", "expected"),
    [
        (
            Department,
            keyed_by(FieldName("company_id")),
            PrimaryKeyOfNonOneToOne(entity=Department, field=FieldName("company_id")),
        ),
        (
            Department,
            keyed_by(FieldName("founded")),
            PrimaryKeyOfNonOneToOne(entity=Department, field=FieldName("founded")),
        ),
        (
            CompanyDetails,
            keyed_by(FieldName("successor_id")),
            OptionalPrimaryKeyReference(
                entity=CompanyDetails, field=FieldName("successor_id")
            ),
        ),
        (
            CompanyDetails,
            keyed_by(FieldName("company_id")).model_copy(
                update={"defaults": {FieldName("id"): AutoNow()}}
            ),
            DefaultForReferencePrimaryKey(
                entity=CompanyDetails, reference=FieldName("company_id")
            ),
        ),
    ],
    ids=["many-to-one", "not a reference", "optional", "default for the id"],
)
def test_a_primary_key_reference_django_cannot_key_by_is_a_diagnostic(
    entity: type[pydantic.BaseModel],
    options: EntityModelOptions,
    expected: DjangoDiagnostic,
) -> None:
    models = config({}, {}).model_copy(
        update={"entity_model_options": {entity: options}}
    )

    assert diagnostics([Company, Department, CompanyDetails], models) == (expected,)


class UnruledId(Id[complex]):
    """No Django field rule holds a `complex`."""


class UnruledDetails(pydantic.BaseModel):
    id: UnruledId
    company_id: Annotated[CompanyId, OneToOne()]


def test_the_id_of_a_reference_keyed_model_is_held_by_the_reference_column() -> None:
    """The `id` needs no Django field rule, and its column is the Reference's,
    which is never NULL."""
    models = config({}, {}).model_copy(
        update={
            "entity_model_options": {UnruledDetails: keyed_by(FieldName("company_id"))}
        }
    )
    graph = EntityGraph.build(
        [Company, UnruledDetails], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()

    _ = DjangoModelsRenderer.render(
        graph, models, first_party=generated_by(models)
    ).unwrap()
    assert DjangoModelsRenderer.columns(graph, models, FieldTypes(overrides={})).of(
        UnruledDetails, FieldName("id")
    ) == DjangoColumn(
        attribute=FieldName("company_id"), optionality=Optionality.REQUIRED
    )


def test_a_catalog_wide_default_for_the_id_of_a_reference_keyed_model_is_left_out() -> (
    None
):
    """The `id` has no column of its own, so the default has no field to go on."""
    models = config({}, {}).model_copy(
        update={
            "model_options": CatalogModelOptions.default().model_copy(
                update={"defaults": {FieldName("id"): Default(value=uuid4())}}
            ),
            "entity_model_options": {CompanyDetails: keyed_by(FieldName("company_id"))},
        }
    )

    text = render([Company, CompanyDetails], models).root

    assert "class CompanyDetails(models.Model):\n    objects" in text
    assert text.count("default=uuid4") == 1


class HolderId(Id[int]):
    pass


class Holder(pydantic.BaseModel):
    """Its Django model has the field `holder_details_by_holder`, the reverse
    accessor of `HolderDetails.holder_id`."""

    id: HolderId
    holder_details_by_holder: Text


class HolderDetailsId(Id[int]):
    pass


class HolderDetails(pydantic.BaseModel):
    id: HolderDetailsId
    holder_id: Annotated[HolderId, OneToOne()]


def test_a_reverse_accessor_named_like_a_field_is_a_diagnostic() -> None:
    assert diagnostics([Holder, HolderDetails], config({}, {})) == (
        ReverseAccessorClash(
            entity=Holder,
            name=FieldName("holder_details_by_holder"),
            references=(
                ReferenceKey(entity=HolderDetails, field=FieldName("holder_id")),
            ),
        ),
    )


@pytest.mark.parametrize(
    "models",
    [config({}, {}), existing_company({})],
    ids=["generated", "existing"],
)
def test_reverse_accessors_named_alike_are_a_diagnostic(models: DjangoModels) -> None:
    """`Owned.team_by_owner_id` and `OwnedByTeam.owner_id` both have the reverse
    accessor `owned_by_team_by_owner`, also on an existing Django model."""

    class OwnedId(Id[int]):
        pass

    class Owned(pydantic.BaseModel):
        id: OwnedId
        team_by_owner_id: Annotated[CompanyId, OneToOne()]

    class OwnedByTeamId(Id[int]):
        pass

    class OwnedByTeam(pydantic.BaseModel):
        id: OwnedByTeamId
        owner_id: Annotated[CompanyId, OneToOne()]

    assert diagnostics([Company, Owned, OwnedByTeam], models) == (
        ReverseAccessorClash(
            entity=Company,
            name=FieldName("owned_by_team_by_owner"),
            references=(
                ReferenceKey(entity=Owned, field=FieldName("team_by_owner_id")),
                ReferenceKey(entity=OwnedByTeam, field=FieldName("owner_id")),
            ),
        ),
    )


def test_a_reverse_accessor_django_rejects_is_a_diagnostic() -> None:
    """An Entity named with `__` gives a reverse accessor holding `__`."""

    class DoubleId(Id[int]):
        pass

    double = pydantic.create_model(
        "Holder__Details",
        id=(DoubleId, ...),
        holder_id=(Annotated[HolderId, OneToOne()], ...),
    )

    assert diagnostics([Holder, double], config({}, {})) == (
        InvalidReverseAccessorName(
            name=FieldName("holder__details_by_holder"),
            reference=ReferenceKey(entity=double, field=FieldName("holder_id")),
        ),
    )


def test_the_invalid_reverse_accessor_message_names_the_reference() -> None:
    diagnostic = InvalidReverseAccessorName(
        name=FieldName("holder__details_by_holder"),
        reference=ReferenceKey(entity=HolderDetails, field=FieldName("holder_id")),
    )

    assert diagnostic.describe().root == (
        "HolderDetails.holder_id: its reverse accessor 'holder__details_by_holder' "
        "cannot be a Django related name, since Django rejects one ending in '_' "
        "or holding '__'. Rename the Entity or the field."
    )


def test_the_reverse_accessor_clash_message_names_the_references() -> None:
    diagnostic = ReverseAccessorClash(
        entity=Holder,
        name=FieldName("holder_details_by_holder"),
        references=(ReferenceKey(entity=HolderDetails, field=FieldName("holder_id")),),
    )

    assert diagnostic.describe().root == (
        "Holder: 'holder_details_by_holder', the reverse accessor of "
        "HolderDetails.holder_id, is bound more than once on its Django model."
    )


@pytest.mark.parametrize(
    ("field_types", "expected"),
    [
        ({}, "id = models.BigIntegerField(primary_key=True)"),
        (
            {CompanyId: DjangoField.builtin(Identifier("BigAutoField"), {})},
            "id = models.BigAutoField(primary_key=True)",
        ),
    ],
)
def test_the_primary_key_is_a_field_of_the_wrapped_type(
    field_types: Mapping[RuntimeAnnotation, DjangoField], expected: str
) -> None:
    text = render([Company], config(field_types, {})).root

    assert f"    {expected}\n" in text


def test_golden_django_models() -> None:
    text = render(
        [Department, Company],
        config(
            {CompanyId: DjangoField.builtin(Identifier("BigAutoField"), {})},
            {
                ReferenceKey(
                    entity=Company, field=FieldName("parent")
                ): OnDelete.SET_NULL
            },
        ),
    )

    assert text == FileText(
        textwrap.dedent(
            """\
        # Generated by functional-relational-python. Do not edit.
        # Regenerate with `frp generate`; `frp check` fails when this is stale.
        # Migrations are yours to make, e.g. with `makemigrations`.

        from typing import ClassVar

        from django.db import models

        from app.queries import CompanyManager, DepartmentManager


        class Company(models.Model):
            objects: ClassVar[CompanyManager] = CompanyManager()
            id = models.BigAutoField(primary_key=True)
            name = models.TextField()
            parent = models.ForeignKey(
                "Company", on_delete=models.SET_NULL, related_name="+", null=True
            )

            class Meta:
                app_label = "app"


        class Department(models.Model):
            objects: ClassVar[DepartmentManager] = DepartmentManager()
            id = models.UUIDField(primary_key=True)
            company = models.ForeignKey("Company", on_delete=models.PROTECT, related_name="+")
            founded = models.DateField()

            class Meta:
                app_label = "app"
        """
        )
    )


def test_the_module_is_written_where_the_catalog_names() -> None:
    graph = EntityGraph.build([Company], IdBase.fake(), OneToOneMarker.fake()).unwrap()
    models = config({}, {})

    files = DjangoModelsRenderer.render(
        graph, models, first_party=generated_by(models)
    ).unwrap()

    assert [file.path for file in files.files] == [models.module.file_path()]
    assert GeneratedFiles.is_generated(files.files[0].text)


def test_on_delete_for_a_field_that_is_not_a_reference_is_a_diagnostic() -> None:
    models = config(
        {},
        {
            ReferenceKey(entity=Company, field=FieldName("name")): OnDelete.CASCADE,
            ReferenceKey(entity=Department, field=FieldName("company_id")): (
                OnDelete.CASCADE
            ),
        },
    )

    assert diagnostics([Company], models) == (
        OnDeleteForNonReference(entity=Company, field=FieldName("name")),
        OnDeleteForNonReference(entity=Department, field=FieldName("company_id")),
    )


def test_set_null_on_a_required_reference_is_a_diagnostic() -> None:
    key = ReferenceKey(entity=Department, field=FieldName("company_id"))

    assert diagnostics([Company, Department], config({}, {key: OnDelete.SET_NULL})) == (
        SetNullOnRequiredReference(entity=Department, field=FieldName("company_id")),
    )


class ParentId(Id[int]):
    pass


class Parent(pydantic.BaseModel):
    id: ParentId


class Clashing(pydantic.BaseModel):
    """`parent_id` becomes the Django field `parent`, which `parent` already is."""

    id: ThingId
    parent_id: ParentId
    parent: Text


def test_a_foreign_key_named_like_another_field_is_a_diagnostic() -> None:
    assert diagnostics([Parent, Clashing], config({}, {})) == (
        DjangoFieldNameClash(
            entity=Clashing,
            name=FieldName("parent"),
            fields=(FieldName("parent"), FieldName("parent_id")),
        ),
    )


def another_company() -> type[pydantic.BaseModel]:
    """An Entity named like `Company`, defined elsewhere."""

    class Company(pydantic.BaseModel):
        id: ThingId

    return Company


def test_entities_with_one_name_are_a_diagnostic() -> None:
    assert diagnostics([Company, another_company()], config({}, {})) == (
        DjangoModelNameClash(name=Identifier("Company")),
    )


def test_an_entity_with_an_existing_model_gets_none_and_is_imported() -> None:
    text = render([Company, Department, CompanyDetails], existing_company({}))

    assert text == FileText(
        textwrap.dedent(
            """\
        # Generated by functional-relational-python. Do not edit.
        # Regenerate with `frp generate`; `frp check` fails when this is stale.
        # Migrations are yours to make, e.g. with `makemigrations`.

        from typing import ClassVar

        from crm.models import Company
        from django.db import models

        from app.queries import CompanyDetailsManager, DepartmentManager


        class CompanyDetails(models.Model):
            objects: ClassVar[CompanyDetailsManager] = CompanyDetailsManager()
            id = models.BigIntegerField(primary_key=True)
            company = models.OneToOneField(
                Company, on_delete=models.PROTECT, related_name="company_details_by_company"
            )
            successor = models.OneToOneField(
                Company,
                on_delete=models.PROTECT,
                related_name="company_details_by_successor",
                null=True,
            )

            class Meta:
                app_label = "app"


        class Department(models.Model):
            objects: ClassVar[DepartmentManager] = DepartmentManager()
            id = models.UUIDField(primary_key=True)
            company = models.ForeignKey(Company, on_delete=models.PROTECT, related_name="+")
            founded = models.DateField()

            class Meta:
                app_label = "app"
        """
        )
    )


def test_an_existing_model_of_a_non_entity_is_a_diagnostic() -> None:
    models = config({}, {}, {Thing: ExistingDjangoModel.fake()})

    assert diagnostics([Company], models) == (ExistingModelOfNonEntity(model=Thing),)


@pytest.mark.parametrize("module", ["app.models", "app.queries"])
def test_an_existing_model_in_a_generated_module_is_a_diagnostic(module: str) -> None:
    existing = ExistingDjangoModel(
        module=ModulePath(module), name=Identifier("Firm"), reverse_accessors={}
    )

    assert diagnostics([Company], config({}, {}, {Company: existing})) == (
        ExistingModelInGeneratedModule(entity=Company, module=ModulePath(module)),
    )


def existing_details(
    reverse_accessors: Mapping[FieldName, FieldName],
) -> DjangoModels:
    """`CompanyDetails`, whose References are one-to-one, has an existing Django
    model declaring `reverse_accessors`."""
    existing = ExistingDjangoModel(
        module=ModulePath("crm.models"),
        name=Identifier("CompanyDetails"),
        reverse_accessors=reverse_accessors,
    )
    return config({}, {}, {CompanyDetails: existing})


def test_reverse_accessors_an_existing_model_declares_amiss_are_diagnostics() -> None:
    models = existing_details(
        {
            FieldName("company_id"): FieldName("details"),
            FieldName("id"): FieldName("details_by_id"),
        }
    )

    assert diagnostics([Company, CompanyDetails], models) == (
        ReverseAccessorOfNonOneToOne(entity=CompanyDetails, field=FieldName("id")),
        UndeclaredReverseAccessor(
            entity=CompanyDetails, field=FieldName("successor_id")
        ),
    )


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("details", ()),
        # Django checks the existing model's names, not frp.
        ("details__", ()),
        (
            "name",
            (
                ReverseAccessorClash(
                    entity=Company,
                    name=FieldName("name"),
                    references=(
                        ReferenceKey(
                            entity=CompanyDetails, field=FieldName("company_id")
                        ),
                    ),
                ),
            ),
        ),
    ],
    ids=["free", "invalid-for-frp", "clashing"],
)
def test_reverse_accessors_of_an_existing_model_are_checked_as_declared(
    declared: str, expected: tuple[ReverseAccessorClash, ...]
) -> None:
    """An existing model's one-to-one Reference to a generated model binds the
    reverse accessor it declares, not the one frp would generate."""
    models = existing_details(
        {
            FieldName("company_id"): FieldName(declared),
            FieldName("successor_id"): FieldName("successor_details"),
        }
    )

    graph = EntityGraph.build(
        [Company, CompanyDetails], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()

    match DjangoModelsRenderer.render(graph, models, first_party=generated_by(models)):
        case Ok():
            reported: tuple[DjangoDiagnostic, ...] = ()
        case Err(error):
            reported = error.diagnostics
    assert reported == expected


def test_on_delete_for_a_reference_of_an_existing_model_is_a_diagnostic() -> None:
    key = ReferenceKey(entity=Company, field=FieldName("parent"))

    assert diagnostics([Company], existing_company({key: OnDelete.CASCADE})) == (
        OnDeleteOnExistingModel(entity=Company, field=FieldName("parent")),
    )


class TypeErrorKind(pydantic.RootModel[str], frozen=True):
    """A kind of error the type checker reports, e.g. `bad-override`."""


def django_project_passes_checks(
    project: Directory,
    apps: Sequence[AppLabel],
    migrated: AppLabel,
    type_errors_off: Sequence[TypeErrorKind],
) -> None:
    """The Django project in `project`, with `apps` installed, passes the type
    checker over `migrated`, but for the kinds of error in `type_errors_off`,
    Django's system checks, `makemigrations` of `migrated`, and `migrate`.
    `django.contrib.postgres` is installed too, for the `ArrayField` of a
    list."""
    directory = project.root
    installed = ["django.contrib.postgres", *(app.root for app in apps)]
    _ = (directory / "settings.py").write_text(
        textwrap.dedent(
            f"""\
            SECRET_KEY = "test"
            INSTALLED_APPS = {installed!r}
            DATABASES = {{
                "default": {{
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": {str(directory / "db.sqlite3")!r},
                }}
            }}
            USE_TZ = True
            """
        ),
        encoding="utf-8",
    )
    _ = (directory / "pyrefly.toml").write_text(
        f'project-includes = ["{migrated.root}"]\n'
        f'search-path = ["{directory}"]\n'
        f'python-interpreter-path = "{sys.executable}"\n'
        'preset = "all"\n'
        "[errors]\n" + "".join(f"{kind.root} = false\n" for kind in type_errors_off),
        encoding="utf-8",
    )

    for command in (
        ["pyrefly", "check", "--config", "pyrefly.toml"],
        ["django", "check", "--settings=settings", "--fail-level=WARNING"],
        ["django", "makemigrations", migrated.root, "--settings=settings"],
        ["django", "migrate", "--settings=settings"],
    ):
        result = subprocess.run(
            [sys.executable, "-m", *command],
            cwd=directory,
            env={**os.environ, "PYTHONPATH": str(directory)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert (directory / migrated.root / "migrations" / "0001_initial.py").exists()


def test_generated_models_referencing_an_existing_model_make_migrations(
    tmp_path: Path,
) -> None:
    """The existing model is the application's own, in the app `crm`, which
    already has its migrations."""
    models = existing_company({})
    graph = EntityGraph.build(
        [Company, Department, CompanyDetails], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    files = GeneratedFiles.merged(
        [
            DjangoModelsRenderer.render(
                graph, models, first_party=generated_by(models)
            ).unwrap(),
            DjangoQueriesRenderer.render(
                graph,
                models,
                (),
                PlacedModels(modules={}),
                first_party=generated_by(models),
            ).unwrap(),
        ]
    ).unwrap()
    files.write(LocalFileSystem(), Directory(tmp_path)).unwrap()
    _ = (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    crm = tmp_path / "crm"
    (crm / "migrations").mkdir(parents=True)
    _ = (crm / "__init__.py").write_text("", encoding="utf-8")
    _ = (crm / "migrations" / "__init__.py").write_text("", encoding="utf-8")
    _ = (crm / "models.py").write_text(
        textwrap.dedent(
            """\
            from django.db import models


            class Company(models.Model):
                id = models.BigIntegerField(primary_key=True)
                name = models.TextField()
                parent = models.ForeignKey(
                    "self", on_delete=models.PROTECT, related_name="+", null=True
                )
            """
        ),
        encoding="utf-8",
    )
    _ = (crm / "migrations" / "0001_initial.py").write_text(
        textwrap.dedent(
            """\
            from django.db import migrations, models


            class Migration(migrations.Migration):
                initial = True
                dependencies = []
                operations = [
                    migrations.CreateModel(
                        name="Company",
                        fields=[
                            (
                                "id",
                                models.BigIntegerField(
                                    primary_key=True, serialize=False
                                ),
                            ),
                            ("name", models.TextField()),
                            (
                                "parent",
                                models.ForeignKey(
                                    null=True,
                                    on_delete=models.PROTECT,
                                    related_name="+",
                                    to="crm.company",
                                ),
                            ),
                        ],
                    ),
                ]
            """
        ),
        encoding="utf-8",
    )

    django_project_passes_checks(
        Directory(tmp_path), (AppLabel("crm"), AppLabel("app")), AppLabel("app"), ()
    )


class Person(pydantic.BaseModel):
    id: ThingId
    email: Email
    nickname: Text | None
    employer_id: CompanyId
    previous_employer_id: CompanyId | None
    department_id: DepartmentId
    heads_id: Annotated[DepartmentId | None, OneToOne()]
    born: Moment


class ProjectBases:
    """What `app.base` of the Django project defines, for its models to use."""

    module: ClassVar[ModulePath] = ModulePath("app.base")
    source: ClassVar[FileText] = FileText(
        textwrap.dedent(
            """\
            import typing
            from collections.abc import Callable

            from django.db import models


            class Stamped(models.Model):
                created_at = models.DateTimeField(auto_now_add=True)

                class Meta:
                    abstract = True


            class Described:
                def describe(self) -> str:
                    return type(self).__name__


            def registered[M: type[models.Model]](model: M) -> M:
                return model


            def following[M: type[models.Model]](
                *, follow: list[str]
            ) -> Callable[[M], M]:
                return lambda model: model


            class LiveQuerySet[T: models.Model](models.QuerySet[T]):
                def newest_first(self) -> typing.Self:
                    return self.order_by("-pk")


            class LiveManager[T: models.Model](models.Manager[T]):
                @typing.override
                def get_queryset(self) -> models.QuerySet[T]:
                    return super().get_queryset().exclude(pk=None)
            """
        )
    )

    @staticmethod
    def symbol(name: Identifier) -> Symbol:
        return Symbol(module=ProjectBases.module, qualname=(name,))


def person_options() -> EntityModelOptions:
    """Every option, for the one Entity `Person`."""
    return EntityModelOptions(
        bases=(
            ProjectBases.symbol(Identifier("Described")),
            ProjectBases.symbol(Identifier("Stamped")),
        ),
        decorators=(
            BareDecorator(function=ProjectBases.symbol(Identifier("registered"))),
            CalledDecorator(
                function=ProjectBases.symbol(Identifier("following")),
                arguments={Identifier("follow"): ValueExpression('["employer"]')},
            ),
        ),
        ordering=(OrderingTerm("-created_at"), OrderingTerm("email")),
        defaults={
            FieldName("nickname"): DatabaseDefault(value=ValueExpression('"anon"')),
            FieldName("born"): AutoNowAdd(),
        },
        foreign_key_constraint=ForeignKeyConstraint.NOT_ENFORCED,
        query_set_base=ModelGenericBase(
            origin=ProjectBases.symbol(Identifier("LiveQuerySet"))
        ),
        manager_base=ModelGenericBase(
            origin=ProjectBases.symbol(Identifier("LiveManager"))
        ),
        db_table=DbTable("people"),
        indexes=(
            DjangoIndex(
                fields=(IndexField("born"), IndexField("-email")),
                name=IndexName("person_born"),
            ),
        ),
        related_names={
            FieldName("employer_id"): RelatedName("employees"),
            FieldName("heads_id"): RelatedName("head"),
        },
        primary_key=OwnPrimaryKey(),
        model_name=DefaultDjangoModelName(),
    )


def uuid4() -> Symbol:
    return Symbol(module=ModulePath("uuid"), qualname=(Identifier("uuid4"),))


def uuid_default() -> EntityModelOptions:
    return EntityModelOptions.inherited().model_copy(
        update={"defaults": {FieldName("id"): Default(value=uuid4())}}
    )


@pytest.mark.parametrize(
    "entity_model_options",
    [
        {},
        {
            Person: person_options(),
            Department: named(DjangoModelName("Unit"), uuid_default()),
            CompanyDetails: named(
                DjangoModelName("DjangoCompanyDetails"),
                keyed_by(FieldName("company_id")),
            ),
        },
    ],
    ids=["django", "options"],
)
def test_generated_models_type_check_and_make_migrations(
    tmp_path: Path,
    entity_model_options: Mapping[type[pydantic.BaseModel], EntityModelOptions],
) -> None:
    """A Django project with the generated models, and the QuerySets their managers
    use, as its one app: they pass the type checker, Django's system checks,
    `makemigrations` and `migrate`, and the database enforces their constraints.
    """
    text = DjangoField.builtin(Identifier("TextField"), {})
    models = config(
        {
            CompanyId: DjangoField.builtin(Identifier("BigAutoField"), {}),
            Email: char_field(),
            Kind: text,
            SubmitState: text,
        },
        {ReferenceKey(entity=Company, field=FieldName("parent")): OnDelete.SET_NULL},
    ).model_copy(
        update={
            "entity_model_options": entity_model_options,
            "constraints": {Campaign: campaign_constraints()},
        }
    )
    graph = EntityGraph.build(
        [Company, CompanyDetails, Department, Person, Ruled, Campaign],
        IdBase.fake(),
        OneToOneMarker.fake(),
    ).unwrap()
    files = GeneratedFiles.merged(
        [
            DjangoModelsRenderer.render(
                graph, models, first_party=generated_by(models)
            ).unwrap(),
            # No Views, so only the QuerySets the models' managers use.
            DjangoQueriesRenderer.render(
                graph,
                models,
                (),
                PlacedModels(modules={}),
                first_party=generated_by(models),
            ).unwrap(),
        ]
    ).unwrap()
    files.write(LocalFileSystem(), Directory(tmp_path)).unwrap()
    _ = (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    _ = (tmp_path / "app" / "base.py").write_text(
        ProjectBases.source.root, encoding="utf-8"
    )

    django_project_passes_checks(
        Directory(tmp_path), (AppLabel("app"),), AppLabel("app"), ()
    )
    _ = (tmp_path / "rejected.py").write_text(
        textwrap.dedent(
            """\
            import datetime
            import json

            import django
            from django.db import IntegrityError, transaction

            django.setup()

            from app.models import Campaign, Company

            company = Company.objects.create(name="Acme")
            now = datetime.datetime.now(tz=datetime.UTC)
            draft = {
                "company": company,
                "kind": "instance",
                "submit_state": "draft",
                "submitted_at": None,
                "lifetime_budget": None,
                "deleted_at": None,
            }
            cases = {
                "draft": {"name": "a"},
                "template": {"name": "b", "kind": "template"},
                "submitted template": {
                    "name": "c",
                    "kind": "template",
                    "submit_state": "submitted",
                    "submitted_at": now,
                },
                "stamped template": {
                    "name": "d",
                    "kind": "template",
                    "submitted_at": now,
                },
                "submitted": {
                    "name": "e",
                    "submit_state": "submitted",
                    "submitted_at": now,
                },
                "unstamped submitted": {"name": "f", "submit_state": "submitted"},
                "positive budget": {"name": "g", "lifetime_budget": 1},
                "zero budget": {"name": "h", "lifetime_budget": 0},
                "live name taken": {"name": "a"},
                "deleted name taken": {"name": "a", "deleted_at": now},
            }
            rejected = {}
            for id, (case, fields) in enumerate(cases.items()):
                try:
                    with transaction.atomic():
                        Campaign.objects.create(id=id, **{**draft, **fields})
                    rejected[case] = False
                except IntegrityError:
                    rejected[case] = True
            print(json.dumps(rejected))
            """
        ),
        encoding="utf-8",
    )
    rejected = subprocess.run(
        [sys.executable, "rejected.py"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path),
            "DJANGO_SETTINGS_MODULE": "settings",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(rejected.stdout) == {
        "draft": False,
        "template": False,
        "submitted template": True,
        "stamped template": True,
        "submitted": False,
        "unstamped submitted": True,
        "positive budget": False,
        "zero budget": True,
        "live name taken": True,
        "deleted name taken": False,
    }


class BadNames(pydantic.BaseModel):
    id: ThingId
    # Names Django rejects.
    pk: Text
    name_: Text
    first__name: Text
    # Names the class body would shadow.
    Meta: Text
    models: Text
    # A Reference loses `_id`, which can leave a keyword or a rejected name.
    from_id: ParentId
    owner__id: ParentId


def test_a_field_name_django_cannot_use_is_a_diagnostic() -> None:
    assert set(diagnostics([Parent, BadNames], config({}, {}))) == {
        InvalidDjangoFieldName(
            entity=BadNames, field=FieldName(field), name=FieldName(name)
        )
        for field, name in (
            ("pk", "pk"),
            ("name_", "name_"),
            ("first__name", "first__name"),
            ("Meta", "Meta"),
            ("models", "models"),
            ("from_id", "from"),
            ("owner__id", "owner_"),
        )
    }


class ShadowsImport(pydantic.BaseModel):
    id: ThingId
    MoneyField: Real


def test_a_field_named_like_an_imported_field_class_is_a_diagnostic() -> None:
    money = DjangoField(
        field_class=Symbol(
            module=ModulePath("app.fields"), qualname=(Identifier("MoneyField"),)
        ),
        arguments={},
    )

    assert diagnostics([ShadowsImport], config({Real: money}, {})) == (
        InvalidDjangoFieldName(
            entity=ShadowsImport,
            field=FieldName("MoneyField"),
            name=FieldName("MoneyField"),
        ),
    )


class OwnerIdTwice(pydantic.BaseModel):
    """The foreign key of `owner_id_id` is `owner_id`, as is the Reference
    `owner_id`'s ID attribute."""

    id: ThingId
    owner_id_id: ParentId
    owner_id: ParentId


class OwnerIdAndText(pydantic.BaseModel):
    id: ThingId
    owner_id_id: ParentId
    owner_id: Text


@pytest.mark.parametrize("model", [OwnerIdTwice, OwnerIdAndText])
def test_a_foreign_key_clashing_with_an_id_attribute_is_a_diagnostic(
    model: type[pydantic.BaseModel],
) -> None:
    assert diagnostics([Parent, model], config({}, {})) == (
        DjangoFieldNameClash(
            entity=model,
            name=FieldName("owner_id"),
            fields=(FieldName("owner_id"), FieldName("owner_id_id")),
        ),
    )


class SlugId(Id[str]):
    pass


class Article(pydantic.BaseModel):
    id: SlugId
    title: Text


def test_a_primary_key_wrapping_str_needs_an_override() -> None:
    assert diagnostics([Article], config({}, {})) == (
        NoDjangoFieldRule(
            entity=Article,
            field=FieldName("id"),
            annotation=AnnotationText.of(SlugId),
            unruled=AnnotationText.of(str),
        ),
    )


def test_a_primary_key_wrapping_str_takes_the_override_for_its_id_type() -> None:
    text = render([Article], config({SlugId: char_field()}, {})).root

    assert "    id = models.CharField(max_length=254, primary_key=True)\n" in text
    assert "    title = models.TextField()\n" in text


class Counter(pydantic.BaseModel):
    id: CompanyId
    count: Integer


def test_an_auto_increment_override_on_another_field_is_a_diagnostic() -> None:
    auto = DjangoField.builtin(Identifier("BigAutoField"), {})

    assert diagnostics([Counter], config({int: auto}, {})) == (
        AutoIncrementOnNonPrimaryKey(entity=Counter, field=FieldName("count")),
    )


def test_golden_django_model_with_options() -> None:
    """`Person`'s options apply over the catalog's, which `Company` and
    `Department` get."""
    models = config({}, {}).model_copy(
        update={"entity_model_options": {Person: person_options()}}
    )

    text = render([Company, Department, Person], models)

    assert text == FileText(
        textwrap.dedent(
            """\
        # Generated by functional-relational-python. Do not edit.
        # Regenerate with `frp generate`; `frp check` fails when this is stale.
        # Migrations are yours to make, e.g. with `makemigrations`.

        from typing import ClassVar

        from django.db import models

        from app.base import Described, Stamped, following, registered
        from app.queries import CompanyManager, DepartmentManager, PersonManager


        class Company(models.Model):
            objects: ClassVar[CompanyManager] = CompanyManager()
            id = models.BigIntegerField(primary_key=True)
            name = models.TextField()
            parent = models.ForeignKey(
                "Company", on_delete=models.PROTECT, related_name="+", null=True
            )

            class Meta:
                app_label = "app"


        class Department(models.Model):
            objects: ClassVar[DepartmentManager] = DepartmentManager()
            id = models.UUIDField(primary_key=True)
            company = models.ForeignKey("Company", on_delete=models.PROTECT, related_name="+")
            founded = models.DateField()

            class Meta:
                app_label = "app"


        @registered
        @following(follow=["employer"])
        class Person(Described, Stamped):
            objects: ClassVar[PersonManager] = PersonManager()
            id = models.BigIntegerField(primary_key=True)
            email = models.TextField()
            nickname = models.TextField(null=True, db_default="anon")
            employer = models.ForeignKey(
                "Company",
                on_delete=models.PROTECT,
                related_name="employees",
                db_constraint=False,
            )
            previous_employer = models.ForeignKey(
                "Company",
                on_delete=models.PROTECT,
                related_name="+",
                null=True,
                db_constraint=False,
            )
            department = models.ForeignKey(
                "Department", on_delete=models.PROTECT, related_name="+", db_constraint=False
            )
            heads = models.OneToOneField(
                "Department",
                on_delete=models.PROTECT,
                related_name="head",
                null=True,
                db_constraint=False,
            )
            born = models.DateTimeField(auto_now_add=True)

            class Meta:
                app_label = "app"
                db_table = "people"
                ordering = ["-created_at", "email"]
                indexes = [models.Index(fields=["born", "-email"], name="person_born")]
        """
        )
    )


class QuotedId(Id[int]):
    pass


class Rate(float, enum.Enum):
    BIG = 1e16


Vast = Annotated[float, annotated_types.Gt(1e20)]


class Quoted(pydantic.BaseModel):
    """Choices that hold quotes, bytes and floats with exponents, and names long
    enough that a line of the model does not fit."""

    id: QuotedId
    said: Literal["it's", 'say "hi"', "plain"]
    raw: Literal[b"it's", b"plain"]
    rate: Rate
    bounded: Vast
    a_reference_with_a_name_long_enough_that_its_line_does_not_fit_id: CompanyId
    an_optional_reference_whose_line_does_not_fit_even_split_over_lines_id: (
        CompanyId | None
    )
    a_reference_named_so_long_that_even_its_line_up_to_an_opening_parenthesis_does_not_fit_id: CompanyId
    a_text_named_so_long_that_its_line_does_not_fit_and_has_nothing_to_split_or_to_be_parenthesized_in: Text
    choices_too_many_for_one_line: Literal[
        "first choice",
        "second choice",
        "third choice",
        "fourth choice",
        "fifth choice",
    ]


class AnEntityWithAManagerTooLongId(Id[int]):
    pass


class AnEntityWithAManagerTooLong(pydantic.BaseModel):
    """Named so that the line of its manager does not fit, but does with its
    value parenthesized."""

    id: AnEntityWithAManagerTooLongId


class AnEntityNamedSoLongThatEvenItsManagerUpToTheValueDoesNotFitId(Id[int]):
    pass


class AnEntityNamedSoLongThatEvenItsManagerUpToTheValueDoesNotFit(pydantic.BaseModel):
    """Named so that the line of its manager does not fit even with its value
    parenthesized."""

    id: AnEntityNamedSoLongThatEvenItsManagerUpToTheValueDoesNotFitId


@pytest.mark.parametrize(
    "entity_model_options",
    [{}, {Person: person_options(), Department: uuid_default()}],
    ids=["django", "options"],
)
def test_generated_models_are_as_ruff_formats_and_sorts_them(
    entity_model_options: Mapping[type[pydantic.BaseModel], EntityModelOptions],
) -> None:
    models = config(
        {
            CompanyId: DjangoField.builtin(Identifier("BigAutoField"), {}),
            Kind: DjangoField.builtin(Identifier("TextField"), {}),
            SubmitState: DjangoField.builtin(Identifier("TextField"), {}),
        },
        {ReferenceKey(entity=Company, field=FieldName("parent")): OnDelete.SET_NULL},
    ).model_copy(
        update={
            "entity_model_options": entity_model_options,
            "constraints": {Campaign: campaign_constraints()},
        }
    )

    texts = (
        render(
            [
                Company,
                Department,
                Person,
                Ruled,
                Campaign,
                Quoted,
                AnEntityWithAManagerTooLong,
                AnEntityNamedSoLongThatEvenItsManagerUpToTheValueDoesNotFit,
            ],
            models,
        ),
        # Their primary keys are of the type `Person`'s is.
        render([Thing], all_rules_for_tags()),
        render([Bounded], config({}, {})),
    )

    assert [ruff_format_diff(text) for text in texts] == [FileText("")] * 3
    first_party = generated_by(models)
    assert [test_symbols.ruff_import_diff(text, first_party) for text in texts] == [
        FileText("")
    ] * 3


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


def test_an_entitys_options_apply_over_the_catalogs() -> None:
    """A setting the Entity inherits is the catalog's, one it sets its own;
    defaults merge, the Entity's winning, and a catalog-wide default applies
    only to an Entity having its field."""
    catalog = CatalogModelOptions.default().model_copy(
        update={
            "ordering": (OrderingTerm("name"),),
            "foreign_key_constraint": ForeignKeyConstraint.NOT_ENFORCED,
            "defaults": {
                FieldName("name"): Default(value=ValueExpression("'x'")),
                FieldName("founded"): AutoNow(),
            },
        }
    )
    own = EntityModelOptions.inherited().model_copy(
        update={
            "ordering": (OrderingTerm("-founded"),),
            "defaults": {FieldName("founded"): AutoNowAdd()},
        }
    )
    models = config({}, {}).model_copy(
        update={"model_options": catalog, "entity_model_options": {Department: own}}
    )

    department = models.options_of(Department)
    company = models.options_of(Company)

    assert department.meta.ordering == own.ordering
    assert department.defaults == own.defaults
    assert department.foreign_key_constraint == catalog.foreign_key_constraint
    assert company.meta.ordering == catalog.ordering
    assert company.defaults == {FieldName("name"): catalog.defaults[FieldName("name")]}


def test_a_catalog_wide_default_for_a_field_an_entity_lacks_is_no_diagnostic() -> None:
    catalog = CatalogModelOptions.default().model_copy(
        update={"defaults": {FieldName("founded"): AutoNowAdd()}}
    )
    models = config({}, {}).model_copy(update={"model_options": catalog})

    text = render([Company, Department], models).root

    assert "    founded = models.DateField(auto_now_add=True)\n" in text


def test_a_default_symbol_is_imported() -> None:
    models = config({}, {}).model_copy(
        update={"entity_model_options": {Department: uuid_default()}}
    )

    text = render([Company, Department], models).root

    assert "\nfrom uuid import uuid4\n" in text
    assert "    id = models.UUIDField(primary_key=True, default=uuid4)\n" in text


def test_a_field_shadowing_a_default_symbol_is_a_diagnostic() -> None:
    """The field `uuid4` would be bound in the class body before `id`'s default
    refers to the imported `uuid4`."""

    class Shadowing(pydantic.BaseModel):
        id: DepartmentId
        uuid4: Text

    models = config({}, {}).model_copy(
        update={"entity_model_options": {Shadowing: uuid_default()}}
    )

    assert diagnostics([Shadowing], models) == (
        InvalidDjangoFieldName(
            entity=Shadowing, field=FieldName("uuid4"), name=FieldName("uuid4")
        ),
    )


@pytest.mark.parametrize(
    ("default", "expected"),
    [
        (Default(value=ValueExpression("'x'")), "default='x'"),
        (DatabaseDefault(value=ValueExpression("'x'")), "db_default='x'"),
        (AutoNow(), "auto_now=True"),
        (AutoNowAdd(), "auto_now_add=True"),
    ],
)
def test_a_default_is_a_keyword_argument(default: FieldDefault, expected: str) -> None:
    models = all_rules_for_tags().model_copy(
        update={
            "model_options": CatalogModelOptions.default().model_copy(
                update={"defaults": {FieldName("text"): default}}
            )
        }
    )

    assert field_line(Thing, FieldName("text"), models) == FileText(
        f"text = models.TextField({expected})"
    )


def test_a_related_name_clashing_with_a_field_is_a_diagnostic() -> None:
    """`Company.name` is a field, so it cannot be the reverse accessor of
    `Department.company_id`."""
    named = EntityModelOptions.inherited().model_copy(
        update={"related_names": {FieldName("company_id"): RelatedName("name")}}
    )
    models = config({}, {}).model_copy(
        update={"entity_model_options": {Department: named}}
    )

    assert diagnostics([Company, Department], models) == (
        ReverseAccessorClash(
            entity=Company,
            name=FieldName("name"),
            references=(
                ReferenceKey(entity=Department, field=FieldName("company_id")),
            ),
        ),
    )


def test_options_naming_what_is_not_there_are_diagnostics() -> None:
    misplaced = EntityModelOptions.inherited().model_copy(
        update={
            "defaults": {FieldName("missing"): AutoNow()},
            "related_names": {FieldName("founded"): RelatedName.fake()},
        }
    )
    models = config({}, {}, {Company: ExistingDjangoModel.fake()}).model_copy(
        update={
            "entity_model_options": {
                Department: misplaced,
                Parent: misplaced,
                Company: EntityModelOptions.inherited(),
            }
        }
    )

    assert set(diagnostics([Company, Department], models)) == {
        ModelOptionsForNonEntity(entity=Parent),
        ModelOptionsForExistingModel(entity=Company),
        DefaultForNonField(entity=Department, field=FieldName("missing")),
        RelatedNameForNonReference(entity=Department, field=FieldName("founded")),
    }


def test_models_sharing_a_table_or_an_index_name_are_diagnostics() -> None:
    """`Department`'s table is the one Django names `Company`'s."""
    index = DjangoIndex(fields=(IndexField("id"),), name=IndexName("by_id"))
    models = config({}, {}).model_copy(
        update={
            "entity_model_options": {
                Company: EntityModelOptions.inherited().model_copy(
                    update={"indexes": (index,)}
                ),
                Department: EntityModelOptions.inherited().model_copy(
                    update={"db_table": DbTable("app_company"), "indexes": (index,)}
                ),
            }
        }
    )

    assert set(diagnostics([Company, Department], models)) == {
        SharedDbTable(table=DbTable("app_company"), entities=(Company, Department)),
        SharedIndexName(name=index.name, entities=(Company, Department)),
    }


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (RelatedName, "ad_sets_"),
        (RelatedName, "ad__sets"),
        (RelatedName, "class"),
        (RelatedName, "+"),
        (DbTable, ""),
        (OrderingTerm, "-"),
        (OrderingTerm, "created at"),
        (IndexField, "company__name"),
        (IndexName, "_born"),
        (IndexName, "a" * 31),
    ],
)
def test_a_value_django_rejects_is_invalid(
    value_type: type[pydantic.RootModel[str]], value: str
) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = value_type(value)


class Kind(enum.StrEnum):
    TEMPLATE = "template"
    INSTANCE = "instance"


class SubmitState(enum.StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"


class CampaignId(Id[int]):
    pass


class Campaign(pydantic.BaseModel):
    id: CampaignId
    company_id: CompanyId
    name: Text
    kind: Kind
    submit_state: SubmitState
    submitted_at: Moment | None
    lifetime_budget: Budget | None
    deleted_at: Moment | None


def campaign_constraints() -> tuple[Constraint, ...]:
    """Flowbase's constraints on a campaign."""
    return (
        Check(
            name=ConstraintName("campaign_template_is_never_submitted"),
            condition=conditions.or_(
                conditions.ne(conditions.field(FieldName("kind")), Kind.TEMPLATE),
                conditions.and_(
                    conditions.eq(
                        conditions.field(FieldName("submit_state")), SubmitState.DRAFT
                    ),
                    conditions.is_null(conditions.field(FieldName("submitted_at"))),
                ),
            ),
        ),
        Check(
            name=ConstraintName("campaign_submitted_is_stamped"),
            condition=conditions.or_(
                conditions.ne(
                    conditions.field(FieldName("submit_state")), SubmitState.SUBMITTED
                ),
                conditions.not_(
                    conditions.is_null(conditions.field(FieldName("submitted_at")))
                ),
            ),
        ),
        Unique(
            name=ConstraintName("campaign_unique_live_name"),
            fields=(FieldName("company_id"), FieldName("name")),
            where=conditions.is_null(conditions.field(FieldName("deleted_at"))),
        ),
    )


def campaign_config(constraints: Iterable[Constraint]) -> DjangoModels:
    """Enums have no built-in rule, so Kind and SubmitState get one."""
    text = DjangoField.builtin(Identifier("TextField"), {})
    return config({Kind: text, SubmitState: text}, {}).model_copy(
        update={"constraints": {Campaign: tuple(constraints)}}
    )


def test_golden_constraints() -> None:
    """A Check is written as the negation of the rows it excludes, a Unique as the
    rows it applies to, and a bound of a field's type as a Check of its own."""
    text = render([Company, Campaign], campaign_config(campaign_constraints())).root

    assert (
        "from functional_relational_python.outputs.django.test_models import "
        "Kind, SubmitState\n"
    ) in text
    assert (
        textwrap.indent(
            textwrap.dedent(
                """\
                class Meta:
                    app_label = "app"
                    constraints = [
                        models.CheckConstraint(
                            condition=~(
                                models.Q(
                                    kind=Kind.TEMPLATE.value,
                                )
                                & (
                                    ~models.Q(
                                        submit_state=SubmitState.DRAFT.value,
                                    )
                                    | models.Q(
                                        submitted_at__isnull=False,
                                    )
                                )
                            ),
                            name="campaign_template_is_never_submitted",
                        ),
                        models.CheckConstraint(
                            condition=~(
                                models.Q(
                                    submit_state=SubmitState.SUBMITTED.value,
                                )
                                & models.Q(
                                    submitted_at__isnull=True,
                                )
                            ),
                            name="campaign_submitted_is_stamped",
                        ),
                        models.UniqueConstraint(
                            fields=["company", "name"],
                            condition=models.Q(
                                deleted_at__isnull=True,
                            ),
                            name="campaign_unique_live_name",
                        ),
                        models.CheckConstraint(
                            condition=models.Q(
                                lifetime_budget__gt=0,
                            ),
                            name="campaign_lifetime_budget_gt",
                        ),
                    ]
                """
            ),
            "    ",
        )
        in text
    )


def test_a_unique_over_every_row_has_no_condition() -> None:
    unique = Unique(
        name=ConstraintName("campaign_unique_name"),
        fields=(FieldName("name"),),
        where=EveryRow(),
    )

    text = render([Company, Campaign], campaign_config([unique])).root

    assert (
        "            models.UniqueConstraint(\n"
        '                fields=["name"],\n'
        '                name="campaign_unique_name",\n'
        "            ),\n"
    ) in text


def test_a_model_without_constraints_has_no_constraints_option() -> None:
    assert "constraints" not in render([Company], config({}, {})).root


class Score(
    pydantic.RootModel[Annotated[float, annotated_types.Interval(ge=0, lt=1)]],
    frozen=True,
):
    pass


Positive = Annotated[int, annotated_types.Gt(0)]
Whole = int
AboveFive = Annotated[pydantic.PositiveInt, annotated_types.Gt(5)]
Huge = Annotated[int, annotated_types.Lt(10**400)]


class Level(enum.IntEnum):
    LOW = 1


AboveLow = Annotated[int, annotated_types.Gt(Level.LOW)]
type Aliased = Annotated[int, annotated_types.Ge(2)]
AfterY2K = Annotated[datetime.date, annotated_types.Gt(datetime.date(2000, 1, 1))]
Finite = Annotated[float, annotated_types.Lt(math.inf)]
NonEmpty = Annotated[str, annotated_types.MinLen(1)]


class Bounded(pydantic.BaseModel):
    """Each field's type sets bounds, or sets some that imply no constraint."""

    id: ThingId
    positive: pydantic.PositiveInt
    optional_positive: Positive | None
    wrapped: Budget
    field_bound: Whole = pydantic.Field(le=10)
    interval: Score
    date_bound: AfterY2K
    infinite_bound: Finite
    length: NonEmpty
    tightest: AboveFive
    huge: Huge
    int_enum_bound: AboveLow
    aliased: Aliased


def test_numeric_bounds_of_a_fields_type_are_check_constraints() -> None:
    """Of two bounds by one operator, the tightest; a huge `int` as written; an
    `IntEnum` bound as its `int`."""
    text = render([Bounded], config({}, {})).root

    assert re.findall(r"condition=models.Q\(\n\s+(\w+=\S+),", text) == [
        "positive__gt=0",
        "optional_positive__gt=0",
        "wrapped__gt=0",
        "field_bound__lte=10",
        "interval__gte=0",
        "interval__lt=1",
        "tightest__gt=5",
        f"huge__lt={10**400}",
        "int_enum_bound__gt=1",
        "aliased__gte=2",
    ]
    assert re.findall(r'name="(\w+)"', text) == [
        "bounded_positive_gt",
        "bounded_optional_positive_gt",
        "bounded_wrapped_gt",
        "bounded_field_bound_lte",
        "bounded_interval_gte",
        "bounded_interval_lt",
        "bounded_tightest_gt",
        "bounded_huge_lt",
        "bounded_int_enum_bound_gt",
        "bounded_aliased_gte",
    ]


def test_an_implied_bound_is_named_after_the_model() -> None:
    models = config({}, {}).model_copy(
        update={"entity_model_options": prefixed(Bounded)}
    )

    text = render([Bounded], models).root

    assert 'name="django_bounded_positive_gt"' in text


def test_a_bound_on_a_field_without_a_numeric_column_implies_no_constraint() -> None:
    """E.g. a `Budget` the catalog stores as text."""
    text_field = DjangoField.builtin(Identifier("TextField"), {})

    models = campaign_config([])
    models = models.model_copy(
        update={"field_types": {**models.field_types, Budget: text_field}}
    )

    text = render([Company, Campaign], models).root

    assert "lifetime_budget__gt" not in text


def test_constraints_for_an_entity_with_an_existing_model_are_a_diagnostic() -> None:
    models = existing_company({}).model_copy(
        update={"constraints": {Company: campaign_constraints()}}
    )

    assert ConstraintsOnExistingModel(entity=Company) in diagnostics(
        [Company, Department], models
    )


def test_constraints_for_a_model_that_is_not_an_entity_are_a_diagnostic() -> None:
    models = campaign_config(campaign_constraints())

    assert diagnostics([Company], models) == (ConstraintsForNonEntity(model=Campaign),)


def test_constraints_named_alike_are_a_diagnostic() -> None:
    """A declared constraint may clash with one a field's type implies."""
    check = Check(
        name=ConstraintName("campaign_lifetime_budget_gt"),
        condition=conditions.is_null(conditions.field(FieldName("deleted_at"))),
    )

    assert diagnostics([Company, Campaign], campaign_config([check])) == (
        ConstraintNameClash(
            name=ConstraintName("campaign_lifetime_budget_gt"), entities=(Campaign,)
        ),
    )


def test_a_defective_condition_is_a_diagnostic_naming_the_constraint() -> None:
    check = Check(
        name=ConstraintName("campaign_kind_is_known"),
        condition=conditions.eq(conditions.field(FieldName("kind")), SubmitState.DRAFT),
    )

    (diagnostic,) = diagnostics([Company, Campaign], campaign_config([check]))

    assert diagnostic.describe().root.startswith(
        "campaign_kind_is_known: a condition compares Campaign.kind"
    )


def test_a_unique_over_an_unknown_field_is_a_diagnostic() -> None:
    unique = Unique(
        name=ConstraintName("campaign_unique_title"),
        fields=(FieldName("title"),),
        where=conditions.is_null(conditions.field(FieldName("removed_at"))),
    )

    assert {
        (type(diagnostic), diagnostic.field)
        for diagnostic in diagnostics([Company, Campaign], campaign_config([unique]))
        if isinstance(diagnostic, UnknownField)
    } == {(UnknownField, FieldName("title")), (UnknownField, FieldName("removed_at"))}


def test_a_condition_on_an_enum_the_models_cannot_import_is_a_diagnostic() -> None:
    class Local(enum.StrEnum):
        ONLY = "only"

    class LocalId(Id[int]):
        pass

    class WithLocal(pydantic.BaseModel):
        id: LocalId
        local: Local

    name = ConstraintName("with_local_is_only")
    check = Check(
        name=name,
        condition=conditions.eq(conditions.field(FieldName("local")), Local.ONLY),
    )
    models = config(
        {Local: DjangoField.builtin(Identifier("TextField"), {})}, {}
    ).model_copy(update={"constraints": {WithLocal: (check,)}})

    assert diagnostics([WithLocal], models) == (
        UnimportableConstraintEnum(
            constraint=name, enum_class=AnnotationText.of(Local)
        ),
    )


def test_a_unique_naming_a_field_twice_is_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match="more than once: name"):
        _ = Unique(
            name=ConstraintName.fake(),
            fields=(FieldName("name"), FieldName("name")),
            where=EveryRow(),
        )


class BranchId(Id[int]):
    pass


class Branch(pydantic.BaseModel):
    """A child of `Company`, named to sort before it."""

    id: BranchId
    company_id: Annotated[CompanyId, ParentLink()]
    motto: Text


def branch_models() -> DjangoModels:
    """A check each on `Company` and `Branch`. Every model is ordered by name,
    but for `Branch`, which Django gives its parent's ordering."""
    return config({}, {}).model_copy(
        update={
            "model_options": CatalogModelOptions.default().model_copy(
                update={"ordering": (OrderingTerm("name"),)}
            ),
            "constraints": {
                Company: [
                    Check(
                        name=ConstraintName("company_is_named"),
                        condition=conditions.ne(
                            conditions.field(FieldName("name")), Text("")
                        ),
                    )
                ],
                Branch: [
                    Check(
                        name=ConstraintName("branch_has_a_motto"),
                        condition=conditions.ne(
                            conditions.field(FieldName("motto")), Text("")
                        ),
                    )
                ],
            },
        }
    )


def test_golden_child_model_subclasses_its_parents() -> None:
    text = render([Branch, Company], branch_models())

    assert text.root.split("\n\n\n", 1)[1] == (
        """\
class Company(models.Model):
    objects: ClassVar[CompanyManager] = CompanyManager()
    id = models.BigIntegerField(primary_key=True)
    name = models.TextField()
    parent = models.ForeignKey(
        "Company", on_delete=models.PROTECT, related_name="+", null=True
    )

    class Meta:
        app_label = "app"
        ordering = ["name"]
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(
                    name="",
                ),
                name="company_is_named",
            ),
        ]


class Branch(Company):
    objects: ClassVar[BranchManager] = BranchManager()
    company = models.OneToOneField(
        "Company",
        on_delete=models.CASCADE,
        related_name="branch_by_company",
        parent_link=True,
        primary_key=True,
    )
    motto = models.TextField()

    class Meta:
        app_label = "app"
        constraints = [
            models.CheckConstraint(
                condition=~models.Q(
                    motto="",
                ),
                name="branch_has_a_motto",
            ),
        ]
"""
    )


def branch_options() -> EntityModelOptions:
    """A base of `Branch`'s own, an ordering of its own, and a decorator
    following its parent link, as `reversion.register(follow=...)` does. It is
    keyed by its parent link, as it is without saying so."""
    return keyed_by(FieldName("company_id")).model_copy(
        update={
            "bases": (ProjectBases.symbol(Identifier("Described")),),
            "ordering": (OrderingTerm("-motto"),),
            "decorators": (
                CalledDecorator(
                    function=ProjectBases.symbol(Identifier("following")),
                    arguments={Identifier("follow"): ValueExpression('["company"]')},
                ),
            ),
        }
    )


def test_a_child_model_has_its_own_bases_after_its_parent_and_its_own_ordering() -> (
    None
):
    models = branch_models().model_copy(
        update={"entity_model_options": {Branch: branch_options()}}
    )

    text = render([Branch, Company], models).root

    assert (
        '\n@following(follow=["company"])\nclass Branch(Company, Described):\n'
    ) in text
    assert (
        '    class Meta:\n        app_label = "app"\n        ordering = ["-motto"]\n'
    ) in text.split("class Branch(", 1)[1]


def test_a_child_model_keyed_otherwise_than_by_its_parent_link_is_a_diagnostic() -> (
    None
):
    models = branch_models().model_copy(
        update={"entity_model_options": {Branch: keyed_by(FieldName("motto"))}}
    )

    assert diagnostics([Branch, Company], models) == (
        ChildNotKeyedByParentLink(
            entity=Branch,
            field=FieldName("motto"),
            parent_link=FieldName("company_id"),
        ),
    )


def test_a_childs_constraint_cannot_test_a_field_it_inherits() -> None:
    """Django constrains a child model's own columns only."""
    models = branch_models().model_copy(
        update={
            "constraints": {
                Branch: [
                    Check(
                        name=ConstraintName("branch_is_named"),
                        condition=conditions.ne(
                            conditions.field(FieldName("name")), Text("")
                        ),
                    )
                ]
            }
        }
    )

    assert diagnostics([Branch, Company], models) == (
        UnknownField(
            owner=ConstraintName("branch_is_named"),
            source=ir.SourceName("Branch"),
            field=FieldName("name"),
            valid=(FieldName("id"), FieldName("company_id"), FieldName("motto")),
        ),
    )


def test_a_child_model_makes_a_multi_table_inheritance_migration(
    tmp_path: Path,
) -> None:
    """A child's manager is not its parent's, which a type checker reports as an
    inconsistent override: the project turns those reports off."""
    models = branch_models().model_copy(
        update={"entity_model_options": {Branch: branch_options()}}
    )
    graph = EntityGraph.build(
        [Branch, Company], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    files = GeneratedFiles.merged(
        [
            DjangoModelsRenderer.render(
                graph, models, first_party=generated_by(models)
            ).unwrap(),
            DjangoQueriesRenderer.render(
                graph,
                models,
                (),
                PlacedModels(modules={}),
                first_party=generated_by(models),
            ).unwrap(),
        ]
    ).unwrap()
    files.write(LocalFileSystem(), Directory(tmp_path)).unwrap()
    _ = (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    _ = (tmp_path / "app" / "base.py").write_text(
        ProjectBases.source.root, encoding="utf-8"
    )

    django_project_passes_checks(
        Directory(tmp_path),
        (AppLabel("app"),),
        AppLabel("app"),
        (TypeErrorKind("bad-override"), TypeErrorKind("missing-override-decorator")),
    )
    migration = (tmp_path / "app" / "migrations" / "0001_initial.py").read_text(
        encoding="utf-8"
    )
    assert "bases=('app.company', app.base.Described)" in migration
