import datetime
import enum
import math
import os
import subprocess
import sys
import textwrap
import types
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import django.db.models
import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    RuntimeAnnotation,
)
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    OneToOne,
    OneToOneMarker,
)
from functional_relational_python.outputs.django.models import (
    AppLabel,
    AutoIncrementOnNonPrimaryKey,
    DjangoDiagnostic,
    DjangoField,
    DjangoFieldNameClash,
    DjangoModelNameClash,
    DjangoModels,
    DjangoModelsRenderer,
    ExistingDjangoModel,
    ExistingModelInGeneratedModule,
    ExistingModelOfNonEntity,
    InvalidDjangoFieldName,
    InvalidReverseAccessorName,
    MixedDjangoChoiceTypes,
    NoDjangoFieldRule,
    OnDelete,
    OnDeleteForNonReference,
    OnDeleteOnExistingModel,
    ReferenceKey,
    ReverseAccessorClash,
    ReverseAccessorOfNonOneToOne,
    SetNullOnRequiredReference,
    UndeclaredReverseAccessor,
    UnimportableDjangoChoices,
    UnwritableDjangoChoices,
)
from functional_relational_python.outputs.django.queries import DjangoQueriesRenderer
from functional_relational_python.outputs.python.plan import PlacedModels
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    ModulePath,
    Symbol,
    ValueExpression,
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
        existing=existing,
    )


def existing_company(on_delete: Mapping[ReferenceKey, OnDelete]) -> DjangoModels:
    """`Company` has an existing Django model, `Company` of `crm.models`."""
    return config({}, on_delete, {Company: ExistingDjangoModel.fake()})


def render(
    entities: Iterable[type[pydantic.BaseModel]], models: DjangoModels
) -> FileText:
    match DjangoModelsRenderer.render(
        EntityGraph.build(entities, IdBase.fake(), OneToOneMarker.fake()).unwrap(),
        models,
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
    ):
        case Ok(files):
            pytest.fail(f"expected diagnostics, got {files}")
        case Err(error):
            return error.diagnostics


def field_line(
    entity: type[pydantic.BaseModel], field: FieldName, models: DjangoModels
) -> FileText:
    """The line declaring `field` of `entity`."""
    (line,) = (
        FileText(line.strip())
        for line in render([entity], models).root.splitlines()
        if line.strip().startswith(f"{field.root} = ")
    )
    return line


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
        "\n\nfrom app.fields import MoneyField\n"
        "from app.queries import ThingManager\n"
        "from django.db import models\n\n\n"
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
            "models.TextField(choices=[('light', 'light'), ('dark', 'dark')])",
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
            "models.TextField(choices=[('area', 'area'), ('money', 'money')])",
        ),
        (
            "maybe_kind",
            "models.TextField(choices=[('area', 'area')], null=True)",
        ),
        ("member", "models.TextField(choices=[('dark', 'dark')])"),
        # A list is an array of the field of its items.
        (
            "shades",
            (
                "ArrayField(base_field=models.TextField("
                "choices=[('light', 'light'), ('dark', 'dark')]))"
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
        ({}, "company = models.ForeignKey('Company', on_delete=models.PROTECT, related_name='+')"),
        (
            {ReferenceKey(entity=Department, field=FieldName("company_id")): OnDelete.CASCADE},
            "company = models.ForeignKey('Company', on_delete=models.CASCADE, related_name='+')",
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
        "    parent = models.ForeignKey('Company', on_delete=models.PROTECT, "
        "related_name='+', null=True)\n"
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
        "    company = models.OneToOneField('Company', on_delete=models.PROTECT, "
        "related_name='company_details_by_company')\n"
        "    successor = models.OneToOneField('Company', on_delete=models.PROTECT, "
        "related_name='company_details_by_successor', null=True)\n"
    ) in text


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

        from app.queries import CompanyManager, DepartmentManager
        from django.db import models


        class Company(models.Model):
            objects = CompanyManager()
            id = models.BigAutoField(primary_key=True)
            name = models.TextField()
            parent = models.ForeignKey('Company', on_delete=models.SET_NULL, related_name='+', null=True)

            class Meta:
                app_label = 'app'


        class Department(models.Model):
            objects = DepartmentManager()
            id = models.UUIDField(primary_key=True)
            company = models.ForeignKey('Company', on_delete=models.PROTECT, related_name='+')
            founded = models.DateField()

            class Meta:
                app_label = 'app'
        """
        )
    )


def test_the_module_is_written_where_the_catalog_names() -> None:
    graph = EntityGraph.build([Company], IdBase.fake(), OneToOneMarker.fake()).unwrap()
    models = config({}, {})

    files = DjangoModelsRenderer.render(graph, models).unwrap()

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

        from app.queries import CompanyDetailsManager, DepartmentManager
        from crm.models import Company
        from django.db import models


        class CompanyDetails(models.Model):
            objects = CompanyDetailsManager()
            id = models.BigIntegerField(primary_key=True)
            company = models.OneToOneField(Company, on_delete=models.PROTECT, related_name='company_details_by_company')
            successor = models.OneToOneField(Company, on_delete=models.PROTECT, related_name='company_details_by_successor', null=True)

            class Meta:
                app_label = 'app'


        class Department(models.Model):
            objects = DepartmentManager()
            id = models.UUIDField(primary_key=True)
            company = models.ForeignKey(Company, on_delete=models.PROTECT, related_name='+')
            founded = models.DateField()

            class Meta:
                app_label = 'app'
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

    match DjangoModelsRenderer.render(graph, models):
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


def django_project_passes_checks(
    project: Directory, apps: Sequence[AppLabel], migrated: AppLabel
) -> None:
    """The Django project in `project`, with `apps` installed, passes the type
    checker over `migrated`, Django's system checks, `makemigrations` of
    `migrated`, and `migrate`. `django.contrib.postgres` is installed too, for
    the `ArrayField` of a list."""
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
        'preset = "all"\n',
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
            DjangoModelsRenderer.render(graph, models).unwrap(),
            DjangoQueriesRenderer.render(
                graph, models, (), PlacedModels(modules={})
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
        Directory(tmp_path), (AppLabel("crm"), AppLabel("app")), AppLabel("app")
    )


def test_generated_models_type_check_and_make_migrations(tmp_path: Path) -> None:
    """A Django project with the generated models, and the QuerySets their managers
    use, as its one app: they pass the type checker, Django's system checks,
    `makemigrations` and `migrate`."""
    models = config(
        {
            CompanyId: DjangoField.builtin(Identifier("BigAutoField"), {}),
            Email: char_field(),
        },
        {ReferenceKey(entity=Company, field=FieldName("parent")): OnDelete.SET_NULL},
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

    graph = EntityGraph.build(
        [Company, Department, Person, Ruled], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    files = GeneratedFiles.merged(
        [
            DjangoModelsRenderer.render(graph, models).unwrap(),
            # No Views, so only the QuerySets the models' managers use.
            DjangoQueriesRenderer.render(
                graph, models, (), PlacedModels(modules={})
            ).unwrap(),
        ]
    ).unwrap()
    files.write(LocalFileSystem(), Directory(tmp_path)).unwrap()
    _ = (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")

    django_project_passes_checks(
        Directory(tmp_path), (AppLabel("app"),), AppLabel("app")
    )


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
