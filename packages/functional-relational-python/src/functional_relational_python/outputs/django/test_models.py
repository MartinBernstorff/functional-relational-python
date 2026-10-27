import datetime
import enum
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated

import annotated_types
import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.compiler.diagnostics import UnknownField
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
    ConstraintNameClash,
    ConstraintsForNonEntity,
    DjangoDiagnostic,
    DjangoField,
    DjangoFieldNameClash,
    DjangoModelNameClash,
    DjangoModels,
    DjangoModelsRenderer,
    InvalidDjangoFieldName,
    InvalidReverseAccessorName,
    NoDjangoFieldRule,
    OnDelete,
    OnDeleteForNonReference,
    ReferenceKey,
    ReverseAccessorClash,
    SetNullOnRequiredReference,
    UnimportableConstraintEnum,
)
from functional_relational_python.outputs.django.queries import DjangoQueriesRenderer
from functional_relational_python.outputs.python.plan import PlacedModels
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    ModulePath,
    Symbol,
    ValueExpression,
)
from functional_relational_python.pipeline import conditions
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
) -> DjangoModels:
    return DjangoModels(
        module=ModulePath("app.models"),
        app_label=AppLabel("app"),
        field_types=field_types,
        on_delete=on_delete,
        queries=ModulePath("app.queries"),
        constraints={},
    )


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


def field_line(field: FieldName, models: DjangoModels) -> FileText:
    """The line declaring `field` of `Thing`."""
    (line,) = (
        FileText(line.strip())
        for line in render([Thing], models).root.splitlines()
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
    assert field_line(FieldName(field), all_rules_for_tags()) == FileText(
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

    assert field_line(FieldName(field), models) == FileText(
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

    assert field_line(FieldName("optional"), models) == FileText(
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
        ),
        NoDjangoFieldRule(
            entity=Untyped,
            field=FieldName("labels"),
            annotation=AnnotationText.of(frozenset[Text]),
        ),
    }


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


def test_reverse_accessors_named_alike_are_a_diagnostic() -> None:
    """`Owned.team_by_owner_id` and `OwnedByTeam.owner_id` both have the reverse
    accessor `owned_by_team_by_owner`."""

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

    assert diagnostics([Company, Owned, OwnedByTeam], config({}, {})) == (
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


def test_generated_models_type_check_and_make_migrations(tmp_path: Path) -> None:
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
    ).model_copy(update={"constraints": {Campaign: campaign_constraints()}})

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
        [Company, Department, Person, Campaign], IdBase.fake(), OneToOneMarker.fake()
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
    _ = (tmp_path / "settings.py").write_text(
        textwrap.dedent(
            f"""\
            SECRET_KEY = "test"
            INSTALLED_APPS = ["app"]
            DATABASES = {{
                "default": {{
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": {str(tmp_path / "db.sqlite3")!r},
                }}
            }}
            USE_TZ = True
            """
        ),
        encoding="utf-8",
    )
    _ = (tmp_path / "pyrefly.toml").write_text(
        f'project-includes = ["app"]\n'
        f'search-path = ["{tmp_path}"]\n'
        f'python-interpreter-path = "{sys.executable}"\n'
        'preset = "all"\n',
        encoding="utf-8",
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
                "stamped template": {"name": "d", "kind": "template", "submitted_at": now},
                "submitted": {"name": "e", "submit_state": "submitted", "submitted_at": now},
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

    for command in (
        ["pyrefly", "check", "--config", "pyrefly.toml"],
        ["django", "check", "--settings=settings", "--fail-level=WARNING"],
        ["django", "makemigrations", "app", "--settings=settings"],
        ["django", "migrate", "--settings=settings"],
    ):
        result = subprocess.run(
            [sys.executable, "-m", *command],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(tmp_path)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "app" / "migrations" / "0001_initial.py").exists()
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
            entity=Article, field=FieldName("id"), annotation=AnnotationText.of(SlugId)
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


class Kind(enum.StrEnum):
    TEMPLATE = "template"
    INSTANCE = "instance"


class SubmitState(enum.StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"


class Budget(pydantic.RootModel[pydantic.PositiveInt], frozen=True):
    """Its `Gt(0)` implies a check constraint."""


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
                    app_label = 'app'
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
                            name='campaign_template_is_never_submitted',
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
                            name='campaign_submitted_is_stamped',
                        ),
                        models.UniqueConstraint(
                            fields=['company', 'name'],
                            condition=models.Q(
                                deleted_at__isnull=True,
                            ),
                            name='campaign_unique_live_name',
                        ),
                        models.CheckConstraint(
                            condition=models.Q(
                                lifetime_budget__gt=0,
                            ),
                            name='campaign_lifetime_budget_gt',
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
        "                fields=['name'],\n"
        "                name='campaign_unique_name',\n"
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
Count = int
AfterY2K = Annotated[datetime.date, annotated_types.Gt(datetime.date(2000, 1, 1))]
Finite = Annotated[float, annotated_types.Lt(math.inf)]
NonEmpty = Annotated[str, annotated_types.MinLen(1)]


class Bounded(pydantic.BaseModel):
    """Each field's type sets bounds, or sets some that imply no constraint."""

    id: ThingId
    positive: pydantic.PositiveInt
    optional_positive: Positive | None
    wrapped: Budget
    field_bound: Count = pydantic.Field(le=10)
    interval: Score
    date_bound: AfterY2K
    infinite_bound: Finite
    length: NonEmpty


def test_numeric_bounds_of_a_fields_type_are_check_constraints() -> None:
    text = render([Bounded], config({}, {})).root

    assert re.findall(r"condition=models.Q\(\n\s+(\w+=\S+),", text) == [
        "positive__gt=0",
        "optional_positive__gt=0",
        "wrapped__gt=0",
        "field_bound__lte=10",
        "interval__gte=0",
        "interval__lt=1",
    ]
    assert re.findall(r"name='(\w+)'", text) == [
        "bounded_positive_gt",
        "bounded_optional_positive_gt",
        "bounded_wrapped_gt",
        "bounded_field_bound_lte",
        "bounded_interval_gte",
        "bounded_interval_lt",
    ]


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
