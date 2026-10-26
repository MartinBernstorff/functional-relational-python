import datetime
import os
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    RuntimeAnnotation,
)
from functional_relational_python.entity_graph.id import Id
from functional_relational_python.outputs.django.models import (
    AppLabel,
    DjangoDiagnostic,
    DjangoField,
    DjangoFieldNameClash,
    DjangoModelNameClash,
    DjangoModels,
    DjangoModelsRenderer,
    NoDjangoFieldRule,
    OnDelete,
    OnDeleteForNonReference,
    ReferenceKey,
    SetNullOnRequiredReference,
)
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
) -> DjangoModels:
    return DjangoModels(
        module=ModulePath("app.models"),
        app_label=AppLabel("app"),
        field_types=field_types,
        on_delete=on_delete,
    )


def render(
    entities: Iterable[type[pydantic.BaseModel]], models: DjangoModels
) -> FileText:
    match DjangoModelsRenderer.render(EntityGraph.build(entities).unwrap(), models):
        case Ok(files):
            (file,) = files.files
            return file.text
        case Err(error):
            pytest.fail(f"expected generated files, got {error}")


def diagnostics(
    entities: Iterable[type[pydantic.BaseModel]], models: DjangoModels
) -> tuple[DjangoDiagnostic, ...]:
    match DjangoModelsRenderer.render(EntityGraph.build(entities).unwrap(), models):
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

    assert "from app.fields import MoneyField\nfrom django.db import models\n" in text
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

        from django.db import models


        class Company(models.Model):
            id = models.BigAutoField(primary_key=True)
            name = models.TextField()
            parent = models.ForeignKey('Company', on_delete=models.SET_NULL, related_name='+', null=True)

            class Meta:
                app_label = 'app'


        class Department(models.Model):
            id = models.UUIDField(primary_key=True)
            company = models.ForeignKey('Company', on_delete=models.PROTECT, related_name='+')
            founded = models.DateField()

            class Meta:
                app_label = 'app'
        """
        )
    )


def test_the_module_is_written_where_the_catalog_names() -> None:
    graph = EntityGraph.build([Company]).unwrap()
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
            name=Identifier("parent"),
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
    """A Django project with the generated models as its one app: they pass the
    type checker, Django's system checks, `makemigrations` and `migrate`."""
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
        born: Moment

    graph = EntityGraph.build([Company, Department, Person]).unwrap()
    files = DjangoModelsRenderer.render(graph, models).unwrap()
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
