"""Renders a Django model per Entity, so a Django project can store Entities.

Everything here is pure: it builds file contents and touches no filesystem, and
it never imports Django. Migrations are the user's to make.
"""

import datetime
import enum
import typing
import uuid
from collections.abc import Iterable, Iterator, Mapping

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    EntityNode,
    FieldName,
    Optionality,
    Reference,
    RuntimeAnnotation,
)
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    ModulePath,
    Symbol,
    TypeExpression,
    ValueExpression,
)
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)
from functional_relational_python.support.filesystem import FileText
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)
from functional_relational_python.support.identifier import Identifier


class DjangoField(pydantic.BaseModel, frozen=True):
    """A Django model field: its class and keyword arguments, e.g.
    `models.CharField(max_length=254)`.

    A field class in `django.db.models` is written as `models.<name>`; any other
    is imported by name.
    """

    models_module: typing.ClassVar[ModulePath] = ModulePath("django.db.models")

    field_class: Symbol
    arguments: Mapping[Identifier, ValueExpression]

    @staticmethod
    def builtin(
        name: Identifier, arguments: Mapping[Identifier, ValueExpression]
    ) -> DjangoField:
        """The field class `name` of `django.db.models`."""
        return DjangoField(
            field_class=Symbol(module=DjangoField.models_module, qualname=(name,)),
            arguments=arguments,
        )

    @staticmethod
    def fake() -> DjangoField:
        return DjangoField.builtin(Identifier("TextField"), {})

    def is_builtin(self) -> bool:
        return self.field_class.module == DjangoField.models_module

    def reference(self) -> TypeExpression:
        """The field class, where the generated module refers to it."""
        if self.is_builtin():
            return TypeExpression(f"models.{self.field_class.reference().root}")
        return self.field_class.reference()


class OnDelete(enum.StrEnum):
    """What deleting a referenced Entity does to the Entities referencing it.

    `SET_DEFAULT` is left out, because generated fields have no default.
    """

    CASCADE = "CASCADE"
    PROTECT = "PROTECT"
    RESTRICT = "RESTRICT"
    SET_NULL = "SET_NULL"
    DO_NOTHING = "DO_NOTHING"


class ReferenceKey(pydantic.BaseModel, frozen=True):
    """A Reference, by the Entity declaring it and its field."""

    entity: type[pydantic.BaseModel]
    field: FieldName


class AppLabel(pydantic.RootModel[str], frozen=True):
    """A Django app label, e.g. `app`. It is a Python identifier."""

    @pydantic.field_validator("root")
    @classmethod
    def _identifier(cls, root: str) -> str:
        _ = Identifier(root)
        return root

    @staticmethod
    def fake() -> AppLabel:
        return AppLabel("app")


class DjangoModels(pydantic.BaseModel, frozen=True):
    """Generate a Django model per Entity, all into `module` with `app_label`.

    `field_types` overrides the built-in rule for a type, see `FieldTypes`.
    `on_delete` sets what deleting the referenced Entity does, per Reference; a
    Reference not in it is `PROTECT`.
    """

    module: ModulePath
    app_label: AppLabel
    field_types: Mapping[RuntimeAnnotation, DjangoField]
    on_delete: Mapping[ReferenceKey, OnDelete]

    @staticmethod
    def fake() -> DjangoModels:
        return DjangoModels(
            module=ModulePath("app.models"),
            app_label=AppLabel.fake(),
            field_types={},
            on_delete={},
        )


class WithoutDjangoModels(pydantic.BaseModel, frozen=True):
    """Generate no Django models."""


class NoDjangoFieldRule(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity field's type, and every type it wraps, has no Django field rule."""

    entity: type[pydantic.BaseModel]
    field: FieldName
    annotation: AnnotationText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: no Django field for "
            f"'{self.annotation.root}'. Add one to DjangoModels.field_types."
        )


class OnDeleteForNonReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`on_delete` names a field that is not a Reference of an Entity."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: DjangoModels.on_delete "
            "names it, but it is not a Reference of an Entity."
        )


class SetNullOnRequiredReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`on_delete` is `SET_NULL` for a Reference that cannot be `None`."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: on_delete is SET_NULL, "
            "but the Reference is not optional."
        )


class DjangoFieldNameClash(pydantic.BaseModel, Diagnostic, frozen=True):
    """Several fields of an Entity would give its Django model one attribute.

    A Reference `x_id` becomes the foreign key `x`, whose ID attribute is `x_id`.
    """

    entity: type[pydantic.BaseModel]
    name: Identifier
    fields: tuple[FieldName, ...]

    def describe(self) -> DiagnosticMessage:
        fields = ", ".join(f"'{field.root}'" for field in self.fields)
        return DiagnosticMessage(
            f"{self.entity.__qualname__}: the fields {fields} would each be the "
            f"Django model attribute '{self.name.root}'."
        )


class DjangoModelNameClash(pydantic.BaseModel, Diagnostic, frozen=True):
    """Several Entities, or an Entity and an import, have one name in the module
    their Django models are generated into."""

    name: Identifier

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"Django models: several Entities or imports are named '{self.name.root}'."
        )


type DjangoDiagnostic = (
    NoDjangoFieldRule
    | OnDeleteForNonReference
    | SetNullOnRequiredReference
    | DjangoFieldNameClash
    | DjangoModelNameClash
)


class NoFieldRuleError(Exception):
    def __init__(self, annotation: RuntimeAnnotation) -> None:
        self.annotation = annotation
        super().__init__(f"No Django field for {annotation!r}")


class FieldTypes:
    """The Django field for an Entity field's type.

    The type is unwrapped one RootModel at a time, e.g. `Email` to `Text` to
    `str`. The first type with a rule, overridden or built in, gives the field.
    The field is nullable when any of those types admits `None`.
    """

    builtin: typing.ClassVar[Mapping[RuntimeAnnotation, DjangoField]] = {
        str: DjangoField.builtin(Identifier("TextField"), {}),
        int: DjangoField.builtin(Identifier("BigIntegerField"), {}),
        bool: DjangoField.builtin(Identifier("BooleanField"), {}),
        float: DjangoField.builtin(Identifier("FloatField"), {}),
        bytes: DjangoField.builtin(Identifier("BinaryField"), {}),
        datetime.datetime: DjangoField.builtin(Identifier("DateTimeField"), {}),
        datetime.date: DjangoField.builtin(Identifier("DateField"), {}),
        datetime.time: DjangoField.builtin(Identifier("TimeField"), {}),
        datetime.timedelta: DjangoField.builtin(Identifier("DurationField"), {}),
        uuid.UUID: DjangoField.builtin(Identifier("UUIDField"), {}),
    }

    def __init__(self, overrides: Mapping[RuntimeAnnotation, DjangoField]) -> None:
        self.overrides = overrides

    def field_of(
        self, annotation: RuntimeAnnotation
    ) -> Result[_TypedField, NoFieldRuleError]:
        optionality = Optionality.REQUIRED
        current = annotation
        while True:
            split = Optionality.of_annotation(current)
            if split.optionality == Optionality.OPTIONAL:
                optionality = Optionality.OPTIONAL
            current = split.required
            if current in self.overrides:
                return Ok(
                    _TypedField(field=self.overrides[current], optionality=optionality)
                )
            if isinstance(current, type) and issubclass(current, pydantic.RootModel):
                current = current.model_fields["root"].annotation
                continue
            if current in FieldTypes.builtin:
                return Ok(
                    _TypedField(
                        field=FieldTypes.builtin[current], optionality=optionality
                    )
                )
            return Err(NoFieldRuleError(annotation))


class _TypedField(pydantic.BaseModel, frozen=True):
    field: DjangoField
    optionality: Optionality


class _Declaration(pydantic.BaseModel, frozen=True):
    """One attribute of a Django model, e.g. `name = models.TextField()`.

    `attributes` are the names it binds on the model: a foreign key `x` also
    binds `x_id`.
    """

    name: Identifier
    field: DjangoField
    positional: tuple[ValueExpression, ...]
    source: FieldName
    attributes: tuple[Identifier, ...]

    def line(self) -> FileText:
        arguments = ", ".join(
            [
                *(value.root for value in self.positional),
                *(
                    f"{name.root}={value.root}"
                    for name, value in self.field.arguments.items()
                ),
            ]
        )
        return FileText(
            f"    {self.name.root} = {self.field.reference().root}({arguments})\n"
        )


class _PlannedModel(pydantic.BaseModel, frozen=True):
    name: Identifier
    declarations: tuple[_Declaration, ...]


class DjangoModelsRenderer:
    """Writes the Django model of every Entity into one module."""

    header: typing.ClassVar[FileText] = FileText(
        f"{GeneratedFiles.header.root}"
        "# Regenerate with `frp generate`; `frp check` fails when this is stale.\n"
        "# Migrations are yours to make, e.g. with `makemigrations`.\n"
    )
    models_import: typing.ClassVar[Identifier] = Identifier("models")

    @staticmethod
    def render(
        graph: EntityGraph, config: DjangoModels
    ) -> Result[GeneratedFiles, DiagnosticsError[DjangoDiagnostic]]:
        """The module of Django models, or every diagnostic.

        Models are in name order, and refer to each other by name, so the order of
        Entities does not matter.
        """
        types = FieldTypes(config.field_types)
        reported: list[DjangoDiagnostic] = []
        planned: list[_PlannedModel] = []
        references = {
            ReferenceKey(entity=model, field=reference.field): reference
            for model, node in graph.entities.items()
            for reference in node.references
        }
        for key, on_delete in config.on_delete.items():
            match references.get(key):
                case None:
                    reported.append(
                        OnDeleteForNonReference(entity=key.entity, field=key.field)
                    )
                case Reference(optionality=Optionality.REQUIRED) if (
                    on_delete == OnDelete.SET_NULL
                ):
                    reported.append(
                        SetNullOnRequiredReference(entity=key.entity, field=key.field)
                    )
                case Reference():
                    pass
        for model, node in sorted(
            graph.entities.items(), key=lambda item: item[0].__name__
        ):
            declarations: list[_Declaration] = []
            for declared in DjangoModelsRenderer._declarations(
                model, node, types, config.on_delete
            ):
                match declared:
                    case _Declaration():
                        declarations.append(declared)
                    case NoDjangoFieldRule():
                        reported.append(declared)
            reported.extend(DjangoModelsRenderer._field_clashes(model, declarations))
            planned.append(
                _PlannedModel(
                    name=Identifier(model.__name__), declarations=tuple(declarations)
                )
            )
        imports = {
            declaration.field.field_class
            for model in planned
            for declaration in model.declarations
            if not declaration.field.is_builtin()
        }
        reported.extend(DjangoModelsRenderer._name_clashes(planned, imports))
        return DiagnosticsError.or_ok(
            sorted(reported, key=lambda diagnostic: diagnostic.describe().root),
            lambda: GeneratedFiles(
                files=(
                    GeneratedFile(
                        path=config.module.file_path(),
                        text=DjangoModelsRenderer._module(
                            config.app_label, planned, imports
                        ),
                    ),
                )
            ),
        )

    @staticmethod
    def _declarations(
        model: type[pydantic.BaseModel],
        node: EntityNode,
        types: FieldTypes,
        on_delete: Mapping[ReferenceKey, OnDelete],
    ) -> Iterator[_Declaration | NoDjangoFieldRule]:
        references = {reference.field: reference for reference in node.references}
        for name, info in model.model_fields.items():
            field = FieldName(name)
            match references.get(field):
                case Reference() as reference:
                    yield DjangoModelsRenderer._foreign_key(model, reference, on_delete)
                    continue
                case None:
                    pass
            match types.field_of(info.annotation):
                case Ok(typed):
                    pass
                case Err():
                    yield NoDjangoFieldRule(
                        entity=model,
                        field=field,
                        annotation=AnnotationText.of(info.annotation),
                    )
                    continue
            arguments = dict(typed.field.arguments)
            if typed.optionality == Optionality.OPTIONAL:
                arguments[Identifier("null")] = ValueExpression("True")
            if field == EntityNode.primary_key:
                arguments[Identifier("primary_key")] = ValueExpression("True")
            yield _Declaration(
                name=Identifier(name),
                field=typed.field.model_copy(update={"arguments": arguments}),
                positional=(),
                source=field,
                attributes=(Identifier(name),),
            )

    @staticmethod
    def _foreign_key(
        model: type[pydantic.BaseModel],
        reference: Reference,
        on_delete: Mapping[ReferenceKey, OnDelete],
    ) -> _Declaration:
        """A foreign key to the referenced Entity's model, named without `_id`.

        Reverse accessors are off (`related_name='+'`), so several References to
        one Entity never clash.
        """
        name = reference.field.root.removesuffix("_id") or reference.field.root
        deleted = on_delete.get(
            ReferenceKey(entity=model, field=reference.field), OnDelete.PROTECT
        )
        arguments = {
            Identifier("on_delete"): ValueExpression(f"models.{deleted.value}"),
            Identifier("related_name"): ValueExpression(repr("+")),
        }
        if reference.optionality == Optionality.OPTIONAL:
            arguments[Identifier("null")] = ValueExpression("True")
        return _Declaration(
            name=Identifier(name),
            field=DjangoField.builtin(Identifier("ForeignKey"), arguments),
            positional=(ValueExpression(repr(reference.target.__name__)),),
            source=reference.field,
            attributes=(Identifier(name), Identifier(f"{name}_id")),
        )

    @staticmethod
    def _field_clashes(
        model: type[pydantic.BaseModel], declarations: Iterable[_Declaration]
    ) -> Iterator[DjangoFieldNameClash]:
        bound: dict[Identifier, list[FieldName]] = {}
        for declaration in declarations:
            for attribute in declaration.attributes:
                bound.setdefault(attribute, []).append(declaration.source)
        for name, fields in bound.items():
            if len(fields) > 1:
                yield DjangoFieldNameClash(
                    entity=model,
                    name=name,
                    fields=tuple(sorted(fields, key=lambda field: field.root)),
                )

    @staticmethod
    def _name_clashes(
        planned: list[_PlannedModel], imports: set[Symbol]
    ) -> Iterator[DjangoModelNameClash]:
        bound: dict[Identifier, list[Symbol | Identifier]] = {
            DjangoModelsRenderer.models_import: [DjangoModelsRenderer.models_import]
        }
        for symbol in {symbol.imported() for symbol in imports}:
            bound.setdefault(symbol.bound_name(), []).append(symbol)
        for model in planned:
            bound.setdefault(model.name, []).append(model.name)
        for name, origins in bound.items():
            if len(origins) > 1:
                yield DjangoModelNameClash(name=name)

    @staticmethod
    def _module(
        app_label: AppLabel, planned: list[_PlannedModel], imports: set[Symbol]
    ) -> FileText:
        lines = [
            f"from {symbol.module.root} import {symbol.bound_name().root}"
            for symbol in sorted(
                {symbol.imported() for symbol in imports},
                key=lambda symbol: (symbol.module.root, symbol.bound_name().root),
            )
        ]
        lines.append("from django.db import models")
        imported = "\n".join(sorted(lines))
        classes = "\n\n".join(
            f"class {model.name.root}(models.Model):\n"
            + "".join(declaration.line().root for declaration in model.declarations)
            + "\n    class Meta:\n"
            + f"        app_label = {app_label.root!r}\n"
            for model in planned
        )
        return FileText(
            f"{DjangoModelsRenderer.header.root}\n{imported}\n\n\n{classes}"
        )
