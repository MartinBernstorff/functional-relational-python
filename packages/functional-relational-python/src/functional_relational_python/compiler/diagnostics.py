import pydantic

from functional_relational_python.compiler import model_names
from functional_relational_python.entity_graph.entity_graph import (
    FieldName,
    RuntimeAnnotation,
)
from functional_relational_python.entity_graph.id import IdType
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.payloads import DefinitionName
from functional_relational_python.pipeline.views import ViewName
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
)


class UnknownField(pydantic.BaseModel, Diagnostic, frozen=True):
    """A step names a field the relation does not have at that point."""

    view: DefinitionName
    source: ir.SourceName
    field: FieldName
    valid: tuple[FieldName, ...]

    def describe(self) -> DiagnosticMessage:
        valid = ", ".join(f"'{name.root}'" for name in self.valid)
        return DiagnosticMessage(
            f"{self.view.root}: {self.source.root} has no field "
            f"'{self.field.root}'. Valid fields: {valid}."
        )


class RenameOntoExistingField(pydantic.BaseModel, Diagnostic, frozen=True):
    """A rename's new name is a field the relation already has at that point."""

    view: DefinitionName
    renamed: FieldName
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot rename '{self.renamed.root}' to "
            f"'{self.field.root}', because it already has a field "
            f"'{self.field.root}'."
        )


class NotAReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`resolve` selected a field that is not typed with an ID type, optional or
    not."""

    view: DefinitionName
    source: ir.SourceName
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot resolve {self.source.root}.{self.field.root}, "
            "because it is not a Reference."
        )


class UnresolvableReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`resolve` selected a Reference whose ID type keys no Entity."""

    view: DefinitionName
    source: ir.SourceName
    field: FieldName
    id_type: IdType

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot resolve {self.source.root}.{self.field.root}, "
            f"because {self.id_type.__qualname__} is the primary key of no entity."
        )


class ResolvedNameCollision(pydantic.BaseModel, Diagnostic, frozen=True):
    """A resolved field's new name is a field the relation already has."""

    view: DefinitionName
    source: ir.SourceName
    field: FieldName
    existing: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: resolving {self.source.root}.{self.field.root} "
            f"would name it '{self.existing.root}', but {self.source.root} already "
            f"has a field '{self.existing.root}'."
        )


class ReverseFromNonEntity(pydantic.BaseModel, Diagnostic, frozen=True):
    """`reverse` was applied to a subject that is not an Entity, so no Reference
    can point to it."""

    view: DefinitionName
    source: ir.SourceName
    entity: type[pydantic.BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot reverse {self.entity.__qualname__} onto "
            f"{self.source.root}, because {self.source.root} is not an entity."
        )


class ReverseOfNonEntity(pydantic.BaseModel, Diagnostic, frozen=True):
    """`reverse` named a model that is not an Entity in the EntityGraph."""

    view: DefinitionName
    source: ir.SourceName
    entity: type[pydantic.BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot reverse {self.entity.__qualname__} onto "
            f"{self.source.root}, because {self.entity.__qualname__} is not an "
            "entity."
        )


class NoOneToOneReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`reverse` named an Entity with no one-to-one Reference to the subject.

    `many_to_one` are its References to the subject that are not marked
    `OneToOne`; following one backwards would give several Entities.
    """

    view: DefinitionName
    source: ir.SourceName
    entity: type[pydantic.BaseModel]
    many_to_one: tuple[FieldName, ...]

    def describe(self) -> DiagnosticMessage:
        message = (
            f"{self.view.root}: cannot reverse {self.entity.__qualname__} onto "
            f"{self.source.root}, because it has no one-to-one Reference to "
            f"{self.source.root}."
        )
        if self.many_to_one:
            fields = ", ".join(f"'{field.root}'" for field in self.many_to_one)
            message += f" Its References {fields} are not marked OneToOne."
        return DiagnosticMessage(message)


class AmbiguousReverse(pydantic.BaseModel, Diagnostic, frozen=True):
    """`reverse` named an Entity with several one-to-one References to the subject,
    without choosing one with `reverse_via`."""

    view: DefinitionName
    source: ir.SourceName
    entity: type[pydantic.BaseModel]
    references: tuple[FieldName, ...]

    def describe(self) -> DiagnosticMessage:
        fields = ", ".join(f"'{field.root}'" for field in self.references)
        return DiagnosticMessage(
            f"{self.view.root}: cannot reverse {self.entity.__qualname__} onto "
            f"{self.source.root}, because each of {fields} is a one-to-one "
            f"Reference to {self.source.root}. Choose one with reverse_via."
        )


class NotAOneToOneReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`reverse_via` named a field that is not one of the Entity's one-to-one
    References to the subject. `valid` are those it has."""

    view: DefinitionName
    source: ir.SourceName
    entity: type[pydantic.BaseModel]
    field: FieldName
    valid: tuple[FieldName, ...]

    def describe(self) -> DiagnosticMessage:
        valid = ", ".join(f"'{field.root}'" for field in self.valid)
        return DiagnosticMessage(
            f"{self.view.root}: cannot reverse {self.entity.__qualname__}."
            f"{self.field.root} onto {self.source.root}, because it is not a "
            f"one-to-one Reference to {self.source.root}. Valid fields: {valid}."
        )


class ReverseOntoExistingField(pydantic.BaseModel, Diagnostic, frozen=True):
    """A reversed field's name is a field the relation already has."""

    view: DefinitionName
    source: ir.SourceName
    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: reversing {self.entity.__qualname__} would name it "
            f"'{self.field.root}', but {self.source.root} already has a field "
            f"'{self.field.root}'."
        )


class SharedModelName(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View and a Payload have the same name, so their models would too.

    Reported once per name, however many Views and Payloads share it.
    """

    view: DefinitionName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: a View and a Payload are both named "
            f"'{self.view.root}'. Model names must be unique across Views and "
            "Payloads."
        )


class RetypeInView(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View's pipeline retypes a field. Its mapping copies each value unchanged,
    so it cannot convert one to another type."""

    view: ViewName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: a View cannot retype its fields, because its "
            "mapping copies each value unchanged. Retype in a Payload instead."
        )


class RetypeNestedField(pydantic.BaseModel, Diagnostic, frozen=True):
    """`retype` selected a resolved field typed by the model of a nested pipeline.

    That model is generated, so there is no type to give `transform`.
    """

    view: DefinitionName
    source: ir.SourceName
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot retype {self.source.root}.{self.field.root}, "
            "because its type is the model of a nested pipeline."
        )


class BaseFieldTypeMismatch(pydantic.BaseModel, Diagnostic, frozen=True):
    """A kept field's type differs from the type the definition's base declares
    for it, so the generated model could not inherit it."""

    view: DefinitionName
    base: type[pydantic.BaseModel]
    field: FieldName
    declared: RuntimeAnnotation
    kept: RuntimeAnnotation

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: the base {self.base.__qualname__} declares "
            f"'{self.field.root}' as {self.declared!r}, but the kept field is "
            f"{self.kept!r}."
        )


class BaseFieldIsNestedModel(pydantic.BaseModel, Diagnostic, frozen=True):
    """A kept field the definition's base declares is a resolved field typed by a
    nested generated model, which the base cannot declare."""

    view: DefinitionName
    base: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: the base {self.base.__qualname__} declares "
            f"'{self.field.root}', but the kept field is typed by a nested "
            "generated model."
        )


class DefaultText(pydantic.RootModel[str], frozen=True):
    """How a default's value reads in a diagnostic: its `repr`."""

    @staticmethod
    def of(value: ir.UncheckedDefault) -> DefaultText:
        return DefaultText(repr(value))


class NonLiteralDefault(pydantic.BaseModel, Diagnostic, frozen=True):
    """`default` got a value that is not a literal a generated model can write out."""

    view: DefinitionName
    value: DefaultText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot use {self.value.root} as a default. A default "
            "is None, MISSING, a finite number, a string, a boolean, an empty list "
            "or dict, or an enum member."
        )


class AddOntoExistingField(pydantic.BaseModel, Diagnostic, frozen=True):
    """`add` named a field the relation already has at that point."""

    view: DefinitionName
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot add '{self.field.root}', because it already "
            f"has a field '{self.field.root}'."
        )


class AddWithoutDefault(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View's `add` gave no default, so its mapping has nothing to fill the
    field with."""

    view: ViewName
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot add '{self.field.root}' without a default, "
            "because a View's mapping fills an added field with its default."
        )


type CompileDiagnostic = (
    UnknownField
    | RenameOntoExistingField
    | model_names.InvalidFieldName
    | model_names.InvalidModelName
    | model_names.InvalidNestedModelName
    | NotAReference
    | UnresolvableReference
    | ResolvedNameCollision
    | ReverseFromNonEntity
    | ReverseOfNonEntity
    | NoOneToOneReference
    | AmbiguousReverse
    | NotAOneToOneReference
    | ReverseOntoExistingField
    | NonLiteralDefault
    | AddOntoExistingField
    | AddWithoutDefault
    | SharedModelName
    | RetypeInView
    | RetypeNestedField
    | BaseFieldTypeMismatch
    | BaseFieldIsNestedModel
)
