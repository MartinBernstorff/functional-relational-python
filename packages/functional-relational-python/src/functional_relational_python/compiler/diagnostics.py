import pydantic

from functional_relational_python.compiler import model_names
from functional_relational_python.entity_graph.entity_graph import (
    FieldName,
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
    """`resolve` selected a field that is not typed with an Id, optional or not."""

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


type CompileDiagnostic = (
    UnknownField
    | RenameOntoExistingField
    | model_names.InvalidFieldName
    | model_names.InvalidModelName
    | model_names.InvalidNestedModelName
    | NotAReference
    | UnresolvableReference
    | ResolvedNameCollision
    | SharedModelName
    | RetypeInView
    | RetypeNestedField
)
