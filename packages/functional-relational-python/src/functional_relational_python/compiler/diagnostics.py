import pydantic

from functional_relational_python.compiler import model_names
from functional_relational_python.entity_graph.entity_graph import (
    FieldName,
)
from functional_relational_python.entity_graph.id import IdType
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.views import ViewName
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
)


class UnknownField(pydantic.BaseModel, Diagnostic, frozen=True):
    """A step names a field the relation does not have at that point."""

    view: ViewName
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

    view: ViewName
    renamed: FieldName
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot rename '{self.renamed.root}' to "
            f"'{self.field.root}', because the View already has a field "
            f"'{self.field.root}'."
        )


class NotAReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`resolve` selected a field that is not typed with an Id, optional or not."""

    view: ViewName
    source: ir.SourceName
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot resolve {self.source.root}.{self.field.root}, "
            "because it is not a Reference."
        )


class UnresolvableReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`resolve` selected a Reference whose ID type keys no Entity."""

    view: ViewName
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

    view: ViewName
    source: ir.SourceName
    field: FieldName
    existing: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: resolving {self.source.root}.{self.field.root} "
            f"would name it '{self.existing.root}', but {self.source.root} already "
            f"has a field '{self.existing.root}'."
        )


class DefaultText(pydantic.RootModel[str], frozen=True):
    """How a default's value reads in a diagnostic: its `repr`."""

    @staticmethod
    def of(value: ir.UncheckedDefault) -> DefaultText:
        return DefaultText(repr(value))


class NonLiteralDefault(pydantic.BaseModel, Diagnostic, frozen=True):
    """`default` got a value that is not a literal a generated model can write out."""

    view: ViewName
    value: DefaultText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: cannot use {self.value.root} as a default. A default "
            "is None, MISSING, a finite number, a string, a boolean, an empty list "
            "or dict, or an enum member."
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
    | NonLiteralDefault
)
