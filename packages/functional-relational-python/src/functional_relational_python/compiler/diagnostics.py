import pydantic

from functional_relational_python.compiler import model_names
from functional_relational_python.domain.entity_graph import (
    FieldName,
)
from functional_relational_python.domain.id import IdType
from functional_relational_python.language import ir
from functional_relational_python.language.views import ViewName
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


type CompileDiagnostic = (
    UnknownField
    | RenameOntoExistingField
    | model_names.InvalidFieldName
    | model_names.InvalidModelName
    | model_names.InvalidNestedModelName
    | model_names.InvalidFunctionName
    | NotAReference
    | UnresolvableReference
    | ResolvedNameCollision
)
