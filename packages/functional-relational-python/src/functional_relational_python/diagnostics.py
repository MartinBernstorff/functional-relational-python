import pydantic

from functional_relational_python import ir
from functional_relational_python.views import ViewName


class DiagnosticMessage(pydantic.RootModel[str], frozen=True):
    pass


class UnknownField(pydantic.BaseModel, frozen=True):
    """A step names a field the relation does not have at that point."""

    view: ViewName
    source: ir.SourceName
    field: ir.FieldName
    valid: tuple[ir.FieldName, ...]

    def message(self) -> DiagnosticMessage:
        valid = ", ".join(f"'{name.root}'" for name in self.valid)
        return DiagnosticMessage(
            f"{self.view.root}: {self.source.root} has no field "
            f"'{self.field.root}'. Valid fields: {valid}."
        )


type Diagnostic = UnknownField
