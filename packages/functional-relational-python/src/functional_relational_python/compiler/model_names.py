"""The names a generated Pydantic model, and the code around it, can use.

Generating Pydantic models is the product, so whether a name works as a model
name, a model field name or a mapping function name is checked here, once, while
compiling. Outputs receive these names already valid, so building identifiers from
them cannot fail.

Clashes that depend on where the output puts things (two models of one name in a
module, a model named like an import) are not checked here: the outputs own them.
"""

import keyword
import re
from collections.abc import Mapping

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.domain.entity_graph import FieldName
from functional_relational_python.language import ir
from functional_relational_python.language.views import ViewName
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)
from functional_relational_python.support.identifier import Identifier


class InvalidModelName(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View's name cannot be used as a class name."""

    view: ViewName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: the View name is not a valid Python class name."
        )


class CandidateName(pydantic.RootModel[str], frozen=True):
    """A name that was derived for something, before it is known to be valid."""


class InvalidNestedModelName(pydantic.BaseModel, Diagnostic, frozen=True):
    """The model of a resolved field with a nested pipeline would get a name that
    is not a valid class name."""

    view: ViewName
    field: FieldName
    name: CandidateName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}.{self.field.root}: the nested model would be named "
            f"'{self.name.root}', which is not a valid Python class name."
        )


class InvalidFieldName(pydantic.BaseModel, Diagnostic, frozen=True):
    """A rename or resolve gives a field a name no View model field can have."""

    view: ViewName
    source: ir.SourceName
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: '{self.field.root}' cannot be a field name. A field "
            "name is a Python identifier that is not a keyword, does not start with "
            "'_', and is not an attribute of pydantic.BaseModel."
        )


type ModelNameDiagnostic = InvalidModelName | InvalidNestedModelName


class ModelName(pydantic.RootModel[str], frozen=True):
    """A valid class name for a generated model, e.g. `CompanySummary`."""

    @pydantic.field_validator("root")
    @classmethod
    def _identifier(cls, root: str) -> str:
        return Identifier(root).root

    @staticmethod
    def fake() -> ModelName:
        return ModelName("CompanySummary")

    def identifier(self) -> Identifier:
        """The name as a plain Identifier, e.g. for a Symbol's qualified name."""
        return Identifier(self.root)


class FunctionName(pydantic.RootModel[str], frozen=True):
    """A valid name for the function mapping a View's Source to its model."""

    @pydantic.field_validator("root")
    @classmethod
    def _identifier(cls, root: str) -> str:
        return Identifier(root).root

    @staticmethod
    def fake() -> FunctionName:
        return FunctionName("company_summary")

    @staticmethod
    def of(model: ModelName) -> FunctionName:
        """The model's name in snake_case: `CompanyPublic` is `company_public`.

        A keyword gets a trailing underscore, as PEP 8 suggests: `Class` is
        `class_`.
        """
        snake = PascalCase(model.root).snake_case().root
        return FunctionName(f"{snake}_" if keyword.iskeyword(snake) else snake)


class PascalCase(pydantic.RootModel[str], frozen=True):
    """A class-style name, such as a View's or an Entity's."""

    def snake_case(self) -> CandidateName:
        """`CompanyPublic` is `company_public`."""
        return CandidateName(re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", self.root).lower())


class ModelNames(pydantic.BaseModel, frozen=True):
    """A generated model's name, and the names of the models nested in it.

    `nested` has an entry for each NestedField of the model's Relation.
    """

    name: ModelName
    nested: Mapping[FieldName, ModelNames]

    def nested_in(self, field: ir.NestedField) -> ModelNames:
        return self.nested[field.name]

    @staticmethod
    def nested_name(parent: ModelName, field: FieldName) -> CandidateName:
        """The model a nested resolved field is typed by: `EmployeeDepartment` for
        the field `department` of `Employee`."""
        pascal = "".join(part[:1].upper() + part[1:] for part in field.root.split("_"))
        return CandidateName(f"{parent.root}{pascal}")

    @staticmethod
    def of(
        view: ViewName, name: ModelName, relation: ir.Relation
    ) -> Result[ModelNames, DiagnosticsError[InvalidNestedModelName]]:
        """The names of `relation`'s model, called `name`, and its nested models."""
        nested: dict[FieldName, ModelNames] = {}
        reported: list[InvalidNestedModelName] = []
        for field in relation.fields:
            match field:
                case ir.Field():
                    continue
                case ir.NestedField(relation=nested_relation):
                    pass
            candidate = ModelNames.nested_name(name, field.name)
            match Identifier.of(candidate):
                case Ok():
                    pass
                case Err():
                    reported.append(
                        InvalidNestedModelName(
                            view=view, field=field.name, name=candidate
                        )
                    )
                    continue
            match ModelNames.of(view, ModelName(candidate.root), nested_relation):
                case Ok(names):
                    nested[field.name] = names
                case Err(error):
                    reported.extend(error.diagnostics)
        return DiagnosticsError.or_ok(
            reported, lambda: ModelNames(name=name, nested=nested)
        )


class ViewModelNames(pydantic.BaseModel, frozen=True):
    """Every name the code generated for one View uses: its models' and its
    mapping function's."""

    model: ModelNames
    function: FunctionName

    @staticmethod
    def check(
        view: ViewName, relation: ir.Relation
    ) -> Result[ViewModelNames, DiagnosticsError[ModelNameDiagnostic]]:
        """The names of the code generated for `view`, whose Relation is
        `relation`, or every reason they cannot be used."""
        match ViewModelNames.model_name(view):
            case Err(error):
                return Err(DiagnosticsError(error.diagnostics))
            case Ok(model):
                pass
        match ModelNames.of(view, model, relation):
            case Err(error):
                return Err(DiagnosticsError(error.diagnostics))
            case Ok(names):
                return Ok(ViewModelNames(model=names, function=FunctionName.of(model)))

    @staticmethod
    def view_diagnostics(view: ViewName) -> tuple[InvalidModelName, ...]:
        """What is wrong with the names taken from the View's name alone.

        For a View that did not compile, whose nested models are unknown.
        """
        match ViewModelNames.model_name(view):
            case Err(error):
                return error.diagnostics
            case Ok(_):
                return ()

    @staticmethod
    def model_name(
        view: ViewName,
    ) -> Result[ModelName, DiagnosticsError[InvalidModelName]]:
        """The name of the View's model: the View's name. Every rule a View's name
        must follow is checked here."""
        reported: list[InvalidModelName] = []
        if isinstance(Identifier.of(view), Err):
            reported.append(InvalidModelName(view=view))
        return DiagnosticsError.or_ok(reported, lambda: ModelName(view.root))

    @staticmethod
    def check_field(
        view: ViewName, source: ir.SourceName, name: FieldName
    ) -> tuple[InvalidFieldName, ...]:
        """Reports `name` unless a View model can have a field of that name."""
        if (
            isinstance(Identifier.of(name), Ok)
            and not name.root.startswith("_")
            and not hasattr(pydantic.BaseModel, name.root)
        ):
            return ()
        return (InvalidFieldName(view=view, source=source, field=name),)
