"""Renders the planned mapping functions: compiled Views as plain Python functions.

Each View becomes a function from one instance of its Source to its View model.
A View that resolves References also takes, per Entity it looks up, a keyword
parameter mapping IDs to Entities; a dangling ID raises `KeyError` there. A View
that reverses a one-to-one Reference takes a parameter mapping the IDs in that
Reference to the Entities holding them; a missing ID gives `None`. A
module of mapping functions imports only the Source, Entity and View model
modules, so it runs and type checks without this library installed.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two agree (the differential test in `test_mappings.py`
checks this).
"""

import typing
from collections.abc import Callable, Sequence

import pydantic

from functional_relational_python.entity_graph.entity_graph import (
    EntityNode,
    Optionality,
)
from functional_relational_python.outputs.python.plan import (
    FunctionPlan,
    ModelPlan,
    PythonPlan,
)
from functional_relational_python.outputs.python.symbols import (
    LineLength,
    ModulePath,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.support.filesystem import FileText
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)


class Expression(pydantic.RootModel[str], frozen=True):
    """A Python expression in generated source, e.g. `source.name`.

    The later lines of a multi-line expression are indented for where it is used.
    """


class LeftOut(pydantic.BaseModel, frozen=True):
    """A field the call leaves out, so the model fills it with its default."""


class FunctionText(pydantic.RootModel[str], frozen=True):
    """The source text of one mapping function."""


class Indent(pydantic.RootModel[int], frozen=True):
    """How many spaces a line of generated source starts with."""

    def pad(self) -> Expression:
        return Expression(" " * self.root)

    def deeper(self) -> Indent:
        return Indent(self.root + 4)


class MappingsRenderer:
    """Writes the source of every planned mapping function."""

    header: typing.ClassVar[FileText] = FileText(
        f"{GeneratedFiles.header.root}"
        "# Regenerate with `frp generate`; `frp check` fails when this is stale.\n"
        "# Maps each View's Source to its View model.\n"
    )

    @staticmethod
    def files(plan: PythonPlan) -> tuple[GeneratedFile, ...]:
        """One file per module, in module order, with its functions in View order.

        `render.PythonRenderer` writes them together with the View models.
        """
        return tuple(
            GeneratedFile(
                path=module.file_path(),
                text=MappingsRenderer._module(plan.functions[module]),
            )
            for module in sorted(plan.functions, key=lambda module: module.root)
        )

    @staticmethod
    def _module(functions: Sequence[FunctionPlan]) -> FileText:
        imports = "\n".join(
            "".join(f"import {module.root}\n" for module in section)
            for section in ModulePath.import_sections(
                module for function in functions for module in function.imports()
            )
        )
        rendered = "".join(
            f"\n\n{MappingsRenderer._function(function).root}" for function in functions
        )
        return FileText(f"{MappingsRenderer.header.root}\n{imports}{rendered}")

    @staticmethod
    def _function(function: FunctionPlan) -> FunctionText:
        lookups = "".join(
            f"    {parameter.name.root}: {FunctionPlan.mapping.root}.Mapping[\n"
            f"        {parameter.id_type.qualified().root},\n"
            f"        {parameter.target.qualified().root},\n"
            "    ],\n"
            for parameter in function.lookups
        )
        keyword_only = f"    *,\n{lookups}" if lookups else ""
        model = function.model.symbol.qualified().root
        call = _Construction(function=function, indent=Indent(4)).call(
            function.model, Expression(FunctionPlan.source_parameter.root)
        )
        return FunctionText(
            f"def {function.symbol.bound_name().root}(\n"
            f"    {FunctionPlan.source_parameter.root}: "
            f"{function.source.qualified().root},\n"
            f"{keyword_only}"
            f") -> {model}:\n"
            f"    return {call.root}\n"
        )


class _Construction:
    """Writes the call that builds a View model, at one indentation.

    The output is ruff-format stable: calls keep a trailing comma, so they stay
    exploded, and a conditional is split over lines only when it does not fit.
    """

    def __init__(self, function: FunctionPlan, indent: Indent) -> None:
        self.function = function
        self.indent = indent

    def call(self, model: ModelPlan, subject: Expression) -> Expression:
        """`model(...)`, with a keyword argument per field, computed from
        `subject`."""
        inner = _Construction(self.function, self.indent.deeper())
        arguments = "".join(
            f"{inner.indent.pad().root}{field.field.name.root}={value.root},\n"
            for field in model.fields
            if isinstance(value := inner.value(model, field.field, subject), Expression)
        )
        name = model.symbol.qualified().root
        if not arguments:
            return Expression(f"{name}()")
        return Expression(f"{name}(\n{arguments}{self.indent.pad().root})")

    def value(
        self, model: ModelPlan, field: ir.RelationField, subject: Expression
    ) -> Expression | LeftOut:
        """The argument for `field` of `model`, on a line at this indentation."""
        match field:
            case ir.Field(value=value):
                match value:
                    case ir.SourceField():
                        return Expression(f"{subject.root}.{value.field.root}")
                    case ir.Lookup():
                        entity = self._entity(value, subject)
                        return self._optional(field, value, subject, lambda _: entity)
                    case ir.ReverseLookup():
                        parameter = self.function.parameter_of(value).root
                        key = self._key(value, subject).root
                        return Expression(f"{parameter}.get({key})")
                    case ir.Added():
                        return LeftOut()
            case ir.NestedField(value=lookup):
                nested = model.nested_in(field)
                return self._optional(
                    field,
                    lookup,
                    subject,
                    lambda construction: construction.call(
                        nested, self._entity(lookup, subject)
                    ),
                )

    def _key(self, lookup: ir.EntityLookup, subject: Expression) -> Expression:
        """The subject's value `lookup` finds its Entity by: the Reference's ID,
        or for a reversed Reference the subject's own ID."""
        match lookup:
            case ir.Lookup():
                field = lookup.reference.field
            case ir.ReverseLookup():
                field = EntityNode.primary_key
        return Expression(f"{subject.root}.{field.root}")

    def _entity(self, lookup: ir.EntityLookup, subject: Expression) -> Expression:
        parameter = self.function.parameter_of(lookup)
        return Expression(f"{parameter.root}[{self._key(lookup, subject).root}]")

    def _optional(
        self,
        field: ir.RelationField,
        lookup: ir.EntityLookup,
        subject: Expression,
        present: Callable[[_Construction], Expression],
    ) -> Expression:
        """`present` itself, or `None` when an optional Reference is absent, or no
        Entity holds a reversed one.

        `present` gets the construction to write its expression at.
        """
        match field.optionality():
            case Optionality.REQUIRED:
                return present(self)
            case Optionality.OPTIONAL:
                pass
        key = self._key(lookup, subject).root
        match lookup:
            case ir.Lookup():
                absent = f"{key} is None"
            case ir.ReverseLookup():
                absent = f"{key} not in {self.function.parameter_of(lookup).root}"
        inline = f"None if {absent} else {present(self).root}"
        line = f"{self.indent.pad().root}{field.name.root}={inline},"
        if "\n" not in inline and len(line) <= LineLength.default().root:
            return Expression(inline)
        inner = _Construction(self.function, self.indent.deeper())
        pad = inner.indent.pad().root
        return Expression(
            f"(\n{pad}None\n{pad}if {absent}\n"
            f"{pad}else {present(inner).root}\n{self.indent.pad().root})"
        )
