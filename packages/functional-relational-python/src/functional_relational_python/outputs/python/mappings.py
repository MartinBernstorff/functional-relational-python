"""Renders the planned mapping functions: compiled Views as plain Python functions.

Each View becomes a function from one instance of its Source to its View model,
or to `None` for a row the View's Filters exclude.
A View that resolves References also takes, per Entity it looks up, a keyword
parameter mapping IDs to Entities; a dangling ID raises `KeyError` there. A
module of mapping functions imports only the Source, Entity and View model
modules, so it runs and type checks without this library installed.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two agree (the differential test in `test_mappings.py`
checks this).
"""

import keyword
import typing
from collections.abc import Callable, Sequence

import pydantic

from functional_relational_python.entity_graph.entity_graph import Optionality
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

    # `frp generate` does not write mapping functions, so the header does not
    # point to it.
    header: typing.ClassVar[FileText] = FileText(
        f"{GeneratedFiles.header.root}# Maps each View's Source to its View model.\n"
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
        returned = f"{model} | None" if function.filters else model
        call = _Construction(function=function, indent=Indent(4)).call(
            function.model, Expression(FunctionPlan.source_parameter.root)
        )
        writer = _FilterWriter(function)
        guards = "".join(writer.guard(kept).root for kept in function.filters)
        return FunctionText(
            f"def {function.symbol.bound_name().root}(\n"
            f"    {FunctionPlan.source_parameter.root}: "
            f"{function.source.qualified().root},\n"
            f"{keyword_only}"
            f") -> {returned}:\n"
            f"{guards}"
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

    def _entity(self, lookup: ir.Lookup, subject: Expression) -> Expression:
        parameter = self.function.parameter_of(lookup)
        return Expression(
            f"{parameter.root}[{subject.root}.{lookup.reference.field.root}]"
        )

    def _optional(
        self,
        field: ir.RelationField,
        lookup: ir.Lookup,
        subject: Expression,
        present: Callable[[_Construction], Expression],
    ) -> Expression:
        """`present` itself, or `None` when an optional Reference is absent.

        `present` gets the construction to write its expression at.
        """
        match field.optionality():
            case Optionality.REQUIRED:
                return present(self)
            case Optionality.OPTIONAL:
                pass
        reference = f"{subject.root}.{lookup.reference.field.root}"
        inline = f"None if {reference} is None else {present(self).root}"
        line = f"{self.indent.pad().root}{field.name.root}={inline},"
        if "\n" not in inline and len(line) <= LineLength.default().root:
            return Expression(inline)
        inner = _Construction(self.function, self.indent.deeper())
        pad = inner.indent.pad().root
        return Expression(
            f"(\n{pad}None\n{pad}if {reference} is None\n"
            f"{pad}else {present(inner).root}\n{self.indent.pad().root})"
        )


class _FilterWriter:
    """Writes the guards that return `None` for a row a Filter excludes.

    Like the calls, a condition is split over lines, at its `and`s and `or`s, only
    when it does not fit.
    """

    def __init__(self, function: FunctionPlan) -> None:
        self.function = function

    def guard(self, kept: ir.Filter) -> Expression:
        """`if not (<kept>): return None`, at the function body's indentation."""
        body = Indent(4)
        line = f"{body.pad().root}if not ({self.inline(kept).root}):"
        if len(line) <= LineLength.default().root:
            condition = line
        else:
            lines = "".join(
                f"{text.root}\n" for text in self._block(kept, body.deeper())
            )
            condition = f"{body.pad().root}if not (\n{lines}{body.pad().root}):"
        return Expression(f"{condition}\n{body.deeper().pad().root}return None\n")

    def inline(self, kept: ir.Filter) -> Expression:
        """`kept` on one line. A nested `and` or `or` is in parentheses."""
        match kept:
            case ir.FilterComparison(operator=operator, field=field, operand=operand):
                return Expression(
                    f"{self._field(field).root} {operator.value} "
                    f"{self._operand(operand).root}"
                )
            case ir.FilterIsNull(field=field):
                return Expression(f"{self._field(field).root} is None")
            case ir.FilterNot(term=ir.FilterIsNull(field=field)):
                return Expression(f"{self._field(field).root} is not None")
            case ir.FilterNot(term=term):
                return Expression(f"not ({self.inline(term).root})")
            case ir.FilterAnd(terms=terms) | ir.FilterOr(terms=terms):
                connective = f" {_FilterWriter._connective(kept).root} "
                return Expression(
                    connective.join(self._term(term).root for term in terms)
                )

    def _term(self, term: ir.Filter) -> Expression:
        match term:
            case ir.FilterAnd() | ir.FilterOr():
                return Expression(f"({self.inline(term).root})")
            case ir.FilterComparison() | ir.FilterIsNull() | ir.FilterNot():
                return self.inline(term)

    @staticmethod
    def _connective(kept: ir.FilterAnd | ir.FilterOr) -> Expression:
        match kept:
            case ir.FilterAnd():
                return Expression("and")
            case ir.FilterOr():
                return Expression("or")

    def _block(self, kept: ir.Filter, indent: Indent) -> tuple[Expression, ...]:
        """`kept` as lines at `indent`, inside parentheses: one line if it fits,
        else split at its `and`s or `or`s, or inside its `not`."""
        pad = indent.pad().root
        line = f"{pad}{self.inline(kept).root}"
        if len(line) <= LineLength.default().root:
            return (Expression(line),)
        match kept:
            case ir.FilterAnd(terms=terms) | ir.FilterOr(terms=terms):
                connective = _FilterWriter._connective(kept).root
                return tuple(
                    text
                    for index, term in enumerate(terms)
                    for text in self._split_term(
                        term,
                        indent,
                        Expression("" if index == 0 else f"{connective} "),
                    )
                )
            case ir.FilterComparison() | ir.FilterIsNull() | ir.FilterNot():
                return self._split_term(kept, indent, Expression(""))

    def _split_term(
        self, term: ir.Filter, indent: Indent, prefix: Expression
    ) -> tuple[Expression, ...]:
        """One term of a split condition, after `prefix`, e.g. `and `: on one line
        if it fits, else split."""
        start = f"{indent.pad().root}{prefix.root}"
        line = f"{start}{self._term(term).root}"
        match term:
            case _ if len(line) <= LineLength.default().root:
                return (Expression(line),)
            case ir.FilterAnd() | ir.FilterOr():
                return self._parenthesised(Expression(f"{start}("), term, indent)
            case ir.FilterNot(term=ir.FilterIsNull()) | ir.FilterIsNull():
                return (Expression(line),)
            case ir.FilterNot(term=negated):
                return self._parenthesised(Expression(f"{start}not ("), negated, indent)
            case ir.FilterComparison(operator=operator, field=field, operand=operand):
                return (
                    Expression(f"{start}{self._field(field).root}"),
                    *self._operand_lines(
                        operand, indent, Expression(f"{operator.value} ")
                    ),
                )

    def _operand_lines(
        self,
        operand: ir.SourceField | ir.FilterLiteral,
        indent: Indent,
        prefix: Expression,
    ) -> tuple[Expression, ...]:
        """`operand` after `prefix`: on one line if it fits, else a RootModel's
        call split around its root."""
        start = f"{indent.pad().root}{prefix.root}"
        line = f"{start}{self._operand(operand).root}"
        match operand:
            case ir.RootModelLiteral(instance=instance, root=root) if (
                len(line) > LineLength.default().root
            ):
                model = self.function.literal_symbol(type(instance)).qualified()
                return (
                    Expression(f"{start}{model.root}("),
                    *self._operand_lines(root, indent.deeper(), Expression("")),
                    Expression(f"{indent.pad().root})"),
                )
            case _:
                return (Expression(line),)

    def _parenthesised(
        self, opening: Expression, term: ir.Filter, indent: Indent
    ) -> tuple[Expression, ...]:
        """`opening`, which ends in `(`, then `term` split one level deeper, then
        `)`."""
        return (
            opening,
            *self._block(term, indent.deeper()),
            Expression(f"{indent.pad().root})"),
        )

    def _field(self, field: ir.SourceField) -> Expression:
        return Expression(f"{FunctionPlan.source_parameter.root}.{field.field.root}")

    def _operand(self, operand: ir.SourceField | ir.FilterLiteral) -> Expression:
        match operand:
            case ir.SourceField():
                return self._field(operand)
            case ir.ScalarLiteral(root=str() as text) if '"' not in text:
                # Double quotes, as ruff writes them.
                return Expression(f'"{repr(text)[1:-1]}"')
            case ir.ScalarLiteral(root=value):
                return Expression(repr(value))
            case ir.EnumLiteral(member=member):
                enum_class = self.function.literal_symbol(type(member)).qualified()
                name = member.name
                if name.isidentifier() and not keyword.iskeyword(name):
                    return Expression(f"{enum_class.root}.{name}")
                return Expression(f"{enum_class.root}[{name!r}]")
            case ir.RootModelLiteral(instance=instance, root=root):
                model = self.function.literal_symbol(type(instance)).qualified()
                return Expression(f"{model.root}({self._operand(root).root})")
