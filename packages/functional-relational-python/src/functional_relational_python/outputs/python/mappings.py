"""Renders the planned mapping functions: compiled Views as plain Python functions.

Each View becomes a function from one instance of its Source to its View model.
A filtered View's function returns `None` for a row its Filters exclude: the
ticket that introduced `where` specifies that it "returns nothing", so `None`
carries that meaning rather than an enum variant. A View that resolves References also takes, per Entity it looks up, a keyword
parameter mapping IDs to Entities; a dangling ID raises `KeyError` there. A
module of mapping functions imports only the Source, Entity and View model
modules, so it runs and type checks without this library installed.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two agree (the differential test in `test_mappings.py`
checks this).
"""

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
    PythonString,
    ValueExpression,
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
        returned = f"{model} | None" if function.filters else model
        if function.filters and len(f") -> {returned}:") > LineLength.default().root:
            # As ruff splits a return annotation that does not fit: in parentheses,
            # then at the `|` if it still does not fit.
            inner = f"    {returned}"
            returned = (
                f"(\n{inner}\n)"
                if len(inner) <= LineLength.default().root
                else f"(\n    {model}\n    | None\n)"
            )
        call = _Construction(function=function, indent=Indent(4)).call(
            function.model, Expression(FunctionPlan.source_parameter.root)
        )
        writer = _FilterWriter(function=function)
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


class _RenderedFilter(pydantic.BaseModel, frozen=True):
    """A Filter with its text on one line, and its terms rendered likewise.

    Each subtree's text is written once, however its lines are split later.
    """

    kept: ir.Filter
    inline: Expression
    terms: tuple[_RenderedFilter, ...]

    def as_term(self) -> Expression:
        """The text as a term of an `and` or `or`: in parentheses if it is one."""
        match self.kept:
            case ir.FilterAnd() | ir.FilterOr():
                return Expression(f"({self.inline.root})")
            case ir.FilterComparison() | ir.FilterIsNull() | ir.FilterNot():
                return self.inline


class _FilterWriter(pydantic.BaseModel, frozen=True):
    """Writes the guards that return `None` for a row a Filter excludes.

    Like the calls, a condition is split over lines only when it does not fit, as
    ruff splits it: at its `and`s and `or`s, inside its parentheses, before a
    comparison's operator, and around a RootModel's root.
    """

    function: FunctionPlan

    def guard(self, kept: ir.Filter) -> Expression:
        """`if not (<kept>): return None`, at the function body's indentation."""
        body = Indent(4)
        rendered = self._rendered(kept)
        line = f"{body.pad().root}if not ({rendered.inline.root}):"
        if len(line) <= LineLength.default().root:
            condition = line
        else:
            lines = "".join(
                f"{text.root}\n" for text in self._block(rendered, body.deeper())
            )
            condition = f"{body.pad().root}if not (\n{lines}{body.pad().root}):"
        return Expression(f"{condition}\n{body.deeper().pad().root}return None\n")

    def _rendered(self, kept: ir.Filter) -> _RenderedFilter:
        terms: tuple[_RenderedFilter, ...]
        match kept:
            case ir.FilterComparison(operator=operator, field=field, operand=operand):
                terms = ()
                inline = (
                    f"{self._field(field).root} {operator.value} "
                    f"{self._operand(operand).root}"
                )
            case ir.FilterIsNull(field=field):
                terms = ()
                inline = f"{self._field(field).root} is None"
            case ir.FilterNot(term=ir.FilterIsNull(field=field)):
                terms = ()
                inline = f"{self._field(field).root} is not None"
            case ir.FilterNot(term=term):
                terms = (self._rendered(term),)
                inline = f"not ({terms[0].inline.root})"
            case ir.FilterAnd(terms=connected) | ir.FilterOr(terms=connected):
                terms = tuple(self._rendered(term) for term in connected)
                connective = f" {_FilterWriter._connective(kept).root} "
                inline = connective.join(term.as_term().root for term in terms)
        return _RenderedFilter(kept=kept, inline=Expression(inline), terms=terms)

    @staticmethod
    def _connective(kept: ir.FilterAnd | ir.FilterOr) -> Expression:
        match kept:
            case ir.FilterAnd():
                return Expression("and")
            case ir.FilterOr():
                return Expression("or")

    def _block(
        self, rendered: _RenderedFilter, indent: Indent
    ) -> tuple[Expression, ...]:
        """`rendered` as lines at `indent`, inside parentheses: one line if it
        fits, else split at its `and`s or `or`s, or as a single term."""
        line = f"{indent.pad().root}{rendered.inline.root}"
        if len(line) <= LineLength.default().root:
            return (Expression(line),)
        match rendered.kept:
            case ir.FilterAnd() | ir.FilterOr() as connected:
                connective = _FilterWriter._connective(connected).root
                return tuple(
                    text
                    for index, term in enumerate(rendered.terms)
                    for text in self._split_term(
                        term,
                        indent,
                        Expression("" if index == 0 else f"{connective} "),
                    )
                )
            case ir.FilterComparison() | ir.FilterIsNull() | ir.FilterNot():
                return self._split_term(rendered, indent, Expression(""))

    def _split_term(
        self, rendered: _RenderedFilter, indent: Indent, prefix: Expression
    ) -> tuple[Expression, ...]:
        """One term of a split condition, after `prefix`, e.g. `and `: on one line
        if it fits, else split."""
        start = f"{indent.pad().root}{prefix.root}"
        line = f"{start}{rendered.as_term().root}"
        match rendered.kept:
            case _ if len(line) <= LineLength.default().root:
                return (Expression(line),)
            case ir.FilterNot(term=ir.FilterIsNull()) | ir.FilterIsNull():
                return (Expression(line),)
            case ir.FilterAnd() | ir.FilterOr():
                return self._parenthesised(Expression(f"{start}("), rendered, indent)
            case ir.FilterNot():
                (negated,) = rendered.terms
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
        self, opening: Expression, rendered: _RenderedFilter, indent: Indent
    ) -> tuple[Expression, ...]:
        """`opening`, which ends in `(`, then `rendered` split one level deeper,
        then `)`."""
        return (
            opening,
            *self._block(rendered, indent.deeper()),
            Expression(f"{indent.pad().root})"),
        )

    def _field(self, field: ir.SourceField) -> Expression:
        return Expression(f"{FunctionPlan.source_parameter.root}.{field.field.root}")

    def _operand(self, operand: ir.SourceField | ir.FilterLiteral) -> Expression:
        match operand:
            case ir.SourceField():
                return self._field(operand)
            case ir.ScalarLiteral(root=str() as text):
                return Expression(PythonString(text).literal().root)
            case ir.ScalarLiteral(root=value):
                return Expression(repr(value))
            case ir.EnumLiteral(member=member):
                enum_class = self.function.literal_symbol(type(member)).qualified()
                return Expression(
                    ValueExpression.of_enum_member(enum_class, member).root
                )
            case ir.RootModelLiteral(instance=instance, root=root):
                model = self.function.literal_symbol(type(instance)).qualified()
                return Expression(f"{model.root}({self._operand(root).root})")
