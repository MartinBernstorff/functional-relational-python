"""Renders the planned mapping functions: compiled Views as plain Python functions.

Each View becomes a function from one instance of its Source to its View model.
A filtered View's function returns `None` for a row its Filters exclude: the
ticket that introduced `where` specifies that it "returns nothing", so `None`
carries that meaning rather than an enum variant. A View that resolves References
also takes, per Entity it looks up, a keyword parameter mapping IDs to Entities; a
dangling ID raises `KeyError` there. A View that reverses a one-to-one Reference
takes a parameter mapping the IDs in that Reference to the Entities holding them;
a missing ID gives `None`. A
module of mapping functions imports only the Source, Entity and View model
modules, so it runs and type checks without this library installed.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two agree (the differential test in `test_mappings.py`
checks this).
"""

import typing
from collections.abc import Callable, Iterable, Sequence

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

    def exploded(
        self, opening: Expression, entries: Iterable[Expression], closing: Expression
    ) -> Expression:
        """`opening`, then each entry on its own line one level deeper with a
        trailing comma, then `closing` at this indentation; `opening` directly
        followed by `closing` when there are no entries.

        The trailing comma keeps ruff format from joining the lines.
        """
        pad = self.deeper().pad().root
        lines = "".join(f"{pad}{entry.root},\n" for entry in entries)
        if not lines:
            return Expression(f"{opening.root}{closing.root}")
        return Expression(f"{opening.root}\n{lines}{self.pad().root}{closing.root}")

    def looked_up(
        self,
        prefix: Expression,
        absent: Sequence[Expression],
        present: Callable[[Indent], Expression],
    ) -> Expression:
        """The value of an entry at this indentation that starts with `prefix`,
        e.g. `mentor=`, holding what a chain of lookups finds: `present` itself, or
        `None if <absent> else ...` for each optional lookup on the way, where
        `absent` is the condition under which it finds nothing, e.g.
        `source.mentor_id is None`.

        The conditionals stay on one line when `present` is one line and the
        entry fits, or else they fit parenthesized on a line of their own;
        otherwise they are split over lines like ruff format does, one chained
        conditional rather than nested ones. `present` gets the
        indentation to write at, and is written once: a one-line expression
        reads the same at any indentation.
        """
        if not absent:
            return present(self)
        inner = self.deeper()
        written = present(inner).root
        conditions = "".join(f"None if {condition.root} else " for condition in absent)
        inline = f"{conditions}{written}"
        line = f"{self.pad().root}{prefix.root}{inline},"
        pad = inner.pad().root
        if "\n" not in written:
            if len(line) <= LineLength.default().root:
                return Expression(inline)
            # Like ruff format: parenthesized onto its own line, if that fits.
            if len(f"{pad}{inline}") <= LineLength.default().root:
                return Expression(f"(\n{pad}{inline}\n{self.pad().root})")
        branches = "".join(
            f"{pad}{'None' if index == 0 else 'else None'}\n{pad}if {condition.root}\n"
            for index, condition in enumerate(absent)
        )
        return Expression(f"(\n{branches}{pad}else {written}\n{self.pad().root})")


class Followed(pydantic.BaseModel, frozen=True):
    """A chain of lookups followed from a subject in generated source: per optional
    lookup, in turn, the condition under which it finds nothing, and the Entity
    the chain ends at."""

    absent: tuple[Expression, ...]
    entity: Expression

    @staticmethod
    def of(
        lookups: Iterable[ir.EntityLookup],
        subject: Expression,
        entity: Callable[[ir.EntityLookup, Expression], Expression],
        absent: Callable[[ir.EntityLookup, Expression], Expression],
    ) -> Followed:
        """`entity` writes what a lookup finds from the expression before it, and
        `absent` the condition under which it finds nothing."""
        conditions: list[Expression] = []
        found = subject
        for lookup in lookups:
            match lookup.optionality:
                case Optionality.REQUIRED:
                    pass
                case Optionality.OPTIONAL:
                    conditions.append(absent(lookup, found))
            found = entity(lookup, found)
        return Followed(absent=tuple(conditions), entity=found)


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


class _Construction(pydantic.BaseModel, frozen=True):
    """Writes the call that builds a View model, at one indentation.

    The output is ruff-format stable: calls keep a trailing comma, so they stay
    exploded, and a conditional is split over lines only when it does not fit.
    """

    function: FunctionPlan
    indent: Indent

    def call(self, model: ModelPlan, subject: Expression) -> Expression:
        """`model(...)`, with a keyword argument per field, computed from
        `subject`."""
        inner = self.model_copy(update={"indent": self.indent.deeper()})
        return self.indent.exploded(
            Expression(f"{model.symbol.qualified().root}("),
            (
                Expression(f"{field.field.name.root}={value.root}")
                for field in model.fields
                if isinstance(
                    value := inner.value(model, field.field, subject), Expression
                )
            ),
            Expression(")"),
        )

    def value(
        self, model: ModelPlan, field: ir.RelationField, subject: Expression
    ) -> Expression | LeftOut:
        """The argument for `field` of `model`, on a line at this indentation."""
        prefix = Expression(f"{field.name.root}=")
        match field:
            case ir.NestedField(value=found):
                nested = model.nested_in(field)
                followed = self._followed(found.lookups(), subject)
                return self.indent.looked_up(
                    prefix,
                    followed.absent,
                    lambda indent: self.model_copy(update={"indent": indent}).call(
                        nested, followed.entity
                    ),
                )
            case ir.Field(value=value):
                pass
        match value:
            case ir.SourceField():
                return Expression(f"{subject.root}.{value.field.root}")
            case ir.ReverseLookup():
                access = self._access(value, subject)
                return Expression(f"{access.parameter.root}.get({access.key.root})")
            case ir.Lookup() | ir.ChainedLookup():
                followed = self._followed(value.lookups(), subject)
                return self.indent.looked_up(
                    prefix, followed.absent, lambda _: followed.entity
                )
            case ir.Lifted(through=through, field=read):
                followed = self._followed(through, subject)
                lifted = Expression(f"{followed.entity.root}.{read.field.root}")
                return self.indent.looked_up(prefix, followed.absent, lambda _: lifted)
            case ir.Added():
                return LeftOut()

    def _followed(
        self, lookups: Iterable[ir.EntityLookup], subject: Expression
    ) -> Followed:
        """`lookups` followed from `subject`: an optional Reference finds nothing
        when its ID is `None`, a reversed one when no Entity holds the ID."""

        def entity(lookup: ir.EntityLookup, found: Expression) -> Expression:
            access = self._access(lookup, found)
            return Expression(f"{access.parameter.root}[{access.key.root}]")

        def absent(lookup: ir.EntityLookup, found: Expression) -> Expression:
            access = self._access(lookup, found)
            match lookup:
                case ir.Lookup():
                    return Expression(f"{access.key.root} is None")
                case ir.ReverseLookup():
                    return Expression(
                        f"{access.key.root} not in {access.parameter.root}"
                    )

        return Followed.of(lookups, subject, entity, absent)

    def _access(self, lookup: ir.EntityLookup, subject: Expression) -> _Access:
        """The parameter `lookup` finds its Entity in, and the subject's value it
        is keyed by: the Reference's ID, or for a reversed Reference the subject's
        own ID."""
        match lookup:
            case ir.Lookup():
                field = lookup.reference.field
            case ir.ReverseLookup():
                field = EntityNode.primary_key
        return _Access(
            parameter=Expression(self.function.parameter_of(lookup).root),
            key=Expression(f"{subject.root}.{field.root}"),
        )


class _Access(pydantic.BaseModel, frozen=True):
    """How a mapping function reaches what a lookup finds: `parameter[key]`."""

    parameter: Expression
    key: Expression


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
