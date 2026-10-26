"""Renders the planned mapping functions: compiled Views as plain Python functions.

Each View becomes a function from one instance of its Source to its View model.
A filtered View's function returns `None` for a row its Filters exclude: the
ticket that introduced `where` specifies that it "returns nothing", so `None`
carries that meaning rather than an enum variant. A View that resolves References
also takes, per Entity it looks up, a keyword parameter mapping IDs to Entities; a
dangling ID raises `KeyError` there. A View that reverses a one-to-one Reference
takes a parameter mapping the IDs in that Reference to the Entities holding them;
a missing ID gives `None`. A value the View requires goes through a function
`_required` the module defines, which raises `ValueError` naming the subject's
field and ID when the value is `None`. A module of mapping functions imports only
the Source, Entity and View model modules, so it runs and type checks without
this library installed.

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
        inner = self.deeper()
        pad = inner.pad().root
        lines = "".join(
            f"{pad}{inner.fitted(entry, Expression(',')).root},\n" for entry in entries
        )
        if not lines:
            return Expression(f"{opening.root}{closing.root}")
        return Expression(f"{opening.root}\n{lines}{self.pad().root}{closing.root}")

    def fitted(self, entry: Expression, suffix: Expression) -> Expression:
        """`entry`, on a line at this indentation followed by `suffix`, split as
        ruff format splits a line that does not fit: inside its last brackets,
        their contents on one line one level deeper if they fit, else one item
        per line with a trailing comma, each split likewise."""
        text = entry.root
        if (
            "\n" in text
            or len(f"{self.pad().root}{text}{suffix.root}") <= LineLength.default().root
        ):
            return entry
        match _Bracketed.of(entry):
            case _Unbracketed():
                return entry
            case _Bracketed(head=head, body=body, tail=tail):
                pass
        inner = self.deeper()
        pad = inner.pad().root
        items = _Bracketed.items(body)
        if len(items) > 1 and len(f"{pad}{body.root}") > LineLength.default().root:
            lines = "".join(
                f"{pad}{inner.fitted(item, Expression(',')).root},\n" for item in items
            )
        else:
            lines = f"{pad}{inner.fitted(body, Expression('')).root}\n"
        return Expression(f"{head.root}\n{lines}{self.pad().root}{tail.root}")

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
            f"{pad}{'None' if index == 0 else 'else None'}\n"
            f"{pad}{inner.condition(condition).root}\n"
            for index, condition in enumerate(absent)
        )
        otherwise = inner.fitted(Expression(f"else {written}"), Expression(""))
        return Expression(f"(\n{branches}{pad}{otherwise.root}\n{self.pad().root})")

    def condition(self, absent: Expression) -> Expression:
        """`if <absent>` on a line at this indentation, split as ruff format
        splits one that does not fit: before its comparison, then its left
        operand as `fitted` splits it."""
        line = Expression(f"if {absent.root}")
        if len(f"{self.pad().root}{line.root}") <= LineLength.default().root:
            return line
        for operator in (" is None", " not in "):
            left, found, right = absent.root.rpartition(operator)
            if found:
                operand = self.fitted(Expression(f"if {left}"), Expression(""))
                compared = f"{operator.strip()} {right}".strip()
                return Expression(f"{operand.root}\n{self.pad().root}{compared}")
        return line


class _Unbracketed(pydantic.BaseModel, frozen=True):
    """A one-line expression without brackets outside its string literals."""


class _Depth(pydantic.RootModel[int], frozen=True):
    """How many brackets are open before a character of a one-line expression,
    outside its string literals."""


class _Bracketed(pydantic.BaseModel, frozen=True):
    """A one-line expression around its last top-level brackets: `head` up to
    and including the opening bracket, `body` inside them, and `tail` from the
    closing bracket on."""

    head: Expression
    body: Expression
    tail: Expression

    @staticmethod
    def of(text: Expression) -> _Bracketed | _Unbracketed:
        depths = _Bracketed._depths(text)
        closings = [
            index
            for index, character in enumerate(text.root)
            if character in ")]}" and depths[index] == _Depth(1)
        ]
        if not closings:
            return _Unbracketed()
        closing = closings[-1]
        opening = max(
            index
            for index in range(closing)
            if text.root[index] in "([{" and depths[index] == _Depth(0)
        )
        return _Bracketed(
            head=Expression(text.root[: opening + 1]),
            body=Expression(text.root[opening + 1 : closing]),
            tail=Expression(text.root[closing:]),
        )

    @staticmethod
    def items(body: Expression) -> tuple[Expression, ...]:
        """`body` split at its top-level commas."""
        depths = _Bracketed._depths(body)
        items: list[Expression] = []
        start = 0
        for index, character in enumerate(body.root):
            if character == "," and depths[index] == _Depth(0):
                items.append(Expression(body.root[start:index].strip()))
                start = index + 1
        items.append(Expression(body.root[start:].strip()))
        return tuple(item for item in items if item.root)

    @staticmethod
    def _depths(text: Expression) -> tuple[_Depth, ...]:
        """The depth before each character; -1 inside a string literal."""
        depths: list[_Depth] = []
        depth = 0
        quote = ""
        escaped = False
        for character in text.root:
            if quote:
                depths.append(_Depth(-1))
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = ""
                continue
            depths.append(_Depth(depth))
            if character in "\"'":
                quote = character
            elif character in "([{":
                depth += 1
            elif character in ")]}":
                depth -= 1
        return tuple(depths)


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
        helper = (
            f"\n\n{MappingsRenderer._required_helper().root}"
            if any(function.model.requires() for function in functions)
            else ""
        )
        rendered = "".join(
            f"\n\n{MappingsRenderer._function(function).root}" for function in functions
        )
        return FileText(f"{MappingsRenderer.header.root}\n{imports}{helper}{rendered}")

    @staticmethod
    def _required_helper() -> FunctionText:
        """The function giving a value a View requires, or raising when it is
        `None`. `of` is the subject holding it, named by its ID if it has one."""
        return FunctionText(
            f"def {FunctionPlan.required_helper.root}"
            "[T](value: T | None, of: object, what: str) -> T:\n"
            '    """`value`, which a View requires: `what` of `of`."""\n'
            "    if value is None:\n"
            f'        key = getattr(of, "{EntityNode.primary_key.root}", of)\n'
            '        msg = f"{what} is required, but there is none for '
            '{type(of).__name__} {key!r}"\n'
            "        raise ValueError(msg)\n"
            "    return value\n"
        )

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
                    lambda indent: self._nested(indent, nested, found, followed),
                )
            case ir.Field(value=value):
                pass
        match value:
            case ir.SourceField() | ir.RequiredField():
                return self.column(value, subject)
            case ir.ReverseLookup():
                return self._reverse(value, subject)
            case ir.Lookup() | ir.ChainedLookup():
                followed = self._followed(value.lookups(), subject)
                return self.indent.looked_up(
                    prefix, followed.absent, lambda _: followed.entity
                )
            case ir.Lifted(through=through, field=read):
                followed = self._followed(through, subject)
                return self.indent.looked_up(
                    prefix,
                    followed.absent,
                    lambda indent: self.model_copy(update={"indent": indent}).column(
                        read, followed.entity
                    ),
                )
            case ir.Added():
                return LeftOut()

    def _followed(
        self, lookups: Iterable[ir.EntityLookup], subject: Expression
    ) -> Followed:
        """`lookups` followed from `subject`: an optional Reference finds nothing
        when its ID is `None`, a reversed one when no Entity holds the ID. One the
        View requires finds something, or raises through `_required`."""

        def absent(lookup: ir.EntityLookup, found: Expression) -> Expression:
            access = self._access(lookup, found)
            match lookup:
                case ir.Lookup():
                    return Expression(f"{access.key.root} is None")
                case ir.ReverseLookup():
                    return Expression(
                        f"{access.key.root} not in {access.parameter.root}"
                    )

        return Followed.of(lookups, subject, self._entity, absent)

    def _entity(self, lookup: ir.EntityLookup, subject: Expression) -> Expression:
        """The Entity `lookup` finds from `subject`, on one line. One the View
        requires goes through `_required`."""
        access = self._access(lookup, subject)
        helper = FunctionPlan.required_helper.root
        match lookup:
            case ir.Lookup(reference=ir.RequiredField() as reference):
                what = _Construction._what(reference).root
                key = f"{helper}({access.key.root}, {subject.root}, {what})"
                return Expression(f"{access.parameter.root}[{key}]")
            case ir.ReverseLookup(optionality=Optionality.REQUIRED):
                what = _Construction._what(lookup).root
                found = self._reversed(lookup, subject).root
                return Expression(f"{helper}({found}, {subject.root}, {what})")
            case ir.Lookup() | ir.ReverseLookup():
                return Expression(f"{access.parameter.root}[{access.key.root}]")

    def column(self, column: ir.SubjectColumn, subject: Expression) -> Expression:
        """The value of `subject`'s field `column`, through `_required` if the View
        requires it."""
        value = Expression(f"{subject.root}.{column.field.root}")
        match column:
            case ir.SourceField():
                return value
            case ir.RequiredField():
                return self.required(value, subject, _Construction._what(column))

    def _reverse(self, lookup: ir.ReverseLookup, subject: Expression) -> Expression:
        """The Entity referencing `subject` by `lookup.via`: through `_required` if
        the View requires it, else `None` if there is none."""
        found = self._reversed(lookup, subject)
        match lookup.optionality:
            case Optionality.OPTIONAL:
                return found
            case Optionality.REQUIRED:
                return self.required(found, subject, _Construction._what(lookup))

    def _nested(
        self,
        indent: Indent,
        nested: ModelPlan,
        found: ir.EntityValue,
        followed: Followed,
    ) -> Expression:
        """The call building `nested` from the Entity `found`, at `indent`.

        A model reading nothing from an Entity the View requires still looks it
        up, so that a missing one raises: the call is the second of a pair with it.
        """
        reads = any(
            not isinstance(field.field, ir.Field)
            or not isinstance(field.field.value, ir.Added)
            for field in nested.fields
        )
        required = any(ModelPlan.required_lookup(lookup) for lookup in found.lookups())
        if reads or not required:
            return self.model_copy(update={"indent": indent}).call(
                nested, followed.entity
            )
        inner = self.model_copy(update={"indent": indent.deeper()})
        return indent.exploded(
            Expression("("),
            (followed.entity, inner.call(nested, followed.entity)),
            Expression(")[1]"),
        )

    def required(
        self, value: Expression, subject: Expression, what: Expression
    ) -> Expression:
        """`value`, a value of `subject` the View requires, through the module's
        `_required`, on a line at this indentation."""
        return self.indent.exploded(
            Expression(f"{FunctionPlan.required_helper.root}("),
            (value, subject, what),
            Expression(")"),
        )

    @staticmethod
    def _what(required: ir.RequiredField | ir.ReverseLookup) -> Expression:
        """What `_required` names a value the View requires, as a string."""
        match required:
            case ir.RequiredField(field=field):
                text = field.root
            case ir.ReverseLookup(target=target, via=via):
                text = f"a {target.__name__} by {via.root}"
        return Expression(PythonString(text).literal().root)

    def _reversed(self, lookup: ir.ReverseLookup, subject: Expression) -> Expression:
        """The Entity referencing `subject` by `lookup.via`, or `None`."""
        access = self._access(lookup, subject)
        return Expression(f"{access.parameter.root}.get({access.key.root})")

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
