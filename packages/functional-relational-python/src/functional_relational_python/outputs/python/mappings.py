"""Renders the planned mapping functions: compiled Views as plain Python functions.

Each View becomes a function from one instance of its Source to its View model.
A View that resolves References also takes, per Entity it looks up, a keyword
parameter mapping IDs to Entities; a dangling ID raises `KeyError` there. A
module of mapping functions imports only the Source, Entity and View model
modules, so it runs and type checks without this library installed.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two agree (the differential test in `test_mappings.py`
checks this).
"""

import typing
from collections.abc import Callable, Iterable, Sequence

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
        guards: Sequence[Expression],
        present: Callable[[Indent], Expression],
    ) -> Expression:
        """The value of an entry at this indentation that starts with `prefix`,
        e.g. `mentor=`, holding what a chain of References finds: `present`
        itself, or `None if <guard> is None else ...` for each optional Reference
        on the way, `guards` holding the expressions of their IDs or rows.

        The conditionals stay on one line when `present` is one line and the
        entry fits, or else they fit parenthesized on a line of their own;
        otherwise they are split over lines like ruff format does, one chained
        conditional rather than nested ones. `present` gets the
        indentation to write at, and is written once: a one-line expression
        reads the same at any indentation.
        """
        if not guards:
            return present(self)
        inner = self.deeper()
        written = present(inner).root
        conditions = "".join(f"None if {guard.root} is None else " for guard in guards)
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
            f"{pad}if {guard.root} is None\n"
            for index, guard in enumerate(guards)
        )
        return Expression(f"(\n{branches}{pad}else {written}\n{self.pad().root})")


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
        match field:
            case ir.Field(value=value):
                match value:
                    case ir.SourceField() | ir.Lookup() | ir.Lifted():
                        return self.computed(field, value, subject)
                    case ir.Added():
                        return LeftOut()
            case ir.NestedField(value=lookup):
                nested = model.nested_in(field)
                return self.indent.looked_up(
                    Expression(f"{field.name.root}="),
                    _Construction._guard(lookup, subject),
                    lambda indent: self.model_copy(update={"indent": indent}).call(
                        nested, self._entity(lookup, subject)
                    ),
                )

    def computed(
        self, field: ir.RelationField, value: ir.SubjectValue, subject: Expression
    ) -> Expression:
        """`value` computed from `subject`, as the argument for `field`."""
        guards, present = self._chain(value, subject)
        return self.indent.looked_up(
            Expression(f"{field.name.root}="), guards, lambda _: present
        )

    def _chain(
        self, value: ir.SubjectValue, subject: Expression
    ) -> tuple[tuple[Expression, ...], Expression]:
        """The IDs of the optional References `value` follows from `subject`, to
        check for `None` in order, and what it computes once none is."""
        match value:
            case ir.SourceField():
                return (), Expression(f"{subject.root}.{value.field.root}")
            case ir.Lookup():
                return _Construction._guard(value, subject), self._entity(
                    value, subject
                )
            case ir.Lifted(lookup=lookup, value=lifted):
                guards, present = self._chain(lifted, self._entity(lookup, subject))
                return (*_Construction._guard(lookup, subject), *guards), present

    @staticmethod
    def _guard(lookup: ir.Lookup, subject: Expression) -> tuple[Expression, ...]:
        """The ID to check for `None`, if `lookup`'s Reference is optional."""
        match lookup.optionality:
            case Optionality.REQUIRED:
                return ()
            case Optionality.OPTIONAL:
                return (Expression(f"{subject.root}.{lookup.reference.field.root}"),)

    def _entity(self, lookup: ir.Lookup, subject: Expression) -> Expression:
        parameter = self.function.parameter_of(lookup)
        return Expression(
            f"{parameter.root}[{subject.root}.{lookup.reference.field.root}]"
        )
