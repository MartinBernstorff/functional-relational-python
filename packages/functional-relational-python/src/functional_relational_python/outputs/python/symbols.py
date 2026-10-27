"""The parts generated Python source is written from: module paths, the Symbols
it imports, and annotations as source text."""

import enum
import keyword
import sys
import typing
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Annotated

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.pipeline import ir
from functional_relational_python.support.filesystem import RelativePath
from functional_relational_python.support.identifier import Identifier


class ModulePath(pydantic.RootModel[str], frozen=True):
    """A dotted Python module path, e.g. `myapp.views.company`."""

    @pydantic.field_validator("root")
    @classmethod
    def _dotted_identifiers(cls, root: str) -> str:
        _ = [Identifier(part) for part in root.split(".")]
        return root

    @staticmethod
    def fake() -> ModulePath:
        return ModulePath("app.views")

    def file_path(self) -> RelativePath:
        """Where the module lives, relative to the directory it is imported from."""
        return RelativePath(PurePosixPath(*self.root.split(".")).with_suffix(".py"))

    def top_level(self) -> Identifier:
        """The name `import <this module>` binds."""
        return Identifier(self.root.split(".")[0])

    @staticmethod
    def ordered(modules: Iterable[ModulePath]) -> tuple[ModulePath, ...]:
        """`modules` in module order, each once."""
        return tuple(sorted(set(modules), key=lambda module: module.root))

    @staticmethod
    def import_sections(
        modules: Iterable[ModulePath],
    ) -> tuple[tuple[ModulePath, ...], ...]:
        """`modules` grouped like isort: the standard library, then the rest.

        Each section is sorted, and an empty one is left out.
        """
        ordered = ModulePath.ordered(modules)
        stdlib = tuple(
            module
            for module in ordered
            if module.top_level().root in sys.stdlib_module_names
        )
        rest = tuple(module for module in ordered if module not in stdlib)
        return tuple(section for section in (stdlib, rest) if section)


class AnnotationText(pydantic.RootModel[str], frozen=True):
    """How an annotation or name reads in a diagnostic, e.g. `Annotated[int, 0]`."""

    @staticmethod
    def of(annotation: ir.RuntimeAnnotation) -> AnnotationText:
        return AnnotationText(repr(annotation))


class UnimportableError(Exception):
    """A class or `type` alias cannot be imported by its module and name."""

    def __init__(self, named: ir.RuntimeAnnotation) -> None:
        self.named = named
        super().__init__(f"{named!r} cannot be imported by its module and name")


class TypeExpression(pydantic.RootModel[str], frozen=True):
    """An annotation as it is written in generated source."""


class Quote(enum.StrEnum):
    """The quote character a string or bytes literal is in."""

    SINGLE = "'"
    DOUBLE = '"'


class PythonString(pydantic.RootModel[str], frozen=True):
    """A string generated source writes as a string literal."""

    def literal(self) -> ValueExpression:
        """The string in the quotes ruff writes: double quotes, unless the string
        has more double quotes than single ones."""
        return PythonString.requoted(
            ValueExpression(repr(self.root)),
            Quote.SINGLE
            if self.root.count('"') > self.root.count("'")
            else Quote.DOUBLE,
        )

    @staticmethod
    def requoted(written: ValueExpression, quote: Quote) -> ValueExpression:
        """`written`, the `repr` of a string or bytes without its prefix, in
        `quote`."""
        body = written.root[1:-1]
        if written.root.startswith("'"):
            # `repr` escapes single quotes only when it quotes with them.
            body = body.replace("\\'", "'")
        escaped = body.replace(quote, f"\\{quote}")
        return ValueExpression(f"{quote}{escaped}{quote}")


class PythonBytes(pydantic.RootModel[bytes], frozen=True):
    """Bytes generated source writes as a bytes literal."""

    def literal(self) -> ValueExpression:
        """The bytes in the quotes ruff writes, which it chooses as for a string."""
        written = PythonString.requoted(
            ValueExpression(repr(self.root).removeprefix("b")),
            Quote.SINGLE
            if self.root.count(b'"') > self.root.count(b"'")
            else Quote.DOUBLE,
        )
        return ValueExpression(f"b{written.root}")


class ValueExpression(pydantic.RootModel[str], frozen=True):
    """A value as it is written in generated source, e.g. a field's default."""

    @staticmethod
    def of_scalar(value: object) -> ValueExpression:
        """`value`, e.g. a string, bytes or a finite number, written as ruff
        formats it: a string or bytes in the quotes ruff writes, and a float's
        exponent without a `+`."""
        match value:
            case str():
                return PythonString(value).literal()
            case bytes():
                return PythonBytes(value).literal()
            case float():
                return ValueExpression(float.__repr__(value).replace("e+", "e"))
            case _:
                return ValueExpression(repr(value))

    @staticmethod
    def of_enum_member(
        enum_class: TypeExpression, member: enum.Enum
    ) -> ValueExpression:
        """`member` as an attribute of `enum_class`, or by subscription when its
        name is no identifier."""
        name = member.name
        if name.isidentifier() and not keyword.iskeyword(name):
            return ValueExpression(f"{enum_class.root}.{name}")
        return ValueExpression(
            f"{enum_class.root}[{PythonString(name).literal().root}]"
        )


class Symbol(pydantic.BaseModel, frozen=True):
    """A module-level object generated source refers to, e.g. `pydantic.BaseModel`.

    `qualname` has several parts for a nested class; only the first is imported.
    """

    module: ModulePath
    qualname: Annotated[tuple[Identifier, ...], pydantic.Field(min_length=1)]

    @staticmethod
    def of(named: ir.RuntimeAnnotation) -> Result[Symbol, UnimportableError]:
        """How source imports `named`, a class, function or `type` alias, by its
        module and qualified name. That fails when `named` is not reachable that
        way, e.g. for a class defined in a function.

        A `type` alias has no qualified name, only a name, so only one defined at
        module level is reachable.
        """
        module = getattr(named, "__module__", "")
        name = (
            named.__name__
            if isinstance(named, typing.TypeAliasType)
            else getattr(named, "__qualname__", "")
        )
        parts = name.split(".")
        reachable = sys.modules.get(module)
        for part in parts:
            reachable = getattr(reachable, part, None)
        if reachable is not named:
            # Also covers `<locals>` in the qualname: no attribute has that name.
            return Err(UnimportableError(named))
        return Ok(
            Symbol(module=ModulePath(module), qualname=tuple(map(Identifier, parts)))
        )

    def bound_name(self) -> Identifier:
        return self.qualname[0]

    def imported(self) -> Symbol:
        """What source imports to refer to this Symbol: `Outer` for `Outer.Inner`.

        Symbols nested in one class import the same thing, so they share a name.
        """
        return Symbol(module=self.module, qualname=(self.bound_name(),))

    def describe(self) -> AnnotationText:
        return AnnotationText(
            ".".join([self.module.root, *(part.root for part in self.qualname)])
        )

    def reference(self) -> TypeExpression:
        """The Symbol where its module is imported from: `Outer.Inner`."""
        return TypeExpression(".".join(part.root for part in self.qualname))

    def qualified(self) -> TypeExpression:
        """The Symbol where only its module is imported: `app.views.Outer.Inner`."""
        return TypeExpression(self.describe().root)


class LineLength(pydantic.RootModel[int], frozen=True):
    """The longest line generated source aims for, in characters."""

    @staticmethod
    def default() -> LineLength:
        """ruff's default, so generated source reads like formatted code."""
        return LineLength(88)
