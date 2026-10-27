"""The parts generated Python source is written from: module paths, the Symbols
it imports, and annotations as source text."""

import enum
import importlib.util
import keyword
import re
import site
import sys
import sysconfig
import typing
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
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


class SitePackages(pydantic.RootModel[frozenset[Path]], frozen=True):
    """The directories distributions are installed into."""

    @staticmethod
    def current() -> SitePackages:
        """Those of the running interpreter."""
        return SitePackages(
            frozenset(
                Path(directory).resolve()
                for directory in (
                    *site.getsitepackages(),
                    site.getusersitepackages(),
                    sysconfig.get_path("purelib"),
                    sysconfig.get_path("platlib"),
                )
            )
        )


class ImportSection(enum.IntEnum):
    """A section of import statements, in the order ruff's isort rules put them."""

    STANDARD_LIBRARY = enum.auto()
    THIRD_PARTY = enum.auto()
    FIRST_PARTY = enum.auto()

    @staticmethod
    def of(module: ModulePath, installed: SitePackages) -> ImportSection:
        """The standard library by name; third party when the module is installed
        into site-packages; first party otherwise, i.e. the project's own modules,
        including generated ones that do not exist yet.

        ruff decides what is first party from its configuration and the project's
        source directories, which frp does not read. This agrees with it where a
        project marks the packages it develops, e.g. editable workspace members,
        as first party.
        """
        top = module.top_level().root
        if top in sys.stdlib_module_names:
            return ImportSection.STANDARD_LIBRARY
        try:
            spec = importlib.util.find_spec(top)
        except ValueError:
            # An imported module without a spec, e.g. `__main__`, is the project's.
            return ImportSection.FIRST_PARTY
        if spec is None:
            return ImportSection.FIRST_PARTY
        locations = (
            *((spec.origin,) if spec.origin is not None else ()),
            *(spec.submodule_search_locations or ()),
        )
        if any(
            Path(location).resolve().is_relative_to(directory)
            for location in locations
            for directory in installed.root
        ):
            return ImportSection.THIRD_PARTY
        return ImportSection.FIRST_PARTY


class MemberKind(enum.IntEnum):
    """What a name a `from` statement imports is, in the order ruff's isort rules
    put the kinds."""

    CONSTANT = enum.auto()
    CLASS = enum.auto()
    VARIABLE = enum.auto()

    @staticmethod
    def of(name: Identifier) -> MemberKind:
        """A constant is in upper case and longer than one character, e.g.
        `TYPE_CHECKING`; a class starts with an upper case letter, e.g. `X`."""
        if len(name.root) > 1 and name.root.isupper():
            return MemberKind.CONSTANT
        if name.root[0].isupper():
            return MemberKind.CLASS
        return MemberKind.VARIABLE


class SortName(pydantic.RootModel[tuple[tuple[str, int, int], ...]], frozen=True):
    """A name as ruff's isort rules compare it: character by character, except
    that runs of digits compare by value, so `v2` comes before `v10`, and of equal
    values the one with more leading zeros first, so `v01` before `v1`.

    A run of digits is keyed as `0`, the lowest digit, then its value: against
    any other character it compares as a digit would.
    """

    @staticmethod
    def of(name: pydantic.RootModel[str]) -> SortName:
        return SortName(
            tuple(
                ("0", int(token), -len(token)) if token.isdigit() else (token, 0, 0)
                for token in re.findall(r"\d+|\D", name.root)
            )
        )

    @staticmethod
    def ignoring_case(name: pydantic.RootModel[str]) -> SortName:
        return SortName.of(pydantic.RootModel[str](name.root.lower()))

    def __lt__(self, other: SortName) -> bool:
        return self.root < other.root


class ImportText(pydantic.RootModel[str], frozen=True):
    """Import statements as generated source, each line ending in a newline."""


class ImportBlock(pydantic.BaseModel, frozen=True):
    """Import statements, written as ruff's isort rule (I001) with its default
    settings writes them.

    They are in a section each for the standard library, third party and first
    party (see `ImportSection.of`), with a blank line between sections. In a
    section, `import` statements come before `from` statements, each in module
    order: by name ignoring case, then by name. A `from` statement's names are
    constants, then classes, then the rest, each in the same order.
    """

    modules: frozenset[ModulePath]
    """Imported by `import <module>`."""
    symbols: frozenset[Symbol]
    """Imported by `from <module> import <name>`, by their bound names."""

    def text(self) -> ImportText:
        """The statements; empty without any."""
        installed = SitePackages.current()
        sections = {
            module: ImportSection.of(module, installed)
            for module in (*self.modules, *(symbol.module for symbol in self.symbols))
        }
        names: dict[ModulePath, set[Identifier]] = {}
        for symbol in self.symbols:
            names.setdefault(symbol.module, set()).add(symbol.bound_name())
        texts: list[str] = []
        for section in ImportSection:
            straight = sorted(
                (module for module in self.modules if sections[module] is section),
                key=ImportBlock._module_key,
            )
            from_modules = sorted(
                (module for module in names if sections[module] is section),
                key=ImportBlock._module_key,
            )
            lines = [f"import {module.root}\n" for module in straight]
            for module in from_modules:
                ordered = [
                    name.root
                    for name in sorted(
                        names[module],
                        key=lambda name: (
                            MemberKind.of(name),
                            SortName.ignoring_case(name),
                            SortName.of(name),
                        ),
                    )
                ]
                line = f"from {module.root} import {', '.join(ordered)}"
                if len(line) > LineLength.default().root:
                    listed = "".join(f"    {name},\n" for name in ordered)
                    line = f"from {module.root} import (\n{listed})"
                lines.append(f"{line}\n")
            if lines:
                texts.append("".join(lines))
        return ImportText("\n".join(texts))

    @staticmethod
    def _module_key(module: ModulePath) -> tuple[SortName, SortName]:
        return (SortName.ignoring_case(module), SortName.of(module))
