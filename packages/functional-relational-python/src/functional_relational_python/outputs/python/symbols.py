"""The parts generated Python source is written from: module paths, the Symbols
it imports, and annotations as source text."""

import sys
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Annotated

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.domain.entity_graph import Optionality
from functional_relational_python.language import ir
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
    def import_sections(
        modules: Iterable[ModulePath],
    ) -> tuple[tuple[ModulePath, ...], ...]:
        """`modules` grouped like isort: the standard library, then the rest.

        Each section is sorted, and an empty one is left out.
        """
        ordered = sorted(set(modules), key=lambda module: module.root)
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

    def optional(self, optionality: Optionality) -> TypeExpression:
        """Written like `ir.Lookup.annotation`, for a model that only exists as source."""
        match optionality:
            case Optionality.REQUIRED:
                return self
            case Optionality.OPTIONAL:
                return TypeExpression(f"{self.root} | None")


class Symbol(pydantic.BaseModel, frozen=True):
    """A module-level object generated source refers to, e.g. `pydantic.BaseModel`.

    `qualname` has several parts for a nested class; only the first is imported.
    """

    module: ModulePath
    qualname: Annotated[tuple[Identifier, ...], pydantic.Field(min_length=1)]

    @staticmethod
    def of(named: ir.RuntimeAnnotation) -> Result[Symbol, UnimportableError]:
        """How source imports `named`, a class or `type` alias, by its module and
        qualified name. That fails when `named` is not reachable that way, e.g. for
        a class defined in a function."""
        module = getattr(named, "__module__", "")
        parts = getattr(named, "__qualname__", "").split(".")
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
