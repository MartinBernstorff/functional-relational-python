"""Generates View model source files from compiled Views.

Everything here is pure: it builds file contents and touches no filesystem.
`generated_files` writes and checks them.

Generation runs in two passes. Planning decides every model's name and module and
renders its field types; rendering then turns each module's models into source.
Model names are explicit (a View's name) for now. A model with a generated name,
e.g. a variant of a union View, is one more `ViewModel` in the plan: the planner
names it, and the name checks and rendering treat it like any other.
"""

import keyword
import sys
import types
import typing
from collections.abc import Callable, Iterable, Sequence
from pathlib import PurePosixPath
from typing import Annotated

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python import compiler
from functional_relational_python.entity_graph import DiagnosticMessage, FieldName
from functional_relational_python.filesystem import FileText, RelativePath
from functional_relational_python.generated_files import GeneratedFile, GeneratedFiles
from functional_relational_python.views import ViewName

type RuntimeAnnotation = object
"""A field's annotation as Python holds it at run time: a class, `None`, a generic
alias such as `list[int]`, a union, a `type` alias, or a `Literal` argument."""


class Identifier(pydantic.RootModel[str], frozen=True):
    """A Python name that is not a keyword, e.g. a class or module name."""

    @pydantic.field_validator("root")
    @classmethod
    def _not_keyword(cls, root: str) -> str:
        if not root.isidentifier() or keyword.iskeyword(root):
            msg = f"'{root}' is not a Python identifier"
            raise ValueError(msg)
        return root

    @staticmethod
    def fake() -> Identifier:
        return Identifier("CompanySummary")


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

    def is_stdlib(self) -> bool:
        return self.root.split(".")[0] in sys.stdlib_module_names


type ModuleLayout = Callable[[compiler.CompiledView], ModulePath]
"""Maps each compiled View to the module its View model is generated into."""


class AnnotationText(pydantic.RootModel[str], frozen=True):
    """How an annotation or name reads in a diagnostic, e.g. `Annotated[int, 0]`."""


class InvalidModelName(pydantic.BaseModel, frozen=True):
    """A View's name cannot be used as a class name."""

    view: ViewName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: the View name is not a valid Python class name."
        )


class DuplicateModelName(pydantic.BaseModel, frozen=True):
    """Several Views generate a model of the same name into the same module."""

    module: ModulePath
    name: Identifier
    views: tuple[ViewName, ...]

    def describe(self) -> DiagnosticMessage:
        views = ", ".join(view.root for view in self.views)
        return DiagnosticMessage(
            f"{self.module.root}: Views {views} all generate a model named "
            f"'{self.name.root}'."
        )


class UnsupportedAnnotation(pydantic.BaseModel, frozen=True):
    """A field's type has a form the generator cannot write out."""

    view: ViewName
    field: FieldName
    annotation: AnnotationText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}.{self.field.root}: cannot generate the type "
            f"'{self.annotation.root}'."
        )


class UnimportableType(pydantic.BaseModel, frozen=True):
    """A field's type cannot be imported by its module and name, e.g. a local class."""

    view: ViewName
    field: FieldName
    annotation: AnnotationText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}.{self.field.root}: '{self.annotation.root}' cannot "
            "be imported by its module and name. Define it at module level."
        )


class NameClash(pydantic.BaseModel, frozen=True):
    """Several different things would be bound to one name in a generated module."""

    module: ModulePath
    name: Identifier
    origins: tuple[AnnotationText, ...]

    def describe(self) -> DiagnosticMessage:
        origins = ", ".join(origin.root for origin in self.origins)
        return DiagnosticMessage(
            f"{self.module.root}: the name '{self.name.root}' would refer to "
            f"each of {origins}."
        )


type Diagnostic = (
    InvalidModelName
    | DuplicateModelName
    | UnsupportedAnnotation
    | UnimportableType
    | NameClash
)


class CodegenError(Exception):
    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "\n".join(diagnostic.describe().root for diagnostic in diagnostics)
        )


class Symbol(pydantic.BaseModel, frozen=True):
    """A module-level object generated source refers to, e.g. `pydantic.BaseModel`.

    `qualname` has several parts for a nested class; only the first is imported.
    """

    module: ModulePath
    qualname: Annotated[tuple[Identifier, ...], pydantic.Field(min_length=1)]

    def bound_name(self) -> Identifier:
        return self.qualname[0]

    def describe(self) -> AnnotationText:
        return AnnotationText(
            ".".join([self.module.root, *(part.root for part in self.qualname)])
        )

    def reference(self) -> TypeExpression:
        return TypeExpression(".".join(part.root for part in self.qualname))


class TypeExpression(pydantic.RootModel[str], frozen=True):
    """An annotation as it is written in generated source."""


class ModelField(pydantic.BaseModel, frozen=True):
    name: FieldName
    type: TypeExpression


class LineLength(pydantic.RootModel[int], frozen=True):
    """The longest line generated source aims for, in characters."""


class ViewModel(pydantic.BaseModel, frozen=True):
    """One model to generate: its name, module, fields and the Symbols it uses."""

    view: ViewName
    name: Identifier
    module: ModulePath
    fields: tuple[ModelField, ...]
    symbols: frozenset[Symbol]

    def symbol(self) -> Symbol:
        return Symbol(module=self.module, qualname=(self.name,))


class _FieldRenderer:
    """Writes one field's annotation as source, collecting what it refers to.

    Problems are collected rather than returned early, so every unsupported part
    of an annotation is reported.
    """

    literal: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("typing"), qualname=(Identifier("Literal"),)
    )

    def __init__(self, view: ViewName, field: FieldName) -> None:
        self.view = view
        self.field = field
        self.symbols: set[Symbol] = set()
        self.diagnostics: list[Diagnostic] = []

    def render(self, annotation: RuntimeAnnotation) -> TypeExpression:
        origin = typing.get_origin(annotation)
        if origin is None:
            return self._unparameterised(annotation)
        return self._parameterised(annotation, origin, typing.get_args(annotation))

    def _unparameterised(self, annotation: RuntimeAnnotation) -> TypeExpression:
        if annotation is None or annotation is types.NoneType:
            return TypeExpression("None")
        if annotation is Ellipsis:
            return TypeExpression("...")
        if isinstance(annotation, (type, typing.TypeAliasType)):
            return self._named(annotation)
        return self._unsupported(annotation)

    def _parameterised(
        self,
        annotation: RuntimeAnnotation,
        origin: RuntimeAnnotation,
        args: Sequence[RuntimeAnnotation],
    ) -> TypeExpression:
        if origin is typing.Union:
            return TypeExpression(" | ".join(self.render(arg).root for arg in args))
        if origin is typing.Literal and all(
            type(arg) in (str, int, bool) for arg in args
        ):
            self.symbols.add(_FieldRenderer.literal)
            values = ", ".join(repr(arg) for arg in args)
            return TypeExpression(
                f"{_FieldRenderer.literal.reference().root}[{values}]"
            )
        if origin in (typing.Literal, typing.Annotated) or not args:
            return self._unsupported(annotation)
        rendered = ", ".join(self.render(arg).root for arg in args)
        return TypeExpression(f"{self.render(origin).root}[{rendered}]")

    def _named(self, annotation: RuntimeAnnotation) -> TypeExpression:
        module = getattr(annotation, "__module__", "")
        parts = getattr(annotation, "__qualname__", "").split(".")
        reachable = sys.modules.get(module)
        for part in parts:
            reachable = getattr(reachable, part, None)
        if reachable is not annotation:
            # Also covers `<locals>` in the qualname: no attribute has that name.
            self.diagnostics.append(
                UnimportableType(
                    view=self.view,
                    field=self.field,
                    annotation=AnnotationText(repr(annotation)),
                )
            )
            return TypeExpression(repr(annotation))
        symbol = Symbol(
            module=ModulePath(module), qualname=tuple(map(Identifier, parts))
        )
        self.symbols.add(symbol)
        return symbol.reference()

    def _unsupported(self, annotation: RuntimeAnnotation) -> TypeExpression:
        self.diagnostics.append(
            UnsupportedAnnotation(
                view=self.view,
                field=self.field,
                annotation=AnnotationText(repr(annotation)),
            )
        )
        return TypeExpression(repr(annotation))


class Codegen:
    """Plans and renders View models. The entry point is `generate`."""

    base: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("pydantic"), qualname=(Identifier("BaseModel"),)
    )
    header: typing.ClassVar[FileText] = FileText(
        "# Generated by functional-relational-python. Do not edit.\n"
        "# Regenerate with `frp generate`; `frp check` fails when this is stale.\n"
    )
    line_length: typing.ClassVar[LineLength] = LineLength(88)

    @staticmethod
    def generate(
        compiled: compiler.CompiledViews, layout: ModuleLayout
    ) -> Result[GeneratedFiles, CodegenError]:
        """Generates one file per module, or returns every diagnostic."""
        diagnostics: list[Diagnostic] = []
        by_module: dict[ModulePath, list[ViewModel]] = {}
        for view in compiled.views:
            match Codegen._plan(view, layout(view)):
                case ViewModel() as model:
                    by_module.setdefault(model.module, []).append(model)
                case planned:
                    diagnostics.extend(planned)
        for module, models in by_module.items():
            diagnostics.extend(Codegen._name_diagnostics(module, models))

        if diagnostics:
            return Err(CodegenError(tuple(diagnostics)))
        return Ok(
            GeneratedFiles(
                files=tuple(
                    GeneratedFile(
                        path=module.file_path(),
                        text=Codegen._render_module(module, by_module[module]),
                    )
                    for module in sorted(by_module, key=lambda module: module.root)
                )
            )
        )

    @staticmethod
    def _plan(
        view: compiler.CompiledView, module: ModulePath
    ) -> ViewModel | list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        name = view.name.root
        if not name.isidentifier() or keyword.iskeyword(name):
            diagnostics.append(InvalidModelName(view=view.name))

        fields: list[ModelField] = []
        symbols: set[Symbol] = {Codegen.base}
        for field in view.relation.fields:
            renderer = _FieldRenderer(view.name, field.name)
            fields.append(
                ModelField(name=field.name, type=renderer.render(field.info.annotation))
            )
            symbols |= renderer.symbols
            diagnostics.extend(renderer.diagnostics)

        if diagnostics:
            return diagnostics
        return ViewModel(
            view=view.name,
            name=Identifier(name),
            module=module,
            fields=tuple(fields),
            symbols=frozenset(symbols),
        )

    @staticmethod
    def _name_diagnostics(
        module: ModulePath, models: Sequence[ViewModel]
    ) -> Iterable[Diagnostic]:
        views_by_name: dict[Identifier, list[ViewName]] = {}
        for model in models:
            views_by_name.setdefault(model.name, []).append(model.view)
        for name, views in views_by_name.items():
            if len(views) > 1:
                yield DuplicateModelName(module=module, name=name, views=tuple(views))

        # Every name the module binds, by what it is bound to. A Symbol in this
        # module (a model) and an import of it are the same binding.
        bindings: dict[Identifier, set[Symbol]] = {}
        for model in models:
            for symbol in {model.symbol(), *model.symbols}:
                bindings.setdefault(symbol.bound_name(), set()).add(
                    Symbol(module=symbol.module, qualname=(symbol.bound_name(),))
                )
        for name, bound in bindings.items():
            if len(bound) > 1:
                yield NameClash(
                    module=module,
                    name=name,
                    origins=tuple(
                        sorted(
                            (symbol.describe() for symbol in bound),
                            key=lambda text: text.root,
                        )
                    ),
                )

    @staticmethod
    def _render_module(module: ModulePath, models: Sequence[ViewModel]) -> FileText:
        imported: dict[ModulePath, set[Identifier]] = {}
        for model in models:
            for symbol in model.symbols:
                if symbol.module.root != "builtins" and symbol.module != module:
                    imported.setdefault(symbol.module, set()).add(symbol.bound_name())

        # Standard library imports first, then the rest, like isort.
        sections: dict[bool, list[str]] = {True: [], False: []}
        for source in sorted(imported, key=lambda source: source.root):
            names = sorted(name.root for name in imported[source])
            line = f"from {source.root} import {', '.join(names)}"
            if len(line) > Codegen.line_length.root:
                listed = "".join(f"    {name},\n" for name in names)
                line = f"from {source.root} import (\n{listed})"
            sections[source.is_stdlib()].append(line)
        imports = "\n\n".join("\n".join(lines) for lines in sections.values() if lines)

        base = Codegen.base.reference().root
        classes = "\n\n".join(
            f"class {model.name.root}({base}, frozen=True):\n"
            + (
                "".join(
                    f"    {field.name.root}: {field.type.root}\n"
                    for field in model.fields
                )
                or "    pass\n"
            )
            for model in models
        )
        return FileText(f"{Codegen.header.root}\n{imports}\n\n\n{classes}")
