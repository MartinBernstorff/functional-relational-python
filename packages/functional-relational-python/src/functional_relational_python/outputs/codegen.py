"""Generates View model source files from compiled Views.

Everything here is pure: it builds file contents and touches no filesystem.
`generated_files` writes and checks them.

Generation runs in two passes. Planning decides every model's module and renders
its field types; rendering then turns each module's models into source. The
compiler has already named every model (`model_names`): a View's model, and the
model of each resolved field with a nested pipeline. A nested model is one more
`ViewModel` in the plan; the clash checks and rendering treat it like any other.
"""

import sys
import types
import typing
from collections.abc import Callable, Iterable, Sequence
from pathlib import PurePosixPath
from typing import Annotated

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler
from functional_relational_python.compiler.model_names import ModelNames
from functional_relational_python.domain.entity_graph import (
    FieldName,
    Optionality,
)
from functional_relational_python.language import ir
from functional_relational_python.language.views import ViewName
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)
from functional_relational_python.support.filesystem import FileText, RelativePath
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)
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


type ModuleLayout = Callable[[compiler.CompiledView], ModulePath]
"""Maps each compiled View to the module its View model is generated into."""


class AnnotationText(pydantic.RootModel[str], frozen=True):
    """How an annotation or name reads in a diagnostic, e.g. `Annotated[int, 0]`."""


class GeneratedIntoDefiningModule(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View model would be generated into a module that defines classes the
    View uses, overwriting their definitions."""

    view: ViewName
    module: ModulePath
    defined: tuple[AnnotationText, ...]

    def describe(self) -> DiagnosticMessage:
        defined = ", ".join(text.root for text in self.defined)
        return DiagnosticMessage(
            f"{self.view.root}: generating into {self.module.root} would overwrite "
            f"the module defining {defined}. Generate into another module."
        )


class DuplicateModelName(pydantic.BaseModel, Diagnostic, frozen=True):
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


class UnsupportedAnnotation(pydantic.BaseModel, Diagnostic, frozen=True):
    """A field's type has a form the generator cannot write out."""

    view: ViewName
    field: FieldName
    annotation: AnnotationText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}.{self.field.root}: cannot generate the type "
            f"'{self.annotation.root}'."
        )


class UnimportableType(pydantic.BaseModel, Diagnostic, frozen=True):
    """A field's type cannot be imported by its module and name, e.g. a local class."""

    view: ViewName
    field: FieldName
    annotation: AnnotationText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}.{self.field.root}: '{self.annotation.root}' cannot "
            "be imported by its module and name. Define it at module level."
        )


class NameClash(pydantic.BaseModel, Diagnostic, frozen=True):
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


type CodegenDiagnostic = (
    DuplicateModelName
    | GeneratedIntoDefiningModule
    | UnsupportedAnnotation
    | UnimportableType
    | NameClash
)


class UnimportableError(Exception):
    """A class or `type` alias cannot be imported by its module and name."""

    def __init__(self, named: ir.RuntimeAnnotation) -> None:
        self.named = named
        super().__init__(f"{named!r} cannot be imported by its module and name")


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
        return TypeExpression(".".join(part.root for part in self.qualname))


class TypeExpression(pydantic.RootModel[str], frozen=True):
    """An annotation as it is written in generated source."""

    def optional(self, optionality: Optionality) -> TypeExpression:
        """Written like `ir.Lookup.annotation`, for a model that only exists as source."""
        match optionality:
            case Optionality.REQUIRED:
                return self
            case Optionality.OPTIONAL:
                return TypeExpression(f"{self.root} | None")


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
        self.diagnostics: list[CodegenDiagnostic] = []

    def render(self, annotation: ir.RuntimeAnnotation) -> TypeExpression:
        origin = typing.get_origin(annotation)
        if origin is None:
            return self._unparameterised(annotation)
        return self._parameterised(annotation, origin, typing.get_args(annotation))

    def _unparameterised(self, annotation: ir.RuntimeAnnotation) -> TypeExpression:
        if annotation is None or annotation is types.NoneType:
            return TypeExpression("None")
        if annotation is Ellipsis:
            return TypeExpression("...")
        if isinstance(annotation, (type, typing.TypeAliasType)):
            return self._named(annotation)
        return self._unsupported(annotation)

    def _parameterised(
        self,
        annotation: ir.RuntimeAnnotation,
        origin: ir.RuntimeAnnotation,
        args: Sequence[ir.RuntimeAnnotation],
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

    def _named(self, annotation: ir.RuntimeAnnotation) -> TypeExpression:
        match Symbol.of(annotation):
            case Ok(symbol):
                self.symbols.add(symbol)
                return symbol.reference()
            case Err():
                self.diagnostics.append(
                    UnimportableType(
                        view=self.view,
                        field=self.field,
                        annotation=AnnotationText(repr(annotation)),
                    )
                )
                return TypeExpression(repr(annotation))

    def _unsupported(self, annotation: ir.RuntimeAnnotation) -> TypeExpression:
        self.diagnostics.append(
            UnsupportedAnnotation(
                view=self.view,
                field=self.field,
                annotation=AnnotationText(repr(annotation)),
            )
        )
        return TypeExpression(repr(annotation))


class _ModelPlanner:
    """Plans the models of one View: its own, and one per nested resolved field.

    Nested models come before the models that refer to them.
    """

    def __init__(self, view: ViewName, module: ModulePath) -> None:
        self.view = view
        self.module = module
        self.models: list[ViewModel] = []
        self.diagnostics: list[CodegenDiagnostic] = []
        # Classes the View uses that `module` defines, so generating would
        # overwrite their definitions.
        self.defined_here: set[AnnotationText] = set()

    def use(self, model: type) -> None:
        """Records that the View uses `model`, without generated code naming it."""
        if model.__module__ == self.module.root:
            self.defined_here.add(
                AnnotationText(f"{model.__module__}.{model.__qualname__}")
            )

    def plan(self, names: ModelNames, relation: ir.Relation) -> None:
        name = names.name.identifier()
        fields: list[ModelField] = []
        symbols: set[Symbol] = {Codegen.base}
        for field in relation.fields:
            renderer = _FieldRenderer(self.view, field.name)
            match field:
                case ir.Field(info=info):
                    rendered = renderer.render(info.annotation)
                case ir.NestedField(value=lookup, relation=nested):
                    nested_names = names.nested_in(field)
                    self.use(lookup.target)
                    self.plan(nested_names, nested)
                    symbol = Symbol(
                        module=self.module,
                        qualname=(nested_names.name.identifier(),),
                    )
                    symbols.add(symbol)
                    rendered = symbol.reference().optional(field.optionality())
            fields.append(ModelField(name=field.name, type=rendered))
            symbols |= renderer.symbols
            self.defined_here |= {
                symbol.describe()
                for symbol in renderer.symbols
                if symbol.module == self.module
            }
            self.diagnostics.extend(renderer.diagnostics)
        self.models.append(
            ViewModel(
                view=self.view,
                name=name,
                module=self.module,
                fields=tuple(fields),
                symbols=frozenset(symbols),
            )
        )


class Codegen:
    """Plans and renders View models. The entry point is `generate`."""

    base: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("pydantic"), qualname=(Identifier("BaseModel"),)
    )
    line_length: typing.ClassVar[LineLength] = LineLength(88)

    @staticmethod
    def generate(
        compiled: compiler.CompiledViews, layout: ModuleLayout
    ) -> Result[GeneratedFiles, DiagnosticsError[CodegenDiagnostic]]:
        """Generates one file per module, or returns every diagnostic."""
        diagnostics: list[CodegenDiagnostic] = []
        by_module: dict[ModulePath, list[ViewModel]] = {}
        for view in compiled.views:
            planned = Codegen._plan(view, layout(view))
            for item in planned:
                match item:
                    case ViewModel():
                        by_module.setdefault(item.module, []).append(item)
                    case _:
                        diagnostics.append(item)
        for module, models in by_module.items():
            diagnostics.extend(Codegen._name_diagnostics(module, models))

        return DiagnosticsError.or_ok(
            diagnostics,
            lambda: GeneratedFiles(
                files=tuple(
                    GeneratedFile(
                        path=module.file_path(),
                        text=Codegen._render_module(module, by_module[module]),
                    )
                    for module in sorted(by_module, key=lambda module: module.root)
                )
            ),
        )

    @staticmethod
    def _plan(
        view: compiler.CompiledView, module: ModulePath
    ) -> list[ViewModel] | list[CodegenDiagnostic]:
        """The View's model, after the nested models it refers to."""
        planner = _ModelPlanner(view.name, module)
        planner.use(view.source)
        planner.plan(view.names.model, view.relation)
        if planner.defined_here:
            planner.diagnostics.append(
                GeneratedIntoDefiningModule(
                    view=view.name,
                    module=module,
                    defined=tuple(
                        sorted(planner.defined_here, key=lambda text: text.root)
                    ),
                )
            )
        if planner.diagnostics:
            return planner.diagnostics
        return planner.models

    @staticmethod
    def _name_diagnostics(
        module: ModulePath, models: Sequence[ViewModel]
    ) -> Iterable[CodegenDiagnostic]:
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

        sections: list[list[str]] = []
        for section in ModulePath.import_sections(imported):
            lines: list[str] = []
            for source in section:
                names = sorted(name.root for name in imported[source])
                line = f"from {source.root} import {', '.join(names)}"
                if len(line) > Codegen.line_length.root:
                    listed = "".join(f"    {name},\n" for name in names)
                    line = f"from {source.root} import (\n{listed})"
                lines.append(line)
            sections.append(lines)
        imports = "\n\n".join("\n".join(lines) for lines in sections)

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
        return FileText(f"{GeneratedFiles.header.root}\n{imports}\n\n\n{classes}")
