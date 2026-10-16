"""The Python emitter: compiled Views as plain Python mapping functions.

Each View becomes a function from one instance of its Source to its View model.
A View that resolves References also takes, per Entity it looks up, a keyword
parameter mapping IDs to Entities; a dangling ID raises `KeyError` there. The
emitted module imports only the Source, Entity and View model modules, so it runs
and type checks without this library installed.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two backends agree (the differential test in
`test_emitter.py` checks this).
"""

import re
import typing
from collections.abc import Callable

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler
from functional_relational_python.domain.entity_graph import (
    DiagnosticMessage,
    Optionality,
)
from functional_relational_python.language import ir
from functional_relational_python.language.views import ViewName
from functional_relational_python.outputs.codegen import (
    Codegen,
    InvalidModelName,
    ModuleLayout,
    ModulePath,
    Symbol,
)
from functional_relational_python.support.identifier import Identifier


class BoundName(pydantic.RootModel[str], frozen=True):
    """A name the emitted module binds: a function, its parameter or an import."""

    @staticmethod
    def fake() -> BoundName:
        return BoundName("company_summary")


class PascalCase(pydantic.RootModel[str], frozen=True):
    """A class-style name, such as a View's or an Entity's."""

    def snake_case(self) -> BoundName:
        """`CompanyPublic` is `company_public`."""
        return BoundName(re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", self.root).lower())


class EmittedModule(pydantic.RootModel[str], frozen=True):
    """The source text of an emitted module."""


class EmittedFunction(pydantic.RootModel[str], frozen=True):
    """The source text of one mapping function."""


class UnimportableSource(pydantic.BaseModel, frozen=True):
    """A Source the emitted module cannot import, e.g. a class defined in a function."""

    view: ViewName
    source: type[pydantic.BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: the Source {self.source.__module__}."
            f"{self.source.__qualname__} cannot be imported by its module and name. "
            "Define it at module level."
        )


class UnimportableReferencedModel(pydantic.BaseModel, frozen=True):
    """An Entity a View looks up, or its ID type, cannot be imported."""

    view: ViewName
    model: type[pydantic.BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: {self.model.__module__}.{self.model.__qualname__} "
            "cannot be imported by its module and name. Define it at module level."
        )


class InvalidFunctionName(pydantic.BaseModel, frozen=True):
    view: ViewName
    function: BoundName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: '{self.function.root}' is not a valid Python "
            "function name."
        )


class DuplicateFunctionName(pydantic.BaseModel, frozen=True):
    """Several View names have the same snake_case form."""

    function: BoundName
    views: tuple[ViewName, ...]

    def describe(self) -> DiagnosticMessage:
        views = ", ".join(view.root for view in self.views)
        return DiagnosticMessage(
            f"Views {views} all emit a function named '{self.function.root}'."
        )


class DuplicateLookupParameter(pydantic.BaseModel, frozen=True):
    """A View looks up several Entities whose names give the same parameter."""

    view: ViewName
    parameter: BoundName
    entities: tuple[type[pydantic.BaseModel], ...]

    def describe(self) -> DiagnosticMessage:
        entities = ", ".join(
            f"{entity.__module__}.{entity.__qualname__}" for entity in self.entities
        )
        return DiagnosticMessage(
            f"{self.view.root}: Entities {entities} all need a parameter named "
            f"'{self.parameter.root}'."
        )


class ShadowedModule(pydantic.BaseModel, frozen=True):
    """A name a View's function binds hides a module the emitted module imports.

    The name is the function's own name, or one of its parameters when the View
    model's module has that parameter's name.
    """

    view: ViewName
    name: BoundName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: the emitted name '{self.name.root}' would shadow the "
            f"imported module '{self.name.root}'."
        )


type Diagnostic = (
    UnimportableSource
    | InvalidModelName
    | UnimportableReferencedModel
    | InvalidFunctionName
    | DuplicateFunctionName
    | DuplicateLookupParameter
    | ShadowedModule
)


class EmitError(Exception):
    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "\n".join(diagnostic.describe().root for diagnostic in diagnostics)
        )


class Expression(pydantic.RootModel[str], frozen=True):
    """A Python expression in emitted source, e.g. `source.name`.

    The later lines of a multi-line expression are indented for where it is used.
    """


class Indent(pydantic.RootModel[int], frozen=True):
    """How many spaces a line of emitted source starts with."""

    def pad(self) -> Expression:
        return Expression(" " * self.root)

    def deeper(self) -> Indent:
        return Indent(self.root + 4)


class PythonEmitter:
    """Emits one module of mapping functions. The entry point is `emit`."""

    parameter: typing.ClassVar[BoundName] = BoundName("source")
    mapping: typing.ClassVar[ModulePath] = ModulePath("collections.abc")

    @staticmethod
    def emit(
        compiled: compiler.CompiledViews, view_models: ModuleLayout
    ) -> Result[EmittedModule, EmitError]:
        """Emits a function per View, in View order, or returns every diagnostic.

        `view_models` gives the module each View's model is imported from. The
        model's class name is the View's name. Nested View models are in the same
        module, named by `Codegen.nested_model_name`.
        """
        diagnostics = PythonEmitter._diagnostics(compiled, view_models)
        if diagnostics:
            return Err(EmitError(diagnostics))
        modules = (
            module
            for view in compiled.views
            for module in PythonEmitter._imports(view, view_models(view))
        )
        imports = "\n".join(
            "".join(f"import {module.root}\n" for module in section)
            for section in ModulePath.import_sections(modules)
        )
        functions = "".join(
            f"\n\n{PythonEmitter._function(view, view_models(view)).root}"
            for view in compiled.views
        )
        return Ok(
            EmittedModule(
                "# Generated by functional-relational-python. Do not edit.\n"
                f"\n{imports}{functions}"
            )
        )

    @staticmethod
    def function_name(view: ViewName) -> BoundName:
        """The View's name in snake_case: `CompanyPublic` is `company_public`."""
        return PascalCase(view.root).snake_case()

    @staticmethod
    def lookup_parameter(entity: type[pydantic.BaseModel]) -> BoundName:
        """The keyword parameter mapping IDs to `entity`, e.g. `department_by_id`."""
        snake = PascalCase(entity.__name__).snake_case()
        return BoundName(f"{snake.root}_by_id")

    @staticmethod
    def _qualified(model: type[pydantic.BaseModel]) -> Expression:
        return Expression(f"{model.__module__}.{model.__qualname__}")

    @staticmethod
    def _imports(
        view: compiler.CompiledView, module: ModulePath
    ) -> frozenset[ModulePath]:
        lookups = view.relation.lookups()
        return frozenset(
            {
                module,
                ModulePath(view.source.__module__),
                *(ModulePath(lookup.target.__module__) for lookup in lookups),
                *(ModulePath(lookup.id_type.__module__) for lookup in lookups),
                *((PythonEmitter.mapping,) if lookups else ()),
            }
        )

    @staticmethod
    def _function(view: compiler.CompiledView, module: ModulePath) -> EmittedFunction:
        lookups = "".join(
            f"    {PythonEmitter.lookup_parameter(lookup.target).root}: "
            f"{PythonEmitter.mapping.root}.Mapping[\n"
            f"        {PythonEmitter._qualified(lookup.id_type).root},\n"
            f"        {PythonEmitter._qualified(lookup.target).root},\n"
            "    ],\n"
            for lookup in view.relation.lookups()
        )
        keyword_only = f"    *,\n{lookups}" if lookups else ""
        model = Expression(f"{module.root}.{view.name.root}")
        call = _Construction(module=module, indent=Indent(4)).call(
            model,
            Identifier(view.name.root),
            view.relation,
            Expression(PythonEmitter.parameter.root),
        )
        return EmittedFunction(
            f"def {PythonEmitter.function_name(view.name).root}(\n"
            f"    {PythonEmitter.parameter.root}: "
            f"{PythonEmitter._qualified(view.source).root},\n"
            f"{keyword_only}"
            f") -> {model.root}:\n"
            f"    return {call.root}\n"
        )

    @staticmethod
    def _diagnostics(
        compiled: compiler.CompiledViews, view_models: ModuleLayout
    ) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        imported = {
            BoundName(module.top_level().root)
            for view in compiled.views
            for module in PythonEmitter._imports(view, view_models(view))
        }
        views_by_function: dict[BoundName, list[ViewName]] = {}
        for view in compiled.views:
            if isinstance(Symbol.of(view.source), Err):
                diagnostics.append(
                    UnimportableSource(view=view.name, source=view.source)
                )
            diagnostics.extend(PythonEmitter._parameter_diagnostics(view, view_models))
            if isinstance(Identifier.of(view.name), Err):
                diagnostics.append(InvalidModelName(view=view.name))
            function = PythonEmitter.function_name(view.name)
            if isinstance(Identifier.of(function), Err):
                diagnostics.append(
                    InvalidFunctionName(view=view.name, function=function)
                )
            views_by_function.setdefault(function, []).append(view.name)
        diagnostics.extend(
            DuplicateFunctionName(function=function, views=tuple(views))
            for function, views in views_by_function.items()
            if len(views) > 1
        )
        diagnostics.extend(
            ShadowedModule(view=view, name=function)
            for function, views in views_by_function.items()
            if function in imported
            for view in views
        )
        return tuple(diagnostics)

    @staticmethod
    def _parameter_diagnostics(
        view: compiler.CompiledView, view_models: ModuleLayout
    ) -> list[Diagnostic]:
        """Problems with a View function's parameters and the types they name."""
        diagnostics: list[Diagnostic] = []
        lookups = view.relation.lookups()
        diagnostics.extend(
            UnimportableReferencedModel(view=view.name, model=model)
            for lookup in lookups
            for model in (lookup.target, lookup.id_type)
            if isinstance(Symbol.of(model), Err)
        )
        entities_by_parameter: dict[BoundName, list[type[pydantic.BaseModel]]] = {}
        for lookup in lookups:
            entities_by_parameter.setdefault(
                PythonEmitter.lookup_parameter(lookup.target), []
            ).append(lookup.target)
        diagnostics.extend(
            DuplicateLookupParameter(
                view=view.name, parameter=parameter, entities=tuple(entities)
            )
            for parameter, entities in entities_by_parameter.items()
            if len(entities) > 1
        )
        # The body refers to the View model's module, so no parameter may hide it.
        diagnostics.extend(
            ShadowedModule(view=view.name, name=parameter)
            for parameter in (PythonEmitter.parameter, *entities_by_parameter)
            if view_models(view).top_level().root == parameter.root
        )
        return diagnostics


class _Construction:
    """Writes the call that builds a View model, at one indentation.

    The output is ruff-format stable: calls keep a trailing comma, so they stay
    exploded, and a conditional is split over lines only when it does not fit.
    """

    def __init__(self, module: ModulePath, indent: Indent) -> None:
        self.module = module
        self.indent = indent

    def call(
        self,
        model: Expression,
        name: Identifier,
        relation: ir.Relation,
        subject: Expression,
    ) -> Expression:
        """`model(...)`, with a keyword argument per field of `relation`.

        `name` is the model's class name, which nested model names start with.
        """
        inner = _Construction(self.module, self.indent.deeper())
        arguments = "".join(
            f"{inner.indent.pad().root}{field.name.root}="
            f"{inner.value(name, field, subject).root},\n"
            for field in relation.fields
        )
        if not arguments:
            return Expression(f"{model.root}()")
        return Expression(f"{model.root}(\n{arguments}{self.indent.pad().root})")

    def value(
        self, model: Identifier, field: ir.RelationField, subject: Expression
    ) -> Expression:
        """The argument for `field` of `model`, on a line at this indentation."""
        match field:
            case ir.Field(value=value):
                match value:
                    case ir.SourceField():
                        return Expression(f"{subject.root}.{value.field.root}")
                    case ir.Lookup():
                        entity = self._entity(value, subject)
                        return self._optional(field, value, subject, lambda _: entity)
            case ir.NestedField(value=lookup, relation=relation):
                nested = Codegen.nested_model_name(model, field.name)
                return self._optional(
                    field,
                    lookup,
                    subject,
                    lambda construction: construction.call(
                        Expression(f"{self.module.root}.{nested.root}"),
                        nested,
                        relation,
                        self._entity(lookup, subject),
                    ),
                )

    def _entity(self, lookup: ir.Lookup, subject: Expression) -> Expression:
        parameter = PythonEmitter.lookup_parameter(lookup.target)
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
        if "\n" not in inline and len(line) <= Codegen.line_length.root:
            return Expression(inline)
        inner = _Construction(self.module, self.indent.deeper())
        pad = inner.indent.pad().root
        return Expression(
            f"(\n{pad}None\n{pad}if {reference} is None\n"
            f"{pad}else {present(inner).root}\n{self.indent.pad().root})"
        )
