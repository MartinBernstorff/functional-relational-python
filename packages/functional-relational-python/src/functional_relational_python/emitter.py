"""The Python emitter: compiled Views as plain Python mapping functions.

Each View becomes a function from one instance of its Source to its View model.
The emitted module imports only the Source and View model modules, so it runs and
type checks without this library installed.

Every primitive the InMemoryViewEngine interprets is emitted here too, so the two
backends agree (the differential test in `test_emitter.py` checks this).
"""

import keyword
import re
import sys
import typing
from collections.abc import Callable

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python import compiler, ir
from functional_relational_python.entity_graph import DiagnosticMessage, FieldName
from functional_relational_python.views import ViewName


class ModulePath(pydantic.RootModel[str], frozen=True):
    """A dotted Python module path, e.g. `myapp.views.company`."""

    @pydantic.field_validator("root")
    @classmethod
    def _dotted_identifiers(cls, root: str) -> str:
        if not all(
            part.isidentifier() and not keyword.iskeyword(part)
            for part in root.split(".")
        ):
            msg = f"'{root}' is not a dotted path of Python identifiers"
            raise ValueError(msg)
        return root

    @staticmethod
    def fake() -> ModulePath:
        return ModulePath("app.views")

    def top_level(self) -> BoundName:
        """The name `import <this module>` binds."""
        return BoundName(self.root.split(".")[0])


type ViewModelLayout = Callable[[compiler.CompiledView], ModulePath]
"""Maps each compiled View to the module its View model is generated into."""


class BoundName(pydantic.RootModel[str], frozen=True):
    """A name the emitted module binds: a function, its parameter or an import."""

    @staticmethod
    def fake() -> BoundName:
        return BoundName("company_summary")

    def is_valid(self) -> bool:
        return self.root.isidentifier() and not keyword.iskeyword(self.root)


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


class ShadowedModule(pydantic.BaseModel, frozen=True):
    """A name a View's function binds hides a module the emitted module imports.

    The name is the function's own name, or its parameter when the View model's
    module has the parameter's name.
    """

    view: ViewName
    name: BoundName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: the emitted name '{self.name.root}' would shadow the "
            f"imported module '{self.name.root}'."
        )


type Diagnostic = (
    UnimportableSource | InvalidFunctionName | DuplicateFunctionName | ShadowedModule
)


class EmitError(Exception):
    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "\n".join(diagnostic.describe().root for diagnostic in diagnostics)
        )


class Expression(pydantic.RootModel[str], frozen=True):
    """A Python expression in emitted source, e.g. `source.name`."""


class Assignment(pydantic.BaseModel, frozen=True):
    """One keyword argument to the View model: a field and the value it gets."""

    field: FieldName
    value: Expression


class PythonEmitter:
    """Emits one module of mapping functions. The entry point is `emit`."""

    parameter: typing.ClassVar[BoundName] = BoundName("source")

    @staticmethod
    def emit(
        compiled: compiler.CompiledViews, view_models: ViewModelLayout
    ) -> Result[EmittedModule, EmitError]:
        """Emits a function per View, in View order, or returns every diagnostic.

        `view_models` gives the module each View's model is imported from. The
        model's class name is the View's name.
        """
        diagnostics = PythonEmitter._diagnostics(compiled, view_models)
        if diagnostics:
            return Err(EmitError(diagnostics))
        modules = {
            ModulePath(PythonEmitter.source(view.node).__module__)
            for view in compiled.views
        } | {view_models(view) for view in compiled.views}
        imports = "".join(
            f"import {module}\n" for module in sorted(module.root for module in modules)
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
        return BoundName(re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", view.root).lower())

    @staticmethod
    def source(node: ir.Node) -> type[pydantic.BaseModel]:
        """The Source a View's function takes an instance of."""
        match node:
            case ir.FromSource():
                return node.source
            case ir.Omit() | ir.Rename():
                return PythonEmitter.source(node.input)

    @staticmethod
    def assignments(node: ir.Node) -> tuple[Assignment, ...]:
        """The View model's fields, each with the expression that computes it.

        Interprets frp-ir like `InMemoryViewEngine._run`; each primitive needs a
        case in both.
        """
        match node:
            case ir.FromSource():
                return tuple(
                    Assignment(
                        field=FieldName(name),
                        value=Expression(f"{PythonEmitter.parameter.root}.{name}"),
                    )
                    for name in node.source.model_fields
                )
            case ir.Omit():
                return tuple(
                    assignment
                    for assignment in PythonEmitter.assignments(node.input)
                    if assignment.field != node.field
                )
            case ir.Rename():
                return tuple(
                    Assignment(field=node.new, value=assignment.value)
                    if assignment.field == node.field
                    else assignment
                    for assignment in PythonEmitter.assignments(node.input)
                )

    @staticmethod
    def _function(view: compiler.CompiledView, module: ModulePath) -> EmittedFunction:
        source = PythonEmitter.source(view.node)
        model = f"{module.root}.{view.name.root}"
        arguments = "".join(
            f"        {assignment.field.root}={assignment.value.root},\n"
            for assignment in PythonEmitter.assignments(view.node)
        )
        call = f"(\n{arguments}    )" if arguments else "()"
        return EmittedFunction(
            f"def {PythonEmitter.function_name(view.name).root}(\n"
            f"    {PythonEmitter.parameter.root}: "
            f"{source.__module__}.{source.__qualname__},\n"
            f") -> {model}:\n"
            f"    return {model}{call}\n"
        )

    @staticmethod
    def _diagnostics(
        compiled: compiler.CompiledViews, view_models: ViewModelLayout
    ) -> tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        imported = {view_models(view).top_level() for view in compiled.views}
        views_by_function: dict[BoundName, list[ViewName]] = {}
        for view in compiled.views:
            source = PythonEmitter.source(view.node)
            reachable: object = sys.modules.get(source.__module__)
            for part in source.__qualname__.split("."):
                reachable = getattr(reachable, part, None)
            if reachable is source:
                imported.add(ModulePath(source.__module__).top_level())
            else:
                diagnostics.append(UnimportableSource(view=view.name, source=source))
            function = PythonEmitter.function_name(view.name)
            if not function.is_valid():
                diagnostics.append(
                    InvalidFunctionName(view=view.name, function=function)
                )
            views_by_function.setdefault(function, []).append(view.name)
            if view_models(view).top_level() == PythonEmitter.parameter:
                diagnostics.append(
                    ShadowedModule(view=view.name, name=PythonEmitter.parameter)
                )
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
