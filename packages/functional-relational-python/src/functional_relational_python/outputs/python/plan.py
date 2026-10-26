"""Plans the Python output of compiled Views and Payloads: what each generated
name refers to.

For each View and Payload the plan decides its model's Symbol and the Symbols of
the models nested in it. For each View, when the layout asks for them, it also
decides its mapping function's Symbol and lookup parameters. Every problem that depends on where things are generated
(a name bound twice in one module, a type that cannot be imported, a module that
defines what it would be overwritten with) is found here, once. The renderers in
`models` and `mappings` read the plan and cannot fail.

The compiler has already checked the names themselves (`model_names`).
"""

import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler
from functional_relational_python.compiler.model_names import ModelNames, PascalCase
from functional_relational_python.entity_graph.entity_graph import FieldName
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    ModulePath,
    Symbol,
    TypeExpression,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.payloads import (
    DefinitionName,
    PayloadName,
)
from functional_relational_python.pipeline.views import ViewName
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)
from functional_relational_python.support.identifier import Identifier

type ModuleLayout = Callable[
    [compiler.CompiledView | compiler.CompiledPayload], ModulePath
]
"""Maps each compiled View or Payload to the module something of it is generated
into. Only Views have mapping functions, so a layout for mappings receives only
Views."""


class WithoutMappings(pydantic.BaseModel, frozen=True):
    """Plan models only, without mapping functions."""


class PythonLayout(pydantic.BaseModel, frozen=True):
    """Where each View's and Payload's model, and each View's mapping function,
    are generated."""

    models: ModuleLayout
    mappings: ModuleLayout | WithoutMappings

    @staticmethod
    def models_only(models: ModuleLayout) -> PythonLayout:
        return PythonLayout(models=models, mappings=WithoutMappings())


class Scope(pydantic.RootModel[str], frozen=True):
    """Where generated source binds names: a module such as `app.views`, or a
    function such as `app.mappings.company_public`."""


class GeneratedIntoDefiningModule(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View's or Payload's model, or a View's mapping function, would be
    generated into a module that defines classes it uses, overwriting their
    definitions."""

    view: DefinitionName
    module: ModulePath
    defined: tuple[AnnotationText, ...]

    def describe(self) -> DiagnosticMessage:
        defined = ", ".join(text.root for text in self.defined)
        return DiagnosticMessage(
            f"{self.view.root}: generating into {self.module.root} would overwrite "
            f"the module defining {defined}. Generate into another module."
        )


class _DefinitionsUsed:
    """The classes a View's or Payload's model, or a View's mapping function, uses
    that the module it is generated into defines. Generating would overwrite their
    definitions."""

    def __init__(self, view: DefinitionName, module: ModulePath) -> None:
        self.view = view
        self.module = module
        self.defined: set[AnnotationText] = set()

    def use(self, used: type) -> None:
        """Records that `used` is used, whether or not it can be imported."""
        if used.__module__ == self.module.root:
            self.defined.add(AnnotationText(f"{used.__module__}.{used.__qualname__}"))

    def use_symbol(self, used: Symbol) -> None:
        if used.module == self.module:
            self.defined.add(used.describe())

    def diagnostics(self) -> tuple[GeneratedIntoDefiningModule, ...]:
        if not self.defined:
            return ()
        return (
            GeneratedIntoDefiningModule(
                view=self.view,
                module=self.module,
                defined=tuple(sorted(self.defined, key=lambda text: text.root)),
            ),
        )


class UnsupportedAnnotation(pydantic.BaseModel, Diagnostic, frozen=True):
    """A field's type, or a type argument of the base, has a form the generator
    cannot write out.

    `field` is the field typed by it, or `None` for the base.
    """

    view: DefinitionName
    field: FieldName | None
    annotation: AnnotationText

    def describe(self) -> DiagnosticMessage:
        where = (
            self.view.root
            if self.field is None
            else f"{self.view.root}.{self.field.root}"
        )
        return DiagnosticMessage(
            f"{where}: cannot generate the type '{self.annotation.root}'."
        )


class Unimportable(pydantic.BaseModel, Diagnostic, frozen=True):
    """Generated source would import something it cannot import by its module and
    name, e.g. a class defined in a function: a field's type, the base, the Source,
    or an Entity the View looks up or its ID type.

    `field` is the field typed by it, or `None` when no model field is: the base,
    the Source, or what only a mapping function's parameters refer to.
    """

    view: DefinitionName
    field: FieldName | None
    annotation: AnnotationText

    def describe(self) -> DiagnosticMessage:
        where = (
            self.view.root
            if self.field is None
            else f"{self.view.root}.{self.field.root}"
        )
        return DiagnosticMessage(
            f"{where}: '{self.annotation.root}' cannot be imported by its "
            "module and name. Define it at module level."
        )


class NameClash(pydantic.BaseModel, Diagnostic, frozen=True):
    """Several things would be bound to one name in a generated module or function:
    two models, two functions, two parameters, or one of them and an import."""

    scope: Scope
    name: Identifier
    origins: tuple[AnnotationText, ...]

    def describe(self) -> DiagnosticMessage:
        origins = ", ".join(origin.root for origin in self.origins)
        return DiagnosticMessage(
            f"{self.scope.root}: the name '{self.name.root}' would refer to "
            f"each of {origins}."
        )


class ModelsAndMappingsInOneModule(pydantic.BaseModel, Diagnostic, frozen=True):
    """The layout generates View models and mapping functions into one module, so
    one file would overwrite the other."""

    module: ModulePath

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.module.root}: View models and mapping functions cannot be "
            "generated into the same module."
        )


type PlanDiagnostic = (
    GeneratedIntoDefiningModule
    | UnsupportedAnnotation
    | Unimportable
    | NameClash
    | ModelsAndMappingsInOneModule
)


class Inherited(pydantic.BaseModel, frozen=True):
    """A field the model's base declares with the same type, so the model does not
    declare it again."""


class ModelField(pydantic.BaseModel, frozen=True):
    """A field of a planned model: the Relation field it holds, and its type as
    source, or that the model inherits it from its base."""

    field: ir.RelationField
    type: TypeExpression | Inherited


class ModelPlan(pydantic.BaseModel, frozen=True):
    """One model to generate, and the models nested in it.

    `base` is the class it subclasses, as source. `uses` is every Symbol its class
    statement refers to: the base, the fields' types and the nested models.
    """

    default_base: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("pydantic"), qualname=(Identifier("BaseModel"),)
    )

    view: DefinitionName
    symbol: Symbol
    base: TypeExpression
    fields: tuple[ModelField, ...]
    uses: frozenset[Symbol]
    nested: Mapping[FieldName, ModelPlan]

    def nested_in(self, field: ir.NestedField) -> ModelPlan:
        return self.nested[field.name]

    def with_nested(self) -> tuple[ModelPlan, ...]:
        """This model and every model nested in it, each after those it refers to."""
        return (
            *(
                model
                for nested in self.nested.values()
                for model in nested.with_nested()
            ),
            self,
        )


class LookupParameter(pydantic.BaseModel, frozen=True):
    """A mapping function's keyword parameter, mapping IDs to the Entity `entity`."""

    name: Identifier
    entity: type[pydantic.BaseModel]
    id_type: Symbol
    target: Symbol

    @staticmethod
    def name_for(entity: type[pydantic.BaseModel]) -> Identifier:
        """The parameter for `entity`, e.g. `department_by_id`."""
        snake = PascalCase(entity.__name__).snake_case()
        return Identifier(f"{snake.root}_by_id")


class FunctionPlan(pydantic.BaseModel, frozen=True):
    """One mapping function to generate: from one instance of the View's Source,
    and an ID mapping per Entity it looks up, to the View's model."""

    source_parameter: typing.ClassVar[Identifier] = Identifier("source")
    mapping: typing.ClassVar[ModulePath] = ModulePath("collections.abc")

    view: ViewName
    symbol: Symbol
    source: Symbol
    model: ModelPlan
    lookups: tuple[LookupParameter, ...]

    @staticmethod
    def of(
        view: compiler.CompiledView, model: ModelPlan, module: ModulePath
    ) -> Result[FunctionPlan, DiagnosticsError[PlanDiagnostic]]:
        """The View's mapping function, generated into `module`, building `model`."""
        lookups = view.relation.lookups()
        used = _DefinitionsUsed(view.name, module)
        symbols: dict[type, Symbol] = {}
        diagnostics: list[PlanDiagnostic] = []
        for named in (
            view.source,
            *(t for lookup in lookups for t in (lookup.target, lookup.id_type)),
        ):
            used.use(named)
            match Symbol.of(named):
                case Ok(symbol):
                    symbols[named] = symbol
                case Err():
                    diagnostics.append(
                        Unimportable(
                            view=view.name,
                            field=None,
                            annotation=AnnotationText.of(named),
                        )
                    )
        diagnostics.extend(used.diagnostics())
        return DiagnosticsError.or_ok(
            diagnostics,
            lambda: FunctionPlan(
                view=view.name,
                symbol=Symbol(
                    module=module, qualname=(Identifier(view.names.function.root),)
                ),
                source=symbols[view.source],
                model=model,
                lookups=tuple(
                    LookupParameter(
                        name=LookupParameter.name_for(lookup.target),
                        entity=lookup.target,
                        target=symbols[lookup.target],
                        id_type=symbols[lookup.id_type],
                    )
                    for lookup in lookups
                ),
            ),
        )

    def parameter_of(self, lookup: ir.Lookup) -> Identifier:
        """The parameter holding the Entities `lookup` finds.

        The plan has a parameter for every Lookup of the View's Relation.
        """
        (name,) = (
            parameter.name
            for parameter in self.lookups
            if parameter.entity is lookup.target
        )
        return name

    def imports(self) -> frozenset[ModulePath]:
        """The modules the function refers to, each imported whole."""
        return frozenset(
            {
                self.model.symbol.module,
                self.source.module,
                *(parameter.target.module for parameter in self.lookups),
                *(parameter.id_type.module for parameter in self.lookups),
                *((FunctionPlan.mapping,) if self.lookups else ()),
            }
        )

    def scope(self) -> Scope:
        return Scope(self.symbol.describe().root)


class _AnnotationRenderer:
    """Writes one field's annotation as source, collecting what it refers to.

    Problems are collected rather than returned early, so every unsupported part
    of an annotation is reported.
    """

    literal: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("typing"), qualname=(Identifier("Literal"),)
    )

    def __init__(self, view: DefinitionName, field: FieldName | None) -> None:
        self.view = view
        self.field = field
        self.symbols: set[Symbol] = set()
        self.diagnostics: list[PlanDiagnostic] = []

    def render_base(self, base: type[pydantic.BaseModel]) -> TypeExpression:
        """`base` as source: `pydantic.BaseModel` as imported from `pydantic`, a
        parameterised generic model as its generic class and type arguments."""
        if base is pydantic.BaseModel:
            self.symbols.add(ModelPlan.default_base)
            return ModelPlan.default_base.reference()
        generic = base.__pydantic_generic_metadata__
        origin = generic["origin"]
        if origin is None:
            return self._named(base)
        arguments = ", ".join(self.render(arg).root for arg in generic["args"])
        return TypeExpression(f"{self._named(origin).root}[{arguments}]")

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
            self.symbols.add(_AnnotationRenderer.literal)
            values = ", ".join(repr(arg) for arg in args)
            return TypeExpression(
                f"{_AnnotationRenderer.literal.reference().root}[{values}]"
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
                    Unimportable(
                        view=self.view,
                        field=self.field,
                        annotation=AnnotationText.of(annotation),
                    )
                )
                return TypeExpression(repr(annotation))

    def _unsupported(self, annotation: ir.RuntimeAnnotation) -> TypeExpression:
        self.diagnostics.append(
            UnsupportedAnnotation(
                view=self.view,
                field=self.field,
                annotation=AnnotationText.of(annotation),
            )
        )
        return TypeExpression(repr(annotation))


class _PlannedModel(pydantic.BaseModel, frozen=True):
    """A View's or Payload's model, planned even when that found problems."""

    plan: ModelPlan
    diagnostics: tuple[PlanDiagnostic, ...]


class _ModelPlanner:
    """Plans the model of one View or Payload, and one per nested resolved field."""

    def __init__(self, view: DefinitionName, module: ModulePath) -> None:
        self.view = view
        self.module = module
        self.diagnostics: list[PlanDiagnostic] = []
        self.used = _DefinitionsUsed(view, module)

    @staticmethod
    def definition_model(
        compiled: compiler.CompiledView | compiler.CompiledPayload, module: ModulePath
    ) -> _PlannedModel:
        planner = _ModelPlanner(compiled.name, module)
        planner.used.use(compiled.source)
        match compiled:
            case compiler.CompiledView(names=names):
                plan = planner.plan(names.model, compiled.relation, compiled.base)
            case compiler.CompiledPayload(names=names):
                plan = planner.plan(names, compiled.relation, compiled.base)
        return _PlannedModel(
            plan=plan,
            diagnostics=(*planner.diagnostics, *planner.used.diagnostics()),
        )

    def plan(
        self,
        names: ModelNames,
        relation: ir.Relation,
        base: type[pydantic.BaseModel],
    ) -> ModelPlan:
        """The model of `relation`, subclassing `base`.

        The compiler has checked that each kept field `base` declares has the type
        it declares, so the model inherits it. Nested models subclass the default
        base.
        """
        fields: list[ModelField] = []
        uses: set[Symbol] = set()
        base_renderer = _AnnotationRenderer(self.view, None)
        rendered_base = base_renderer.render_base(base)
        self._record(base_renderer, uses)
        nested: dict[FieldName, ModelPlan] = {}
        for field in relation.fields:
            renderer = _AnnotationRenderer(self.view, field.name)
            match field:
                case _ if field.name.root in base.model_fields:
                    rendered = Inherited()
                case ir.Field():
                    rendered = renderer.render(field.annotation())
                case ir.NestedField(value=lookup, relation=nested_relation):
                    self.used.use(lookup.target)
                    model = self.plan(
                        names.nested_in(field), nested_relation, pydantic.BaseModel
                    )
                    nested[field.name] = model
                    uses.add(model.symbol)
                    rendered = field.optionality().annotate_text(
                        model.symbol.reference()
                    )
            fields.append(ModelField(field=field, type=rendered))
            self._record(renderer, uses)
        return ModelPlan(
            view=self.view,
            symbol=Symbol(module=self.module, qualname=(names.name.identifier(),)),
            base=rendered_base,
            fields=tuple(fields),
            uses=frozenset(uses),
            nested=nested,
        )

    def _record(self, renderer: _AnnotationRenderer, uses: set[Symbol]) -> None:
        """Adds what `renderer` referred to to `uses`, and keeps its diagnostics."""
        uses |= renderer.symbols
        for symbol in renderer.symbols:
            self.used.use_symbol(symbol)
        self.diagnostics.extend(renderer.diagnostics)


class PythonPlan(pydantic.BaseModel, frozen=True):
    """Every model and mapping function to generate, by module: View models in
    View order, then Payload models in Payload order.

    `models` holds each View's and Payload's own model; its nested models are in
    it.
    """

    models: Mapping[ModulePath, tuple[ModelPlan, ...]]
    functions: Mapping[ModulePath, tuple[FunctionPlan, ...]]

    @staticmethod
    def of(
        compiled: compiler.CompiledCatalog, layout: PythonLayout
    ) -> Result[PythonPlan, DiagnosticsError[PlanDiagnostic]]:
        """The plan, or every diagnostic. Each diagnostic is reported once, even
        when both a model and a mapping function run into it. A type a model field
        cannot import is reported for that field only, not again for the mapping
        function that looks it up."""
        diagnostics: dict[PlanDiagnostic, None] = {}
        models: dict[ModulePath, list[ModelPlan]] = {}
        functions: dict[ModulePath, list[FunctionPlan]] = {}
        for view in compiled.views:
            model = _ModelPlanner.definition_model(view, layout.models(view))
            diagnostics.update(dict.fromkeys(model.diagnostics))
            models.setdefault(model.plan.symbol.module, []).append(model.plan)
            match layout.mappings:
                case WithoutMappings():
                    continue
                case mappings:
                    pass
            match FunctionPlan.of(view, model.plan, mappings(view)):
                case Ok(function):
                    functions.setdefault(function.symbol.module, []).append(function)
                case Err(error):
                    unimportable_fields = {
                        diagnostic.annotation
                        for diagnostic in model.diagnostics
                        if isinstance(diagnostic, Unimportable)
                    }
                    diagnostics.update(
                        dict.fromkeys(
                            diagnostic
                            for diagnostic in error.diagnostics
                            if not isinstance(diagnostic, Unimportable)
                            or diagnostic.annotation not in unimportable_fields
                        )
                    )
        for payload in compiled.payloads:
            model = _ModelPlanner.definition_model(payload, layout.models(payload))
            diagnostics.update(dict.fromkeys(model.diagnostics))
            models.setdefault(model.plan.symbol.module, []).append(model.plan)
        plan = PythonPlan(
            models={module: tuple(planned) for module, planned in models.items()},
            functions={module: tuple(planned) for module, planned in functions.items()},
        )
        diagnostics.update(dict.fromkeys(plan._layout_diagnostics()))
        return DiagnosticsError.or_ok(diagnostics, lambda: plan)

    def _layout_diagnostics(self) -> Iterable[PlanDiagnostic]:
        for module, models in self.models.items():
            namespace = _Namespace(Scope(module.root))
            for model in (m for own in models for m in own.with_nested()):
                match model.view:
                    case ViewName():
                        definition = f"View {model.view.root}"
                    case PayloadName():
                        definition = f"Payload {model.view.root}"
                namespace.define(
                    model.symbol.bound_name(),
                    AnnotationText(
                        f"{model.symbol.describe().root} (the model of {definition})"
                    ),
                )
                for symbol in model.uses:
                    if symbol.module != module:
                        namespace.refer(
                            symbol.bound_name(), symbol.imported().describe()
                        )
            yield from namespace.clashes()
        for module, functions in self.functions.items():
            namespace = _Namespace(Scope(module.root))
            for function in functions:
                namespace.define(
                    function.symbol.bound_name(),
                    AnnotationText(
                        f"{function.symbol.describe().root} "
                        f"(the mapping function of View {function.view.root})"
                    ),
                )
                for imported in function.imports():
                    namespace.refer_to_module(imported)
            yield from namespace.clashes()
            for function in functions:
                yield from _Namespace.of_function(function).clashes()
        yield from (
            ModelsAndMappingsInOneModule(module=module)
            for module in sorted(
                self.models.keys() & self.functions.keys(),
                key=lambda module: module.root,
            )
        )


class _Namespace:
    """The names generated source binds in one Scope, with what each refers to."""

    def __init__(self, scope: Scope) -> None:
        self.scope = scope
        self.bound: dict[Identifier, list[AnnotationText]] = {}

    def define(self, name: Identifier, origin: AnnotationText) -> None:
        """Binds `name` to a thing generated here. Each binding counts, so two
        Views generating one name clash."""
        self.bound.setdefault(name, []).append(origin)

    def refer(self, name: Identifier, origin: AnnotationText) -> None:
        """Refers to `name`, imported or built in. Referring to the same thing
        twice is one binding."""
        origins = self.bound.setdefault(name, [])
        if origin not in origins:
            origins.append(origin)

    def refer_to_module(self, module: ModulePath) -> None:
        """Refers to `module` by its full path, so to its top-level package."""
        top_level = module.top_level()
        self.refer(top_level, AnnotationText(f"the module {top_level.root}"))

    @staticmethod
    def of_function(function: FunctionPlan) -> _Namespace:
        """A function's parameters, and the module its body builds models from."""
        namespace = _Namespace(function.scope())
        namespace.define(
            FunctionPlan.source_parameter,
            AnnotationText(
                f"the parameter for the Source {function.source.describe().root}"
            ),
        )
        for parameter in function.lookups:
            namespace.define(
                parameter.name,
                AnnotationText(f"the parameter for {parameter.target.describe().root}"),
            )
        namespace.refer_to_module(function.model.symbol.module)
        return namespace

    def clashes(self) -> Iterable[NameClash]:
        for name, origins in self.bound.items():
            if len(origins) > 1:
                yield NameClash(
                    scope=self.scope,
                    name=name,
                    origins=tuple(sorted(origins, key=lambda text: text.root)),
                )
