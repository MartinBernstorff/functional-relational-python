"""Plans the Python output of compiled Views, Payloads and Unions: what each
generated name refers to.

For each View and Payload the plan decides its model's Symbol and the Symbols of
the models nested in it, and for each Union, of Views or of Payloads, the Symbol
of its model, a `type` alias. For each View and Union of Views, when the layout
asks for them, it also decides its mapping function's Symbol and lookup
parameters. Every problem that depends on where things are generated (a name
bound twice in one module, a type that cannot be imported, a module that defines
what it would be overwritten with) is found here, once. The renderers in
`models` and `mappings` read the plan and cannot fail.

The compiler has already checked the names themselves (`model_names`).
"""

import enum
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler, narrowing
from functional_relational_python.compiler.model_names import ModelNames, PascalCase
from functional_relational_python.entity_graph.entity_graph import (
    FieldName,
    Optionality,
)
from functional_relational_python.entity_graph.id import OneToOneMarker
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    ModulePath,
    PythonString,
    Symbol,
    TypeExpression,
    ValueExpression,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.payloads import (
    DefinitionName,
    PayloadName,
)
from functional_relational_python.pipeline.unions import UnionName
from functional_relational_python.pipeline.views import ViewName
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)
from functional_relational_python.support.identifier import Identifier

type ModuleLayout = Callable[
    [
        compiler.CompiledView
        | compiler.CompiledPayload
        | compiler.CompiledUnion
        | compiler.CompiledPayloadUnion
    ],
    ModulePath,
]
"""Maps each compiled View, Payload or Union to the module something of it is
generated into. Only Views and Unions of Views have mapping functions, so a
layout for mappings receives only those."""


type _Placed = (
    compiler.CompiledView
    | compiler.CompiledPayload
    | compiler.CompiledUnion
    | compiler.CompiledPayloadUnion
)
"""Anything a layout places."""


class WithoutMappings(pydantic.BaseModel, frozen=True):
    """Plan models only, without mapping functions."""


class LayoutRole(enum.Enum):
    """What a layout places: models, or mapping functions."""

    MODELS = "models"
    MAPPINGS = "mappings"


class ReturnedText(pydantic.RootModel[str], frozen=True):
    """How a value a layout returned reads in a diagnostic: its `repr`."""

    @staticmethod
    def of(returned: object) -> ReturnedText:
        return ReturnedText(repr(returned))


class NoModuleFromLayout(pydantic.BaseModel, Diagnostic, frozen=True):
    """A layout returned something other than a `ModulePath` for a View, Payload
    or Union, e.g. `None` from a layout that does not handle Unions of Payloads."""

    definition: DefinitionName | UnionName
    role: LayoutRole
    returned: ReturnedText

    @staticmethod
    def check(
        layout: ModuleLayout, role: LayoutRole, compiled: _Placed
    ) -> ModulePath | NoModuleFromLayout:
        """The module `layout` gives `compiled`, or why it gives none."""
        returned: object = layout(compiled)
        if isinstance(returned, ModulePath):
            return returned
        return NoModuleFromLayout(
            definition=compiled.name, role=role, returned=ReturnedText.of(returned)
        )

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.definition.root}: the {self.role.value} layout returned "
            f"{self.returned.root} for it, not a ModulePath. A layout returns a "
            "ModulePath for every View, Payload and Union it receives."
        )


class PlacedModels(pydantic.BaseModel, frozen=True):
    """The module a models layout gives each View's, Payload's and Union's model,
    checked once, so that every output places the models alike and a layout's
    defects are reported once."""

    modules: Mapping[DefinitionName | UnionName, ModulePath]

    @staticmethod
    def of(
        layout: ModuleLayout, placed: Iterable[_Placed]
    ) -> Result[PlacedModels, DiagnosticsError[NoModuleFromLayout]]:
        """The module of each of `placed`, or every one `layout` gives none."""
        modules: dict[DefinitionName | UnionName, ModulePath] = {}
        unplaced: list[NoModuleFromLayout] = []
        for compiled in placed:
            match NoModuleFromLayout.check(layout, LayoutRole.MODELS, compiled):
                case NoModuleFromLayout() as diagnostic:
                    unplaced.append(diagnostic)
                case module:
                    modules[compiled.name] = module
        return DiagnosticsError.or_ok(unplaced, lambda: PlacedModels(modules=modules))

    @staticmethod
    def of_catalog(
        layout: ModuleLayout, catalog: compiler.CompiledCatalog
    ) -> Result[PlacedModels, DiagnosticsError[NoModuleFromLayout]]:
        """The module of every model of `catalog`."""
        return PlacedModels.of(
            layout,
            (
                *catalog.views,
                *catalog.payloads,
                *catalog.unions,
                *catalog.payload_unions,
            ),
        )

    def module_of(self, compiled: _Placed) -> ModulePath:
        """The module of a placed definition's model: a layout that gives every
        model the module it was placed in."""
        return self.modules[compiled.name]


class PythonLayout(pydantic.BaseModel, frozen=True):
    """Where each View's and Payload's model, and each View's mapping function,
    are generated."""

    models: ModuleLayout
    mappings: ModuleLayout | WithoutMappings

    @staticmethod
    def models_only(models: ModuleLayout) -> PythonLayout:
        return PythonLayout(models=models, mappings=WithoutMappings())

    @staticmethod
    def mappings_into(models: ModuleLayout, mappings: ModulePath) -> PythonLayout:
        """Every View's and Union's mapping function into the one module
        `mappings`."""

        def into(
            _: compiler.CompiledView
            | compiler.CompiledPayload
            | compiler.CompiledUnion
            | compiler.CompiledPayloadUnion,
        ) -> ModulePath:
            return mappings

        return PythonLayout(models=models, mappings=into)

    def model_of(self, compiled: _Placed) -> Symbol | NoModuleFromLayout:
        """The Symbol of a View's, Payload's or Union's own model, or why the
        layout gives it no module."""
        match compiled:
            case compiler.CompiledView(names=names):
                name = names.model.name
            case compiler.CompiledPayload(names=names):
                name = names.name
            case compiler.CompiledUnion(names=names):
                name = names.model
            case compiler.CompiledPayloadUnion(model=name):
                pass
        match NoModuleFromLayout.check(self.models, LayoutRole.MODELS, compiled):
            case NoModuleFromLayout() as unplaced:
                return unplaced
            case module:
                return Symbol(module=module, qualname=(name.identifier(),))


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
    """The layout generates View or Payload models and mapping functions into one
    module, so one file would overwrite the other."""

    module: ModulePath

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.module.root}: models and mapping functions cannot be "
            "generated into the same module."
        )


type PlanDiagnostic = (
    GeneratedIntoDefiningModule
    | UnsupportedAnnotation
    | Unimportable
    | NameClash
    | ModelsAndMappingsInOneModule
    | NoModuleFromLayout
)


class Inherited(pydantic.BaseModel, frozen=True):
    """A field the model's base declares with the same type, so the model does not
    declare it again."""


class ModelField(pydantic.BaseModel, frozen=True):
    """A field of a planned model: the Relation field it holds, and its type and
    default as source, or that the model inherits it from its base."""

    field: ir.RelationField
    type: TypeExpression | Inherited
    default: ValueExpression | ir.NoDefault


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

    def requires(self) -> bool:
        """Whether building this model, or a model nested in it, reads a value the
        View requires."""
        return any(ModelPlan._required(field.field) for field in self.fields) or any(
            nested.requires() for nested in self.nested.values()
        )

    @staticmethod
    def _required(field: ir.RelationField) -> bool:
        match field:
            case ir.Field(value=value):
                pass
            case ir.NestedField(value=found):
                value = found
        required = any(ModelPlan.required_lookup(lookup) for lookup in value.lookups())
        match value:
            case ir.RequiredField() | ir.Lifted(field=ir.RequiredField()):
                return True
            case (
                ir.SourceField()
                | ir.Added()
                | ir.Lifted()
                | ir.Lookup()
                | ir.ReverseLookup()
                | ir.ChainedLookup()
            ):
                return required

    @staticmethod
    def required_lookup(lookup: ir.EntityLookup) -> bool:
        """Whether the View requires `lookup` to find an Entity, which it might
        not."""
        match lookup:
            case ir.Lookup(reference=reference):
                return isinstance(reference, ir.RequiredField)
            case ir.ReverseLookup(optionality=optionality):
                return optionality is Optionality.REQUIRED

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


class UnionPlan(pydantic.BaseModel, frozen=True):
    """One Union model to generate: a `type` alias of the `Annotated` union of
    `variants`, the Variants' models, discriminated by `discriminator`."""

    annotated: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("typing"), qualname=(Identifier("Annotated"),)
    )
    discriminator_class: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("pydantic"), qualname=(Identifier("Discriminator"),)
    )

    union: UnionName
    symbol: Symbol
    discriminator: FieldName
    variants: tuple[Symbol, ...]

    @staticmethod
    def of(
        union: compiler.CompiledUnion | compiler.CompiledPayloadUnion,
        layout: PythonLayout,
    ) -> Result[UnionPlan, DiagnosticsError[NoModuleFromLayout]]:
        """The model of a Union of Views or of Payloads, where `layout` puts it,
        or why the layout gives it or a Variant's model no module."""
        placed = (layout.model_of(union), *map(layout.model_of, union.variants))
        symbols = [symbol for symbol in placed if isinstance(symbol, Symbol)]
        return DiagnosticsError.or_ok(
            (found for found in placed if isinstance(found, NoModuleFromLayout)),
            lambda: UnionPlan(
                union=union.name,
                symbol=symbols[0],
                discriminator=union.discriminator,
                variants=tuple(symbols[1:]),
            ),
        )

    def uses(self) -> frozenset[Symbol]:
        """Every Symbol the alias refers to."""
        return frozenset(
            {UnionPlan.annotated, UnionPlan.discriminator_class, *self.variants}
        )


class LookupParameter(pydantic.BaseModel, frozen=True):
    """A mapping function's keyword parameter, mapping each value of the field
    `keyed_by` of the Entity `entity` to the Entity: its primary key for a
    resolved Reference, its one-to-one Reference for a reversed one."""

    name: Identifier
    entity: type[pydantic.BaseModel]
    keyed_by: FieldName
    id_type: Symbol
    target: Symbol

    @staticmethod
    def name_for(entity: type[pydantic.BaseModel], keyed_by: FieldName) -> Identifier:
        """The parameter for `entity` by `keyed_by`, e.g. `department_by_id`, or
        `meta_campaign_by_channel_campaign_id`."""
        snake = PascalCase(entity.__name__).snake_case()
        return Identifier(f"{snake.root}_by_{keyed_by.root}")


class MappingFunctionPlan(pydantic.BaseModel, frozen=True):
    """What every mapping function to generate has: its Symbol, the Source it
    maps, and its keyword parameters, an ID mapping per Entity and field it looks
    up."""

    symbol: Symbol
    source: Symbol
    lookups: tuple[LookupParameter, ...]

    def parameter_imports(self) -> frozenset[ModulePath]:
        """The modules the function's parameters refer to, each imported whole."""
        return frozenset(
            {
                self.source.module,
                *(parameter.target.module for parameter in self.lookups),
                *(parameter.id_type.module for parameter in self.lookups),
                *((FunctionPlan.mapping,) if self.lookups else ()),
            }
        )

    def scope(self) -> Scope:
        return Scope(self.symbol.describe().root)


class FunctionPlan(MappingFunctionPlan, frozen=True):
    """One mapping function to generate: from one instance of the View's Source,
    and an ID mapping per Entity and field it looks up, to the View's model, or to
    `None` for a row the View's Filters exclude.

    `literals` has the Symbol of every class the Filters' literals are instances
    of, e.g. an enum or a RootModel.
    """

    source_parameter: typing.ClassVar[Identifier] = Identifier("source")
    required_helper: typing.ClassVar[Identifier] = Identifier("_required")
    """The function a mapping module defines when a View requires a value. It
    gives the value, or raises `ValueError` when it is `None`."""
    mapping: typing.ClassVar[ModulePath] = ModulePath("collections.abc")

    view: ViewName
    model: ModelPlan
    filters: tuple[ir.Filter, ...]
    literals: Mapping[type, Symbol]

    @staticmethod
    def of(
        view: compiler.CompiledView, model: ModelPlan, module: ModulePath
    ) -> Result[FunctionPlan, DiagnosticsError[PlanDiagnostic]]:
        """The View's mapping function, generated into `module`, building `model`."""
        lookups = view.relation.lookups()
        used = _DefinitionsUsed(view.name, module)
        symbols: dict[type, Symbol] = {}
        diagnostics: list[PlanDiagnostic] = []
        literal_types = tuple(
            dict.fromkeys(
                literal_type
                for kept in view.relation.filters
                for literal_type in FunctionPlan._literal_types(kept)
            )
        )
        for named in (
            view.source,
            *(t for lookup in lookups for t in (lookup.target, lookup.id_type)),
            *literal_types,
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
                        name=LookupParameter.name_for(lookup.target, lookup.keyed_by()),
                        entity=lookup.target,
                        keyed_by=lookup.keyed_by(),
                        target=symbols[lookup.target],
                        id_type=symbols[lookup.id_type],
                    )
                    for lookup in lookups
                ),
                filters=view.relation.filters,
                literals={
                    literal_type: symbols[literal_type]
                    for literal_type in literal_types
                },
            ),
        )

    @staticmethod
    def _literal_types(kept: ir.Filter) -> tuple[type, ...]:
        """The class of every enum member and RootModel among `kept`'s literals."""
        match kept:
            case ir.FilterComparison(operand=operand):
                return FunctionPlan._classes_of(operand)
            case ir.FilterIsNull():
                return ()
            case ir.FilterAnd(terms=terms) | ir.FilterOr(terms=terms):
                return tuple(
                    literal_type
                    for term in terms
                    for literal_type in FunctionPlan._literal_types(term)
                )
            case ir.FilterNot(term=term):
                return FunctionPlan._literal_types(term)

    @staticmethod
    def _classes_of(operand: ir.SourceField | ir.FilterLiteral) -> tuple[type, ...]:
        match operand:
            case ir.SourceField() | ir.ScalarLiteral():
                return ()
            case ir.EnumLiteral(member=member):
                return (type(member),)
            case ir.RootModelLiteral(instance=instance, root=root):
                return (type(instance), *FunctionPlan._classes_of(root))

    def literal_symbol(self, literal_type: type) -> Symbol:
        """The Symbol of a class one of the Filters' literals is an instance of.

        The plan has one for every such class.
        """
        return self.literals[literal_type]

    def parameter_of(self, lookup: ir.EntityLookup) -> Identifier:
        """The parameter holding the Entities `lookup` finds.

        The plan has a parameter for every lookup of the View's Relation.
        """
        (name,) = (
            parameter.name
            for parameter in self.lookups
            if parameter.entity is lookup.target
            and parameter.keyed_by == lookup.keyed_by()
        )
        return name

    def imports(self) -> frozenset[ModulePath]:
        """The modules the function refers to, each imported whole."""
        return self.parameter_imports() | {
            self.model.symbol.module,
            *(symbol.module for symbol in self.literals.values()),
        }


class VariantCall(pydantic.BaseModel, frozen=True):
    """A call of a Variant's mapping function from a Union's: `function`, passed
    the Source and each of `lookups`, its keyword parameters, by name."""

    function: Symbol
    lookups: tuple[Identifier, ...]


class UnionFunctionPlan(MappingFunctionPlan, frozen=True):
    """One Union's mapping function to generate: from one instance of the Source
    its Variants read, and the lookup parameters of every Variant's mapping
    function, to the model of the first Variant whose function maps it, or to
    `None` when none does.

    `lookups` has each parameter of a Variant's function once.
    """

    found: typing.ClassVar[Identifier] = Identifier("found")
    """The variable holding what the Variants' functions have mapped so far."""

    union: UnionName
    model: Symbol
    variants: tuple[VariantCall, ...]

    @staticmethod
    def of(
        union: compiler.CompiledUnion,
        model: Symbol,
        module: ModulePath,
        variants: Sequence[FunctionPlan | UnionFunctionPlan],
    ) -> UnionFunctionPlan:
        """The Union's mapping function, generated into `module`, building `model`
        by calling `variants`, the Variants' mapping functions, in turn."""
        return UnionFunctionPlan(
            union=union.name,
            symbol=Symbol(
                module=module, qualname=(Identifier(union.names.function.root),)
            ),
            source=variants[0].source,
            model=model,
            lookups=tuple(
                dict.fromkeys(
                    parameter for variant in variants for parameter in variant.lookups
                )
            ),
            variants=tuple(
                VariantCall(
                    function=variant.symbol,
                    lookups=tuple(parameter.name for parameter in variant.lookups),
                )
                for variant in variants
            ),
        )

    def called(self, call: VariantCall) -> Symbol | Identifier:
        """How the function refers to a Variant's: by name in its own module, else
        by its Symbol, its module imported whole."""
        if call.function.module == self.symbol.module:
            return call.function.bound_name()
        return call.function

    def imports(self) -> frozenset[ModulePath]:
        """The modules the function refers to, each imported whole."""
        return self.parameter_imports() | {
            self.model.module,
            *(
                called.module
                for call in self.variants
                if isinstance(called := self.called(call), Symbol)
            ),
        }


class _AnnotationRenderer:
    """Writes one field's annotation and default as source, collecting what they
    refer to.

    Problems are collected rather than returned early, so every unsupported part
    of an annotation is reported.
    """

    literal: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("typing"), qualname=(Identifier("Literal"),)
    )
    missing: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("pydantic"), qualname=(Identifier("MISSING"),)
    )

    def __init__(
        self,
        view: DefinitionName,
        field: FieldName | None,
        one_to_one_marker: OneToOneMarker,
    ) -> None:
        self.view = view
        self.field = field
        self.one_to_one_marker = one_to_one_marker
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
        if annotation is pydantic.MISSING:
            self.symbols.add(_AnnotationRenderer.missing)
            return _AnnotationRenderer.missing.reference()
        if isinstance(annotation, (type, typing.TypeAliasType)):
            return self._named(annotation)
        return self._unsupported(annotation)

    def render_default(
        self, default: ir.DefaultValue | ir.NoDefault
    ) -> ValueExpression | ir.NoDefault:
        match default:
            case ir.NoDefault():
                return default
            case ir.ScalarDefault(root=str() as text):
                return PythonString(text).literal()
            case ir.ScalarDefault(root=value):
                written = repr(value)
            case ir.MissingDefault():
                self.symbols.add(_AnnotationRenderer.missing)
                written = _AnnotationRenderer.missing.reference().root
            case ir.EmptyListDefault():
                written = "[]"
            case ir.EmptyDictDefault():
                written = "{}"
            case ir.EnumDefault(member=member):
                return ValueExpression.of_enum_member(self._named(type(member)), member)
        return ValueExpression(written)

    def _parameterised(
        self,
        annotation: ir.RuntimeAnnotation,
        origin: ir.RuntimeAnnotation,
        args: Sequence[ir.RuntimeAnnotation],
    ) -> TypeExpression:
        if origin is typing.Union:
            # `None` last, so a field widened with `MISSING` reads `X | MISSING | None`.
            ordered = sorted(args, key=lambda arg: arg is types.NoneType)
            return TypeExpression(" | ".join(self.render(arg).root for arg in ordered))
        if origin is typing.Literal and all(
            _AnnotationRenderer._writable_in_literal(arg) for arg in args
        ):
            self.symbols.add(_AnnotationRenderer.literal)
            values = ", ".join(self._literal_value(arg).root for arg in args)
            return TypeExpression(
                f"{_AnnotationRenderer.literal.reference().root}[{values}]"
            )
        if origin is typing.Annotated and all(
            self.one_to_one_marker.marks(marker) for marker in args[1:]
        ):
            # A generated model is not an Entity, so it leaves the one-to-one
            # marker out, as it does when the marker marks the whole field.
            return self.render(args[0])
        if origin in (typing.Literal, typing.Annotated) or not args:
            return self._unsupported(annotation)
        rendered = ", ".join(self.render(arg).root for arg in args)
        return TypeExpression(f"{self.render(origin).root}[{rendered}]")

    @staticmethod
    def _writable_in_literal(arg: ir.RuntimeAnnotation) -> bool:
        """Whether `arg` can be written in a `Literal`: a string, an integer, a
        boolean, `None`, or an enum member named by an identifier, as a narrowing
        Filter gives."""
        match arg:
            case enum.Enum():
                return narrowing.nameable(arg)
            case None:
                return True
            case _:
                return type(arg) in (str, int, bool)

    def _literal_value(self, arg: ir.RuntimeAnnotation) -> ValueExpression:
        match arg:
            case enum.Enum():
                return ValueExpression.of_enum_member(self._named(type(arg)), arg)
            case str():
                return PythonString(arg).literal()
            case _:
                return ValueExpression(repr(arg))

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

    def __init__(
        self,
        view: DefinitionName,
        module: ModulePath,
        one_to_one_marker: OneToOneMarker,
    ) -> None:
        self.view = view
        self.module = module
        self.one_to_one_marker = one_to_one_marker
        self.diagnostics: list[PlanDiagnostic] = []
        self.used = _DefinitionsUsed(view, module)

    @staticmethod
    def definition_model(
        compiled: compiler.CompiledView | compiler.CompiledPayload,
        module: ModulePath,
        one_to_one_marker: OneToOneMarker,
    ) -> _PlannedModel:
        planner = _ModelPlanner(compiled.name, module, one_to_one_marker)
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
        it declares, so the model inherits it, unless a step gave it a default,
        which only a declaration can set. Nested models subclass the default base.
        """
        fields: list[ModelField] = []
        uses: set[Symbol] = set()
        base_renderer = _AnnotationRenderer(self.view, None, self.one_to_one_marker)
        rendered_base = base_renderer.render_base(base)
        self._record(base_renderer, uses)
        nested: dict[FieldName, ModelPlan] = {}
        for field in relation.fields:
            renderer = _AnnotationRenderer(
                self.view, field.name, self.one_to_one_marker
            )
            match field:
                case _ if (
                    field.name.root in base.model_fields
                    and field.default == ir.NoDefault()
                ):
                    rendered = Inherited()
                case ir.Field():
                    rendered = renderer.render(field.annotation())
                case ir.NestedField(value=found, relation=nested_relation):
                    self.used.use(found.lookups()[-1].target)
                    model = self.plan(
                        names.nested_in(field), nested_relation, pydantic.BaseModel
                    )
                    nested[field.name] = model
                    uses.add(model.symbol)
                    rendered = field.optionality().annotate_text(
                        model.symbol.reference()
                    )
            fields.append(
                ModelField(
                    field=field,
                    type=rendered,
                    default=renderer.render_default(field.default),
                )
            )
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


class _PlannedUnions(pydantic.BaseModel, frozen=True):
    """Every Union's model, and each Union of Views' mapping function, by
    module."""

    models: Mapping[ModulePath, tuple[UnionPlan, ...]]
    functions: Mapping[ModulePath, tuple[UnionFunctionPlan, ...]]
    diagnostics: tuple[PlanDiagnostic, ...]

    @staticmethod
    def of(
        unions: Iterable[compiler.CompiledUnion],
        payload_unions: Iterable[compiler.CompiledPayloadUnion],
        layout: PythonLayout,
        mapped: Mapping[ViewName | UnionName, FunctionPlan | UnionFunctionPlan],
    ) -> _PlannedUnions:
        """`unions` and `payload_unions` have each Union after those nested in
        it, and `mapped` the planned mapping function of each View.

        A Union gets no mapping function when a Variant's could not be planned,
        which is reported already. A Union of Payloads gets none at all. A Union
        the layout gives no module is reported, and gets neither.
        """
        functions_of = dict(mapped)
        models: dict[ModulePath, list[UnionPlan]] = {}
        functions: dict[ModulePath, list[UnionFunctionPlan]] = {}
        reported: list[PlanDiagnostic] = []
        for union in unions:
            match UnionPlan.of(union, layout):
                case Ok(planned):
                    pass
                case Err(error):
                    reported.extend(error.diagnostics)
                    continue
            symbol = planned.symbol
            models.setdefault(symbol.module, []).append(planned)
            variants = tuple(
                functions_of[variant.name]
                for variant in union.variants
                if variant.name in functions_of
            )
            match layout.mappings:
                case WithoutMappings():
                    continue
                case mappings if len(variants) == len(union.variants):
                    pass
                case _:
                    continue
            match NoModuleFromLayout.check(mappings, LayoutRole.MAPPINGS, union):
                case NoModuleFromLayout() as unplaced:
                    reported.append(unplaced)
                case module:
                    function = UnionFunctionPlan.of(union, symbol, module, variants)
                    functions.setdefault(function.symbol.module, []).append(function)
                    functions_of[union.name] = function
        for payload_union in payload_unions:
            match UnionPlan.of(payload_union, layout):
                case Ok(planned):
                    models.setdefault(planned.symbol.module, []).append(planned)
                case Err(error):
                    reported.extend(error.diagnostics)
        return _PlannedUnions(
            models={module: tuple(planned) for module, planned in models.items()},
            functions={module: tuple(planned) for module, planned in functions.items()},
            diagnostics=tuple(reported),
        )


class PythonPlan(pydantic.BaseModel, frozen=True):
    """Every model and mapping function to generate, by module: View models in
    View order, then Payload models in Payload order, then the models of Unions of
    Views, then of Unions of Payloads, each Union after the Unions nested in it.

    `models` holds each View's and Payload's own model; its nested models are in
    it. `unions` holds each Union's model, and `union_functions` each Union of
    Views' mapping function. `requiring` are the modules of mapping functions that read
    a value a View requires, so they define `FunctionPlan.required_helper`.
    """

    models: Mapping[ModulePath, tuple[ModelPlan, ...]]
    unions: Mapping[ModulePath, tuple[UnionPlan, ...]]
    functions: Mapping[ModulePath, tuple[FunctionPlan, ...]]
    union_functions: Mapping[ModulePath, tuple[UnionFunctionPlan, ...]]
    requiring: frozenset[ModulePath]

    def model_modules(self) -> tuple[ModulePath, ...]:
        """Every module models go to, in module order."""
        return ModulePath.ordered(self.models.keys() | self.unions.keys())

    def function_modules(self) -> tuple[ModulePath, ...]:
        """Every module mapping functions go to, in module order."""
        return ModulePath.ordered(self.functions.keys() | self.union_functions.keys())

    @staticmethod
    def of(
        compiled: compiler.CompiledCatalog, layout: PythonLayout
    ) -> Result[PythonPlan, DiagnosticsError[PlanDiagnostic]]:
        """The plan, or every diagnostic. Each diagnostic is reported once, even
        when both a model and a mapping function run into it. A type a model field
        cannot import is reported for that field only, not again for the mapping
        function that looks it up. Models leave the catalog's one-to-one markers out.

        A View, Payload or Union the layout gives no `ModulePath` is reported, and
        not planned."""
        diagnostics: dict[PlanDiagnostic, None] = {}
        models: dict[ModulePath, list[ModelPlan]] = {}
        functions: dict[ModulePath, list[FunctionPlan]] = {}
        mapped: dict[ViewName | UnionName, FunctionPlan | UnionFunctionPlan] = {}
        for view in compiled.views:
            match NoModuleFromLayout.check(layout.models, LayoutRole.MODELS, view):
                case NoModuleFromLayout() as unplaced:
                    diagnostics[unplaced] = None
                    continue
                case module:
                    pass
            model = _ModelPlanner.definition_model(
                view, module, compiled.one_to_one_marker
            )
            diagnostics.update(dict.fromkeys(model.diagnostics))
            models.setdefault(model.plan.symbol.module, []).append(model.plan)
            match layout.mappings:
                case WithoutMappings():
                    continue
                case mappings:
                    pass
            match NoModuleFromLayout.check(mappings, LayoutRole.MAPPINGS, view):
                case NoModuleFromLayout() as unplaced:
                    diagnostics[unplaced] = None
                    continue
                case module:
                    pass
            match FunctionPlan.of(view, model.plan, module):
                case Ok(function):
                    functions.setdefault(function.symbol.module, []).append(function)
                    mapped[view.name] = function
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
            match NoModuleFromLayout.check(layout.models, LayoutRole.MODELS, payload):
                case NoModuleFromLayout() as unplaced:
                    diagnostics[unplaced] = None
                    continue
                case module:
                    pass
            model = _ModelPlanner.definition_model(
                payload, module, compiled.one_to_one_marker
            )
            diagnostics.update(dict.fromkeys(model.diagnostics))
            models.setdefault(model.plan.symbol.module, []).append(model.plan)
        unions = _PlannedUnions.of(
            compiled.unions, compiled.payload_unions, layout, mapped
        )
        diagnostics.update(dict.fromkeys(unions.diagnostics))
        plan = PythonPlan(
            models={module: tuple(planned) for module, planned in models.items()},
            unions=unions.models,
            functions={module: tuple(planned) for module, planned in functions.items()},
            union_functions=unions.functions,
            requiring=frozenset(
                module
                for module, planned in functions.items()
                if any(function.model.requires() for function in planned)
            ),
        )
        diagnostics.update(dict.fromkeys(plan._layout_diagnostics()))
        return DiagnosticsError.or_ok(diagnostics, lambda: plan)

    def _layout_diagnostics(self) -> Iterable[PlanDiagnostic]:
        for module in self.model_modules():
            yield from self._model_clashes(module)
        for module in self.function_modules():
            yield from self._function_clashes(module)
        yield from (
            ModelsAndMappingsInOneModule(module=module)
            for module in ModulePath.ordered(
                set(self.model_modules()) & set(self.function_modules())
            )
        )

    def _model_clashes(self, module: ModulePath) -> Iterable[NameClash]:
        """The names the models of `module` and their imports bind twice."""
        namespace = Namespace(Scope(module.root))
        defined: list[tuple[Symbol, AnnotationText, Iterable[Symbol]]] = [
            (
                union.symbol,
                AnnotationText(f"the model of Union {union.union.root}"),
                union.uses(),
            )
            for union in self.unions.get(module, ())
        ]
        for model in (
            nested
            for own in self.models.get(module, ())
            for nested in own.with_nested()
        ):
            match model.view:
                case ViewName():
                    definition = f"View {model.view.root}"
                case PayloadName():
                    definition = f"Payload {model.view.root}"
            defined.append(
                (model.symbol, AnnotationText(f"the model of {definition}"), model.uses)
            )
        for symbol, role, uses in defined:
            namespace.define(
                symbol.bound_name(),
                AnnotationText(f"{symbol.describe().root} ({role.root})"),
            )
            for used in uses:
                if used.module != module:
                    namespace.refer(used.bound_name(), used.imported().describe())
        return namespace.clashes()

    def _function_clashes(self, module: ModulePath) -> Iterable[NameClash]:
        """The names the mapping functions of `module`, their imports and the
        scopes of each bind twice."""
        functions = self.functions.get(module, ())
        union_functions = self.union_functions.get(module, ())
        namespace = Namespace(Scope(module.root))
        for function in (*union_functions, *functions):
            match function:
                case FunctionPlan(view=name):
                    role = f"the mapping function of View {name.root}"
                case UnionFunctionPlan(union=name):
                    role = f"the mapping function of Union {name.root}"
            namespace.define(
                function.symbol.bound_name(),
                AnnotationText(f"{function.symbol.describe().root} ({role})"),
            )
            for imported in function.imports():
                namespace.refer_to_module(imported)
        if module in self.requiring:
            namespace.define(
                FunctionPlan.required_helper,
                AnnotationText(
                    f"{module.root}.{FunctionPlan.required_helper.root} (the "
                    "function giving each value a View requires)"
                ),
            )
        yield from namespace.clashes()
        for function in functions:
            yield from Namespace.of_function(function).clashes()
        for union_function in union_functions:
            yield from Namespace.of_union_function(union_function).clashes()


class Namespace:
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
    def of_function(function: FunctionPlan) -> Namespace:
        """A function's parameters, and the module its body builds models from."""
        namespace = Namespace(function.scope())
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

    @staticmethod
    def of_union_function(function: UnionFunctionPlan) -> Namespace:
        """A Union function's parameters and variable, the functions it calls by
        name, and the modules it refers to."""
        namespace = Namespace(function.scope())
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
        namespace.define(
            UnionFunctionPlan.found,
            AnnotationText("the variable holding the mapped Variant"),
        )
        for call in function.variants:
            match function.called(call):
                case Identifier() as name:
                    namespace.refer(name, call.function.describe())
                case Symbol() as symbol:
                    namespace.refer_to_module(symbol.module)
        namespace.refer_to_module(function.model.module)
        return namespace

    def clashes(self) -> Iterable[NameClash]:
        for name, origins in self.bound.items():
            if len(origins) > 1:
                yield NameClash(
                    scope=self.scope,
                    name=name,
                    origins=tuple(sorted(origins, key=lambda text: text.root)),
                )
