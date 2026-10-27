"""Renders Django query code for every View and Union whose Source is an Entity, so
a Django project can fill View models from its database while keeping its own
filtering, ordering and pagination.

Per View it writes two functions. `<view>_queryset` makes a QuerySet of the
Source's Django model keep the rows the View's Filters keep, and load every Entity
the View looks up, in the same query. `<view>_from_row` builds the View model from
one row of that QuerySet. Per Union it writes the same two functions:
`<union>_queryset` keeps the rows any Variant keeps, loads what every Variant looks
up, and annotates each row with the name of the first Variant that keeps it;
`<union>_from_row` calls that Variant's `<view>_from_row`. Per Entity it writes a
QuerySet with a method `as_<view>()` per View and Union over the Entity, and a
Manager, which is the Entity's Django model's manager `objects`. The method yields
View models from the QuerySet as filtered, ordered and sliced so far, so a list
costs one query. An Entity whose Django model the application already has, see
`ExistingDjangoModel`, gets no QuerySet or Manager, since its model's manager is
its own; its Views and Unions still get their two functions.

The View's Filters apply before a slice the QuerySet has taken: `[:10]` gives the
first ten rows the View keeps, not those of the first ten rows it keeps. A Filter
has Python's two-valued null semantics, as in the InMemoryViewEngine: `None == x`
is false and `None != x` is true, also when `x` is another field, so `None == None`
is true. SQL's three-valued comparisons get explicit `IS NULL` terms for that; no
negation is left around a comparison that may be NULL. An enum literal is
compared with its member's value, which is what Django stores for an enum field
it has a built-in rule for.

`as_<view>()` loads every field the View needs: it undoes `only()` and `defer()`,
which would otherwise cost a query per row or conflict with `select_related`. It
cannot follow `values()`, `values_list()` or a combinator such as `union()`, whose
rows are not model instances; Django raises when it is called after them.

The Manager overrides the Manager methods that return a QuerySet (`all`, `filter`,
`exclude`, `order_by`) to return the Entity's QuerySet, so a type checker sees
`as_<view>()` on `Model.objects.filter(...)`.

A row is validated non-strictly, so a View model with a strict base accepts what
Django returns, e.g. a `str` for an enum field. A field typed by a `Literal` of
enum members, e.g. one a Filter narrows, is read through its enum, since pydantic
does not take a member's value for a `Literal` of a plain `enum.Enum`. A field
lifted out of an Entity that declares it as a required RootModel whose column
may be NULL, e.g. `Rank(RootModel[int | None])`, is validated as that RootModel,
so a NULL of an Entity found is `Rank(None)`, not the `None` of no Entity found.
A field the View requires that is `None` fails that validation with a
`pydantic.ValidationError` naming the View model and the field. An Entity the
View requires that the row does not have raises `ValueError` through the
function `_required` the module defines, as in a module of mapping functions: a
field read through it may be optional, so validation alone would not catch it.
Both are a `ValueError`.

A one-to-one Reference is followed backwards through the reverse accessor the
Django models give it. Django raises for a reverse accessor that found no row, so
the row reader asks `hasattr` first. Following a one-to-one Reference and then
back over it returns to the row it came from: Django fills that in itself, and
drops what it selected related past it, so the reader resumes at that row, and
selects related from there.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two agree (the differential test in `test_queries.py`
checks this). The generated module imports the Django models only for type
checking, and refers to them only in quoted annotations, because the models
import their Managers from it.
"""

import enum
import typing
from collections.abc import Iterable, Iterator, Mapping, Sequence

import pydantic
from pydantic.fields import FieldInfo
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler, narrowing
from functional_relational_python.compiler.model_names import FunctionName
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
)
from functional_relational_python.outputs.django import conditions
from functional_relational_python.outputs.django.models import (
    DjangoModels,
    DjangoModelsRenderer,
    FieldTypes,
)
from functional_relational_python.outputs.python.mappings import (
    Expression,
    Indent,
    LeftOut,
    MappingsRenderer,
)
from functional_relational_python.outputs.python.plan import (
    FunctionPlan,
    ModelPlan,
    NameClash,
    Namespace,
    PlacedModels,
    Scope,
    Unimportable,
)
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    LineLength,
    ModulePath,
    PythonString,
    Symbol,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.unions import UnionName
from functional_relational_python.pipeline.views import ViewName
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)
from functional_relational_python.support.filesystem import FileText
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)
from functional_relational_python.support.identifier import Identifier


class QueriesIntoDjangoModelsModule(pydantic.BaseModel, Diagnostic, frozen=True):
    """`DjangoModels.queries` is `DjangoModels.module`, so one file would
    overwrite the other."""

    module: ModulePath

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.module.root}: Django models and Django queries cannot be "
            "generated into the same module. Set DjangoModels.queries to another "
            "module."
        )


class ViewModelIntoQueriesModule(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View's or Union's model would be generated into the module of Django
    queries, so one file would overwrite the other."""

    view: ViewName | UnionName
    module: ModulePath

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: its model would be generated into "
            f"{self.module.root}, the module of Django queries. Generate it into "
            "another module."
        )


class ViewModelIntoDjangoModelsModule(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View's or Union's model would be generated into the module of Django
    models, which imports the Django queries, which import that model."""

    view: ViewName | UnionName
    module: ModulePath

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: its model would be generated into "
            f"{self.module.root}, the module of Django models, which the Django "
            "queries importing the model are imported by. Generate it into another "
            "module."
        )


type DjangoQueriesDiagnostic = (
    QueriesIntoDjangoModelsModule
    | ViewModelIntoQueriesModule
    | ViewModelIntoDjangoModelsModule
    | Unimportable
    | NameClash
)


class RelatedPath(pydantic.RootModel[str], frozen=True):
    """A chain of relations `select_related` follows, e.g. `department__company`:
    foreign keys, and reverse accessors of one-to-one References."""

    @staticmethod
    def of(relations: Sequence[FieldName]) -> RelatedPath:
        return RelatedPath("__".join(relation.root for relation in relations))


class QueryNames(pydantic.BaseModel, frozen=True):
    """The names of the query code of a View or Union, from the name of its
    mapping function."""

    function: FunctionName

    def queryset_function(self) -> Identifier:
        return Identifier(f"{self.function.root}_queryset")

    def from_row_function(self) -> Identifier:
        return Identifier(f"{self.function.root}_from_row")

    def method(self) -> Identifier:
        """The method of the Source's QuerySet and Manager yielding its models."""
        return Identifier(f"as_{self.function.root}")


class ViewQueries(pydantic.BaseModel, frozen=True):
    """The query code of one View whose Source is an Entity: `model` is its View
    model, `source` the Django model of its Source."""

    view: compiler.CompiledView
    model: Symbol
    source: Symbol

    @staticmethod
    def of(
        view: compiler.CompiledView, config: DjangoModels, module: ModulePath
    ) -> ViewQueries:
        """`module` is the one the View's model is generated into."""
        return ViewQueries(
            view=view,
            model=Symbol(module=module, qualname=(view.names.model.name.identifier(),)),
            source=config.model_of(view.source),
        )

    def entity(self) -> type[pydantic.BaseModel]:
        return self.view.source

    def describe(self) -> AnnotationText:
        return AnnotationText(f"View {self.view.name.root}")

    def names(self) -> QueryNames:
        return QueryNames(function=self.view.names.function)

    def related_paths(self, config: DjangoModels) -> tuple[RelatedPath, ...]:
        """Every chain of relations to the Entities the View looks up, in the
        order they are first looked up."""
        return tuple(
            dict.fromkeys(
                RelatedPath.of(path)
                for path in ViewQueries._paths(
                    self.view.relation,
                    _Trail.start(self.view.source),
                    ((),),
                    config,
                )
            )
        )

    @staticmethod
    def _paths(
        relation: ir.Relation,
        trail: _Trail,
        paths: tuple[tuple[FieldName, ...], ...],
        config: DjangoModels,
    ) -> Iterator[tuple[FieldName, ...]]:
        """`trail` is the way to the subject of `relation`, and `paths` the chain
        of relations to each Entity on it, the subject's last. A Hop back needs
        no relation: Django fills it in."""
        for field in relation.fields:
            at, here = trail, paths
            for lookup in field.value.lookups():
                hop, at = at.hop(lookup)
                match hop:
                    case _Follow():
                        path = (
                            *here[-1],
                            DjangoQueriesRenderer.related(lookup, config),
                        )
                        yield path
                        here = (*here, path)
                    case _Back():
                        here = here[:-1]
            match field:
                case ir.Field():
                    pass
                case ir.NestedField(relation=nested):
                    yield from ViewQueries._paths(nested, at, here, config)


class UnionQueries(pydantic.BaseModel, frozen=True):
    """The query code of one Union whose Source is an Entity: `model` is its
    model, `source` the Django model of its Source, and `leaves` the query code
    of the Views a row may be a result of, in the order they are tried."""

    union: compiler.CompiledUnion
    model: Symbol
    source: Symbol
    leaves: tuple[ViewQueries, ...]

    @staticmethod
    def of(
        union: compiler.CompiledUnion,
        config: DjangoModels,
        placed: PlacedModels,
    ) -> UnionQueries:
        """`placed` has the module of the Union's model, and of each of its
        leaves'."""
        return UnionQueries(
            union=union,
            model=Symbol(
                module=placed.module_of(union),
                qualname=(union.names.model.identifier(),),
            ),
            source=config.model_of(union.source),
            leaves=tuple(
                ViewQueries.of(leaf, config, placed.module_of(leaf))
                for leaf in union.leaves()
            ),
        )

    def entity(self) -> type[pydantic.BaseModel]:
        return self.union.source

    def describe(self) -> AnnotationText:
        return AnnotationText(f"Union {self.union.name.root}")

    def names(self) -> QueryNames:
        return QueryNames(function=self.union.names.function)

    def related_paths(self, config: DjangoModels) -> tuple[RelatedPath, ...]:
        """Every Variant's chains of relations, each once, in Variant order."""
        return tuple(
            dict.fromkeys(
                path for leaf in self.leaves for path in leaf.related_paths(config)
            )
        )


class _Follow(pydantic.BaseModel, frozen=True):
    """Follows `lookup` from the Entity found so far."""

    lookup: ir.EntityLookup


class _Back(pydantic.BaseModel, frozen=True):
    """`lookup` follows the one-to-one Reference the lookup before it followed,
    the other way, so it finds the Entity found before that one.

    Django fills in the way back when it selects related: the row it reaches
    is the one it came from, not one of its own, so what it selects related
    past it is lost. So the way back is not followed; reading resumes at the
    Entity found before.
    """

    lookup: ir.EntityLookup


type _Hop = _Follow | _Back


class _Trail(pydantic.BaseModel, frozen=True):
    """The way from the row a query reads to an Entity `subject`: the lookups
    followed and not returned over, each with the Entity it was followed
    from. It spans nested relations, since Django fills in a way back across
    them too."""

    subject: type[pydantic.BaseModel]
    followed: tuple[tuple[type[pydantic.BaseModel], ir.EntityLookup], ...]

    @staticmethod
    def start(subject: type[pydantic.BaseModel]) -> _Trail:
        return _Trail(subject=subject, followed=())

    def hop(self, lookup: ir.EntityLookup) -> tuple[_Hop, _Trail]:
        """`lookup` from `subject`: a Hop back when it returns over the Reference
        the last lookup followed, and the trail after it."""
        match self.followed:
            case (*rest, (before, previous)) if _Trail._returns(
                before, previous, lookup
            ):
                return _Back(lookup=lookup), _Trail(
                    subject=before, followed=tuple(rest)
                )
            case _:
                return _Follow(lookup=lookup), _Trail(
                    subject=lookup.target,
                    followed=(*self.followed, (self.subject, lookup)),
                )

    @staticmethod
    def _returns(
        before: type[pydantic.BaseModel],
        previous: ir.EntityLookup,
        lookup: ir.EntityLookup,
    ) -> bool:
        """Whether `lookup` follows backwards the Reference `previous` followed
        from an Entity `before`."""
        match previous, lookup:
            case ir.ReverseLookup(via=via), ir.Lookup(reference=reference):
                return reference.field == via
            case ir.Lookup(reference=reference), ir.ReverseLookup(via=via):
                return via == reference.field and lookup.target is before
            case _:
                return False


type ReadQueries = ViewQueries | UnionQueries
"""The query code of a View or a Union: what an `as_<name>()` method reads."""


class DjangoQueriesRenderer(pydantic.BaseModel, frozen=True):
    """Writes the query code of every View and Union whose Source is an Entity,
    and a QuerySet and Manager per Entity, into one module.

    The output is ruff-format stable: what ruff format could split is split as it
    would, or exploded with a trailing comma.
    """

    typing_module: typing.ClassVar[ModulePath] = ModulePath("typing")
    iterator_module: typing.ClassVar[ModulePath] = ModulePath("collections.abc")
    django_models: typing.ClassVar[ModulePath] = ModulePath("django.db.models")
    expressions: typing.ClassVar[ModulePath] = ModulePath(
        "django.db.models.expressions"
    )
    queryset_parameter: typing.ClassVar[Identifier] = Identifier("queryset")
    kept_variable: typing.ClassVar[Identifier] = Identifier("kept")
    """The QuerySet a queryset function filters, once it has prepared it."""
    row_parameter: typing.ClassVar[Identifier] = Identifier("row")
    as_manager: typing.ClassVar[Identifier] = Identifier("as_manager")
    map_builtin: typing.ClassVar[Identifier] = Identifier("map")
    vars_builtin: typing.ClassVar[Identifier] = Identifier("vars")
    required_helper: typing.ClassVar[Identifier] = FunctionPlan.required_helper
    """The function the module defines when a View requires a lookup to find an
    Entity, as a module of mapping functions does."""
    variant_annotation: typing.ClassVar[Identifier] = Identifier("_variant")
    """The annotation naming the Variant a row of a Union is a result of. No
    Entity field starts with `_`, so it names no field of a Django model."""

    header: typing.ClassVar[FileText] = FileText(
        f"{GeneratedFiles.header.root}"
        "# Regenerate with `frp generate`; `frp check` fails when this is stale.\n"
        "# Fills each View over an Entity from the Entity's Django model.\n"
    )

    graph: EntityGraph
    config: DjangoModels
    queries: tuple[ViewQueries, ...]
    unions: tuple[UnionQueries, ...]
    classes: Mapping[type, Symbol]
    """Every class the code refers to: the enum of a Filter's literal, and every
    class a field is read through."""
    writer: conditions.ConditionWriter
    """Writes the Filters of every View, once for all of them."""

    @staticmethod
    def related(lookup: ir.EntityLookup, config: DjangoModels) -> FieldName:
        """The relation `select_related` follows for `lookup`: the foreign key of
        its Reference, or for a ReverseLookup that Reference's reverse
        accessor."""
        match lookup:
            case ir.Lookup():
                return DjangoModelsRenderer.foreign_key_name(lookup.reference.field)
            case ir.ReverseLookup():
                return config.reverse_accessor_of(lookup.target, lookup.via)

    @staticmethod
    def render(
        graph: EntityGraph,
        config: DjangoModels,
        views: Iterable[compiler.CompiledView],
        placed: PlacedModels,
        unions: Iterable[compiler.CompiledUnion] = (),
    ) -> Result[GeneratedFiles, DiagnosticsError[DjangoQueriesDiagnostic]]:
        """The module of query code, or every diagnostic.

        `placed` has the module of each View's and Union's model, and `views`
        has every View a Union has as a Variant. Views and Unions over a Source
        that is not an Entity get no query code. QuerySets are in Entity name
        order.
        """
        views = tuple(views)
        unions = tuple(unions)
        queries = tuple(
            ViewQueries.of(view, config, placed.module_of(view))
            for view in views
            if view.source in graph.entities
        )
        classes: dict[type, Symbol] = {}
        unimportable: list[DjangoQueriesDiagnostic] = []
        for query in queries:
            for used, field in dict.fromkeys(
                _ClassesUsed.by(
                    query.view, graph, FieldTypes(overrides=config.field_types)
                )
            ):
                match Symbol.of(used):
                    case Ok(symbol):
                        classes[used] = symbol
                    case Err():
                        unimportable.append(
                            Unimportable(
                                view=query.view.name,
                                field=field,
                                annotation=AnnotationText.of(used),
                            )
                        )
        renderer = DjangoQueriesRenderer(
            graph=graph,
            config=config,
            queries=queries,
            unions=tuple(
                UnionQueries.of(union, config, placed)
                for union in unions
                if union.source in graph.entities
            ),
            classes=classes,
            writer=conditions.ConditionWriter(
                columns=DjangoModelsRenderer.columns(
                    graph, FieldTypes(overrides=config.field_types)
                ),
                models=Expression(DjangoQueriesRenderer.django_models.root),
                enums={
                    used: symbol.qualified()
                    for used, symbol in classes.items()
                    if issubclass(used, enum.Enum)
                },
            ),
        )
        entities = sorted(
            (entity for entity in graph.entities if config.generates(entity)),
            key=lambda entity: entity.__name__,
        )
        return DiagnosticsError.or_ok(
            (
                *DjangoQueriesRenderer._modules(config, (*views, *unions), placed),
                *unimportable,
                *renderer._clashes(entities),
            ),
            lambda: GeneratedFiles(
                files=(
                    GeneratedFile(
                        path=config.queries.file_path(),
                        text=renderer._module(entities),
                    ),
                )
            ),
        )

    @staticmethod
    def _modules(
        config: DjangoModels,
        defined: Iterable[compiler.CompiledView | compiler.CompiledUnion],
        placed: PlacedModels,
    ) -> Iterator[DjangoQueriesDiagnostic]:
        """The Django models, the Django queries and the View and Union models
        each need their own module."""
        if config.queries == config.module:
            yield QueriesIntoDjangoModelsModule(module=config.module)
        for compiled in defined:
            module = placed.module_of(compiled)
            if module == config.queries:
                yield ViewModelIntoQueriesModule(view=compiled.name, module=module)
            if module == config.module:
                yield ViewModelIntoDjangoModelsModule(view=compiled.name, module=module)

    def _read(self) -> tuple[ReadQueries, ...]:
        return (*self.queries, *self.unions)

    def _imports(self) -> tuple[ModulePath, ...]:
        """The modules imported when the module runs."""
        return (
            DjangoQueriesRenderer.typing_module,
            DjangoQueriesRenderer.django_models,
            DjangoQueriesRenderer.expressions,
            *((DjangoQueriesRenderer.iterator_module,) if self._read() else ()),
            *(query.model.module for query in self._read()),
            *(symbol.module for symbol in self.classes.values()),
        )

    def _django_model_modules(self) -> tuple[ModulePath, ...]:
        """The modules of the Django models, generated and existing, imported only
        for type checking."""
        return ModulePath.ordered(
            (
                self.config.module,
                *(
                    existing.module
                    for entity, existing in self.config.existing.items()
                    if entity in self.graph.entities
                ),
            )
        )

    def _clashes(
        self, entities: Sequence[type[pydantic.BaseModel]]
    ) -> Iterator[NameClash]:
        """Names bound twice. Entities named alike are left to the Django models
        to report, so their QuerySets and Managers are counted once."""
        module = Namespace(Scope(self.config.queries.root))
        for query in self._read():
            for function, role in (
                (query.names().queryset_function(), "QuerySet function"),
                (query.names().from_row_function(), "row function"),
            ):
                module.define(
                    function,
                    AnnotationText(
                        f"{self.config.queries.root}.{function.root} "
                        f"(the {role} of {query.describe().root})"
                    ),
                )
        named = {entity.__name__: entity for entity in reversed(entities)}
        for entity in named.values():
            for symbol, role in (
                (self.config.query_set_of(entity), "QuerySet"),
                (self.config.manager_of(entity), "Manager"),
            ):
                module.define(
                    symbol.bound_name(),
                    AnnotationText(
                        f"{symbol.describe().root} (the {role} of Entity "
                        f"{entity.__qualname__})"
                    ),
                )
        for imported in (*self._imports(), *self._django_model_modules()):
            module.refer_to_module(imported)
        module.refer(
            DjangoQueriesRenderer.map_builtin, AnnotationText("the built-in map")
        )
        if self.unions:
            module.refer(
                DjangoQueriesRenderer.vars_builtin, AnnotationText("the built-in vars")
            )
        if self._requiring():
            module.define(
                DjangoQueriesRenderer.required_helper,
                AnnotationText(
                    f"{self.config.queries.root}."
                    f"{DjangoQueriesRenderer.required_helper.root} (the function "
                    "raising for a value a View requires)"
                ),
            )
        yield from module.clashes()
        for query in self._read():
            yield from self._local_clashes(query)
        for entity in named.values():
            query_set = self.config.query_set_of(entity)
            methods = Namespace(Scope(query_set.describe().root))
            methods.refer(
                DjangoQueriesRenderer.as_manager,
                AnnotationText(
                    f"{DjangoQueriesRenderer.django_models.root}.QuerySet."
                    f"{DjangoQueriesRenderer.as_manager.root}"
                ),
            )
            for query in self._over(entity):
                method = query.names().method()
                methods.define(
                    method,
                    AnnotationText(
                        f"{query_set.describe().root}.{method.root} (the method of "
                        f"{query.describe().root})"
                    ),
                )
            yield from methods.clashes()

    def _local_clashes(self, query: ReadQueries) -> Iterator[NameClash]:
        """Names the functions of `query` bind that shadow a module they refer
        to."""
        class_modules = tuple(symbol.module for symbol in self.classes.values())
        row = Namespace(
            Scope(
                f"{self.config.queries.root}.{query.names().from_row_function().root}"
            )
        )
        row.define(
            DjangoQueriesRenderer.row_parameter,
            AnnotationText(f"the parameter for {query.source.describe().root}"),
        )
        for module in (query.model.module, *class_modules):
            row.refer_to_module(module)
        yield from row.clashes()
        queryset = Namespace(
            Scope(
                f"{self.config.queries.root}.{query.names().queryset_function().root}"
            )
        )
        queryset.define(
            DjangoQueriesRenderer.queryset_parameter,
            AnnotationText(
                f"the parameter for the QuerySet of {query.source.describe().root}"
            ),
        )
        queryset.define(
            DjangoQueriesRenderer.kept_variable,
            AnnotationText("the variable holding the QuerySet of kept rows"),
        )
        for module in (DjangoQueriesRenderer.django_models, *class_modules):
            queryset.refer_to_module(module)
        yield from queryset.clashes()

    def _over(self, entity: type[pydantic.BaseModel]) -> Iterator[ReadQueries]:
        """The Views, then the Unions, whose Source is `entity`, in order."""
        return (query for query in self._read() if query.entity() is entity)

    def _module(self, entities: Sequence[type[pydantic.BaseModel]]) -> FileText:
        imports = "\n".join(
            "".join(f"import {module.root}\n" for module in section)
            for section in ModulePath.import_sections(self._imports())
        )
        type_checking = "".join(
            (
                f"\nif {DjangoQueriesRenderer.typing_module.root}.TYPE_CHECKING:\n",
                *(
                    f"    import {module.root}\n"
                    for module in self._django_model_modules()
                ),
            )
        )
        functions = "".join(
            f"\n\n{self._queryset_function(query).root}"
            f"\n\n{self._from_row_function(query).root}"
            for query in self.queries
        )
        union_functions = "".join(
            f"\n\n{self._union_queryset_function(union).root}"
            f"\n\n{DjangoQueriesRenderer._union_from_row_function(union).root}"
            for union in self.unions
        )
        classes = "".join(
            f"\n\n{self._query_set(entity).root}\n\n{self._manager(entity).root}"
            for entity in entities
        )
        helper = (
            f"\n\n{MappingsRenderer.required_function().root}"
            if self._requiring()
            else ""
        )
        return FileText(
            f"{DjangoQueriesRenderer.header.root}\n{imports}{type_checking}"
            f"{helper}{functions}{union_functions}{classes}"
        )

    @staticmethod
    def _queryset_annotation(query: ReadQueries) -> Expression:
        """Quoted, because the Django models are imported only for type
        checking."""
        return Expression(
            f'"{DjangoQueriesRenderer.django_models.root}.QuerySet'
            f'[{query.source.qualified().root}]"'
        )

    def _prepared(self, query: ReadQueries) -> Expression:
        """The QuerySet loading every field and every Entity `query` reads, at the
        indentation of a function body."""
        prepared = f"{DjangoQueriesRenderer.queryset_parameter.root}.defer(None)"
        paths = query.related_paths(self.config)
        if not paths:
            return Expression(prepared)
        return Indent(4).exploded(
            Expression(f"{prepared}.select_related("),
            (Expression(PythonString(path.root).literal().root) for path in paths),
            Expression(")"),
        )

    @staticmethod
    def _queryset_signature(query: ReadQueries) -> FileText:
        annotation = DjangoQueriesRenderer._queryset_annotation(query).root
        return FileText(
            f"def {query.names().queryset_function().root}(\n"
            f"    {DjangoQueriesRenderer.queryset_parameter.root}: {annotation},\n"
            f") -> {annotation}:\n"
        )

    def _queryset_function(self, query: ViewQueries) -> FileText:
        prepared = self._prepared(query)
        signature = DjangoQueriesRenderer._queryset_signature(query).root
        writer = self.writer
        match DjangoQueriesRenderer._kept_by(writer, query.view):
            case _KeepsEvery():
                return FileText(f"{signature}    return {prepared.root}\n")
            case condition:
                pass
        kept = DjangoQueriesRenderer.kept_variable.root
        added = DjangoQueriesRenderer._added(writer.written(condition, Indent(8)))
        return FileText(
            f"{signature}    {kept} = {prepared.root}\n{added.root}    return {kept}\n"
        )

    @staticmethod
    def _added(condition: Expression) -> FileText:
        """`kept.query.add_q(<condition>)`, a statement of a function body.

        Unlike `filter()`, `add_q()` takes a QuerySet that has been sliced, and
        keeps the slice, so that it is a slice of the kept rows.
        """
        kept = DjangoQueriesRenderer.kept_variable.root
        added = Indent(4).exploded(
            Expression(f"{kept}.query.add_q("), (condition,), Expression(")")
        )
        return FileText(f"    {added.root}\n")

    @staticmethod
    def _kept_by(
        writer: conditions.ConditionWriter, view: compiler.CompiledView
    ) -> conditions.Q | _KeepsEvery:
        """True for the rows `view`'s Filters all keep, or `_KeepsEvery` when it
        has none."""
        match view.relation.filters:
            case ():
                return _KeepsEvery()
            case (single,):
                return writer.kept(single, view.source)
            case filters:
                return writer.kept(ir.FilterAnd(terms=filters), view.source)

    def _union_queryset_function(self, union: UnionQueries) -> FileText:
        """Annotates each row with the name of the first Variant whose Filters keep
        it, and keeps only the rows one keeps: those with a name. Only the last
        Variant may keep every row, the compiler checks, so it is the default."""
        kept = DjangoQueriesRenderer.kept_variable.root
        models = DjangoQueriesRenderer.django_models.root
        writer = self.writer
        cases: list[Expression] = []
        default: list[Expression] = []
        for leaf in union.leaves:
            name = Expression(
                f"{models}.Value({PythonString(leaf.view.name.root).literal().root})"
            )
            match DjangoQueriesRenderer._kept_by(writer, leaf.view):
                case _KeepsEvery():
                    default.append(Expression(f"default={name.root}"))
                    break
                case condition:
                    pass
            cases.append(
                Indent(12).exploded(
                    Expression(f"{models}.When("),
                    (
                        writer.written(condition, Indent(16)),
                        Expression(f"then={name.root}"),
                    ),
                    Expression(")"),
                )
            )
        annotated = Indent(4).exploded(
            Expression(f"{kept}.annotate("),
            (
                Indent(8).exploded(
                    Expression(
                        f"{DjangoQueriesRenderer.variant_annotation.root}="
                        f"{models}.Case("
                    ),
                    (*cases, *default),
                    Expression(")"),
                ),
            ),
            Expression(")"),
        )
        named = conditions.QLookup(
            entries=(
                conditions.QEntry(
                    lookup=Expression(
                        f"{DjangoQueriesRenderer.variant_annotation.root}__isnull"
                    ),
                    value=conditions.NullTest.NOT_NULL.written(),
                ),
            )
        )
        added = (
            ""
            if default
            else DjangoQueriesRenderer._added(writer.written(named, Indent(8))).root
        )
        return FileText(
            f"{DjangoQueriesRenderer._queryset_signature(union).root}"
            f"    {kept} = {self._prepared(union).root}\n"
            f"    {kept} = {annotated.root}\n"
            f"{added}"
            f"    return {kept}\n"
        )

    @staticmethod
    def _returns(model: Symbol) -> FileText:
        """The line closing a row function's signature, `) -> <model>:`. Like ruff
        format: parenthesized onto its own line, if that makes it fit."""
        qualified = model.qualified().root
        returns = f") -> {qualified}:"
        if len(returns) > LineLength.default().root >= len(qualified) + 4:
            return FileText(f") -> (\n    {qualified}\n):")
        return FileText(returns)

    def _from_row_function(self, query: ViewQueries) -> FileText:
        row = Expression(DjangoQueriesRenderer.row_parameter.root)
        model = query.model.qualified().root
        reader = _RowReader(
            graph=self.graph, config=self.config, classes=self.classes, indent=Indent(8)
        )
        start = _Reached.start(query.view.source, row)
        read = reader.fields(query.view.relation, start)
        unread = "".join(
            statement.root
            for statement in reader.unread_requirements(query.view.relation, start, ())
        )
        validated = Indent(4).exploded(
            Expression(f"{model}.model_validate("),
            (read, Expression("strict=False")),
            Expression(")"),
        )
        return FileText(
            f"def {query.names().from_row_function().root}(\n"
            f'    {row.root}: "{query.source.qualified().root}",\n'
            f"{DjangoQueriesRenderer._returns(query.model).root}\n"
            f"{unread}"
            f"    return {validated.root}\n"
        )

    def _requiring(self) -> bool:
        """Whether a View requires a lookup to find an Entity, so that the module
        needs `_required`."""
        return any(
            ModelPlan.required_lookup(lookup)
            for query in self.queries
            for lookup in _RowReader.every_lookup(query.view.relation)
        )

    @staticmethod
    def _union_from_row_function(union: UnionQueries) -> FileText:
        """Calls the row function of the Variant the row is annotated with. The
        QuerySet keeps only rows a Variant keeps, so a row not annotated with an
        earlier one is the last one's."""
        row = DjangoQueriesRenderer.row_parameter.root
        variant = (
            f"{DjangoQueriesRenderer.vars_builtin.root}({row})"
            f'["{DjangoQueriesRenderer.variant_annotation.root}"]'
        )
        *earlier, last = union.leaves
        cases = "".join(
            f"        case {PythonString(leaf.view.name.root).literal().root}:\n"
            f"            return {leaf.names().from_row_function().root}({row})\n"
            for leaf in earlier
        )
        return FileText(
            f"def {union.names().from_row_function().root}(\n"
            f'    {row}: "{union.source.qualified().root}",\n'
            f"{DjangoQueriesRenderer._returns(union.model).root}\n"
            f"    match {variant}:\n"
            f"{cases}"
            "        case _:\n"
            f"            return {last.names().from_row_function().root}({row})\n"
        )

    def _class_statement(self, name: Identifier, base: Expression) -> FileText:
        """`class <name>(<base>):`, where `base` subscripts a Django class with the
        quoted Django model, split over lines like ruff format does."""
        statement = f"class {name.root}({base.root}):\n"
        if len(statement) - 1 <= LineLength.default().root:
            return FileText(statement)
        split = DjangoQueriesRenderer._subscript(Indent(4), base, Expression(","))
        return FileText(f"class {name.root}(\n{split.root}\n):\n")

    @staticmethod
    def _subscript(
        indent: Indent, subscripted: Expression, after: Expression
    ) -> Expression:
        """A line at `indent` holding `subscripted`, e.g. `X[Y]`, then `after`; its
        argument on a line of its own when it does not fit, like ruff format."""
        line = f"{indent.pad().root}{subscripted.root}{after.root}"
        if len(line) <= LineLength.default().root:
            return Expression(line)
        origin, _, argument = subscripted.root.partition("[")
        return Expression(
            f"{indent.pad().root}{origin}[\n{indent.deeper().pad().root}"
            f"{argument.removesuffix(']')}\n{indent.pad().root}]{after.root}"
        )

    def _iterator_of(self, query: ReadQueries) -> Expression:
        """The method's return annotation line, `    ) -> Iterator[<model>]:`."""
        return DjangoQueriesRenderer._subscript(
            Indent(4),
            Expression(
                f") -> {DjangoQueriesRenderer.iterator_module.root}.Iterator"
                f"[{query.model.qualified().root}]"
            ),
            Expression(":"),
        )

    @staticmethod
    def _method_docstring(query: ReadQueries) -> FileText:
        """For a View with Filters, or a Union, what its Filters do to a slice the
        QuerySet has taken: they apply first, so it is a slice of the kept rows.
        A View without Filters keeps every row, so it needs none."""
        match query:
            case ViewQueries(view=view) if not view.relation.filters:
                return FileText("")
            case ViewQueries() | UnionQueries():
                return FileText(
                    '        """The models of the rows of this QuerySet that the '
                    "Filters keep.\n"
                    "\n"
                    "        The Filters apply before a slice the QuerySet has "
                    "taken: `[:10]` gives\n"
                    "        the first ten rows they keep.\n"
                    '        """\n'
                )

    def _query_set(self, entity: type[pydantic.BaseModel]) -> FileText:
        model = self.config.model_of(entity).qualified().root
        statement = self._class_statement(
            self.config.query_set_of(entity).bound_name(),
            Expression(
                f'{DjangoQueriesRenderer.django_models.root}.QuerySet["{model}"]'
            ),
        )
        methods = "\n".join(
            f"    def {query.names().method().root}(\n"
            "        self,\n"
            f"{self._iterator_of(query).root}\n"
            f"{DjangoQueriesRenderer._method_docstring(query).root}"
            f"        return {DjangoQueriesRenderer.map_builtin.root}(\n"
            f"            {query.names().from_row_function().root},\n"
            f"            {query.names().queryset_function().root}(self),\n"
            "        )\n"
            for query in self._over(entity)
        )
        return FileText(
            f"{statement.root}{methods}" if methods else f"{statement.root}    pass\n"
        )

    def _manager(self, entity: type[pydantic.BaseModel]) -> FileText:
        """The Manager, with the Manager methods returning a QuerySet typed to
        return the Entity's QuerySet, and its `as_<view>()` methods."""
        model = self.config.model_of(entity).qualified().root
        query_set = self.config.query_set_of(entity).bound_name().root
        statement = self._class_statement(
            self.config.manager_of(entity).bound_name(),
            Expression(
                f'{DjangoQueriesRenderer.django_models.root}.Manager["{model}"]'
            ),
        )
        combinable = f"{DjangoQueriesRenderer.expressions.root}.Combinable"
        methods = [
            (
                "    @typing.override\n"
                "    def get_queryset(\n"
                "        self,\n"
                f"    ) -> {query_set}:\n"
                f"        return {query_set}(\n"
                "            self.model,\n"
                "            using=self._db,\n"
                "        )\n"
            ),
            (
                "    @typing.override\n"
                "    def all(\n"
                "        self,\n"
                f"    ) -> {query_set}:\n"
                "        return self.get_queryset()\n"
            ),
            *(
                "    @typing.override\n"
                f"    def {method}(\n"
                "        self,\n"
                "        *args: object,\n"
                "        **kwargs: object,\n"
                f"    ) -> {query_set}:\n"
                f"        return self.get_queryset().{method}(*args, **kwargs)\n"
                for method in ("filter", "exclude")
            ),
            (
                "    @typing.override\n"
                "    def order_by(\n"
                "        self,\n"
                f"        *field_names: str | {combinable},\n"
                f"    ) -> {query_set}:\n"
                "        return self.get_queryset().order_by(*field_names)\n"
            ),
            *(
                f"    def {query.names().method().root}(\n"
                "        self,\n"
                f"{self._iterator_of(query).root}\n"
                f"        return self.get_queryset().{query.names().method().root}()\n"
                for query in self._over(entity)
            ),
        ]
        return FileText(f"{statement.root}{'\n'.join(methods)}")


class _ClassesUsed:
    """The classes a View's query code refers to, each with the field read
    through it, or `None` for the enum of a Filter's literal."""

    @staticmethod
    def by(
        view: compiler.CompiledView, graph: EntityGraph, field_types: FieldTypes
    ) -> Iterator[tuple[type, FieldName | None]]:
        """The enum of every Filter's enum literal, and the class every field is
        read through, nested ones included."""
        for kept in view.relation.filters:
            for enum_class in conditions.ConditionWriter.enums_in(kept):
                yield enum_class, None
        yield from _ClassesUsed._in_relation(view.relation, graph, field_types)

    @staticmethod
    def _in_relation(
        relation: ir.Relation, graph: EntityGraph, field_types: FieldTypes
    ) -> Iterator[tuple[type, FieldName]]:
        for field in relation.fields:
            match field:
                case ir.NestedField(relation=nested):
                    yield from _ClassesUsed._in_relation(nested, graph, field_types)
                case ir.Field():
                    match _ReadThrough.of(field, graph, field_types):
                        case (
                            _ThroughEnum(enum_class=through)
                            | _ThroughRootModel(root_model=through)
                        ):
                            yield through, field.name
                        case _ReadAsStored():
                            pass


class _ReadAsStored(pydantic.BaseModel, frozen=True):
    """A field read as Django returns it."""


class _ThroughEnum(pydantic.BaseModel, frozen=True):
    """A field read from one of the subject's own fields, typed by a `Literal` of
    members of `enum_class`, e.g. one a Filter narrows to `Literal[Role.LEAD]`.

    pydantic does not take a member's value for a `Literal` of members of a plain
    `enum.Enum`, even non-strictly, so the value Django returns goes through
    `enum_class` first.
    """

    enum_class: type[enum.Enum]


class _ThroughRootModel(pydantic.BaseModel, frozen=True):
    """A field lifted out of an Entity that declares it as `root_model`, required,
    whose column may still be NULL because a type it wraps admits `None`, e.g.
    `Rank(RootModel[int | None])`.

    Lifted through an optional Lookup, the field is `Rank | None`, which pydantic
    validates a NULL as `None`, not as the `Rank(None)` the Entity holds, so the
    value Django returns is validated as `root_model` first, non-strictly like
    the row.
    """

    root_model: type[pydantic.BaseModel]


class _ReadThrough:
    """How the value Django returns for a field is read."""

    @staticmethod
    def of(
        field: ir.Field, graph: EntityGraph, field_types: FieldTypes
    ) -> _ThroughEnum | _ThroughRootModel | _ReadAsStored:
        match field.value:
            case ir.SourceField() | ir.RequiredField():
                return _ReadThrough._literal_enum(field.annotation())
            case ir.Lifted() as lifted:
                return _ReadThrough._wrapping_none(lifted, graph, field_types)
            case ir.Lookup() | ir.ReverseLookup() | ir.ChainedLookup() | ir.Added():
                return _ReadAsStored()

    @staticmethod
    def _literal_enum(
        annotation: ir.RuntimeAnnotation,
    ) -> _ThroughEnum | _ReadAsStored:
        split = narrowing.split(annotation)
        members = typing.get_args(split.required)
        classes = {type(member) for member in members}
        match split.optionality, typing.get_origin(split.required), tuple(classes):
            case Optionality.REQUIRED, typing.Literal, (enum_class,) if issubclass(
                enum_class, enum.Enum
            ):
                return _ThroughEnum(enum_class=enum_class)
            case _:
                return _ReadAsStored()

    @staticmethod
    def _wrapping_none(
        lifted: ir.Lifted, graph: EntityGraph, field_types: FieldTypes
    ) -> _ThroughRootModel | _ReadAsStored:
        match lifted.declared():
            case FieldInfo(annotation=annotation):
                split = narrowing.split(annotation)
            case ir.Undeclared():
                return _ReadAsStored()
        column = DjangoModelsRenderer.column_optionality(
            graph.entities[lifted.through[-1].target], lifted.field.field, field_types
        )
        match split.optionality, column, split.required:
            case Optionality.REQUIRED, Optionality.OPTIONAL, type() as root_model if (
                issubclass(root_model, pydantic.RootModel)
            ):
                return _ThroughRootModel(root_model=root_model)
            case _:
                return _ReadAsStored()


class _KeepsEvery(pydantic.BaseModel, frozen=True):
    """A View without Filters: it keeps every row."""


class _RowReader(pydantic.BaseModel, frozen=True):
    """Writes the dict a View model is validated from, read from a row and the
    rows it selected related, at one indentation.

    The output is ruff-format stable: dicts keep a trailing comma, so they stay
    exploded, and a conditional is split over lines only when it does not fit.
    """

    graph: EntityGraph
    config: DjangoModels
    classes: Mapping[type, Symbol]
    indent: Indent

    def fields(self, relation: ir.Relation, reached: _Reached) -> Expression:
        """A dict with an entry per field of `relation` that is read from the row
        `reached`."""
        inner = self.at(self.indent.deeper())
        return self._dict(
            (field.name, value)
            for field in relation.fields
            if isinstance(value := inner.value(field, reached), Expression)
        )

    def at(self, indent: Indent) -> _RowReader:
        return self.model_copy(update={"indent": indent})

    def entity(self, entity: type[pydantic.BaseModel], row: Expression) -> Expression:
        """A dict with every field of `entity`, read from its Django model's `row`."""
        node = self.graph.entities[entity]
        return self._dict(
            (
                FieldName(name),
                Expression(
                    f"{row.root}."
                    f"{DjangoModelsRenderer.attribute_of(node, FieldName(name)).root}"
                ),
            )
            for name in entity.model_fields
        )

    def _dict(self, entries: Iterable[tuple[FieldName, Expression]]) -> Expression:
        return self.indent.exploded(
            Expression("{"),
            (Expression(f'"{key.root}": {value.root}') for key, value in entries),
            Expression("}"),
        )

    def value(self, field: ir.RelationField, reached: _Reached) -> Expression | LeftOut:
        """The entry for `field`, on a line at this indentation. What a Lookup
        finds is read off the row the QuerySet selected related."""
        prefix = Expression(f'"{field.name.root}": ')
        match field:
            case ir.NestedField(value=found, relation=nested):
                followed = self._followed(found.lookups(), reached)
                return self.indent.looked_up(
                    prefix,
                    followed.absent,
                    lambda indent: self.at(indent).fields(nested, followed.reached),
                )
            case ir.Field(value=value):
                field_types = FieldTypes(overrides=self.config.field_types)
                read_as = _ReadThrough.of(field, self.graph, field_types)
        row = reached.row()
        match value:
            case ir.SourceField() | ir.RequiredField():
                # A required field that is `None` fails the View model's validation.
                node = self.graph.entities[reached.trail.subject]
                attribute = DjangoModelsRenderer.attribute_of(node, value.field)
                return self._through(
                    read_as, Expression(f"{row.root}.{attribute.root}")
                )
            case ir.Lookup() | ir.ReverseLookup() | ir.ChainedLookup():
                followed = self._followed(value.lookups(), reached)
                return self.indent.looked_up(
                    prefix,
                    followed.absent,
                    lambda indent: self.at(indent).entity(
                        value.lookups()[-1].target, followed.reached.row()
                    ),
                )
            case ir.Lifted(through=through, field=read_field):
                followed = self._followed(through, reached)
                node = self.graph.entities[through[-1].target]
                attribute = DjangoModelsRenderer.attribute_of(node, read_field.field)
                lifted = self._through(
                    read_as,
                    Expression(f"{followed.reached.row().root}.{attribute.root}"),
                )
                return self.indent.looked_up(prefix, followed.absent, lambda _: lifted)
            case ir.Added():
                return LeftOut()

    def _through(
        self,
        through: _ThroughEnum | _ThroughRootModel | _ReadAsStored,
        read: Expression,
    ) -> Expression:
        """`read`, a value Django returns, read as `through` says."""
        match through:
            case _ThroughEnum(enum_class=enum_class):
                return Expression(
                    f"{self.classes[enum_class].qualified().root}({read.root})"
                )
            case _ThroughRootModel(root_model=root_model):
                validated = (
                    f"{self.classes[root_model].qualified().root}.model_validate"
                )
                return Expression(f"{validated}({read.root}, strict=False)")
            case _ReadAsStored():
                return read

    def _followed(
        self, lookups: Iterable[ir.EntityLookup], reached: _Reached
    ) -> _Followed:
        """`lookups` followed from the row `reached` through the rows the
        QuerySet selected related, each optional one absent when its row is
        `None`, or for a ReverseLookup when the reverse accessor found no row. A
        Hop back resumes at the row found before.

        A reverse accessor is read with `getattr`, cast to the Django model it
        holds, because the Django models do not declare it.

        One the View requires goes through `_required`, which raises when it
        finds nothing, like the mapping functions do: a field read through it
        may be optional, so the View model's validation would not fail.
        """
        conditions: list[Expression] = []
        for lookup in lookups:
            hop, trail = reached.trail.hop(lookup)
            match hop:
                case _Back():
                    # It finds the row it came from, which is there.
                    rows = reached.rows[:-1]
                case _Follow():
                    rows = (*reached.rows, self._found(lookup, reached.row()))
                    if not ModelPlan.required_lookup(lookup) and (
                        lookup.optionality is Optionality.OPTIONAL
                    ):
                        conditions.append(self._absent(lookup, reached.row()))
            reached = _Reached(trail=trail, rows=rows)
        return _Followed(absent=tuple(conditions), reached=reached)

    @staticmethod
    def every_lookup(relation: ir.Relation) -> Iterator[ir.EntityLookup]:
        """Every lookup of every field of `relation`, nested ones included, as
        often as it is used."""
        for field in relation.fields:
            yield from field.value.lookups()
            match field:
                case ir.NestedField(relation=nested):
                    yield from _RowReader.every_lookup(nested)
                case ir.Field():
                    pass

    def _found(self, lookup: ir.EntityLookup, row: Expression) -> Expression:
        """The row `lookup` finds from `row`: through `_required` if the View
        requires it."""
        related = DjangoQueriesRenderer.related(lookup, self.config).root
        name = PythonString(related).literal()
        cast = DjangoQueriesRenderer.typing_module.root
        model = self.config.model_of(lookup.target).qualified().root
        required = DjangoQueriesRenderer.required_helper.root
        match lookup:
            case ir.Lookup(reference=ir.RequiredField(field=field)):
                what = PythonString(field.root).literal().root
                return Expression(
                    f"{required}({row.root}.{related}, {row.root}, {what})"
                )
            case ir.Lookup():
                return Expression(f"{row.root}.{related}")
            case ir.ReverseLookup(optionality=Optionality.REQUIRED):
                what = PythonString(
                    f"a {lookup.target.__name__} by {lookup.via.root}"
                ).literal()
                return Expression(
                    f'{required}({cast}.cast("{model} | None", getattr({row.root}, '
                    f"{name.root}, None)), {row.root}, {what.root})"
                )
            case ir.ReverseLookup():
                return Expression(
                    f'{cast}.cast("{model}", getattr({row.root}, {name.root}))'
                )

    def _absent(self, lookup: ir.EntityLookup, row: Expression) -> Expression:
        """The condition under which an optional `lookup` finds no row from
        `row`."""
        name = DjangoQueriesRenderer.related(lookup, self.config).root
        match lookup:
            case ir.Lookup():
                return Expression(f"{row.root}.{name} is None")
            case ir.ReverseLookup():
                return Expression(
                    f"not hasattr({row.root}, {PythonString(name).literal().root})"
                )

    def unread_requirements(
        self, relation: ir.Relation, reached: _Reached, absent: tuple[Expression, ...]
    ) -> tuple[Expression, ...]:
        """A statement in the row function per Entity the View requires that the
        model built from `reached`, or one nested in it, reads nothing from: it
        looks the Entity up, so that a missing one raises. `absent` are the
        conditions under which a lookup on the way finds nothing."""
        statements: list[Expression] = []
        for field in relation.fields:
            match field:
                case ir.NestedField(value=found, relation=nested):
                    pass
                case ir.Field():
                    continue
            followed = self._followed(found.lookups(), reached)
            conditions = (*absent, *followed.absent)
            reads = any(
                not isinstance(read, ir.Field) or not isinstance(read.value, ir.Added)
                for read in nested.fields
            )
            if not reads and any(map(ModelPlan.required_lookup, found.lookups())):
                statements.append(
                    Indent(4).looked_up_statement(conditions, followed.reached.row())
                )
            statements.extend(
                self.unread_requirements(nested, followed.reached, conditions)
            )
        return tuple(statements)


class _Reached(pydantic.BaseModel, frozen=True):
    """A row the row reader has reached: the way there, and the expression of
    each row on it, the last the row itself."""

    trail: _Trail
    rows: tuple[Expression, ...]

    @staticmethod
    def start(subject: type[pydantic.BaseModel], row: Expression) -> _Reached:
        return _Reached(trail=_Trail.start(subject), rows=(row,))

    def row(self) -> Expression:
        return self.rows[-1]


class _Followed(pydantic.BaseModel, frozen=True):
    """Lookups followed: the conditions under which each optional one finds
    nothing, and the row they reach otherwise."""

    absent: tuple[Expression, ...]
    reached: _Reached
