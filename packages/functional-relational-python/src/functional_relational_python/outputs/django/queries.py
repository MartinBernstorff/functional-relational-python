"""Renders Django query code for every View whose Source is an Entity, so a Django
project can fill View models from its database while keeping its own filtering,
ordering and pagination.

Per View it writes two functions. `<view>_queryset` makes a QuerySet of the
Source's Django model load every Entity the View looks up, in the same query.
`<view>_from_row` builds the View model from one row of that QuerySet. Per Entity
it writes a QuerySet with a method `as_<view>()` per View over the Entity, and a
Manager, which is the Entity's Django model's manager `objects`. The method yields
View models from the QuerySet as filtered, ordered and sliced so far, so a list
costs one query.

`as_<view>()` loads every field the View needs: it undoes `only()` and `defer()`,
which would otherwise cost a query per row or conflict with `select_related`. It
cannot follow `values()`, `values_list()` or a combinator such as `union()`, whose
rows are not model instances; Django raises when it is called after them.

The Manager overrides the Manager methods that return a QuerySet (`all`, `filter`,
`exclude`, `order_by`) to return the Entity's QuerySet, so a type checker sees
`as_<view>()` on `Model.objects.filter(...)`.

A row is validated non-strictly, so a View model with a strict base accepts what
Django returns, e.g. a `str` for an enum field.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two agree (the differential test in `test_queries.py`
checks this). The generated module imports the Django models only for type
checking, and refers to them only in quoted annotations, because the models
import their Managers from it.
"""

import typing
from collections.abc import Callable, Iterable, Iterator, Sequence

import pydantic
from safe_result import Result

from functional_relational_python.compiler import compiler
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
)
from functional_relational_python.outputs.django.models import (
    DjangoModels,
    DjangoModelsRenderer,
)
from functional_relational_python.outputs.python.mappings import (
    Expression,
    Indent,
    LeftOut,
)
from functional_relational_python.outputs.python.plan import (
    ModuleLayout,
    NameClash,
    Namespace,
    Scope,
)
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    LineLength,
    ModulePath,
    Symbol,
)
from functional_relational_python.pipeline import ir
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
    """A View's model would be generated into the module of Django queries, so one
    file would overwrite the other."""

    view: ViewName
    module: ModulePath

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.view.root}: its model would be generated into "
            f"{self.module.root}, the module of Django queries. Generate it into "
            "another module."
        )


class ViewModelIntoDjangoModelsModule(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View's model would be generated into the module of Django models, which
    imports the Django queries, which import the View's model."""

    view: ViewName
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
    | NameClash
)


class RelatedPath(pydantic.RootModel[str], frozen=True):
    """A chain of foreign keys `select_related` follows, e.g. `department__company`."""

    @staticmethod
    def of(foreign_keys: Sequence[FieldName]) -> RelatedPath:
        return RelatedPath("__".join(key.root for key in foreign_keys))


class ViewQueries(pydantic.BaseModel, frozen=True):
    """The query code of one View whose Source is an Entity: `model` is its View
    model, `source` the Django model of its Source."""

    view: compiler.CompiledView
    model: Symbol
    source: Symbol

    def queryset_function(self) -> Identifier:
        return Identifier(f"{self.view.names.function.root}_queryset")

    def from_row_function(self) -> Identifier:
        return Identifier(f"{self.view.names.function.root}_from_row")

    def method(self) -> Identifier:
        """The method of the Source's QuerySet and Manager yielding View models."""
        return Identifier(f"as_{self.view.names.function.root}")

    def related_paths(self) -> tuple[RelatedPath, ...]:
        """Every chain of foreign keys to the Entities the View looks up, in the
        order they are first looked up."""
        return tuple(
            dict.fromkeys(
                RelatedPath.of(path)
                for path in ViewQueries._paths(self.view.relation, ())
            )
        )

    @staticmethod
    def _paths(
        relation: ir.Relation, prefix: tuple[FieldName, ...]
    ) -> Iterator[tuple[FieldName, ...]]:
        """`prefix` is the chain of foreign keys to the subject of `relation`."""
        for field in relation.fields:
            match field:
                case ir.Field(value=ir.Lookup() | ir.Lifted() as value):
                    yield from ViewQueries._value_paths(value, prefix)
                case ir.Field():
                    pass
                case ir.NestedField(value=lookup, relation=nested):
                    path = (*prefix, DjangoQueriesRenderer.foreign_key(lookup))
                    yield path
                    yield from ViewQueries._paths(nested, path)

    @staticmethod
    def _value_paths(
        value: ir.Lookup | ir.Lifted, prefix: tuple[FieldName, ...]
    ) -> Iterator[tuple[FieldName, ...]]:
        """The chains of foreign keys to each Entity `value` looks up on its way."""
        match value:
            case ir.Lookup():
                yield (*prefix, DjangoQueriesRenderer.foreign_key(value))
            case ir.Lifted(lookup=lookup, value=lifted):
                path = (*prefix, DjangoQueriesRenderer.foreign_key(lookup))
                yield path
                match lifted:
                    case ir.SourceField():
                        pass
                    case ir.Lookup() | ir.Lifted():
                        yield from ViewQueries._value_paths(lifted, path)


class DjangoQueriesRenderer(pydantic.BaseModel, frozen=True):
    """Writes the query code of every View whose Source is an Entity, and a
    QuerySet and Manager per Entity, into one module.

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
    row_parameter: typing.ClassVar[Identifier] = Identifier("row")
    as_manager: typing.ClassVar[Identifier] = Identifier("as_manager")
    map_builtin: typing.ClassVar[Identifier] = Identifier("map")

    header: typing.ClassVar[FileText] = FileText(
        f"{GeneratedFiles.header.root}"
        "# Regenerate with `frp generate`; `frp check` fails when this is stale.\n"
        "# Fills each View over an Entity from the Entity's Django model.\n"
    )

    graph: EntityGraph
    config: DjangoModels
    queries: tuple[ViewQueries, ...]

    @staticmethod
    def foreign_key(lookup: ir.Lookup) -> FieldName:
        """The foreign key of the Reference `lookup` follows."""
        return DjangoModelsRenderer.foreign_key_name(lookup.reference.field)

    @staticmethod
    def render(
        graph: EntityGraph,
        config: DjangoModels,
        views: Iterable[compiler.CompiledView],
        layout: ModuleLayout,
    ) -> Result[GeneratedFiles, DiagnosticsError[DjangoQueriesDiagnostic]]:
        """The module of query code, or every diagnostic.

        `layout` gives the module of each View's model. Views over a Source that
        is not an Entity get no query code. QuerySets are in Entity name order.
        """
        views = tuple(views)
        renderer = DjangoQueriesRenderer(
            graph=graph,
            config=config,
            queries=tuple(
                ViewQueries(
                    view=view,
                    model=Symbol(
                        module=layout(view),
                        qualname=(view.names.model.name.identifier(),),
                    ),
                    source=config.model_of(view.source),
                )
                for view in views
                if view.source in graph.entities
            ),
        )
        entities = sorted(graph.entities, key=lambda entity: entity.__name__)
        return DiagnosticsError.or_ok(
            (
                *DjangoQueriesRenderer._modules(config, views, layout),
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
        views: Iterable[compiler.CompiledView],
        layout: ModuleLayout,
    ) -> Iterator[DjangoQueriesDiagnostic]:
        """The Django models, the Django queries and the View models each need
        their own module."""
        if config.queries == config.module:
            yield QueriesIntoDjangoModelsModule(module=config.module)
        for view in views:
            module = layout(view)
            if module == config.queries:
                yield ViewModelIntoQueriesModule(view=view.name, module=module)
            if module == config.module:
                yield ViewModelIntoDjangoModelsModule(view=view.name, module=module)

    def _imports(self) -> tuple[ModulePath, ...]:
        """The modules imported when the module runs."""
        return (
            DjangoQueriesRenderer.typing_module,
            DjangoQueriesRenderer.django_models,
            DjangoQueriesRenderer.expressions,
            *((DjangoQueriesRenderer.iterator_module,) if self.queries else ()),
            *(query.model.module for query in self.queries),
        )

    def _clashes(
        self, entities: Sequence[type[pydantic.BaseModel]]
    ) -> Iterator[NameClash]:
        """Names bound twice. Entities named alike are left to the Django models
        to report, so their QuerySets and Managers are counted once."""
        module = Namespace(Scope(self.config.queries.root))
        for query in self.queries:
            view = query.view.name.root
            for function, role in (
                (query.queryset_function(), "QuerySet function"),
                (query.from_row_function(), "row function"),
            ):
                module.define(
                    function,
                    AnnotationText(
                        f"{self.config.queries.root}.{function.root} "
                        f"(the {role} of View {view})"
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
        for imported in (*self._imports(), self.config.module):
            module.refer_to_module(imported)
        module.refer(
            DjangoQueriesRenderer.map_builtin, AnnotationText("the built-in map")
        )
        yield from module.clashes()
        for query in self.queries:
            row = Namespace(
                Scope(f"{self.config.queries.root}.{query.from_row_function().root}")
            )
            row.define(
                DjangoQueriesRenderer.row_parameter,
                AnnotationText(f"the parameter for {query.source.describe().root}"),
            )
            row.refer_to_module(query.model.module)
            yield from row.clashes()
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
                methods.define(
                    query.method(),
                    AnnotationText(
                        f"{query_set.describe().root}.{query.method().root} (the "
                        f"method of View {query.view.name.root})"
                    ),
                )
            yield from methods.clashes()

    def _over(self, entity: type[pydantic.BaseModel]) -> Iterator[ViewQueries]:
        """The Views whose Source is `entity`, in View order."""
        return (query for query in self.queries if query.view.source is entity)

    def _module(self, entities: Sequence[type[pydantic.BaseModel]]) -> FileText:
        imports = "\n".join(
            "".join(f"import {module.root}\n" for module in section)
            for section in ModulePath.import_sections(self._imports())
        )
        type_checking = (
            f"\nif {DjangoQueriesRenderer.typing_module.root}.TYPE_CHECKING:\n"
            f"    import {self.config.module.root}\n"
        )
        functions = "".join(
            f"\n\n{DjangoQueriesRenderer._queryset_function(query).root}"
            f"\n\n{self._from_row_function(query).root}"
            for query in self.queries
        )
        classes = "".join(
            f"\n\n{self._query_set(entity).root}\n\n{self._manager(entity).root}"
            for entity in entities
        )
        return FileText(
            f"{DjangoQueriesRenderer.header.root}\n{imports}{type_checking}"
            f"{functions}{classes}"
        )

    @staticmethod
    def _queryset_function(query: ViewQueries) -> FileText:
        """Its annotations are quoted, because the Django models are imported only
        for type checking."""
        queryset = DjangoQueriesRenderer.queryset_parameter.root
        annotation = (
            f'"{DjangoQueriesRenderer.django_models.root}.QuerySet'
            f'[{query.source.qualified().root}]"'
        )
        prepared = f"{queryset}.defer(None)"
        paths = query.related_paths()
        if paths:
            body = Indent(4).exploded(
                Expression(f"{prepared}.select_related("),
                (Expression(f'"{path.root}"') for path in paths),
                Expression(")"),
            )
        else:
            body = Expression(prepared)
        return FileText(
            f"def {query.queryset_function().root}(\n"
            f"    {queryset}: {annotation},\n"
            f") -> {annotation}:\n"
            f"    return {body.root}\n"
        )

    def _from_row_function(self, query: ViewQueries) -> FileText:
        row = Expression(DjangoQueriesRenderer.row_parameter.root)
        model = query.model.qualified().root
        read = _RowReader(graph=self.graph, indent=Indent(8)).fields(
            query.view.relation, query.view.source, row
        )
        validated = Indent(4).exploded(
            Expression(f"{model}.model_validate("),
            (read, Expression("strict=False")),
            Expression(")"),
        )
        returns = f") -> {model}:"
        # Like ruff format: parenthesized onto its own line, if that makes it fit.
        if len(returns) > LineLength.default().root >= len(model) + 4:
            returns = f") -> (\n    {model}\n):"
        return FileText(
            f"def {query.from_row_function().root}(\n"
            f'    {row.root}: "{query.source.qualified().root}",\n'
            f"{returns}\n"
            f"    return {validated.root}\n"
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

    def _iterator_of(self, query: ViewQueries) -> Expression:
        """The method's return annotation line, `    ) -> Iterator[<model>]:`."""
        return DjangoQueriesRenderer._subscript(
            Indent(4),
            Expression(
                f") -> {DjangoQueriesRenderer.iterator_module.root}.Iterator"
                f"[{query.model.qualified().root}]"
            ),
            Expression(":"),
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
            f"    def {query.method().root}(\n"
            "        self,\n"
            f"{self._iterator_of(query).root}\n"
            f"        return {DjangoQueriesRenderer.map_builtin.root}(\n"
            f"            {query.from_row_function().root},\n"
            f"            {query.queryset_function().root}(self),\n"
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
                f"    def {query.method().root}(\n"
                "        self,\n"
                f"{self._iterator_of(query).root}\n"
                f"        return self.get_queryset().{query.method().root}()\n"
                for query in self._over(entity)
            ),
        ]
        return FileText(f"{statement.root}{'\n'.join(methods)}")


class _RowReader(pydantic.BaseModel, frozen=True):
    """Writes the dict a View model is validated from, read from a row and the
    rows it selected related, at one indentation.

    The output is ruff-format stable: dicts keep a trailing comma, so they stay
    exploded, and a conditional is split over lines only when it does not fit.
    """

    graph: EntityGraph
    indent: Indent

    def fields(
        self,
        relation: ir.Relation,
        subject: type[pydantic.BaseModel],
        row: Expression,
    ) -> Expression:
        """A dict with an entry per field of `relation` that is read from `row`,
        the Django model instance of the Entity `subject`."""
        inner = self.at(self.indent.deeper())
        return self._dict(
            (field.name, value)
            for field in relation.fields
            if isinstance(value := inner.value(field, subject, row), Expression)
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

    def value(
        self,
        field: ir.RelationField,
        subject: type[pydantic.BaseModel],
        row: Expression,
    ) -> Expression | LeftOut:
        """The entry for `field`, on a line at this indentation."""
        match field:
            case ir.Field(value=value):
                match value:
                    case ir.SourceField() | ir.Lookup() | ir.Lifted():
                        return self.computed(field, value, subject, row)
                    case ir.Added():
                        return LeftOut()
            case ir.NestedField(value=lookup, relation=nested):
                related = _RowReader._related(lookup, row)
                return self.indent.looked_up(
                    Expression(f'"{field.name.root}": '),
                    _RowReader._guard(lookup, related),
                    lambda indent: self.at(indent).fields(
                        nested, lookup.target, related
                    ),
                )

    def computed(
        self,
        field: ir.RelationField,
        value: ir.SubjectValue,
        subject: type[pydantic.BaseModel],
        row: Expression,
    ) -> Expression:
        """`value` read from `row`, the Django model instance of `subject`, as the
        entry for `field`."""
        guards, present = self._chain(value, subject, row)
        return self.indent.looked_up(
            Expression(f'"{field.name.root}": '), guards, present
        )

    def _chain(
        self, value: ir.SubjectValue, subject: type[pydantic.BaseModel], row: Expression
    ) -> tuple[tuple[Expression, ...], Callable[[Indent], Expression]]:
        """The rows of the optional References `value` follows from `row`, to
        check for `None` in order, and what it reads once none is. A lifted value
        is read off the row its lookup selected related."""
        match value:
            case ir.SourceField():
                node = self.graph.entities[subject]
                attribute = DjangoModelsRenderer.attribute_of(node, value.field)
                read = Expression(f"{row.root}.{attribute.root}")
                return (), lambda _indent: read
            case ir.Lookup():
                related = _RowReader._related(value, row)
                return _RowReader._guard(value, related), lambda indent: self.at(
                    indent
                ).entity(value.target, related)
            case ir.Lifted(lookup=lookup, value=lifted):
                related = _RowReader._related(lookup, row)
                guards, present = self._chain(lifted, lookup.target, related)
                return (*_RowReader._guard(lookup, related), *guards), present

    @staticmethod
    def _related(lookup: ir.Lookup, row: Expression) -> Expression:
        """The row `lookup` finds, which the QuerySet selected related."""
        foreign_key = DjangoQueriesRenderer.foreign_key(lookup)
        return Expression(f"{row.root}.{foreign_key.root}")

    @staticmethod
    def _guard(lookup: ir.Lookup, related: Expression) -> tuple[Expression, ...]:
        """The row `lookup` finds, to check for `None`, if its Reference is
        optional."""
        match lookup.optionality:
            case Optionality.REQUIRED:
                return ()
            case Optionality.OPTIONAL:
                return (related,)
