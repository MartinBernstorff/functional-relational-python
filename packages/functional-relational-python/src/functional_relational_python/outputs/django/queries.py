"""Renders Django query code for every View whose Source is an Entity, so a Django
project can fill View models from its database while keeping its own filtering,
ordering and pagination.

Per View it writes two functions. `<view>_queryset` makes a QuerySet of the
Source's Django model load every Entity the View looks up, in the same query.
`<view>_from_row` builds the View model from one row of that QuerySet. Per Entity
it writes a QuerySet, which the Entity's Django model uses as its manager, with a
method `as_<view>()` per View over the Entity. The method yields View models from
the QuerySet as filtered, ordered and sliced so far, so a list costs one query.

Like the InMemoryViewEngine, it computes each field from the Values of the
compiled Relation, so the two agree (the differential test in `test_queries.py`
checks this). The generated module imports the Django models only for type
checking, because the models import their QuerySets from it.
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
from functional_relational_python.support.diagnostic import DiagnosticsError
from functional_relational_python.support.filesystem import FileText
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)
from functional_relational_python.support.identifier import Identifier


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
        """The method of the Source's QuerySet yielding View models."""
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
                case ir.Field(value=ir.Lookup() as lookup):
                    yield (*prefix, DjangoQueriesRenderer.foreign_key(lookup))
                case ir.Field():
                    pass
                case ir.NestedField(value=lookup, relation=nested):
                    path = (*prefix, DjangoQueriesRenderer.foreign_key(lookup))
                    yield path
                    yield from ViewQueries._paths(nested, path)


class DjangoQueriesRenderer:
    """Writes the query code of every View whose Source is an Entity, and a
    QuerySet per Entity, into one module."""

    header: typing.ClassVar[FileText] = FileText(
        f"{GeneratedFiles.header.root}"
        "# Regenerate with `frp generate`; `frp check` fails when this is stale.\n"
        "# Fills each View over an Entity from the Entity's Django model.\n"
    )
    typing_module: typing.ClassVar[ModulePath] = ModulePath("typing")
    iterator_module: typing.ClassVar[ModulePath] = ModulePath("collections.abc")
    django_models: typing.ClassVar[ModulePath] = ModulePath("django.db.models")
    queryset_parameter: typing.ClassVar[Identifier] = Identifier("queryset")
    row_parameter: typing.ClassVar[Identifier] = Identifier("row")
    as_manager: typing.ClassVar[Identifier] = Identifier("as_manager")
    map_builtin: typing.ClassVar[Identifier] = Identifier("map")

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
    ) -> Result[GeneratedFiles, DiagnosticsError[NameClash]]:
        """The module of query code, or every name it would bind twice.

        `layout` gives the module of each View's model. Views over a Source that
        is not an Entity are left out. QuerySets are in Entity name order.
        """
        queries = tuple(
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
        )
        entities = sorted(graph.entities, key=lambda entity: entity.__name__)
        renderer = DjangoQueriesRenderer(graph, config, queries)
        return DiagnosticsError.or_ok(
            renderer._clashes(entities),
            lambda: GeneratedFiles(
                files=(
                    GeneratedFile(
                        path=config.queries.file_path(),
                        text=renderer._module(entities),
                    ),
                )
            ),
        )

    def __init__(
        self,
        graph: EntityGraph,
        config: DjangoModels,
        queries: tuple[ViewQueries, ...],
    ) -> None:
        self.graph = graph
        self.config = config
        self.queries = queries

    def _imports(self) -> tuple[ModulePath, ...]:
        """The modules imported when the module runs."""
        return (
            DjangoQueriesRenderer.typing_module,
            DjangoQueriesRenderer.django_models,
            *((DjangoQueriesRenderer.iterator_module,) if self.queries else ()),
            *(query.model.module for query in self.queries),
        )

    def _clashes(
        self, entities: Sequence[type[pydantic.BaseModel]]
    ) -> Iterator[NameClash]:
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
        for entity in entities:
            query_set = self.config.query_set_of(entity)
            module.define(
                query_set.bound_name(),
                AnnotationText(
                    f"{query_set.describe().root} (the QuerySet of Entity "
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
        for entity in entities:
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
        classes = "".join(f"\n\n{self._query_set(entity).root}" for entity in entities)
        return FileText(
            f"{DjangoQueriesRenderer.header.root}\n{imports}{type_checking}"
            f"{functions}{classes}"
        )

    @staticmethod
    def _queryset_function(query: ViewQueries) -> FileText:
        queryset = DjangoQueriesRenderer.queryset_parameter.root
        annotation = (
            f"{DjangoQueriesRenderer.django_models.root}.QuerySet"
            f"[{query.source.qualified().root}]"
        )
        paths = query.related_paths()
        if paths:
            listed = "".join(f'        "{path.root}",\n' for path in paths)
            body = f"{queryset}.select_related(\n{listed}    )"
        else:
            body = queryset
        return FileText(
            f"def {query.queryset_function().root}(\n"
            f"    {queryset}: {annotation},\n"
            f") -> {annotation}:\n"
            f"    return {body}\n"
        )

    def _from_row_function(self, query: ViewQueries) -> FileText:
        row = Expression(DjangoQueriesRenderer.row_parameter.root)
        model = query.model.qualified().root
        read = _RowReader(self.graph, Indent(8)).fields(
            query.view.relation, query.view.source, row
        )
        if "\n" in read.root:
            validated = f"(\n        {read.root}\n    )"
        else:
            validated = f"({read.root})"
        return FileText(
            f"def {query.from_row_function().root}(\n"
            f"    {row.root}: {query.source.qualified().root},\n"
            f") -> {model}:\n"
            f"    return {model}.model_validate{validated}\n"
        )

    def _query_set(self, entity: type[pydantic.BaseModel]) -> FileText:
        model = self.config.model_of(entity).qualified().root
        base = f'{DjangoQueriesRenderer.django_models.root}.QuerySet["{model}"]'
        name = self.config.query_set_of(entity).bound_name().root
        statement = f"class {name}({base}):\n"
        if len(statement) - 1 > LineLength.default().root:
            statement = f"class {name}(\n    {base},\n):\n"
        methods = "\n".join(
            f"    def {query.method().root}(\n"
            "        self,\n"
            f"    ) -> {DjangoQueriesRenderer.iterator_module.root}.Iterator"
            f"[{query.model.qualified().root}]:\n"
            f"        return {DjangoQueriesRenderer.map_builtin.root}(\n"
            f"            {query.from_row_function().root},\n"
            f"            {query.queryset_function().root}(self),\n"
            "        )\n"
            for query in self._over(entity)
        )
        return FileText(
            f"{statement}{methods}" if methods else f"{statement}    pass\n"
        )


class _RowReader:
    """Writes the dict a View model is validated from, read from a row and the
    rows it selected related, at one indentation.

    The output is ruff-format stable: dicts keep a trailing comma, so they stay
    exploded, and a conditional is split over lines only when it does not fit.
    """

    def __init__(self, graph: EntityGraph, indent: Indent) -> None:
        self.graph = graph
        self.indent = indent

    def fields(
        self,
        relation: ir.Relation,
        subject: type[pydantic.BaseModel],
        row: Expression,
    ) -> Expression:
        """A dict with an entry per field of `relation` that is read from `row`,
        the Django model instance of the Entity `subject`."""
        inner = _RowReader(self.graph, self.indent.deeper())
        return self._dict(
            (field.name, value)
            for field in relation.fields
            if isinstance(value := inner.value(field, subject, row), Expression)
        )

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
        inner = self.indent.deeper().pad().root
        lines = "".join(
            f'{inner}"{key.root}": {value.root},\n' for key, value in entries
        )
        if not lines:
            return Expression("{}")
        return Expression(f"{{\n{lines}{self.indent.pad().root}}}")

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
                    case ir.SourceField():
                        node = self.graph.entities[subject]
                        attribute = DjangoModelsRenderer.attribute_of(node, value.field)
                        return Expression(f"{row.root}.{attribute.root}")
                    case ir.Lookup():
                        related = _RowReader._related(value, row)
                        return self._optional(
                            field,
                            related,
                            lambda reader: reader.entity(value.target, related),
                        )
                    case ir.Added():
                        return LeftOut()
            case ir.NestedField(value=lookup, relation=nested):
                related = _RowReader._related(lookup, row)
                return self._optional(
                    field,
                    related,
                    lambda reader: reader.fields(nested, lookup.target, related),
                )

    @staticmethod
    def _related(lookup: ir.Lookup, row: Expression) -> Expression:
        """The row `lookup` finds, which the QuerySet selected related."""
        foreign_key = DjangoQueriesRenderer.foreign_key(lookup)
        return Expression(f"{row.root}.{foreign_key.root}")

    def _optional(
        self,
        field: ir.RelationField,
        related: Expression,
        present: Callable[[_RowReader], Expression],
    ) -> Expression:
        """`present` itself, or `None` when an optional Reference is absent.

        `present` gets the reader to write its expression at.
        """
        match field.optionality():
            case Optionality.REQUIRED:
                return present(self)
            case Optionality.OPTIONAL:
                pass
        inline = f"None if {related.root} is None else {present(self).root}"
        line = f'{self.indent.pad().root}"{field.name.root}": {inline},'
        if "\n" not in inline and len(line) <= LineLength.default().root:
            return Expression(inline)
        inner = _RowReader(self.graph, self.indent.deeper())
        pad = inner.indent.pad().root
        return Expression(
            f"(\n{pad}None\n{pad}if {related.root} is None\n"
            f"{pad}else {present(inner).root}\n{self.indent.pad().root})"
        )
