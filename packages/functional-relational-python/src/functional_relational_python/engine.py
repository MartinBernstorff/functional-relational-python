"""The InMemoryViewEngine: the reference semantics for compiled Views.

It runs compiled Views over Entities held in memory. Users can use it as a
stateful fake in their own tests, and every other backend is compared against it.
"""

from collections.abc import Collection, Mapping

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python import compiler, ir
from functional_relational_python.entity_graph import EntityGraph, FieldName
from functional_relational_python.views import ViewName


class ViewRow(pydantic.RootModel[Mapping[FieldName, object]], frozen=True):
    """One View result: the value of each field the View's Relation has.

    Values are the Entity's own field values, so their types are the user's.
    """

    @staticmethod
    def fake() -> ViewRow:
        return ViewRow({FieldName.fake(): ViewName.fake()})

    def renamed(self, old: FieldName, new: FieldName) -> ViewRow:
        return ViewRow(
            {new if name == old else name: value for name, value in self.root.items()}
        )

    def without(self, field: FieldName) -> ViewRow:
        return ViewRow(
            {name: value for name, value in self.root.items() if name != field}
        )

    def keeping(self, fields: Collection[FieldName]) -> ViewRow:
        return ViewRow(
            {name: value for name, value in self.root.items() if name in fields}
        )


class NotAnEntityError(Exception):
    """`put` got an instance of a model that is not an Entity in the EntityGraph."""

    def __init__(self, model: type[pydantic.BaseModel]) -> None:
        self.model = model
        super().__init__(f"{model.__name__} is not an Entity in the EntityGraph.")


class UnknownViewError(Exception):
    def __init__(self, view: ViewName, known: tuple[ViewName, ...]) -> None:
        self.view = view
        self.known = known
        names = ", ".join(f"'{name.root}'" for name in known)
        super().__init__(f"No compiled View named '{view.root}'. Known Views: {names}.")


class InMemoryViewEngine:
    """Holds Entities by ID and runs compiled Views over them.

    `put` inserts an Entity, or replaces the one with the same ID. `read` returns
    the View results, one per row of the View's Source, in insertion order.
    """

    def __init__(
        self, entity_graph: EntityGraph, views: compiler.CompiledViews
    ) -> None:
        self._entity_graph = entity_graph
        self._views = {view.name: view for view in views.views}
        self._entities: dict[type[pydantic.BaseModel], dict[object, ViewRow]] = {}

    def put(self, entity: pydantic.BaseModel) -> Result[None, NotAnEntityError]:
        model = type(entity)
        if model not in self._entity_graph.entities:
            return Err(NotAnEntityError(model))
        row = ViewRow(
            {FieldName(name): getattr(entity, name) for name in model.model_fields}
        )
        self._entities.setdefault(model, {})[row.root[FieldName("id")]] = row
        return Ok(None)

    def read(self, view: ViewName) -> Result[tuple[ViewRow, ...], UnknownViewError]:
        match self._views.get(view):
            case None:
                return Err(UnknownViewError(view, tuple(self._views)))
            case compiled:
                return Ok(self._run(compiled.node))

    def _run(self, node: ir.Node) -> tuple[ViewRow, ...]:
        match node:
            case ir.FromSource():
                return tuple(self._entities.get(node.source, {}).values())
            case ir.Omit():
                return tuple(row.without(node.field) for row in self._run(node.input))
            case ir.Rename():
                return tuple(
                    row.renamed(node.field, node.new) for row in self._run(node.input)
                )
            case ir.Project():
                return tuple(row.keeping(node.fields) for row in self._run(node.input))
