"""The InMemoryViewEngine: the reference semantics for compiled Views.

It runs compiled Views over Entities held in memory. Users can use it as a
stateful fake in their own tests, and every other backend is compared against it.
"""

from collections.abc import Mapping

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    EntityNode,
    FieldName,
)
from functional_relational_python.entity_graph.id import IdValue
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.views import ViewName
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)


class ViewRow(pydantic.RootModel[Mapping[FieldName, object]], frozen=True):
    """One View result: the value of each field the View's Relation has.

    Values are the Entity's own field values, so their types are the user's. A
    resolved field holds the referenced Entity, or a ViewRow if it has a nested
    pipeline.
    """

    @staticmethod
    def fake() -> ViewRow:
        return ViewRow({FieldName.fake(): ViewName.fake()})


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


class NonEntitySourceError(Exception):
    """A View's Source is not an Entity, so the engine holds no rows to read."""

    def __init__(self, view: ViewName, source: type[pydantic.BaseModel]) -> None:
        self.view = view
        self.source = source
        super().__init__(
            f"View '{view.root}' reads {source.__name__}, which is not an Entity in "
            "the EntityGraph, so the engine holds no rows for it."
        )


class UndefaultedFieldError(Exception):
    """An added field has no default to fill it with. The compiler rejects such a
    View, so this means the compiler let one through."""

    def __init__(self, field: FieldName) -> None:
        self.field = field
        super().__init__(
            f"Added field '{field.root}' has no default; the compiler should have "
            "rejected its View."
        )


class DanglingReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """A Reference field whose IDs key no stored Entity."""

    entity: type[pydantic.BaseModel]
    field: FieldName
    missing: tuple[IdValue, ...]

    def describe(self) -> DiagnosticMessage:
        missing = ", ".join(repr(id_value.root) for id_value in self.missing)
        return DiagnosticMessage(
            f"{self.entity.__name__}.{self.field.root} references missing IDs: "
            f"{missing}"
        )


class SharedOneToOneReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """A one-to-one Reference field that several stored Entities share IDs in, so
    following it backwards finds more than one Entity."""

    entity: type[pydantic.BaseModel]
    field: FieldName
    shared: tuple[IdValue, ...]

    def describe(self) -> DiagnosticMessage:
        shared = ", ".join(repr(id_value.root) for id_value in self.shared)
        return DiagnosticMessage(
            f"{self.entity.__name__}.{self.field.root} is one-to-one, but several "
            f"{self.entity.__name__} entities reference each of: {shared}"
        )


type ReadDiagnostic = DanglingReference | SharedOneToOneReference


type FieldValue = object
"""A View result's field value: the Entity's own value, an Entity, or a ViewRow."""


class _IdsFound:
    """IDs found while running a View, by Entity and Reference field, each once."""

    def __init__(self) -> None:
        self.ids: dict[tuple[type[pydantic.BaseModel], FieldName], list[IdValue]] = {}

    def add(
        self, entity: type[pydantic.BaseModel], field: FieldName, id_value: IdValue
    ) -> None:
        found = self.ids.setdefault((entity, field), [])
        if id_value not in found:
            found.append(id_value)


class _Problems:
    """The IDs running a View found a problem with.

    `missing` are IDs no stored Entity has, by the Entity and field holding them.
    `shared` are IDs several Entities hold in a one-to-one field, by that Entity
    and field.
    """

    def __init__(self) -> None:
        self.missing = _IdsFound()
        self.shared = _IdsFound()

    def diagnostics(self) -> tuple[ReadDiagnostic, ...]:
        """One per Entity and field, each with every ID found."""
        return (
            *(
                DanglingReference(entity=entity, field=field, missing=tuple(ids))
                for (entity, field), ids in self.missing.ids.items()
            ),
            *(
                SharedOneToOneReference(entity=entity, field=field, shared=tuple(ids))
                for (entity, field), ids in self.shared.ids.items()
            ),
        )


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
        self._entities: dict[
            type[pydantic.BaseModel], dict[IdValue, pydantic.BaseModel]
        ] = {}

    def put(self, entity: pydantic.BaseModel) -> Result[None, NotAnEntityError]:
        model = type(entity)
        if model not in self._entity_graph.entities:
            return Err(NotAnEntityError(model))
        primary_key = getattr(entity, EntityNode.primary_key.root)
        self._entities.setdefault(model, {})[primary_key] = entity
        return Ok(None)

    def read(
        self, view: ViewName
    ) -> Result[
        tuple[ViewRow, ...],
        UnknownViewError | NonEntitySourceError | DiagnosticsError[ReadDiagnostic],
    ]:
        """The View's results, or every Reference that points to a missing Entity
        and every ID several Entities share in a one-to-one Reference the View
        follows backwards.

        Only Entities are stored, so a View over any other Source is an error.
        """
        match self._views.get(view):
            case None:
                return Err(UnknownViewError(view, tuple(self._views)))
            case compiled:
                pass
        if compiled.source not in self._entity_graph.entities:
            return Err(NonEntitySourceError(view, compiled.source))
        problems = _Problems()
        rows = tuple(
            self._row(compiled.relation, entity, problems)
            for entity in self._entities.get(compiled.source, {}).values()
        )
        return DiagnosticsError.or_ok(problems.diagnostics(), lambda: rows)

    def _row(
        self, relation: ir.Relation, subject: pydantic.BaseModel, problems: _Problems
    ) -> ViewRow:
        return ViewRow(
            {
                field.name: self._field(field, subject, problems)
                for field in relation.fields
            }
        )

    def _field(
        self, field: ir.RelationField, subject: pydantic.BaseModel, problems: _Problems
    ) -> FieldValue:
        match field:
            case ir.Field(value=value):
                match value:
                    case ir.SourceField():
                        return getattr(subject, value.field.root)
                    case ir.Lookup() | ir.ReverseLookup():
                        return self._lookup(value, subject, problems)
                    case ir.Added():
                        return InMemoryViewEngine._default(field)
            case ir.NestedField(value=lookup, relation=relation):
                match self._lookup(lookup, subject, problems):
                    case None:
                        return None
                    case entity:
                        return self._row(relation, entity, problems)

    @staticmethod
    def _default(field: ir.Field) -> FieldValue:
        """The default an added field's model fills it with."""
        match field.default:
            case ir.ScalarDefault(root=value):
                return value
            case ir.MissingDefault():
                return pydantic.MISSING
            case ir.EmptyListDefault():
                return list[FieldValue]()
            case ir.EmptyDictDefault():
                return dict[FieldValue, FieldValue]()
            case ir.EnumDefault(member=member):
                return member
            case ir.NoDefault():
                raise UndefaultedFieldError(field.name)

    def _lookup(
        self,
        lookup: ir.EntityLookup,
        subject: pydantic.BaseModel,
        problems: _Problems,
    ) -> pydantic.BaseModel | None:
        """The Entity `lookup` finds from `subject`, or `None`.

        A Lookup gives `None` for an absent optional Reference, or when the Entity
        is missing, which `problems` records. A ReverseLookup gives `None` when no
        Entity references the subject, and the first in insertion order when
        several do, which `problems` records.
        """
        match lookup:
            case ir.Lookup():
                reference = lookup.reference.field
                id_value = getattr(subject, reference.root)
                if id_value is None:
                    return None
                match self._entities.get(lookup.target, {}).get(id_value):
                    case None:
                        problems.missing.add(type(subject), reference, id_value)
                        return None
                    case entity:
                        return entity
            case ir.ReverseLookup():
                id_value = getattr(subject, EntityNode.primary_key.root)
                found = [
                    entity
                    for entity in self._entities.get(lookup.target, {}).values()
                    if getattr(entity, lookup.via.root) == id_value
                ]
                if not found:
                    return None
                if len(found) > 1:
                    problems.shared.add(lookup.target, lookup.via, id_value)
                return found[0]
