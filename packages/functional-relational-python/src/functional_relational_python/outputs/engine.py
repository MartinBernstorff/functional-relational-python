"""The InMemoryViewEngine: the reference semantics for compiled Views.

It runs compiled Views over Entities held in memory. Users can use it as a
stateful fake in their own tests, and every other backend is compared against it.
"""

from collections.abc import Mapping
from typing import Protocol

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


type FieldValue = object
"""A View result's field value: the Entity's own value, an Entity, or a ViewRow."""


class _Missing:
    """The missing IDs found while running a View, by Entity and field."""

    def __init__(self) -> None:
        self.ids: dict[tuple[type[pydantic.BaseModel], FieldName], list[IdValue]] = {}

    def add(
        self, entity: type[pydantic.BaseModel], field: FieldName, id_value: IdValue
    ) -> None:
        missing = self.ids.setdefault((entity, field), [])
        if id_value not in missing:
            missing.append(id_value)

    def dangling(self) -> tuple[DanglingReference, ...]:
        """One per Entity and field, each with every missing ID."""
        return tuple(
            DanglingReference(entity=entity, field=field, missing=tuple(ids))
            for (entity, field), ids in self.ids.items()
        )


class _Comparable(Protocol):
    """A field value a Filter compares. The compiler only lets through an
    ordering of values whose type defines it."""

    def __lt__(self, other: object, /) -> object: ...

    def __le__(self, other: object, /) -> object: ...

    def __gt__(self, other: object, /) -> object: ...

    def __ge__(self, other: object, /) -> object: ...


class InMemoryViewEngine:
    """Holds Entities by ID and runs compiled Views over them.

    `put` inserts an Entity, or replaces the one with the same ID. `read` returns
    the View results, one per row of the View's Source its Filters keep, in
    insertion order.
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
        UnknownViewError | NonEntitySourceError | DiagnosticsError[DanglingReference],
    ]:
        """The View's results, or every Reference that points to a missing Entity.

        Only Entities are stored, so a View over any other Source is an error. A
        row the View's Filters exclude is not looked up, so its References are not
        reported.
        """
        match self._views.get(view):
            case None:
                return Err(UnknownViewError(view, tuple(self._views)))
            case compiled:
                pass
        if compiled.source not in self._entity_graph.entities:
            return Err(NonEntitySourceError(view, compiled.source))
        missing = _Missing()
        rows = tuple(
            self._row(compiled.relation, entity, missing)
            for entity in self._entities.get(compiled.source, {}).values()
            if all(
                InMemoryViewEngine._holds(kept, entity)
                for kept in compiled.relation.filters
            )
        )
        return DiagnosticsError.or_ok(missing.dangling(), lambda: rows)

    def _row(
        self, relation: ir.Relation, subject: pydantic.BaseModel, missing: _Missing
    ) -> ViewRow:
        return ViewRow(
            {
                field.name: self._field(field, subject, missing)
                for field in relation.fields
            }
        )

    def _field(
        self, field: ir.RelationField, subject: pydantic.BaseModel, missing: _Missing
    ) -> FieldValue:
        match field:
            case ir.Field(value=value):
                match value:
                    case ir.SourceField():
                        return getattr(subject, value.field.root)
                    case ir.Lookup():
                        return self._lookup(value, subject, missing)
                    case ir.Added():
                        return InMemoryViewEngine._default(field)
            case ir.NestedField(value=lookup, relation=relation):
                match self._lookup(lookup, subject, missing):
                    case None:
                        return None
                    case entity:
                        return self._row(relation, entity, missing)

    @staticmethod
    def _holds(kept: ir.Filter, subject: pydantic.BaseModel) -> bool:
        """Whether `kept` is true of `subject`, with Python's two-valued null
        semantics: `None == x` is false and `None != x` is true."""
        match kept:
            case ir.FilterComparison(operator=operator, field=field, operand=operand):
                match operand:
                    case ir.SourceField():
                        right = getattr(subject, operand.field.root)
                    case ir.ScalarLiteral() | ir.EnumLiteral() | ir.RootModelLiteral():
                        right = operand.value()
                return InMemoryViewEngine._compare(operator, subject, field, right)
            case ir.FilterIsNull(field=field):
                return getattr(subject, field.field.root) is None
            case ir.FilterAnd(terms=terms):
                return all(InMemoryViewEngine._holds(term, subject) for term in terms)
            case ir.FilterOr(terms=terms):
                return any(InMemoryViewEngine._holds(term, subject) for term in terms)
            case ir.FilterNot(term=term):
                return not InMemoryViewEngine._holds(term, subject)

    @staticmethod
    def _compare(
        operator: ir.ComparisonOperator,
        subject: pydantic.BaseModel,
        field: ir.SourceField,
        right: FieldValue,
    ) -> bool:
        """`subject.<field> <operator> right` in Python. The compiler only lets
        through an ordering of non-optional fields whose type defines it."""
        left: _Comparable = getattr(subject, field.field.root)
        compared: object
        match operator:
            case ir.ComparisonOperator.EQ:
                compared = left == right
            case ir.ComparisonOperator.NE:
                compared = left != right
            case ir.ComparisonOperator.LT:
                compared = left < right
            case ir.ComparisonOperator.LE:
                compared = left <= right
            case ir.ComparisonOperator.GT:
                compared = left > right
            case ir.ComparisonOperator.GE:
                compared = left >= right
        return bool(compared)

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
        self, lookup: ir.Lookup, subject: pydantic.BaseModel, missing: _Missing
    ) -> pydantic.BaseModel | None:
        """The referenced Entity; `None` for an absent optional Reference, or when
        the Entity is missing, which `missing` records."""
        id_value = getattr(subject, lookup.reference.field.root)
        if id_value is None:
            return None
        match self._entities.get(lookup.target, {}).get(id_value):
            case None:
                missing.add(type(subject), lookup.reference.field, id_value)
                return None
            case entity:
                return entity
