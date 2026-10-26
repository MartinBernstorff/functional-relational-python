"""The InMemoryViewEngine: the reference semantics for compiled Views.

It runs compiled Views over Entities held in memory. Users can use it as a
stateful fake in their own tests, and every other backend is compared against it.
"""

from collections.abc import Iterable, Mapping
from typing import Protocol

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    EntityNode,
    FieldName,
    Optionality,
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


class UnmetRequirement(pydantic.BaseModel, Diagnostic, frozen=True):
    """A field a View requires is `None` in the stored Entities with `ids`."""

    entity: type[pydantic.BaseModel]
    field: FieldName
    ids: tuple[IdValue, ...]

    def describe(self) -> DiagnosticMessage:
        ids = ", ".join(repr(id_value.root) for id_value in self.ids)
        return DiagnosticMessage(
            f"{self.entity.__name__}.{self.field.root} is required, but is None "
            f"for IDs: {ids}"
        )


class UnmetReverseRequirement(pydantic.BaseModel, Diagnostic, frozen=True):
    """A View requires an `entity` whose one-to-one Reference `via` points to the
    subject, but none points to the subjects with `ids`."""

    entity: type[pydantic.BaseModel]
    via: FieldName
    ids: tuple[IdValue, ...]

    def describe(self) -> DiagnosticMessage:
        ids = ", ".join(repr(id_value.root) for id_value in self.ids)
        return DiagnosticMessage(
            f"A {self.entity.__name__} referencing each subject by "
            f"{self.entity.__name__}.{self.via.root} is required, but none "
            f"references IDs: {ids}"
        )


type ReadDiagnostic = (
    DanglingReference
    | SharedOneToOneReference
    | UnmetRequirement
    | UnmetReverseRequirement
)


type FieldValue = object
"""A View result's field value: the Entity's own value, an Entity, or a ViewRow."""


class _IdsFound:
    """IDs found while running a View, by Entity and field, each once."""

    def __init__(self) -> None:
        self.ids: dict[tuple[type[pydantic.BaseModel], FieldName], list[IdValue]] = {}

    def add(
        self, entity: type[pydantic.BaseModel], field: FieldName, id_value: IdValue
    ) -> None:
        found = self.ids.setdefault((entity, field), [])
        if id_value not in found:
            found.append(id_value)


type _Stored = Mapping[type[pydantic.BaseModel], Mapping[IdValue, pydantic.BaseModel]]
"""The stored Entities, by model and ID."""


class _Read:
    """One run of a View: the IDs it found a problem with, and an index per
    one-to-one Reference it follows backwards.

    `missing` are IDs no stored Entity has, by the Entity and field holding them.
    `shared` are IDs several Entities hold in a one-to-one field, by that Entity
    and field. Only IDs the View looks up are reported. `unmet` are the IDs of
    Entities whose required field is `None`, by that Entity and field.
    `unreferenced` are the IDs of subjects no Entity references by a one-to-one
    field the View requires, by that Entity and field.
    """

    def __init__(self, stored: _Stored) -> None:
        self.missing = _IdsFound()
        self.shared = _IdsFound()
        self.unmet = _IdsFound()
        self.unreferenced = _IdsFound()
        self._stored = stored
        self._indexes: dict[
            tuple[type[pydantic.BaseModel], FieldName],
            dict[IdValue, list[pydantic.BaseModel]],
        ] = {}

    def referencing(
        self, lookup: ir.ReverseLookup, id_value: IdValue
    ) -> pydantic.BaseModel | None:
        """The Entity whose `lookup.via` holds `id_value`: the first in insertion
        order when several do, which `shared` records, or `None`.

        The index of `lookup.via` is built the first time it is followed.
        """
        key = (lookup.target, lookup.via)
        if key not in self._indexes:
            index: dict[IdValue, list[pydantic.BaseModel]] = {}
            for entity in self._stored.get(lookup.target, {}).values():
                index.setdefault(getattr(entity, lookup.via.root), []).append(entity)
            self._indexes[key] = index
        found = self._indexes[key].get(id_value, [])
        if len(found) > 1:
            self.shared.add(lookup.target, lookup.via, id_value)
        return found[0] if found else None

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
            *(
                UnmetRequirement(entity=entity, field=field, ids=tuple(ids))
                for (entity, field), ids in self.unmet.ids.items()
            ),
            *(
                UnmetReverseRequirement(entity=entity, via=via, ids=tuple(ids))
                for (entity, via), ids in self.unreferenced.ids.items()
            ),
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
        UnknownViewError | NonEntitySourceError | DiagnosticsError[ReadDiagnostic],
    ]:
        """The View's results, or every Reference that points to a missing Entity,
        every ID several Entities share in a one-to-one Reference the View follows
        backwards, and every requirement of the View a row does not meet.

        Only Entities are stored, so a View over any other Source is an error. A
        row the View's Filters exclude is not looked up, so its References and
        requirements are not reported.
        """
        match self._views.get(view):
            case None:
                return Err(UnknownViewError(view, tuple(self._views)))
            case compiled:
                pass
        if compiled.source not in self._entity_graph.entities:
            return Err(NonEntitySourceError(view, compiled.source))
        run = _Read(self._entities)
        rows = tuple(
            self._row(compiled.relation, entity, run)
            for entity in self._entities.get(compiled.source, {}).values()
            if all(
                InMemoryViewEngine._holds(kept, entity)
                for kept in compiled.relation.filters
            )
        )
        return DiagnosticsError.or_ok(run.diagnostics(), lambda: rows)

    def _row(
        self, relation: ir.Relation, subject: pydantic.BaseModel, run: _Read
    ) -> ViewRow:
        return ViewRow(
            {field.name: self._field(field, subject, run) for field in relation.fields}
        )

    def _field(
        self, field: ir.RelationField, subject: pydantic.BaseModel, run: _Read
    ) -> FieldValue:
        match field:
            case ir.Field(value=value):
                pass
            case ir.NestedField(value=found, relation=relation):
                entity = self._follow(found.lookups(), subject, run)
                return None if entity is None else self._row(relation, entity, run)
        match value:
            case ir.SourceField() | ir.RequiredField():
                InMemoryViewEngine._check_required(value, subject, run)
                return getattr(subject, value.field.root)
            case ir.Lookup() | ir.ReverseLookup() | ir.ChainedLookup():
                return self._follow(value.lookups(), subject, run)
            case ir.Lifted(through=through, field=read):
                entity = self._follow(through, subject, run)
                if entity is None:
                    return None
                InMemoryViewEngine._check_required(read, entity, run)
                return getattr(entity, read.field.root)
            case ir.Added():
                return InMemoryViewEngine._default(field)

    def _follow(
        self,
        lookups: Iterable[ir.EntityLookup],
        subject: pydantic.BaseModel,
        run: _Read,
    ) -> pydantic.BaseModel | None:
        """The Entity found by following each of `lookups` in turn from `subject`,
        or `None` once one finds none."""
        found = subject
        for lookup in lookups:
            match self._lookup(lookup, found, run):
                case None:
                    return None
                case entity:
                    found = entity
        return found

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
        self,
        lookup: ir.EntityLookup,
        subject: pydantic.BaseModel,
        run: _Read,
    ) -> pydantic.BaseModel | None:
        """The Entity `lookup` finds from `subject`, or `None`.

        A Lookup gives `None` for an absent optional Reference, or when the Entity
        is missing, which `run` records. A ReverseLookup gives `None` when no
        Entity references the subject, and the first in insertion order when
        several do, which `run` records. When the View requires the lookup, `run`
        records finding nothing.
        """
        match lookup:
            case ir.Lookup():
                reference = lookup.reference.field
                InMemoryViewEngine._check_required(lookup.reference, subject, run)
                id_value = getattr(subject, reference.root)
                if id_value is None:
                    return None
                match self._entities.get(lookup.target, {}).get(id_value):
                    case None:
                        run.missing.add(type(subject), reference, id_value)
                        return None
                    case entity:
                        return entity
            case ir.ReverseLookup():
                subject_id = getattr(subject, EntityNode.primary_key.root)
                found = run.referencing(lookup, subject_id)
                match lookup.optionality, found:
                    case Optionality.REQUIRED, None:
                        run.unreferenced.add(lookup.target, lookup.via, subject_id)
                    case ((Optionality.REQUIRED | Optionality.OPTIONAL), _):
                        pass
                return found

    @staticmethod
    def _check_required(
        column: ir.SubjectColumn, subject: pydantic.BaseModel, run: _Read
    ) -> None:
        """Records in `run` that `subject`'s field `column` is `None`, if the View
        requires it."""
        match column:
            case ir.RequiredField(field=field) if getattr(subject, field.root) is None:
                run.unmet.add(
                    type(subject), field, getattr(subject, EntityNode.primary_key.root)
                )
            case ir.SourceField() | ir.RequiredField():
                pass
