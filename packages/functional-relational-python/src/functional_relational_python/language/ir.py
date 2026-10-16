"""frp-ir: the relational expression a View's pipeline builds over its Source.

Steps only build nodes; nothing here is checked. The compiler checks a node tree
and derives its Relation: the fields a View result will have, each with its type
and the Value that computes it. Backends run the Relation, not the node tree.
"""

from collections.abc import Callable, Collection

import pydantic
from pydantic.fields import FieldInfo

from functional_relational_python.domain.entity_graph import (
    FieldName,
    Optionality,
    RuntimeAnnotation,
)
from functional_relational_python.domain.id import IdType


class SourceName(pydantic.RootModel[str], frozen=True):
    @staticmethod
    def fake() -> SourceName:
        return SourceName("Company")


class FromSource(pydantic.BaseModel, frozen=True):
    """The root of every pipeline: all fields of a Source.

    A Source is any Pydantic model. It need not be an Entity in the EntityGraph.
    """

    source: type[pydantic.BaseModel]

    def source_name(self) -> SourceName:
        return SourceName(self.source.__name__)


class Omit(pydantic.BaseModel, frozen=True):
    input: Node
    field: FieldName


class Rename(pydantic.BaseModel, frozen=True):
    input: Node
    field: FieldName
    new: FieldName


class Project(pydantic.BaseModel, frozen=True):
    """Keeps only `fields`, in the input's field order: the complement of Omit."""

    input: Node
    fields: tuple[FieldName, ...]


class Select(pydantic.BaseModel, frozen=True):
    """The fields of a relation that satisfy `predicate`, in field order."""

    predicate: FieldPredicate


class Resolve(pydantic.BaseModel, frozen=True):
    """Replaces each selected Reference with the Entity it points to.

    `then` is applied to the referenced Entity, as if it were a View's Source.
    """

    input: Node
    fields: Select
    then: Callable[[Node], Node]


type Node = FromSource | Omit | Rename | Project | Resolve


class SourceField(pydantic.BaseModel, frozen=True):
    """The value of one of the subject's own fields.

    The subject is the View's Source, or inside a resolved field the referenced
    Entity.
    """

    field: FieldName


class Lookup(pydantic.BaseModel, frozen=True):
    """The Entity a Reference points to, or `None` for an absent optional one."""

    reference: SourceField
    id_type: IdType
    target: type[pydantic.BaseModel]
    optionality: Optionality

    def annotation(self) -> type[pydantic.BaseModel | None]:
        """The looked-up value's type: the target Entity, optional if the
        Reference is."""
        return self.optionality.annotate(self.target)


type Value = SourceField | Lookup
"""How a field of a View result is computed from the subject."""


class Field(pydantic.BaseModel, frozen=True):
    """A field typed by `info.annotation`: a Source's own type, or an Entity."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: FieldName
    info: FieldInfo
    value: Value

    @staticmethod
    def looked_up(name: FieldName, lookup: Lookup) -> Field:
        """The field holding the Entity `lookup` finds, typed by `lookup.annotation()`."""
        return Field(
            name=name, info=FieldInfo.from_annotation(lookup.annotation()), value=lookup
        )

    def annotation(self) -> RuntimeAnnotation:
        return self.info.annotation

    def optionality(self) -> Optionality:
        return Optionality.of_annotation(self.annotation()).optionality


class NestedField(pydantic.BaseModel, frozen=True):
    """A resolved field whose Entity went through a nested pipeline.

    Its type is a View model generated from `relation`, built from the Entity
    `value` looks up.
    """

    name: FieldName
    value: Lookup
    relation: Relation

    def optionality(self) -> Optionality:
        return self.value.optionality


type RelationField = Field | NestedField

type FieldPredicate = Callable[[RelationField], bool]


class Relation(pydantic.BaseModel, frozen=True):
    """The checked shape of a node: the Source it starts from, and its fields."""

    source: SourceName
    fields: tuple[RelationField, ...]

    @staticmethod
    def of(node: FromSource) -> Relation:
        return Relation(
            source=node.source_name(),
            fields=tuple(
                Field(
                    name=FieldName(name),
                    info=info,
                    value=SourceField(field=FieldName(name)),
                )
                for name, info in node.source.model_fields.items()
            ),
        )

    def field_names(self) -> tuple[FieldName, ...]:
        return tuple(field.name for field in self.fields)

    def lookups(self) -> tuple[Lookup, ...]:
        """One Lookup per Entity this relation looks up, nested ones included.

        They are in the order the Entities are first looked up.
        """
        used: list[Lookup] = []
        for field in self.fields:
            match field:
                case Field(value=value):
                    match value:
                        case SourceField():
                            pass
                        case Lookup():
                            used.append(value)
                case NestedField(value=lookup, relation=nested):
                    used.extend((lookup, *nested.lookups()))
        first: dict[type[pydantic.BaseModel], Lookup] = {}
        for lookup in used:
            if lookup.target not in first:
                first[lookup.target] = lookup
        return tuple(first.values())

    def renamed(self, old: FieldName, new: FieldName) -> Relation:
        """Renames `old` to `new` in place, keeping its type."""
        return Relation(
            source=self.source,
            fields=tuple(
                field.model_copy(update={"name": new}) if field.name == old else field
                for field in self.fields
            ),
        )

    def without(self, name: FieldName) -> Relation:
        return Relation(
            source=self.source,
            fields=tuple(field for field in self.fields if field.name != name),
        )

    def keeping(self, names: Collection[FieldName]) -> Relation:
        return Relation(
            source=self.source,
            fields=tuple(field for field in self.fields if field.name in names),
        )
