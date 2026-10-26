"""frp-ir: the relational expression a View's pipeline builds over its Source.

Steps only build nodes; nothing here is checked. The compiler checks a node tree
and derives its Relation: the fields a View result will have, each with its type
and the Value that computes it. Backends run the Relation, not the node tree.
"""

import copy
from collections.abc import Callable, Collection, Mapping

import pydantic
from pydantic.fields import FieldInfo

from functional_relational_python.entity_graph.entity_graph import (
    FieldName,
    Optionality,
    RuntimeAnnotation,
)
from functional_relational_python.entity_graph.id import IdType


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
    """Removes one named field, or every selected field."""

    input: Node
    fields: FieldName | Select


class Rename(pydantic.BaseModel, frozen=True):
    input: Node
    field: FieldName
    new: FieldName


type FieldRenaming = Callable[[FieldName], FieldName]
"""Gives a field its new name from its old one."""


class RenameSelected(pydantic.BaseModel, frozen=True):
    """Renames every selected field to `new` of its name, all at once.

    Because the renames happen together, selected fields can swap names.
    """

    input: Node
    fields: Select
    new: FieldRenaming


type TypeTransform = Callable[[RuntimeAnnotation], RuntimeAnnotation]
"""Gives a field its new type from its old one."""


class Retype(pydantic.BaseModel, frozen=True):
    """Gives the named field, or every selected field, the type `transform` maps
    its type to. Only a Payload may retype; a View's mapping copies each value
    unchanged."""

    input: Node
    fields: FieldName | Select
    transform: TypeTransform


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


type Node = FromSource | Omit | Rename | RenameSelected | Retype | Project | Resolve


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

    def retyped(self, annotation: RuntimeAnnotation) -> Field:
        """This field typed by `annotation`. It keeps the rest of its `info`, such
        as its default, but not the old type's constraints."""
        info = copy.copy(self.info)
        # `FieldInfo.annotation` is typed narrower than the annotations a Source
        # can have, e.g. `int | None`, so set it untyped.
        object.__setattr__(info, "annotation", annotation)
        object.__setattr__(info, "metadata", [])
        return self.model_copy(update={"info": info})

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

    def selected(self, fields: Select) -> tuple[FieldName, ...]:
        """The names of the fields `fields` selects, in field order."""
        return tuple(field.name for field in self.fields if fields.predicate(field))

    def renamed(self, names: Mapping[FieldName, FieldName]) -> Relation:
        """Renames each key of `names` to its value in place, keeping its type."""
        return Relation(
            source=self.source,
            fields=tuple(
                field.model_copy(update={"name": names[field.name]})
                if field.name in names
                else field
                for field in self.fields
            ),
        )

    def without(self, names: Collection[FieldName]) -> Relation:
        return Relation(
            source=self.source,
            fields=tuple(field for field in self.fields if field.name not in names),
        )

    def keeping(self, names: Collection[FieldName]) -> Relation:
        return Relation(
            source=self.source,
            fields=tuple(field for field in self.fields if field.name in names),
        )
