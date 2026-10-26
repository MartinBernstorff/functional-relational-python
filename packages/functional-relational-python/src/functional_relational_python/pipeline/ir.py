"""frp-ir: the relational expression a View's pipeline builds over its Source.

Steps only build nodes; nothing here is checked. The compiler checks a node tree
and derives its Relation: the fields a View result will have, each with its type
and the Value that computes it. Backends run the Relation, not the node tree.
"""

import enum
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


type UncheckedDefault = object
"""A value given to `default`, before the compiler checks that it is a literal."""


class Default(pydantic.BaseModel, frozen=True):
    """Sets the default of one named field, or of every selected field, in the
    generated model.

    `value` is unchecked: the compiler checks that it is a literal a generated
    model can write out.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    input: Node
    fields: FieldName | Select
    value: UncheckedDefault


type AddedType = type[object]
"""The type `add` gives a field, e.g. `Literal["company"]` or `Text | None`."""


class Add(pydantic.BaseModel, frozen=True):
    """Declares a new field, typed by `info.annotation`, that has no value from the
    subject.

    `default` is unchecked, or `NoDefault` when none was given: the compiler checks
    that it is a literal, and that a View's added field has one.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    input: Node
    name: FieldName
    info: FieldInfo
    default: UncheckedDefault | NoDefault


type Node = (
    FromSource | Omit | Rename | RenameSelected | Project | Resolve | Default | Add
)


class NoDefault(pydantic.BaseModel, frozen=True):
    """The field is required: its generated model gives it no default."""


class ScalarDefault(pydantic.RootModel[None | bool | int | float | str], frozen=True):
    """A default written as its `repr`: `None`, a boolean, a finite number or a
    string."""


class MissingDefault(pydantic.BaseModel, frozen=True):
    """Pydantic's `MISSING` sentinel: the field is left out when unset."""


class EmptyListDefault(pydantic.BaseModel, frozen=True):
    pass


class EmptyDictDefault(pydantic.BaseModel, frozen=True):
    pass


class EnumDefault(pydantic.BaseModel, frozen=True):
    member: enum.Enum


type DefaultValue = (
    ScalarDefault | MissingDefault | EmptyListDefault | EmptyDictDefault | EnumDefault
)
"""A literal a generated model can write out as a field's default."""


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


class Added(pydantic.BaseModel, frozen=True):
    """No value from the subject: a field `add` declared.

    A mapping leaves it out, so the model fills it with its default.
    """


type Value = SourceField | Lookup | Added
"""How a field of a View result is computed from the subject."""


class Field(pydantic.BaseModel, frozen=True):
    """A field typed by `info.annotation`: a Source's own type, or an Entity."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: FieldName
    info: FieldInfo
    value: Value
    default: DefaultValue | NoDefault

    @staticmethod
    def looked_up(
        name: FieldName, lookup: Lookup, default: DefaultValue | NoDefault
    ) -> Field:
        """The field holding the Entity `lookup` finds, typed by `lookup.annotation()`."""
        return Field(
            name=name,
            info=FieldInfo.from_annotation(lookup.annotation()),
            value=lookup,
            default=default,
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
    default: DefaultValue | NoDefault

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
                    default=NoDefault(),
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
                        case SourceField() | Added():
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

    def added(
        self, name: FieldName, info: FieldInfo, default: DefaultValue | NoDefault
    ) -> Relation:
        """Appends a field `add` declared, named `name` and typed by `info`."""
        return Relation(
            source=self.source,
            fields=(
                *self.fields,
                Field(
                    name=name,
                    info=info,
                    value=Added(),
                    default=default,
                ),
            ),
        )

    def defaulted(
        self, names: Collection[FieldName], default: DefaultValue
    ) -> Relation:
        """Gives each field in `names` the default `default`, replacing any other."""
        return Relation(
            source=self.source,
            fields=tuple(
                field.model_copy(update={"default": default})
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
