"""frp-ir: the relational expression a View's pipeline builds over its Source.

Steps only build nodes; nothing here is checked. The compiler checks a node tree
and derives its Relation: the fields a View result will have, each with its type
and the Value that computes it. Backends run the Relation, not the node tree.
"""

import copy
import enum
import typing
from collections.abc import Callable, Collection, Mapping
from typing import Annotated

import pydantic
from pydantic.fields import FieldInfo
from safe_result import Err, Ok

from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    EntityNode,
    FieldName,
    NoEntityError,
    NotAReferenceError,
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


class References(pydantic.BaseModel, frozen=True):
    """The References of a relation, in field order.

    Which fields are References depends on the catalog's `IdBase`, so unlike a
    Select, the compiler decides what this selects.
    """

    def selects(self, field: RelationField, entity_graph: EntityGraph) -> bool:
        """Whether `field` is a Reference, as `EntityGraph.reference_of` decides.

        A Reference to an ID type keying no Entity is selected, so that resolving
        it reports why it cannot be resolved.
        """
        match field:
            case Field(info=info, value=SourceField(field=source)):
                pass
            case Field() | NestedField():
                return False
        match entity_graph.reference_of(source, info):
            case Ok():
                return True
            case Err(error):
                pass
        match error:
            case NoEntityError():
                return True
            case NotAReferenceError():
                return False


class Resolve(pydantic.BaseModel, frozen=True):
    """Replaces each selected Reference with the Entity it points to.

    `then` is applied to the referenced Entity, as if it were a View's Source.
    """

    input: Node
    fields: Select | References
    then: Callable[[Node], Node]


class SoleReference(pydantic.BaseModel, frozen=True):
    """The one one-to-one Reference an Entity has to the subject, whichever field
    it is."""


class Reverse(pydantic.BaseModel, frozen=True):
    """Adds a field holding the `entity` whose one-to-one Reference `via` points to
    the subject, or `None` when none does.

    The subject is the View's Source, or inside a nested pipeline the Entity it
    runs on. `then` is applied to `entity`, as if it were a View's Source.
    """

    input: Node
    entity: type[pydantic.BaseModel]
    via: FieldName | SoleReference
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


class FieldRef(pydantic.BaseModel, frozen=True):
    """A Condition's reference to a top-level field, by its name at that point in
    the pipeline."""

    name: FieldName


type UncheckedLiteral = object
"""A value a Condition compares with, before the compiler checks it."""


class ConditionLiteral(pydantic.BaseModel, frozen=True):
    """A value a Condition compares a field with, e.g. `Text("Acme")`.

    `value` is unchecked: the compiler checks that it is an instance of the
    field's type, and a literal generated code can write out.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    value: UncheckedLiteral


class ComparisonOperator(enum.StrEnum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    def is_ordering(self) -> bool:
        """Whether the operator orders its operands, rather than equating them."""
        match self:
            case ComparisonOperator.EQ | ComparisonOperator.NE:
                return False
            case (
                ComparisonOperator.LT
                | ComparisonOperator.LE
                | ComparisonOperator.GT
                | ComparisonOperator.GE
            ):
                return True


class Comparison(pydantic.BaseModel, frozen=True):
    """`field <operator> operand`, with Python's two-valued null semantics."""

    operator: ComparisonOperator
    field: FieldRef
    operand: FieldRef | ConditionLiteral


class IsNull(pydantic.BaseModel, frozen=True):
    field: FieldRef


class And(pydantic.BaseModel, frozen=True):
    terms: Annotated[tuple[Condition, ...], pydantic.Field(min_length=1)]


class Or(pydantic.BaseModel, frozen=True):
    terms: Annotated[tuple[Condition, ...], pydantic.Field(min_length=1)]


class Not(pydantic.BaseModel, frozen=True):
    term: Condition


type Condition = Comparison | IsNull | And | Or | Not
"""What `where` keeps a row by, before the compiler checks it."""


class Where(pydantic.BaseModel, frozen=True):
    """Keeps only the rows for which `condition` is true.

    Only a View's own pipeline may filter, not a nested one or a Payload's.
    """

    input: Node
    condition: Condition


type Node = (
    FromSource
    | Omit
    | Rename
    | RenameSelected
    | Retype
    | Project
    | Resolve
    | Reverse
    | Default
    | Add
    | Where
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

    def keyed_by(self) -> FieldName:
        """The target's field the subject's ID is matched against: its primary
        key."""
        return EntityNode.primary_key


class ReverseLookup(pydantic.BaseModel, frozen=True):
    """The Entity whose one-to-one Reference `via` holds the subject's primary key,
    or `None` when no Entity's does.

    `id_type` is the subject's primary key type, which `via` is typed with.
    """

    optionality: typing.ClassVar[Optionality] = Optionality.OPTIONAL
    """Whether a referencing Entity exists is up to the data, so it is always
    optional."""

    target: type[pydantic.BaseModel]
    via: FieldName
    id_type: IdType

    def annotation(self) -> type[pydantic.BaseModel | None]:
        """The looked-up value's type: the target Entity, or `None`."""
        return self.optionality.annotate(self.target)

    def keyed_by(self) -> FieldName:
        """The target's field the subject's ID is matched against: `via`."""
        return self.via


type EntityLookup = Lookup | ReverseLookup
"""How a field holding an Entity finds it from the subject."""


class Added(pydantic.BaseModel, frozen=True):
    """No value from the subject: a field `add` declared.

    A mapping leaves it out, so the model fills it with its default.
    """


type Value = SourceField | Lookup | ReverseLookup | Added
"""How a field of a View result is computed from the subject."""


type LiteralValue = object
"""The value a Filter's literal stands for, e.g. `Text("Acme")`."""

type RootModelInstance = Annotated[pydantic.RootModel[object], pydantic.SkipValidation]
"""An instance of any RootModel. Pydantic would check against `RootModel[object]`,
which no RootModel subclasses, and convert the instance, so skip it."""


class ScalarLiteral(pydantic.RootModel[bool | int | float | str], frozen=True):
    """A Filter's literal written as its `repr`: a boolean, a finite number or a
    string."""

    def value(self) -> LiteralValue:
        return self.root


class EnumLiteral(pydantic.BaseModel, frozen=True):
    member: enum.Enum

    def value(self) -> LiteralValue:
        return self.member


class RootModelLiteral(pydantic.BaseModel, frozen=True):
    """A RootModel instance, e.g. `Text("Acme")`, written as its class called with
    its root."""

    instance: RootModelInstance
    root: FilterLiteral

    def value(self) -> LiteralValue:
        return self.instance


type FilterLiteral = ScalarLiteral | EnumLiteral | RootModelLiteral
"""A value a Filter compares with, as generated code can write it out. Never
`None`: a comparison with `None` is checked into a FilterIsNull."""


class FilterComparison(pydantic.BaseModel, frozen=True):
    """`field <operator> operand`, over the subject's own fields."""

    operator: ComparisonOperator
    field: SourceField
    operand: SourceField | FilterLiteral


class FilterIsNull(pydantic.BaseModel, frozen=True):
    field: SourceField


class FilterAnd(pydantic.BaseModel, frozen=True):
    """At least two terms: an `and` of one term is checked into that term."""

    terms: Annotated[tuple[Filter, ...], pydantic.Field(min_length=2)]


class FilterOr(pydantic.BaseModel, frozen=True):
    """At least two terms: an `or` of one term is checked into that term."""

    terms: Annotated[tuple[Filter, ...], pydantic.Field(min_length=2)]


class FilterNot(pydantic.BaseModel, frozen=True):
    term: Filter


type Filter = FilterComparison | FilterIsNull | FilterAnd | FilterOr | FilterNot
"""A checked Condition: its fields are the subject's own fields, whatever names
the pipeline gave them, and its literals have the types of the fields they are
compared with."""


class Field(pydantic.BaseModel, frozen=True):
    """A field typed by `info.annotation`: a Source's own type, or an Entity."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: FieldName
    info: FieldInfo
    value: Value
    default: DefaultValue | NoDefault

    @staticmethod
    def looked_up(
        name: FieldName, lookup: EntityLookup, default: DefaultValue | NoDefault
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

    def retyped(self, annotation: RuntimeAnnotation) -> Field:
        """This field typed by `annotation`, keeping its default but not the old
        type's constraints."""
        info = copy.copy(self.info)
        # `FieldInfo.annotation` is typed narrower than the annotations a Source
        # can have, e.g. `int | None`, so set it untyped.
        object.__setattr__(info, "annotation", annotation)
        object.__setattr__(info, "metadata", [])
        return self.model_copy(update={"info": info})

    def optionality(self) -> Optionality:
        return Optionality.of_annotation(self.annotation()).optionality


class NestedField(pydantic.BaseModel, frozen=True):
    """A resolved or reversed field whose Entity went through a nested pipeline.

    Its type is a View model generated from `relation`, built from the Entity
    `value` looks up.
    """

    name: FieldName
    value: EntityLookup
    relation: Relation
    default: DefaultValue | NoDefault

    def optionality(self) -> Optionality:
        return self.value.optionality


type RelationField = Field | NestedField

type FieldPredicate = Callable[[RelationField], bool]


class Relation(pydantic.BaseModel, frozen=True):
    """The checked shape of a node: the Source it starts from, its fields, and the
    Filters a row must pass to be a View result.

    A row is kept when every Filter in `filters` is true. Only a View's own
    Relation has Filters, not a nested one or a Payload's.
    """

    source: SourceName
    fields: tuple[RelationField, ...]
    filters: tuple[Filter, ...]

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
            filters=(),
        )

    def field_names(self) -> tuple[FieldName, ...]:
        return tuple(field.name for field in self.fields)

    def lookups(self) -> tuple[EntityLookup, ...]:
        """One lookup per Entity this relation looks up and the field it is
        matched on, nested ones included.

        Every Lookup of one Entity matches its primary key, so they count once;
        each ReverseLookup counts once per Entity and `via`. They are in the
        order they are first used.
        """
        used: list[EntityLookup] = []
        for field in self.fields:
            match field:
                case Field(value=value):
                    match value:
                        case SourceField() | Added():
                            pass
                        case Lookup() | ReverseLookup():
                            used.append(value)
                case NestedField(value=lookup, relation=nested):
                    used.extend((lookup, *nested.lookups()))
        first: dict[tuple[type[pydantic.BaseModel], FieldName], EntityLookup] = {}
        for lookup in used:
            _ = first.setdefault((lookup.target, lookup.keyed_by()), lookup)
        return tuple(first.values())

    def selected(self, fields: Select) -> tuple[FieldName, ...]:
        """The names of the fields `fields` selects, in field order."""
        return tuple(field.name for field in self.fields if fields.predicate(field))

    def with_fields(self, fields: tuple[RelationField, ...]) -> Relation:
        """This relation with `fields` instead of its own, keeping its Filters."""
        return Relation(source=self.source, fields=fields, filters=self.filters)

    def filtered(self, kept: Filter) -> Relation:
        """Also keeps only the rows for which `kept` is true."""
        return Relation(
            source=self.source, fields=self.fields, filters=(*self.filters, kept)
        )

    def unfiltered(self) -> Relation:
        """This relation without its Filters, keeping every row."""
        return Relation(source=self.source, fields=self.fields, filters=())

    def renamed(self, names: Mapping[FieldName, FieldName]) -> Relation:
        """Renames each key of `names` to its value in place, keeping its type."""
        return self.with_fields(
            tuple(
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
        return self.with_fields(
            (
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
        return self.with_fields(
            tuple(
                field.model_copy(update={"default": default})
                if field.name in names
                else field
                for field in self.fields
            ),
        )

    def without(self, names: Collection[FieldName]) -> Relation:
        return self.with_fields(
            tuple(field for field in self.fields if field.name not in names),
        )

    def keeping(self, names: Collection[FieldName]) -> Relation:
        return self.with_fields(
            tuple(field for field in self.fields if field.name in names),
        )
