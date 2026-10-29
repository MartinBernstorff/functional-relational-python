"""frp-ir: the relational expression a View's pipeline builds over its Source.

Steps only build nodes; nothing here is checked. The compiler checks a node tree
and derives its Relation: the fields a View result will have, each with its type
and the Value that computes it. Backends run the Relation, not the node tree.
"""

import copy
import enum
import typing
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Annotated

import pydantic
from pydantic.fields import FieldInfo
from safe_result import Err, Ok

from functional_relational_python.entity_graph.entity_graph import (
    AnnotationMetadata,
    EntityGraph,
    EntityNode,
    FieldName,
    NoEntityError,
    NoParent,
    NotAReferenceError,
    Optionality,
    Parent,
    RuntimeAnnotation,
    SpecialForm,
)
from functional_relational_python.entity_graph.id import IdType


class SourceName(pydantic.RootModel[str], frozen=True):
    @staticmethod
    def fake() -> SourceName:
        return SourceName("Company")


class FromSource(pydantic.BaseModel, frozen=True):
    """The root of every pipeline: all fields of a Source, a child Entity's
    inherited ones included, see `EntityGraph.fields_of`.

    A Source is any Pydantic model. It need not be an Entity in the EntityGraph.
    """

    source: type[pydantic.BaseModel]

    def source_name(self) -> SourceName:
        return SourceName(self.source.__name__)


class Omit(pydantic.BaseModel, frozen=True):
    """Removes every named or selected field."""

    input: Node
    fields: FieldSelection


class Rename(pydantic.BaseModel, frozen=True):
    input: Node
    field: FieldName
    new: FieldName


type FieldRenaming = Callable[[FieldName], FieldName]
"""Gives a field its new name from its old one."""


class RenameSelected(pydantic.BaseModel, frozen=True):
    """Renames every named or selected field to `new` of its name, all at once.

    Because the renames happen together, selected fields can swap names.
    """

    input: Node
    fields: FieldSelection
    new: FieldRenaming


type TypeTransform = Callable[[RuntimeAnnotation], RuntimeAnnotation]
"""Gives a field its new type from its old one."""


class MapType(pydantic.BaseModel, frozen=True):
    """Gives every named or selected field the type `transform` maps its type to.
    Only a Payload may map types; a View's mapping copies each value unchanged."""

    input: Node
    fields: FieldSelection
    transform: TypeTransform


class Pick(pydantic.BaseModel, frozen=True):
    """Keeps only the named or selected fields, in the input's field order: the
    complement of Omit."""

    input: Node
    fields: FieldSelection


class Named(pydantic.BaseModel, frozen=True):
    """The fields named `names`, in field order. Naming a field the relation does
    not have is a diagnostic."""

    names: tuple[FieldName, ...]


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
            case NestedField():
                return False
            case Field():
                pass
        match field.reference_source(entity_graph):
            case (source, info):
                pass
            case Field():
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


type Selector = Select | References
"""Fields chosen by what they are rather than by name. Selecting no field is no
defect."""

type FieldSelection = Named | Selector
"""The fields a step acts on."""


class Resolve(pydantic.BaseModel, frozen=True):
    """Replaces each selected Reference with the Entity it points to.

    `then` is applied to the referenced Entity, as if it were a View's Source.
    """

    input: Node
    fields: FieldSelection
    then: Callable[[Node], Node]


class Flatten(pydantic.BaseModel, frozen=True):
    """Replaces every named or selected resolved field with the fields of the
    Entity it holds."""

    input: Node
    fields: FieldSelection


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
    """Sets the default of every named or selected field in the generated model.

    `value` is unchecked: the compiler checks that it is a literal a generated
    model can write out.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    input: Node
    fields: FieldSelection
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


class NonNullable(pydantic.BaseModel, frozen=True):
    """Narrows every named or selected field from `X | None` to `X`.

    It drops no row: a `None` there is an unmet requirement when the View runs. A
    View may make fields non-nullable anywhere, a nested pipeline included; a
    Payload may not.
    """

    input: Node
    fields: FieldSelection


type Node = (
    FromSource
    | Omit
    | Rename
    | RenameSelected
    | MapType
    | Pick
    | Resolve
    | Flatten
    | Reverse
    | Default
    | Add
    | Where
    | NonNullable
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
    """The value of one of the subject's own fields, or of a child Entity's
    inherited ones.

    The subject is the View's Source, or inside a resolved field the referenced
    Entity.
    """

    field: FieldName

    def lookups(self) -> tuple[EntityLookup, ...]:
        return ()


class RequiredField(pydantic.BaseModel, frozen=True):
    """The value of one of the subject's optional fields, which the View requires:
    `None` there is an unmet requirement, not a value.

    Like a SourceField, it reads the subject's field `field`.
    """

    field: FieldName

    def lookups(self) -> tuple[EntityLookup, ...]:
        return ()


type SubjectColumn = SourceField | RequiredField
"""A value read from one of the subject's own fields, required or not."""


class _FindsEntity(pydantic.BaseModel, frozen=True):
    """What every lookup has: the Entity it finds, and whether it may find none."""

    target: type[pydantic.BaseModel]
    optionality: Optionality

    def annotation(self) -> type[pydantic.BaseModel | None]:
        """The looked-up value's type: the target Entity, optional if the lookup
        is."""
        return self.optionality.annotate(self.target)

    @staticmethod
    def optionality_of(lookups: Iterable[EntityLookup]) -> Optionality:
        """Whether following `lookups` in turn may find no Entity: whether any of
        their References is optional."""
        if any(lookup.optionality is Optionality.OPTIONAL for lookup in lookups):
            return Optionality.OPTIONAL
        return Optionality.REQUIRED


class Lookup(_FindsEntity, frozen=True):
    """The Entity a Reference points to, or `None` for an absent optional one.

    A RequiredField `reference` is an optional Reference the View requires, so
    the lookup is required and an absent Reference an unmet requirement.
    """

    reference: SubjectColumn
    id_type: IdType

    def required(self) -> Lookup:
        """This lookup, required by the View: an optional Reference's ID is
        required. A required Reference is left as it is."""
        match self.optionality:
            case Optionality.REQUIRED:
                return self
            case Optionality.OPTIONAL:
                return self.model_copy(
                    update={
                        "reference": RequiredField(field=self.reference.field),
                        "optionality": Optionality.REQUIRED,
                    }
                )

    def lookups(self) -> tuple[EntityLookup, ...]:
        return (self,)

    def keyed_by(self) -> FieldName:
        """The target's field the subject's ID is matched against: its primary
        key."""
        return EntityNode.primary_key


class ReverseLookup(_FindsEntity, frozen=True):
    """The Entity whose one-to-one Reference `via` holds the subject's primary key,
    or `None` when no Entity's does.

    `id_type` is the subject's primary key type, which `via` is typed with.

    Whether a referencing Entity exists is up to the data, so a ReverseLookup is
    optional unless the View requires it; then finding none is an unmet
    requirement.
    """

    via: FieldName
    id_type: IdType

    def required(self) -> ReverseLookup:
        """This lookup, required by the View."""
        return self.model_copy(update={"optionality": Optionality.REQUIRED})

    def lookups(self) -> tuple[EntityLookup, ...]:
        return (self,)

    def keyed_by(self) -> FieldName:
        """The target's field the subject's ID is matched against: `via`."""
        return self.via


type EntityLookup = Lookup | ReverseLookup
"""How a field holding an Entity finds it from the subject."""


class ChainedLookup(pydantic.BaseModel, frozen=True):
    """The Entity found by following each of `chain` in turn from the subject, or
    `None` when one finds none: a resolved field `flatten` lifted, or a lifted
    Reference resolved.
    """

    chain: Annotated[tuple[EntityLookup, ...], pydantic.Field(min_length=2)]

    def lookups(self) -> tuple[EntityLookup, ...]:
        return self.chain

    def required(self) -> ChainedLookup:
        """This lookup, required by the View: each lookup on the way must find an
        Entity."""
        return ChainedLookup(chain=tuple(lookup.required() for lookup in self.chain))

    def annotation(self) -> type[pydantic.BaseModel | None]:
        """The found value's type: the last target Entity, optional if any
        Reference on the way is."""
        return Lookup.optionality_of(self.chain).annotate(self.chain[-1].target)


type EntityValue = EntityLookup | ChainedLookup
"""The Entity a resolved field holds."""


class Added(pydantic.BaseModel, frozen=True):
    """No value from the subject: a field `add` declared.

    A mapping leaves it out, so the model fills it with its default.
    """

    def lookups(self) -> tuple[EntityLookup, ...]:
        return ()


class Undeclared(pydantic.BaseModel, frozen=True):
    """A lifted field the Entity found does not declare: one recovered from a
    defective step."""


class Lifted(pydantic.BaseModel, frozen=True):
    """The value of `field` of the Entity found by following each of `through` in
    turn from the subject: a field `flatten` lifted out of a resolved field. It is
    `None` when a Lookup on the way finds no Entity."""

    through: Annotated[tuple[EntityLookup, ...], pydantic.Field(min_length=1)]
    field: SubjectColumn

    def lookups(self) -> tuple[EntityLookup, ...]:
        return self.through

    def declared(self, entity_graph: EntityGraph) -> FieldInfo | Undeclared:
        """The field this value reads, as the Entity found declares it or
        inherits it."""
        fields = entity_graph.fields_of(self.through[-1].target)
        declared = fields.get(self.field.field)
        if declared is None:
            return Undeclared()
        return declared

    def required(self, entity_graph: EntityGraph) -> Lifted:
        """This value, required by the View: every Lookup on the way must find an
        Entity, and its field must hold a value. An Undeclared field is left as it
        is."""
        match self.declared(entity_graph):
            case FieldInfo(annotation=annotation):
                optionality = Optionality.of_aliased_annotation(annotation).optionality
            case Undeclared():
                optionality = Optionality.REQUIRED
        match optionality:
            case Optionality.OPTIONAL:
                field: SubjectColumn = RequiredField(field=self.field.field)
            case Optionality.REQUIRED:
                field = self.field
        return Lifted(
            through=tuple(lookup.required() for lookup in self.through),
            field=field,
        )


type SubjectValue = SourceField | RequiredField | EntityValue | Lifted
"""A value computed from the subject's fields."""

type Value = SubjectValue | Added
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
        name: FieldName, value: EntityValue, default: DefaultValue | NoDefault
    ) -> Field:
        """The field holding the Entity `value` finds, typed by
        `value.annotation()`."""
        return Field(
            name=name,
            info=FieldInfo.from_annotation(value.annotation()),
            value=value,
            default=default,
        )

    def annotation(self) -> RuntimeAnnotation:
        return self.info.annotation

    def retyped(self, annotation: RuntimeAnnotation) -> Field:
        """This field typed by `annotation`, keeping its default but not the old
        type's constraints."""
        return self._annotated(annotation, ())

    def reference_source(
        self, entity_graph: EntityGraph
    ) -> tuple[FieldName, FieldInfo] | Field:
        """The subject's field this field's ID is read from, with that field's
        info, if it is read unchanged from one: the subject's own, or one of the
        Entity a lifted field is read off. Otherwise, this field itself.

        Whether it is a Reference depends on the field's own type, not on the
        optional one a lifted field may have.
        """
        match self.value:
            case SourceField(field=source) | RequiredField(field=source):
                return (source, self.info)
            case Lifted(field=read) as lifted:
                # A field recovered from a defective step may name no field of
                # the Entity; it is no Reference then.
                match lifted.declared(entity_graph):
                    case FieldInfo() as declared:
                        return (read.field, declared)
                    case Undeclared():
                        return self
            case Lookup() | ReverseLookup() | ChainedLookup() | Added():
                return self

    def narrowed(self, annotation: RuntimeAnnotation) -> Field:
        """This field typed by `annotation`, a narrower type than its own, keeping
        its default and its type's constraints, which still hold."""
        return self._annotated(annotation, self.info.metadata)

    def without_none(self) -> Field:
        """This field typed without `None`, keeping its default, its type's
        constraints, and those of an `Annotated` inside the `None`: a field typed
        `PastAwareDatetime | None` is typed `PastAwareDatetime`.

        Those constraints move to the field, where pydantic records the metadata
        of a field typed `Annotated[X, ...]`.
        """
        split = Optionality.of_aliased_annotation(self.annotation())
        match split.optionality:
            case Optionality.REQUIRED:
                return self
            case Optionality.OPTIONAL:
                return self._annotated(
                    split.required, (*self.info.metadata, *split.metadata)
                )

    def admitting_none(self) -> Field:
        """This field typed to admit `None` too, keeping its default and its
        type's constraints.

        Unless its type admits `None` already, those constraints move inside the
        `None`, where pydantic records those of `Annotated[X, ...] | None`, so that
        they do not apply to `None`.
        """
        annotation = self.annotation()
        if (
            not self.info.metadata
            or Optionality.of_aliased_annotation(annotation).optionality
            is Optionality.OPTIONAL
        ):
            return self._annotated(
                Optionality.OPTIONAL.annotate(annotation), self.info.metadata
            )
        annotated = typing.cast("SpecialForm", Annotated)[
            (annotation, *self.info.metadata)
        ]
        return self._annotated(Optionality.OPTIONAL.annotate(annotated), ())

    def _annotated(
        self, annotation: RuntimeAnnotation, metadata: Iterable[AnnotationMetadata]
    ) -> Field:
        """This field typed by `annotation` with `metadata`, with a copy of its
        info."""
        info = copy.copy(self.info)
        # `FieldInfo.annotation` is typed narrower than the annotations a Source
        # can have, e.g. `int | None`, so set it untyped.
        object.__setattr__(info, "annotation", annotation)
        object.__setattr__(info, "metadata", list(metadata))
        return self.model_copy(update={"info": info})

    def optionality(self) -> Optionality:
        return Optionality.of_annotation(self.annotation()).optionality


class NestedField(pydantic.BaseModel, frozen=True):
    """A resolved or reversed field whose Entity went through a nested pipeline.

    Its type is a View model generated from `relation`, built from the Entity
    `value` finds.
    """

    name: FieldName
    value: EntityValue
    relation: Relation
    default: DefaultValue | NoDefault

    def optionality(self) -> Optionality:
        return Lookup.optionality_of(self.value.lookups())


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
    def of(node: FromSource, entity_graph: EntityGraph) -> Relation:
        return Relation(
            source=node.source_name(),
            fields=tuple(
                Field(
                    name=name,
                    info=info,
                    value=SourceField(field=name),
                    default=NoDefault(),
                )
                for name, info in entity_graph.fields_of(node.source).items()
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
                    used.extend(value.lookups())
                case NestedField(value=value, relation=nested):
                    used.extend((*value.lookups(), *nested.lookups()))
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


class Parents(pydantic.BaseModel, frozen=True):
    """The Lookup that finds each child Entity's parent by its parent link, see
    `ParentLink`.

    A Relation reads a field a child inherits as one of the child's own, as a
    Django child model holds it. A backend holding each Entity without the fields
    it inherits reads such a field off the parent instead.
    """

    lookups: Mapping[type[pydantic.BaseModel], Lookup]

    @staticmethod
    def of(entity_graph: EntityGraph) -> Parents:
        lookups: dict[type[pydantic.BaseModel], Lookup] = {}
        for model, node in entity_graph.entities.items():
            match node.parent:
                case Parent(link=link):
                    lookups[model] = Lookup(
                        reference=SourceField(field=link.field),
                        id_type=link.id_type,
                        target=link.target,
                        optionality=Optionality.REQUIRED,
                    )
                case NoParent():
                    pass
        return Parents(lookups=lookups)

    def through(
        self, subject: type[pydantic.BaseModel], field: FieldName
    ) -> tuple[Lookup, ...]:
        """The Lookups that find the Entity declaring `subject`'s `field`, from
        `subject`: none when `subject` declares it, or no ancestor does."""
        found: list[Lookup] = []
        while field.root not in subject.model_fields and subject in self.lookups:
            found.append(self.lookups[subject])
            subject = self.lookups[subject].target
        return tuple(found) if field.root in subject.model_fields else ()

    def read_off_parents(
        self, relation: Relation, subject: type[pydantic.BaseModel]
    ) -> Relation:
        """`relation` over `subject`, with each value it reads off a field a child
        inherits read off the parent declaring it, through `through`. Its Filters
        are unchanged: they read the subject's fields as its own."""
        return relation.with_fields(
            tuple(self._field(field, subject) for field in relation.fields)
        )

    def _field(
        self, field: RelationField, subject: type[pydantic.BaseModel]
    ) -> RelationField:
        match field:
            case NestedField(value=found, relation=nested):
                return field.model_copy(
                    update={
                        "value": self._found(found, subject),
                        "relation": self.read_off_parents(
                            nested, found.lookups()[-1].target
                        ),
                    }
                )
            case Field(value=value):
                pass
        read: Value
        match value:
            case SourceField() | RequiredField():
                through = self.through(subject, value.field)
                read = Lifted(through=through, field=value) if through else value
            case Lookup() | ReverseLookup() | ChainedLookup():
                read = self._found(value, subject)
            case Lifted(through=through, field=column):
                read = Lifted(
                    through=(
                        *self._chain(through, subject),
                        *self.through(through[-1].target, column.field),
                    ),
                    field=column,
                )
            case Added():
                read = value
        return field.model_copy(update={"value": read})

    def _chain(
        self, lookups: Iterable[EntityLookup], subject: type[pydantic.BaseModel]
    ) -> tuple[EntityLookup, ...]:
        """`lookups`, followed from `subject`, each Lookup of an inherited
        Reference after the Lookups that find the parent declaring it."""
        chain: list[EntityLookup] = []
        for lookup in lookups:
            match lookup:
                case Lookup(reference=reference):
                    chain.extend(self.through(subject, reference.field))
                case ReverseLookup():
                    pass
            chain.append(lookup)
            subject = lookup.target
        return tuple(chain)

    def _found(
        self, found: EntityValue, subject: type[pydantic.BaseModel]
    ) -> EntityValue:
        match self._chain(found.lookups(), subject):
            case (only,):
                return only
            case chain:
                return ChainedLookup(chain=chain)
