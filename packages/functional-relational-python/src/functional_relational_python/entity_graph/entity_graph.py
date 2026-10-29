import functools
import operator
import types
import typing
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, RootModel
from pydantic.fields import FieldInfo
from safe_result import Err, Ok, Result

from functional_relational_python.entity_graph.id import (
    IdBase,
    IdType,
    OneToOne,
    OneToOneMarker,
    ParentLink,
)
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)


class FieldName(RootModel[str]):
    model_config = ConfigDict(frozen=True)

    @staticmethod
    def fake() -> FieldName:
        return FieldName("department_id")


type RuntimeAnnotation = object
"""A field's annotation as Python holds it at run time: a class, `None`, a generic
alias such as `list[int]`, a union, a `type` alias, or a `Literal` argument."""


type AnnotationMetadata = object
"""One of the objects after the type in `Annotated[X, ...]`, e.g. `OneToOne()`."""


class SpecialForm(typing.Protocol):
    """A special form such as `Literal` or `Annotated`, subscripted at run time. A
    type checker takes only types written out in one, so it is subscripted as this
    protocol."""

    def __getitem__(self, parameters: object, /) -> RuntimeAnnotation: ...


class Optionality(StrEnum):
    """Whether a value may be `None`.

    An optional annotation is `X | None`. This is the one place that says so: it
    builds optional annotations, at run time and as source text, and recognises
    them.
    """

    REQUIRED = "required"
    OPTIONAL = "optional"

    @staticmethod
    def of_annotation(annotation: RuntimeAnnotation) -> SplitAnnotation:
        """Whether `annotation` admits `None`, and `annotation` without `None`:
        `X | None` is an optional `X`, `X | Y | None` an optional `X | Y`.

        `Annotated[X, ...] | None` is an optional `X` with that metadata. Pydantic
        moves the metadata of `Annotated[X | None, ...]` to the field, but leaves
        an `Annotated` inside a union in the annotation.
        """
        args = typing.get_args(annotation)
        if typing.get_origin(annotation) is typing.Union and types.NoneType in args:
            rest = [arg for arg in args if arg is not types.NoneType]
            match rest:
                case [annotated] if typing.get_origin(annotated) is typing.Annotated:
                    required, *metadata = typing.get_args(annotated)
                    return SplitAnnotation(
                        optionality=Optionality.OPTIONAL,
                        required=required,
                        metadata=tuple(metadata),
                    )
                case _:
                    return SplitAnnotation(
                        optionality=Optionality.OPTIONAL,
                        required=functools.reduce(operator.or_, rest),
                        metadata=(),
                    )
        return SplitAnnotation(
            optionality=Optionality.REQUIRED, required=annotation, metadata=()
        )

    @staticmethod
    def of_aliased_annotation(annotation: RuntimeAnnotation) -> SplitAnnotation:
        """As `of_annotation`, looking through `type` aliases of the whole
        annotation, so `type MaybeText = Text | None` is an optional `Text`."""
        while isinstance(annotation, typing.TypeAliasType):
            annotation = annotation.__value__
        return Optionality.of_annotation(annotation)

    @typing.overload
    def annotate[T](self, annotation: type[T]) -> type[T | None]: ...

    @typing.overload
    def annotate(self, annotation: RuntimeAnnotation) -> RuntimeAnnotation: ...

    def annotate(self, annotation: RuntimeAnnotation) -> RuntimeAnnotation:
        """`annotation` as a run time annotation of this optionality. An optional
        annotation that admits `None` already is unchanged."""
        match self:
            case Optionality.REQUIRED:
                return annotation
            case Optionality.OPTIONAL:
                return functools.reduce(operator.or_, (annotation, types.NoneType))

    def annotate_text[T: RootModel[str]](self, annotation: T) -> T:
        """`annotation`, written as source, as an annotation of this optionality."""
        match self:
            case Optionality.REQUIRED:
                return annotation
            case Optionality.OPTIONAL:
                return annotation.model_validate(f"{annotation.root} | None")


class SplitAnnotation(BaseModel):
    """An annotation, told apart into whether it admits `None`, the rest, and the
    metadata of an `Annotated` the rest was wrapped in inside the union."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    optionality: Optionality
    required: RuntimeAnnotation
    metadata: tuple[AnnotationMetadata, ...]


class Cardinality(StrEnum):
    """How many Entities' References may point to one target Entity.

    A Reference is many-to-one unless the catalog's `OneToOneMarker` marks its
    field. Only a one-to-one Reference can be followed backwards to a single
    Entity.
    """

    MANY_TO_ONE = "many-to-one"
    ONE_TO_ONE = "one-to-one"

    @staticmethod
    def of_field(info: FieldInfo, one_to_one_marker: OneToOneMarker) -> Cardinality:
        """One-to-one if `one_to_one_marker` marks the field, e.g. with frp's
        `OneToOne` as `Annotated[X, OneToOne()]`, `Annotated[X | None, OneToOne()]`
        or `Annotated[X, OneToOne()] | None`."""
        if any(
            one_to_one_marker.marks(marker) for marker in Cardinality.markers_of(info)
        ):
            return Cardinality.ONE_TO_ONE
        return Cardinality.MANY_TO_ONE

    @staticmethod
    def markers_of(info: FieldInfo) -> tuple[AnnotationMetadata, ...]:
        """The metadata of the field, and of an `Annotated` inside its `None`."""
        return (
            *info.metadata,
            *Optionality.of_annotation(info.annotation).metadata,
        )


class ReferenceRole(StrEnum):
    """Whether a one-to-one Reference is the Entity's parent link, which
    `ParentLink` marks."""

    PLAIN = "plain"
    PARENT_LINK = "parent-link"

    @staticmethod
    def of_field(info: FieldInfo, cardinality: Cardinality) -> ReferenceRole:
        if cardinality == Cardinality.ONE_TO_ONE and any(
            isinstance(marker, ParentLink) for marker in Cardinality.markers_of(info)
        ):
            return ReferenceRole.PARENT_LINK
        return ReferenceRole.PLAIN


class Reference(BaseModel):
    """A field typed with an ID type, resolved to the Entity it points to."""

    model_config = ConfigDict(frozen=True)

    field: FieldName
    id_type: IdType
    target: type[BaseModel]
    optionality: Optionality
    cardinality: Cardinality


class Parent(BaseModel):
    """The Entity inherits the fields of `link.target`, its parent, through its
    parent link `link`."""

    model_config = ConfigDict(frozen=True)

    link: Reference

    @staticmethod
    def linked_by(
        entity: type[BaseModel], links: Sequence[Reference]
    ) -> Parent | NoParent | OptionalParentLink | SeveralParentLinks:
        """The parent of `entity`, whose Reference `links` are marked as parent
        links, or why it has none."""
        match links:
            case []:
                return NoParent()
            case [Reference(optionality=Optionality.REQUIRED) as link]:
                return Parent(link=link)
            case [link]:
                return OptionalParentLink(entity=entity, field=link.field)
            case _:
                return SeveralParentLinks(
                    entity=entity, fields=tuple(link.field for link in links)
                )


class NoParent(BaseModel):
    """The Entity has no parent link, so it inherits no fields."""

    model_config = ConfigDict(frozen=True)


class EntityNode(BaseModel):
    """An Entity in the EntityGraph: its model, primary key type, References, and
    the parent it inherits fields from."""

    model_config = ConfigDict(frozen=True)

    primary_key: ClassVar[FieldName] = FieldName("id")
    """The name of every Entity's primary key. It is never a Reference."""

    model: type[BaseModel]
    id_type: IdType
    references: tuple[Reference, ...]
    parent: Parent | NoParent


class DuplicatePrimaryKey(BaseModel, Diagnostic):
    """An ID type is the primary key of several Entities."""

    model_config = ConfigDict(frozen=True)

    id_type: IdType
    entities: frozenset[type[BaseModel]]

    def describe(self) -> DiagnosticMessage:
        names = ", ".join(sorted(entity.__qualname__ for entity in self.entities))
        return DiagnosticMessage(
            f"{self.id_type.__qualname__} is the primary key of several entities: {names}"
        )


class UnresolvedReference(BaseModel, Diagnostic):
    """A Reference whose ID type is the primary key of no Entity."""

    model_config = ConfigDict(frozen=True)

    entity: type[BaseModel]
    field: FieldName
    id_type: IdType

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root} references "
            f"{self.id_type.__qualname__}, which is the primary key of no entity"
        )


class NotAnEntity(BaseModel, Diagnostic):
    """A model given as an Entity has no `id` field typed with an ID type."""

    model_config = ConfigDict(frozen=True)

    model: type[BaseModel]
    id_base: IdBase

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.model.__qualname__} is not an entity: its `id` field must be "
            f"typed with a subclass of {self.id_base.root.__qualname__}"
        )


class OneToOneOnNonReference(BaseModel, Diagnostic):
    """A field marked one-to-one is not a Reference: it is the primary key, or not
    typed `X` or `X | None` for an ID type `X`."""

    model_config = ConfigDict(frozen=True)

    entity: type[BaseModel]
    field: FieldName
    one_to_one_marker: OneToOneMarker

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root} is marked "
            f"{self.one_to_one_marker.root.__qualname__}, but is not a Reference"
        )


class UnconfiguredOneToOne(BaseModel, Diagnostic):
    """A field is marked with frp's `OneToOne`, but the catalog's
    `OneToOneMarker` is another class, so `OneToOne` would be ignored."""

    model_config = ConfigDict(frozen=True)

    entity: type[BaseModel]
    field: FieldName
    one_to_one_marker: OneToOneMarker

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root} is marked with frp's "
            f"OneToOne, but the catalog marks one-to-one References with "
            f"{self.one_to_one_marker.root.__qualname__}"
        )


class SeveralParentLinks(BaseModel, Diagnostic):
    """An Entity marks more than one Reference as its parent link."""

    model_config = ConfigDict(frozen=True)

    entity: type[BaseModel]
    fields: tuple[FieldName, ...]

    def describe(self) -> DiagnosticMessage:
        names = ", ".join(field.root for field in self.fields)
        return DiagnosticMessage(
            f"{self.entity.__qualname__} has several parent links: {names}"
        )


class OptionalParentLink(BaseModel, Diagnostic):
    """An Entity's parent link may be `None`, but every child has a parent."""

    model_config = ConfigDict(frozen=True)

    entity: type[BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root} is a parent link, but "
            "may be None"
        )


class InheritanceCycle(BaseModel, Diagnostic):
    """Following parent links from these Entities leads back to them."""

    model_config = ConfigDict(frozen=True)

    entities: frozenset[type[BaseModel]]

    def describe(self) -> DiagnosticMessage:
        names = ", ".join(sorted(entity.__qualname__ for entity in self.entities))
        return DiagnosticMessage(f"These entities inherit from each other: {names}")


class InheritedFieldClash(BaseModel, Diagnostic):
    """A child Entity declares a field its ancestor `parent` declares too. Only
    `id` may be declared again: the child's takes the place of the parent's."""

    model_config = ConfigDict(frozen=True)

    entity: type[BaseModel]
    field: FieldName
    parent: type[BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root} is inherited from "
            f"{self.parent.__qualname__}, so {self.entity.__qualname__} cannot "
            "declare it"
        )


type EntityGraphDiagnostic = (
    DuplicatePrimaryKey
    | UnresolvedReference
    | NotAnEntity
    | OneToOneOnNonReference
    | UnconfiguredOneToOne
    | SeveralParentLinks
    | OptionalParentLink
    | InheritanceCycle
    | InheritedFieldClash
)


class NotAnIdAnnotationError(Exception):
    """A field's annotation is neither `X` nor `X | None` for an ID type `X`."""

    def __init__(self, info: FieldInfo, id_base: IdBase) -> None:
        self.info = info
        super().__init__(
            f"{info.annotation!r} is not a subclass of "
            f"{id_base.root.__qualname__}, optional or not"
        )


class NoEntityError(Exception):
    """An ID type is the primary key of no Entity in the EntityGraph."""

    def __init__(self, id_type: IdType) -> None:
        self.id_type = id_type
        super().__init__(f"{id_type.__qualname__} is the primary key of no entity")


class IdAnnotation(BaseModel):
    """A field annotated `X` or `X | None`, where `X` is an ID type, and whether
    it is marked one-to-one.

    Only these two shapes are References; `list[X]` and the like are not.
    """

    model_config = ConfigDict(frozen=True)

    id_type: IdType
    optionality: Optionality
    cardinality: Cardinality

    @staticmethod
    def of(
        info: FieldInfo, id_base: IdBase, one_to_one_marker: OneToOneMarker
    ) -> Result[IdAnnotation, NotAnIdAnnotationError]:
        split = Optionality.of_annotation(info.annotation)
        if id_base.includes(split.required):
            return Ok(
                IdAnnotation(
                    id_type=split.required,
                    optionality=split.optionality,
                    cardinality=Cardinality.of_field(info, one_to_one_marker),
                )
            )
        return Err(NotAnIdAnnotationError(info, id_base))

    @staticmethod
    def of_reference(
        field: FieldName,
        info: FieldInfo,
        id_base: IdBase,
        one_to_one_marker: OneToOneMarker,
    ) -> Result[IdAnnotation, NotAReferenceError]:
        """The ID type and optionality of `field`, if it is a Reference.

        `field` is the subject's own field the value comes from, so a renamed
        primary key is still the primary key.
        """
        if field == EntityNode.primary_key:
            return Err(NotAReferenceError(field))
        match IdAnnotation.of(info, id_base, one_to_one_marker):
            case Ok(annotation):
                return Ok(annotation)
            case Err():
                return Err(NotAReferenceError(field))


class NotAReferenceError(Exception):
    """A field is the primary key, or is not typed `X` or `X | None` for an ID `X`."""

    def __init__(self, field: FieldName) -> None:
        self.field = field
        super().__init__(f"{field.root} is not a Reference")


class _ReferenceField(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: FieldName
    annotation: IdAnnotation
    role: ReferenceRole


class _ModelFields(BaseModel):
    """A model's References, and the diagnostics of how its fields are marked."""

    model_config = ConfigDict(frozen=True)

    references: tuple[_ReferenceField, ...]
    diagnostics: tuple[OneToOneOnNonReference | UnconfiguredOneToOne, ...]

    @staticmethod
    def of(
        model: type[BaseModel], id_base: IdBase, one_to_one_marker: OneToOneMarker
    ) -> _ModelFields:
        references: list[_ReferenceField] = []
        diagnostics: list[OneToOneOnNonReference | UnconfiguredOneToOne] = []
        for name, info in model.model_fields.items():
            field = FieldName(name)
            match IdAnnotation.of_reference(field, info, id_base, one_to_one_marker):
                case Ok(annotation):
                    references.append(
                        _ReferenceField(
                            field=field,
                            annotation=annotation,
                            role=ReferenceRole.of_field(info, annotation.cardinality),
                        )
                    )
                    cardinality = annotation.cardinality
                case Err():
                    cardinality = Cardinality.of_field(info, one_to_one_marker)
                    if cardinality == Cardinality.ONE_TO_ONE:
                        diagnostics.append(
                            OneToOneOnNonReference(
                                entity=model,
                                field=field,
                                one_to_one_marker=one_to_one_marker,
                            )
                        )
            if cardinality == Cardinality.MANY_TO_ONE and any(
                isinstance(marker, OneToOne) for marker in Cardinality.markers_of(info)
            ):
                diagnostics.append(
                    UnconfiguredOneToOne(
                        entity=model, field=field, one_to_one_marker=one_to_one_marker
                    )
                )
        return _ModelFields(
            references=tuple(references), diagnostics=tuple(diagnostics)
        )


class EntityGraph(BaseModel):
    """All Entities, with every Reference resolved to the Entity it points to.

    `id_base` decides which types are ID types, and so which fields are References.
    `one_to_one_marker` decides which References are one-to-one.
    """

    model_config = ConfigDict(frozen=True)

    entities: Mapping[type[BaseModel], EntityNode]
    id_base: IdBase
    one_to_one_marker: OneToOneMarker

    @staticmethod
    def build(
        entities: Iterable[type[BaseModel]],
        id_base: IdBase,
        one_to_one_marker: OneToOneMarker,
    ) -> Result[EntityGraph, DiagnosticsError[EntityGraphDiagnostic]]:
        """Build the graph, or return every diagnostic at once.

        The order of `entities` does not affect the result.
        """
        diagnostics: set[EntityGraphDiagnostic] = set()
        keyed_by: defaultdict[IdType, set[type[BaseModel]]] = defaultdict(set)
        reference_fields: dict[type[BaseModel], tuple[_ReferenceField, ...]] = {}
        for model in set(entities):
            match model.model_fields.get(EntityNode.primary_key.root):
                case FieldInfo(annotation=id_type) if id_base.includes(id_type):
                    keyed_by[id_type].add(model)
                case _:
                    diagnostics.add(NotAnEntity(model=model, id_base=id_base))
            fields = _ModelFields.of(model, id_base, one_to_one_marker)
            reference_fields[model] = fields.references
            diagnostics.update(fields.diagnostics)

        owners: dict[IdType, type[BaseModel]] = {}
        for id_type, models in keyed_by.items():
            if len(models) > 1:
                diagnostics.add(
                    DuplicatePrimaryKey(id_type=id_type, entities=frozenset(models))
                )
            else:
                (owners[id_type],) = models

        nodes: dict[type[BaseModel], EntityNode] = {}
        for id_type, models in keyed_by.items():
            for model in models:
                references: list[Reference] = []
                for reference in reference_fields[model]:
                    annotation = reference.annotation
                    if annotation.id_type in owners:
                        references.append(
                            Reference(
                                field=reference.field,
                                id_type=annotation.id_type,
                                target=owners[annotation.id_type],
                                optionality=annotation.optionality,
                                cardinality=annotation.cardinality,
                            )
                        )
                    elif annotation.id_type not in keyed_by:
                        # An ID type keying several Entities is already a diagnostic,
                        # so References to it are not reported again.
                        diagnostics.add(
                            UnresolvedReference(
                                entity=model,
                                field=reference.field,
                                id_type=annotation.id_type,
                            )
                        )
                links = {
                    reference.field
                    for reference in reference_fields[model]
                    if reference.role is ReferenceRole.PARENT_LINK
                }
                parent = Parent.linked_by(
                    model,
                    [reference for reference in references if reference.field in links],
                )
                if not isinstance(parent, Parent | NoParent):
                    diagnostics.add(parent)
                    parent = NoParent()
                nodes[model] = EntityNode(
                    model=model,
                    id_type=id_type,
                    references=tuple(references),
                    parent=parent,
                )

        graph = EntityGraph(
            entities=nodes, id_base=id_base, one_to_one_marker=one_to_one_marker
        )
        diagnostics.update(graph._inheritance_diagnostics())

        # Sorted, so the order of `entities` does not affect the order reported.
        return DiagnosticsError.or_ok(
            sorted(diagnostics, key=lambda diagnostic: diagnostic.describe().root),
            lambda: graph,
        )

    def _inheritance_diagnostics(
        self,
    ) -> set[InheritanceCycle | InheritedFieldClash]:
        """A diagnostic per cycle of parent links, however many Entities lead into
        it; else one per field a child declares that it inherits."""
        cycles: set[InheritanceCycle | InheritedFieldClash] = set()
        for model in self.entities:
            path: list[type[BaseModel]] = []
            current = model
            while current not in path:
                path.append(current)
                match self.parent_of(current):
                    case Parent(link=link):
                        current = link.target
                    case NoParent():
                        break
            else:
                cycles.add(
                    InheritanceCycle(entities=frozenset(path[path.index(current) :]))
                )
        if cycles:
            # Following parent links around a cycle never ends.
            return cycles
        return {
            InheritedFieldClash(
                entity=model,
                field=FieldName(name),
                parent=self.declaring(parent, FieldName(name)),
            )
            for model in self.entities
            for parent in self.ancestors(model)[:1]
            for name in model.model_fields
            if FieldName(name) != EntityNode.primary_key
            and FieldName(name) in self.fields_of(parent)
        }

    def parent_of(self, model: type[BaseModel]) -> Parent | NoParent:
        """The parent `model` inherits fields from. A Source that is no Entity
        has none."""
        match self.entities.get(model):
            case None:
                return NoParent()
            case node:
                return node.parent

    def ancestors(self, model: type[BaseModel]) -> tuple[type[BaseModel], ...]:
        """The Entities `model` inherits fields from: its parent first, then the
        parent's parent, and so on."""
        match self.parent_of(model):
            case Parent(link=link):
                return (link.target, *self.ancestors(link.target))
            case NoParent():
                return ()

    def fields_of(self, source: type[BaseModel]) -> Mapping[FieldName, FieldInfo]:
        """The fields a View or Payload over `source` starts with: its own, and
        for a child Entity first those it inherits from its parent, in the
        parent's order, its own `id` in place of the parent's."""
        own = {FieldName(name): info for name, info in source.model_fields.items()}
        match self.parent_of(source):
            case Parent(link=link):
                return {**self.fields_of(link.target), **own}
            case NoParent():
                return own

    def ancestry(
        self, model: type[BaseModel], field: FieldName
    ) -> tuple[Reference, ...]:
        """The parent links followed from `model` to the Entity declaring `field`:
        none for a field `model` declares itself, or no ancestor declares."""
        if field.root in model.model_fields:
            return ()
        match self.parent_of(model):
            case Parent(link=link):
                inherited = self.ancestry(link.target, field)
                if inherited or field.root in link.target.model_fields:
                    return (link, *inherited)
                return ()
            case NoParent():
                return ()

    def declaring(self, model: type[BaseModel], field: FieldName) -> type[BaseModel]:
        """The Entity among `model` and its ancestors that declares `field`, see
        `ancestry`."""
        ancestry = self.ancestry(model, field)
        return ancestry[-1].target if ancestry else model

    def entity_keyed_by(
        self, id_type: IdType
    ) -> Result[type[BaseModel], NoEntityError]:
        """The Entity whose primary key `id_type` is."""
        for model, node in self.entities.items():
            if node.id_type is id_type:
                return Ok(model)
        return Err(NoEntityError(id_type))

    def reference_of(
        self, field: FieldName, info: FieldInfo
    ) -> Result[Reference, NotAReferenceError | NoEntityError]:
        """The Entity the Reference `field`, typed by `info`, points to.

        `field` is the subject's own field the value comes from.
        """
        match IdAnnotation.of_reference(
            field, info, self.id_base, self.one_to_one_marker
        ):
            case Err() as not_a_reference:
                return not_a_reference
            case Ok(annotation):
                pass
        match self.entity_keyed_by(annotation.id_type):
            case Err() as no_entity:
                return no_entity
            case Ok(target):
                return Ok(
                    Reference(
                        field=field,
                        id_type=annotation.id_type,
                        target=target,
                        optionality=annotation.optionality,
                        cardinality=annotation.cardinality,
                    )
                )
