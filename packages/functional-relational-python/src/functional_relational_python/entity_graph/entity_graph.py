import functools
import operator
import types
import typing
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, RootModel
from pydantic.fields import FieldInfo
from safe_result import Err, Ok, Result

from functional_relational_python.entity_graph.id import Id, IdType, OneToOne
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
        `X | None` is an optional `X`, `X | Y | None` an optional `X | Y`."""
        args = typing.get_args(annotation)
        if typing.get_origin(annotation) is typing.Union and types.NoneType in args:
            rest = [arg for arg in args if arg is not types.NoneType]
            return SplitAnnotation(
                optionality=Optionality.OPTIONAL,
                required=functools.reduce(operator.or_, rest),
            )
        return SplitAnnotation(optionality=Optionality.REQUIRED, required=annotation)

    def annotate[T](self, annotation: type[T]) -> type[T | None]:
        """`annotation` as a run time annotation of this optionality."""
        match self:
            case Optionality.REQUIRED:
                return annotation
            case Optionality.OPTIONAL:
                return annotation | None

    def annotate_text[T: RootModel[str]](self, annotation: T) -> T:
        """`annotation`, written as source, as an annotation of this optionality."""
        match self:
            case Optionality.REQUIRED:
                return annotation
            case Optionality.OPTIONAL:
                return annotation.model_validate(f"{annotation.root} | None")


class SplitAnnotation(BaseModel):
    """An annotation, told apart into whether it admits `None` and the rest."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    optionality: Optionality
    required: RuntimeAnnotation


class Cardinality(StrEnum):
    """How many Entities' References may point to one target Entity.

    A Reference is many-to-one unless its field is marked `OneToOne`. Only a
    one-to-one Reference can be followed backwards to a single Entity.
    """

    MANY_TO_ONE = "many-to-one"
    ONE_TO_ONE = "one-to-one"

    @staticmethod
    def of_field(info: FieldInfo) -> Cardinality:
        if any(isinstance(marker, OneToOne) for marker in info.metadata):
            return Cardinality.ONE_TO_ONE
        return Cardinality.MANY_TO_ONE


class Reference(BaseModel):
    """A field typed with an `Id` subclass, resolved to the Entity it points to."""

    model_config = ConfigDict(frozen=True)

    field: FieldName
    id_type: IdType
    target: type[BaseModel]
    optionality: Optionality
    cardinality: Cardinality


class EntityNode(BaseModel):
    """An Entity in the EntityGraph: its model, primary key type and References."""

    model_config = ConfigDict(frozen=True)

    primary_key: ClassVar[FieldName] = FieldName("id")
    """The name of every Entity's primary key. It is never a Reference."""

    model: type[BaseModel]
    id_type: IdType
    references: tuple[Reference, ...]


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
    """A model given as an Entity has no `id` field typed with an `Id` subclass."""

    model_config = ConfigDict(frozen=True)

    model: type[BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.model.__qualname__} is not an entity: its `id` field must be "
            f"typed with an Id subclass"
        )


class OneToOneOnNonReference(BaseModel, Diagnostic):
    """A field marked `OneToOne` is not a Reference: it is the primary key, or not
    typed `X` or `X | None` for an `Id` subclass `X`."""

    model_config = ConfigDict(frozen=True)

    entity: type[BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root} is marked OneToOne, but "
            "is not a Reference"
        )


type EntityGraphDiagnostic = (
    DuplicatePrimaryKey | UnresolvedReference | NotAnEntity | OneToOneOnNonReference
)


class NotAnIdAnnotationError(Exception):
    """A field's annotation is neither `X` nor `X | None` for an `Id` subclass `X`."""

    def __init__(self, info: FieldInfo) -> None:
        self.info = info
        super().__init__(f"{info.annotation!r} is not an Id type, optional or not")


class NoEntityError(Exception):
    """An ID type is the primary key of no Entity in the EntityGraph."""

    def __init__(self, id_type: IdType) -> None:
        self.id_type = id_type
        super().__init__(f"{id_type.__qualname__} is the primary key of no entity")


class IdAnnotation(BaseModel):
    """A field annotated `X` or `X | None`, where `X` is an `Id` subclass, and
    whether it is marked `OneToOne`.

    Only these two shapes are References; `list[X]` and the like are not.
    """

    model_config = ConfigDict(frozen=True)

    id_type: IdType
    optionality: Optionality
    cardinality: Cardinality

    @staticmethod
    def of(info: FieldInfo) -> Result[IdAnnotation, NotAnIdAnnotationError]:
        split = Optionality.of_annotation(info.annotation)
        match split.required:
            case type() as id_type if issubclass(id_type, Id):
                return Ok(
                    IdAnnotation(
                        id_type=id_type,
                        optionality=split.optionality,
                        cardinality=Cardinality.of_field(info),
                    )
                )
            case _:
                return Err(NotAnIdAnnotationError(info))

    @staticmethod
    def of_reference(
        field: FieldName, info: FieldInfo
    ) -> Result[IdAnnotation, NotAReferenceError]:
        """The ID type and optionality of `field`, if it is a Reference.

        `field` is the subject's own field the value comes from, so a renamed
        primary key is still the primary key.
        """
        if field == EntityNode.primary_key:
            return Err(NotAReferenceError(field))
        match IdAnnotation.of(info):
            case Ok(annotation):
                return Ok(annotation)
            case Err():
                return Err(NotAReferenceError(field))


class NotAReferenceError(Exception):
    """A field is the primary key, or is not typed `X` or `X | None` for an Id `X`."""

    def __init__(self, field: FieldName) -> None:
        self.field = field
        super().__init__(f"{field.root} is not a Reference")


class _ReferenceField(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: FieldName
    annotation: IdAnnotation


class EntityGraph(BaseModel):
    """All Entities, with every Reference resolved to the Entity it points to."""

    model_config = ConfigDict(frozen=True)

    entities: Mapping[type[BaseModel], EntityNode]

    @staticmethod
    def build(
        entities: Iterable[type[BaseModel]],
    ) -> Result[EntityGraph, DiagnosticsError[EntityGraphDiagnostic]]:
        """Build the graph, or return every diagnostic at once.

        The order of `entities` does not affect the result.
        """
        diagnostics: set[EntityGraphDiagnostic] = set()
        keyed_by: defaultdict[IdType, set[type[BaseModel]]] = defaultdict(set)
        for model in set(entities):
            match model.model_fields.get(EntityNode.primary_key.root):
                case FieldInfo(annotation=type() as id_type) if issubclass(id_type, Id):
                    keyed_by[id_type].add(model)
                case _:
                    diagnostics.add(NotAnEntity(model=model))
            diagnostics.update(
                OneToOneOnNonReference(entity=model, field=FieldName(name))
                for name, info in model.model_fields.items()
                if Cardinality.of_field(info) == Cardinality.ONE_TO_ONE
                and isinstance(IdAnnotation.of_reference(FieldName(name), info), Err)
            )

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
                for reference in EntityGraph._reference_fields(model):
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
                nodes[model] = EntityNode(
                    model=model, id_type=id_type, references=tuple(references)
                )

        # Sorted, so the order of `entities` does not affect the order reported.
        return DiagnosticsError.or_ok(
            sorted(diagnostics, key=lambda diagnostic: diagnostic.describe().root),
            lambda: EntityGraph(entities=nodes),
        )

    @staticmethod
    def _reference_fields(model: type[BaseModel]) -> Iterator[_ReferenceField]:
        for name, info in model.model_fields.items():
            field = FieldName(name)
            match IdAnnotation.of_reference(field, info):
                case Ok(annotation):
                    yield _ReferenceField(field=field, annotation=annotation)
                case Err():
                    pass

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
        match IdAnnotation.of_reference(field, info):
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
