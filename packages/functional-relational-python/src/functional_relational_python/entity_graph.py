import types
import typing
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, RootModel
from pydantic.fields import FieldInfo
from safe_result import Err, Ok, Result

from functional_relational_python.id import Id, IdType


class FieldName(RootModel[str]):
    model_config = ConfigDict(frozen=True)

    @staticmethod
    def fake() -> FieldName:
        return FieldName("department_id")


class DiagnosticMessage(RootModel[str]):
    model_config = ConfigDict(frozen=True)


class Optionality(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class Reference(BaseModel):
    """A field typed with an `Id` subclass, resolved to the Entity it points to."""

    model_config = ConfigDict(frozen=True)

    field: FieldName
    id_type: IdType
    target: type[BaseModel]
    optionality: Optionality


class EntityNode(BaseModel):
    """An Entity in the EntityGraph: its model, primary key type and References."""

    model_config = ConfigDict(frozen=True)

    model: type[BaseModel]
    id_type: IdType
    references: tuple[Reference, ...]


class DuplicatePrimaryKey(BaseModel):
    """An ID type is the primary key of several Entities."""

    model_config = ConfigDict(frozen=True)

    id_type: IdType
    entities: frozenset[type[BaseModel]]

    def describe(self) -> DiagnosticMessage:
        names = ", ".join(sorted(entity.__qualname__ for entity in self.entities))
        return DiagnosticMessage(
            f"{self.id_type.__qualname__} is the primary key of several entities: {names}"
        )


class UnresolvedReference(BaseModel):
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


class NotAnEntity(BaseModel):
    """A model given as an Entity has no `id` field typed with an `Id` subclass."""

    model_config = ConfigDict(frozen=True)

    model: type[BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.model.__qualname__} is not an entity: its `id` field must be "
            f"typed with an Id subclass"
        )


type EntityGraphDiagnostic = DuplicatePrimaryKey | UnresolvedReference | NotAnEntity


class EntityGraphError(Exception):
    """Every problem found while building an EntityGraph."""

    def __init__(self, diagnostics: frozenset[EntityGraphDiagnostic]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "\n".join(sorted(diagnostic.describe().root for diagnostic in diagnostics))
        )


class _ReferenceField(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: FieldName
    id_type: IdType
    optionality: Optionality


class EntityGraph(BaseModel):
    """All Entities, with every Reference resolved to the Entity it points to."""

    model_config = ConfigDict(frozen=True)

    entities: Mapping[type[BaseModel], EntityNode]

    @staticmethod
    def build(
        entities: Iterable[type[BaseModel]],
    ) -> Result[EntityGraph, EntityGraphError]:
        """Build the graph, or return every diagnostic at once.

        The order of `entities` does not affect the result.
        """
        diagnostics: set[EntityGraphDiagnostic] = set()
        keyed_by: defaultdict[IdType, set[type[BaseModel]]] = defaultdict(set)
        for model in set(entities):
            match model.model_fields.get("id"):
                case FieldInfo(annotation=type() as id_type) if issubclass(id_type, Id):
                    keyed_by[id_type].add(model)
                case _:
                    diagnostics.add(NotAnEntity(model=model))

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
                    if reference.id_type in owners:
                        references.append(
                            Reference(
                                field=reference.field,
                                id_type=reference.id_type,
                                target=owners[reference.id_type],
                                optionality=reference.optionality,
                            )
                        )
                    elif reference.id_type not in keyed_by:
                        # An ID type keying several Entities is already a diagnostic,
                        # so References to it are not reported again.
                        diagnostics.add(
                            UnresolvedReference(
                                entity=model,
                                field=reference.field,
                                id_type=reference.id_type,
                            )
                        )
                nodes[model] = EntityNode(
                    model=model, id_type=id_type, references=tuple(references)
                )

        if diagnostics:
            return Err(EntityGraphError(frozenset(diagnostics)))
        return Ok(EntityGraph(entities=nodes))

    @staticmethod
    def _reference_fields(model: type[BaseModel]) -> Iterator[_ReferenceField]:
        for name, field in model.model_fields.items():
            if name == "id":
                continue
            annotation = field.annotation
            match annotation:
                case type() as id_type if issubclass(id_type, Id):
                    yield _ReferenceField(
                        field=FieldName(name),
                        id_type=id_type,
                        optionality=Optionality.REQUIRED,
                    )
                case _:
                    pass
            match (typing.get_origin(annotation), typing.get_args(annotation)):
                case (
                    types.UnionType,
                    (type() as id_type, types.NoneType)
                    | (types.NoneType, type() as id_type),
                ) if issubclass(id_type, Id):
                    yield _ReferenceField(
                        field=FieldName(name),
                        id_type=id_type,
                        optionality=Optionality.OPTIONAL,
                    )
                case _:
                    pass
