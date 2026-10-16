"""frp-ir: the relational expression a View's pipeline builds over its Source.

Steps only build nodes; nothing here is checked. The compiler checks a node tree
and derives its Relation (the fields a View result will have).
"""

import pydantic
from pydantic.fields import FieldInfo

from functional_relational_python.entity_graph import FieldName


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


type Node = FromSource | Omit | Rename


class Field(pydantic.BaseModel, frozen=True):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: FieldName
    info: FieldInfo


class Relation(pydantic.BaseModel, frozen=True):
    """The checked shape of a node: the Source it starts from, and its fields."""

    source: SourceName
    fields: tuple[Field, ...]

    @staticmethod
    def of(node: FromSource) -> Relation:
        return Relation(
            source=node.source_name(),
            fields=tuple(
                Field(name=FieldName(name), info=info)
                for name, info in node.source.model_fields.items()
            ),
        )

    def field_names(self) -> tuple[FieldName, ...]:
        return tuple(field.name for field in self.fields)

    def renamed(self, old: FieldName, new: FieldName) -> Relation:
        """Renames `old` to `new` in place, keeping its type."""
        return Relation(
            source=self.source,
            fields=tuple(
                Field(name=new, info=field.info) if field.name == old else field
                for field in self.fields
            ),
        )

    def without(self, name: FieldName) -> Relation:
        return Relation(
            source=self.source,
            fields=tuple(field for field in self.fields if field.name != name),
        )
