"""Constraints on the stored rows of an Entity, which the database enforces.

A Check holds of every row, a Unique says no two rows share values. Both use the
Conditions of `where`, over the Entity's own fields:

>>> Check(
...     name=ConstraintName("campaign_template_is_never_submitted"),
...     condition=conditions.or_(
...         conditions.ne(conditions.field(FieldName("kind")), Kind.TEMPLATE),
...         conditions.is_null(conditions.field(FieldName("submitted_at"))),
...     ),
... )
>>> Unique(
...     name=ConstraintName("frame_parameter_unique_identifier"),
...     fields=(FieldName("frame_id"), FieldName("identifier")),
...     where=conditions.is_null(conditions.field(FieldName("deleted_at"))),
... )
"""

from typing import Annotated

import pydantic

from functional_relational_python.entity_graph.entity_graph import FieldName
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.payloads import DefinitionName


class ConstraintName(pydantic.RootModel[str], frozen=True):
    """The name of a constraint in the database, e.g.
    `campaign_lifetime_budget_gt`. It is a Python identifier, and unique among
    the constraints of every Entity."""

    @pydantic.field_validator("root")
    @classmethod
    def _identifier(cls, root: str) -> str:
        if not root.isidentifier():
            msg = f"'{root}' is not an identifier"
            raise ValueError(msg)
        return root

    @staticmethod
    def fake() -> ConstraintName:
        return ConstraintName("company_name_is_set")


class Check(pydantic.BaseModel, frozen=True):
    """Every row satisfies `condition`, with the semantics it has in `where`:
    `None == x` is false, so a row whose optional field is `None` violates a
    comparison of it unless the condition allows `None`."""

    name: ConstraintName
    condition: ir.Condition


class EveryRow(pydantic.BaseModel, frozen=True):
    """A Unique over every row, not only those a Condition keeps."""


class Unique(pydantic.BaseModel, frozen=True):
    """No two rows `where` keeps have equal values in all of `fields`.

    As in SQL, a row with a `None` in any of `fields` equals no other row.
    """

    name: ConstraintName
    fields: Annotated[tuple[FieldName, ...], pydantic.Field(min_length=1)]
    where: ir.Condition | EveryRow

    @pydantic.field_validator("fields")
    @classmethod
    def _distinct(cls, fields: tuple[FieldName, ...]) -> tuple[FieldName, ...]:
        repeated = sorted({field.root for field in fields if fields.count(field) > 1})
        if repeated:
            msg = f"fields are named more than once: {', '.join(repeated)}"
            raise ValueError(msg)
        return fields


type Constraint = Check | Unique


type ConditionOwner = DefinitionName | ConstraintName
"""What a Condition belongs to: the View or Payload whose `where` it is, or the
constraint it states."""
