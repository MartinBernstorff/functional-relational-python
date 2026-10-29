"""Selectors choose the fields a step acts on by what they are, rather than by name.

Selecting no field is no defect, unlike naming a field the relation lacks.
"""

from functional_relational_python.entity_graph.entity_graph import Optionality
from functional_relational_python.pipeline import ir


def all_fields() -> ir.Select:
    """Every field."""

    def predicate(_field: ir.RelationField) -> bool:
        return True

    return ir.Select(predicate=predicate)


def optional() -> ir.Select:
    """Every field whose type admits `None`."""

    def predicate(field: ir.RelationField) -> bool:
        return field.optionality() is Optionality.OPTIONAL

    return ir.Select(predicate=predicate)


def matching(predicate: ir.FieldPredicate) -> ir.Select:
    """The fields `predicate` accepts."""
    return ir.Select(predicate=predicate)


def references() -> ir.References:
    """Every Reference: fields typed `X` or `X | None` for an ID `X`.

    Not a predicate for `matching`, since which types are IDs depends on the
    catalog's `IdBase`.
    """
    return ir.References()
