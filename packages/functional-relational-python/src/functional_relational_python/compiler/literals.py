"""Which values generated code can write out as literals, shared by defaults and
Conditions."""

import enum
import math

import pydantic

from functional_relational_python.pipeline import ir


class NotALiteral(pydantic.BaseModel, frozen=True):
    """A value generated code cannot write out."""


def classify_scalar(
    value: ir.UncheckedLiteral,
) -> ir.ScalarLiteral | ir.EnumLiteral | NotALiteral:
    """`value` as a boolean, a finite number, a string or a named enum member.

    Only the exact built-in types count, so a subclass that might `repr` as
    something else is not a literal. Neither is a float without a literal, such as
    `nan`, nor a combination of Flag members, which has no name to write out.
    """
    match value:
        case enum.Enum() if type(value).__members__.get(value.name) is value:
            return ir.EnumLiteral(member=value)
        case bool() | int() | str() if type(value) in (bool, int, str):
            return ir.ScalarLiteral(value)
        case float() if type(value) is float and math.isfinite(value):
            return ir.ScalarLiteral(value)
        case _:
            return NotALiteral()
