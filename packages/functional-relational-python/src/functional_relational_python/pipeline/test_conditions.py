from collections.abc import Callable

import pytest

from functional_relational_python.entity_graph.entity_graph import FieldName
from functional_relational_python.pipeline import conditions, ir

status = conditions.field(FieldName("status"))
name = conditions.field(FieldName("name"))
other = conditions.field(FieldName("other"))


@pytest.mark.parametrize(
    ("method", "function", "operator"),
    [
        (ir.FieldRef.eq, conditions.eq, ir.ComparisonOperator.EQ),
        (ir.FieldRef.ne, conditions.ne, ir.ComparisonOperator.NE),
        (ir.FieldRef.lt, conditions.lt, ir.ComparisonOperator.LT),
        (ir.FieldRef.le, conditions.le, ir.ComparisonOperator.LE),
        (ir.FieldRef.gt, conditions.gt, ir.ComparisonOperator.GT),
        (ir.FieldRef.ge, conditions.ge, ir.ComparisonOperator.GE),
    ],
    ids=["eq", "ne", "lt", "le", "gt", "ge"],
)
@pytest.mark.parametrize(
    ("operand", "compared"),
    [
        (other, other),
        ("active", ir.ConditionLiteral(value="active")),
    ],
    ids=["field", "literal"],
)
def test_a_comparison_method_builds_what_its_function_does(
    method: Callable[[ir.FieldRef, conditions.Operand], ir.Comparison],
    function: Callable[[ir.FieldRef, conditions.Operand], ir.Comparison],
    operator: ir.ComparisonOperator,
    operand: ir.FieldRef | str,
    compared: ir.FieldRef | ir.ConditionLiteral,
) -> None:
    expected = ir.Comparison(operator=operator, field=status, operand=compared)

    assert method(status, operand) == expected
    assert function(status, operand) == expected


def test_is_null_method_builds_what_its_function_does() -> None:
    assert name.is_null() == conditions.is_null(name) == ir.IsNull(field=name)


def test_operators_combine_conditions_as_the_functions_do() -> None:
    active, unnamed = status.eq("active"), name.is_null()

    assert active & unnamed == conditions.and_(active, unnamed)
    assert active | unnamed == conditions.or_(active, unnamed)
    assert ~unnamed == conditions.not_(unnamed)
    assert (active & ~unnamed) | unnamed == conditions.or_(
        conditions.and_(active, conditions.not_(unnamed)), unnamed
    )
