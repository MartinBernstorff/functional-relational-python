"""Conditions for `steps.where`. Like steps, they build frp-ir and check nothing.

>>> conditions.and_(
...     conditions.eq(conditions.field(FieldName("status")), Status.ACTIVE),
...     conditions.not_(conditions.is_null(conditions.field(FieldName("name")))),
... )

A field reference names a top-level field by its name at that point in the
pipeline. A literal is an instance of the field's type, or `None` for an optional
field.
"""

from functional_relational_python.entity_graph.entity_graph import FieldName
from functional_relational_python.pipeline import ir

type Operand = ir.FieldRef | ir.UncheckedLiteral
"""What a field is compared with: another field, or a literal."""


def field(name: FieldName) -> ir.FieldRef:
    return ir.FieldRef(name=name)


def _compare(
    operator: ir.ComparisonOperator, left: ir.FieldRef, right: Operand
) -> ir.Comparison:
    match right:
        case ir.FieldRef():
            operand = right
        case _:
            operand = ir.ConditionLiteral(value=right)
    return ir.Comparison(operator=operator, field=left, operand=operand)


def eq(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    """`left == right`. `None == x` is false."""
    return _compare(ir.ComparisonOperator.EQ, left, right)


def ne(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    """`left != right`. `None != x` is true."""
    return _compare(ir.ComparisonOperator.NE, left, right)


def lt(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    """`left < right`. Neither may be optional."""
    return _compare(ir.ComparisonOperator.LT, left, right)


def le(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    return _compare(ir.ComparisonOperator.LE, left, right)


def gt(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    return _compare(ir.ComparisonOperator.GT, left, right)


def ge(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    return _compare(ir.ComparisonOperator.GE, left, right)


def is_null(of: ir.FieldRef) -> ir.IsNull:
    return ir.IsNull(field=of)


def and_(first: ir.Condition, *rest: ir.Condition) -> ir.And:
    return ir.And(terms=(first, *rest))


def or_(first: ir.Condition, *rest: ir.Condition) -> ir.Or:
    return ir.Or(terms=(first, *rest))


def not_(term: ir.Condition) -> ir.Not:
    return ir.Not(term=term)
