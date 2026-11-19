"""Conditions for `steps.where`. Like steps, they build frp-ir and check nothing.

A Condition is built with the methods of a field reference, and combined with
`&`, `|` and `~`:

>>> status, name = field(FieldName("status")), field(FieldName("name"))
>>> status.eq(Status.ACTIVE) & ~name.is_null()

The functions here build the same Conditions:

>>> and_(eq(status, Status.ACTIVE), not_(is_null(name)))

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


def eq(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    """`left == right`. `None == x` is false."""
    return left.eq(right)


def ne(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    """`left != right`. `None != x` is true."""
    return left.ne(right)


def lt(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    """`left < right`. Neither may be optional."""
    return left.lt(right)


def le(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    return left.le(right)


def gt(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    return left.gt(right)


def ge(left: ir.FieldRef, right: Operand) -> ir.Comparison:
    return left.ge(right)


def is_null(of: ir.FieldRef) -> ir.IsNull:
    return of.is_null()


def and_(first: ir.Condition, *rest: ir.Condition) -> ir.And:
    return ir.And(terms=(first, *rest))


def or_(first: ir.Condition, *rest: ir.Condition) -> ir.Or:
    return ir.Or(terms=(first, *rest))


def not_(term: ir.Condition) -> ir.Not:
    return ir.Not(term=term)
