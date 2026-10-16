"""Steps build frp-ir and check nothing.

A pipeline is itself a step, so user-defined steps compose exactly like built-in
ones.
"""

import functools
from collections.abc import Callable

from functional_relational_python import ir

type Step = Callable[[ir.Node], ir.Node]


def omit(field: ir.FieldName) -> Step:
    def step(node: ir.Node) -> ir.Node:
        return ir.Omit(input=node, field=field)

    return step


def rename(field: ir.FieldName, new: ir.FieldName) -> Step:
    def step(node: ir.Node) -> ir.Node:
        return ir.Rename(input=node, field=field, new=new)

    return step


def pipeline(*steps: Step) -> Step:
    """Applies `steps` left to right. An empty pipeline is the identity."""

    def step(node: ir.Node) -> ir.Node:
        return functools.reduce(lambda acc, next_step: next_step(acc), steps, node)

    return step
