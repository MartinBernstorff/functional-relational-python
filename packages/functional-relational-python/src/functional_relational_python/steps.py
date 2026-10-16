"""Steps build frp-ir and check nothing.

A pipeline is itself a step, so user-defined steps compose exactly like built-in
ones.
"""

import functools
from collections.abc import Callable, Iterable

from functional_relational_python import ir

type Step = Callable[[ir.Node], ir.Node]


def omit(field: ir.FieldName) -> Step:
    def step(node: ir.Node) -> ir.Node:
        return ir.Omit(input=node, field=field)

    return step


def project(fields: Iterable[ir.FieldName]) -> Step:
    """Keeps only `fields`, in the Source's field order."""
    kept = tuple(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.Project(input=node, fields=kept)

    return step


def pipeline(*steps: Step) -> Step:
    """Applies `steps` left to right. An empty pipeline is the identity."""

    def step(node: ir.Node) -> ir.Node:
        return functools.reduce(lambda acc, next_step: next_step(acc), steps, node)

    return step
