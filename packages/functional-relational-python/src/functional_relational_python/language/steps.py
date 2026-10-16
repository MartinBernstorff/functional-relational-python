"""Steps build frp-ir and check nothing.

A pipeline is itself a step, so user-defined steps compose exactly like built-in
ones.
"""

import functools
from collections.abc import Callable, Iterable

from safe_result import Ok

from functional_relational_python.domain.entity_graph import IdAnnotation
from functional_relational_python.language import ir

type Step = Callable[[ir.Node], ir.Node]


def omit(field: ir.FieldName) -> Step:
    def step(node: ir.Node) -> ir.Node:
        return ir.Omit(input=node, field=field)

    return step


def rename(field: ir.FieldName, new: ir.FieldName) -> Step:
    def step(node: ir.Node) -> ir.Node:
        return ir.Rename(input=node, field=field, new=new)

    return step


def project(fields: Iterable[ir.FieldName]) -> Step:
    """Keeps only `fields`, in the Source's field order."""
    kept = tuple(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.Project(input=node, fields=kept)

    return step


def select(predicate: ir.FieldPredicate) -> ir.Select:
    """The fields `predicate` accepts, for a step that acts on several fields."""
    return ir.Select(predicate=predicate)


def is_reference(field: ir.RelationField) -> bool:
    """Whether `field` is a Reference: typed `X` or `X | None` for an Id `X`.

    The primary key is not a Reference, under whatever name it has been renamed to.
    It is told apart by where its value comes from, the subject's primary key.
    """
    match field:
        case ir.Field(info=info, value=ir.SourceField(field=source)):
            return isinstance(IdAnnotation.of_reference(source, info), Ok)
        case ir.Field() | ir.NestedField():
            return False


def resolve(fields: ir.Select, *then: Step) -> Step:
    """Replaces each Reference in `fields` with the Entity it points to.

    A `department_id` field becomes `department`. The steps in `then` form a
    pipeline applied to each referenced Entity; without them, the field holds the
    Entity itself.
    """
    nested = pipeline(*then)

    def step(node: ir.Node) -> ir.Node:
        return ir.Resolve(input=node, fields=fields, then=nested)

    return step


def hydrate(*then: Step) -> Step:
    """Resolves every Reference: `resolve(select(is_reference), *then)`."""
    return resolve(select(is_reference), *then)


def pipeline(*steps: Step) -> Step:
    """Applies `steps` left to right. An empty pipeline is the identity."""

    def step(node: ir.Node) -> ir.Node:
        return functools.reduce(lambda acc, next_step: next_step(acc), steps, node)

    return step
