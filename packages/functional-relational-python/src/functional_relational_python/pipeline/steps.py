"""Steps build frp-ir and check nothing.

A pipeline is itself a step, so user-defined steps compose exactly like built-in
ones.
"""

import functools
from collections.abc import Callable, Iterable
from typing import overload

from safe_result import Ok

from functional_relational_python.entity_graph.entity_graph import (
    EntityNode,
    FieldName,
    IdAnnotation,
)
from functional_relational_python.pipeline import ir

type Step = Callable[[ir.Node], ir.Node]


def omit(fields: FieldName | ir.Select) -> Step:
    """Removes the field named `fields`, or every field `fields` selects."""

    def step(node: ir.Node) -> ir.Node:
        return ir.Omit(input=node, fields=fields)

    return step


@overload
def rename(fields: FieldName, new: FieldName) -> Step: ...


@overload
def rename(fields: ir.Select, new: ir.FieldRenaming) -> Step: ...


def rename(fields: FieldName | ir.Select, new: FieldName | ir.FieldRenaming) -> Step:
    """Renames the field named `fields` to `new`.

    With a Select, renames every selected field to `new` of its old name.
    """
    # The overloads keep a Select with a name, and a name with a function, out;
    # this is what they would mean.
    match new:
        case FieldName() as name:

            def renaming(_old: FieldName) -> FieldName:
                return name

        case _:
            renaming = new

    def step(node: ir.Node) -> ir.Node:
        match fields:
            case FieldName():
                return ir.Rename(input=node, field=fields, new=renaming(fields))
            case ir.Select():
                return ir.RenameSelected(input=node, fields=fields, new=renaming)

    return step


def retype(fields: FieldName | ir.Select, transform: ir.TypeTransform) -> Step:
    """Gives the field named `fields`, or every field `fields` selects, the type
    `transform` maps its type to.

    Only a Payload may retype. E.g. an update payload widens its fields with
    `retype(select(...), lambda annotation: annotation | MISSING)`, then defaults
    them to `MISSING`.
    """

    def step(node: ir.Node) -> ir.Node:
        return ir.Retype(input=node, fields=fields, transform=transform)

    return step


def retype_to(fields: FieldName | ir.Select, annotation: ir.RuntimeAnnotation) -> Step:
    """Gives the field named `fields`, or every field `fields` selects, the type
    `annotation`."""

    def transform(_old: ir.RuntimeAnnotation) -> ir.RuntimeAnnotation:
        return annotation

    return retype(fields, transform)


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


def is_primary_key(field: ir.RelationField) -> bool:
    """Whether `field` is the primary key, under whatever name it has been renamed to.

    Like `is_reference`, it is told apart by where its value comes from: the
    subject's `id` field. A Reference renamed to `id` is not the primary key.
    """
    match field:
        case ir.Field(value=ir.SourceField(field=source)):
            return source == EntityNode.primary_key
        case ir.Field() | ir.NestedField():
            return False


def hydrate(*then: Step) -> Step:
    """Resolves every Reference: `resolve(select(is_reference), *then)`."""
    return resolve(select(is_reference), *then)


def pipeline(*steps: Step) -> Step:
    """Applies `steps` left to right. An empty pipeline is the identity."""

    def step(node: ir.Node) -> ir.Node:
        return functools.reduce(lambda acc, next_step: next_step(acc), steps, node)

    return step
