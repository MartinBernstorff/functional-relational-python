"""Steps build frp-ir and check nothing.

A pipeline is itself a step, so user-defined steps compose exactly like built-in
ones.
"""

import functools
from collections.abc import Callable, Iterable
from typing import overload

from pydantic.fields import FieldInfo

from functional_relational_python.entity_graph.entity_graph import (
    EntityNode,
    FieldName,
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


def default(fields: FieldName | ir.Select, value: ir.UncheckedDefault) -> Step:
    """Sets the default of the field named `fields`, or of every field `fields`
    selects, in the generated model.

    `value` is a literal: `None`, `MISSING`, a number, a string, a boolean, an
    empty list or dict, or an enum member. A mapping still fills every field from
    the Source; the default only appears in the model.
    """

    def step(node: ir.Node) -> ir.Node:
        return ir.Default(input=node, fields=fields, value=value)

    return step


@overload
def add(name: FieldName, annotation: ir.AddedType) -> Step: ...


@overload
def add(
    name: FieldName, annotation: ir.AddedType, default: ir.UncheckedDefault
) -> Step: ...


def add(
    name: FieldName,
    annotation: ir.AddedType,
    default: ir.UncheckedDefault = ir.NoDefault,
) -> Step:
    """Declares a new field `name`, typed `annotation`, with no value from the
    Source, e.g. the literal tag of a tagged union.

    `default` is a literal, as for `default`. A View's added field needs one: its
    mapping leaves the field out, so the model fills it with its default. In a
    Payload it is optional.
    """
    # The class stands for "no default given", which the overloads spell out;
    # unlike an instance, it calls nothing in the signature.
    given = ir.NoDefault() if default is ir.NoDefault else default

    def step(node: ir.Node) -> ir.Node:
        return ir.Add(
            input=node,
            name=name,
            info=FieldInfo.from_annotation(annotation),
            default=given,
        )

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


def references() -> ir.References:
    """Every Reference, for `resolve`: fields typed `X` or `X | None` for an ID `X`.

    Not a predicate for `select`, since which types are IDs depends on the
    catalog's `IdBase`.
    """
    return ir.References()


def resolve(fields: ir.Select | ir.References, *then: Step) -> Step:
    """Replaces each Reference in `fields` with the Entity it points to.

    A `department_id` field becomes `department`. The steps in `then` form a
    pipeline applied to each referenced Entity; without them, the field holds the
    Entity itself.
    """
    nested = pipeline(*then)

    def step(node: ir.Node) -> ir.Node:
        return ir.Resolve(input=node, fields=fields, then=nested)

    return step


def flatten(fields: FieldName | ir.Select) -> Step:
    """Replaces the resolved field named `fields`, or every resolved field `fields`
    selects, with the fields of the Entity it holds, in its place.

    `resolve(select(...), omit("id"))` then `flatten("department")` gives the
    Department's `name` at top level. A lifted field may not take a name the
    relation already has: rename or omit it in the resolve's nested pipeline. A
    field lifted out of an optional Reference is optional, `None` when the
    Reference is.
    """

    def step(node: ir.Node) -> ir.Node:
        return ir.Flatten(input=node, fields=fields)

    return step


def is_primary_key(field: ir.RelationField) -> bool:
    """Whether `field` is the primary key, under whatever name it has been renamed to.

    Like a Reference, it is told apart by where its value comes from: the
    subject's `id` field. A Reference renamed to `id` is not the primary key.
    """
    match field:
        case ir.Field(value=ir.SourceField(field=source)):
            return source == EntityNode.primary_key
        case ir.Field() | ir.NestedField():
            return False


def hydrate(*then: Step) -> Step:
    """Resolves every Reference: `resolve(references(), *then)`."""
    return resolve(references(), *then)


def pipeline(*steps: Step) -> Step:
    """Applies `steps` left to right. An empty pipeline is the identity."""

    def step(node: ir.Node) -> ir.Node:
        return functools.reduce(lambda acc, next_step: next_step(acc), steps, node)

    return step
