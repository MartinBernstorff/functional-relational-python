"""Steps build frp-ir and check nothing.

A pipeline is itself a step, so user-defined steps compose exactly like built-in
ones.
"""

import functools
import operator
from collections.abc import Callable, Iterable, Sequence
from typing import overload

import pydantic
from pydantic import MISSING
from pydantic.fields import FieldInfo

from functional_relational_python.entity_graph.entity_graph import (
    EntityNode,
    FieldName,
)
from functional_relational_python.pipeline import ir, selectors

type Step = Callable[[ir.Node], ir.Node]

type Name = str
"""A field's name."""

type Fields = Name | Sequence[Name] | ir.Selector
"""The fields a step acts on: one name, several names, or a Selector (see
`selectors`). Naming a field the relation lacks is a diagnostic."""


def _selection(fields: Fields) -> ir.FieldSelection:
    if isinstance(fields, str):
        return ir.Named(names=(FieldName(fields),))
    match fields:
        case ir.Select() | ir.References():
            return fields
        case _:
            return ir.Named(names=tuple(FieldName(name) for name in fields))


def omit(fields: Fields) -> Step:
    """Removes `fields`."""
    selection = _selection(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.Omit(input=node, fields=selection)

    return step


def pick(fields: Fields) -> Step:
    """Keeps only `fields`, in the Source's field order."""
    selection = _selection(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.Pick(input=node, fields=selection)

    return step


@overload
def rename(fields: Name, new: Name) -> Step: ...


@overload
def rename(fields: Fields, new: ir.FieldRenaming) -> Step: ...


def rename(fields: Fields, new: Name | ir.FieldRenaming) -> Step:
    """Renames the field named `fields` to `new`.

    With several fields, renames each to `new` of its old name.
    """
    # The overloads keep several fields with a name, and a name with a function,
    # out; this is what they would mean.
    match new:
        case str():
            name = FieldName(new)

            def renaming(_old: FieldName) -> FieldName:
                return name

        case _:
            renaming = new

    def step(node: ir.Node) -> ir.Node:
        if isinstance(fields, str):
            field = FieldName(fields)
            return ir.Rename(input=node, field=field, new=renaming(field))
        return ir.RenameSelected(input=node, fields=_selection(fields), new=renaming)

    return step


def map_type(fields: Fields, transform: ir.TypeTransform) -> Step:
    """Gives each of `fields` the type `transform` maps its type to.

    Only a Payload may map types. E.g. an update payload makes its fields
    `partial`.
    """
    selection = _selection(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.MapType(input=node, fields=selection, transform=transform)

    return step


def set_type(fields: Fields, annotation: ir.RuntimeAnnotation) -> Step:
    """Gives each of `fields` the type `annotation`."""

    def transform(_old: ir.RuntimeAnnotation) -> ir.RuntimeAnnotation:
        return annotation

    return map_type(fields, transform)


def widen(fields: Fields, variant: ir.RuntimeAnnotation) -> Step:
    """Gives each of `fields` its type or `variant`, e.g. `X | None` for `X`."""

    def transform(old: ir.RuntimeAnnotation) -> ir.RuntimeAnnotation:
        return operator.or_(old, variant)

    return map_type(fields, transform)


def default(fields: Fields, value: ir.UncheckedDefault) -> Step:
    """Sets the default of each of `fields` in the generated model.

    `value` is a literal: `None`, `MISSING`, a number, a string, a boolean, an
    empty list or dict, or an enum member. A mapping still fills every field from
    the Source; the default only appears in the model.
    """
    selection = _selection(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.Default(input=node, fields=selection, value=value)

    return step


def partial(fields: Fields) -> Step:
    """Lets each of `fields` be left out: widens it with `MISSING`, its default.

    Only a Payload may be partial, e.g. an update payload.
    """
    return pipeline(widen(fields, MISSING), default(fields, MISSING))


@overload
def add(name: Name, annotation: ir.AddedType) -> Step: ...


@overload
def add(name: Name, annotation: ir.AddedType, default: ir.UncheckedDefault) -> Step: ...


def add(
    name: Name,
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
            name=FieldName(name),
            info=FieldInfo.from_annotation(annotation),
            default=given,
        )

    return step


def where(condition: ir.Condition) -> Step:
    """Keeps only the Source rows for which `condition` is true; see `conditions`.

    Each kept row still gives one View result. Only a View's own pipeline may
    filter, not a nested `resolve` pipeline or a Payload.
    """

    def step(node: ir.Node) -> ir.Node:
        return ir.Where(input=node, condition=condition)

    return step


def non_nullable(fields: Fields) -> Step:
    """Narrows each of `fields` from `X | None` to `X`, e.g. a submitted
    campaign's `name`.

    A non-nullable Reference, resolved or reversed, holds its Entity rather than
    `None`. Narrowing drops no row: a View result whose field is `None` is an
    unmet requirement, which the InMemoryViewEngine reports and a mapping function
    raises. Narrowing a field that is not optional, or in a Payload, is a
    diagnostic.
    """
    selection = _selection(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.NonNullable(input=node, fields=selection)

    return step


def resolve(fields: Fields, *then: Step) -> Step:
    """Replaces each Reference in `fields` with the Entity it points to.

    A `department_id` field becomes `department`. The steps in `then` form a
    pipeline applied to each referenced Entity; without them, the field holds the
    Entity itself.
    """
    selection = _selection(fields)
    nested = pipeline(*then)

    def step(node: ir.Node) -> ir.Node:
        return ir.Resolve(input=node, fields=selection, then=nested)

    return step


def reverse(entity: type[pydantic.BaseModel], *then: Step) -> Step:
    """Adds a field holding the `entity` that references the subject, or `None`
    when none does.

    `entity` must have exactly one one-to-one Reference (see `OneToOneMarker`) to the
    subject; with several, choose one with `reverse_via`. The field is named after
    `entity` in snake_case, e.g. `meta_campaign` for `MetaCampaign`. The steps in
    `then` form a pipeline applied to `entity`; without them, the field holds the
    Entity itself.
    """
    return _reversing(entity, ir.SoleReference(), then)


def reverse_via(entity: type[pydantic.BaseModel], via: Name, *then: Step) -> Step:
    """Like `reverse`, following `entity`'s one-to-one Reference `via` backwards.

    For an `entity` with several one-to-one References to the subject.
    """
    return _reversing(entity, FieldName(via), then)


def _reversing(
    entity: type[pydantic.BaseModel],
    via: FieldName | ir.SoleReference,
    then: Iterable[Step],
) -> Step:
    nested = pipeline(*then)

    def step(node: ir.Node) -> ir.Node:
        return ir.Reverse(input=node, entity=entity, via=via, then=nested)

    return step


def flatten(fields: Fields) -> Step:
    """Replaces each resolved field in `fields` with the fields of the Entity it
    holds, in its place.

    `resolve("department_id", omit("id"))` then `flatten("department")` gives the
    Department's `name` at top level. A Selector passes over the fields that are
    not resolved; naming one is a diagnostic. A lifted field may not take a name the
    relation already has: rename or omit it in the resolve's nested pipeline. A
    field lifted out of an optional Reference is optional, `None` when the
    Reference is. A lifted Reference can be resolved, and a lifted resolved field
    flattened, like any other.
    """

    selection = _selection(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.Flatten(input=node, fields=selection)

    return step


def is_primary_key(field: ir.RelationField) -> bool:
    """Whether `field` is the primary key, under whatever name it has been renamed to.

    Like a Reference, it is told apart by where its value comes from: the
    subject's `id` field. A Reference renamed to `id` is not the primary key.
    """
    match field:
        case ir.Field(
            value=ir.SourceField(field=source) | ir.RequiredField(field=source)
        ):
            return source == EntityNode.primary_key
        case ir.Field() | ir.NestedField():
            return False


def hydrate(*then: Step) -> Step:
    """Resolves every Reference: `resolve(selectors.references(), *then)`."""
    return resolve(selectors.references(), *then)


def pipeline(*steps: Step) -> Step:
    """Applies `steps` left to right. An empty pipeline is the identity."""

    def step(node: ir.Node) -> ir.Node:
        return functools.reduce(lambda acc, next_step: next_step(acc), steps, node)

    return step
