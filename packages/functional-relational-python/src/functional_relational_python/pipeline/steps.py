"""Steps build frp-ir and check nothing.

A pipeline is itself a step, so user-defined steps compose exactly like built-in
ones.
"""

import functools
from collections.abc import Callable, Iterable
from typing import overload

import pydantic
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

    Only a WriteModel may retype. E.g. an update write model widens its fields with
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
    the from_model; the default only appears in the model.
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
    from_model, e.g. the literal tag of a tagged union.

    `default` is a literal, as for `default`. A View's added field needs one: its
    mapping leaves the field out, so the model fills it with its default. In a
    WriteModel it is optional.
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


def pick(fields: Iterable[ir.FieldName]) -> Step:
    """Keeps only `fields`, in the from_model's field order."""
    kept = tuple(fields)

    def step(node: ir.Node) -> ir.Node:
        return ir.Pick(input=node, fields=kept)

    return step


def where(condition: ir.Condition) -> Step:
    """Keeps only the from_model rows for which `condition` is true; see `conditions`.

    Each kept row still gives one View result. Only a View's own pipeline may
    filter, not a nested `resolve` pipeline or a WriteModel.
    """

    def step(node: ir.Node) -> ir.Node:
        return ir.Where(input=node, condition=condition)

    return step


def require(fields: FieldName | ir.Select) -> Step:
    """Narrows the field named `fields`, or every field `fields` selects, from
    `X | None` to `X`, e.g. a submitted campaign's `name`.

    A required Reference, resolved or reversed, holds its Entity rather than
    `None`. Requiring drops no row: a View result whose field is `None` is an
    unmet requirement, which the InMemoryViewEngine reports and a mapping function
    raises. Requiring a field that is not optional, or in a WriteModel, is a
    diagnostic.
    """

    def step(node: ir.Node) -> ir.Node:
        return ir.Require(input=node, fields=fields)

    return step


class Selecting:
    """Builds a Select, for a step that acts on several fields.

    >>> select(lambda field: field.name.root.endswith("_at"))
    >>> select.all()
    """

    def __call__(self, predicate: ir.FieldPredicate) -> ir.Select:
        """The fields `predicate` accepts."""
        return ir.Select(predicate=predicate)

    @staticmethod
    def all() -> ir.Select:
        """Every field."""
        return ir.Select(predicate=_every_field)


def _every_field(_field: ir.RelationField) -> bool:
    return True


select = Selecting()


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


def reverse_via(entity: type[pydantic.BaseModel], via: FieldName, *then: Step) -> Step:
    """Like `reverse`, following `entity`'s one-to-one Reference `via` backwards.

    For an `entity` with several one-to-one References to the subject.
    """
    return _reversing(entity, via, then)


def _reversing(
    entity: type[pydantic.BaseModel],
    via: FieldName | ir.SoleReference,
    then: Iterable[Step],
) -> Step:
    nested = pipeline(*then)

    def step(node: ir.Node) -> ir.Node:
        return ir.Reverse(input=node, entity=entity, via=via, then=nested)

    return step


def flatten(fields: FieldName | ir.Select) -> Step:
    """Replaces the resolved field named `fields`, or every resolved field `fields`
    selects, with the fields of the Entity it holds, in its place.

    `resolve(select(...), omit("id"))` then `flatten("department")` gives the
    Department's `name` at top level. A Select passes over the fields that are not
    resolved; naming one is a diagnostic. A lifted field may not take a name the
    relation already has: rename or omit it in the resolve's nested pipeline. A
    field lifted out of an optional Reference is optional, `None` when the
    Reference is. A lifted Reference can be resolved, and a lifted resolved field
    flattened, like any other.
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
        case ir.Field(value=ir.SubjectField(field=read) | ir.RequiredField(field=read)):
            return read == EntityNode.primary_key
        case ir.Field() | ir.NestedField():
            return False


def pipeline(*steps: Step) -> Step:
    """Applies `steps` left to right. An empty pipeline is the identity."""

    def step(node: ir.Node) -> ir.Node:
        return functools.reduce(lambda acc, next_step: next_step(acc), steps, node)

    return step


class Query(pydantic.BaseModel, frozen=True):
    """A model and the steps applied to it: what a View or WriteModel reads.

    Only a Query has a from_model. A nested pipeline, e.g. in `resolve`, starts
    from the Entity it is given, so it takes Steps.
    """

    from_model: type[pydantic.BaseModel]
    step: Step

    def then(self, *steps: Step) -> Query:
        """This query followed by `steps`, e.g. a shared filter extended by one View.

        >>> active = from_(Company, where(...))
        >>> active.then(omit(FieldName("secret")))
        """
        return Query(from_model=self.from_model, step=pipeline(self.step, *steps))

    def node(self) -> ir.Node:
        """The node tree the steps build over every field of `from_model`."""
        return self.step(ir.FromModel(from_model=self.from_model))


def from_(model: type[pydantic.BaseModel], *steps: Step) -> Query:
    """A query over every field of `model`, followed by `steps` left to right.

    `model` is any Pydantic model, often an Entity. A View over a model that is not
    an Entity cannot be run by the InMemoryViewEngine.
    """
    return Query(from_model=model, step=pipeline(*steps))
