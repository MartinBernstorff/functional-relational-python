"""Narrows the types of a View's fields: by the Filters its rows pass, and by
`require`.

A Filter refers to the subject's own fields, so narrowing by it narrows every
field whose value is the field it tests, under whatever name.
"""

import enum
import keyword
import typing

from functional_relational_python.compiler import diagnostics
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
    RuntimeAnnotation,
    SplitAnnotation,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.payloads import DefinitionName


def split(annotation: RuntimeAnnotation) -> SplitAnnotation:
    """`annotation`, apart from `None`, with `type` aliases looked through, of
    the whole annotation and of the rest."""
    whole = Optionality.of_aliased_annotation(annotation)
    required = whole.required
    while isinstance(required, typing.TypeAliasType):
        required = required.__value__
    return SplitAnnotation(
        optionality=whole.optionality, required=required, metadata=whole.metadata
    )


def without_none(annotation: RuntimeAnnotation) -> RuntimeAnnotation:
    """`annotation` without `None`, keeping the metadata of an `Annotated` inside
    the union, e.g. `PastAwareDatetime | None` to `PastAwareDatetime`."""
    split_off = Optionality.of_aliased_annotation(annotation)
    if not split_off.metadata:
        return split_off.required
    return typing.cast("_SpecialForm", typing.Annotated)[
        (split_off.required, *split_off.metadata)
    ]


class _SpecialForm(typing.Protocol):
    """A special form such as `Literal`, subscripted at run time."""

    def __getitem__(self, parameters: object, /) -> RuntimeAnnotation: ...


def _literal(member: enum.Enum) -> RuntimeAnnotation:
    """`Literal[member]`. A type checker takes only a member written out in a
    `Literal`, so it is subscripted as the object it is at run time."""
    return typing.cast("_SpecialForm", typing.Literal)[member]


def nameable(member: enum.Enum) -> bool:
    """Whether `member` can be written in a `Literal`: as `<Enum>.<name>`."""
    return member.name.isidentifier() and not keyword.iskeyword(member.name)


def by_filter(
    relation: ir.Relation, kept: ir.Filter, entity_graph: EntityGraph
) -> ir.Relation:
    """`relation` with its fields narrowed by `kept`, a Filter every row passes.

    Only the terms `kept` `and`s together at its top narrow: `== <enum member>`
    to `Literal[<member>]`, `is None` to `Literal[None]`, and `is not None` drops
    `None`. An `or`, or any other `not`, narrows nothing.

    A member whose name is no identifier cannot be written in a `Literal`, so it
    narrows nothing. Nor does `is None` narrow a field that is not optional, or a
    Reference, which stays one, so that it can still be resolved.
    """
    terms = _conjuncts(kept)
    return relation.with_fields(
        tuple(_narrowed(field, terms, entity_graph) for field in relation.fields)
    )


def _conjuncts(kept: ir.Filter) -> tuple[ir.Filter, ...]:
    """The terms `kept` `and`s together, with nested `and`s flattened."""
    match kept:
        case ir.FilterAnd(terms=terms):
            return tuple(term for and_ed in terms for term in _conjuncts(and_ed))
        case ir.FilterComparison() | ir.FilterIsNull() | ir.FilterOr() | ir.FilterNot():
            return (kept,)


def _narrowed(
    field: ir.RelationField, terms: tuple[ir.Filter, ...], entity_graph: EntityGraph
) -> ir.RelationField:
    """`field` narrowed by every term in `terms` that tests its value."""
    match field:
        case ir.Field(
            value=ir.SourceField(field=column) | ir.RequiredField(field=column)
        ):
            pass
        case ir.Field() | ir.NestedField():
            return field
    annotation = field.annotation()
    # A Reference stays one, so that it can still be resolved.
    narrowing = (
        tuple(term for term in terms if not isinstance(term, ir.FilterIsNull))
        if ir.References().selects(field, entity_graph)
        else terms
    )
    for term in narrowing:
        annotation = _narrowed_by(term, column, annotation)
    if annotation == field.annotation():
        return field
    return field.narrowed(annotation)


def _narrowed_by(
    term: ir.Filter, column: FieldName, annotation: RuntimeAnnotation
) -> RuntimeAnnotation:
    """`annotation`, the type of a field whose value is `column`, narrowed by
    `term`."""
    match term:
        case ir.FilterComparison(
            operator=ir.ComparisonOperator.EQ,
            field=ir.SourceField(field=tested),
            operand=ir.EnumLiteral(member=member),
        ) if tested == column and nameable(member):
            return _literal(member)
        case ir.FilterIsNull(field=ir.SourceField(field=tested)) if (
            tested == column and split(annotation).optionality is Optionality.OPTIONAL
        ):
            return typing.Literal[None]
        case ir.FilterNot(term=ir.FilterIsNull(field=ir.SourceField(field=tested))) if (
            tested == column
        ):
            return without_none(annotation)
        case _:
            return annotation


def required(
    view: DefinitionName, source: ir.SourceName, field: ir.RelationField
) -> ir.RelationField | diagnostics.CompileDiagnostic:
    """`field` narrowed from `X | None` to `X`, with a value the View requires; or
    why it cannot be."""
    match field:
        case ir.Field():
            annotation = field.annotation()
        case ir.NestedField(value=lookup):
            annotation = lookup.annotation()
    narrowed = split(annotation)
    if narrowed.required == typing.Literal[None]:
        return diagnostics.RequireOnAlwaysNoneField(
            view=view, source=source, field=field.name
        )
    match narrowed.optionality:
        case Optionality.REQUIRED:
            return diagnostics.RequireOnRequiredField(
                view=view, source=source, field=field.name, annotation=annotation
            )
        case Optionality.OPTIONAL:
            pass
    match field:
        case ir.NestedField(value=lookup):
            return field.model_copy(update={"value": lookup.required()})
        case ir.Field(value=value):
            pass
    value_required: ir.Value
    match value:
        case ir.SourceField(field=column) | ir.RequiredField(field=column):
            value_required = ir.RequiredField(field=column)
            narrowed_type = without_none(annotation)
        case ir.Lifted():
            value_required = value.required()
            narrowed_type = without_none(annotation)
        case ir.Lookup() | ir.ReverseLookup() | ir.ChainedLookup():
            value_required = value.required()
            narrowed_type = value_required.annotation()
        case ir.Added():
            return diagnostics.RequireAddedField(view=view, field=field.name)
    return field.narrowed(narrowed_type).model_copy(update={"value": value_required})
