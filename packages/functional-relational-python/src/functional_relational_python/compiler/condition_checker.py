"""Checks a `where` step's Condition against the relation at that point, giving
the Filter backends run."""

import types
import typing
from typing import Annotated

import pydantic

from functional_relational_python.compiler import diagnostics, literals, narrowing
from functional_relational_python.entity_graph.entity_graph import (
    Optionality,
    RuntimeAnnotation,
    SplitAnnotation,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.constraints import ConditionOwner


class Defects(pydantic.BaseModel, frozen=True):
    """Why a Condition, or a part of one, is defective: at least one diagnostic."""

    diagnostics: Annotated[
        tuple[diagnostics.CompileDiagnostic, ...], pydantic.Field(min_length=1)
    ]

    @staticmethod
    def among(
        checked: tuple[Checked, ...],
    ) -> tuple[diagnostics.CompileDiagnostic, ...]:
        """Every diagnostic of the Defects in `checked`, in order."""
        return tuple(
            diagnostic
            for part in checked
            if isinstance(part, Defects)
            for diagnostic in part.diagnostics
        )


class _Column(pydantic.BaseModel, frozen=True):
    """A field a Condition names, the subject's own field that is its value, and
    that field's type as the subject declares it.

    A Condition is checked against the declared type, not the one an earlier
    `where` or `require` narrowed the View's field to: it tests the subject's
    field, whatever the View makes of it.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    field: ir.Field
    source: ir.SourceField
    annotation: RuntimeAnnotation

    def split_annotation(self) -> SplitAnnotation:
        """The declared type, apart from `None`, with `type` aliases looked
        through."""
        return narrowing.split(self.annotation)


type Checked = (
    ir.Filter
    | ir.FilterLiteral
    | ir.SourceField
    | ir.ConditionLiteral
    | _Column
    | Defects
)
"""A checked Condition or part of one, or why it is defective."""


class ConditionChecker(pydantic.BaseModel, frozen=True):
    """Checks a Condition against `relation`, the relation at its `where`.

    Every defective term is reported, not only the first. `owner` is what the
    Condition belongs to: a View's or Payload's `where`, or a constraint, whose
    relation is its Entity's.
    """

    owner: ConditionOwner
    relation: ir.Relation
    subject: type[pydantic.BaseModel]

    def check(self, condition: ir.Condition) -> ir.Filter | Defects:
        checked: ir.Filter | Defects
        match condition:
            case ir.Comparison():
                checked = self._check_comparison(condition)
            case ir.IsNull(field=reference):
                match self._column(reference):
                    case _Column(source=source):
                        checked = ir.FilterIsNull(field=source)
                    case Defects() as defects:
                        checked = defects
            case ir.And(terms=terms):
                checked = self._check_connected(terms, ir.FilterAnd)
            case ir.Or(terms=terms):
                checked = self._check_connected(terms, ir.FilterOr)
            case ir.Not(term=term):
                match self.check(term):
                    case Defects() as defects:
                        checked = defects
                    case negated:
                        checked = ir.FilterNot(term=negated)
        return checked

    def _check_connected(
        self,
        terms: tuple[ir.Condition, ...],
        connective: type[ir.FilterAnd] | type[ir.FilterOr],
    ) -> ir.Filter | Defects:
        """The terms joined by `connective`, or the only term."""
        checked = tuple(self.check(term) for term in terms)
        if reported := Defects.among(checked):
            return Defects(diagnostics=reported)
        filters = tuple(term for term in checked if not isinstance(term, Defects))
        match filters:
            case (only,):
                return only
            case _:
                return connective(terms=filters)

    def _column(self, reference: ir.FieldRef) -> _Column | Defects:
        """The field `reference` names, if its value is the subject's own field."""
        named = [
            field for field in self.relation.fields if field.name == reference.name
        ]
        match named:
            case [
                ir.Field(
                    value=ir.SourceField(field=column) | ir.RequiredField(field=column)
                ) as field,
                *_,
            ]:
                declared = self.subject.model_fields.get(column.root)
                return _Column(
                    field=field,
                    source=ir.SourceField(field=column),
                    # A field recovered from a defective step may name no field
                    # of the subject; it keeps the type it was given.
                    annotation=field.annotation()
                    if declared is None
                    else declared.annotation,
                )
            case [_, *_]:
                diagnostic = diagnostics.NotASourceColumn(
                    owner=self.owner, source=self.relation.source, field=reference.name
                )
            case _:
                diagnostic = diagnostics.UnknownField(
                    owner=self.owner,
                    source=self.relation.source,
                    field=reference.name,
                    valid=self.relation.field_names(),
                )
        return Defects(diagnostics=(diagnostic,))

    def _check_comparison(self, comparison: ir.Comparison) -> ir.Filter | Defects:
        left = self._column(comparison.field)
        match comparison.operand:
            case ir.FieldRef() as reference:
                right = self._column(reference)
            case ir.ConditionLiteral() as literal:
                right = literal
        match left, right:
            case _Column(), ir.ConditionLiteral(value=None):
                return self._check_null_test(left, comparison.operator)
            case _Column(), _Column() | ir.ConditionLiteral():
                pass
            case _:
                return Defects(diagnostics=Defects.among((left, right)))
        operator = comparison.operator
        operand = self._check_operand(left, right)
        reported = (
            *self._ordering_defects(left, operator),
            *(
                self._ordering_defects(right, operator)
                if isinstance(right, _Column)
                else ()
            ),
            *Defects.among((operand,)),
        )
        if reported or isinstance(operand, Defects):
            return Defects(diagnostics=reported)
        return ir.FilterComparison(
            operator=operator, field=left.source, operand=operand
        )

    def _check_null_test(
        self, column: _Column, operator: ir.ComparisonOperator
    ) -> ir.Filter | Defects:
        """`== None` as is-null, `!= None` as its negation. Only an optional field
        may be `None`, and `None` has no order."""
        ordering = self._ordering_defects(column, operator)
        match column.split_annotation().optionality:
            case Optionality.REQUIRED:
                return Defects(
                    diagnostics=(*ordering, self._literal_mismatch(column, None))
                )
            case Optionality.OPTIONAL:
                pass
        is_null = ir.FilterIsNull(field=column.source)
        match operator:
            case ir.ComparisonOperator.EQ:
                return is_null
            case ir.ComparisonOperator.NE:
                return ir.FilterNot(term=is_null)
            case (
                ir.ComparisonOperator.LT
                | ir.ComparisonOperator.LE
                | ir.ComparisonOperator.GT
                | ir.ComparisonOperator.GE
            ):
                # An ordering of an optional field always has a diagnostic.
                return Defects(diagnostics=ordering)

    def _check_operand(
        self, left: _Column, right: _Column | ir.ConditionLiteral
    ) -> ir.SourceField | ir.FilterLiteral | Defects:
        """What `left` is compared with: a field of its type, or a literal
        generated code can write out, which is an instance of its type."""
        match right:
            case _Column():
                if (
                    left.split_annotation().required
                    == right.split_annotation().required
                ):
                    return right.source
                return Defects(
                    diagnostics=(
                        diagnostics.ComparedTypesDiffer(
                            owner=self.owner,
                            source=self.relation.source,
                            field=left.field.name,
                            annotation=left.annotation,
                            other=right.field.name,
                            other_annotation=right.annotation,
                        ),
                    )
                )
            case ir.ConditionLiteral(value=value):
                pass
        match ConditionChecker._written(value):
            case literals.NotALiteral():
                return Defects(
                    diagnostics=(
                        diagnostics.NonLiteralConditionValue(
                            owner=self.owner, value=diagnostics.LiteralText.of(value)
                        ),
                    )
                )
            case written:
                pass
        if not ConditionChecker._admits(left.annotation, value):
            return Defects(diagnostics=(self._literal_mismatch(left, value),))
        return written

    def _literal_mismatch(
        self, column: _Column, value: ir.UncheckedLiteral
    ) -> diagnostics.LiteralTypeMismatch:
        return diagnostics.LiteralTypeMismatch(
            owner=self.owner,
            source=self.relation.source,
            field=column.field.name,
            annotation=column.annotation,
            value=diagnostics.LiteralText.of(value),
        )

    def _ordering_defects(
        self, column: _Column, operator: ir.ComparisonOperator
    ) -> tuple[diagnostics.CompileDiagnostic, ...]:
        """For an ordering `operator`, a diagnostic if `column` may be `None`, and
        one if its type does not define `operator`."""
        match operator:
            case ir.ComparisonOperator.EQ | ir.ComparisonOperator.NE:
                return ()
            case ir.ComparisonOperator.LT:
                method = "__lt__"
            case ir.ComparisonOperator.LE:
                method = "__le__"
            case ir.ComparisonOperator.GT:
                method = "__gt__"
            case ir.ComparisonOperator.GE:
                method = "__ge__"
        split = column.split_annotation()
        reported: list[diagnostics.CompileDiagnostic] = []
        match split.optionality:
            case Optionality.REQUIRED:
                pass
            case Optionality.OPTIONAL:
                reported.append(
                    diagnostics.OrderingOnOptionalField(
                        owner=self.owner,
                        source=self.relation.source,
                        field=column.field.name,
                        operator=operator,
                    )
                )
        required = split.required
        # Every class inherits `object`'s method, which orders nothing.
        if not isinstance(required, type) or getattr(required, method) is getattr(
            object, method
        ):
            reported.append(
                diagnostics.UnorderedFieldType(
                    owner=self.owner,
                    source=self.relation.source,
                    field=column.field.name,
                    annotation=required,
                    operator=operator,
                )
            )
        return tuple(reported)

    @staticmethod
    def _written(
        value: ir.UncheckedLiteral,
    ) -> ir.FilterLiteral | literals.NotALiteral:
        """`value` as a literal generated code can write out: a scalar or enum
        member as for a default, or a RootModel of one."""
        match value:
            case pydantic.RootModel(root=root):
                # A RootModel's root has the type of its parameter, unknown here.
                inner: ir.UncheckedLiteral = root
                match ConditionChecker._written(inner):
                    case literals.NotALiteral() as not_a_literal:
                        return not_a_literal
                    case written:
                        return ir.RootModelLiteral(instance=value, root=written)
            case _:
                return literals.classify_scalar(value)

    @staticmethod
    def _admits(annotation: RuntimeAnnotation, value: ir.UncheckedLiteral) -> bool:
        """Whether `value` is an instance of `annotation`.

        An int is accepted for a float, as Python's numeric tower has it, but a
        bool is not accepted for an int.
        """
        origin = typing.get_origin(annotation)
        arguments = typing.get_args(annotation)
        if isinstance(annotation, typing.TypeAliasType):
            admitted = ConditionChecker._admits(annotation.__value__, value)
        elif origin is typing.Union or origin is types.UnionType:
            admitted = any(ConditionChecker._admits(arg, value) for arg in arguments)
        elif origin is typing.Annotated:
            admitted = ConditionChecker._admits(arguments[0], value)
        elif origin is typing.Literal:
            admitted = any(
                type(arg) is type(value) and arg == value for arg in arguments
            )
        elif origin is not None or not isinstance(annotation, type):
            admitted = False
        elif isinstance(value, bool):
            admitted = issubclass(annotation, bool)
        elif annotation is float and type(value) is int:
            admitted = True
        else:
            admitted = isinstance(value, annotation)
        return admitted
