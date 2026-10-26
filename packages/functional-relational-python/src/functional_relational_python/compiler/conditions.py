"""Checks a `where` step's Condition against the relation at that point, giving
the Filter backends run."""

import enum
import math
import types
import typing
from collections.abc import Callable

import pydantic

from functional_relational_python.compiler import diagnostics
from functional_relational_python.entity_graph.entity_graph import (
    Optionality,
    RuntimeAnnotation,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.payloads import DefinitionName


class Defective(pydantic.BaseModel, frozen=True):
    """A Condition, or a part of one, that a diagnostic was reported for."""


class _Column(pydantic.BaseModel, frozen=True):
    """A field a Condition names, and the subject's own field that is its value."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    field: ir.Field
    source: ir.SourceField


class _NoneLiteral(pydantic.BaseModel, frozen=True):
    """A comparison with `None`, which is checked into an is-null test."""


class ConditionChecker:
    """Checks the Conditions of one `where` against `relation`, collecting a
    diagnostic for every defect, so each defective term is reported."""

    def __init__(self, view: DefinitionName, relation: ir.Relation) -> None:
        self.view = view
        self.relation = relation
        self.diagnostics: list[diagnostics.CompileDiagnostic] = []

    def check(self, condition: ir.Condition) -> ir.Filter | Defective:
        match condition:
            case ir.Comparison():
                return self._comparison(condition)
            case ir.IsNull(field=reference):
                match self._column(reference):
                    case _Column(source=source):
                        return ir.FilterIsNull(field=source)
                    case Defective() as defective:
                        return defective
            case ir.And(terms=terms):
                return self._combined(
                    terms, lambda checked: ir.FilterAnd(terms=checked)
                )
            case ir.Or(terms=terms):
                return self._combined(terms, lambda checked: ir.FilterOr(terms=checked))
            case ir.Not(term=term):
                return self._combined(
                    (term,), lambda checked: ir.FilterNot(term=checked[0])
                )

    def _combined(
        self,
        terms: tuple[ir.Condition, ...],
        combine: Callable[[tuple[ir.Filter, ...]], ir.Filter],
    ) -> ir.Filter | Defective:
        """`combine` of every term checked, or Defective if any term is. Each term
        is checked either way, so each defect is reported."""
        checked = [self.check(term) for term in terms]
        filters = tuple(term for term in checked if not isinstance(term, Defective))
        if len(filters) < len(checked):
            return Defective()
        return combine(filters)

    def _column(self, reference: ir.FieldRef) -> _Column | Defective:
        """The field `reference` names, if its value is the subject's own field."""
        named = [
            field for field in self.relation.fields if field.name == reference.name
        ]
        match named:
            case [ir.Field(value=ir.SourceField() as source) as field, *_]:
                return _Column(field=field, source=source)
            case [_, *_]:
                self.diagnostics.append(
                    diagnostics.NotASourceColumn(
                        view=self.view,
                        source=self.relation.source,
                        field=reference.name,
                    )
                )
            case _:
                self.diagnostics.append(
                    diagnostics.UnknownField(
                        view=self.view,
                        source=self.relation.source,
                        field=reference.name,
                        valid=self.relation.field_names(),
                    )
                )
        return Defective()

    def _comparison(self, comparison: ir.Comparison) -> ir.Filter | Defective:
        left = self._column(comparison.field)
        match comparison.operand:
            case ir.FieldRef() as reference:
                right = self._column(reference)
            case ir.ConditionLiteral() as literal:
                right = literal
        match left, right:
            case _Column(), _Column() | ir.ConditionLiteral():
                pass
            case _:
                return Defective()
        reported = len(self.diagnostics)
        operator = comparison.operator
        if operator.is_ordering():
            self._check_ordered(left, operator)
            if isinstance(right, _Column):
                self._check_ordered(right, operator)
        operand: ir.SourceField | ir.FilterLiteral | _NoneLiteral | Defective
        match right:
            case _Column():
                self._check_same_type(left, right)
                operand = right.source
            case ir.ConditionLiteral(value=None):
                operand = self._none(left)
            case ir.ConditionLiteral(value=value):
                operand = self._literal(left, value)
        if isinstance(operand, Defective) or len(self.diagnostics) > reported:
            return Defective()
        match operand:
            case _NoneLiteral():
                return ConditionChecker._null_test(left, operator)
            case _:
                return ir.FilterComparison(
                    operator=operator, field=left.source, operand=operand
                )

    def _check_same_type(self, left: _Column, right: _Column) -> None:
        left_type = Optionality.of_annotation(left.field.annotation()).required
        right_type = Optionality.of_annotation(right.field.annotation()).required
        if left_type != right_type:
            self.diagnostics.append(
                diagnostics.ComparedTypesDiffer(
                    view=self.view,
                    source=self.relation.source,
                    field=left.field.name,
                    annotation=left.field.annotation(),
                    other=right.field.name,
                    other_annotation=right.field.annotation(),
                )
            )

    def _none(self, column: _Column) -> _NoneLiteral | Defective:
        """`None`, if `column` is optional."""
        match column.field.optionality():
            case Optionality.REQUIRED:
                self.diagnostics.append(self._mismatch(column.field, None))
                return Defective()
            case Optionality.OPTIONAL:
                return _NoneLiteral()

    @staticmethod
    def _null_test(column: _Column, operator: ir.ComparisonOperator) -> ir.Filter:
        """`== None` as is-null, `!= None` as its negation."""
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
                # Ordering an optional field is reported, so this is not reached.
                raise AssertionError(operator)

    def _literal(
        self, column: _Column, value: ir.UncheckedLiteral
    ) -> ir.FilterLiteral | Defective:
        match ConditionChecker._written(value):
            case Defective():
                self.diagnostics.append(
                    diagnostics.NonLiteralConditionValue(
                        view=self.view, value=diagnostics.LiteralText.of(value)
                    )
                )
                return Defective()
            case literal:
                pass
        if not ConditionChecker._admits(column.field.annotation(), value):
            self.diagnostics.append(self._mismatch(column.field, value))
            return Defective()
        return literal

    def _mismatch(
        self, field: ir.Field, value: ir.UncheckedLiteral
    ) -> diagnostics.LiteralTypeMismatch:
        return diagnostics.LiteralTypeMismatch(
            view=self.view,
            source=self.relation.source,
            field=field.name,
            annotation=field.annotation(),
            value=diagnostics.LiteralText.of(value),
        )

    def _check_ordered(self, column: _Column, operator: ir.ComparisonOperator) -> None:
        """A diagnostic if `column` may be `None`, or its type does not define
        `operator`."""
        field = column.field
        split = Optionality.of_annotation(field.annotation())
        match split.optionality:
            case Optionality.REQUIRED:
                pass
            case Optionality.OPTIONAL:
                self.diagnostics.append(
                    diagnostics.OrderingOnOptionalField(
                        view=self.view,
                        source=self.relation.source,
                        field=field.name,
                        operator=operator,
                    )
                )
        match operator:
            case ir.ComparisonOperator.LT:
                method = "__lt__"
            case ir.ComparisonOperator.LE:
                method = "__le__"
            case ir.ComparisonOperator.GT:
                method = "__gt__"
            case ir.ComparisonOperator.GE:
                method = "__ge__"
            case ir.ComparisonOperator.EQ | ir.ComparisonOperator.NE:
                return
        required = split.required
        # Every class inherits `object`'s method, which orders nothing.
        if not isinstance(required, type) or getattr(required, method) is getattr(
            object, method
        ):
            self.diagnostics.append(
                diagnostics.UnorderedFieldType(
                    view=self.view,
                    source=self.relation.source,
                    field=field.name,
                    annotation=required,
                    operator=operator,
                )
            )

    @staticmethod
    def _written(value: ir.UncheckedLiteral) -> ir.FilterLiteral | Defective:
        """`value` as a literal generated code can write out, if it is one.

        As for a default, only the exact built-in types count, and an enum member
        must have a name.
        """
        match value:
            case enum.Enum() if type(value).__members__.get(value.name) is value:
                return ir.EnumLiteral(member=value)
            case bool() | int() | str() if type(value) in (bool, int, str):
                return ir.ScalarLiteral(value)
            case float() if type(value) is float and math.isfinite(value):
                return ir.ScalarLiteral(value)
            case pydantic.RootModel(root=root):
                # A RootModel's root has the type of its parameter, unknown here.
                inner: ir.UncheckedLiteral = root
                match ConditionChecker._written(inner):
                    case Defective() as defective:
                        return defective
                    case written:
                        return ir.RootModelLiteral(instance=value, root=written)
            case _:
                return Defective()

    @staticmethod
    def _admits(annotation: RuntimeAnnotation, value: ir.UncheckedLiteral) -> bool:
        """Whether `value` is an instance of `annotation`."""
        if isinstance(annotation, typing.TypeAliasType):
            return ConditionChecker._admits(annotation.__value__, value)
        origin = typing.get_origin(annotation)
        arguments = typing.get_args(annotation)
        if origin is typing.Union or origin is types.UnionType:
            return any(ConditionChecker._admits(arg, value) for arg in arguments)
        if origin is typing.Annotated:
            return ConditionChecker._admits(arguments[0], value)
        if origin is typing.Literal:
            return any(type(arg) is type(value) and arg == value for arg in arguments)
        return (
            origin is None
            and isinstance(annotation, type)
            and isinstance(value, annotation)
        )
