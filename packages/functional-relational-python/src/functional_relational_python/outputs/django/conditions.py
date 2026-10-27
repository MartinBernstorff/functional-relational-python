"""Writes Filters as Django `Q` conditions, for the queries of Views and the
constraints of Django models.

A Filter has Python's two-valued null semantics, as in the InMemoryViewEngine:
`None == x` is false and `None != x` is true, also when `x` is another field, so
`None == None` is true. SQL's three-valued comparisons get explicit `IS NULL`
terms for that; no negation is left around a comparison that may be NULL. An
enum literal is compared with its member's value, which is what Django stores for
an enum field it has a built-in rule for.

It never imports Django, and knows the Django models only by their columns, so
that the module writing them can use it.
"""

import enum
from collections.abc import Iterator, Mapping, Sequence

import pydantic

from functional_relational_python.entity_graph.entity_graph import (
    FieldName,
    Optionality,
)
from functional_relational_python.outputs.python.mappings import Expression, Indent
from functional_relational_python.outputs.python.symbols import (
    PythonString,
    TypeExpression,
    ValueExpression,
)
from functional_relational_python.pipeline import ir


class DjangoColumn(pydantic.BaseModel, frozen=True):
    """Where a Django model holds an Entity field, and whether it may be NULL."""

    attribute: FieldName
    optionality: Optionality


class DjangoColumns(pydantic.BaseModel, frozen=True):
    """The column of every field of every Entity's Django model."""

    of_entity: Mapping[type[pydantic.BaseModel], Mapping[FieldName, DjangoColumn]]

    def of(self, subject: type[pydantic.BaseModel], field: FieldName) -> DjangoColumn:
        return self.of_entity[subject][field]


class _Sense(enum.Enum):
    """Whether a condition is written for the rows a Filter keeps, or for those
    it excludes."""

    KEPT = "kept"
    EXCLUDED = "excluded"

    def flipped(self) -> _Sense:
        match self:
            case _Sense.KEPT:
                return _Sense.EXCLUDED
            case _Sense.EXCLUDED:
                return _Sense.KEPT


class NullTest(enum.Enum):
    """What an `__isnull` lookup asks of a column."""

    NULL = "null"
    NOT_NULL = "not null"

    def written(self) -> Expression:
        match self:
            case NullTest.NULL:
                return Expression("True")
            case NullTest.NOT_NULL:
                return Expression("False")


class QEntry(pydantic.BaseModel, frozen=True):
    """A keyword argument of a `Q`, e.g. `nickname__isnull=True`."""

    lookup: Expression
    value: Expression


class QLookup(pydantic.BaseModel, frozen=True):
    """`Q(<entries>)`: true where every entry is. A comparison in it is NULL in
    SQL where a column it compares is NULL, and its negation may be too, or may
    be true, depending on how Django negates it. So a negated one is only ever
    combined with explicit `IS NULL` terms that decide the rows where its
    columns are NULL: it is `and`-ed with their being not NULL, or `or`-ed with a
    term true wherever one is NULL.

    The explicit terms are needed: Django's own negation of `f == g` on two
    nullable columns is true where both are NULL, and of `f == f` where `f` is,
    but `None != None` is false."""

    entries: tuple[QEntry, ...]


class QNegated(pydantic.BaseModel, frozen=True):
    """`~Q(<entries>)`."""

    lookup: QLookup


class QAllOf(pydantic.BaseModel, frozen=True):
    """`(<a> & <b> ...)`, at least two terms."""

    terms: tuple[Q, ...]

    @staticmethod
    def of(terms: Sequence[Q]) -> Q:
        """The `and` of `terms`, nested `and`s flattened; one term is itself."""
        flat = tuple(
            inner
            for term in terms
            for inner in (term.terms if isinstance(term, QAllOf) else (term,))
        )
        return flat[0] if len(flat) == 1 else QAllOf(terms=flat)


class QAnyOf(pydantic.BaseModel, frozen=True):
    """`(<a> | <b> ...)`, at least two terms."""

    terms: tuple[Q, ...]

    @staticmethod
    def of(terms: Sequence[Q]) -> Q:
        """The `or` of `terms`, nested `or`s flattened; one term is itself."""
        flat = tuple(
            inner
            for term in terms
            for inner in (term.terms if isinstance(term, QAnyOf) else (term,))
        )
        return flat[0] if len(flat) == 1 else QAnyOf(terms=flat)


type Q = QLookup | QNegated | QAllOf | QAnyOf
"""A condition on a Django model's rows, written with `Q` objects. True in SQL
exactly where it is in Python: each NULL a comparison may give is either covered
by an explicit `IS NULL` term or ends up false."""


class ConditionWriter(pydantic.BaseModel, frozen=True):
    """Writes Filters as `Q` conditions with Python's two-valued null semantics.

    A Filter is written for the rows it keeps or for those it excludes, so that
    no negation is left around a comparison that may be NULL: `not` flips which,
    and turns an `and` into an `or` of the excluded terms, and vice versa. Each
    comparison is then written with explicit `IS NULL` terms.

    `models` is how the generated module refers to `django.db.models`, and
    `enums` how it refers to each enum a Filter's literal is a member of.
    """

    columns: DjangoColumns
    models: Expression
    enums: Mapping[type[enum.Enum], TypeExpression]

    def kept(self, kept: ir.Filter, subject: type[pydantic.BaseModel]) -> Q:
        """True for the rows of `subject`'s Django model `kept` keeps."""
        return self._condition(kept, subject, _Sense.KEPT)

    def excluded(self, kept: ir.Filter, subject: type[pydantic.BaseModel]) -> Q:
        """True for the rows of `subject`'s Django model `kept` excludes."""
        return self._condition(kept, subject, _Sense.EXCLUDED)

    def _condition(
        self, kept: ir.Filter, subject: type[pydantic.BaseModel], sense: _Sense
    ) -> Q:
        match kept:
            case ir.FilterIsNull(field=field):
                match sense:
                    case _Sense.KEPT:
                        null = NullTest.NULL
                    case _Sense.EXCLUDED:
                        null = NullTest.NOT_NULL
                return QLookup(entries=(self._is_null(subject, field, null),))
            case ir.FilterNot(term=term):
                return self._condition(term, subject, sense.flipped())
            case ir.FilterAnd(terms=terms) | ir.FilterOr(terms=terms):
                written = [self._condition(term, subject, sense) for term in terms]
                match kept, sense:
                    case (ir.FilterAnd(), _Sense.KEPT) | (
                        ir.FilterOr(),
                        _Sense.EXCLUDED,
                    ):
                        return QAllOf.of(written)
                    case _:
                        return QAnyOf.of(written)
            case ir.FilterComparison():
                return self._comparison(kept, subject, sense)

    def _comparison(
        self,
        compared: ir.FilterComparison,
        subject: type[pydantic.BaseModel],
        sense: _Sense,
    ) -> Q:
        """`field <operator> operand`. `!=` is written as `==` the other way."""
        left = compared.field
        match compared.operand:
            case ir.SourceField() as right:
                value = Expression(
                    f"{self.models.root}.F("
                    f"{PythonString(self._column(subject, right).root).literal().root})"
                )
                fields: tuple[ir.SourceField, ...] = (left, right)
            case literal:
                value = self._literal(literal)
                fields = (left,)
        column = self._column(subject, left).root
        match compared.operator:
            case ir.ComparisonOperator.EQ:
                pass
            case ir.ComparisonOperator.NE:
                sense = sense.flipped()
            case ordering:
                suffix = ConditionWriter.suffix(ordering).root
                compares = QLookup(
                    entries=(
                        QEntry(lookup=Expression(f"{column}__{suffix}"), value=value),
                    )
                )
                return self._ordering(subject, compares, fields, sense)
        equal = QLookup(entries=(QEntry(lookup=Expression(column), value=value),))
        return self._equality(subject, equal, fields, sense)

    def _ordering(
        self,
        subject: type[pydantic.BaseModel],
        compares: QLookup,
        fields: Sequence[ir.SourceField],
        sense: _Sense,
    ) -> Q:
        """`compares`, an ordering of `fields`. The compiler lets through only
        fields that are never `None`, but a column may still be NULL, e.g. for
        a RootModel of an optional type. SQL orders no NULL, so the ordering is
        false there, and its negation true, explicitly, rather than NULL or
        whatever Django's negation makes of it."""
        nullable = tuple(
            dict.fromkeys(
                field
                for field in fields
                if self._optionality(subject, field) is Optionality.OPTIONAL
            )
        )
        match sense:
            case _Sense.KEPT:
                return compares
            case _Sense.EXCLUDED if not nullable:
                return QNegated(lookup=compares)
            case _Sense.EXCLUDED:
                return QAnyOf.of(
                    [
                        QAllOf.of(
                            [
                                QNegated(lookup=compares),
                                self._nulls(subject, nullable, ()),
                            ]
                        ),
                        *(self._nulls(subject, (side,), (side,)) for side in nullable),
                    ]
                )

    def _equality(
        self,
        subject: type[pydantic.BaseModel],
        equal: QLookup,
        fields: Sequence[ir.SourceField],
        sense: _Sense,
    ) -> Q:
        """`equal`, which compares `fields` with each other or with a literal,
        with Python's semantics where a field is NULL: `None == None` is true, so
        a field also equals itself where it is NULL.

        `fields` has a field twice when it is compared with itself."""
        nullable = tuple(
            field
            for field in fields
            if self._optionality(subject, field) is Optionality.OPTIONAL
        )
        distinct = tuple(dict.fromkeys(nullable))
        every_side = len(fields) > 1 and len(nullable) == len(fields)
        match sense:
            case _Sense.KEPT:
                if every_side:
                    return QAnyOf.of([equal, self._nulls(subject, distinct, distinct)])
                return equal
            case _Sense.EXCLUDED:
                # Both values and unequal, or one side `None` and the other not.
                unequal = (
                    QAllOf.of(
                        [QNegated(lookup=equal), self._nulls(subject, distinct, ())]
                    )
                    if distinct
                    else QNegated(lookup=equal)
                )
                one_null = (
                    ()
                    if len(distinct) < len(nullable)
                    else tuple(
                        self._nulls(subject, distinct, (side,)) for side in distinct
                    )
                )
                return QAnyOf.of([unequal, *one_null])

    def _nulls(
        self,
        subject: type[pydantic.BaseModel],
        fields: Sequence[ir.SourceField],
        null: Sequence[ir.SourceField],
    ) -> QLookup:
        """True where each of `fields` is NULL if it is in `null`, else not."""
        return QLookup(
            entries=tuple(
                self._is_null(
                    subject,
                    field,
                    NullTest.NULL if field in null else NullTest.NOT_NULL,
                )
                for field in fields
            )
        )

    def _is_null(
        self, subject: type[pydantic.BaseModel], field: ir.SourceField, null: NullTest
    ) -> QEntry:
        return QEntry(
            lookup=Expression(f"{self._column(subject, field).root}__isnull"),
            value=null.written(),
        )

    def _column(
        self, subject: type[pydantic.BaseModel], field: ir.SourceField
    ) -> FieldName:
        """The attribute of `subject`'s Django model holding `field`."""
        return self.columns.of(subject, field.field).attribute

    def _optionality(
        self, subject: type[pydantic.BaseModel], field: ir.SourceField
    ) -> Optionality:
        """Whether the column of `field` may be NULL."""
        return self.columns.of(subject, field.field).optionality

    @staticmethod
    def enums_in(kept: ir.Filter) -> Iterator[type[enum.Enum]]:
        """The enums of `kept`'s literals, which the written condition refers
        to."""
        match kept:
            case ir.FilterComparison(operand=operand):
                yield from ConditionWriter._enums_in_operand(operand)
            case ir.FilterIsNull():
                pass
            case ir.FilterAnd(terms=terms) | ir.FilterOr(terms=terms):
                for term in terms:
                    yield from ConditionWriter.enums_in(term)
            case ir.FilterNot(term=term):
                yield from ConditionWriter.enums_in(term)

    @staticmethod
    def _enums_in_operand(
        operand: ir.SourceField | ir.FilterLiteral,
    ) -> Iterator[type[enum.Enum]]:
        match operand:
            case ir.SourceField() | ir.ScalarLiteral():
                pass
            case ir.EnumLiteral(member=member):
                yield type(member)
            case ir.RootModelLiteral(root=root):
                yield from ConditionWriter._enums_in_operand(root)

    @staticmethod
    def suffix(ordering: ir.ComparisonOperator) -> Expression:
        """The lookup comparing a column with `ordering`, e.g. `gt` for `>`."""
        match ordering:
            case ir.ComparisonOperator.LT:
                return Expression("lt")
            case ir.ComparisonOperator.LE:
                return Expression("lte")
            case ir.ComparisonOperator.GT:
                return Expression("gt")
            case ir.ComparisonOperator.GE:
                return Expression("gte")
            case ir.ComparisonOperator.EQ | ir.ComparisonOperator.NE:
                return Expression("exact")

    def _literal(self, literal: ir.FilterLiteral) -> Expression:
        """`literal` as Django stores it: a RootModel as its root, and an enum
        member as its value."""
        match literal:
            case ir.ScalarLiteral(root=str() as text):
                return Expression(PythonString(text).literal().root)
            case ir.ScalarLiteral(root=value):
                return Expression(repr(value))
            case ir.EnumLiteral(member=member):
                written = ValueExpression.of_enum_member(
                    self.enums[type(member)], member
                )
                return Expression(f"{written.root}.value")
            case ir.RootModelLiteral(root=root):
                return self._literal(root)

    def written(self, condition: Q, indent: Indent) -> Expression:
        """`condition` as an expression starting on a line at `indent`."""
        match condition:
            case QLookup(entries=entries):
                return indent.exploded(
                    Expression(f"{self.models.root}.Q("),
                    (
                        Expression(f"{entry.lookup.root}={entry.value.root}")
                        for entry in entries
                    ),
                    Expression(")"),
                )
            case QNegated(lookup=lookup):
                return Expression(f"~{self.written(lookup, indent).root}")
            case QAllOf(terms=terms) | QAnyOf(terms=terms):
                operator = "&" if isinstance(condition, QAllOf) else "|"
                inner = indent.deeper()
                lines = "".join(
                    f"{inner.pad().root}{'' if index == 0 else f'{operator} '}"
                    f"{self.written(term, inner).root}\n"
                    for index, term in enumerate(terms)
                )
                return Expression(f"(\n{lines}{indent.pad().root})")

    def written_negation(self, condition: Q, indent: Indent) -> Expression:
        """`~<condition>`, as an expression starting on a line at `indent`. A
        negated lookup is the lookup itself."""
        match condition:
            case QNegated(lookup=lookup):
                return self.written(lookup, indent)
            case QLookup() | QAllOf() | QAnyOf():
                return Expression(f"~{self.written(condition, indent).root}")
