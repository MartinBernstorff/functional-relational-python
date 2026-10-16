"""Compiles Views to checked frp-ir.

Checking uses the poisoned pattern: a failed check reports a diagnostic and
recovers with the best relation it can build, so later steps and other Views are
still checked and independent defects are all reported, once each.
"""

from collections.abc import Iterable
from typing import Annotated

import pydantic
from pydantic.fields import FieldInfo
from safe_result import Err, Ok, Result

from functional_relational_python import diagnostics, ir
from functional_relational_python.entity_graph import EntityGraph
from functional_relational_python.views import View, ViewName

type Diagnostics = Annotated[
    tuple[diagnostics.Diagnostic, ...], pydantic.Field(min_length=1)
]


class Poisoned(pydantic.BaseModel, frozen=True):
    """A node whose check reported at least one diagnostic.

    `recovered` is the relation later steps are checked against. A Poisoned node
    cannot exist without the diagnostics that poisoned it.
    """

    recovered: ir.Relation
    diagnostics: Diagnostics


class CompiledView(pydantic.BaseModel, frozen=True):
    name: ViewName
    node: ir.Node
    relation: ir.Relation


class PoisonedView(pydantic.BaseModel, frozen=True):
    name: ViewName
    diagnostics: Diagnostics


class CompiledViews(pydantic.BaseModel, frozen=True):
    views: tuple[CompiledView, ...]


class CompileError(Exception):
    def __init__(self, diagnostics: tuple[diagnostics.Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "\n".join(diagnostic.describe().root for diagnostic in diagnostics)
        )


def compile_views(
    _entity_graph: EntityGraph, views: Iterable[View], /
) -> Result[CompiledViews, CompileError]:
    """Compiles every View, or returns every diagnostic from all of them.

    The EntityGraph is unused until References are resolved (MB-181).
    """
    checked = check(views)
    poisoned = [view for view in checked if isinstance(view, PoisonedView)]
    if poisoned:
        return Err(
            CompileError(
                tuple(
                    diagnostic for view in poisoned for diagnostic in view.diagnostics
                )
            )
        )
    return Ok(
        CompiledViews(
            views=tuple(view for view in checked if isinstance(view, CompiledView))
        )
    )


def check(views: Iterable[View]) -> tuple[CompiledView | PoisonedView, ...]:
    """Checks every View independently, so a defect in one never hides another."""
    return tuple(_check_view(view) for view in views)


def _check_view(view: View) -> CompiledView | PoisonedView:
    node = view.step(ir.FromSource(source=view.source))
    match _check_node(view.name, node):
        case ir.Relation() as relation:
            return CompiledView(name=view.name, node=node, relation=relation)
        case Poisoned(diagnostics=reported):
            return PoisonedView(name=view.name, diagnostics=reported)


def _check_node(view: ViewName, node: ir.Node) -> ir.Relation | Poisoned:
    match node:
        case ir.FromSource():
            return ir.Relation.of(node)
        case ir.Omit() | ir.Rename() | ir.Project():
            match _check_node(view, node.input):
                case ir.Relation() as relation:
                    return _check_step(view, node, relation)
                case Poisoned(recovered=recovered, diagnostics=earlier):
                    return _poison(_check_step(view, node, recovered), earlier)


def _check_step(
    view: ViewName, node: ir.Omit | ir.Rename | ir.Project, relation: ir.Relation
) -> ir.Relation | Poisoned:
    """Checks one step against the relation its input has."""
    match node:
        case ir.Omit():
            return _check_omit(view, node, relation)
        case ir.Rename():
            return _check_rename(view, node, relation)
        case ir.Project():
            return _check_project(view, node, relation)


def _check_omit(
    view: ViewName, node: ir.Omit, relation: ir.Relation
) -> ir.Relation | Poisoned:
    if node.field in relation.field_names():
        return relation.without(node.field)
    # Omitting a field that is not there leaves the relation unchanged, so that is
    # an exact recovery.
    return Poisoned(
        recovered=relation,
        diagnostics=(
            diagnostics.UnknownField(
                view=view,
                source=relation.source,
                field=node.field,
                valid=relation.field_names(),
            ),
        ),
    )


def _check_rename(
    view: ViewName, node: ir.Rename, relation: ir.Relation
) -> ir.Relation | Poisoned:
    names = relation.field_names()
    reported: list[diagnostics.Diagnostic] = []
    if node.field not in names:
        reported.append(
            diagnostics.UnknownField(
                view=view, source=relation.source, field=node.field, valid=names
            )
        )
    if node.new in names:
        reported.append(
            diagnostics.RenameOntoExistingField(
                view=view, renamed=node.field, field=node.new
            )
        )
    if not reported:
        return relation.renamed(node.field, node.new)
    # Later steps may use the new name, so the recovered relation has it either
    # way: the existing field if there is one, else a field of unknown type.
    recovered = (
        relation.without(node.field)
        if node.new in names
        else ir.Relation(
            source=relation.source,
            fields=(*relation.fields, ir.Field(name=node.new, info=FieldInfo())),
        )
    )
    return Poisoned(recovered=recovered, diagnostics=tuple(reported))


def _check_project(
    view: ViewName, node: ir.Project, relation: ir.Relation
) -> ir.Relation | Poisoned:
    kept = relation.keeping(node.fields)
    unknown = tuple(
        diagnostics.UnknownField(
            view=view,
            source=relation.source,
            field=field,
            valid=relation.field_names(),
        )
        for field in node.fields
        if field not in relation.field_names()
    )
    if not unknown:
        return kept
    # The known fields are kept, so later steps are checked against them.
    return Poisoned(recovered=kept, diagnostics=unknown)


def _poison(
    checked: ir.Relation | Poisoned, earlier: tuple[diagnostics.Diagnostic, ...]
) -> Poisoned:
    """Carries diagnostics from a poisoned input onto the node built from it."""
    match checked:
        case ir.Relation():
            return Poisoned(recovered=checked, diagnostics=earlier)
        case Poisoned():
            return Poisoned(
                recovered=checked.recovered,
                diagnostics=(*earlier, *checked.diagnostics),
            )
