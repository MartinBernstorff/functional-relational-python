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

from functional_relational_python.compiler import diagnostics
from functional_relational_python.domain.entity_graph import (
    EntityGraph,
    FieldName,
    IdAnnotation,
    Optionality,
)
from functional_relational_python.language import ir
from functional_relational_python.language.views import View, ViewName
from functional_relational_python.support.diagnostic import DiagnosticsError
from functional_relational_python.support.identifier import Identifier

type Diagnostics = Annotated[
    tuple[diagnostics.CompileDiagnostic, ...], pydantic.Field(min_length=1)
]


class Poisoned(pydantic.BaseModel, frozen=True):
    """A node whose check reported at least one diagnostic.

    `recovered` is the relation later steps are checked against. A Poisoned node
    cannot exist without the diagnostics that poisoned it.
    """

    recovered: ir.Relation
    diagnostics: Diagnostics


class CompiledView(pydantic.BaseModel, frozen=True):
    """A checked View: its Source, and the Relation backends compute from it."""

    name: ViewName
    source: type[pydantic.BaseModel]
    relation: ir.Relation


class PoisonedView(pydantic.BaseModel, frozen=True):
    name: ViewName
    diagnostics: Diagnostics


class CompiledViews(pydantic.BaseModel, frozen=True):
    views: tuple[CompiledView, ...]


def compile_views(
    entity_graph: EntityGraph, views: Iterable[View], /
) -> Result[CompiledViews, DiagnosticsError[diagnostics.CompileDiagnostic]]:
    """Compiles every View, or returns every diagnostic from all of them.

    `entity_graph` is what References are resolved against.
    """
    checked = check(entity_graph, views)
    return DiagnosticsError.or_ok(
        (
            diagnostic
            for view in checked
            if isinstance(view, PoisonedView)
            for diagnostic in view.diagnostics
        ),
        lambda: CompiledViews(
            views=tuple(view for view in checked if isinstance(view, CompiledView))
        ),
    )


def check(
    entity_graph: EntityGraph, views: Iterable[View]
) -> tuple[CompiledView | PoisonedView, ...]:
    """Checks every View independently, so a defect in one never hides another."""
    return tuple(_check_view(entity_graph, view) for view in views)


def _check_view(entity_graph: EntityGraph, view: View) -> CompiledView | PoisonedView:
    node = view.step(ir.FromSource(source=view.source))
    match _check_node(entity_graph, view.name, node):
        case ir.Relation() as relation:
            return CompiledView(name=view.name, source=view.source, relation=relation)
        case Poisoned(diagnostics=reported):
            return PoisonedView(name=view.name, diagnostics=reported)


def _check_node(
    entity_graph: EntityGraph, view: ViewName, node: ir.Node
) -> ir.Relation | Poisoned:
    match node:
        case ir.FromSource():
            return ir.Relation.of(node)
        case ir.Omit() | ir.Rename() | ir.Project() | ir.Resolve():
            match _check_node(entity_graph, view, node.input):
                case ir.Relation() as relation:
                    return _check_step(entity_graph, view, node, relation)
                case Poisoned(recovered=recovered, diagnostics=earlier):
                    return _poison(
                        _check_step(entity_graph, view, node, recovered), earlier
                    )


def _check_step(
    entity_graph: EntityGraph,
    view: ViewName,
    node: ir.Omit | ir.Rename | ir.Project | ir.Resolve,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Checks one step against the relation its input has."""
    match node:
        case ir.Omit():
            return _check_omit(view, node, relation)
        case ir.Rename():
            return _check_rename(view, node, relation)
        case ir.Project():
            return _check_project(view, node, relation)
        case ir.Resolve():
            return _check_resolve(entity_graph, view, node, relation)


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
    reported: list[diagnostics.CompileDiagnostic] = []
    if node.field not in names:
        reported.append(
            diagnostics.UnknownField(
                view=view, source=relation.source, field=node.field, valid=names
            )
        )
    # Renaming a field to its own name changes nothing, so it is no defect.
    if node.new in names and node.new != node.field:
        reported.append(
            diagnostics.RenameOntoExistingField(
                view=view, renamed=node.field, field=node.new
            )
        )
    reported.extend(_field_name_diagnostics(view, relation, node.new))
    if not reported:
        return relation.renamed(node.field, node.new)
    return Poisoned(
        recovered=_recovered_rename(node, relation), diagnostics=tuple(reported)
    )


def _recovered_rename(node: ir.Rename, relation: ir.Relation) -> ir.Relation:
    """The relation after a defective rename.

    Later steps may use the new name, so it has a field of that name: the existing
    field if there is one, else the renamed field, else a field of unknown type.
    """
    names = relation.field_names()
    if node.new in names:
        return relation.without(node.field)
    if node.field in names:
        return relation.renamed(node.field, node.new)
    return ir.Relation(
        source=relation.source,
        fields=(
            *relation.fields,
            ir.Field(
                name=node.new, info=FieldInfo(), value=ir.SourceField(field=node.field)
            ),
        ),
    )


def _field_name_diagnostics(
    view: ViewName, relation: ir.Relation, name: FieldName
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    """Reports `name` unless a View model can have a field of that name."""
    if (
        isinstance(Identifier.of(name), Ok)
        and not name.root.startswith("_")
        and not hasattr(pydantic.BaseModel, name.root)
    ):
        return ()
    return (
        diagnostics.InvalidFieldName(view=view, source=relation.source, field=name),
    )


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


class _CheckedField(pydantic.BaseModel, frozen=True):
    """One field after `resolve`, with the diagnostics resolving it reported.

    When resolving fails, `field` is the unresolved field, so later steps still
    find it under its old name.
    """

    field: ir.RelationField
    diagnostics: tuple[diagnostics.CompileDiagnostic, ...]


def _check_resolve(
    entity_graph: EntityGraph, view: ViewName, node: ir.Resolve, relation: ir.Relation
) -> ir.Relation | Poisoned:
    checked = tuple(
        _resolve_field(entity_graph, view, node, relation, field)
        if node.fields.predicate(field)
        else _CheckedField(field=field, diagnostics=())
        for field in relation.fields
    )
    resolved = ir.Relation(
        source=relation.source, fields=tuple(check.field for check in checked)
    )
    reported = tuple(d for check in checked for d in check.diagnostics)
    if not reported:
        return resolved
    return Poisoned(recovered=resolved, diagnostics=reported)


def _resolve_field(
    entity_graph: EntityGraph,
    view: ViewName,
    node: ir.Resolve,
    relation: ir.Relation,
    field: ir.RelationField,
) -> _CheckedField:
    match _lookup(entity_graph, view, relation, field):
        case _CheckedField() as unresolved:
            return unresolved
        case ir.Lookup() as lookup:
            pass
    # A field named just `_id` keeps its name rather than losing all of it.
    name = FieldName(field.name.root.removesuffix("_id") or field.name.root)
    if name != field.name and name in relation.field_names():
        return _CheckedField(
            field=field,
            diagnostics=(
                diagnostics.ResolvedNameCollision(
                    view=view, source=relation.source, field=field.name, existing=name
                ),
            ),
        )
    if name != field.name and (
        invalid := _field_name_diagnostics(view, relation, name)
    ):
        return _CheckedField(field=field, diagnostics=invalid)
    return _resolved(entity_graph, view, node, name, lookup)


def _lookup(
    entity_graph: EntityGraph,
    view: ViewName,
    relation: ir.Relation,
    field: ir.RelationField,
) -> ir.Lookup | _CheckedField:
    """How `field` is looked up, or the field unresolved with why it can't be."""
    not_a_reference = _CheckedField(
        field=field,
        diagnostics=(
            diagnostics.NotAReference(
                view=view, source=relation.source, field=field.name
            ),
        ),
    )
    match field:
        case ir.Field(info=info, value=ir.SourceField() as reference):
            pass
        case ir.Field() | ir.NestedField():
            return not_a_reference
    match IdAnnotation.of(info):
        case Err():
            return not_a_reference
        case Ok(annotation):
            pass
    match entity_graph.entity_keyed_by(annotation.id_type):
        case Err():
            return _CheckedField(
                field=field,
                diagnostics=(
                    diagnostics.UnresolvableReference(
                        view=view,
                        source=relation.source,
                        field=field.name,
                        id_type=annotation.id_type,
                    ),
                ),
            )
        case Ok(target):
            return ir.Lookup(
                reference=reference,
                id_type=annotation.id_type,
                target=target,
                optionality=annotation.optionality,
            )


def _resolved(
    entity_graph: EntityGraph,
    view: ViewName,
    node: ir.Resolve,
    name: FieldName,
    lookup: ir.Lookup,
) -> _CheckedField:
    """The field holding the looked-up Entity, after `node.then` if it does anything.

    Without a nested pipeline the field is typed by the Entity class itself.
    """
    root = ir.FromSource(source=lookup.target)
    nested = node.then(root)
    if nested == root:
        entity = (
            lookup.target
            if lookup.optionality is Optionality.REQUIRED
            else lookup.target | None
        )
        return _CheckedField(
            field=ir.Field(
                name=name, info=FieldInfo.from_annotation(entity), value=lookup
            ),
            diagnostics=(),
        )
    match _check_node(entity_graph, view, nested):
        case ir.Relation() as relation:
            return _CheckedField(
                field=ir.NestedField(name=name, value=lookup, relation=relation),
                diagnostics=(),
            )
        case Poisoned(recovered=recovered, diagnostics=reported):
            return _CheckedField(
                field=ir.NestedField(name=name, value=lookup, relation=recovered),
                diagnostics=reported,
            )


def _poison(
    checked: ir.Relation | Poisoned, earlier: tuple[diagnostics.CompileDiagnostic, ...]
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
