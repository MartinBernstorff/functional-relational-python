"""Compiles Views and Payloads to checked frp-ir.

Checking uses the poisoned pattern: a failed check reports a diagnostic and
recovers with the best relation it can build, so later steps and other Views and
Payloads are still checked and independent defects are all reported, once each.
"""

from collections.abc import Iterable
from typing import Annotated

import pydantic
from pydantic.fields import FieldInfo
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import diagnostics
from functional_relational_python.compiler.model_names import (
    ModelNames,
    ViewModelNames,
)
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    NoEntityError,
    NotAReferenceError,
)
from functional_relational_python.pipeline import ir, steps
from functional_relational_python.pipeline.payloads import (
    DefinitionName,
    Payload,
    PayloadName,
)
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.support.diagnostic import DiagnosticsError

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
    """A checked View: its Source, the Relation backends compute from it, and the
    names its generated code uses."""

    name: ViewName
    source: type[pydantic.BaseModel]
    relation: ir.Relation
    names: ViewModelNames


class PoisonedView(pydantic.BaseModel, frozen=True):
    name: ViewName
    diagnostics: Diagnostics


class CompiledViews(pydantic.BaseModel, frozen=True):
    views: tuple[CompiledView, ...]


class CompiledPayload(pydantic.BaseModel, frozen=True):
    """A checked Payload: its Source, the Relation its model is generated from,
    and the names of its models. A Payload has no mapping function."""

    name: PayloadName
    source: type[pydantic.BaseModel]
    relation: ir.Relation
    names: ModelNames


class CompiledCatalog(pydantic.BaseModel, frozen=True):
    """Every View and Payload of a Catalog, checked."""

    views: tuple[CompiledView, ...]
    payloads: tuple[CompiledPayload, ...]


def compile_views(
    entity_graph: EntityGraph, views: Iterable[View], /
) -> Result[CompiledViews, DiagnosticsError[diagnostics.CompileDiagnostic]]:
    """Compiles every View, or returns every diagnostic from all of them.

    `entity_graph` is what References are resolved against.
    """
    match compile_catalog(entity_graph, views, ()):
        case Ok(compiled):
            return Ok(CompiledViews(views=compiled.views))
        case Err(error):
            return Err(error)


def compile_catalog(
    entity_graph: EntityGraph,
    views: Iterable[View],
    payloads: Iterable[Payload],
    /,
) -> Result[CompiledCatalog, DiagnosticsError[diagnostics.CompileDiagnostic]]:
    """Compiles every View and Payload, or returns every diagnostic from all of
    them, including one per name a View and a Payload share."""
    checked_views = check(entity_graph, views)
    checked_payloads = tuple(
        _check_payload(entity_graph, payload) for payload in payloads
    )
    payload_names = {payload.name.root for payload in checked_payloads}
    # One diagnostic per shared name, however many Views and Payloads have it.
    shared_names = dict.fromkeys(
        view.name for view in checked_views if view.name.root in payload_names
    )
    return DiagnosticsError.or_ok(
        (
            *(
                diagnostic
                for view in checked_views
                if isinstance(view, PoisonedView)
                for diagnostic in view.diagnostics
            ),
            *(
                diagnostic
                for payload in checked_payloads
                if isinstance(payload, PoisonedPayload)
                for diagnostic in payload.diagnostics
            ),
            *(diagnostics.SharedModelName(view=name) for name in shared_names),
        ),
        lambda: CompiledCatalog(
            views=tuple(v for v in checked_views if isinstance(v, CompiledView)),
            payloads=tuple(
                p for p in checked_payloads if isinstance(p, CompiledPayload)
            ),
        ),
    )


def check(
    entity_graph: EntityGraph, views: Iterable[View]
) -> tuple[CompiledView | PoisonedView, ...]:
    """Checks every View independently, so a defect in one never hides another."""
    return tuple(_check_view(entity_graph, view) for view in views)


class PoisonedPayload(pydantic.BaseModel, frozen=True):
    name: PayloadName
    diagnostics: Diagnostics


def _check_relation(
    entity_graph: EntityGraph,
    name: DefinitionName,
    source: type[pydantic.BaseModel],
    step: steps.Step,
) -> ir.Relation | Poisoned:
    """The Relation a View's or Payload's pipeline builds over its Source, before
    its model names are checked."""
    match _check_node(entity_graph, name, step(ir.FromSource(source=source))):
        case ir.Relation() as relation:
            return relation
        case Poisoned(recovered=recovered, diagnostics=reported):
            # Nested model names depend on the Relation, which is only partly known.
            return Poisoned(
                recovered=recovered,
                diagnostics=(*reported, *ModelNames.name_diagnostics(name)),
            )


def _check_view(entity_graph: EntityGraph, view: View) -> CompiledView | PoisonedView:
    match _check_relation(entity_graph, view.name, view.source, view.step):
        case ir.Relation() as relation:
            pass
        case Poisoned(diagnostics=reported):
            return PoisonedView(name=view.name, diagnostics=reported)
    match ViewModelNames.check(view.name, relation):
        case Ok(names):
            return CompiledView(
                name=view.name, source=view.source, relation=relation, names=names
            )
        case Err(error):
            return PoisonedView(name=view.name, diagnostics=error.diagnostics)


def _check_payload(
    entity_graph: EntityGraph, payload: Payload
) -> CompiledPayload | PoisonedPayload:
    match _check_relation(entity_graph, payload.name, payload.source, payload.step):
        case ir.Relation() as relation:
            pass
        case Poisoned(diagnostics=reported):
            return PoisonedPayload(name=payload.name, diagnostics=reported)
    match ModelNames.check(payload.name, relation):
        case Ok(names):
            return CompiledPayload(
                name=payload.name,
                source=payload.source,
                relation=relation,
                names=names,
            )
        case Err(error):
            return PoisonedPayload(name=payload.name, diagnostics=error.diagnostics)


def _check_node(
    entity_graph: EntityGraph, view: DefinitionName, node: ir.Node
) -> ir.Relation | Poisoned:
    match node:
        case ir.FromSource():
            return ir.Relation.of(node)
        case (
            ir.Omit() | ir.Rename() | ir.RenameSelected() | ir.Project() | ir.Resolve()
        ):
            match _check_node(entity_graph, view, node.input):
                case ir.Relation() as relation:
                    return _check_step(entity_graph, view, node, relation)
                case Poisoned(recovered=recovered, diagnostics=earlier):
                    return _poison(
                        _check_step(entity_graph, view, node, recovered), earlier
                    )


def _check_step(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.Omit | ir.Rename | ir.RenameSelected | ir.Project | ir.Resolve,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Checks one step against the relation its input has."""
    match node:
        case ir.Omit():
            return _check_omit(view, node, relation)
        case ir.Rename():
            return _check_rename(view, node, relation)
        case ir.RenameSelected():
            return _check_rename_selected(view, node, relation)
        case ir.Project():
            return _check_project(view, node, relation)
        case ir.Resolve():
            return _check_resolve(entity_graph, view, node, relation)


def _check_omit(
    view: DefinitionName, node: ir.Omit, relation: ir.Relation
) -> ir.Relation | Poisoned:
    match node.fields:
        case ir.Select() as selected:
            # Selecting no field is no defect, as for `resolve`.
            return relation.without(relation.selected(selected))
        case FieldName() as field if field in relation.field_names():
            return relation.without((field,))
        case FieldName() as field:
            # Omitting a field that is not there leaves the relation unchanged, so
            # that is an exact recovery.
            return Poisoned(
                recovered=relation,
                diagnostics=(
                    diagnostics.UnknownField(
                        view=view,
                        source=relation.source,
                        field=field,
                        valid=relation.field_names(),
                    ),
                ),
            )


def _check_rename(
    view: DefinitionName, node: ir.Rename, relation: ir.Relation
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
    reported.extend(ViewModelNames.check_field(view, relation.source, node.new))
    if not reported:
        return relation.renamed({node.field: node.new})
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
        return relation.without((node.field,))
    if node.field in names:
        return relation.renamed({node.field: node.new})
    return ir.Relation(
        source=relation.source,
        fields=(
            *relation.fields,
            ir.Field(
                name=node.new, info=FieldInfo(), value=ir.SourceField(field=node.field)
            ),
        ),
    )


def _check_rename_selected(
    view: DefinitionName, node: ir.RenameSelected, relation: ir.Relation
) -> ir.Relation | Poisoned:
    """Renames every selected field at once, so selected fields can swap names.

    A selected field renamed onto a name that an unselected field, a selected
    field renamed to its own name, or an earlier selected field ends up with is a
    diagnostic. It is dropped, so the field that has the name keeps it, as for a
    single rename.
    """
    new_names = {old: node.new(old) for old in relation.selected(node.fields)}
    kept = {
        name for name in relation.field_names() if new_names.get(name, name) == name
    }
    taken = set(kept)
    renames: dict[FieldName, FieldName] = {}
    dropped: list[FieldName] = []
    reported: list[diagnostics.CompileDiagnostic] = []
    for old, new in new_names.items():
        if old in kept:
            continue
        reported.extend(ViewModelNames.check_field(view, relation.source, new))
        if new in taken:
            reported.append(
                diagnostics.RenameOntoExistingField(view=view, renamed=old, field=new)
            )
            dropped.append(old)
        else:
            taken.add(new)
            renames[old] = new
    renamed = relation.without(dropped).renamed(renames)
    if not reported:
        return renamed
    return Poisoned(recovered=renamed, diagnostics=tuple(reported))


def _check_project(
    view: DefinitionName, node: ir.Project, relation: ir.Relation
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
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.Resolve,
    relation: ir.Relation,
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
    view: DefinitionName,
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
        invalid := ViewModelNames.check_field(view, relation.source, name)
    ):
        return _CheckedField(field=field, diagnostics=invalid)
    return _resolved(entity_graph, view, node, name, lookup)


def _lookup(
    entity_graph: EntityGraph,
    view: DefinitionName,
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
        case ir.Field(info=info, value=ir.SourceField(field=source)):
            pass
        case ir.Field() | ir.NestedField():
            return not_a_reference
    match entity_graph.reference_of(source, info):
        case Ok(reference):
            return ir.Lookup(
                reference=ir.SourceField(field=reference.field),
                id_type=reference.id_type,
                target=reference.target,
                optionality=reference.optionality,
            )
        case Err(error):
            pass
    match error:
        case NotAReferenceError():
            return not_a_reference
        case NoEntityError(id_type=id_type):
            return _CheckedField(
                field=field,
                diagnostics=(
                    diagnostics.UnresolvableReference(
                        view=view,
                        source=relation.source,
                        field=field.name,
                        id_type=id_type,
                    ),
                ),
            )


def _resolved(
    entity_graph: EntityGraph,
    view: DefinitionName,
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
        return _CheckedField(field=ir.Field.looked_up(name, lookup), diagnostics=())
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
