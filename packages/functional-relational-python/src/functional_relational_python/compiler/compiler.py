"""Compiles Views and Payloads to checked frp-ir.

Checking uses the poisoned pattern: a failed check reports a diagnostic and
recovers with the best relation it can build, so later steps and other Views and
Payloads are still checked and independent defects are all reported, once each.
"""

import enum
import math
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
from functional_relational_python.pipeline import ir
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
    """A checked View: its Source, the Relation backends compute from it, the
    base its model subclasses, and the names its generated code uses.

    Every kept field `base` declares has the type it declares."""

    name: ViewName
    source: type[pydantic.BaseModel]
    base: type[pydantic.BaseModel]
    relation: ir.Relation
    names: ViewModelNames


class PoisonedView(pydantic.BaseModel, frozen=True):
    name: ViewName
    diagnostics: Diagnostics


class CompiledViews(pydantic.BaseModel, frozen=True):
    views: tuple[CompiledView, ...]


class CompiledPayload(pydantic.BaseModel, frozen=True):
    """A checked Payload: its Source, the Relation its model is generated from,
    the base its model subclasses, and the names of its models. A Payload has no
    mapping function.

    Every kept field `base` declares has the type it declares."""

    name: PayloadName
    source: type[pydantic.BaseModel]
    base: type[pydantic.BaseModel]
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
                for checked in (*checked_views, *checked_payloads)
                if isinstance(checked, PoisonedView | PoisonedPayload)
                for diagnostic in checked.diagnostics
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
    entity_graph: EntityGraph, definition: View | Payload
) -> ir.Relation | Poisoned:
    """The Relation a View's or Payload's pipeline builds over its Source, checked
    against its base, before its model names are checked."""
    name = definition.name
    match _check_node(
        entity_graph, name, definition.step(ir.FromSource(source=definition.source))
    ):
        case ir.Relation() as relation:
            reported = _check_base(name, definition.base, relation)
            if not reported:
                return relation
            return Poisoned(recovered=relation, diagnostics=reported)
        case Poisoned(recovered=recovered, diagnostics=reported):
            # Nested model names depend on the Relation, which is only partly known.
            return Poisoned(
                recovered=recovered,
                diagnostics=(
                    *reported,
                    *_check_base(name, definition.base, recovered),
                    *ModelNames.name_diagnostics(name),
                ),
            )


def _check_base(
    name: DefinitionName, base: type[pydantic.BaseModel], relation: ir.Relation
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    """A diagnostic per kept field `base` declares with another type.

    The generated model inherits the fields the base declares with the same type,
    rather than declaring them again.
    """
    reported: list[diagnostics.CompileDiagnostic] = []
    for field in relation.fields:
        declared = base.model_fields.get(field.name.root)
        if declared is None:
            continue
        match field:
            case ir.Field() if field.annotation() == declared.annotation:
                pass
            case ir.Field():
                reported.append(
                    diagnostics.BaseFieldTypeMismatch(
                        view=name,
                        base=base,
                        field=field.name,
                        declared=declared.annotation,
                        kept=field.annotation(),
                    )
                )
            case ir.NestedField():
                reported.append(
                    diagnostics.BaseFieldIsNestedModel(
                        view=name, base=base, field=field.name
                    )
                )
    return tuple(reported)


def _check_view(entity_graph: EntityGraph, view: View) -> CompiledView | PoisonedView:
    match _check_relation(entity_graph, view):
        case ir.Relation() as relation:
            pass
        case Poisoned(diagnostics=reported):
            return PoisonedView(name=view.name, diagnostics=reported)
    match ViewModelNames.check(view.name, relation):
        case Ok(names):
            return CompiledView(
                name=view.name,
                source=view.source,
                base=view.base,
                relation=relation,
                names=names,
            )
        case Err(error):
            return PoisonedView(name=view.name, diagnostics=error.diagnostics)


def _check_payload(
    entity_graph: EntityGraph, payload: Payload
) -> CompiledPayload | PoisonedPayload:
    match _check_relation(entity_graph, payload):
        case ir.Relation() as relation:
            pass
        case Poisoned(diagnostics=reported):
            return PoisonedPayload(name=payload.name, diagnostics=reported)
    match ModelNames.check(payload.name, relation):
        case Ok(names):
            return CompiledPayload(
                name=payload.name,
                source=payload.source,
                base=payload.base,
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
            ir.Omit()
            | ir.Rename()
            | ir.RenameSelected()
            | ir.Retype()
            | ir.Project()
            | ir.Resolve()
            | ir.Default()
            | ir.Add()
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
    node: ir.Omit
    | ir.Rename
    | ir.RenameSelected
    | ir.Retype
    | ir.Project
    | ir.Resolve
    | ir.Default
    | ir.Add,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Checks one step against the relation its input has."""
    checked: ir.Relation | Poisoned
    match node:
        case ir.Omit():
            checked = _check_omit(view, node, relation)
        case ir.Rename():
            checked = _check_rename(view, node, relation)
        case ir.RenameSelected():
            checked = _check_rename_selected(view, node, relation)
        case ir.Retype():
            checked = _check_retype(view, node, relation)
        case ir.Project():
            checked = _check_project(view, node, relation)
        case ir.Resolve():
            checked = _check_resolve(entity_graph, view, node, relation)
        case ir.Default():
            checked = _check_default(view, node, relation)
        case ir.Add():
            checked = _check_add(view, node, relation)
    return checked


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
                name=node.new,
                info=FieldInfo(),
                value=ir.SourceField(field=node.field),
                default=ir.NoDefault(),
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


def _check_retype(
    view: DefinitionName, node: ir.Retype, relation: ir.Relation
) -> ir.Relation | Poisoned:
    """Retypes the named field, or every selected field, in place.

    In a View, retyping is a diagnostic. Its fields are still retyped, so later
    steps are checked against the types the pipeline asks for.
    """
    reported: list[diagnostics.CompileDiagnostic] = []
    match view:
        case ViewName():
            reported.append(diagnostics.RetypeInView(view=view))
        case PayloadName():
            pass
    match node.fields:
        case ir.Select() as selected:
            retyped = relation.selected(selected)
        case FieldName() as field:
            retyped = (field,)
            if field not in relation.field_names():
                reported.append(
                    diagnostics.UnknownField(
                        view=view,
                        source=relation.source,
                        field=field,
                        valid=relation.field_names(),
                    )
                )
    fields: list[ir.RelationField] = []
    for field in relation.fields:
        if field.name not in retyped:
            fields.append(field)
            continue
        match field:
            case ir.Field():
                fields.append(field.retyped(node.transform(field.annotation())))
            case ir.NestedField():
                # Left as it is, so later steps still find it.
                reported.append(
                    diagnostics.RetypeNestedField(
                        view=view, source=relation.source, field=field.name
                    )
                )
                fields.append(field)
    checked = ir.Relation(source=relation.source, fields=tuple(fields))
    if not reported:
        return checked
    return Poisoned(recovered=checked, diagnostics=tuple(reported))


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


def _check_default(
    view: DefinitionName, node: ir.Default, relation: ir.Relation
) -> ir.Relation | Poisoned:
    """Gives the named or selected fields the default, if it is a literal.

    A defective default leaves the relation unchanged, so that is an exact
    recovery: later steps never depend on a field's default.
    """
    reported: list[diagnostics.CompileDiagnostic] = []
    match node.fields:
        case ir.Select() as selected:
            # Selecting no field is no defect, as for `omit`.
            names = relation.selected(selected)
        case FieldName() as field if field in relation.field_names():
            names = (field,)
        case FieldName() as field:
            names = ()
            reported.append(
                diagnostics.UnknownField(
                    view=view,
                    source=relation.source,
                    field=field,
                    valid=relation.field_names(),
                )
            )
    match _default_value(view, node.value):
        case diagnostics.NonLiteralDefault() as invalid:
            return Poisoned(recovered=relation, diagnostics=(*reported, invalid))
        case default:
            pass
    defaulted = relation.defaulted(names, default)
    if not reported:
        return defaulted
    return Poisoned(recovered=defaulted, diagnostics=tuple(reported))


def _check_add(
    view: DefinitionName, node: ir.Add, relation: ir.Relation
) -> ir.Relation | Poisoned:
    """Appends the declared field, if its name is new and its default a literal.

    A View's added field needs a default. A defective default recovers with the
    field added without one, so later steps still find it. A name the relation
    already has recovers with the relation unchanged, so the existing field keeps
    it, as for a rename.
    """
    reported: list[diagnostics.CompileDiagnostic] = []
    collides = node.name in relation.field_names()
    if collides:
        reported.append(diagnostics.AddOntoExistingField(view=view, field=node.name))
    reported.extend(ViewModelNames.check_field(view, relation.source, node.name))
    default: ir.DefaultValue | ir.NoDefault
    match node.default:
        case ir.NoDefault():
            default = node.default
            if isinstance(view, ViewName):
                reported.append(
                    diagnostics.AddWithoutDefault(view=view, field=node.name)
                )
        case value:
            match _default_value(view, value):
                case diagnostics.NonLiteralDefault() as invalid:
                    default = ir.NoDefault()
                    reported.append(invalid)
                case literal:
                    default = literal
    added = relation if collides else relation.added(node.name, node.info, default)
    if not reported:
        return added
    return Poisoned(recovered=added, diagnostics=tuple(reported))


def _default_value(
    view: DefinitionName, value: ir.UncheckedDefault
) -> ir.DefaultValue | diagnostics.NonLiteralDefault:
    """`value` as a literal default, if it is one.

    Only the exact built-in types count, so a subclass that might `repr` as
    something else is not a literal. Neither is a float without a literal, such as
    `nan`.
    """
    default: ir.DefaultValue
    match value:
        case pydantic.MISSING:
            default = ir.MissingDefault()
        case enum.Enum() if type(value).__members__.get(value.name) is value:
            # A combination of Flag members, or an empty Flag, has no name to
            # write out.
            default = ir.EnumDefault(member=value)
        case None:
            default = ir.ScalarDefault(None)
        case bool() | int() | str() if type(value) in (bool, int, str):
            default = ir.ScalarDefault(value)
        case float() if type(value) is float and math.isfinite(value):
            default = ir.ScalarDefault(value)
        case list() if type(value) is list and not value:
            default = ir.EmptyListDefault()
        case dict() if type(value) is dict and not value:
            default = ir.EmptyDictDefault()
        case _:
            return diagnostics.NonLiteralDefault(
                view=view, value=diagnostics.DefaultText.of(value)
            )
    return default


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
    def selected(field: ir.RelationField) -> bool:
        match node.fields:
            case ir.Select(predicate=predicate):
                return predicate(field)
            case ir.References() as references:
                return references.selects(field, entity_graph.id_base)

    checked = tuple(
        _resolve_field(entity_graph, view, node, relation, field)
        if selected(field)
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

    Without a nested pipeline the field is typed by the Entity class itself. The
    field is new, so it has no default: the Reference's default was for an ID, and
    need not fit the Entity.
    """
    default = ir.NoDefault()
    root = ir.FromSource(source=lookup.target)
    nested = node.then(root)
    if nested == root:
        return _CheckedField(
            field=ir.Field.looked_up(name, lookup, default), diagnostics=()
        )
    match _check_node(entity_graph, view, nested):
        case ir.Relation() as relation:
            return _CheckedField(
                field=ir.NestedField(
                    name=name, value=lookup, relation=relation, default=default
                ),
                diagnostics=(),
            )
        case Poisoned(recovered=recovered, diagnostics=reported):
            return _CheckedField(
                field=ir.NestedField(
                    name=name, value=lookup, relation=recovered, default=default
                ),
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
