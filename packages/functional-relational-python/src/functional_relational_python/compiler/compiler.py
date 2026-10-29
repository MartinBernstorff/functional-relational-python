"""Compiles Views and Payloads to checked frp-ir.

Checking uses the poisoned pattern: a failed check reports a diagnostic and
recovers with the best relation it can build, so later steps and other Views and
Payloads are still checked and independent defects are all reported, once each.
"""

import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated

import pydantic
from pydantic.fields import FieldInfo
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import (
    condition_checker,
    diagnostics,
    literals,
    narrowing,
)
from functional_relational_python.compiler.model_names import (
    ModelName,
    ModelNames,
    PascalCase,
    UnionModelNames,
    ViewModelNames,
)
from functional_relational_python.entity_graph.entity_graph import (
    Cardinality,
    EntityGraph,
    FieldName,
    NoEntityError,
    NotAReferenceError,
    Optionality,
)
from functional_relational_python.entity_graph.id import OneToOneMarker
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.payloads import (
    DefinitionName,
    Payload,
    PayloadName,
)
from functional_relational_python.pipeline.unions import Union, UnionName
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

    def leaves(self) -> tuple[CompiledView]:
        """This View, as the one leaf of itself, like a Union's leaves."""
        return (self,)


class PoisonedView(pydantic.BaseModel, frozen=True):
    name: ViewName
    diagnostics: Diagnostics


class CompiledUnion(pydantic.BaseModel, frozen=True):
    """A checked Union of Views: the Source its Variants read, its Discriminator,
    its checked Variants, and the names its generated code uses.

    Every View among its leaves has the Discriminator, typed by a Tag no other
    Variant has; a nested Union's leaves share theirs.
    """

    name: UnionName
    source: type[pydantic.BaseModel]
    discriminator: FieldName
    variants: tuple[CompiledView | CompiledUnion, ...]
    names: UnionModelNames

    def leaves(self) -> tuple[CompiledView, ...]:
        """The Views a Source row may be a result of, in the order they are tried:
        the first whose Filters keep the row gives the result."""
        return tuple(leaf for variant in self.variants for leaf in variant.leaves())


class PoisonedUnion(pydantic.BaseModel, frozen=True):
    name: UnionName
    diagnostics: Diagnostics


class BlockedUnion(pydantic.BaseModel, frozen=True):
    """A Union with no defect of its own that is not compiled, because a Variant
    of it did not compile. That Variant's diagnostics are reported."""

    name: UnionName


class CompiledViews(pydantic.BaseModel, frozen=True):
    views: tuple[CompiledView, ...]
    unions: tuple[CompiledUnion, ...]


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

    def leaves(self) -> tuple[CompiledPayload]:
        """This Payload, as the one leaf of itself, like a Union's leaves."""
        return (self,)


class CompiledPayloadUnion(pydantic.BaseModel, frozen=True):
    """A checked Union of Payloads: its Discriminator, its checked Variants, and
    the name of its model. Its Variants may read different Sources. Like a
    Payload, it has no mapping function.

    Every Payload among its leaves has the Discriminator, typed by a Tag no other
    Variant has; a nested Union's leaves share theirs.
    """

    name: UnionName
    discriminator: FieldName
    variants: tuple[CompiledPayload | CompiledPayloadUnion, ...]
    model: ModelName

    def leaves(self) -> tuple[CompiledPayload, ...]:
        """The Payloads among its Variants and those of the Unions nested in it,
        in Variant order."""
        return tuple(leaf for variant in self.variants for leaf in variant.leaves())


class CompiledCatalog(pydantic.BaseModel, frozen=True):
    """Every View, Payload and Union of a Catalog, checked, and the Entity graph's
    `OneToOneMarker`, which generated models leave out.

    `views` has every View a Union has as a Variant, and `payloads` every
    Payload. `unions` has the Unions of Views and `payload_unions` the Unions of
    Payloads, each including every Union nested in another, after the Unions
    nested in it.
    """

    views: tuple[CompiledView, ...]
    payloads: tuple[CompiledPayload, ...]
    unions: tuple[CompiledUnion, ...]
    payload_unions: tuple[CompiledPayloadUnion, ...]
    one_to_one_marker: OneToOneMarker


def compile_views(
    entity_graph: EntityGraph,
    views: Iterable[View],
    unions: Iterable[Union] = (),
    /,
) -> Result[CompiledViews, DiagnosticsError[diagnostics.CompileDiagnostic]]:
    """Compiles every View and Union, or returns every diagnostic from all of
    them.

    `entity_graph` is what References are resolved against. A Union of Payloads
    is a diagnostic, as it is a write model, and is not compiled;
    `compile_catalog` compiles it.
    """
    unions = tuple(unions)
    misplaced: tuple[diagnostics.CompileDiagnostic, ...] = tuple(
        diagnostics.PayloadUnionInViews(view=union.name)
        for union in unions
        if not union.views()
    )
    read_unions = (union for union in unions if union.views())
    match compile_catalog(entity_graph, views, (), read_unions):
        case Ok(compiled):
            return DiagnosticsError.or_ok(
                misplaced,
                lambda: CompiledViews(views=compiled.views, unions=compiled.unions),
            )
        case Err(error):
            return Err(DiagnosticsError((*error.diagnostics, *misplaced)))


def compile_catalog(
    entity_graph: EntityGraph,
    views: Iterable[View],
    payloads: Iterable[Payload],
    unions: Iterable[Union],
    /,
) -> Result[CompiledCatalog, DiagnosticsError[diagnostics.CompileDiagnostic]]:
    """Compiles every View, Payload and Union, or returns every diagnostic from
    all of them, including one per name a View and a Payload share.

    A Union's Variants are compiled as if `views` or `payloads` listed them, and
    the Unions nested in it as if `unions` did. A View, Payload or Union listed or
    nested more than once is compiled once.
    """
    unions = tuple(unions)
    all_unions = tuple(
        dict.fromkeys(nested for union in unions for nested in union.unions())
    )
    all_views = tuple(
        dict.fromkeys((*views, *(view for union in unions for view in union.views())))
    )
    all_payloads = tuple(
        dict.fromkeys(
            (*payloads, *(payload for union in unions for payload in union.payloads()))
        )
    )
    checked_views = check(entity_graph, all_views)
    checked_payloads = tuple(
        _check_payload(entity_graph, payload) for payload in all_payloads
    )
    checked_unions = _check_unions(
        all_unions,
        dict(zip(all_views, checked_views, strict=True)),
        dict(zip(all_payloads, checked_payloads, strict=True)),
    )
    payload_names = {payload.name.root for payload in checked_payloads}
    # One diagnostic per shared name, however many Views and Payloads have it.
    shared_names = dict.fromkeys(
        view.name for view in checked_views if view.name.root in payload_names
    )
    other_names = {view.name.root for view in all_views} | payload_names
    union_names = [union.name.root for union in all_unions]
    shared_union_names = dict.fromkeys(
        union.name
        for union in all_unions
        if union.name.root in other_names or union_names.count(union.name.root) > 1
    )
    return DiagnosticsError.or_ok(
        (
            *(
                diagnostic
                for checked in (*checked_views, *checked_payloads, *checked_unions)
                if isinstance(checked, PoisonedView | PoisonedPayload | PoisonedUnion)
                for diagnostic in checked.diagnostics
            ),
            *(diagnostics.SharedModelName(view=name) for name in shared_names),
            *(diagnostics.SharedUnionName(view=name) for name in shared_union_names),
        ),
        lambda: CompiledCatalog(
            views=tuple(v for v in checked_views if isinstance(v, CompiledView)),
            payloads=tuple(
                p for p in checked_payloads if isinstance(p, CompiledPayload)
            ),
            unions=tuple(u for u in checked_unions if isinstance(u, CompiledUnion)),
            payload_unions=tuple(
                u for u in checked_unions if isinstance(u, CompiledPayloadUnion)
            ),
            one_to_one_marker=entity_graph.one_to_one_marker,
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
        entity_graph,
        name,
        definition.step(ir.FromSource(source=definition.source)),
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
            reported = ()
        case Poisoned(recovered=relation, diagnostics=reported):
            pass
    # One per `where`: a Payload is a write model, with no rows to filter.
    misplaced = tuple(
        diagnostics.WhereInPayload(view=payload.name) for _ in relation.filters
    )
    if reported or misplaced:
        return PoisonedPayload(name=payload.name, diagnostics=(*reported, *misplaced))
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
            | ir.MapType()
            | ir.Pick()
            | ir.Resolve()
            | ir.Flatten()
            | ir.Reverse()
            | ir.Default()
            | ir.Add()
            | ir.Where()
            | ir.NonNullable()
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
    | ir.MapType
    | ir.Pick
    | ir.Resolve
    | ir.Flatten
    | ir.Reverse
    | ir.Default
    | ir.Add
    | ir.Where
    | ir.NonNullable,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Checks one step against the relation its input has."""
    checked: ir.Relation | Poisoned
    match node:
        case ir.Omit():
            checked = _check_omit(entity_graph, view, node, relation)
        case ir.Rename():
            checked = _check_rename(view, node, relation)
        case ir.RenameSelected():
            checked = _check_rename_selected(entity_graph, view, node, relation)
        case ir.MapType():
            checked = _check_map_type(entity_graph, view, node, relation)
        case ir.Pick():
            checked = _check_pick(entity_graph, view, node, relation)
        case ir.Resolve():
            checked = _check_resolve(entity_graph, view, node, relation)
        case ir.Flatten():
            checked = _check_flatten(entity_graph, view, node, relation)
        case ir.Reverse():
            checked = _check_reverse(entity_graph, view, node, relation)
        case ir.Default():
            checked = _check_default(entity_graph, view, node, relation)
        case ir.Add():
            checked = _check_add(view, node, relation)
        case ir.Where():
            checked = _check_where(entity_graph, view, node, relation)
        case ir.NonNullable():
            checked = _check_non_nullable(entity_graph, view, node, relation)
    return checked


class _Selection(pydantic.BaseModel, frozen=True):
    """The fields a step acts on, in field order, with each name it gave that the
    relation does not have."""

    names: tuple[FieldName, ...]
    unknown: tuple[diagnostics.UnknownField, ...]

    @staticmethod
    def of(
        entity_graph: EntityGraph,
        view: DefinitionName,
        fields: ir.FieldSelection,
        relation: ir.Relation,
    ) -> _Selection:
        match fields:
            case ir.Named(names=named):
                return _Selection(
                    names=tuple(
                        name for name in relation.field_names() if name in named
                    ),
                    unknown=tuple(
                        diagnostics.UnknownField(
                            owner=view,
                            source=relation.source,
                            field=name,
                            valid=relation.field_names(),
                        )
                        for name in named
                        if name not in relation.field_names()
                    ),
                )
            case ir.Select():
                return _Selection(names=relation.selected(fields), unknown=())
            case ir.References():
                return _Selection(
                    names=tuple(
                        field.name
                        for field in relation.fields
                        if fields.selects(field, entity_graph)
                    ),
                    unknown=(),
                )


def _check_omit(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.Omit,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    selection = _Selection.of(entity_graph, view, node.fields, relation)
    omitted = relation.without(selection.names)
    if not selection.unknown:
        return omitted
    # Omitting a field that is not there leaves it out anyway, so that is an
    # exact recovery.
    return Poisoned(recovered=omitted, diagnostics=selection.unknown)


def _check_rename(
    view: DefinitionName, node: ir.Rename, relation: ir.Relation
) -> ir.Relation | Poisoned:
    names = relation.field_names()
    reported: list[diagnostics.CompileDiagnostic] = []
    if node.field not in names:
        reported.append(
            diagnostics.UnknownField(
                owner=view, source=relation.source, field=node.field, valid=names
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
    return relation.with_fields(
        (
            *relation.fields,
            ir.Field(
                name=node.new,
                info=FieldInfo(),
                value=ir.SourceField(field=node.field),
                default=ir.NoDefault(),
            ),
        )
    )


def _check_rename_selected(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.RenameSelected,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Renames every named or selected field at once, so they can swap names.

    A selected field renamed onto a name that an unselected field, a selected
    field renamed to its own name, or an earlier selected field ends up with is a
    diagnostic. It is dropped, so the field that has the name keeps it, as for a
    single rename.
    """
    selection = _Selection.of(entity_graph, view, node.fields, relation)
    new_names = {old: node.new(old) for old in selection.names}
    kept = {
        name for name in relation.field_names() if new_names.get(name, name) == name
    }
    taken = set(kept)
    renames: dict[FieldName, FieldName] = {}
    dropped: list[FieldName] = []
    reported: list[diagnostics.CompileDiagnostic] = list(selection.unknown)
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


def _check_map_type(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.MapType,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Gives every named or selected field, in place, the type the transform maps
    its type to.

    In a View, mapping types is a diagnostic. Its fields still get their new
    types, so later steps are checked against the types the pipeline asks for.
    """
    reported: list[diagnostics.CompileDiagnostic] = []
    match view:
        case ViewName():
            reported.append(diagnostics.MapTypeInView(view=view))
        case PayloadName():
            pass
    selection = _Selection.of(entity_graph, view, node.fields, relation)
    reported.extend(selection.unknown)
    fields: list[ir.RelationField] = []
    for field in relation.fields:
        if field.name not in selection.names:
            fields.append(field)
            continue
        match field:
            case ir.Field():
                fields.append(field.retyped(node.transform(field.annotation())))
            case ir.NestedField():
                # Left as it is, so later steps still find it.
                reported.append(
                    diagnostics.MapTypeNestedField(
                        view=view, source=relation.source, field=field.name
                    )
                )
                fields.append(field)
    checked = relation.with_fields(tuple(fields))
    if not reported:
        return checked
    return Poisoned(recovered=checked, diagnostics=tuple(reported))


def _check_pick(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.Pick,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    selection = _Selection.of(entity_graph, view, node.fields, relation)
    kept = relation.keeping(selection.names)
    if not selection.unknown:
        return kept
    # The known fields are kept, so later steps are checked against them.
    return Poisoned(recovered=kept, diagnostics=selection.unknown)


def _check_default(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.Default,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Gives the named or selected fields the default, if it is a literal.

    A defective default leaves the relation unchanged, so that is an exact
    recovery: later steps never depend on a field's default.
    """
    selection = _Selection.of(entity_graph, view, node.fields, relation)
    match _default_value(view, node.value):
        case diagnostics.NonLiteralDefault() as invalid:
            return Poisoned(
                recovered=relation, diagnostics=(*selection.unknown, invalid)
            )
        case default:
            pass
    defaulted = relation.defaulted(selection.names, default)
    if not selection.unknown:
        return defaulted
    return Poisoned(recovered=defaulted, diagnostics=selection.unknown)


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


def _check_where(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.Where,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Adds the checked Condition to the relation's Filters, and narrows the
    fields it tests, as `narrowing.by_filter` does.

    A defective Condition leaves the relation unchanged. Where a Filter is not
    allowed, in a nested pipeline or a Payload, is reported once the pipeline is
    checked.
    """
    match condition_checker.ConditionChecker(
        owner=view, relation=relation, subject=_subject(node).source
    ).check(node.condition):
        case condition_checker.Defects(diagnostics=reported):
            return Poisoned(recovered=relation, diagnostics=reported)
        case checked:
            return narrowing.by_filter(
                relation.filtered(checked), checked, entity_graph
            )


def _subject(node: ir.Node) -> ir.FromSource:
    """What the pipeline that `node` ends started from, however the relation has
    changed since: the View's Source, or the Entity a nested pipeline runs on."""
    while not isinstance(node, ir.FromSource):
        node = node.input
    return node


def _check_non_nullable(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.NonNullable,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Narrows the named or selected fields from `X | None` to `X`.

    Each field that cannot be made non-nullable is a diagnostic, and is left as it
    is, so later steps still find it. Doing so in a Payload is a diagnostic; its
    fields are still narrowed, so later steps are checked against the types the
    pipeline asks for, as for `map_type` in a View.
    """
    reported: list[diagnostics.CompileDiagnostic] = []
    match view:
        case PayloadName():
            reported.append(diagnostics.NonNullableInPayload(view=view))
        case ViewName():
            pass
    selection = _Selection.of(entity_graph, view, node.fields, relation)
    reported.extend(selection.unknown)
    fields: list[ir.RelationField] = []
    for field in relation.fields:
        if field.name not in selection.names:
            fields.append(field)
            continue
        match narrowing.required(view, relation.source, field):
            case ir.Field() | ir.NestedField() as narrowed:
                fields.append(narrowed)
            case diagnostic:
                reported.append(diagnostic)
                fields.append(field)
    checked = relation.with_fields(tuple(fields))
    if not reported:
        return checked
    return Poisoned(recovered=checked, diagnostics=tuple(reported))


def _default_value(
    view: DefinitionName, value: ir.UncheckedDefault
) -> ir.DefaultValue | diagnostics.NonLiteralDefault:
    """`value` as a literal default, if it is one: `None`, `MISSING`, an empty
    list or dict, or a scalar or enum member as `literals.classify_scalar` has it.
    """
    default: ir.DefaultValue | diagnostics.NonLiteralDefault
    match value:
        case pydantic.MISSING:
            default = ir.MissingDefault()
        case None:
            default = ir.ScalarDefault(None)
        case list() if type(value) is list and not value:
            default = ir.EmptyListDefault()
        case dict() if type(value) is dict and not value:
            default = ir.EmptyDictDefault()
        case _:
            match literals.classify_scalar(value):
                case ir.ScalarLiteral(root=scalar):
                    default = ir.ScalarDefault(scalar)
                case ir.EnumLiteral(member=member):
                    default = ir.EnumDefault(member=member)
                case literals.NotALiteral():
                    default = diagnostics.NonLiteralDefault(
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
    selection = _Selection.of(entity_graph, view, node.fields, relation)
    checked = tuple(
        _resolve_field(entity_graph, view, node, relation, field)
        if field.name in selection.names
        else _CheckedField(field=field, diagnostics=())
        for field in relation.fields
    )
    resolved = relation.with_fields(tuple(check.field for check in checked))
    reported = (
        *selection.unknown,
        *(d for check in checked for d in check.diagnostics),
    )
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
        case ir.Lookup() | ir.ReverseLookup() | ir.ChainedLookup() as found:
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
    return _resolved(entity_graph, view, node.then, name, found)


def _lookup(
    entity_graph: EntityGraph,
    view: DefinitionName,
    relation: ir.Relation,
    field: ir.RelationField,
) -> ir.EntityValue | _CheckedField:
    """How the Entity `field` references is found, or the field unresolved with
    why it can't be."""
    not_a_reference = _CheckedField(
        field=field,
        diagnostics=(
            diagnostics.NotAReference(
                view=view, source=relation.source, field=field.name
            ),
        ),
    )
    match field:
        case ir.Field():
            pass
        case ir.NestedField():
            return not_a_reference
    match field.reference_source():
        case (source, info):
            pass
        case ir.Field():
            return not_a_reference
    match entity_graph.reference_of(source, info):
        case Ok(reference):
            lookup = ir.Lookup(
                reference=ir.SourceField(field=reference.field),
                id_type=reference.id_type,
                target=reference.target,
                optionality=reference.optionality,
            )
            match field.value:
                case ir.RequiredField() | ir.Lifted(field=ir.RequiredField()):
                    # The field's type is narrowed, but the Reference it reads is
                    # optional, and the View requires it: `None` there is unmet.
                    lookup = lookup.model_copy(
                        update={"optionality": Optionality.OPTIONAL}
                    ).required()
                case _:
                    pass
            # A lifted Reference is followed from the Entity it is read off.
            match field.value:
                case ir.Lifted(through=through):
                    return ir.ChainedLookup(chain=(*through, lookup))
                case _:
                    return lookup
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


def _check_reverse(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.Reverse,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Appends the field holding the Entity whose one-to-one Reference points to
    the subject, named after that Entity.

    A name the relation already has recovers with the relation unchanged, so the
    existing field keeps it, as for `add`. A Reference that cannot be followed
    backwards recovers with a field of unknown type, so later steps still find it.
    """
    source = relation.source
    name = FieldName(PascalCase(node.entity.__name__).snake_case().root)
    reported: list[diagnostics.CompileDiagnostic] = []
    collides = name in relation.field_names()
    if collides:
        reported.append(
            diagnostics.ReverseOntoExistingField(
                view=view, source=source, entity=node.entity, field=name
            )
        )
    reported.extend(ViewModelNames.check_field(view, source, name))
    subject = _subject(node)
    non_entities: list[diagnostics.CompileDiagnostic] = []
    if subject.source not in entity_graph.entities:
        non_entities.append(
            diagnostics.ReverseFromNonEntity(
                view=view, source=source, entity=node.entity
            )
        )
    if node.entity not in entity_graph.entities:
        non_entities.append(
            diagnostics.ReverseOfNonEntity(view=view, source=source, entity=node.entity)
        )
    reported.extend(non_entities)
    field: ir.RelationField = ir.Field(
        name=name, info=FieldInfo(), value=ir.Added(), default=ir.NoDefault()
    )
    if not non_entities:
        to_subject = tuple(
            reference
            for reference in entity_graph.entities[node.entity].references
            if reference.target is subject.source
        )
        one_to_one = tuple(
            reference
            for reference in to_subject
            if reference.cardinality == Cardinality.ONE_TO_ONE
        )
        # What is reported unless exactly one Reference is chosen.
        unchosen: diagnostics.CompileDiagnostic
        match node.via:
            case ir.SoleReference():
                chosen = one_to_one
                unchosen = (
                    diagnostics.AmbiguousReverse(
                        view=view,
                        source=source,
                        entity=node.entity,
                        references=tuple(reference.field for reference in chosen),
                    )
                    if chosen
                    else diagnostics.NoOneToOneReference(
                        view=view,
                        source=source,
                        entity=node.entity,
                        many_to_one=tuple(
                            reference.field
                            for reference in to_subject
                            if reference.cardinality == Cardinality.MANY_TO_ONE
                        ),
                        one_to_one_marker=entity_graph.one_to_one_marker,
                    )
                )
            case FieldName() as via:
                chosen = tuple(
                    reference for reference in one_to_one if reference.field == via
                )
                unchosen = diagnostics.NotAOneToOneReference(
                    view=view,
                    source=source,
                    entity=node.entity,
                    field=via,
                    valid=tuple(reference.field for reference in one_to_one),
                )
        if len(chosen) == 1:
            (reference,) = chosen
            lookup = ir.ReverseLookup(
                optionality=Optionality.OPTIONAL,
                target=node.entity,
                via=reference.field,
                id_type=reference.id_type,
            )
            checked = _resolved(entity_graph, view, node.then, name, lookup)
            field = checked.field
            reported.extend(checked.diagnostics)
        else:
            reported.append(unchosen)
    reversed_relation = (
        relation if collides else relation.with_fields((*relation.fields, field))
    )
    if not reported:
        return reversed_relation
    return Poisoned(recovered=reversed_relation, diagnostics=tuple(reported))


def _resolved(
    entity_graph: EntityGraph,
    view: DefinitionName,
    then: Callable[[ir.Node], ir.Node],
    name: FieldName,
    found: ir.EntityValue,
) -> _CheckedField:
    """The field holding the looked-up Entity, after `then` if it does anything.

    Without a nested pipeline the field is typed by the Entity class itself. The
    field is new, so it has no default: a resolved Reference's default was for an
    ID, and need not fit the Entity.
    """
    default = ir.NoDefault()
    root = ir.FromSource(source=found.lookups()[-1].target)
    nested = then(root)
    if nested == root:
        return _CheckedField(
            field=ir.Field.looked_up(name, found, default), diagnostics=()
        )
    match _check_node(entity_graph, view, nested):
        case ir.Relation() as relation:
            reported = ()
        case Poisoned(recovered=relation, diagnostics=reported):
            pass
    # One per `where`: each looked-up Entity is one field, with no rows to filter.
    misplaced = tuple(
        diagnostics.WhereInNestedPipeline(view=view, source=relation.source)
        for _ in relation.filters
    )
    return _CheckedField(
        field=ir.NestedField(
            name=name, value=found, relation=relation.unfiltered(), default=default
        ),
        diagnostics=(*reported, *misplaced),
    )


class _HeldEntity(pydantic.BaseModel, frozen=True):
    """The Entity a resolved field holds: the Lookups that find it from the
    subject, in turn, and its fields."""

    chain: tuple[ir.EntityLookup, ...]
    relation: ir.Relation

    @staticmethod
    def of(field: ir.RelationField) -> _HeldEntity | ir.Field:
        """The Entity `field` holds, or `field` itself when it is not resolved."""
        match field:
            case ir.Field(
                value=ir.Lookup() | ir.ReverseLookup() | ir.ChainedLookup() as found
            ):
                chain = found.lookups()
                return _HeldEntity(
                    chain=chain,
                    relation=ir.Relation.of(ir.FromSource(source=chain[-1].target)),
                )
            case ir.NestedField(value=found, relation=relation):
                return _HeldEntity(chain=found.lookups(), relation=relation)
            case ir.Field():
                return field

    def lifted(self, field: ir.RelationField) -> ir.RelationField:
        """`field` of the Entity, computed from the subject through `chain`.

        Lifted out of an optional Reference, a field is optional too. An added
        field has no value from the Entity, so it is lifted as it is.
        """
        match field:
            case ir.NestedField(value=found):
                return field.model_copy(
                    update={
                        "value": ir.ChainedLookup(chain=(*self.chain, *found.lookups()))
                    }
                )
            case ir.Field(value=value):
                pass
        lifted_value: ir.Value
        match value:
            case ir.Added():
                return field
            case ir.SourceField() | ir.RequiredField():
                lifted_value = ir.Lifted(through=self.chain, field=value)
            case ir.Lifted(through=through, field=read):
                lifted_value = ir.Lifted(through=(*self.chain, *through), field=read)
            case ir.Lookup() | ir.ReverseLookup() | ir.ChainedLookup():
                lifted_value = ir.ChainedLookup(chain=(*self.chain, *value.lookups()))
        lifted = field.model_copy(update={"value": lifted_value})
        match ir.Lookup.optionality_of(self.chain):
            case Optionality.REQUIRED:
                return lifted
            case Optionality.OPTIONAL:
                return lifted.admitting_none()


def _check_flatten(
    entity_graph: EntityGraph,
    view: DefinitionName,
    node: ir.Flatten,
    relation: ir.Relation,
) -> ir.Relation | Poisoned:
    """Replaces each named or selected resolved field with the fields of the
    Entity it holds, each computed from the subject through the Lookups that find
    the Entity.

    A Selector ignores the fields that are not resolved. A named field that is not
    resolved is a diagnostic, and stays. A lifted field named like a field the
    relation keeps, or an earlier lifted field, is a diagnostic, and is dropped:
    the existing field keeps its name, as for a rename.
    """
    held: dict[FieldName, _HeldEntity] = {}
    for field in relation.fields:
        match _HeldEntity.of(field):
            case _HeldEntity() as entity:
                held[field.name] = entity
            case ir.Field():
                pass
    selection = _Selection.of(entity_graph, view, node.fields, relation)
    reported: list[diagnostics.CompileDiagnostic] = list(selection.unknown)
    match node.fields:
        case ir.Named():
            reported.extend(
                diagnostics.NotAResolvedField(
                    view=view, source=relation.source, field=name
                )
                for name in selection.names
                if name not in held
            )
        case ir.Select() | ir.References():
            pass
    flattened = {name for name in selection.names if name in held}
    # The fields that stay, so their names are taken before any field is lifted.
    taken = {field.name for field in relation.fields if field.name not in flattened}
    fields: list[ir.RelationField] = []
    for field in relation.fields:
        if field.name not in flattened:
            fields.append(field)
            continue
        entity = held[field.name]
        for inner in entity.relation.fields:
            if inner.name in taken:
                reported.append(
                    diagnostics.FlattenOntoExistingField(
                        view=view, flattened=field.name, field=inner.name
                    )
                )
                continue
            taken.add(inner.name)
            fields.append(entity.lifted(inner))
    flattened_relation = relation.with_fields(tuple(fields))
    if not reported:
        return flattened_relation
    return Poisoned(recovered=flattened_relation, diagnostics=tuple(reported))


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


type _CheckedUnion = CompiledUnion | CompiledPayloadUnion | PoisonedUnion | BlockedUnion


def _check_unions(
    unions: Iterable[Union],
    views: Mapping[View, CompiledView | PoisonedView],
    payloads: Mapping[Payload, CompiledPayload | PoisonedPayload],
) -> tuple[_CheckedUnion, ...]:
    """Checks every Union against its checked Variants. `unions` has each Union
    nested in another before it, and `views` and `payloads` every View and
    Payload among their Variants."""
    checked: dict[Union, _CheckedUnion] = {}
    for union in unions:
        checked[union] = _check_union(union, views, payloads, checked)
    return tuple(checked.values())


def _check_union(
    union: Union,
    views: Mapping[View, CompiledView | PoisonedView],
    payloads: Mapping[Payload, CompiledPayload | PoisonedPayload],
    unions: Mapping[Union, _CheckedUnion],
) -> _CheckedUnion:
    """Checks the Union's name, that its Variants are all Views or Unions of them
    over one Source, or all Payloads or Unions of them, and their Tags.

    A Variant that did not compile is left out of the checks, so the Variants
    that did are still checked.
    """
    variants = _SortedVariants.of(union, views, payloads, unions)
    mixed = variants.kind_diagnostics(union)
    # Variants mixing kinds are reported once, and only the Views checked further.
    tagged = (*variants.compiled_views, *(() if mixed else variants.compiled_payloads))
    reported: list[diagnostics.CompileDiagnostic] = [
        *mixed,
        *_tag_diagnostics(union, tagged),
        *_unreachable_diagnostics(union, variants.compiled_views),
    ]
    sources = tuple(variant.source for variant in variants.compiled_views)
    if len(set(sources)) > 1:
        reported.append(
            diagnostics.VariantSourcesDiffer(
                view=union.name,
                variants=tuple(variant.name for variant in variants.compiled_views),
                sources=tuple(ir.SourceName(source.__name__) for source in sources),
            )
        )
    match UnionModelNames.check(union.name):
        case Err(error):
            return PoisonedUnion(
                name=union.name, diagnostics=(*error.diagnostics, *reported)
            )
        case Ok(names):
            pass
    if reported:
        return PoisonedUnion(name=union.name, diagnostics=tuple(reported))
    if variants.failed:
        return BlockedUnion(name=union.name)
    # Without a kind diagnostic, the Variants are all of one kind.
    if variants.compiled_payloads:
        return CompiledPayloadUnion(
            name=union.name,
            discriminator=union.discriminator,
            variants=variants.compiled_payloads,
            model=names.model,
        )
    return CompiledUnion(
        name=union.name,
        source=sources[0],
        discriminator=union.discriminator,
        variants=variants.compiled_views,
        names=names,
    )


class _SortedVariants(pydantic.BaseModel, frozen=True):
    """A Union's Variants by kind: the names of the read models (Views and
    Unions of them) and of the write models (Payloads and Unions of them); the
    checked Variants of each kind that compiled, and the names of those that did
    not. A nested Union that did not compile is of neither kind, as its kind is
    not known.
    """

    read_models: tuple[ViewName | UnionName, ...]
    write_models: tuple[PayloadName | UnionName, ...]
    compiled_views: tuple[CompiledView | CompiledUnion, ...]
    compiled_payloads: tuple[CompiledPayload | CompiledPayloadUnion, ...]
    failed: tuple[ViewName | PayloadName | UnionName, ...]

    @staticmethod
    def of(
        union: Union,
        checked_views: Mapping[View, CompiledView | PoisonedView],
        checked_payloads: Mapping[Payload, CompiledPayload | PoisonedPayload],
        unions: Mapping[Union, _CheckedUnion],
    ) -> _SortedVariants:
        read_models: list[ViewName | UnionName] = []
        write_models: list[PayloadName | UnionName] = []
        compiled_views: list[CompiledView | CompiledUnion] = []
        compiled_payloads: list[CompiledPayload | CompiledPayloadUnion] = []
        failed: list[ViewName | PayloadName | UnionName] = []
        for variant in union.variants:
            checked: (
                CompiledView | PoisonedView | CompiledPayload | PoisonedPayload
            ) | _CheckedUnion
            match variant:
                case View():
                    read_models.append(variant.name)
                    checked = checked_views[variant]
                case Payload():
                    write_models.append(variant.name)
                    checked = checked_payloads[variant]
                case Union():
                    checked = unions[variant]
                    match checked:
                        case CompiledUnion():
                            read_models.append(variant.name)
                        case CompiledPayloadUnion():
                            write_models.append(variant.name)
                        case PoisonedUnion() | BlockedUnion():
                            pass
            match checked:
                case CompiledView() | CompiledUnion():
                    compiled_views.append(checked)
                case CompiledPayload() | CompiledPayloadUnion():
                    compiled_payloads.append(checked)
                case (
                    PoisonedView()
                    | PoisonedPayload()
                    | PoisonedUnion()
                    | BlockedUnion()
                ):
                    failed.append(variant.name)
        return _SortedVariants(
            read_models=tuple(read_models),
            write_models=tuple(write_models),
            compiled_views=tuple(compiled_views),
            compiled_payloads=tuple(compiled_payloads),
            failed=tuple(failed),
        )

    def kind_diagnostics(
        self, union: Union
    ) -> tuple[diagnostics.ViewsMixedWithPayloads, ...]:
        """Why the Variants are not all of one kind, if they are not."""
        if self.read_models and self.write_models:
            return (
                diagnostics.ViewsMixedWithPayloads(
                    view=union.name, views=self.read_models, payloads=self.write_models
                ),
            )
        return ()


def _unreachable_diagnostics(
    union: Union, variants: Sequence[CompiledView | CompiledUnion]
) -> tuple[diagnostics.UnreachableVariant, ...]:
    """A diagnostic per Variant after one that keeps every row, which leaves it
    none: a View without Filters, or a Union with such a View."""
    for index, variant in enumerate(variants):
        if any(not leaf.relation.filters for leaf in variant.leaves()):
            return tuple(
                diagnostics.UnreachableVariant(
                    view=union.name, variant=later.name, kept_by=variant.name
                )
                for later in variants[index + 1 :]
            )
    return ()


type Tag = ir.EnumLiteral | ir.ScalarLiteral
"""The one value a Variant's Discriminator is typed a `Literal` of: an enum
member, a string or an integer."""


type _CompiledVariant = (
    CompiledView | CompiledUnion | CompiledPayload | CompiledPayloadUnion
)


def _tag_diagnostics(
    union: Union, variants: Iterable[_CompiledVariant]
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    """A diagnostic per View or Payload with no Tag for the Union's
    Discriminator, per nested Union whose leaves have several, and per Tag
    several Variants share."""
    reported: list[diagnostics.CompileDiagnostic] = []
    # By the value, as pydantic tells Tags apart: a `StrEnum` member is its string.
    tagged: dict[object, list[ViewName | PayloadName | UnionName]] = {}
    for variant in variants:
        leaves = variant.leaves()
        found: list[Tag] = []
        for leaf in leaves:
            match _tag(union, leaf):
                case (
                    diagnostics.VariantWithoutDiscriminator()
                    | diagnostics.UntaggedVariant()
                    | diagnostics.DiscriminatorDefaultNotTag() as untagged
                ):
                    reported.append(untagged)
                case tag:
                    found.append(tag)
        distinct = tuple({tag.value(): tag for tag in found}.values())
        if len(distinct) > 1 and isinstance(
            variant, CompiledUnion | CompiledPayloadUnion
        ):
            reported.append(
                diagnostics.NestedUnionTagsDiffer(
                    view=union.name,
                    nested=variant.name,
                    discriminator=union.discriminator,
                    tags=tuple(diagnostics.TagText.of(tag.value()) for tag in distinct),
                )
            )
        elif len(found) == len(leaves) and len(distinct) == 1:
            tagged.setdefault(distinct[0].value(), []).append(variant.name)
    reported.extend(
        diagnostics.SharedTag(
            view=union.name,
            discriminator=union.discriminator,
            tag=diagnostics.TagText.of(tag),
            variants=tuple(sharing),
        )
        for tag, sharing in tagged.items()
        if len(sharing) > 1
    )
    return tuple(reported)


def _tag(
    union: Union, leaf: CompiledView | CompiledPayload
) -> (
    Tag
    | diagnostics.VariantWithoutDiscriminator
    | diagnostics.UntaggedVariant
    | diagnostics.DiscriminatorDefaultNotTag
):
    """The Tag `leaf`, a View or a Payload, has for the Union's Discriminator: the
    one value the field's type is a `Literal` of.

    The field may default to its Tag only: another default, such as `MISSING`,
    would let the model be built without its Tag."""
    fields = {field.name: field for field in leaf.relation.fields}
    match fields.get(union.discriminator):
        case None:
            return diagnostics.VariantWithoutDiscriminator(
                view=union.name, variant=leaf.name, discriminator=union.discriminator
            )
        case ir.Field() as field:
            annotation = field.annotation()
            default = field.default
        case ir.NestedField(value=found):
            annotation = found.annotation()
            default = ir.NoDefault()
    split = narrowing.split(annotation)
    values = typing.get_args(split.required)
    if (
        split.optionality is Optionality.REQUIRED
        and typing.get_origin(split.required) is typing.Literal
        and len(values) == 1
    ):
        match literals.classify_scalar(values[0]):
            case ir.ScalarLiteral(root=bool() | float()) | literals.NotALiteral():
                pass
            case ir.EnumLiteral() | ir.ScalarLiteral() as tag:
                if _defaults_to(default, tag):
                    return tag
                return diagnostics.DiscriminatorDefaultNotTag(
                    view=union.name,
                    variant=leaf.name,
                    discriminator=union.discriminator,
                    tag=diagnostics.TagText.of(tag.value()),
                    default=diagnostics.FieldDefaultText.of(default),
                )
    return diagnostics.UntaggedVariant(
        view=union.name,
        variant=leaf.name,
        discriminator=union.discriminator,
        annotation=annotation,
    )


def _defaults_to(default: ir.DefaultValue | ir.NoDefault, tag: Tag) -> bool:
    """Whether a field with `default` has no default, or defaults to `tag`."""
    match default:
        case ir.NoDefault():
            return True
        case ir.EnumDefault(member=member):
            return member is tag.value()
        case ir.ScalarDefault(root=value):
            # `True == 1`, so the types must match too.
            return type(value) is type(tag.value()) and value == tag.value()
        case ir.MissingDefault() | ir.EmptyListDefault() | ir.EmptyDictDefault():
            return False
