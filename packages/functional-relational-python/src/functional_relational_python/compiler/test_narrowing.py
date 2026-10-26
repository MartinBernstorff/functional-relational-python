import datetime
import enum
from typing import Annotated, Literal

import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler, diagnostics
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
    RuntimeAnnotation,
)
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    OneToOne,
    OneToOneMarker,
)
from functional_relational_python.pipeline import conditions, ir, steps
from functional_relational_python.pipeline.payloads import Payload, PayloadName
from functional_relational_python.pipeline.views import View, ViewName


class Text(pydantic.RootModel[str], frozen=True):
    pass


class Kind(enum.Enum):
    TEMPLATE = "template"
    INSTANCE = "instance"


class SubmitState(enum.StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"


class Mood(enum.Enum):
    # A member named no identifier, which no `Literal` can name.
    vars()["not ok"] = "not ok"


class SubmittedAt(pydantic.RootModel[datetime.datetime], frozen=True):
    pass


class OwnerId(Id[int]):
    pass


class Owner(pydantic.BaseModel):
    id: OwnerId
    nickname: Text | None


class CampaignId(Id[int]):
    pass


class Campaign(pydantic.BaseModel):
    """Shaped like a Meta campaign: a template, a draft or a submitted one."""

    id: CampaignId
    kind: Kind
    submit_state: SubmitState | None
    submitted_at: SubmittedAt | None
    name: Text | None
    mood: Mood | None
    owner_id: OwnerId | None


class DetailId(Id[int]):
    pass


class Detail(pydantic.BaseModel):
    """At most one per Campaign."""

    id: DetailId
    campaign_id: Annotated[CampaignId, OneToOne()]
    objective: Text | None


def graph() -> EntityGraph:
    return EntityGraph.build(
        [Owner, Campaign, Detail], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()


def compile_campaign(step: steps.Step) -> compiler.CompiledView:
    match compiler.compile_views(
        graph(), [View(name=ViewName.fake(), source=Campaign, step=step)]
    ):
        case Ok(compiled):
            return compiled.views[0]
        case Err(error):
            pytest.fail(f"expected a compiled view, got {error}")


def campaign_diagnostics(
    step: steps.Step,
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    match compiler.compile_views(
        graph(), [View(name=ViewName.fake(), source=Campaign, step=step)]
    ):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            return error.diagnostics


def field_of(view: compiler.CompiledView, name: FieldName) -> ir.RelationField:
    (field,) = (field for field in view.relation.fields if field.name == name)
    return field


def annotation_of(view: compiler.CompiledView, name: FieldName) -> RuntimeAnnotation:
    match field_of(view, name):
        case ir.Field() as field:
            return field.annotation()
        case ir.NestedField(value=lookup):
            return lookup.annotation()


def named(*fields: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in fields)


kind = conditions.field(FieldName("kind"))
submit_state = conditions.field(FieldName("submit_state"))
submitted_at = conditions.field(FieldName("submitted_at"))


# Narrowing by a Filter.


def test_a_template_narrows_its_tags_and_its_unset_timestamp() -> None:
    view = compile_campaign(
        steps.where(
            conditions.and_(
                conditions.eq(kind, Kind.TEMPLATE),
                conditions.is_null(submitted_at),
            )
        )
    )

    assert annotation_of(view, FieldName("kind")) == Literal[Kind.TEMPLATE]
    assert annotation_of(view, FieldName("submitted_at")) == Literal[None]


@pytest.mark.parametrize(
    "condition",
    [
        conditions.ne(submitted_at, None),
        conditions.not_(conditions.is_null(submitted_at)),
    ],
    ids=["ne-none", "not-is-null"],
)
def test_is_not_none_drops_none(condition: ir.Condition) -> None:
    view = compile_campaign(steps.where(condition))

    assert annotation_of(view, FieldName("submitted_at")) == SubmittedAt


def test_an_optional_enum_equal_to_a_member_is_that_member() -> None:
    view = compile_campaign(
        steps.where(conditions.eq(submit_state, SubmitState.SUBMITTED))
    )

    assert (
        annotation_of(view, FieldName("submit_state")) == Literal[SubmitState.SUBMITTED]
    )


def test_and_ed_terms_narrow_however_deeply_they_are_and_ed() -> None:
    view = compile_campaign(
        steps.pipeline(
            steps.where(conditions.eq(kind, Kind.INSTANCE)),
            steps.where(
                conditions.and_(
                    conditions.and_(
                        conditions.eq(submit_state, SubmitState.SUBMITTED),
                        conditions.ne(submitted_at, None),
                    )
                )
            ),
        )
    )

    assert {
        name: annotation_of(view, FieldName(name))
        for name in ("kind", "submit_state", "submitted_at")
    } == {
        "kind": Literal[Kind.INSTANCE],
        "submit_state": Literal[SubmitState.SUBMITTED],
        "submitted_at": SubmittedAt,
    }


def test_narrowing_follows_a_renamed_field() -> None:
    view = compile_campaign(
        steps.pipeline(
            steps.rename(FieldName("kind"), FieldName("tag")),
            steps.where(
                conditions.eq(conditions.field(FieldName("tag")), Kind.TEMPLATE)
            ),
            steps.rename(FieldName("tag"), FieldName("variant")),
        )
    )

    assert annotation_of(view, FieldName("variant")) == Literal[Kind.TEMPLATE]


@pytest.mark.parametrize(
    "condition",
    [
        conditions.or_(
            conditions.eq(kind, Kind.TEMPLATE), conditions.ne(submitted_at, None)
        ),
        conditions.not_(conditions.eq(kind, Kind.TEMPLATE)),
        conditions.ne(kind, Kind.TEMPLATE),
        conditions.eq(conditions.field(FieldName("name")), Text("Launch")),
        conditions.not_(conditions.not_(conditions.is_null(submitted_at))),
        # A member whose name no `Literal` can write.
        conditions.eq(conditions.field(FieldName("mood")), Mood("not ok")),
        # `kind` is not optional: no row passes, and its type stays.
        conditions.is_null(kind),
        # A Reference stays one, so that it can still be resolved.
        conditions.is_null(conditions.field(FieldName("owner_id"))),
    ],
    ids=[
        "or",
        "not",
        "ne",
        "non-enum",
        "double-not",
        "unnameable-member",
        "null-required",
        "null-reference",
    ],
)
def test_a_condition_that_does_not_narrow_keeps_every_type(
    condition: ir.Condition,
) -> None:
    view = compile_campaign(steps.where(condition))

    assert view.relation.fields == compile_campaign(steps.pipeline()).relation.fields


def test_a_reference_filtered_not_none_resolves_to_a_required_lookup() -> None:
    view = compile_campaign(
        steps.pipeline(
            steps.where(conditions.ne(conditions.field(FieldName("owner_id")), None)),
            steps.resolve(named(FieldName("owner_id"))),
        )
    )

    assert annotation_of(view, FieldName("owner")) == Owner


# Requiring.


def test_require_narrows_an_optional_field_and_requires_its_value() -> None:
    view = compile_campaign(steps.require(FieldName("name")))

    field = field_of(view, FieldName("name"))
    assert isinstance(field, ir.Field)
    assert (field.annotation(), field.value) == (
        Text,
        ir.RequiredField(field=FieldName("name")),
    )


def test_require_with_a_select_requires_each_selected_field() -> None:
    view = compile_campaign(
        steps.require(named(FieldName("name"), FieldName("submitted_at")))
    )

    assert (
        annotation_of(view, FieldName("name")),
        annotation_of(view, FieldName("submitted_at")),
    ) == (Text, SubmittedAt)


def owner_lookup() -> ir.Lookup:
    return ir.Lookup(
        reference=ir.SourceField(field=FieldName("owner_id")),
        id_type=OwnerId,
        target=Owner,
        optionality=Optionality.OPTIONAL,
    )


@pytest.mark.parametrize(
    "step",
    [
        steps.pipeline(
            steps.require(FieldName("owner_id")),
            steps.resolve(named(FieldName("owner_id"))),
        ),
        steps.pipeline(
            steps.resolve(named(FieldName("owner_id"))),
            steps.require(FieldName("owner")),
        ),
    ],
    ids=["before-resolve", "after-resolve"],
)
def test_a_required_reference_resolves_to_a_required_lookup(step: steps.Step) -> None:
    view = compile_campaign(step)

    field = field_of(view, FieldName("owner"))
    assert isinstance(field, ir.Field)
    assert (field.annotation(), field.value) == (Owner, owner_lookup().required())


def test_requiring_a_resolved_field_with_a_nested_pipeline_requires_its_lookup() -> (
    None
):
    view = compile_campaign(
        steps.pipeline(
            steps.resolve(named(FieldName("owner_id")), steps.omit(FieldName("id"))),
            steps.require(FieldName("owner")),
        )
    )

    field = field_of(view, FieldName("owner"))
    assert isinstance(field, ir.NestedField)
    assert field.value == owner_lookup().required()


def test_require_in_a_nested_pipeline_requires_the_referenced_entitys_field() -> None:
    view = compile_campaign(
        steps.resolve(
            named(FieldName("owner_id")), steps.require(FieldName("nickname"))
        )
    )

    field = field_of(view, FieldName("owner"))
    assert isinstance(field, ir.NestedField)
    nickname = field.relation.fields[-1]
    assert isinstance(nickname, ir.Field)
    assert (nickname.name, nickname.annotation(), nickname.value) == (
        FieldName("nickname"),
        Text,
        ir.RequiredField(field=FieldName("nickname")),
    )


def test_requiring_a_reversed_field_requires_its_lookup() -> None:
    view = compile_campaign(
        steps.pipeline(steps.reverse(Detail), steps.require(FieldName("detail")))
    )

    field = field_of(view, FieldName("detail"))
    assert isinstance(field, ir.Field)
    assert field.value == ir.ReverseLookup(
        optionality=Optionality.REQUIRED,
        target=Detail,
        via=FieldName("campaign_id"),
        id_type=CampaignId,
    )
    assert field.annotation() == Detail


def test_a_required_field_can_still_be_filtered_on() -> None:
    view = compile_campaign(
        steps.pipeline(
            steps.require(FieldName("submit_state")),
            steps.where(conditions.eq(submit_state, SubmitState.DRAFT)),
        )
    )

    field = field_of(view, FieldName("submit_state"))
    assert isinstance(field, ir.Field)
    assert (field.annotation(), field.value) == (
        Literal[SubmitState.DRAFT],
        ir.RequiredField(field=FieldName("submit_state")),
    )


@pytest.mark.parametrize(
    ("step", "field", "annotation"),
    [
        (steps.require(FieldName("kind")), FieldName("kind"), Kind),
        (
            steps.pipeline(
                steps.require(FieldName("name")), steps.require(FieldName("name"))
            ),
            FieldName("name"),
            Text,
        ),
        (
            steps.pipeline(
                steps.where(conditions.ne(submitted_at, None)),
                steps.require(FieldName("submitted_at")),
            ),
            FieldName("submitted_at"),
            SubmittedAt,
        ),
    ],
    ids=["required", "twice", "filtered"],
)
def test_requiring_a_field_that_is_not_optional_is_a_diagnostic(
    step: steps.Step, field: FieldName, annotation: RuntimeAnnotation
) -> None:
    assert campaign_diagnostics(step) == (
        diagnostics.RequireOnRequiredField(
            view=ViewName.fake(),
            source=ir.SourceName("Campaign"),
            field=field,
            annotation=annotation,
        ),
    )


def test_requiring_an_added_field_is_a_diagnostic() -> None:
    assert campaign_diagnostics(
        steps.pipeline(
            steps.add(FieldName("note"), Text | None, default=None),
            steps.require(FieldName("note")),
        )
    ) == (diagnostics.RequireAddedField(view=ViewName.fake(), field=FieldName("note")),)


def test_requiring_an_unknown_field_is_a_diagnostic_and_later_steps_are_checked() -> (
    None
):
    reported = campaign_diagnostics(
        steps.pipeline(
            steps.require(FieldName("missing")), steps.require(FieldName("kind"))
        )
    )

    assert [type(diagnostic) for diagnostic in reported] == [
        diagnostics.UnknownField,
        diagnostics.RequireOnRequiredField,
    ]


@pytest.mark.parametrize(
    "step",
    [
        steps.require(FieldName("name")),
        steps.resolve(
            named(FieldName("owner_id")), steps.require(FieldName("nickname"))
        ),
    ],
    ids=["own", "nested"],
)
def test_require_in_a_payload_is_a_diagnostic(step: steps.Step) -> None:
    payload = PayloadName("CampaignUpdate")

    match compiler.compile_catalog(
        graph(), [], [Payload(name=payload, source=Campaign, step=step)]
    ):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            assert error.diagnostics == (diagnostics.RequireInPayload(view=payload),)


def test_building_a_require_checks_nothing() -> None:
    root = ir.FromSource(source=Campaign)

    assert steps.require(FieldName("missing"))(root) == ir.Require(
        input=root, fields=FieldName("missing")
    )


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            diagnostics.RequireInPayload(view=PayloadName("CampaignUpdate")),
            (
                "CampaignUpdate: a Payload cannot `require` a field, because it is "
                "a write model with no values to require."
            ),
        ),
        (
            diagnostics.RequireOnRequiredField(
                view=ViewName("Submitted"),
                source=ir.SourceName("Campaign"),
                field=FieldName("kind"),
                annotation=int,
            ),
            (
                "Submitted: cannot require Campaign.kind, because its type "
                "<class 'int'> is already required."
            ),
        ),
        (
            diagnostics.RequireAddedField(
                view=ViewName("Submitted"), field=FieldName("note")
            ),
            (
                "Submitted: cannot require 'note', because `add` declared it, so it "
                "has no value to require."
            ),
        ),
    ],
    ids=["payload", "required", "added"],
)
def test_a_require_diagnostic_message_names_the_offending_values(
    diagnostic: diagnostics.CompileDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


def detail_lookup() -> ir.ReverseLookup:
    return ir.ReverseLookup(
        optionality=Optionality.OPTIONAL,
        target=Detail,
        via=FieldName("campaign_id"),
        id_type=CampaignId,
    )


def test_requiring_a_lifted_field_requires_the_entity_and_its_field() -> None:
    view = compile_campaign(
        steps.pipeline(
            steps.reverse(Detail, steps.project([FieldName("objective")])),
            steps.flatten(FieldName("detail")),
            steps.require(FieldName("objective")),
        )
    )

    field = field_of(view, FieldName("objective"))
    assert isinstance(field, ir.Field)
    assert (field.annotation(), field.value) == (
        Text,
        ir.Lifted(
            through=(detail_lookup().required(),),
            field=ir.RequiredField(field=FieldName("objective")),
        ),
    )


def test_requiring_a_field_lifted_through_an_optional_entity_requires_the_entity() -> (
    None
):
    view = compile_campaign(
        steps.pipeline(
            steps.reverse(
                Detail,
                steps.project([FieldName("id")]),
                steps.rename(FieldName("id"), FieldName("detail_id")),
            ),
            steps.flatten(FieldName("detail")),
            steps.require(FieldName("detail_id")),
        )
    )

    field = field_of(view, FieldName("detail_id"))
    assert isinstance(field, ir.Field)
    assert (field.annotation(), field.value) == (
        DetailId,
        ir.Lifted(
            through=(detail_lookup().required(),),
            field=ir.SourceField(field=FieldName("id")),
        ),
    )


def test_requiring_a_field_a_where_keeps_only_none_in_is_a_diagnostic() -> None:
    assert campaign_diagnostics(
        steps.pipeline(
            steps.where(conditions.eq(submitted_at, None)),
            steps.require(FieldName("submitted_at")),
        )
    ) == (
        diagnostics.RequireOnAlwaysNoneField(
            view=ViewName.fake(),
            source=ir.SourceName("Campaign"),
            field=FieldName("submitted_at"),
        ),
    )


def test_the_always_none_requirement_message_names_the_field() -> None:
    diagnostic = diagnostics.RequireOnAlwaysNoneField(
        view=ViewName("Template"),
        source=ir.SourceName("Campaign"),
        field=FieldName("submitted_at"),
    )

    assert diagnostic.describe().root == (
        "Template: cannot require Campaign.submitted_at, because a `where` keeps "
        "only the rows where it is None, so the requirement is never met."
    )


@pytest.mark.parametrize(
    "later",
    [
        conditions.eq(kind, Kind.INSTANCE),
        conditions.eq(submitted_at, None),
        conditions.eq(kind, conditions.field(FieldName("kind"))),
    ],
    ids=["other-member", "none-after-not-none", "same-field"],
)
def test_a_condition_is_checked_against_the_type_the_source_declares(
    later: ir.Condition,
) -> None:
    view = compile_campaign(
        steps.pipeline(
            steps.where(
                conditions.and_(
                    conditions.eq(kind, Kind.TEMPLATE),
                    conditions.ne(submitted_at, None),
                )
            ),
            steps.where(later),
        )
    )

    assert len(view.relation.filters) == len((kind, later))


def test_a_condition_on_a_required_field_may_test_for_none() -> None:
    view = compile_campaign(
        steps.pipeline(
            steps.require(FieldName("name")),
            steps.where(conditions.is_null(conditions.field(FieldName("name")))),
        )
    )

    assert view.relation.filters == (
        ir.FilterIsNull(field=ir.SourceField(field=FieldName("name"))),
    )


def unchanged(label: Text | None) -> Text | None:
    return label


class Constrained(pydantic.BaseModel):
    id: CampaignId
    label: Annotated[Text | None, pydantic.AfterValidator(unchanged)]


@pytest.mark.parametrize(
    "step",
    [
        steps.require(FieldName("label")),
        steps.where(conditions.ne(conditions.field(FieldName("label")), None)),
    ],
    ids=["require", "where"],
)
def test_narrowing_keeps_the_fields_constraints(step: steps.Step) -> None:
    (compiled,) = (
        compiler.compile_views(
            graph(), [View(name=ViewName.fake(), source=Constrained, step=step)]
        )
        .unwrap()
        .views
    )
    (_, label) = compiled.relation.fields

    assert isinstance(label, ir.Field)
    assert label.info.metadata == Constrained.model_fields["label"].metadata != []


type MaybeObjective = Text | None


class AliasedDetail(pydantic.BaseModel):
    """At most one per Campaign, with an optional field typed by an alias."""

    id: DetailId
    campaign_id: Annotated[CampaignId, OneToOne()]
    objective: MaybeObjective


def test_requiring_a_lifted_field_typed_by_an_optional_alias_requires_it() -> None:
    aliased = EntityGraph.build(
        [Owner, Campaign, AliasedDetail], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()
    (compiled,) = (
        compiler.compile_views(
            aliased,
            [
                View(
                    name=ViewName.fake(),
                    source=Campaign,
                    step=steps.pipeline(
                        steps.reverse(
                            AliasedDetail, steps.project([FieldName("objective")])
                        ),
                        steps.flatten(FieldName("aliased_detail")),
                        steps.require(FieldName("objective")),
                    ),
                )
            ],
        )
        .unwrap()
        .views
    )
    (objective,) = (
        field
        for field in compiled.relation.fields
        if field.name == FieldName("objective")
    )

    assert isinstance(objective, ir.Field)
    assert isinstance(objective.value, ir.Lifted)
    assert objective.value.field == ir.RequiredField(field=FieldName("objective"))
