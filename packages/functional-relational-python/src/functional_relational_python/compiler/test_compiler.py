import enum
import functools
import types
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Annotated, Literal

import pydantic
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from pydantic import MISSING
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler, diagnostics, model_names
from functional_relational_python.entity_graph.entity_graph import (
    Cardinality,
    EntityGraph,
    FieldName,
    Optionality,
    RuntimeAnnotation,
)
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    IdType,
    OneToOne,
    OneToOneMarker,
)
from functional_relational_python.pipeline import conditions, ir, steps, test_ir
from functional_relational_python.pipeline.payloads import Payload, PayloadName
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.support.diagnostic import DiagnosticsError


class Text(pydantic.RootModel[str], frozen=True):
    pass


class Company(pydantic.BaseModel):
    name: Text
    secret: Text
    revenue: Text


def empty_graph() -> EntityGraph:
    # The Sources here are not Entities; compile does not require them to be.
    return EntityGraph.build([], IdBase.fake(), OneToOneMarker.fake()).unwrap()


def field_names(view: compiler.CompiledView) -> list[FieldName]:
    return [field.name for field in view.relation.fields]


def names(*fields: Text) -> list[FieldName]:
    return [FieldName(field.root) for field in fields]


def compile_one(step: steps.Step) -> compiler.CompiledView:
    match compiler.compile_views(
        empty_graph(), [View(name=ViewName.fake(), source=Company, step=step)]
    ):
        case Ok(compiled):
            return compiled.views[0]
        case Err(error):
            pytest.fail(f"expected a compiled view, got {error}")


def diagnostics_of(views: list[View]) -> tuple[diagnostics.CompileDiagnostic, ...]:
    match compiler.compile_views(empty_graph(), views):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            return error.diagnostics


def test_omit_removes_the_field_from_the_source() -> None:
    view = compile_one(steps.omit(FieldName("secret")))

    assert field_names(view) == names(Text("name"), Text("revenue"))


def test_project_keeps_only_the_listed_fields_in_source_order() -> None:
    view = compile_one(steps.project([FieldName("revenue"), FieldName("name")]))

    assert field_names(view) == names(Text("name"), Text("revenue"))


def test_each_unknown_projected_field_is_a_diagnostic() -> None:
    unknown = [FieldName("revnue"), FieldName("nme")]
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.project([FieldName("secret"), *unknown]),
    )

    assert diagnostics_of([view]) == tuple(
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Company"),
            field=field,
            valid=tuple(map(FieldName, Company.model_fields)),
        )
        for field in unknown
    )


def test_steps_after_a_defective_project_are_checked_against_its_known_fields() -> None:
    unknown = FieldName("revnue")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.pipeline(
            steps.project([FieldName("name"), unknown]),
            steps.omit(FieldName("secret")),
        ),
    )

    assert diagnostics_of([view]) == (
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Company"),
            field=unknown,
            valid=tuple(map(FieldName, Company.model_fields)),
        ),
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Company"),
            field=FieldName("secret"),
            valid=(FieldName("name"),),
        ),
    )


@pytest.mark.parametrize(
    "step",
    [
        steps.pipeline(
            steps.omit(FieldName("secret")), steps.omit(FieldName("revenue"))
        ),
        steps.pipeline(
            steps.pipeline(steps.omit(FieldName("secret"))),
            steps.omit(FieldName("revenue")),
        ),
        steps.pipeline(
            steps.omit(FieldName("secret")),
            steps.pipeline(steps.pipeline(), steps.omit(FieldName("revenue"))),
        ),
    ],
    ids=["flat", "nested-first", "nested-last-with-empty"],
)
def test_nested_pipelines_compile_like_flat_ones(step: steps.Step) -> None:
    view = compile_one(step)

    assert field_names(view) == names(Text("name"))


def test_an_empty_pipeline_keeps_every_field() -> None:
    view = compile_one(steps.pipeline())

    assert field_names(view) == names(*map(Text, Company.model_fields))


def test_a_user_defined_step_composes_like_a_built_in_one() -> None:
    def without_financials(node: ir.Node) -> ir.Node:
        return steps.omit(FieldName("revenue"))(node)

    view = compile_one(
        steps.pipeline(without_financials, steps.omit(FieldName("secret")))
    )

    assert field_names(view) == names(Text("name"))


def test_an_unknown_field_names_the_source_the_field_and_the_valid_names() -> None:
    unknown = FieldName("revnue")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.pipeline(steps.omit(FieldName("secret")), steps.omit(unknown)),
    )

    assert diagnostics_of([view]) == (
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Company"),
            field=unknown,
            valid=(FieldName("name"), FieldName("revenue")),
        ),
    )


def test_the_unknown_field_message_contains_the_offending_values() -> None:
    message = diagnostics.UnknownField(
        view=ViewName("CompanySummary"),
        source=ir.SourceName("Company"),
        field=FieldName("revnue"),
        valid=(FieldName("name"), FieldName("revenue")),
    ).describe()

    assert message.root == (
        "CompanySummary: Company has no field 'revnue'. "
        "Valid fields: 'name', 'revenue'."
    )


def test_compiled_views_keep_their_explicit_names_in_order() -> None:
    names = [ViewName("CompanyPublic"), ViewName("CompanyInternal")]

    match compiler.compile_views(
        empty_graph(),
        [View(name=name, source=Company, step=steps.pipeline()) for name in names],
    ):
        case Ok(compiled):
            assert [view.name for view in compiled.views] == names
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def test_building_a_pipeline_checks_nothing() -> None:
    unknown = FieldName("does_not_exist")
    root = ir.FromSource(source=Company)

    node = steps.pipeline(steps.omit(unknown), steps.rename(unknown, unknown))(root)

    assert node == ir.Rename(
        input=ir.Omit(input=root, fields=unknown), field=unknown, new=unknown
    )


def test_rename_keeps_the_field_in_place_with_its_type() -> None:
    view = compile_one(steps.rename(FieldName("secret"), FieldName("code")))

    assert view.relation.fields == tuple(
        ir.Field(
            name=FieldName("code") if name == "secret" else FieldName(name),
            info=info,
            value=ir.SourceField(field=FieldName(name)),
            default=ir.NoDefault(),
        )
        for name, info in Company.model_fields.items()
    )


def test_renaming_onto_an_existing_field_is_a_diagnostic() -> None:
    existing = FieldName("name")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.rename(FieldName("secret"), existing),
    )

    assert diagnostics_of([view]) == (
        diagnostics.RenameOntoExistingField(
            view=view.name,
            renamed=FieldName("secret"),
            field=existing,
        ),
    )


def test_renaming_a_field_to_its_own_name_changes_nothing() -> None:
    field = FieldName("secret")

    view = compile_one(steps.pipeline(steps.rename(field, field), steps.omit(field)))

    assert field_names(view) == names(Text("name"), Text("revenue"))


def test_a_renamed_field_can_be_omitted_by_its_new_name() -> None:
    new = FieldName("code")

    view = compile_one(
        steps.pipeline(steps.rename(FieldName("secret"), new), steps.omit(new))
    )

    assert field_names(view) == names(Text("name"), Text("revenue"))


@pytest.mark.parametrize(
    "later",
    [
        steps.omit(FieldName("code")),
        steps.rename(FieldName("code"), FieldName("other")),
    ],
    ids=["omit", "rename"],
)
def test_renaming_an_unknown_field_is_one_diagnostic_even_if_later_steps_use_it(
    later: steps.Step,
) -> None:
    unknown = FieldName("secrt")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.pipeline(steps.rename(unknown, FieldName("code")), later),
    )

    assert diagnostics_of([view]) == (
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Company"),
            field=unknown,
            valid=tuple(map(FieldName, Company.model_fields)),
        ),
    )


def test_renaming_an_unknown_field_onto_an_existing_one_reports_both() -> None:
    unknown = FieldName("secrt")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.rename(unknown, FieldName("name")),
    )

    assert diagnostics_of([view]) == (
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Company"),
            field=unknown,
            valid=tuple(map(FieldName, Company.model_fields)),
        ),
        diagnostics.RenameOntoExistingField(
            view=view.name,
            renamed=unknown,
            field=FieldName("name"),
        ),
    )


def test_the_rename_onto_existing_field_message_contains_the_offending_values() -> None:
    message = diagnostics.RenameOntoExistingField(
        view=ViewName("CompanySummary"),
        renamed=FieldName("secret"),
        field=FieldName("name"),
    ).describe()

    assert message.root == (
        "CompanySummary: cannot rename 'secret' to 'name', because it "
        "already has a field 'name'."
    )


def prefixed(prefix: Text) -> ir.FieldRenaming:
    return lambda field: FieldName(f"{prefix.root}_{field.root}")


def test_omit_with_a_select_removes_every_selected_field() -> None:
    view = compile_one(steps.omit(named(FieldName("secret"), FieldName("revenue"))))

    assert field_names(view) == names(Text("name"))


def test_omit_with_a_select_that_selects_nothing_changes_nothing() -> None:
    view = compile_one(steps.omit(named(FieldName("does_not_exist"))))

    assert field_names(view) == names(*map(Text, Company.model_fields))


def test_rename_with_a_select_renames_each_selected_field_in_place() -> None:
    view = compile_one(
        steps.rename(
            named(FieldName("name"), FieldName("revenue")), prefixed(Text("public"))
        )
    )

    assert values(view) == {
        FieldName("public_name"): ir.SourceField(field=FieldName("name")),
        FieldName("secret"): ir.SourceField(field=FieldName("secret")),
        FieldName("public_revenue"): ir.SourceField(field=FieldName("revenue")),
    }
    assert field_names(view) == names(
        Text("public_name"), Text("secret"), Text("public_revenue")
    )


def test_rename_with_a_select_renames_all_fields_at_once_so_they_can_swap() -> None:
    first, second = FieldName("name"), FieldName("secret")

    view = compile_one(
        steps.rename(
            named(first, second), lambda field: second if field == first else first
        )
    )

    assert values(view) == {
        second: ir.SourceField(field=first),
        first: ir.SourceField(field=second),
        FieldName("revenue"): ir.SourceField(field=FieldName("revenue")),
    }


def test_rename_with_a_select_onto_an_unselected_field_is_a_diagnostic() -> None:
    existing = FieldName("name")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.rename(named(FieldName("secret")), lambda _field: existing),
    )

    assert diagnostics_of([view]) == (
        diagnostics.RenameOntoExistingField(
            view=view.name, renamed=FieldName("secret"), field=existing
        ),
    )


def test_a_selected_field_renamed_to_its_own_name_keeps_it() -> None:
    kept, renamed = FieldName("secret"), FieldName("name")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.rename(named(renamed, kept), lambda _field: kept),
    )

    assert diagnostics_of([view]) == (
        diagnostics.RenameOntoExistingField(
            view=view.name, renamed=renamed, field=kept
        ),
    )


def test_renaming_two_selected_fields_onto_one_name_reports_the_second() -> None:
    new = FieldName("code")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.pipeline(
            steps.rename(
                named(FieldName("secret"), FieldName("revenue")), lambda _field: new
            ),
            # The first renamed field keeps the name, so a later step finds it.
            steps.omit(new),
        ),
    )

    assert diagnostics_of([view]) == (
        diagnostics.RenameOntoExistingField(
            view=view.name, renamed=FieldName("revenue"), field=new
        ),
    )


def test_rename_with_a_select_reports_each_invalid_new_name() -> None:
    selected = (FieldName("name"), FieldName("secret"))
    private = prefixed(Text("_"))
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.rename(named(*selected), private),
    )

    assert diagnostics_of([view]) == tuple(
        model_names.InvalidFieldName(
            view=view.name, source=ir.SourceName("Company"), field=private(field)
        )
        for field in selected
    )


def test_building_steps_with_a_select_checks_nothing() -> None:
    root = ir.FromSource(source=Company)
    selection = named(FieldName("does_not_exist"))
    renaming = prefixed(Text("public"))

    node = steps.pipeline(steps.omit(selection), steps.rename(selection, renaming))(
        root
    )

    assert node == ir.RenameSelected(
        input=ir.Omit(input=root, fields=selection), fields=selection, new=renaming
    )


class Colour(enum.Enum):
    RED = "red"


def defaults(view: compiler.CompiledView) -> dict[FieldName, ir.DefaultValue]:
    """The defaults the View's fields have, by name; required fields are left out."""
    return {
        field.name: field.default
        for field in view.relation.fields
        if not isinstance(field.default, ir.NoDefault)
    }


@pytest.mark.parametrize(
    ("value", "default"),
    [
        (None, ir.ScalarDefault(None)),
        (pydantic.MISSING, ir.MissingDefault()),
        (3, ir.ScalarDefault(3)),
        (0.5, ir.ScalarDefault(0.5)),
        ("shop", ir.ScalarDefault("shop")),
        (True, ir.ScalarDefault(root=True)),
        ([], ir.EmptyListDefault()),
        ({}, ir.EmptyDictDefault()),
        (Colour.RED, ir.EnumDefault(member=Colour.RED)),
    ],
    ids=["none", "missing", "int", "float", "str", "bool", "list", "dict", "enum"],
)
def test_default_sets_a_literal_as_the_named_fields_default(
    value: ir.UncheckedDefault, default: ir.DefaultValue
) -> None:
    field = FieldName("secret")

    view = compile_one(steps.default(field, value))

    assert defaults(view) == {field: default}


def test_default_with_a_select_sets_each_selected_fields_default() -> None:
    selected = (FieldName("name"), FieldName("revenue"))

    view = compile_one(steps.default(named(*selected), None))

    assert defaults(view) == dict.fromkeys(selected, ir.ScalarDefault(None))


def test_a_later_default_replaces_an_earlier_one_and_a_rename_keeps_it() -> None:
    old, new = FieldName("secret"), FieldName("hidden")

    view = compile_one(
        steps.pipeline(
            steps.default(old, None),
            steps.default(old, "redacted"),
            steps.rename(old, new),
        )
    )

    assert defaults(view) == {new: ir.ScalarDefault("redacted")}


class Count(int):
    pass


class Permission(enum.Flag):
    READ = enum.auto()
    WRITE = enum.auto()


@pytest.mark.parametrize(
    "value",
    [
        object(),
        (),
        [None],
        {"key": None},
        float("nan"),
        Count(1),
        Text("shop"),
        Permission.READ | Permission.WRITE,
        Permission(0),
    ],
    ids=[
        "object",
        "tuple",
        "non-empty list",
        "non-empty dict",
        "nan",
        "int subclass",
        "root model",
        "combined flag",
        "empty flag",
    ],
)
def test_a_default_that_is_not_a_literal_is_a_diagnostic(
    value: ir.UncheckedDefault,
) -> None:
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.default(FieldName("secret"), value),
    )

    assert diagnostics_of([view]) == (
        diagnostics.NonLiteralDefault(
            view=view.name, value=diagnostics.DefaultText.of(value)
        ),
    )


def test_a_default_for_an_unknown_field_that_is_not_a_literal_reports_both() -> None:
    unknown, value = FieldName("revnue"), object()
    view = View(
        name=ViewName.fake(), source=Company, step=steps.default(unknown, value)
    )

    assert diagnostics_of([view]) == (
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Company"),
            field=unknown,
            valid=tuple(map(FieldName, Company.model_fields)),
        ),
        diagnostics.NonLiteralDefault(
            view=view.name, value=diagnostics.DefaultText.of(value)
        ),
    )


def test_the_non_literal_default_message_contains_the_value() -> None:
    value = (Text("shop"),)
    message = diagnostics.NonLiteralDefault(
        view=ViewName.fake(), value=diagnostics.DefaultText.of(value)
    ).describe()

    assert repr(value) in message.root


def test_building_a_default_checks_nothing() -> None:
    root = ir.FromSource(source=Company)
    unknown, value = FieldName("does_not_exist"), object()

    node = steps.default(unknown, value)(root)

    assert node == ir.Default(input=root, fields=unknown, value=value)


class AddedField(pydantic.BaseModel, frozen=True):
    """An added field as the compiler sees it: name, type, value and default."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: FieldName
    annotation: RuntimeAnnotation
    value: ir.Value
    default: ir.DefaultValue | ir.NoDefault

    @staticmethod
    def of(field: ir.RelationField) -> AddedField:
        match field:
            case ir.Field():
                return AddedField(
                    name=field.name,
                    annotation=field.annotation(),
                    value=field.value,
                    default=field.default,
                )
            case ir.NestedField():
                pytest.fail(f"expected an added field, got {field}")


def test_add_appends_a_field_with_no_value_from_the_source() -> None:
    kind = FieldName("kind")

    view = compile_one(steps.add(kind, Literal["company"], default="company"))

    assert field_names(view) == [*map(FieldName, Company.model_fields), kind]
    assert AddedField.of(view.relation.fields[-1]) == AddedField(
        name=kind,
        annotation=Literal["company"],
        value=ir.Added(),
        default=ir.ScalarDefault("company"),
    )


def test_a_later_default_replaces_an_added_fields_default() -> None:
    kind = FieldName("kind")

    view = compile_one(
        steps.pipeline(
            steps.add(kind, str | None, default="company"), steps.default(kind, None)
        )
    )

    assert defaults(view) == {kind: ir.ScalarDefault(None)}


def test_a_view_that_adds_a_field_without_a_default_is_a_diagnostic() -> None:
    kind = FieldName("kind")
    view = View(
        name=ViewName.fake(), source=Company, step=steps.add(kind, Literal["company"])
    )

    assert diagnostics_of([view]) == (
        diagnostics.AddWithoutDefault(view=view.name, field=kind),
    )


def test_a_payload_may_add_a_field_without_a_default() -> None:
    kind = FieldName("kind")
    payload = Payload(
        name=PayloadName.fake(), source=Company, step=steps.add(kind, Literal["a"])
    )

    (compiled,) = compile_catalog([], [payload]).unwrap().payloads

    assert AddedField.of(compiled.relation.fields[-1]) == AddedField(
        name=kind, annotation=Literal["a"], value=ir.Added(), default=ir.NoDefault()
    )


def test_adding_a_field_the_relation_already_has_is_a_diagnostic() -> None:
    existing = FieldName("name")
    view = View(
        name=ViewName.fake(), source=Company, step=steps.add(existing, str, default="")
    )

    assert diagnostics_of([view]) == (
        diagnostics.AddOntoExistingField(view=view.name, field=existing),
    )


def test_steps_after_a_defective_add_find_the_added_field() -> None:
    kind, value = FieldName("kind"), object()
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.pipeline(
            steps.add(kind, str, default=value), steps.rename(kind, FieldName("tag"))
        ),
    )

    assert diagnostics_of([view]) == (
        diagnostics.NonLiteralDefault(
            view=view.name, value=diagnostics.DefaultText.of(value)
        ),
    )


def test_the_add_diagnostic_messages_contain_the_field() -> None:
    field = FieldName("kind")

    messages = [
        diagnostics.AddOntoExistingField(view=ViewName.fake(), field=field).describe(),
        diagnostics.AddWithoutDefault(view=ViewName.fake(), field=field).describe(),
    ]

    assert all(field.root in message.root for message in messages)


def test_building_an_add_checks_nothing() -> None:
    root = ir.FromSource(source=Company)
    existing, value = FieldName("name"), object()

    node = steps.add(existing, str, default=value)(root)

    match node:
        case ir.Add(input=input_node, name=name, info=info, default=default):
            assert (input_node, name, info.annotation, default) == (
                root,
                existing,
                str,
                value,
            )
        case _:
            pytest.fail(f"expected an Add, got {node}")


# The prefix keeps these from ever being a field of `Company` or `Wide`.
unknown_field_name = st.from_regex(r"unknown_[a-z]{1,6}", fullmatch=True)


class Wide(pydantic.BaseModel):
    alpha: Text
    beta: Text
    gamma: Text
    delta: Text
    epsilon: Text


class ViewsWithDefects(pydantic.BaseModel, frozen=True):
    views: tuple[View, ...]
    unknown_fields: tuple[FieldName, ...]
    defective_views: frozenset[ViewName]


@st.composite
def views_with_defects(draw: st.DrawFn) -> ViewsWithDefects:
    """Views with known unknown-field omits.

    Every unknown name is independent of the others: it is never a field of the
    Source, and valid omits use each field at most once.
    """
    views: list[View] = []
    unknown_fields: list[FieldName] = []
    defective_views: set[ViewName] = set()
    for index in range(draw(st.integers(min_value=0, max_value=4))):
        source = draw(st.sampled_from([Company, Wide]))
        valid = draw(st.lists(st.sampled_from(list(source.model_fields)), unique=True))
        unknown = draw(st.lists(unknown_field_name, max_size=3))
        omits = draw(st.permutations([*valid, *unknown]))
        name = ViewName(f"View{index}")
        unknown_fields.extend(map(FieldName, unknown))
        if unknown:
            defective_views.add(name)
        views.append(
            View(
                name=name,
                source=source,
                step=steps.pipeline(*(steps.omit(FieldName(f)) for f in omits)),
            )
        )
    return ViewsWithDefects(
        views=tuple(views),
        unknown_fields=tuple(unknown_fields),
        defective_views=frozenset(defective_views),
    )


@given(views_with_defects())
def test_n_independent_defects_give_exactly_n_diagnostics(
    case: ViewsWithDefects,
) -> None:
    match compiler.compile_views(empty_graph(), case.views):
        case Ok(_):
            assert case.unknown_fields == ()
        case Err(error):
            unknown = [
                d for d in error.diagnostics if isinstance(d, diagnostics.UnknownField)
            ]
            assert len(unknown) == len(error.diagnostics)
            assert sorted(d.field.root for d in unknown) == sorted(
                field.root for field in case.unknown_fields
            )


@given(views_with_defects())
def test_only_defective_views_are_poisoned_and_each_names_its_diagnostics(
    case: ViewsWithDefects,
) -> None:
    checked = compiler.check(empty_graph(), case.views)

    poisoned = [view for view in checked if isinstance(view, compiler.PoisonedView)]
    assert {view.name for view in poisoned} == case.defective_views
    for view in poisoned:
        assert view.diagnostics
        assert all(d.view == view.name for d in view.diagnostics)


class Projection(pydantic.BaseModel, frozen=True):
    source: type[pydantic.BaseModel]
    kept: tuple[FieldName, ...]


@st.composite
def projections(draw: st.DrawFn) -> Projection:
    source = draw(st.sampled_from([Company, Wide]))
    kept = draw(st.lists(st.sampled_from(list(source.model_fields)), unique=True))
    return Projection(source=source, kept=tuple(map(FieldName, kept)))


@given(projections())
def test_project_equals_omitting_every_other_field(case: Projection) -> None:
    def compile_relation(step: steps.Step) -> ir.Relation:
        view = View(name=ViewName.fake(), source=case.source, step=step)
        match compiler.compile_views(empty_graph(), [view]):
            case Ok(compiled):
                return compiled.views[0].relation
            case Err(error):
                pytest.fail(f"expected a compiled view, got {error}")

    others = [
        FieldName(name)
        for name in case.source.model_fields
        if FieldName(name) not in case.kept
    ]

    assert compile_relation(steps.project(case.kept)) == compile_relation(
        steps.pipeline(*map(steps.omit, others))
    )


# Resolving References against `hr_graph()`.


class DepartmentId(Id[int]):
    pass


class Department(pydantic.BaseModel):
    id: DepartmentId
    name: Text


class EmployeeId(Id[int]):
    pass


class Employee(pydantic.BaseModel):
    id: EmployeeId
    name: Text
    department_id: DepartmentId
    mentor_id: EmployeeId | None


class Clash(pydantic.BaseModel):
    """`department_id` would resolve onto the existing `department`."""

    department: Text
    department_id: DepartmentId


class ProjectId(Id[int]):
    """The primary key of no Entity in `hr_graph()`."""


class Memo(pydantic.BaseModel):
    """A Source that is not an Entity, with a Reference that resolves to nothing."""

    project_id: ProjectId
    department_id: DepartmentId


def hr_graph() -> EntityGraph:
    return EntityGraph.build(
        [Department, Employee], IdBase.fake(), OneToOneMarker.fake()
    ).unwrap()


def named(*fields: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in fields)


def is_lifted_reference(
    field: ir.RelationField, references: Collection[FieldName], prefix: Text
) -> bool:
    """Whether `field` is one of `references`, lifted with `prefix` by a flatten
    straight out of the Entity that has them."""
    match field:
        case ir.Field(value=ir.Lifted(through=(_,), field=read)):
            return read.field in references and field.name.root.startswith(
                f"{prefix.root}_"
            )
        case ir.Field() | ir.NestedField():
            return False


def compile_resolving(
    source: type[pydantic.BaseModel], step: steps.Step
) -> compiler.CompiledView:
    match compiler.compile_views(
        hr_graph(), [View(name=ViewName.fake(), source=source, step=step)]
    ):
        case Ok(compiled):
            return compiled.views[0]
        case Err(error):
            pytest.fail(f"expected a compiled view, got {error}")


def resolve_diagnostics(
    source: type[pydantic.BaseModel], step: steps.Step
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    match compiler.compile_views(
        hr_graph(), [View(name=ViewName.fake(), source=source, step=step)]
    ):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            return error.diagnostics


def values(view: compiler.CompiledView) -> dict[FieldName, ir.Value]:
    """The values of the View's fields, by name, for asserting what resolved."""
    return {field.name: field.value for field in view.relation.fields}


def test_hydrate_resolves_every_reference_and_drops_the_id_suffix() -> None:
    view = compile_resolving(Employee, steps.hydrate())

    assert [
        (field.name, field.info.annotation)
        for field in view.relation.fields
        if isinstance(field, ir.Field)
    ] == [
        (FieldName("id"), EmployeeId),
        (FieldName("name"), Text),
        (FieldName("department"), Department),
        (FieldName("mentor"), Employee | None),
    ]


class AppId(pydantic.RootModel[int], frozen=True):
    """An ID base an application defines, rather than subclassing frp's `Id`."""


class TeamId(AppId):
    pass


class Team(pydantic.BaseModel):
    id: TeamId


class MemberId(AppId):
    pass


class Member(pydantic.BaseModel):
    id: MemberId
    team_id: TeamId
    department_id: DepartmentId


def test_hydrate_resolves_the_references_the_id_base_decides() -> None:
    graph = EntityGraph.build(
        [Team, Member], IdBase(AppId), OneToOneMarker.fake()
    ).unwrap()

    match compiler.compile_views(
        graph, [View(name=ViewName.fake(), source=Member, step=steps.hydrate())]
    ):
        case Ok(compiled):
            pass
        case Err(error):
            pytest.fail(f"expected a compiled view, got {error}")

    assert compiled.views[0].relation.field_names() == (
        FieldName("id"),
        FieldName("team"),
        FieldName("department_id"),
    )


def test_a_reference_without_the_id_suffix_keeps_its_name() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            steps.rename(FieldName("department_id"), FieldName("team")),
            steps.resolve(named(FieldName("team"))),
        ),
    )

    assert values(view)[FieldName("team")] == ir.Lookup(
        reference=ir.SourceField(field=FieldName("department_id")),
        id_type=DepartmentId,
        target=Department,
        optionality=Optionality.REQUIRED,
    )


def test_resolve_leaves_unselected_references_alone() -> None:
    view = compile_resolving(Employee, steps.resolve(named(FieldName("mentor_id"))))

    assert field_names(view) == names(
        Text("id"), Text("name"), Text("department_id"), Text("mentor")
    )


@pytest.mark.parametrize(
    "then", [(), (steps.project([FieldName("name")]),)], ids=["entity", "nested"]
)
def test_a_resolved_reference_loses_its_default(then: tuple[steps.Step, ...]) -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            steps.default(FieldName("mentor_id"), None),
            steps.resolve(named(FieldName("mentor_id")), *then),
        ),
    )

    assert defaults(view) == {}


def department_lookup() -> ir.Lookup:
    """How an Employee's `department_id` is resolved."""
    return ir.Lookup(
        reference=ir.SourceField(field=FieldName("department_id")),
        id_type=DepartmentId,
        target=Department,
        optionality=Optionality.REQUIRED,
    )


def test_hydrate_leaves_a_renamed_primary_key_alone() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            steps.project([FieldName("id"), FieldName("department_id")]),
            steps.rename(FieldName("id"), FieldName("key")),
            steps.hydrate(),
        ),
    )

    assert values(view) == {
        FieldName("key"): ir.SourceField(field=FieldName("id")),
        FieldName("department"): department_lookup(),
    }


def test_hydrate_resolves_a_reference_renamed_to_id() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            steps.project([FieldName("department_id")]),
            steps.rename(FieldName("department_id"), FieldName("id")),
            steps.hydrate(),
        ),
    )

    assert values(view) == {FieldName("id"): department_lookup()}


def test_a_nested_pipeline_types_the_field_by_its_relation() -> None:
    view = compile_resolving(
        Employee,
        steps.resolve(
            named(FieldName("mentor_id")), steps.omit(FieldName("mentor_id"))
        ),
    )

    match view.relation.fields[-1]:
        case ir.NestedField(name=name, relation=relation) as nested:
            assert (name, nested.optionality(), relation.field_names()) == (
                FieldName("mentor"),
                Optionality.OPTIONAL,
                (FieldName("id"), FieldName("name"), FieldName("department_id")),
            )
        case field:
            pytest.fail(f"expected a nested field, got {field}")


def test_an_empty_nested_pipeline_resolves_to_the_entity_itself() -> None:
    nested = compile_resolving(
        Employee, steps.resolve(named(FieldName("department_id")), steps.pipeline())
    )
    plain = compile_resolving(
        Employee, steps.resolve(named(FieldName("department_id")))
    )

    # FieldInfo compares by identity, so compare what each field holds instead.
    assert [(type(f), f.name, f.value) for f in nested.relation.fields] == [
        (type(f), f.name, f.value) for f in plain.relation.fields
    ]


def test_resolving_onto_an_existing_field_names_the_source_and_both_fields() -> None:
    assert resolve_diagnostics(Clash, steps.hydrate()) == (
        diagnostics.ResolvedNameCollision(
            view=ViewName.fake(),
            source=ir.SourceName("Clash"),
            field=FieldName("department_id"),
            existing=FieldName("department"),
        ),
    )


def test_resolving_a_field_that_is_not_a_reference_is_a_diagnostic() -> None:
    assert resolve_diagnostics(Employee, steps.resolve(named(FieldName("name")))) == (
        diagnostics.NotAReference(
            view=ViewName.fake(),
            source=ir.SourceName("Employee"),
            field=FieldName("name"),
        ),
    )


def test_resolving_the_primary_key_is_a_diagnostic() -> None:
    assert resolve_diagnostics(Employee, steps.resolve(named(FieldName("id")))) == (
        diagnostics.NotAReference(
            view=ViewName.fake(),
            source=ir.SourceName("Employee"),
            field=FieldName("id"),
        ),
    )


def test_a_reference_to_no_entity_is_a_diagnostic_and_others_still_resolve() -> None:
    match compiler.check(
        hr_graph(),
        [
            View(
                name=ViewName.fake(),
                source=Memo,
                step=steps.pipeline(
                    steps.hydrate(), steps.omit(FieldName("department"))
                ),
            )
        ],
    ):
        case (compiler.PoisonedView(diagnostics=reported),):
            assert reported == (
                diagnostics.UnresolvableReference(
                    view=ViewName.fake(),
                    source=ir.SourceName("Memo"),
                    field=FieldName("project_id"),
                    id_type=ProjectId,
                ),
            )
        case checked:
            pytest.fail(f"expected one poisoned view, got {checked}")


def test_a_defect_in_a_nested_pipeline_names_the_referenced_entity() -> None:
    unknown = FieldName("budget")

    assert resolve_diagnostics(
        Employee, steps.resolve(named(FieldName("department_id")), steps.omit(unknown))
    ) == (
        diagnostics.UnknownField(
            view=ViewName.fake(),
            source=ir.SourceName("Department"),
            field=unknown,
            valid=(FieldName("id"), FieldName("name")),
        ),
    )


def test_building_a_resolve_checks_nothing() -> None:
    root = ir.FromSource(source=Memo)
    selection = named(FieldName("does_not_exist"))

    node = steps.resolve(selection)(root)

    match node:
        case ir.Resolve(input=input_node, fields=fields):
            assert (input_node, fields) == (root, selection)
        case _:
            pytest.fail(f"expected a Resolve node, got {node}")


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            diagnostics.NotAReference(
                view=ViewName("EmployeeView"),
                source=ir.SourceName("Employee"),
                field=FieldName("name"),
            ),
            (
                "EmployeeView: cannot resolve Employee.name, because it is not a "
                "Reference."
            ),
        ),
        (
            diagnostics.UnresolvableReference(
                view=ViewName("MemoView"),
                source=ir.SourceName("Memo"),
                field=FieldName("project_id"),
                id_type=ProjectId,
            ),
            (
                "MemoView: cannot resolve Memo.project_id, because ProjectId is the "
                "primary key of no entity."
            ),
        ),
        (
            diagnostics.ResolvedNameCollision(
                view=ViewName("ClashView"),
                source=ir.SourceName("Clash"),
                field=FieldName("department_id"),
                existing=FieldName("department"),
            ),
            (
                "ClashView: resolving Clash.department_id would name it "
                "'department', but Clash already has a field 'department'."
            ),
        ),
    ],
    ids=["not-a-reference", "unresolvable", "collision"],
)
def test_a_resolve_diagnostic_message_names_the_offending_values(
    diagnostic: diagnostics.CompileDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


# Reversing one-to-one References against `campaign_graph()`.


class ChannelCampaignId(Id[int]):
    pass


class ChannelCampaign(pydantic.BaseModel):
    id: ChannelCampaignId
    name: Text


class MetaCampaignId(Id[int]):
    pass


class MetaCampaign(pydantic.BaseModel):
    """A detail row: at most one per ChannelCampaign."""

    id: MetaCampaignId
    channel_campaign_id: Annotated[ChannelCampaignId, OneToOne()]
    ad_account: Text


class RelaunchId(Id[int]):
    pass


class Relaunch(pydantic.BaseModel):
    """Two one-to-one References to ChannelCampaign, so reversing it needs one
    chosen."""

    id: RelaunchId
    old_campaign_id: Annotated[ChannelCampaignId, OneToOne()]
    new_campaign_id: Annotated[ChannelCampaignId | None, OneToOne()]


class CampaignNoteId(Id[int]):
    pass


class CampaignNote(pydantic.BaseModel):
    """A many-to-one Reference to ChannelCampaign, which cannot be reversed."""

    id: CampaignNoteId
    campaign_id: ChannelCampaignId


class CopyId(Id[int]):
    pass


class Copy(pydantic.BaseModel):
    """Reversing it would name the field `copy`, an attribute of BaseModel."""

    id: CopyId
    campaign_id: Annotated[ChannelCampaignId, OneToOne()]


def campaign_graph() -> EntityGraph:
    return EntityGraph.build(
        [ChannelCampaign, MetaCampaign, Relaunch, CampaignNote, Copy],
        IdBase.fake(),
        OneToOneMarker.fake(),
    ).unwrap()


def compile_reversing(
    source: type[pydantic.BaseModel], step: steps.Step
) -> compiler.CompiledView:
    match compiler.compile_views(
        campaign_graph(), [View(name=ViewName.fake(), source=source, step=step)]
    ):
        case Ok(compiled):
            return compiled.views[0]
        case Err(error):
            pytest.fail(f"expected a compiled view, got {error}")


def reverse_diagnostics(
    source: type[pydantic.BaseModel], step: steps.Step
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    match compiler.compile_views(
        campaign_graph(), [View(name=ViewName.fake(), source=source, step=step)]
    ):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            return error.diagnostics


def meta_campaign_lookup() -> ir.ReverseLookup:
    """How a ChannelCampaign's MetaCampaign is found."""
    return ir.ReverseLookup(
        target=MetaCampaign,
        via=FieldName("channel_campaign_id"),
        id_type=ChannelCampaignId,
    )


def test_reverse_appends_the_referencing_entity_as_an_optional_field() -> None:
    view = compile_reversing(ChannelCampaign, steps.reverse(MetaCampaign))

    assert [
        (field.name, field.info.annotation, field.value)
        for field in view.relation.fields
        if isinstance(field, ir.Field)
    ] == [
        (FieldName("id"), ChannelCampaignId, ir.SourceField(field=FieldName("id"))),
        (FieldName("name"), Text, ir.SourceField(field=FieldName("name"))),
        (FieldName("meta_campaign"), MetaCampaign | None, meta_campaign_lookup()),
    ]


def test_flattening_a_reversed_entity_lifts_its_fields_as_optional() -> None:
    view = compile_reversing(
        ChannelCampaign,
        steps.pipeline(
            steps.reverse(MetaCampaign, steps.project([FieldName("ad_account")])),
            steps.flatten(FieldName("meta_campaign")),
        ),
    )

    assert [
        (field.name, field.info.annotation, field.value)
        for field in view.relation.fields
        if isinstance(field, ir.Field)
    ] == [
        (FieldName("id"), ChannelCampaignId, ir.SourceField(field=FieldName("id"))),
        (FieldName("name"), Text, ir.SourceField(field=FieldName("name"))),
        (
            FieldName("ad_account"),
            Text | None,
            ir.Lifted(
                through=(meta_campaign_lookup(),),
                field=ir.SourceField(field=FieldName("ad_account")),
            ),
        ),
    ]


def test_a_reversed_nested_pipeline_types_the_field_by_its_relation() -> None:
    view = compile_reversing(
        ChannelCampaign,
        steps.reverse(MetaCampaign, steps.omit(FieldName("channel_campaign_id"))),
    )

    match view.relation.fields[-1]:
        case ir.NestedField(name=name, value=lookup, relation=relation) as field:
            assert (name, lookup, field.optionality(), relation.field_names()) == (
                FieldName("meta_campaign"),
                meta_campaign_lookup(),
                Optionality.OPTIONAL,
                (FieldName("id"), FieldName("ad_account")),
            )
        case field:
            pytest.fail(f"expected a nested field, got {field}")


def test_reverse_via_follows_the_chosen_one_to_one_reference() -> None:
    view = compile_reversing(
        ChannelCampaign, steps.reverse_via(Relaunch, FieldName("new_campaign_id"))
    )

    assert values(view)[FieldName("relaunch")] == ir.ReverseLookup(
        target=Relaunch, via=FieldName("new_campaign_id"), id_type=ChannelCampaignId
    )


def test_reverse_in_a_nested_pipeline_starts_from_the_resolved_entity() -> None:
    view = compile_reversing(
        MetaCampaign,
        steps.resolve(
            named(FieldName("channel_campaign_id")),
            steps.reverse_via(Relaunch, FieldName("old_campaign_id")),
        ),
    )

    match view.relation.fields[1]:
        case ir.NestedField(relation=relation):
            assert relation.fields[-1].value == ir.ReverseLookup(
                target=Relaunch,
                via=FieldName("old_campaign_id"),
                id_type=ChannelCampaignId,
            )
        case field:
            pytest.fail(f"expected a nested field, got {field}")


def test_reverse_starts_from_the_source_however_its_fields_changed() -> None:
    view = compile_reversing(
        ChannelCampaign,
        steps.pipeline(steps.omit(FieldName("id")), steps.reverse(MetaCampaign)),
    )

    assert values(view) == {
        FieldName("name"): ir.SourceField(field=FieldName("name")),
        FieldName("meta_campaign"): meta_campaign_lookup(),
    }


@pytest.mark.parametrize(
    ("source", "step", "expected"),
    [
        (
            Company,
            steps.reverse(MetaCampaign),
            diagnostics.ReverseFromNonEntity(
                view=ViewName.fake(),
                source=ir.SourceName("Company"),
                entity=MetaCampaign,
            ),
        ),
        (
            ChannelCampaign,
            steps.reverse(Company),
            diagnostics.ReverseOfNonEntity(
                view=ViewName.fake(),
                source=ir.SourceName("ChannelCampaign"),
                entity=Company,
            ),
        ),
        (
            ChannelCampaign,
            steps.reverse(CampaignNote),
            diagnostics.NoOneToOneReference(
                view=ViewName.fake(),
                source=ir.SourceName("ChannelCampaign"),
                entity=CampaignNote,
                many_to_one=(FieldName("campaign_id"),),
            ),
        ),
        (
            MetaCampaign,
            steps.reverse(ChannelCampaign),
            diagnostics.NoOneToOneReference(
                view=ViewName.fake(),
                source=ir.SourceName("MetaCampaign"),
                entity=ChannelCampaign,
                many_to_one=(),
            ),
        ),
        (
            ChannelCampaign,
            steps.reverse(Relaunch),
            diagnostics.AmbiguousReverse(
                view=ViewName.fake(),
                source=ir.SourceName("ChannelCampaign"),
                entity=Relaunch,
                references=(FieldName("old_campaign_id"), FieldName("new_campaign_id")),
            ),
        ),
        (
            ChannelCampaign,
            steps.reverse_via(CampaignNote, FieldName("campaign_id")),
            diagnostics.NotAOneToOneReference(
                view=ViewName.fake(),
                source=ir.SourceName("ChannelCampaign"),
                entity=CampaignNote,
                field=FieldName("campaign_id"),
                valid=(),
            ),
        ),
        (
            ChannelCampaign,
            steps.reverse_via(Relaunch, FieldName("id")),
            diagnostics.NotAOneToOneReference(
                view=ViewName.fake(),
                source=ir.SourceName("ChannelCampaign"),
                entity=Relaunch,
                field=FieldName("id"),
                valid=(FieldName("old_campaign_id"), FieldName("new_campaign_id")),
            ),
        ),
        (
            ChannelCampaign,
            steps.pipeline(
                steps.rename(FieldName("name"), FieldName("meta_campaign")),
                steps.reverse(MetaCampaign),
            ),
            diagnostics.ReverseOntoExistingField(
                view=ViewName.fake(),
                source=ir.SourceName("ChannelCampaign"),
                entity=MetaCampaign,
                field=FieldName("meta_campaign"),
            ),
        ),
        (
            ChannelCampaign,
            steps.reverse(Copy),
            model_names.InvalidFieldName(
                view=ViewName.fake(),
                source=ir.SourceName("ChannelCampaign"),
                field=FieldName("copy"),
            ),
        ),
    ],
    ids=[
        "from-non-entity",
        "of-non-entity",
        "many-to-one",
        "no-reference",
        "ambiguous",
        "via-many-to-one",
        "via-non-reference",
        "onto-existing-field",
        "invalid-name",
    ],
)
def test_a_reverse_defect_is_one_diagnostic(
    source: type[pydantic.BaseModel],
    step: steps.Step,
    expected: diagnostics.CompileDiagnostic,
) -> None:
    assert reverse_diagnostics(source, step) == (expected,)


def test_steps_after_a_defective_reverse_find_the_reversed_field() -> None:
    assert reverse_diagnostics(
        ChannelCampaign,
        steps.pipeline(
            steps.reverse(CampaignNote), steps.omit(FieldName("campaign_note"))
        ),
    ) == (
        diagnostics.NoOneToOneReference(
            view=ViewName.fake(),
            source=ir.SourceName("ChannelCampaign"),
            entity=CampaignNote,
            many_to_one=(FieldName("campaign_id"),),
        ),
    )


def test_a_defect_in_a_reversed_nested_pipeline_names_the_reversed_entity() -> None:
    unknown = FieldName("budget")

    assert reverse_diagnostics(
        ChannelCampaign, steps.reverse(MetaCampaign, steps.omit(unknown))
    ) == (
        diagnostics.UnknownField(
            view=ViewName.fake(),
            source=ir.SourceName("MetaCampaign"),
            field=unknown,
            valid=tuple(map(FieldName, MetaCampaign.model_fields)),
        ),
    )


@pytest.mark.parametrize(
    ("step", "via"),
    [
        (steps.reverse(Company), ir.SoleReference()),
        (
            steps.reverse_via(Company, FieldName("does_not_exist")),
            FieldName("does_not_exist"),
        ),
    ],
    ids=["sole", "via"],
)
def test_building_a_reverse_checks_nothing(
    step: steps.Step, via: FieldName | ir.SoleReference
) -> None:
    root = ir.FromSource(source=Memo)

    node = step(root)

    match node:
        case ir.Reverse(input=input_node, entity=entity, via=built_via):
            assert (input_node, entity, built_via) == (root, Company, via)
        case _:
            pytest.fail(f"expected a Reverse node, got {node}")


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            diagnostics.ReverseFromNonEntity(
                view=ViewName("CompanyView"),
                source=ir.SourceName("Company"),
                entity=MetaCampaign,
            ),
            (
                "CompanyView: cannot reverse MetaCampaign onto Company, because "
                "Company is not an entity."
            ),
        ),
        (
            diagnostics.ReverseOfNonEntity(
                view=ViewName("CampaignView"),
                source=ir.SourceName("ChannelCampaign"),
                entity=Company,
            ),
            (
                "CampaignView: cannot reverse Company onto ChannelCampaign, because "
                "Company is not an entity."
            ),
        ),
        (
            diagnostics.NoOneToOneReference(
                view=ViewName("CampaignView"),
                source=ir.SourceName("ChannelCampaign"),
                entity=CampaignNote,
                many_to_one=(FieldName("campaign_id"),),
            ),
            (
                "CampaignView: cannot reverse CampaignNote onto ChannelCampaign, "
                "because it has no one-to-one Reference to ChannelCampaign. Its "
                "References 'campaign_id' are not marked one-to-one."
            ),
        ),
        (
            diagnostics.NoOneToOneReference(
                view=ViewName("MetaView"),
                source=ir.SourceName("MetaCampaign"),
                entity=ChannelCampaign,
                many_to_one=(),
            ),
            (
                "MetaView: cannot reverse ChannelCampaign onto MetaCampaign, "
                "because it has no one-to-one Reference to MetaCampaign."
            ),
        ),
        (
            diagnostics.AmbiguousReverse(
                view=ViewName("CampaignView"),
                source=ir.SourceName("ChannelCampaign"),
                entity=Relaunch,
                references=(FieldName("old_campaign_id"), FieldName("new_campaign_id")),
            ),
            (
                "CampaignView: cannot reverse Relaunch onto ChannelCampaign, because "
                "each of 'old_campaign_id', 'new_campaign_id' is a one-to-one "
                "Reference to ChannelCampaign. Choose one with reverse_via."
            ),
        ),
        (
            diagnostics.NotAOneToOneReference(
                view=ViewName("CampaignView"),
                source=ir.SourceName("ChannelCampaign"),
                entity=Relaunch,
                field=FieldName("id"),
                valid=(FieldName("old_campaign_id"), FieldName("new_campaign_id")),
            ),
            (
                "CampaignView: cannot reverse Relaunch.id onto ChannelCampaign, "
                "because it is not a one-to-one Reference to ChannelCampaign. Valid "
                "fields: 'old_campaign_id', 'new_campaign_id'."
            ),
        ),
        (
            diagnostics.ReverseOntoExistingField(
                view=ViewName("CampaignView"),
                source=ir.SourceName("ChannelCampaign"),
                entity=MetaCampaign,
                field=FieldName("meta_campaign"),
            ),
            (
                "CampaignView: reversing MetaCampaign would name it "
                "'meta_campaign', but ChannelCampaign already has a field "
                "'meta_campaign'."
            ),
        ),
    ],
    ids=[
        "from-non-entity",
        "of-non-entity",
        "many-to-one",
        "no-reference",
        "ambiguous",
        "via",
        "onto-existing-field",
    ],
)
def test_a_reverse_diagnostic_message_names_the_offending_values(
    diagnostic: diagnostics.CompileDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


# Flattening resolved fields.


def mentor_lookup() -> ir.Lookup:
    """How an Employee's `mentor_id` is resolved."""
    return ir.Lookup(
        reference=ir.SourceField(field=FieldName("mentor_id")),
        id_type=EmployeeId,
        target=Employee,
        optionality=Optionality.OPTIONAL,
    )


def own(name: Text) -> ir.SourceField:
    return ir.SourceField(field=FieldName(name.root))


def resolve_mentor(*then: steps.Step) -> steps.Step:
    """Resolves the mentor, prefixing its fields so none collides when lifted."""
    return steps.resolve(
        named(FieldName("mentor_id")),
        steps.rename(steps.select(lambda _field: True), prefixed(Text("mentor"))),
        *then,
    )


def lifted(*through: ir.Lookup, field: Text) -> ir.Lifted:
    return ir.Lifted(through=through, field=own(field))


def test_flatten_lifts_an_entitys_fields_in_place_of_the_resolved_field() -> None:
    view = compile_resolving(
        Memo,
        steps.pipeline(
            steps.resolve(named(FieldName("department_id"))),
            steps.flatten(FieldName("department")),
        ),
    )

    assert values(view) == {
        FieldName("project_id"): own(Text("project_id")),
        FieldName("id"): lifted(department_lookup(), field=Text("id")),
        FieldName("name"): lifted(department_lookup(), field=Text("name")),
    }
    assert annotations(view.relation) == {
        FieldName("project_id"): ProjectId,
        FieldName("id"): DepartmentId,
        FieldName("name"): Text,
    }


def test_flatten_lifts_the_fields_of_a_nested_pipeline_with_their_defaults() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            steps.resolve(
                named(FieldName("department_id")),
                steps.omit(FieldName("id")),
                steps.rename(FieldName("name"), FieldName("department_name")),
                steps.default(FieldName("department_name"), "Sales"),
            ),
            steps.flatten(FieldName("department")),
        ),
    )

    assert values(view) == {
        FieldName("id"): own(Text("id")),
        FieldName("name"): own(Text("name")),
        FieldName("department_name"): lifted(department_lookup(), field=Text("name")),
        FieldName("mentor_id"): own(Text("mentor_id")),
    }
    assert defaults(view) == {FieldName("department_name"): ir.ScalarDefault("Sales")}


def test_flatten_with_a_select_passes_over_fields_that_are_not_resolved() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            steps.resolve(
                named(FieldName("department_id")),
                steps.project([FieldName("name")]),
                steps.rename(FieldName("name"), FieldName("department_name")),
            ),
            steps.flatten(named(FieldName("name"), FieldName("department"))),
        ),
    )

    assert values(view) == {
        FieldName("id"): own(Text("id")),
        FieldName("name"): own(Text("name")),
        FieldName("department_name"): lifted(department_lookup(), field=Text("name")),
        FieldName("mentor_id"): own(Text("mentor_id")),
    }


def test_fields_lifted_out_of_an_optional_reference_are_optional() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(resolve_mentor(), steps.flatten(FieldName("mentor"))),
    )

    # `mentor_mentor_id` was optional already, and stays `EmployeeId | None`.
    assert annotations(view.relation) == {
        FieldName("id"): EmployeeId,
        FieldName("name"): Text,
        FieldName("department_id"): DepartmentId,
        FieldName("mentor_id"): EmployeeId | None,
        FieldName("mentor_name"): Text | None,
        FieldName("mentor_department_id"): DepartmentId | None,
        FieldName("mentor_mentor_id"): EmployeeId | None,
    }


def mentor_department() -> ir.ChainedLookup:
    """How an Employee's mentor's department is found."""
    return ir.ChainedLookup(chain=(mentor_lookup(), department_lookup()))


def test_a_resolved_field_lifted_out_follows_the_flattened_entitys_lookup() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            resolve_mentor(steps.resolve(named(FieldName("mentor_department_id")))),
            steps.flatten(FieldName("mentor")),
        ),
    )

    assert values(view)[FieldName("mentor_department")] == mentor_department()
    assert annotations(view.relation)[FieldName("mentor_department")] == (
        Department | None
    )
    assert view.relation.lookups() == (mentor_lookup(), department_lookup())


def nested_field(view: compiler.CompiledView, name: FieldName) -> ir.NestedField:
    match next(field for field in view.relation.fields if field.name == name):
        case ir.NestedField() as nested:
            return nested
        case field:
            pytest.fail(f"expected a nested field, got {field}")


def test_a_nested_model_lifted_out_follows_the_flattened_entitys_lookup() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            resolve_mentor(
                steps.resolve(
                    named(FieldName("mentor_department_id")),
                    steps.omit(FieldName("id")),
                )
            ),
            steps.flatten(FieldName("mentor")),
        ),
    )

    nested = nested_field(view, FieldName("mentor_department"))
    assert (nested.value, nested.relation.field_names()) == (
        mentor_department(),
        (FieldName("name"),),
    )


def department_name() -> steps.Step:
    """Resolves the mentor's department to its name, `department_name`."""
    return steps.resolve(
        named(FieldName("mentor_department_id")),
        steps.project([FieldName("name")]),
        steps.rename(FieldName("name"), FieldName("department_name")),
    )


@pytest.mark.parametrize(
    "flattening",
    [
        steps.pipeline(
            resolve_mentor(
                department_name(), steps.flatten(FieldName("mentor_department"))
            ),
            steps.flatten(FieldName("mentor")),
        ),
        steps.pipeline(
            resolve_mentor(department_name()),
            steps.flatten(FieldName("mentor")),
            steps.flatten(FieldName("mentor_department")),
        ),
    ],
    ids=["flattened before it is lifted", "flattened after it is lifted"],
)
def test_a_resolved_field_can_be_flattened_before_or_after_it_is_lifted(
    flattening: steps.Step,
) -> None:
    view = compile_resolving(Employee, flattening)

    assert values(view)[FieldName("department_name")] == lifted(
        mentor_lookup(), department_lookup(), field=Text("name")
    )


@pytest.mark.parametrize(
    "resolving",
    [steps.resolve(named(FieldName("mentor_department_id"))), steps.hydrate()],
    ids=["named", "hydrate"],
)
def test_a_lifted_reference_resolves_through_the_flattened_entity(
    resolving: steps.Step,
) -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(resolve_mentor(), steps.flatten(FieldName("mentor")), resolving),
    )

    assert values(view)[FieldName("mentor_department")] == mentor_department()
    assert annotations(view.relation)[FieldName("mentor_department")] == (
        Department | None
    )


def test_a_lifted_reference_resolves_through_a_nested_pipeline() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            resolve_mentor(),
            steps.flatten(FieldName("mentor")),
            steps.resolve(
                named(FieldName("mentor_department_id")), steps.omit(FieldName("id"))
            ),
        ),
    )

    nested = nested_field(view, FieldName("mentor_department"))
    assert (nested.value, nested.relation.field_names()) == (
        mentor_department(),
        (FieldName("name"),),
    )


def flatten_diagnostics(
    *pipeline: steps.Step,
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    return resolve_diagnostics(Employee, steps.pipeline(*pipeline))


@pytest.mark.parametrize(
    "then",
    [(), (steps.omit(FieldName("id")), steps.omit(FieldName("name")))],
    ids=["alone", "steps after it use the existing fields"],
)
def test_a_lifted_field_onto_an_existing_name_is_a_diagnostic_per_field(
    then: tuple[steps.Step, ...],
) -> None:
    assert flatten_diagnostics(
        steps.resolve(named(FieldName("department_id"))),
        steps.flatten(FieldName("department")),
        *then,
    ) == tuple(
        diagnostics.FlattenOntoExistingField(
            view=ViewName.fake(), flattened=FieldName("department"), field=field
        )
        for field in (FieldName("id"), FieldName("name"))
    )


def test_two_flattened_fields_lifting_one_name_report_the_second() -> None:
    label = FieldName("label")

    reported = flatten_diagnostics(
        steps.resolve(
            named(FieldName("department_id")),
            steps.omit(FieldName("id")),
            steps.rename(FieldName("name"), label),
        ),
        steps.resolve(
            named(FieldName("mentor_id")),
            steps.project([]),
            steps.add(label, str, default="mentor"),
        ),
        steps.flatten(named(FieldName("department"), FieldName("mentor"))),
    )

    assert reported == (
        diagnostics.FlattenOntoExistingField(
            view=ViewName.fake(), flattened=FieldName("mentor"), field=label
        ),
    )


@pytest.mark.parametrize(
    ("named_field", "expected"),
    [
        (
            FieldName("name"),
            diagnostics.NotAResolvedField(
                view=ViewName.fake(),
                source=ir.SourceName("Employee"),
                field=FieldName("name"),
            ),
        ),
        (
            FieldName("budget"),
            diagnostics.UnknownField(
                view=ViewName.fake(),
                source=ir.SourceName("Employee"),
                field=FieldName("budget"),
                valid=(
                    FieldName("id"),
                    FieldName("name"),
                    FieldName("department_id"),
                    FieldName("mentor_id"),
                ),
            ),
        ),
    ],
    ids=["not resolved", "unknown"],
)
def test_flattening_a_named_field_it_cannot_lift_is_one_diagnostic(
    named_field: FieldName, expected: diagnostics.CompileDiagnostic
) -> None:
    assert flatten_diagnostics(steps.flatten(named_field)) == (expected,)


def test_building_a_flatten_checks_nothing() -> None:
    root = ir.FromSource(source=Memo)
    unknown = FieldName("does_not_exist")

    node = steps.flatten(unknown)(root)

    match node:
        case ir.Flatten(input=input_node, fields=fields):
            assert (input_node, fields) == (root, unknown)
        case _:
            pytest.fail(f"expected a Flatten node, got {node}")


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            diagnostics.NotAResolvedField(
                view=ViewName("EmployeeView"),
                source=ir.SourceName("Employee"),
                field=FieldName("name"),
            ),
            (
                "EmployeeView: cannot flatten Employee.name, because it is not a "
                "resolved Reference."
            ),
        ),
        (
            diagnostics.FlattenOntoExistingField(
                view=ViewName("EmployeeView"),
                flattened=FieldName("department"),
                field=FieldName("name"),
            ),
            (
                "EmployeeView: cannot lift 'name' out of 'department', because it "
                "already has a field 'name'. Rename or omit it in the nested "
                "pipeline of the resolve."
            ),
        ),
    ],
    ids=["not resolved", "collision"],
)
def test_a_flatten_diagnostic_message_names_the_offending_values(
    diagnostic: diagnostics.CompileDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


# Property-based tests over EntityGraphs generated per example. The Entities are
# built at run time, so each example has its own shape of graph: any number of
# Entities, References between any of them (self-References included), required or
# optional, with or without the `_id` suffix.


class Depth(pydantic.RootModel[int], frozen=True):
    """How many more levels of nested pipelines a pipeline may contain."""


def new_id_type(entity: ir.SourceName) -> IdType:
    id_type = types.new_class(f"{entity.root}Id", (Id[int],))
    assert issubclass(id_type, Id)
    return id_type


@st.composite
def entity_model(
    draw: st.DrawFn, name: ir.SourceName, id_type: IdType, targets: Sequence[IdType]
) -> type[pydantic.BaseModel]:
    """An Entity keyed by `id_type`, with Text fields and References to `targets`.

    Field names never collide, before or after resolving: the j-th Reference is
    `r{j}_id` or `r{j}`, and Text fields are `f{j}`.
    """
    fields: dict[FieldName, ir.RuntimeAnnotation] = {FieldName("id"): id_type}
    referenced = draw(st.lists(st.sampled_from(targets), max_size=3))
    for index, target in enumerate(referenced):
        field = draw(st.sampled_from([f"r{index}_id", f"r{index}"]))
        fields[FieldName(field)] = draw(st.sampled_from([target, target | None]))
    for index in range(draw(st.integers(min_value=0, max_value=2))):
        fields[FieldName(f"f{index}")] = Text
    order = draw(st.permutations(list(fields)))
    return test_ir.new_model(name, {field: fields[field] for field in order})


class Models(pydantic.BaseModel, frozen=True):
    """Generated Entities, valid by construction, and Sources that are not Entities.

    Every Reference of an Entity points to one of `entities`, so the EntityGraph
    always builds. `memo` has a Reference to an ID type that keys no Entity. `clash`
    has a field `r0` that resolving its Reference `r0_id` would collide with.
    """

    entities: tuple[type[pydantic.BaseModel], ...]
    memo: type[pydantic.BaseModel]
    clash: type[pydantic.BaseModel]

    def graph(self) -> EntityGraph:
        return EntityGraph.build(
            self.entities, IdBase.fake(), OneToOneMarker.fake()
        ).unwrap()

    def sources(self) -> tuple[type[pydantic.BaseModel], ...]:
        return (*self.entities, self.memo, self.clash)

    def field_names(self) -> tuple[FieldName, ...]:
        """Every field name of every Source, before and after resolving."""
        return tuple(
            dict.fromkeys(
                FieldName(name)
                for model in self.sources()
                for field in model.model_fields
                for name in (field, field.removesuffix("_id"))
            )
        )


@st.composite
def models(draw: st.DrawFn) -> Models:
    names = [
        ir.SourceName(f"Entity{index}")
        for index in range(draw(st.integers(min_value=1, max_value=4)))
    ]
    id_types = [new_id_type(name) for name in names]
    entities = tuple(
        draw(entity_model(name, id_type, id_types))
        for name, id_type in zip(names, id_types, strict=True)
    )
    memo = test_ir.new_model(
        ir.SourceName("Memo"),
        {
            FieldName("r0_id"): draw(st.sampled_from(id_types)),
            FieldName("r1_id"): new_id_type(ir.SourceName("Unkeyed")),
            FieldName("f0"): Text,
        },
    )
    clash = test_ir.new_model(
        ir.SourceName("Clash"),
        {
            FieldName("r0_id"): draw(st.sampled_from(id_types)),
            FieldName("r0"): Text,
        },
    )
    return Models(entities=entities, memo=memo, clash=clash)


texts = st.builds(Text, st.text(max_size=3))


@st.composite
def stored_entities(draw: st.DrawFn, drawn: Models) -> tuple[pydantic.BaseModel, ...]:
    """At least one instance of each Entity, with no dangling Reference.

    IDs are distinct per Entity, so no instance replaces another in the engine.
    Entities are drawn in the order of `drawn.entities`, not of the EntityGraph,
    whose order changes between runs.
    """
    nodes = drawn.graph().entities
    ids = {
        model: draw(
            st.lists(
                st.builds(nodes[model].id_type, st.integers()),
                unique=True,
                min_size=1,
                max_size=3,
            )
        )
        for model in drawn.entities
    }
    stored: list[pydantic.BaseModel] = []
    for model in drawn.entities:
        references = {
            reference.field.root: reference for reference in nodes[model].references
        }
        for id_value in ids[model]:
            values: dict[str, object] = {"id": id_value}
            for field in model.model_fields:
                if field == "id":
                    continue
                match references.get(field):
                    case None:
                        values[field] = draw(texts)
                    case reference:
                        targets = st.sampled_from(ids[reference.target])
                        values[field] = draw(
                            targets
                            if reference.optionality is Optionality.REQUIRED
                            else st.none() | targets
                        )
            stored.append(model(**values))
    return tuple(stored)


def keep_target(
    references: dict[FieldName, type[pydantic.BaseModel]],
    old: FieldName,
    new: FieldName,
) -> None:
    """Moves the target of `old`, if it is a Reference, to its new name `new`."""
    if old in references:
        references[new] = references.pop(old)


def flattening(
    graph: EntityGraph,
    reference: FieldName,
    target: type[pydantic.BaseModel],
    nested: steps.Step,
    prefix: Text,
) -> steps.Step:
    """Resolves `reference` to `target` through `nested`, then flattens it, then
    resolves the References it lifted out of `target`.

    `prefix` keeps the lifted fields from colliding with any other.
    """
    return steps.pipeline(
        steps.resolve(
            named(reference),
            nested,
            steps.rename(steps.select(lambda _field: True), prefixed(prefix)),
        ),
        steps.flatten(FieldName(reference.root.removesuffix("_id") or reference.root)),
        steps.resolve(
            steps.select(
                functools.partial(
                    is_lifted_reference,
                    references={
                        lifted.field for lifted in graph.entities[target].references
                    },
                    prefix=prefix,
                )
            )
        ),
    )


@st.composite
def pipelines(
    draw: st.DrawFn,
    graph: EntityGraph,
    source: type[pydantic.BaseModel],
    depth: Depth,
) -> steps.Step:
    """A valid pipeline of omits, renames, projects, adds, resolves, reverses
    and flattens over `source`.

    Omits and renames name one field or select several. It may start by adding a
    field.

    Each step names only fields the earlier steps kept, and each rename gets a name
    no field has, so it never collides. A resolve selects Reference fields only,
    and its nested pipeline is drawn the same way over the referenced Entity. A
    reverse follows a one-to-one Reference to `source` backwards, onto a name no
    field has, and its nested pipeline is drawn over the referencing Entity. A
    flatten lifts the fields of one such resolve, each renamed to a name no other
    field has. Later steps leave the lifted fields alone.
    """
    fields = list(map(FieldName, source.model_fields))
    references = {
        reference.field: reference.target
        for reference in graph.entities[source].references
    }
    # The name each reverse gives its field, the referencing Entity and its
    # Reference. Sorted, because the order of the graph's Entities changes between
    # runs.
    reversible = sorted(
        (
            (
                FieldName(model_names.PascalCase(model.__name__).snake_case().root),
                model,
                reference.field,
            )
            for model, node in graph.entities.items()
            for reference in node.references
            if reference.target is source
            and reference.cardinality == Cardinality.ONE_TO_ONE
        ),
        key=lambda reversal: (reversal[0].root, reversal[2].root),
    )
    # An added field comes first, so later steps can omit, rename or project it.
    added = [FieldName("added") for _ in range(draw(st.integers(0, 1)))]
    fields.extend(added)
    drawn: list[steps.Step] = [
        steps.add(name, str, default=draw(st.sampled_from(["tag", pydantic.MISSING])))
        for name in added
    ]
    for index in range(draw(st.integers(min_value=0, max_value=len(fields)))):
        if not fields:
            break
        field = draw(st.sampled_from(fields))
        match draw(
            st.sampled_from(
                [
                    "omit",
                    "rename",
                    "rename_selected",
                    "project",
                    "resolve",
                    "flatten",
                    "reverse",
                    "reverse_flatten",
                ]
            )
        ):
            case "omit":
                omitted = draw(st.lists(st.sampled_from(fields), unique=True))
                fields = [kept for kept in fields if kept not in omitted]
                # Each by name, or all at once by a Select.
                drawn.extend(
                    draw(
                        st.sampled_from(
                            [
                                tuple(map(steps.omit, omitted)),
                                (steps.omit(named(*omitted)),),
                            ]
                        )
                    )
                )
            case "rename":
                new = FieldName(f"renamed_{index}")
                fields[fields.index(field)] = new
                keep_target(references, field, new)
                drawn.append(steps.rename(field, new))
            case "rename_selected":
                renamed = draw(st.lists(st.sampled_from(fields), unique=True))
                renaming = prefixed(Text(f"renamed_{index}"))
                fields = [
                    renaming(kept) if kept in renamed else kept for kept in fields
                ]
                references = {
                    renaming(kept) if kept in renamed else kept: target
                    for kept, target in references.items()
                }
                drawn.append(steps.rename(named(*renamed), renaming))
            case "project":
                projected = draw(st.lists(st.sampled_from(fields), unique=True))
                fields = [kept for kept in fields if kept in projected]
                drawn.append(steps.project(projected))
            # Without a Reference to flatten, this falls through.
            case "flatten" if depth.root > 0 and (
                resolvable := [kept for kept in fields if kept in references]
            ):
                flattened = draw(st.sampled_from(resolvable))
                fields.remove(flattened)
                target = references.pop(flattened)
                drawn.append(
                    flattening(
                        graph,
                        flattened,
                        target,
                        draw(pipelines(graph, target, Depth(depth.root - 1))),
                        Text(f"flat_{index}"),
                    )
                )
            # A reverse_flatten lifts the reversed field's fields, each renamed to a
            # name no other field has. Without a reverse to draw, this falls
            # through to a resolve.
            case "reverse" | "reverse_flatten" as action if depth.root > 0 and (
                unused := [
                    reversal for reversal in reversible if reversal[0] not in fields
                ]
            ):
                name, model, via = draw(st.sampled_from(unused))
                nested = draw(pipelines(graph, model, Depth(depth.root - 1)))
                drawn.append(
                    steps.pipeline(
                        steps.reverse_via(
                            model,
                            via,
                            nested,
                            *(
                                [
                                    steps.rename(
                                        steps.select(lambda _field: True),
                                        prefixed(Text(f"flat_{index}")),
                                    )
                                ]
                                if action == "reverse_flatten"
                                else []
                            ),
                        ),
                        *([steps.flatten(name)] if action == "reverse_flatten" else []),
                    )
                )
                fields.extend([] if action == "reverse_flatten" else [name])
            case _:
                resolved = [kept for kept in fields if kept in references]
                if not resolved or depth.root == 0:
                    continue
                target = references[draw(st.sampled_from(resolved))]
                chosen = [name for name in resolved if references[name] is target]
                nested = draw(pipelines(graph, target, Depth(depth.root - 1)))
                for name in chosen:
                    del references[name]
                    fields[fields.index(name)] = FieldName(
                        name.root.removesuffix("_id")
                    )
                drawn.append(steps.resolve(named(*chosen), nested))
    return steps.pipeline(*drawn)


@st.composite
def any_steps(draw: st.DrawFn, pool: Sequence[FieldName], depth: Depth) -> steps.Step:
    """Any step, naming fields from `pool` whether or not they exist.

    Unlike `pipelines`, it may omit unknown fields, rename onto existing ones or
    several fields onto one, resolve fields that are not References, resolve onto
    an existing name, or flatten fields that are not resolved.
    """
    names = st.sampled_from(pool)
    match draw(
        st.sampled_from(
            ["omit", "flatten", "rename", "rename_selected", "project", "resolve"]
        )
    ):
        case "omit" | "flatten" as action:
            # One name, or a Select of any names.
            fields = draw(names | st.lists(names).map(lambda chosen: named(*chosen)))
            return {"omit": steps.omit, "flatten": steps.flatten}[action](fields)
        case "rename":
            return steps.rename(draw(names), draw(names))
        case "rename_selected":
            renames = draw(st.dictionaries(names, names))
            return steps.rename(
                named(*renames), lambda field: renames.get(field, field)
            )
        case "project":
            return steps.project(draw(st.lists(names, unique=True)))
        case _:
            selection = draw(
                st.just(steps.references())
                | st.lists(names).map(lambda chosen: named(*chosen))
            )
            nested = (
                draw(any_pipelines(pool, Depth(depth.root - 1)))
                if depth.root > 0
                else steps.pipeline()
            )
            return steps.resolve(selection, nested)


def any_pipelines(
    pool: Sequence[FieldName], depth: Depth
) -> st.SearchStrategy[steps.Step]:
    return st.lists(any_steps(pool, depth), max_size=4).map(
        lambda drawn: steps.pipeline(*drawn)
    )


class ViewCase(pydantic.BaseModel, frozen=True):
    """Views over generated models; those in `valid` are valid by construction."""

    models: Models
    views: tuple[View, ...]
    valid: frozenset[ViewName]


@st.composite
def view_cases(draw: st.DrawFn) -> ViewCase:
    drawn_models = draw(models())
    graph = drawn_models.graph()
    pool = (*drawn_models.field_names(), FieldName("unknown"))
    views: list[View] = []
    valid: set[ViewName] = set()
    for index in range(draw(st.integers(min_value=1, max_value=4))):
        name = ViewName(f"View{index}")
        source = draw(st.sampled_from(drawn_models.sources()))
        if source in graph.entities and draw(st.booleans()):
            valid.add(name)
            step = draw(pipelines(graph, source, Depth(2)))
        else:
            # A single step often has a single defect, which a longer pipeline
            # would rarely leave as the only one.
            step = draw(any_steps(pool, Depth(2)) | any_pipelines(pool, Depth(2)))
        views.append(View(name=name, source=source, step=step))
    return ViewCase(models=drawn_models, views=tuple(views), valid=frozenset(valid))


class PlainShape(pydantic.BaseModel, frozen=True):
    """An `ir.Field` without its FieldInfo, which compares by identity."""

    name: FieldName
    annotation: ir.RuntimeAnnotation
    value: ir.Value


class NestedShape(pydantic.BaseModel, frozen=True):
    name: FieldName
    value: ir.EntityValue
    relation: RelationShape


class RelationShape(pydantic.BaseModel, frozen=True):
    """An `ir.Relation` that compares by what its fields hold."""

    source: ir.SourceName
    fields: tuple[PlainShape | NestedShape, ...]

    @staticmethod
    def of(relation: ir.Relation) -> RelationShape:
        return RelationShape(
            source=relation.source,
            fields=tuple(
                PlainShape(
                    name=field.name, annotation=field.info.annotation, value=field.value
                )
                if isinstance(field, ir.Field)
                else NestedShape(
                    name=field.name,
                    value=field.value,
                    relation=RelationShape.of(field.relation),
                )
                for field in relation.fields
            ),
        )


class CompiledShape(pydantic.BaseModel, frozen=True):
    name: ViewName
    source: type[pydantic.BaseModel]
    relation: RelationShape


def shapes(
    checked: Iterable[compiler.CompiledView | compiler.PoisonedView],
) -> tuple[CompiledShape | compiler.PoisonedView, ...]:
    return tuple(
        CompiledShape(
            name=view.name, source=view.source, relation=RelationShape.of(view.relation)
        )
        if isinstance(view, compiler.CompiledView)
        else view
        for view in checked
    )


class InvarianceCase(pydantic.BaseModel, frozen=True):
    """The same Views over the generated Entities, and over `varied`.

    `varied` is the generated Entities reordered, plus `unreferenced`: Entities that
    may reference the others, but that nothing references.
    """

    views: ViewCase
    unreferenced: tuple[type[pydantic.BaseModel], ...]
    varied: tuple[type[pydantic.BaseModel], ...]


@st.composite
def invariance_cases(draw: st.DrawFn) -> InvarianceCase:
    views = draw(view_cases())
    nodes = views.models.graph().entities
    targets = [nodes[model].id_type for model in views.models.entities]
    unreferenced: list[type[pydantic.BaseModel]] = []
    for index in range(draw(st.integers(min_value=0, max_value=2))):
        name = ir.SourceName(f"Unreferenced{index}")
        unreferenced.append(draw(entity_model(name, new_id_type(name), targets)))
    varied = draw(st.permutations([*views.models.entities, *unreferenced]))
    return InvarianceCase(
        views=views, unreferenced=tuple(unreferenced), varied=tuple(varied)
    )


@given(invariance_cases())
def test_unreferenced_entities_and_entity_order_do_not_change_the_result(
    case: InvarianceCase,
) -> None:
    original = compiler.check(case.views.models.graph(), case.views.views)
    varied = compiler.check(
        EntityGraph.build(case.varied, IdBase.fake(), OneToOneMarker.fake()).unwrap(),
        case.views.views,
    )

    reordered = [e for e in case.varied if e not in case.unreferenced] != list(
        case.views.models.entities
    )
    event(f"entities reordered: {reordered}")
    event(f"unreferenced entities added: {len(case.unreferenced)}")
    for view in original:
        event(f"view: {type(view).__name__}")
    assert shapes(varied) == shapes(original)


class PipelineCase(pydantic.BaseModel, frozen=True):
    """A valid pipeline over one of the generated Entities."""

    models: Models
    source: type[pydantic.BaseModel]
    step: steps.Step


@st.composite
def pipeline_cases(draw: st.DrawFn) -> PipelineCase:
    drawn_models = draw(models())
    source = draw(st.sampled_from(drawn_models.entities))
    step = draw(pipelines(drawn_models.graph(), source, Depth(1)))
    return PipelineCase(models=drawn_models, source=source, step=step)


@given(pipeline_cases())
def test_resolve_accepts_exactly_the_fields_references_selects(
    case: PipelineCase,
) -> None:
    def compile_step(step: steps.Step) -> compiler.CompiledView | None:
        match compiler.compile_views(
            case.models.graph(),
            [View(name=ViewName.fake(), source=case.source, step=step)],
        ):
            case Ok(compiled):
                return compiled.views[0]
            case Err():
                return None

    compiled = compile_step(case.step)
    assert compiled is not None
    for field in compiled.relation.fields:
        resolved = compile_step(
            steps.pipeline(case.step, steps.resolve(named(field.name)))
        )
        is_reference = steps.references().selects(field, case.models.graph())
        event(f"is_reference: {is_reference}")
        assert is_reference == (resolved is not None)


# Properties of the Relations compiled Views have.


@st.composite
def compiled_views(draw: st.DrawFn) -> compiler.CompiledView:
    """A valid View over a generated Entity, compiled."""
    drawn = draw(models())
    graph = drawn.graph()
    source = draw(st.sampled_from(drawn.entities))
    step = draw(pipelines(graph, source, Depth(2)))
    view = View(name=ViewName.fake(), source=source, step=step)
    return compiler.compile_views(graph, [view]).unwrap().views[0]


def resolved_fields(
    relation: ir.Relation,
) -> tuple[tuple[ir.RelationField, ir.EntityValue], ...]:
    """Every field holding a found Entity, nested ones included."""
    found: list[tuple[ir.RelationField, ir.EntityValue]] = []
    for field in relation.fields:
        match field:
            case ir.Field(
                value=ir.Lookup() | ir.ReverseLookup() | ir.ChainedLookup() as value
            ):
                found.append((field, value))
            case ir.NestedField(value=value, relation=nested):
                found.extend(((field, value), *resolved_fields(nested)))
            case ir.Field():
                pass
    return tuple(found)


@given(compiled_views())
def test_a_resolved_field_is_optional_iff_a_lookup_on_its_way_is(
    view: compiler.CompiledView,
) -> None:
    for field, value in resolved_fields(view.relation):
        assert field.optionality() == ir.Lookup.optionality_of(value.lookups())


def lifted_fields(relation: ir.Relation) -> tuple[tuple[ir.Field, ir.Lifted], ...]:
    """Every field `flatten` lifted, nested ones included."""
    found: list[tuple[ir.Field, ir.Lifted]] = []
    for field in relation.fields:
        match field:
            case ir.Field(value=ir.Lifted() as value):
                found.append((field, value))
            case ir.NestedField(relation=nested):
                found.extend(lifted_fields(nested))
            case ir.Field():
                pass
    return tuple(found)


@given(compiled_views())
def test_a_field_lifted_through_an_optional_lookup_is_optional(
    view: compiler.CompiledView,
) -> None:
    lifted = lifted_fields(view.relation)
    event(f"lifted fields: {bool(lifted)}")
    for field, value in lifted:
        if ir.Lookup.optionality_of(value.lookups()) is Optionality.OPTIONAL:
            assert field.optionality() is Optionality.OPTIONAL


@given(compiled_views())
def test_lookups_name_each_looked_up_entity_once(
    view: compiler.CompiledView,
) -> None:
    targets = [found.target for found in view.relation.lookups()]

    assert sorted(targets, key=lambda target: target.__name__) == sorted(
        {
            lookup.target
            for _, value in (
                *resolved_fields(view.relation),
                *lifted_fields(view.relation),
            )
            for lookup in value.lookups()
        },
        key=lambda target: target.__name__,
    )


# Payloads.


def compile_catalog(
    views: Sequence[View], payloads: Sequence[Payload]
) -> Result[compiler.CompiledCatalog, DiagnosticsError[diagnostics.CompileDiagnostic]]:
    return compiler.compile_catalog(empty_graph(), views, payloads)


def test_a_payload_compiles_to_the_relation_its_pipeline_builds() -> None:
    payload = Payload(
        name=PayloadName.fake(),
        source=Company,
        step=steps.omit(FieldName("secret")),
    )

    (compiled,) = compile_catalog([], [payload]).unwrap().payloads

    assert compiled.name == payload.name
    assert [field.name for field in compiled.relation.fields] == names(
        Text("name"), Text("revenue")
    )


def test_a_payload_step_defect_is_reported_under_the_payload_name() -> None:
    payload = Payload(
        name=PayloadName.fake(),
        source=Company,
        step=steps.omit(FieldName("missing")),
    )

    match compile_catalog([], [payload]):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            assert [d.view for d in error.diagnostics] == [payload.name]


def test_a_name_shared_by_a_view_and_a_payload_is_one_diagnostic() -> None:
    name = Text("Company")
    view = View(name=ViewName(name.root), source=Company, step=steps.pipeline())
    payload = Payload(
        name=PayloadName(name.root), source=Company, step=steps.pipeline()
    )

    match compile_catalog([view, view], [payload]):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            assert error.diagnostics == (diagnostics.SharedModelName(view=view.name),)


def test_an_update_payload_composes_the_create_pipeline_with_more_steps() -> None:
    create = steps.omit(FieldName("secret"))
    update = Payload(
        name=PayloadName.fake(),
        source=Company,
        step=steps.pipeline(create, steps.omit(FieldName("revenue"))),
    )

    (compiled,) = compile_catalog([], [update]).unwrap().payloads

    assert [field.name for field in compiled.relation.fields] == names(Text("name"))


def test_views_and_payloads_of_distinct_names_compile() -> None:
    view = View(name=ViewName.fake(), source=Company, step=steps.pipeline())
    payload = Payload(name=PayloadName.fake(), source=Company, step=steps.pipeline())

    compiled = compile_catalog([view], [payload]).unwrap()

    assert [v.name for v in compiled.views] == [view.name]
    assert [p.name for p in compiled.payloads] == [payload.name]


# retype.


def compile_employee_payload(
    step: steps.Step,
) -> Result[compiler.CompiledCatalog, DiagnosticsError[diagnostics.CompileDiagnostic]]:
    payload = Payload(name=PayloadName.fake(), source=Employee, step=step)
    return compiler.compile_catalog(hr_graph(), [], [payload])


def employee_payload_diagnostics(
    step: steps.Step,
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    match compile_employee_payload(step):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            return error.diagnostics


def annotations(relation: ir.Relation) -> dict[FieldName, ir.RuntimeAnnotation]:
    return {
        field.name: field.annotation()
        for field in relation.fields
        if isinstance(field, ir.Field)
    }


def test_retype_maps_each_selected_fields_type_with_the_transform() -> None:
    (compiled,) = (
        compile_employee_payload(
            steps.retype(
                named(FieldName("name"), FieldName("mentor_id")),
                lambda annotation: annotation | MISSING,
            ),
        )
        .unwrap()
        .payloads
    )

    assert annotations(compiled.relation) == {
        FieldName("id"): EmployeeId,
        FieldName("name"): Text | MISSING,
        FieldName("department_id"): DepartmentId,
        FieldName("mentor_id"): EmployeeId | None | MISSING,
    }


def test_retype_to_gives_the_named_field_one_type() -> None:
    (compiled,) = (
        compile_employee_payload(steps.retype_to(FieldName("name"), int))
        .unwrap()
        .payloads
    )

    assert annotations(compiled.relation)[FieldName("name")] is int


class Constrained(pydantic.BaseModel):
    code: Annotated[Text, "constraint"]


def test_a_retyped_field_drops_the_old_types_constraints() -> None:
    payload = Payload(
        name=PayloadName.fake(),
        source=Constrained,
        step=steps.retype_to(FieldName("code"), int),
    )

    (compiled,) = compile_catalog([], [payload]).unwrap().payloads

    match compiled.relation.fields:
        case (ir.Field(info=info),):
            assert info.metadata == []
        case fields:
            pytest.fail(f"expected one own field, got {fields}")


def test_retyping_an_unknown_field_is_a_diagnostic() -> None:
    unknown = FieldName("nme")

    assert employee_payload_diagnostics(steps.retype_to(unknown, int)) == (
        diagnostics.UnknownField(
            view=PayloadName.fake(),
            source=ir.SourceName("Employee"),
            field=unknown,
            valid=tuple(map(FieldName, Employee.model_fields)),
        ),
    )


def test_retyping_a_field_resolved_through_a_nested_pipeline_is_a_diagnostic() -> None:
    step = steps.pipeline(
        steps.resolve(named(FieldName("department_id")), steps.omit(FieldName("name"))),
        steps.retype_to(FieldName("department"), int),
    )

    assert employee_payload_diagnostics(step) == (
        diagnostics.RetypeNestedField(
            view=PayloadName.fake(),
            source=ir.SourceName("Employee"),
            field=FieldName("department"),
        ),
    )


@pytest.mark.parametrize(
    "step",
    [
        steps.retype_to(FieldName("name"), int),
        steps.resolve(
            named(FieldName("department_id")),
            steps.retype(named(FieldName("name")), lambda annotation: annotation),
        ),
    ],
    ids=["in the view", "in a nested pipeline"],
)
def test_retype_in_a_view_is_a_diagnostic(step: steps.Step) -> None:
    assert resolve_diagnostics(Employee, step) == (
        diagnostics.RetypeInView(view=ViewName.fake()),
    )


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            diagnostics.RetypeInView(view=ViewName("CompanySummary")),
            (
                "CompanySummary: a View cannot retype its fields, because its "
                "mapping copies each value unchanged. Retype in a Payload instead."
            ),
        ),
        (
            diagnostics.RetypeNestedField(
                view=PayloadName("CompanyUpdate"),
                source=ir.SourceName("Company"),
                field=FieldName("parent"),
            ),
            (
                "CompanyUpdate: cannot retype Company.parent, because its type is "
                "the model of a nested pipeline."
            ),
        ),
    ],
    ids=["in-view", "nested"],
)
def test_a_retype_diagnostic_message_names_the_offending_values(
    diagnostic: diagnostics.CompileDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


# Base classes.


class NamedBase(pydantic.BaseModel):
    name: Text


class Keyed[T](pydantic.BaseModel):
    name: T


class DepartmentNamedBase(pydantic.BaseModel):
    name: DepartmentId


@pytest.mark.parametrize("base", [NamedBase, Keyed[Text]])
def test_a_kept_field_the_base_declares_with_the_same_type_compiles(
    base: type[pydantic.BaseModel],
) -> None:
    payload = Payload(
        name=PayloadName.fake(), source=Company, step=steps.pipeline(), base=base
    )

    (compiled,) = compile_catalog([], [payload]).unwrap().payloads

    # The model inherits `name`; the relation still holds it for the mapping.
    assert compiled.base is base
    assert [field.name for field in compiled.relation.fields] == names(
        Text("name"), Text("secret"), Text("revenue")
    )


@pytest.mark.parametrize("base", [DepartmentNamedBase, Keyed[DepartmentId]])
def test_a_kept_field_the_base_declares_with_another_type_is_a_diagnostic(
    base: type[pydantic.BaseModel],
) -> None:
    view = View(name=ViewName.fake(), source=Company, step=steps.pipeline(), base=base)

    match compile_catalog([view], []):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            assert error.diagnostics == (
                diagnostics.BaseFieldTypeMismatch(
                    view=view.name,
                    base=base,
                    field=FieldName("name"),
                    declared=DepartmentId,
                    kept=Text,
                ),
            )


def test_a_field_the_base_declares_is_not_checked_once_omitted() -> None:
    payload = Payload(
        name=PayloadName.fake(),
        source=Company,
        step=steps.omit(FieldName("name")),
        base=DepartmentNamedBase,
    )

    assert isinstance(compile_catalog([], [payload]), Ok)


def test_the_base_field_type_mismatch_message_contains_the_offending_values() -> None:
    view = ViewName.fake()
    message = diagnostics.BaseFieldTypeMismatch(
        view=view,
        base=Keyed[DepartmentId],
        field=FieldName("name"),
        declared=DepartmentId,
        kept=Text,
    ).describe()

    assert all(
        part in message.root
        for part in (view.root, "Keyed[", "'name'", "DepartmentId", "Text")
    )


class DepartmentBase(pydantic.BaseModel):
    department: Department


def test_a_nested_field_the_base_declares_is_a_diagnostic() -> None:
    view = View(
        name=ViewName.fake(),
        source=Employee,
        step=steps.resolve(
            named(FieldName("department_id")), steps.project([FieldName("name")])
        ),
        base=DepartmentBase,
    )

    match compiler.compile_views(hr_graph(), [view]):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            assert error.diagnostics == (
                diagnostics.BaseFieldIsNestedModel(
                    view=view.name, base=DepartmentBase, field=FieldName("department")
                ),
            )


# where.


class Stage(enum.Enum):
    LEAD = "lead"
    WON = "won"


class Score(pydantic.RootModel[int], frozen=True):
    """A RootModel with an order, so conditions can order it."""

    def __lt__(self, other: Score) -> bool:
        return self.root < other.root

    def __le__(self, other: Score) -> bool:
        return self.root <= other.root

    def __gt__(self, other: Score) -> bool:
        return self.root > other.root

    def __ge__(self, other: Score) -> bool:
        return self.root >= other.root


class Deal(pydantic.BaseModel):
    name: Text
    nickname: Text | None
    stage: Stage
    score: Score
    target: Score
    bonus: Score | None


def compile_deal(step: steps.Step) -> compiler.CompiledView:
    match compiler.compile_views(
        empty_graph(), [View(name=ViewName.fake(), source=Deal, step=step)]
    ):
        case Ok(compiled):
            return compiled.views[0]
        case Err(error):
            pytest.fail(f"expected a compiled view, got {error}")


def deal_diagnostics(step: steps.Step) -> tuple[diagnostics.CompileDiagnostic, ...]:
    return diagnostics_of([View(name=ViewName.fake(), source=Deal, step=step)])


def column(name: FieldName) -> ir.SourceField:
    return ir.SourceField(field=name)


def text_literal(value: Text) -> ir.RootModelLiteral:
    return ir.RootModelLiteral(instance=value, root=ir.ScalarLiteral(value.root))


def test_where_keeps_every_field_and_filters_by_the_sources_own_field() -> None:
    acme = Text("Acme")
    view = compile_deal(
        steps.where(conditions.eq(conditions.field(FieldName("name")), acme))
    )

    assert view.relation.fields == ir.Relation.of(ir.FromSource(source=Deal)).fields
    assert view.relation.filters == (
        ir.FilterComparison(
            operator=ir.ComparisonOperator.EQ,
            field=column(FieldName("name")),
            operand=text_literal(acme),
        ),
    )


def test_a_condition_names_a_field_by_its_name_after_renames() -> None:
    view = compile_deal(
        steps.pipeline(
            steps.rename(FieldName("score"), FieldName("points")),
            steps.rename(FieldName("target"), FieldName("score")),
            steps.where(
                conditions.lt(
                    conditions.field(FieldName("points")),
                    conditions.field(FieldName("score")),
                )
            ),
        )
    )

    assert view.relation.filters == (
        ir.FilterComparison(
            operator=ir.ComparisonOperator.LT,
            field=column(FieldName("score")),
            operand=column(FieldName("target")),
        ),
    )


def test_a_filter_survives_omitting_the_field_it_uses() -> None:
    view = compile_deal(
        steps.pipeline(
            steps.where(conditions.is_null(conditions.field(FieldName("nickname")))),
            steps.omit(FieldName("nickname")),
        )
    )

    assert view.relation.filters == (
        ir.FilterIsNull(field=column(FieldName("nickname"))),
    )


@pytest.mark.parametrize(
    ("condition", "checked"),
    [
        (
            conditions.eq(conditions.field(FieldName("nickname")), None),
            ir.FilterIsNull(field=column(FieldName("nickname"))),
        ),
        (
            conditions.ne(conditions.field(FieldName("nickname")), None),
            ir.FilterNot(term=ir.FilterIsNull(field=column(FieldName("nickname")))),
        ),
    ],
    ids=["eq", "ne"],
)
def test_comparing_with_none_is_checked_into_is_null(
    condition: ir.Condition, checked: ir.Filter
) -> None:
    view = compile_deal(steps.where(condition))

    assert view.relation.filters == (checked,)


def test_and_or_and_not_keep_their_structure() -> None:
    view = compile_deal(
        steps.where(
            conditions.or_(
                conditions.and_(
                    conditions.eq(conditions.field(FieldName("stage")), Stage.WON),
                    conditions.ge(conditions.field(FieldName("score")), Score(3)),
                ),
                conditions.not_(
                    conditions.is_null(conditions.field(FieldName("bonus")))
                ),
            )
        )
    )

    assert view.relation.filters == (
        ir.FilterOr(
            terms=(
                ir.FilterAnd(
                    terms=(
                        ir.FilterComparison(
                            operator=ir.ComparisonOperator.EQ,
                            field=column(FieldName("stage")),
                            operand=ir.EnumLiteral(member=Stage.WON),
                        ),
                        ir.FilterComparison(
                            operator=ir.ComparisonOperator.GE,
                            field=column(FieldName("score")),
                            operand=ir.RootModelLiteral(
                                instance=Score(3), root=ir.ScalarLiteral(3)
                            ),
                        ),
                    )
                ),
                ir.FilterNot(term=ir.FilterIsNull(field=column(FieldName("bonus")))),
            )
        ),
    )


def test_each_where_adds_one_filter_in_order() -> None:
    first = conditions.is_null(conditions.field(FieldName("nickname")))
    second = conditions.is_null(conditions.field(FieldName("bonus")))

    view = compile_deal(steps.pipeline(steps.where(first), steps.where(second)))

    assert view.relation.filters == (
        ir.FilterIsNull(field=column(FieldName("nickname"))),
        ir.FilterIsNull(field=column(FieldName("bonus"))),
    )


def test_an_optional_field_equals_a_literal_of_its_type() -> None:
    nickname = Text("Ace")
    view = compile_deal(
        steps.where(conditions.eq(conditions.field(FieldName("nickname")), nickname))
    )

    assert view.relation.filters == (
        ir.FilterComparison(
            operator=ir.ComparisonOperator.EQ,
            field=column(FieldName("nickname")),
            operand=text_literal(nickname),
        ),
    )


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (
            steps.where(conditions.is_null(conditions.field(FieldName("nicknme")))),
            diagnostics.UnknownField(
                view=ViewName.fake(),
                source=ir.SourceName("Deal"),
                field=FieldName("nicknme"),
                valid=tuple(map(FieldName, Deal.model_fields)),
            ),
        ),
        (
            steps.pipeline(
                steps.add(FieldName("kind"), Literal["deal"], default="deal"),
                steps.where(conditions.eq(conditions.field(FieldName("kind")), "deal")),
            ),
            diagnostics.NotASourceColumn(
                view=ViewName.fake(),
                source=ir.SourceName("Deal"),
                field=FieldName("kind"),
            ),
        ),
        (
            steps.where(
                conditions.eq(conditions.field(FieldName("name")), [Text("Acme")])
            ),
            diagnostics.NonLiteralConditionValue(
                view=ViewName.fake(),
                value=diagnostics.LiteralText.of([Text("Acme")]),
            ),
        ),
        (
            steps.where(conditions.eq(conditions.field(FieldName("name")), "Acme")),
            diagnostics.LiteralTypeMismatch(
                view=ViewName.fake(),
                source=ir.SourceName("Deal"),
                field=FieldName("name"),
                annotation=Text,
                value=diagnostics.LiteralText.of("Acme"),
            ),
        ),
        (
            steps.where(conditions.eq(conditions.field(FieldName("name")), None)),
            diagnostics.LiteralTypeMismatch(
                view=ViewName.fake(),
                source=ir.SourceName("Deal"),
                field=FieldName("name"),
                annotation=Text,
                value=diagnostics.LiteralText.of(None),
            ),
        ),
        (
            steps.where(
                conditions.eq(
                    conditions.field(FieldName("name")),
                    conditions.field(FieldName("score")),
                )
            ),
            diagnostics.ComparedTypesDiffer(
                view=ViewName.fake(),
                source=ir.SourceName("Deal"),
                field=FieldName("name"),
                annotation=Text,
                other=FieldName("score"),
                other_annotation=Score,
            ),
        ),
        (
            steps.where(conditions.gt(conditions.field(FieldName("bonus")), Score(1))),
            diagnostics.OrderingOnOptionalField(
                view=ViewName.fake(),
                source=ir.SourceName("Deal"),
                field=FieldName("bonus"),
                operator=ir.ComparisonOperator.GT,
            ),
        ),
        (
            steps.where(
                conditions.le(
                    conditions.field(FieldName("score")),
                    conditions.field(FieldName("bonus")),
                )
            ),
            diagnostics.OrderingOnOptionalField(
                view=ViewName.fake(),
                source=ir.SourceName("Deal"),
                field=FieldName("bonus"),
                operator=ir.ComparisonOperator.LE,
            ),
        ),
        (
            steps.where(conditions.lt(conditions.field(FieldName("name")), Text("M"))),
            diagnostics.UnorderedFieldType(
                view=ViewName.fake(),
                source=ir.SourceName("Deal"),
                field=FieldName("name"),
                annotation=Text,
                operator=ir.ComparisonOperator.LT,
            ),
        ),
    ],
    ids=[
        "unknown-field",
        "added-field",
        "non-literal",
        "unwrapped-root",
        "none-for-required",
        "types-differ",
        "ordering-optional-field",
        "ordering-optional-operand",
        "unordered-type",
    ],
)
def test_a_defective_condition_is_a_diagnostic(
    step: steps.Step, expected: diagnostics.CompileDiagnostic
) -> None:
    assert deal_diagnostics(step) == (expected,)


def test_a_condition_on_a_resolved_field_is_a_diagnostic() -> None:
    reported = resolve_diagnostics(
        Employee,
        steps.pipeline(
            steps.hydrate(),
            steps.where(conditions.is_null(conditions.field(FieldName("mentor")))),
        ),
    )

    assert reported == (
        diagnostics.NotASourceColumn(
            view=ViewName.fake(),
            source=ir.SourceName("Employee"),
            field=FieldName("mentor"),
        ),
    )


def test_every_defective_term_of_a_condition_is_reported() -> None:
    unknown = FieldName("nicknme")

    reported = deal_diagnostics(
        steps.where(
            conditions.and_(
                conditions.is_null(conditions.field(unknown)),
                conditions.not_(
                    conditions.lt(conditions.field(FieldName("name")), Text("M"))
                ),
            )
        )
    )

    assert [type(diagnostic) for diagnostic in reported] == [
        diagnostics.UnknownField,
        diagnostics.UnorderedFieldType,
    ]


def test_steps_after_a_defective_where_are_still_checked() -> None:
    reported = deal_diagnostics(
        steps.pipeline(
            steps.where(conditions.eq(conditions.field(FieldName("name")), "Acme")),
            steps.omit(FieldName("secret")),
        )
    )

    assert [type(diagnostic) for diagnostic in reported] == [
        diagnostics.LiteralTypeMismatch,
        diagnostics.UnknownField,
    ]


def test_each_where_in_a_payload_is_a_diagnostic() -> None:
    where = steps.where(conditions.is_null(conditions.field(FieldName("nickname"))))
    payload = Payload(
        name=PayloadName.fake(), source=Deal, step=steps.pipeline(where, where)
    )

    match compile_catalog([], [payload]):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            assert (
                error.diagnostics
                == (diagnostics.WhereInPayload(view=payload.name),) * 2
            )


def test_each_where_in_a_nested_pipeline_is_a_diagnostic() -> None:
    where = steps.where(conditions.is_null(conditions.field(FieldName("mentor_id"))))

    reported = resolve_diagnostics(
        Employee,
        steps.resolve(named(FieldName("mentor_id")), where, where),
    )

    assert (
        reported
        == (
            diagnostics.WhereInNestedPipeline(
                view=ViewName.fake(), source=ir.SourceName("Employee")
            ),
        )
        * 2
    )


type MaybeScore = Score | None


class Rival(pydantic.BaseModel):
    """A Source whose optional field is typed by a `type` alias."""

    score: MaybeScore


@pytest.mark.parametrize(
    ("condition", "outcome"),
    [
        (
            conditions.eq(conditions.field(FieldName("score")), None),
            ir.FilterIsNull(field=ir.SourceField(field=FieldName("score"))),
        ),
        (
            conditions.gt(conditions.field(FieldName("score")), Score(1)),
            diagnostics.OrderingOnOptionalField(
                view=ViewName.fake(),
                source=ir.SourceName("Rival"),
                field=FieldName("score"),
                operator=ir.ComparisonOperator.GT,
            ),
        ),
    ],
    ids=["eq-none", "ordering"],
)
def test_a_field_typed_by_an_optional_alias_is_optional(
    condition: ir.Condition, outcome: ir.Filter | diagnostics.CompileDiagnostic
) -> None:
    match compiler.compile_views(
        empty_graph(),
        [View(name=ViewName.fake(), source=Rival, step=steps.where(condition))],
    ):
        case Ok(compiled):
            assert compiled.views[0].relation.filters == (outcome,)
        case Err(error):
            assert error.diagnostics == (outcome,)


type Ratio = float
type Tally = int


class Measure(pydantic.BaseModel):
    ratio: Ratio
    count: Tally


def measure_view(field: FieldName, value: ir.UncheckedLiteral) -> View:
    return View(
        name=ViewName.fake(),
        source=Measure,
        step=steps.where(conditions.eq(conditions.field(field), value)),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [(FieldName("ratio"), 1), (FieldName("ratio"), 0.5), (FieldName("count"), 1)],
    ids=["int-for-float", "float-for-float", "int-for-int"],
)
def test_a_number_of_the_fields_kind_is_a_literal_of_its_type(
    field: FieldName, value: ir.UncheckedLiteral
) -> None:
    view = compile_view_of(measure_view(field, value))

    assert len(view.relation.filters) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [(FieldName("count"), True), (FieldName("ratio"), True), (FieldName("count"), 0.5)],
    ids=["bool-for-int", "bool-for-float", "float-for-int"],
)
def test_a_bool_or_float_for_a_narrower_number_field_is_a_diagnostic(
    field: FieldName, value: ir.UncheckedLiteral
) -> None:
    view = measure_view(field, value)

    assert diagnostics_of([view]) == (
        diagnostics.LiteralTypeMismatch(
            view=view.name,
            source=ir.SourceName("Measure"),
            field=field,
            annotation=Measure.model_fields[field.root].annotation,
            value=diagnostics.LiteralText.of(value),
        ),
    )


def compile_view_of(view: View) -> compiler.CompiledView:
    match compiler.compile_views(empty_graph(), [view]):
        case Ok(compiled):
            return compiled.views[0]
        case Err(error):
            pytest.fail(f"expected a compiled view, got {error}")


def test_building_a_where_checks_nothing() -> None:
    root = ir.FromSource(source=Deal)
    condition = conditions.eq(conditions.field(FieldName("missing")), object())

    assert steps.where(condition)(root) == ir.Where(input=root, condition=condition)


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            diagnostics.WhereInPayload(view=PayloadName("DealCreate")),
            (
                "DealCreate: a Payload cannot filter with `where`, because it is a "
                "write model with no rows to keep."
            ),
        ),
        (
            diagnostics.WhereInNestedPipeline(
                view=ViewName("EmployeeView"), source=ir.SourceName("Employee")
            ),
            (
                "EmployeeView: cannot filter the resolved Employee with `where`, "
                "because only a View's own pipeline can filter its rows."
            ),
        ),
        (
            diagnostics.NotASourceColumn(
                view=ViewName("DealView"),
                source=ir.SourceName("Deal"),
                field=FieldName("kind"),
            ),
            (
                "DealView: a condition cannot use 'kind', because its value is not "
                "one of Deal's own fields."
            ),
        ),
        (
            diagnostics.NonLiteralConditionValue(
                view=ViewName("DealView"), value=diagnostics.LiteralText.of([1])
            ),
            (
                "DealView: a condition cannot compare with [1]. A literal is None, a "
                "boolean, a finite number, a string, an enum member, or a RootModel "
                "of one of these."
            ),
        ),
        (
            diagnostics.LiteralTypeMismatch(
                view=ViewName("DealView"),
                source=ir.SourceName("Deal"),
                field=FieldName("name"),
                annotation=int,
                value=diagnostics.LiteralText.of("Acme"),
            ),
            (
                "DealView: a condition compares Deal.name, typed <class 'int'>, with "
                "'Acme', which is not of that type."
            ),
        ),
        (
            diagnostics.ComparedTypesDiffer(
                view=ViewName("DealView"),
                source=ir.SourceName("Deal"),
                field=FieldName("name"),
                annotation=int,
                other=FieldName("score"),
                other_annotation=str,
            ),
            (
                "DealView: a condition compares Deal.name, typed <class 'int'>, with "
                "'score', typed <class 'str'>."
            ),
        ),
        (
            diagnostics.OrderingOnOptionalField(
                view=ViewName("DealView"),
                source=ir.SourceName("Deal"),
                field=FieldName("bonus"),
                operator=ir.ComparisonOperator.GT,
            ),
            "DealView: cannot compare Deal.bonus with '>', because it is optional.",
        ),
        (
            diagnostics.UnorderedFieldType(
                view=ViewName("DealView"),
                source=ir.SourceName("Deal"),
                field=FieldName("name"),
                annotation=int,
                operator=ir.ComparisonOperator.LT,
            ),
            (
                "DealView: cannot compare Deal.name with '<', because its type "
                "<class 'int'> does not define it."
            ),
        ),
    ],
    ids=[
        "payload",
        "nested",
        "not-a-column",
        "non-literal",
        "literal-type",
        "types-differ",
        "ordering-optional",
        "unordered",
    ],
)
def test_a_where_diagnostic_message_names_the_offending_values(
    diagnostic: diagnostics.CompileDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


@pytest.mark.parametrize(
    "combine", [conditions.and_, conditions.or_], ids=["and", "or"]
)
def test_an_and_or_an_or_of_one_term_is_that_term(
    combine: Callable[[ir.Condition], ir.Condition],
) -> None:
    term = conditions.is_null(conditions.field(FieldName("nickname")))

    view = compile_deal(steps.where(combine(term)))

    assert view.relation.filters == (
        ir.FilterIsNull(field=column(FieldName("nickname"))),
    )
