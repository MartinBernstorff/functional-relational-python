import enum
import types
from collections.abc import Iterable, Sequence
from typing import Annotated, Literal

import pydantic
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from pydantic import MISSING
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler, diagnostics, model_names
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
    RuntimeAnnotation,
)
from functional_relational_python.entity_graph.id import Id, IdBase, IdType
from functional_relational_python.pipeline import ir, steps, test_ir
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
    return EntityGraph.build([], IdBase.fake()).unwrap()


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
    return EntityGraph.build([Department, Employee], IdBase.fake()).unwrap()


def named(*fields: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in fields)


def is_nested(field: ir.RelationField) -> bool:
    return isinstance(field, ir.NestedField)


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
    graph = EntityGraph.build([Team, Member], IdBase(AppId)).unwrap()

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
        case ir.NestedField(name=name, value=lookup, relation=relation):
            assert (name, lookup.optionality, relation.field_names()) == (
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
        FieldName("id"): ir.Lifted(lookup=department_lookup(), value=own(Text("id"))),
        FieldName("name"): ir.Lifted(
            lookup=department_lookup(), value=own(Text("name"))
        ),
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
            steps.flatten(named(FieldName("department"))),
        ),
    )

    assert values(view) == {
        FieldName("id"): own(Text("id")),
        FieldName("name"): own(Text("name")),
        FieldName("department_name"): ir.Lifted(
            lookup=department_lookup(), value=own(Text("name"))
        ),
        FieldName("mentor_id"): own(Text("mentor_id")),
    }
    assert defaults(view) == {FieldName("department_name"): ir.ScalarDefault("Sales")}


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


def test_a_resolved_field_lifted_out_looks_up_through_the_flattened_entity() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            resolve_mentor(steps.resolve(named(FieldName("mentor_department_id")))),
            steps.flatten(FieldName("mentor")),
        ),
    )

    assert values(view)[FieldName("mentor_department")] == ir.Lifted(
        lookup=mentor_lookup(), value=department_lookup()
    )
    assert annotations(view.relation)[FieldName("mentor_department")] == (
        Department | None
    )
    assert view.relation.lookups() == (mentor_lookup(), department_lookup())


def test_a_field_flattened_in_a_nested_pipeline_can_be_lifted_again() -> None:
    view = compile_resolving(
        Employee,
        steps.pipeline(
            resolve_mentor(
                steps.resolve(
                    named(FieldName("mentor_department_id")),
                    steps.omit(FieldName("id")),
                    steps.rename(FieldName("name"), FieldName("department_name")),
                ),
                steps.flatten(FieldName("mentor_department")),
            ),
            steps.flatten(FieldName("mentor")),
        ),
    )

    assert values(view)[FieldName("department_name")] == ir.Lifted(
        lookup=mentor_lookup(),
        value=ir.Lifted(lookup=department_lookup(), value=own(Text("name"))),
    )


def flatten_diagnostics(
    *pipeline: steps.Step,
) -> tuple[diagnostics.CompileDiagnostic, ...]:
    return resolve_diagnostics(Employee, steps.pipeline(*pipeline))


def department_collisions() -> tuple[diagnostics.CompileDiagnostic, ...]:
    """Flattening a whole Department into an Employee: `id` and `name` collide."""
    return tuple(
        diagnostics.FlattenOntoExistingField(
            view=ViewName.fake(), flattened=FieldName("department"), field=field
        )
        for field in (FieldName("id"), FieldName("name"))
    )


@pytest.mark.parametrize(
    "then",
    [(), (steps.omit(FieldName("id")), steps.omit(FieldName("name")))],
    ids=["alone", "steps after it use the existing fields"],
)
def test_a_lifted_field_onto_an_existing_name_is_a_diagnostic_per_field(
    then: tuple[steps.Step, ...],
) -> None:
    assert (
        flatten_diagnostics(
            steps.resolve(named(FieldName("department_id"))),
            steps.flatten(FieldName("department")),
            *then,
        )
        == department_collisions()
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
    ("pipeline", "expected"),
    [
        (
            (steps.flatten(FieldName("name")),),
            diagnostics.NotAResolvedField(
                view=ViewName.fake(),
                source=ir.SourceName("Employee"),
                field=FieldName("name"),
            ),
        ),
        (
            (steps.flatten(FieldName("budget")),),
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
        (
            (
                resolve_mentor(
                    steps.resolve(
                        named(FieldName("mentor_department_id")),
                        steps.omit(FieldName("id")),
                    )
                ),
                steps.flatten(FieldName("mentor")),
                # The fields it could lift are still there.
                steps.project([FieldName("mentor_name")]),
            ),
            diagnostics.FlattenNestedModel(
                view=ViewName.fake(),
                flattened=FieldName("mentor"),
                field=FieldName("mentor_department"),
            ),
        ),
    ],
    ids=["not resolved", "unknown", "nested model"],
)
def test_flattening_what_it_cannot_lift_is_one_diagnostic(
    pipeline: tuple[steps.Step, ...], expected: diagnostics.CompileDiagnostic
) -> None:
    assert flatten_diagnostics(*pipeline) == (expected,)


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
        (
            diagnostics.FlattenNestedModel(
                view=ViewName("EmployeeView"),
                flattened=FieldName("mentor"),
                field=FieldName("department"),
            ),
            (
                "EmployeeView: cannot lift 'department' out of 'mentor', because "
                "its type is the model of a nested pipeline. Flatten or omit it in "
                "the nested pipeline of the resolve."
            ),
        ),
    ],
    ids=["not resolved", "collision", "nested model"],
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
        return EntityGraph.build(self.entities, IdBase.fake()).unwrap()

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


def lifting(reference: FieldName, nested: steps.Step, prefix: Text) -> steps.Step:
    """Resolves `reference` through `nested`, then flattens it.

    Nested models cannot be lifted, so they are omitted. `prefix` keeps the lifted
    fields from colliding with any other.
    """
    return steps.pipeline(
        steps.resolve(
            named(reference),
            nested,
            steps.omit(steps.select(is_nested)),
            steps.rename(steps.select(lambda _field: True), prefixed(prefix)),
        ),
        steps.flatten(FieldName(reference.root.removesuffix("_id") or reference.root)),
    )


@st.composite
def pipelines(
    draw: st.DrawFn,
    graph: EntityGraph,
    source: type[pydantic.BaseModel],
    depth: Depth,
) -> steps.Step:
    """A valid pipeline of omits, renames, projects, adds, resolves and flattens
    over `source`.

    Omits and renames name one field or select several. It may start by adding a
    field.

    Each step names only fields the earlier steps kept, and each rename gets a name
    no field has, so it never collides. A resolve selects Reference fields only,
    and its nested pipeline is drawn the same way over the referenced Entity. A
    flatten lifts the fields of one such resolve, each renamed to a name no other
    field has. Later steps leave the lifted fields alone.
    """
    fields = list(map(FieldName, source.model_fields))
    references = {
        reference.field: reference.target
        for reference in graph.entities[source].references
    }
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
                for old in renamed:
                    fields[fields.index(old)] = renaming(old)
                    keep_target(references, old, renaming(old))
                drawn.append(steps.rename(named(*renamed), renaming))
            case "project":
                projected = draw(st.lists(st.sampled_from(fields), unique=True))
                fields = [kept for kept in fields if kept in projected]
                drawn.append(steps.project(projected))
            case "flatten":
                resolvable = [kept for kept in fields if kept in references]
                if not resolvable or depth.root == 0:
                    continue
                flattened = draw(st.sampled_from(resolvable))
                target = references.pop(flattened)
                nested = draw(pipelines(graph, target, Depth(depth.root - 1)))
                fields.remove(flattened)
                drawn.append(lifting(flattened, nested, Text(f"flat_{index}")))
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
    value: ir.Lookup
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
        EntityGraph.build(case.varied, IdBase.fake()).unwrap(), case.views.views
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
) -> tuple[tuple[ir.RelationField, ir.Lookup], ...]:
    """Every field holding a looked-up Entity, nested ones included."""
    found: list[tuple[ir.RelationField, ir.Lookup]] = []
    for field in relation.fields:
        match field:
            case ir.Field(value=ir.Lookup() as value):
                found.append((field, value))
            case ir.NestedField(value=value, relation=nested):
                found.extend(((field, value), *resolved_fields(nested)))
            case ir.Field():
                pass
    return tuple(found)


@given(compiled_views())
def test_a_resolved_field_is_optional_iff_its_lookup_is(
    view: compiler.CompiledView,
) -> None:
    for field, value in resolved_fields(view.relation):
        assert field.optionality() == value.optionality


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
        if any(
            lookup.optionality is Optionality.OPTIONAL for lookup in value.lookups()
        ):
            assert field.optionality() is Optionality.OPTIONAL


@given(compiled_views())
def test_lookups_name_each_looked_up_entity_once(
    view: compiler.CompiledView,
) -> None:
    targets = [found.target for found in view.relation.lookups()]

    assert sorted(targets, key=lambda target: target.__name__) == sorted(
        {
            *(value.target for _, value in resolved_fields(view.relation)),
            *(
                lookup.target
                for _, value in lifted_fields(view.relation)
                for lookup in value.lookups()
            ),
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
