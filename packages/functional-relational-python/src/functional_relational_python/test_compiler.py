import pydantic
import pytest
from hypothesis import given
from hypothesis import strategies as st
from safe_result import Err, Ok

from functional_relational_python import compiler, diagnostics, ir, steps
from functional_relational_python.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
)
from functional_relational_python.id import Id
from functional_relational_python.views import View, ViewName


class Text(pydantic.RootModel[str], frozen=True):
    pass


class Company(pydantic.BaseModel):
    name: Text
    secret: Text
    revenue: Text


def empty_graph() -> EntityGraph:
    # The Sources here are not Entities; compile does not require them to be.
    return EntityGraph.build([]).unwrap()


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


def diagnostics_of(views: list[View]) -> tuple[diagnostics.Diagnostic, ...]:
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
        input=ir.Omit(input=root, field=unknown), field=unknown, new=unknown
    )


def test_rename_keeps_the_field_in_place_with_its_type() -> None:
    view = compile_one(steps.rename(FieldName("secret"), FieldName("code")))

    assert view.relation.fields == tuple(
        ir.Field(
            name=FieldName("code") if name == "secret" else FieldName(name),
            info=info,
            value=ir.SourceField(field=FieldName(name)),
        )
        for name, info in Company.model_fields.items()
    )


@pytest.mark.parametrize(
    "existing",
    [FieldName("name"), FieldName("secret")],
    ids=["other-field", "itself"],
)
def test_renaming_onto_an_existing_field_is_a_diagnostic(
    existing: FieldName,
) -> None:
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
        "CompanySummary: cannot rename 'secret' to 'name', because the View "
        "already has a field 'name'."
    )


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
            assert sorted(d.field.root for d in error.diagnostics) == sorted(
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
    return EntityGraph.build([Department, Employee]).unwrap()


def named(*fields: FieldName) -> ir.Select:
    return steps.select(lambda field: field.name in fields)


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
) -> tuple[diagnostics.Diagnostic, ...]:
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
    diagnostic: diagnostics.Diagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message
