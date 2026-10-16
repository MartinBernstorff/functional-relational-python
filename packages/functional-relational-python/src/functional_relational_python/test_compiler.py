import pydantic
import pytest
from hypothesis import given
from hypothesis import strategies as st
from safe_result import Err, Ok

from functional_relational_python import compiler, diagnostics, ir, steps
from functional_relational_python.entity_graph import EntityGraph
from functional_relational_python.views import View, ViewName


class Text(pydantic.RootModel[str], frozen=True):
    pass


class Company(pydantic.BaseModel):
    name: Text
    secret: Text
    revenue: Text


def field_names(view: compiler.CompiledView) -> list[ir.FieldName]:
    return [field.name for field in view.relation.fields]


def names(*fields: Text) -> list[ir.FieldName]:
    return [ir.FieldName(field.root) for field in fields]


def compile_one(step: steps.Step) -> compiler.CompiledView:
    match compiler.compile_views(
        EntityGraph.fake(), [View(name=ViewName.fake(), source=Company, step=step)]
    ):
        case Ok(compiled):
            return compiled.views[0]
        case Err(error):
            pytest.fail(f"expected a compiled view, got {error}")


def diagnostics_of(views: list[View]) -> tuple[diagnostics.Diagnostic, ...]:
    match compiler.compile_views(EntityGraph.fake(), views):
        case Ok(compiled):
            pytest.fail(f"expected diagnostics, got {compiled}")
        case Err(error):
            return error.diagnostics


def test_omit_removes_the_field_from_the_source() -> None:
    view = compile_one(steps.omit(ir.FieldName("secret")))

    assert field_names(view) == names(Text("name"), Text("revenue"))


@pytest.mark.parametrize(
    "step",
    [
        steps.pipeline(
            steps.omit(ir.FieldName("secret")), steps.omit(ir.FieldName("revenue"))
        ),
        steps.pipeline(
            steps.pipeline(steps.omit(ir.FieldName("secret"))),
            steps.omit(ir.FieldName("revenue")),
        ),
        steps.pipeline(
            steps.omit(ir.FieldName("secret")),
            steps.pipeline(steps.pipeline(), steps.omit(ir.FieldName("revenue"))),
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
        return steps.omit(ir.FieldName("revenue"))(node)

    view = compile_one(
        steps.pipeline(without_financials, steps.omit(ir.FieldName("secret")))
    )

    assert field_names(view) == names(Text("name"))


def test_an_unknown_field_names_the_source_the_field_and_the_valid_names() -> None:
    unknown = ir.FieldName("revnue")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.pipeline(steps.omit(ir.FieldName("secret")), steps.omit(unknown)),
    )

    assert diagnostics_of([view]) == (
        diagnostics.UnknownField(
            view=view.name,
            source=ir.SourceName("Company"),
            field=unknown,
            valid=(ir.FieldName("name"), ir.FieldName("revenue")),
        ),
    )


def test_the_unknown_field_message_contains_the_offending_values() -> None:
    message = diagnostics.UnknownField(
        view=ViewName("CompanySummary"),
        source=ir.SourceName("Company"),
        field=ir.FieldName("revnue"),
        valid=(ir.FieldName("name"), ir.FieldName("revenue")),
    ).message()

    assert message.root == (
        "CompanySummary: Company has no field 'revnue'. "
        "Valid fields: 'name', 'revenue'."
    )


def test_compiled_views_keep_their_explicit_names_in_order() -> None:
    names = [ViewName("CompanyPublic"), ViewName("CompanyInternal")]

    match compiler.compile_views(
        EntityGraph.fake(),
        [View(name=name, source=Company, step=steps.pipeline()) for name in names],
    ):
        case Ok(compiled):
            assert [view.name for view in compiled.views] == names
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def test_building_a_pipeline_checks_nothing() -> None:
    unknown = ir.FieldName("does_not_exist")
    root = ir.FromSource(source=Company)

    node = steps.pipeline(steps.omit(unknown))(root)

    assert node == ir.Omit(input=root, field=unknown)


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
    unknown_fields: tuple[ir.FieldName, ...]
    defective_views: frozenset[ViewName]


@st.composite
def views_with_defects(draw: st.DrawFn) -> ViewsWithDefects:
    """Views with known unknown-field omits.

    Every unknown name is independent of the others: it is never a field of the
    Source, and valid omits use each field at most once.
    """
    views: list[View] = []
    unknown_fields: list[ir.FieldName] = []
    defective_views: set[ViewName] = set()
    for index in range(draw(st.integers(min_value=0, max_value=4))):
        source = draw(st.sampled_from([Company, Wide]))
        valid = draw(st.lists(st.sampled_from(list(source.model_fields)), unique=True))
        unknown = draw(st.lists(unknown_field_name, max_size=3))
        omits = draw(st.permutations([*valid, *unknown]))
        name = ViewName(f"View{index}")
        unknown_fields.extend(map(ir.FieldName, unknown))
        if unknown:
            defective_views.add(name)
        views.append(
            View(
                name=name,
                source=source,
                step=steps.pipeline(*(steps.omit(ir.FieldName(f)) for f in omits)),
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
    match compiler.compile_views(EntityGraph.fake(), case.views):
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
    checked = compiler.check(case.views)

    poisoned = [view for view in checked if isinstance(view, compiler.PoisonedView)]
    assert {view.name for view in poisoned} == case.defective_views
    for view in poisoned:
        assert view.diagnostics
        assert all(d.view == view.name for d in view.diagnostics)
