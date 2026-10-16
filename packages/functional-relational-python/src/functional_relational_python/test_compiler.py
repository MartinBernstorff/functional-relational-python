import types
from collections.abc import Iterable, Mapping, Sequence

import pydantic
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from safe_result import Err, Ok

from functional_relational_python import compiler, diagnostics, engine, ir, steps
from functional_relational_python.codegen import RuntimeAnnotation
from functional_relational_python.entity_graph import (
    EntityGraph,
    FieldName,
    Optionality,
)
from functional_relational_python.id import Id, IdType
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


def new_model(
    name: ir.SourceName, fields: Mapping[FieldName, RuntimeAnnotation]
) -> type[pydantic.BaseModel]:
    annotations = {field.root: annotation for field, annotation in fields.items()}
    model = types.new_class(
        name.root,
        (pydantic.BaseModel,),
        {"frozen": True},
        lambda namespace: namespace.update(
            {"__module__": __name__, "__annotations__": annotations}
        ),
    )
    assert issubclass(model, pydantic.BaseModel)
    return model


@st.composite
def entity_model(
    draw: st.DrawFn, name: ir.SourceName, id_type: IdType, targets: Sequence[IdType]
) -> type[pydantic.BaseModel]:
    """An Entity keyed by `id_type`, with Text fields and References to `targets`.

    Field names never collide, before or after resolving: the j-th Reference is
    `r{j}_id` or `r{j}`, and Text fields are `f{j}`.
    """
    fields: dict[FieldName, RuntimeAnnotation] = {FieldName("id"): id_type}
    referenced = draw(st.lists(st.sampled_from(targets), max_size=3))
    for index, target in enumerate(referenced):
        field = draw(st.sampled_from([f"r{index}_id", f"r{index}"]))
        fields[FieldName(field)] = draw(st.sampled_from([target, target | None]))
    for index in range(draw(st.integers(min_value=0, max_value=2))):
        fields[FieldName(f"f{index}")] = Text
    order = draw(st.permutations(list(fields)))
    return new_model(name, {field: fields[field] for field in order})


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
        return EntityGraph.build(self.entities).unwrap()

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
    memo = new_model(
        ir.SourceName("Memo"),
        {
            FieldName("r0_id"): draw(st.sampled_from(id_types)),
            FieldName("r1_id"): new_id_type(ir.SourceName("Unkeyed")),
            FieldName("f0"): Text,
        },
    )
    clash = new_model(
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


@st.composite
def pipelines(
    draw: st.DrawFn,
    graph: EntityGraph,
    source: type[pydantic.BaseModel],
    depth: Depth,
) -> steps.Step:
    """A valid pipeline of omits, renames, projects and resolves over `source`.

    Each step names only fields the earlier steps kept, and each rename gets a name
    no field has, so it never collides. A resolve selects Reference fields only,
    and its nested pipeline is drawn the same way over the referenced Entity.
    """
    fields = list(map(FieldName, source.model_fields))
    references = {
        reference.field: reference.target
        for reference in graph.entities[source].references
    }
    drawn: list[steps.Step] = []
    for index in range(draw(st.integers(min_value=0, max_value=len(fields)))):
        if not fields:
            break
        field = draw(st.sampled_from(fields))
        match draw(st.sampled_from(["omit", "rename", "project", "resolve"])):
            case "omit":
                fields.remove(field)
                drawn.append(steps.omit(field))
            case "rename":
                new = FieldName(f"renamed_{index}")
                fields[fields.index(field)] = new
                if field in references:
                    references[new] = references.pop(field)
                drawn.append(steps.rename(field, new))
            case "project":
                projected = draw(st.lists(st.sampled_from(fields), unique=True))
                fields = [kept for kept in fields if kept in projected]
                drawn.append(steps.project(projected))
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

    Unlike `pipelines`, it may omit unknown fields, rename onto existing ones,
    resolve fields that are not References, or resolve onto an existing name.
    """
    names = st.sampled_from(pool)
    match draw(st.sampled_from(["omit", "rename", "project", "resolve"])):
        case "omit":
            return steps.omit(draw(names))
        case "rename":
            return steps.rename(draw(names), draw(names))
        case "project":
            return steps.project(draw(st.lists(names, unique=True)))
        case _:
            selection = draw(
                st.just(steps.select(steps.is_reference))
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


class TotalityCase(pydantic.BaseModel, frozen=True):
    views: ViewCase
    stored: tuple[pydantic.BaseModel, ...]


@st.composite
def totality_cases(draw: st.DrawFn) -> TotalityCase:
    views = draw(view_cases())
    return TotalityCase(views=views, stored=draw(stored_entities(views.models)))


def has_unique_field_names(relation: ir.Relation) -> bool:
    """Whether no two fields share a name, in nested relations too."""
    names = relation.field_names()
    return len(set(names)) == len(names) and all(
        has_unique_field_names(field.relation)
        for field in relation.fields
        if isinstance(field, ir.NestedField)
    )


@given(totality_cases())
def test_every_view_either_gives_results_or_reports_diagnostics(
    case: TotalityCase,
) -> None:
    graph = case.views.models.graph()
    checked = compiler.check(graph, case.views.views)
    compiled = tuple(v for v in checked if isinstance(v, compiler.CompiledView))
    view_engine = engine.InMemoryViewEngine(
        graph, compiler.CompiledViews(views=compiled)
    )
    for entity in case.stored:
        view_engine.put(entity).unwrap()

    assert [view.name for view in checked] == [view.name for view in case.views.views]
    assert case.views.valid <= {view.name for view in compiled}
    for view in checked:
        event(f"valid by construction: {view.name in case.views.valid}")
        match view:
            case compiler.CompiledView():
                assert has_unique_field_names(view.relation)
                rows = view_engine.read(view.name).unwrap()
                event(f"compiled view has rows: {bool(rows)}")
                assert all(
                    tuple(row.root) == view.relation.field_names() for row in rows
                )
                assert len(rows) == sum(
                    type(entity) is view.source for entity in case.stored
                )
            case compiler.PoisonedView(diagnostics=reported):
                for diagnostic in reported:
                    event(f"diagnostic: {type(diagnostic).__name__}")
                    assert diagnostic.view == view.name
                    assert diagnostic.describe().root
    match compiler.compile_views(graph, case.views.views):
        case Ok(all_compiled):
            assert shapes(all_compiled.views) == shapes(compiled)
        case Err(error):
            assert error.diagnostics == tuple(
                diagnostic
                for view in checked
                if isinstance(view, compiler.PoisonedView)
                for diagnostic in view.diagnostics
            )


class PlainShape(pydantic.BaseModel, frozen=True):
    """An `ir.Field` without its FieldInfo, which compares by identity."""

    name: FieldName
    annotation: RuntimeAnnotation
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
    varied = compiler.check(EntityGraph.build(case.varied).unwrap(), case.views.views)

    reordered = [e for e in case.varied if e not in case.unreferenced] != list(
        case.views.models.entities
    )
    event(f"entities reordered: {reordered}")
    event(f"unreferenced entities added: {len(case.unreferenced)}")
    for view in original:
        event(f"view: {type(view).__name__}")
    assert shapes(varied) == shapes(original)


class RoundTripCase(pydantic.BaseModel, frozen=True):
    models: Models
    source: type[pydantic.BaseModel]
    resolved: tuple[FieldName, ...]
    stored: tuple[pydantic.BaseModel, ...]


@st.composite
def round_trip_cases(draw: st.DrawFn) -> RoundTripCase:
    drawn_models = draw(models())
    graph = drawn_models.graph()
    # Only a graph without References gives a source that resolves nothing.
    referencing = [
        model for model in drawn_models.entities if graph.entities[model].references
    ]
    source = draw(st.sampled_from(referencing or drawn_models.entities))
    references = [reference.field for reference in graph.entities[source].references]
    resolved = (
        draw(st.lists(st.sampled_from(references), unique=True, min_size=1))
        if references
        else []
    )
    return RoundTripCase(
        models=drawn_models,
        source=source,
        resolved=tuple(resolved),
        stored=draw(stored_entities(drawn_models)),
    )


def with_ids_again(
    source: type[pydantic.BaseModel], relation: ir.Relation, row: engine.ViewRow
) -> pydantic.BaseModel:
    """The `source` instance `row` was built from: each resolved Entity is replaced
    by its ID, under the Reference's own name."""
    values: dict[str, object] = {}
    for field in relation.fields:
        value = row.root[field.name]
        match field.value:
            case ir.SourceField(field=name):
                values[name.root] = value
            case ir.Lookup(reference=reference):
                values[reference.field.root] = (
                    vars(value)["id"] if isinstance(value, pydantic.BaseModel) else None
                )
    return source(**values)


@given(round_trip_cases())
def test_resolving_then_replacing_entities_by_their_ids_gives_the_original(
    case: RoundTripCase,
) -> None:
    graph = case.models.graph()
    view = View(
        name=ViewName.fake(),
        source=case.source,
        step=steps.resolve(named(*case.resolved)),
    )
    compiled = compiler.compile_views(graph, [view]).unwrap()
    view_engine = engine.InMemoryViewEngine(graph, compiled)
    for entity in case.stored:
        view_engine.put(entity).unwrap()

    rows = view_engine.read(view.name).unwrap()

    (compiled_view,) = compiled.views
    absent = [
        field.name
        for field in compiled_view.relation.fields
        for row in rows
        if isinstance(field.value, ir.Lookup) and row.root[field.name] is None
    ]
    event(f"references resolved: {len(case.resolved)}")
    event(f"an optional reference is absent: {bool(absent)}")
    assert [
        with_ids_again(case.source, compiled_view.relation, row) for row in rows
    ] == [entity for entity in case.stored if type(entity) is case.source]
