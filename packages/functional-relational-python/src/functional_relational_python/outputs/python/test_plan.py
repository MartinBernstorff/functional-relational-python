import enum
import typing
from collections.abc import Iterable

import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler, test_compiler
from functional_relational_python.entity_graph.entity_graph import (
    EntityGraph,
    EntityNode,
    FieldName,
)
from functional_relational_python.entity_graph.id import Id, IdBase, OneToOneMarker
from functional_relational_python.outputs.python.plan import (
    FunctionPlan,
    GeneratedIntoDefiningModule,
    LayoutRole,
    LookupParameter,
    ModelsAndMappingsInOneModule,
    NameClash,
    NoModuleFromLayout,
    PlanDiagnostic,
    PythonLayout,
    PythonPlan,
    ReturnedText,
    Scope,
    Unimportable,
    UnsupportedAnnotation,
)
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    ModulePath,
    Symbol,
)
from functional_relational_python.outputs.python.test_mappings import (
    Archive,
    Company,
    CompanyId,
    Department,
    DepartmentId,
    Employee,
    Profile,
    Text,
    Transfer,
    compile_all,
    employee_union,
    into,
    named,
)
from functional_relational_python.outputs.python.test_models import WithUnsupported
from functional_relational_python.pipeline import conditions, steps
from functional_relational_python.pipeline.unions import UnionName
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.support.identifier import Identifier


def layout(models: ModulePath, mappings: ModulePath) -> PythonLayout:
    return PythonLayout(models=into(models), mappings=into(mappings))


def app_layout() -> PythonLayout:
    """Models and mappings in two modules of one package."""
    return layout(ModulePath("app.views"), ModulePath("app.mappings"))


def plan_of(compiled: compiler.CompiledViews, python: PythonLayout) -> PythonPlan:
    match PythonPlan.of(
        compiler.CompiledCatalog(
            views=compiled.views,
            payloads=(),
            unions=compiled.unions,
            payload_unions=(),
            one_to_one_marker=OneToOneMarker.fake(),
        ),
        python,
    ):
        case Ok(plan):
            return plan
        case Err(error):
            pytest.fail(f"expected a plan, got {error}")


def diagnostics_of(
    compiled: compiler.CompiledViews, python: PythonLayout
) -> tuple[PlanDiagnostic, ...]:
    match PythonPlan.of(
        compiler.CompiledCatalog(
            views=compiled.views,
            payloads=(),
            unions=compiled.unions,
            payload_unions=(),
            one_to_one_marker=OneToOneMarker.fake(),
        ),
        python,
    ):
        case Ok(plan):
            pytest.fail(f"expected diagnostics, got {plan}")
        case Err(error):
            return error.diagnostics


def symbol(named_object: type) -> Symbol:
    return Symbol.of(named_object).unwrap()


def overview() -> View:
    """A View with a nested model, looking up two Entities."""
    return View(
        name=ViewName("EmployeeOverview"),
        source=Employee,
        step=steps.resolve(named(FieldName("department_id")), steps.hydrate()),
    )


def test_a_view_is_planned_into_the_modules_its_layout_gives() -> None:
    models, mappings = ModulePath("app.views"), ModulePath("app.mappings")

    plan = plan_of(compile_all([overview()]), layout(models, mappings))

    ((function,),) = plan.functions.values()
    assert plan.models == {models: (function.model,)}
    assert [model.symbol for model in function.model.with_nested()] == [
        Symbol(module=models, qualname=(Identifier("EmployeeOverviewDepartment"),)),
        Symbol(module=models, qualname=(Identifier("EmployeeOverview"),)),
    ]
    assert function.symbol == Symbol(
        module=mappings, qualname=(Identifier("employee_overview"),)
    )
    assert function.source == symbol(Employee)
    assert [
        (parameter.name, parameter.target, parameter.id_type)
        for parameter in function.lookups
    ] == [
        (
            LookupParameter.name_for(entity, EntityNode.primary_key),
            symbol(entity),
            symbol(id_type),
        )
        for entity, id_type in [(Department, DepartmentId), (Company, CompanyId)]
    ]


def test_a_models_only_layout_plans_no_functions() -> None:
    plan = plan_of(
        compile_all([overview()]), PythonLayout.models_only(into(ModulePath.fake()))
    )

    assert (len(plan.models), plan.functions) == (1, {})


@pytest.mark.parametrize(
    ("entity", "keyed_by", "expected"),
    [
        (Department, EntityNode.primary_key, "department_by_id"),
        (Profile, FieldName("employee_id"), "profile_by_employee_id"),
    ],
    ids=["resolved", "reversed"],
)
def test_lookup_parameters_are_named_after_the_entity_and_its_key(
    entity: type[pydantic.BaseModel], keyed_by: FieldName, expected: str
) -> None:
    assert LookupParameter.name_for(entity, keyed_by) == Identifier(expected)


def local_entity_and_source() -> tuple[type[pydantic.BaseModel], ...]:
    """An Entity, its ID type and a Source, all defined in a function."""

    class LocalTeamId(Id[int]):
        pass

    class Team(pydantic.BaseModel, frozen=True):
        id: LocalTeamId

    class Player(pydantic.BaseModel, frozen=True):
        team_id: LocalTeamId

    return Team, LocalTeamId, Player


def test_whatever_cannot_be_imported_is_reported_once() -> None:
    team, team_id, player = local_entity_and_source()
    view = View(name=ViewName.fake(), source=player, step=steps.hydrate())
    compiled = compiler.compile_views(
        EntityGraph.build([team], IdBase.fake(), OneToOneMarker.fake()).unwrap(), [view]
    ).unwrap()

    # The Entity is a field's type and a looked-up Entity, yet reported once,
    # for the field.
    assert diagnostics_of(compiled, app_layout()) == (
        Unimportable(
            view=view.name,
            field=FieldName("team"),
            annotation=AnnotationText.of(team),
        ),
        *(
            Unimportable(
                view=view.name, field=None, annotation=AnnotationText.of(unimportable)
            )
            for unimportable in (player, team_id)
        ),
    )


def test_a_models_only_layout_reports_only_what_the_models_cannot_import() -> None:
    team, _, player = local_entity_and_source()
    view = View(name=ViewName.fake(), source=player, step=steps.hydrate())
    compiled = compiler.compile_views(
        EntityGraph.build([team], IdBase.fake(), OneToOneMarker.fake()).unwrap(), [view]
    ).unwrap()

    assert diagnostics_of(
        compiled, PythonLayout.models_only(into(ModulePath.fake()))
    ) == (
        Unimportable(
            view=view.name,
            field=FieldName("team"),
            annotation=AnnotationText.of(team),
        ),
    )


def test_a_base_that_cannot_be_imported_is_reported() -> None:
    class LocalBase(pydantic.BaseModel):
        pass

    view = View(
        name=ViewName.fake(), source=Company, step=steps.pipeline(), base=LocalBase
    )

    assert diagnostics_of(compile_all([view]), app_layout()) == (
        Unimportable(
            view=view.name, field=None, annotation=AnnotationText.of(LocalBase)
        ),
    )


def test_a_default_whose_enum_cannot_be_imported_is_reported_for_its_field() -> None:
    class LocalColour(enum.Enum):
        RED = "red"

    field = FieldName("name")
    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.default(field, LocalColour.RED),
    )

    assert diagnostics_of(
        compile_all([view]), PythonLayout.models_only(into(ModulePath.fake()))
    ) == (
        Unimportable(
            view=view.name, field=field, annotation=AnnotationText.of(LocalColour)
        ),
    )


def test_a_filter_literal_whose_class_cannot_be_imported_is_reported() -> None:
    class LocalText(Text, frozen=True):
        pass

    view = View(
        name=ViewName.fake(),
        source=Company,
        step=steps.where(
            conditions.eq(conditions.field(FieldName("name")), LocalText("Acme"))
        ),
    )

    assert diagnostics_of(compile_all([view]), app_layout()) == (
        Unimportable(
            view=view.name, field=None, annotation=AnnotationText.of(LocalText)
        ),
    )


def test_every_unsupported_annotation_is_reported() -> None:
    view = View(name=ViewName.fake(), source=WithUnsupported, step=steps.pipeline())

    assert [
        diagnostic.field
        for diagnostic in diagnostics_of(compile_all([view]), app_layout())
        if isinstance(diagnostic, UnsupportedAnnotation)
    ] == [FieldName(name) for name in WithUnsupported.model_fields]


def company_view(name: ViewName) -> View:
    return View(name=name, source=Company, step=steps.pipeline())


def hydrated(source: type[pydantic.BaseModel]) -> View:
    return View(name=ViewName.fake(), source=source, step=steps.hydrate())


def lookup_parameter_module() -> ModulePath:
    return ModulePath(
        f"{LookupParameter.name_for(Department, EntityNode.primary_key).root}.views"
    )


@pytest.mark.parametrize(
    ("views", "python", "clash"),
    [
        (
            [company_view(ViewName("Same")), company_view(ViewName("Same"))],
            PythonLayout.models_only(into(ModulePath("app.views"))),
            (Scope("app.views"), Identifier("Same")),
        ),
        (
            [company_view(ViewName(Text.__name__))],
            app_layout(),
            (Scope("app.views"), Identifier(Text.__name__)),
        ),
        (
            [
                company_view(ViewName("CompanyPublic")),
                company_view(ViewName("companyPublic")),
            ],
            app_layout(),
            (Scope("app.mappings"), Identifier("company_public")),
        ),
        (
            [company_view(ViewName("App"))],
            app_layout(),
            (Scope("app.mappings"), Identifier("app")),
        ),
        (
            [company_view(ViewName.fake())],
            layout(
                ModulePath(f"{FunctionPlan.source_parameter.root}.views"),
                ModulePath("app.mappings"),
            ),
            (Scope("app.mappings.company_summary"), FunctionPlan.source_parameter),
        ),
        (
            [hydrated(Employee)],
            layout(lookup_parameter_module(), ModulePath("app.mappings")),
            (
                Scope("app.mappings.company_summary"),
                LookupParameter.name_for(Department, EntityNode.primary_key),
            ),
        ),
        (
            [hydrated(Transfer)],
            app_layout(),
            (
                Scope("app.mappings.company_summary"),
                LookupParameter.name_for(Department, EntityNode.primary_key),
            ),
        ),
    ],
    ids=[
        "two-models",
        "model-and-import",
        "two-functions",
        "function-and-module",
        "source-parameter-and-module",
        "lookup-parameter-and-module",
        "two-lookup-parameters",
    ],
)
def test_a_name_bound_to_several_things_is_one_clash(
    views: Iterable[View], python: PythonLayout, clash: tuple[Scope, Identifier]
) -> None:
    graph = EntityGraph.build(
        [Company, Department, Employee, Archive.Department],
        IdBase.fake(),
        OneToOneMarker.fake(),
    )
    compiled = compiler.compile_views(graph.unwrap(), views).unwrap()

    assert [
        (diagnostic.scope, diagnostic.name)
        if isinstance(diagnostic, NameClash)
        else diagnostic
        for diagnostic in diagnostics_of(compiled, python)
    ] == [clash]


def test_a_clash_names_everything_bound_to_the_name() -> None:
    clashing = company_view(ViewName(Text.__name__))
    module = ModulePath.fake()

    assert diagnostics_of(
        compile_all([clashing]), PythonLayout.models_only(into(module))
    ) == (
        NameClash(
            scope=Scope(module.root),
            name=Identifier(Text.__name__),
            origins=(
                AnnotationText(
                    f"{module.root}.{Text.__name__} (the model of View "
                    f"{clashing.name.root})"
                ),
                symbol(Text).describe(),
            ),
        ),
    )


class Outer:
    class First(pydantic.RootModel[str], frozen=True):
        pass

    class Second(pydantic.RootModel[str], frozen=True):
        pass


class UsesNestedClasses(pydantic.BaseModel, frozen=True):
    first: Outer.First
    second: Outer.Second


def test_two_classes_nested_in_one_imported_class_do_not_clash() -> None:
    view = View(name=ViewName.fake(), source=UsesNestedClasses, step=steps.pipeline())

    plan = plan_of(
        compile_all([view]), PythonLayout.models_only(into(ModulePath.fake()))
    )

    assert len(plan.models) == 1


def test_one_name_in_different_modules_is_allowed() -> None:
    views = [company_view(ViewName("Same")), company_view(ViewName("Same"))]
    modules = iter([ModulePath("app.one"), ModulePath("app.two")])

    plan = plan_of(
        compile_all(views), PythonLayout.models_only(lambda _view: next(modules))
    )

    assert len(plan.models) == len(views)


@pytest.mark.parametrize(
    "python",
    [
        PythonLayout.models_only(into(ModulePath(Company.__module__))),
        layout(ModulePath.fake(), ModulePath(Company.__module__)),
    ],
    ids=["models", "mappings"],
)
def test_generating_into_a_module_that_defines_a_used_class_is_a_diagnostic(
    python: PythonLayout,
) -> None:
    view = View(
        name=ViewName.fake(), source=Company, step=steps.project([FieldName("name")])
    )

    assert [
        (type(diagnostic), diagnostic.module)
        for diagnostic in diagnostics_of(compile_all([view]), python)
        if isinstance(diagnostic, GeneratedIntoDefiningModule)
    ] == [(GeneratedIntoDefiningModule, ModulePath(Company.__module__))]


def test_models_and_mappings_in_one_module_is_a_diagnostic() -> None:
    module = ModulePath.fake()

    assert diagnostics_of(
        compile_all([company_view(ViewName.fake())]), layout(module, module)
    ) == (ModelsAndMappingsInOneModule(module=module),)


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            Unimportable(
                view=ViewName("CompanyPublic"),
                field=FieldName("local"),
                annotation=AnnotationText("app.Local"),
            ),
            (
                "CompanyPublic.local: 'app.Local' cannot be imported by its module "
                "and name. Define it at module level."
            ),
        ),
        (
            Unimportable(
                view=ViewName("CompanyPublic"),
                field=None,
                annotation=AnnotationText("app.Local"),
            ),
            (
                "CompanyPublic: 'app.Local' cannot be imported by its module and "
                "name. Define it at module level."
            ),
        ),
        (
            NameClash(
                scope=Scope("app.views"),
                name=Identifier("Text"),
                origins=(AnnotationText("app.views.Text"), AnnotationText("app.Text")),
            ),
            (
                "app.views: the name 'Text' would refer to each of app.views.Text, "
                "app.Text."
            ),
        ),
        (
            ModelsAndMappingsInOneModule(module=ModulePath("app.views")),
            (
                "app.views: models and mapping functions cannot be generated "
                "into the same module."
            ),
        ),
        (
            GeneratedIntoDefiningModule(
                view=ViewName("CompanyPublic"),
                module=ModulePath("app"),
                defined=(AnnotationText("app.Company"),),
            ),
            (
                "CompanyPublic: generating into app would overwrite the module "
                "defining app.Company. Generate into another module."
            ),
        ),
        (
            UnsupportedAnnotation(
                view=ViewName("CompanyPublic"),
                field=FieldName("tags"),
                annotation=AnnotationText("Annotated[int, 0]"),
            ),
            "CompanyPublic.tags: cannot generate the type 'Annotated[int, 0]'.",
        ),
    ],
    ids=[
        "unimportable-field",
        "unimportable",
        "clash",
        "one-module",
        "defining-module",
        "unsupported",
    ],
)
def test_a_diagnostic_message_names_the_offending_values(
    diagnostic: PlanDiagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


def test_a_union_is_planned_into_the_modules_its_layout_gives() -> None:
    union = employee_union()

    plan = plan_of(compile_all([], [union]), app_layout())

    assert [
        (planned.union, planned.symbol.describe().root)
        for planned in plan.unions[ModulePath("app.views")]
    ] == [
        (UnionName("UnmentoredEmployee"), "app.views.UnmentoredEmployee"),
        (union.name, "app.views.EmployeeByMentor"),
    ]
    assert [
        function.symbol.describe().root
        for function in plan.union_functions[ModulePath("app.mappings")]
    ] == ["app.mappings.unmentored_employee", "app.mappings.employee_by_mentor"]


def test_a_models_only_layout_plans_no_union_functions() -> None:
    plan = plan_of(
        compile_all([], [employee_union()]),
        PythonLayout.models_only(into(ModulePath("app.views"))),
    )

    assert (len(plan.unions), plan.union_functions) == (1, {})


def test_a_union_named_like_a_class_its_module_imports_clashes() -> None:
    union = employee_union().model_copy(update={"name": UnionName(Text.__name__)})
    module = ModulePath("app.views")

    assert diagnostics_of(
        compile_all([], [union]), PythonLayout.models_only(into(module))
    ) == (
        NameClash(
            scope=Scope(module.root),
            name=Identifier(Text.__name__),
            origins=(
                AnnotationText(
                    f"{module.root}.{Text.__name__} (the model of Union "
                    f"{union.name.root})"
                ),
                symbol(Text).describe(),
            ),
        ),
    )


def unplaced(
    _: compiler.CompiledView
    | compiler.CompiledPayload
    | compiler.CompiledUnion
    | compiler.CompiledPayloadUnion,
) -> ModulePath:
    """A layout returning `None`, as one that does not handle what it receives
    does, though its type says otherwise."""
    returned: object = None
    return typing.cast("ModulePath", returned)


def test_a_layout_giving_a_union_of_payloads_no_module_is_a_diagnostic() -> None:
    union = test_compiler.paid_or_free()
    compiled = compiler.compile_catalog(
        test_compiler.empty_graph(), (), (), [union]
    ).unwrap()

    def views_and_payloads(
        placed: compiler.CompiledView
        | compiler.CompiledPayload
        | compiler.CompiledUnion
        | compiler.CompiledPayloadUnion,
    ) -> ModulePath:
        match placed:
            case compiler.CompiledPayloadUnion():
                return unplaced(placed)
            case _:
                return ModulePath("app.payloads")

    match PythonPlan.of(compiled, PythonLayout.models_only(views_and_payloads)):
        case Ok(plan):
            pytest.fail(f"expected diagnostics, got {plan}")
        case Err(error):
            assert error.diagnostics == (
                NoModuleFromLayout(
                    definition=union.name,
                    role=LayoutRole.MODELS,
                    returned=ReturnedText.of(None),
                ),
            )


def test_a_mappings_layout_giving_a_view_no_module_is_a_diagnostic() -> None:
    view = overview()

    assert diagnostics_of(
        compile_all([view]),
        PythonLayout(
            models=into(ModulePath("app.views")),
            mappings=unplaced,
        ),
    ) == (
        NoModuleFromLayout(
            definition=view.name,
            role=LayoutRole.MAPPINGS,
            returned=ReturnedText.of(None),
        ),
    )


def test_a_layout_diagnostic_message_names_the_definition() -> None:
    diagnostic = NoModuleFromLayout(
        definition=UnionName("Writes"),
        role=LayoutRole.MODELS,
        returned=ReturnedText.of(None),
    )

    assert diagnostic.describe().root == (
        "Writes: the models layout returned None for it, not a ModulePath. A "
        "layout returns a ModulePath for every View, Payload and Union it receives."
    )
