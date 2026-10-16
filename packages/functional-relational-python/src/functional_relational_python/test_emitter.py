import contextlib
import sys
import types
from collections.abc import Generator, Iterable
from pathlib import Path

import pydantic
import pytest
from hypothesis import given
from hypothesis import strategies as st
from safe_result import Err, Ok

from functional_relational_python import compiler, emitter, engine, steps
from functional_relational_python.codegen import ModulePath
from functional_relational_python.entity_graph import EntityGraph, FieldName
from functional_relational_python.id import Id
from functional_relational_python.views import View, ViewName


class Text(pydantic.RootModel[str], frozen=True):
    pass


class CompanyId(Id[int]):
    pass


class Company(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text
    secret: Text


class WideId(Id[int]):
    pass


class Wide(pydantic.BaseModel, frozen=True):
    id: WideId
    alpha: Text
    beta: Text | None
    gamma: Text
    delta: Text


# The View models the golden module refers to. Codegen (MB-182) generates these;
# here they are written by hand so the golden module type checks.
class CompanyFull(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text
    secret: Text


class CompanyPublic(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text


class CompanyName(pydantic.BaseModel, frozen=True):
    name: Text


class CompanySecret(pydantic.BaseModel, frozen=True):
    secret: Text


def empty_graph() -> EntityGraph:
    return EntityGraph.build([]).unwrap()


def compile_all(views: Iterable[View]) -> compiler.CompiledViews:
    match compiler.compile_views(empty_graph(), views):
        case Ok(compiled):
            return compiled
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def this_module(_view: compiler.CompiledView) -> ModulePath:
    return ModulePath(__name__)


def emit(views: Iterable[View]) -> emitter.EmittedModule:
    match emitter.PythonEmitter.emit(compile_all(views), this_module):
        case Ok(emitted):
            return emitted
        case Err(error):
            pytest.fail(f"expected an emitted module, got {error}")


def emit_diagnostics(views: Iterable[View]) -> tuple[emitter.Diagnostic, ...]:
    match emitter.PythonEmitter.emit(compile_all(views), this_module):
        case Ok(emitted):
            pytest.fail(f"expected diagnostics, got {emitted.root}")
        case Err(error):
            return error.diagnostics


def test_the_emitted_module_matches_the_golden_file() -> None:
    """The golden file is type checked with the rest of the package (`:typecheck`)."""
    golden = Path(__file__).with_name("test_emitter_golden.py")
    views = [
        View(name=ViewName("CompanyFull"), source=Company, step=steps.pipeline()),
        View(
            name=ViewName("CompanyPublic"),
            source=Company,
            step=steps.omit(FieldName("secret")),
        ),
        View(
            name=ViewName("CompanyName"),
            source=Company,
            step=steps.pipeline(
                steps.omit(FieldName("secret")), steps.omit(FieldName("id"))
            ),
        ),
        View(
            name=ViewName("CompanySecret"),
            source=Company,
            step=steps.project([FieldName("secret")]),
        ),
    ]

    assert emit(views).root == golden.read_text()


@pytest.mark.parametrize(
    ("view", "function"),
    [
        ("CompanyPublic", "company_public"),
        ("Company", "company"),
        ("Company2Public", "company2_public"),
        ("HTTPServer", "httpserver"),
    ],
)
def test_a_function_is_named_after_its_view_in_snake_case(
    view: str, function: str
) -> None:
    assert emitter.PythonEmitter.function_name(ViewName(view)) == (
        emitter.BoundName(function)
    )


def test_a_source_that_cannot_be_imported_is_a_diagnostic() -> None:
    class Local(pydantic.BaseModel, frozen=True):
        name: Text

    view = View(name=ViewName.fake(), source=Local, step=steps.pipeline())

    assert emit_diagnostics([view]) == (
        emitter.UnimportableSource(view=view.name, source=Local),
    )


@pytest.mark.parametrize("name", ["Company Public", "Class"], ids=["space", "keyword"])
def test_a_view_name_that_is_no_function_name_is_a_diagnostic(name: str) -> None:
    view = View(name=ViewName(name), source=Company, step=steps.pipeline())

    assert emit_diagnostics([view]) == (
        emitter.InvalidFunctionName(
            view=view.name, function=emitter.PythonEmitter.function_name(view.name)
        ),
    )


def test_views_with_the_same_function_name_are_a_diagnostic() -> None:
    views = [
        View(name=ViewName(name), source=Company, step=steps.pipeline())
        for name in ["CompanyPublic", "companyPublic"]
    ]

    assert emit_diagnostics(views) == (
        emitter.DuplicateFunctionName(
            function=emitter.BoundName("company_public"),
            views=tuple(view.name for view in views),
        ),
    )


def test_a_function_named_like_an_imported_module_is_a_diagnostic() -> None:
    root = __name__.split(".")[0]
    view = View(
        name=ViewName(root.title().replace("_", "")),
        source=Company,
        step=steps.pipeline(),
    )

    assert emit_diagnostics([view]) == (
        emitter.ShadowedModule(view=view.name, name=emitter.BoundName(root)),
    )


def test_a_view_model_module_named_like_the_parameter_is_a_diagnostic() -> None:
    view = View(name=ViewName.fake(), source=Company, step=steps.pipeline())

    match emitter.PythonEmitter.emit(
        compile_all([view]),
        lambda _view: ModulePath(f"{emitter.PythonEmitter.parameter.root}.views"),
    ):
        case Ok(emitted):
            pytest.fail(f"expected diagnostics, got {emitted.root}")
        case Err(error):
            assert error.diagnostics == (
                emitter.ShadowedModule(
                    view=view.name, name=emitter.PythonEmitter.parameter
                ),
            )


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    [
        (
            emitter.UnimportableSource(view=ViewName("CompanyPublic"), source=Company),
            (
                f"CompanyPublic: the Source {__name__}.Company cannot be imported "
                "by its module and name. Define it at module level."
            ),
        ),
        (
            emitter.InvalidFunctionName(
                view=ViewName("Company Public"),
                function=emitter.BoundName("company public"),
            ),
            "Company Public: 'company public' is not a valid Python function name.",
        ),
        (
            emitter.DuplicateFunctionName(
                function=emitter.BoundName("company_public"),
                views=(ViewName("CompanyPublic"), ViewName("companyPublic")),
            ),
            (
                "Views CompanyPublic, companyPublic all emit a function named "
                "'company_public'."
            ),
        ),
        (
            emitter.ShadowedModule(view=ViewName("App"), name=emitter.BoundName("app")),
            "App: the emitted name 'app' would shadow the imported module 'app'.",
        ),
    ],
    ids=["unimportable", "invalid", "duplicate", "shadowed"],
)
def test_a_diagnostic_message_names_the_offending_values(
    diagnostic: emitter.Diagnostic, message: str
) -> None:
    assert diagnostic.describe().root == message


# Differential test: for generated Sources and pipelines, the emitted functions and
# the InMemoryViewEngine give the same View results.

texts = st.builds(Text, st.text(max_size=3))

companies = st.lists(
    st.builds(
        Company, id=st.builds(CompanyId, st.integers()), name=texts, secret=texts
    ),
    unique_by=lambda company: company.id,
    max_size=3,
)

wides = st.lists(
    st.builds(
        Wide,
        id=st.builds(WideId, st.integers()),
        alpha=texts,
        beta=st.none() | texts,
        gamma=texts,
        delta=texts,
    ),
    unique_by=lambda wide: wide.id,
    max_size=3,
)


class DifferentialCase(pydantic.BaseModel, frozen=True):
    views: tuple[View, ...]
    entities: tuple[pydantic.BaseModel, ...]


@st.composite
def differential_cases(draw: st.DrawFn) -> DifferentialCase:
    """Views over a pool of Sources, each omitting and projecting random fields.

    IDs are distinct per Source, so no Entity replaces another in the engine.
    """
    views: list[View] = []
    for index in range(draw(st.integers(min_value=1, max_value=4))):
        source = draw(st.sampled_from([Company, Wide]))
        fields = list(map(FieldName, source.model_fields))
        omitted = draw(st.lists(st.sampled_from(fields), unique=True))
        kept = [field for field in fields if field not in omitted]
        # `project` may only name fields the omits kept.
        projected = draw(
            st.none() | st.lists(st.sampled_from(kept), unique=True)
            if kept
            else st.none()
        )
        views.append(
            View(
                name=ViewName(f"View{index}"),
                source=source,
                step=steps.pipeline(
                    *map(steps.omit, omitted),
                    *([] if projected is None else [steps.project(projected)]),
                ),
            )
        )
    return DifferentialCase(
        views=tuple(views), entities=(*draw(companies), *draw(wides))
    )


def row_of(model: pydantic.BaseModel) -> engine.ViewRow:
    return engine.ViewRow(
        {FieldName(name): getattr(model, name) for name in type(model).model_fields}
    )


@contextlib.contextmanager
def emitted_module(compiled: compiler.CompiledViews) -> Generator[types.ModuleType]:
    """Runs the emitted module against View models built for `compiled`.

    Codegen (MB-182) generates View models; until this test uses it, each one is
    built here from its View's Relation. They forbid extra fields, because pydantic
    would otherwise ignore a field the emitted function wrongly passes.
    """
    # A top-level name, so `import <name>` binds the module itself.
    view_models = types.ModuleType("frp_differential_view_models")
    view_models.__dict__["pydantic"] = pydantic
    for view in compiled.views:
        body = ""
        for index, field in enumerate(view.relation.fields):
            view_models.__dict__[f"{view.name.root}_{index}"] = field.info.annotation
            body += f"    {field.name.root}: {view.name.root}_{index}\n"
        exec(
            f"class {view.name.root}(pydantic.BaseModel, frozen=True, extra='forbid'):\n"
            f"{body or '    pass\n'}",
            view_models.__dict__,
        )
    sys.modules[view_models.__name__] = view_models
    try:
        match emitter.PythonEmitter.emit(
            compiled, lambda _view: ModulePath(view_models.__name__)
        ):
            case Ok(emitted):
                module = types.ModuleType(f"{__name__}_emitted")
                exec(emitted.root, module.__dict__)
                yield module
            case Err(error):
                pytest.fail(f"expected an emitted module, got {error}")
    finally:
        del sys.modules[view_models.__name__]


@given(differential_cases())
def test_emitted_functions_give_the_same_view_results_as_the_engine(
    case: DifferentialCase,
) -> None:
    graph = EntityGraph.build([Company, Wide]).unwrap()
    compiled = compile_all(case.views)
    view_engine = engine.InMemoryViewEngine(graph, compiled)
    for entity in case.entities:
        view_engine.put(entity).unwrap()

    with emitted_module(compiled) as module:
        for view in case.views:
            function = getattr(
                module, emitter.PythonEmitter.function_name(view.name).root
            )
            emitted_rows = [
                list(row_of(function(entity)).root.items())
                for entity in case.entities
                if isinstance(entity, view.source)
            ]
            engine_rows = [
                list(row.root.items()) for row in view_engine.read(view.name).unwrap()
            ]
            assert emitted_rows == engine_rows
