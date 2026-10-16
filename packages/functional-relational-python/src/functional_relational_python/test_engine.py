import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python import compiler, engine, steps
from functional_relational_python.entity_graph import EntityGraph, FieldName
from functional_relational_python.id import Id
from functional_relational_python.views import View, ViewName


class Text(pydantic.RootModel[str], frozen=True):
    pass


class CompanyId(Id[int]):
    @staticmethod
    def fake() -> CompanyId:
        return CompanyId(1)


class Company(pydantic.BaseModel, frozen=True):
    id: CompanyId
    name: Text
    secret: Text

    @staticmethod
    def fake(company_id: CompanyId) -> Company:
        return Company(id=company_id, name=Text("Acme"), secret=Text("hunter2"))


class Memo(pydantic.BaseModel, frozen=True):
    """A Source that is not an Entity: it has no ID."""

    body: Text


def make_engine(*views: View) -> engine.InMemoryViewEngine:
    graph = EntityGraph.build([Company]).unwrap()
    match compiler.compile_views(graph, views):
        case Ok(compiled):
            return engine.InMemoryViewEngine(graph, compiled)
        case Err(error):
            pytest.fail(f"expected compiled views, got {error}")


def view_of(step: steps.Step) -> View:
    return View(name=ViewName.fake(), source=Company, step=step)


def read(
    view_engine: engine.InMemoryViewEngine, view: ViewName
) -> tuple[engine.ViewRow, ...]:
    match view_engine.read(view):
        case Ok(rows):
            return rows
        case Err(error):
            pytest.fail(f"expected rows, got {error}")


def row_of(company: Company, *omitted: FieldName) -> engine.ViewRow:
    return engine.ViewRow(
        {
            FieldName(name): getattr(company, name)
            for name in Company.model_fields
            if FieldName(name) not in omitted
        }
    )


@pytest.mark.parametrize(
    "omitted",
    [(), (FieldName("secret"),), (FieldName("secret"), FieldName("name"))],
    ids=["none", "one", "two"],
)
def test_a_view_returns_each_entity_without_the_omitted_fields(
    omitted: tuple[FieldName, ...],
) -> None:
    view = view_of(steps.pipeline(*map(steps.omit, omitted)))
    view_engine = make_engine(view)
    companies = [Company.fake(CompanyId(1)), Company.fake(CompanyId(2))]
    for company in companies:
        view_engine.put(company).unwrap()

    assert read(view_engine, view.name) == tuple(
        row_of(company, *omitted) for company in companies
    )


def test_put_replaces_the_entity_with_the_same_id() -> None:
    view = view_of(steps.pipeline())
    view_engine = make_engine(view)
    original = Company.fake(CompanyId.fake())
    replacement = Company(
        id=original.id, name=Text(f"{original.name.root} 2"), secret=original.secret
    )
    view_engine.put(original).unwrap()
    view_engine.put(replacement).unwrap()

    assert read(view_engine, view.name) == (row_of(replacement),)


def test_a_view_over_an_empty_engine_has_no_rows() -> None:
    view = view_of(steps.pipeline())

    assert read(make_engine(view), view.name) == ()


def test_views_over_the_same_entities_are_independent() -> None:
    full = View(name=ViewName("Full"), source=Company, step=steps.pipeline())
    public = View(
        name=ViewName("Public"), source=Company, step=steps.omit(FieldName("secret"))
    )
    view_engine = make_engine(full, public)
    company = Company.fake(CompanyId.fake())
    view_engine.put(company).unwrap()

    assert (read(view_engine, full.name), read(view_engine, public.name)) == (
        (row_of(company),),
        (row_of(company, FieldName("secret")),),
    )


def test_put_rejects_a_model_that_is_not_an_entity() -> None:
    match make_engine().put(Memo(body=Text("note"))):
        case Ok():
            pytest.fail("expected NotAnEntityError")
        case Err(error):
            assert error.model is Memo


def test_reading_an_unknown_view_names_it_and_the_known_views() -> None:
    known = view_of(steps.pipeline())
    unknown = ViewName(f"{known.name.root}Typo")

    match make_engine(known).read(unknown):
        case Ok(rows):
            pytest.fail(f"expected UnknownViewError, got {rows}")
        case Err(error):
            assert (error.view, error.known) == (unknown, (known.name,))
