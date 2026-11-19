from typing import Literal

import pydantic
import pytest
from safe_result import Err, Ok

import functional_relational_python as frp
from functional_relational_python import steps
from functional_relational_python.django import WithoutDjangoModels
from functional_relational_python.testing import ViewRow, engine_for


class Text(pydantic.RootModel[str], frozen=True):
    pass


class CompanyId(frp.Id[int]):
    pass


class Company(pydantic.BaseModel):
    id: CompanyId
    name: Text


def one_module(
    _compiled: frp.CompiledView
    | frp.CompiledWriteModel
    | frp.CompiledUnion
    | frp.CompiledWriteModelUnion,
) -> frp.ModulePath:
    return frp.ModulePath("app.models")


def catalog_of(*views: frp.View) -> frp.Catalog:
    return frp.Catalog(
        entities=[Company],
        views=views,
        write_models=[],
        unions=[],
        layout=one_module,
        mappings=frp.WithoutMappings(),
        django=WithoutDjangoModels(),
    )


def test_engine_for_reads_the_catalogs_views() -> None:
    view = frp.View(
        name=frp.ViewName("CompanyName"),
        query=frp.from_(Company, steps.omit(frp.FieldName("id"))),
    )
    engine = engine_for(catalog_of(view)).unwrap()
    _ = engine.put(Company(id=CompanyId(1), name=Text("Acme"))).unwrap()

    assert engine.read(view.name) == Ok(
        (ViewRow({frp.FieldName("name"): Text("Acme")}),)
    )


def test_engine_for_reports_the_catalogs_diagnostics() -> None:
    view = frp.View(
        name=frp.ViewName("CompanyName"),
        query=frp.from_(Company, steps.omit(frp.FieldName("nmae"))),
    )

    match engine_for(catalog_of(view)):
        case Err(error):
            assert "nmae" in str(error)
        case Ok(_):
            pytest.fail("expected the misspelt field to be reported")


def test_engine_for_leaves_out_unions_of_write_models() -> None:
    view = frp.View(name=frp.ViewName("CompanyCard"), query=frp.from_(Company))

    write_union = frp.Union(
        name=frp.UnionName("CompanyCommand"),
        discriminator=frp.FieldName("kind"),
        variants=(
            frp.WriteModel(
                name=frp.WriteModelName("CompanyCreate"),
                query=frp.from_(
                    Company,
                    steps.add(frp.FieldName("kind"), Literal["create"], "create"),
                ),
            ),
            frp.WriteModel(
                name=frp.WriteModelName("CompanyRename"),
                query=frp.from_(
                    Company,
                    steps.add(frp.FieldName("kind"), Literal["rename"], "rename"),
                ),
            ),
        ),
    )
    catalog = catalog_of(view).model_copy(update={"unions": [write_union]})

    engine = engine_for(catalog).unwrap()

    assert engine.read(view.name) == Ok(())
