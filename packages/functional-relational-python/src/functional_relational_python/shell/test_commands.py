import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler
from functional_relational_python.entity_graph.id import Id
from functional_relational_python.outputs.python.symbols import ModulePath
from functional_relational_python.pipeline import steps
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.shell import commands
from functional_relational_python.shell.catalog import (
    Catalog,
    CatalogAttributeMissingError,
    CatalogModuleNotFoundError,
    ImportPath,
    NotACatalogError,
)
from functional_relational_python.shell.project_config import ConfigFile, ProjectConfig
from functional_relational_python.support.diagnostic import DiagnosticsError
from functional_relational_python.support.filesystem import (
    Directory,
    FileText,
    InMemoryFileSystem,
)
from functional_relational_python.support.generated_files import (
    StaleFilesError,
    Staleness,
)


class Text(pydantic.RootModel[str], frozen=True):
    pass


class CompanyId(Id[int]):
    pass


class Company(pydantic.BaseModel):
    id: CompanyId
    name: Text


def cards_module(_view: compiler.CompiledView) -> ModulePath:
    return ModulePath("app.cards")


catalog = Catalog(
    entities=[Company],
    views=[View(name=ViewName("CompanyCard"), source=Company, step=steps.pipeline())],
    layout=cards_module,
)

invalid_catalog = catalog.model_copy(
    update={
        "views": [
            View(name=ViewName("not valid"), source=Company, step=steps.pipeline())
        ]
    }
)

not_a_catalog = Text("catalog")


def project(catalog_path: ImportPath) -> tuple[InMemoryFileSystem, Directory]:
    filesystem, root = InMemoryFileSystem(), Directory.fake()
    config = ProjectConfig.default().model_copy(update={"catalog": catalog_path})
    filesystem.write(root.file(ConfigFile.name), config.to_toml())
    return filesystem, root


def in_this_module(attribute: Text) -> ImportPath:
    return ImportPath(f"{__name__}:{attribute.root}")


def test_generate_writes_files_that_check_accepts() -> None:
    filesystem, root = project(in_this_module(Text("catalog")))

    written = commands.generate(filesystem, root).unwrap()

    assert written == (root.file(ModulePath("app.cards").file_path()),)
    assert commands.check(filesystem, root) == Ok(None)


def test_check_reports_an_edited_file() -> None:
    filesystem, root = project(in_this_module(Text("catalog")))
    (path,) = commands.generate(filesystem, root).unwrap()
    filesystem.write(path, FileText(f"{filesystem.files[path].root}# edited\n"))

    match commands.check(filesystem, root):
        case Ok(_):
            pytest.fail("expected stale files")
        case Err(error):
            assert isinstance(error, StaleFilesError)
            assert [(file.path, file.staleness) for file in error.stale] == [
                (path, Staleness.OUTDATED)
            ]


@pytest.mark.parametrize(
    ("catalog_path", "error_type"),
    [
        (ImportPath("no_such_module_for_frp:catalog"), CatalogModuleNotFoundError),
        (in_this_module(Text("no_such_attribute")), CatalogAttributeMissingError),
        (in_this_module(Text("not_a_catalog")), NotACatalogError),
        (in_this_module(Text("invalid_catalog")), DiagnosticsError),
    ],
)
def test_generate_reports_why_it_cannot_run(
    catalog_path: ImportPath, error_type: type[commands.GenerateError]
) -> None:
    filesystem, root = project(catalog_path)

    match commands.generate(filesystem, root):
        case Ok(written):
            pytest.fail(f"expected an error, wrote {written}")
        case Err(error):
            assert isinstance(error, error_type)
    assert list(filesystem.files) == [root.file(ConfigFile.name)]
