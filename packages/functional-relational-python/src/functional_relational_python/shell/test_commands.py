import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.compiler import compiler
from functional_relational_python.compiler.model_names import InvalidModelName
from functional_relational_python.entity_graph.entity_graph import FieldName
from functional_relational_python.entity_graph.id import Id, IdBase, OneToOneMarker
from functional_relational_python.outputs.django.models import (
    DjangoModels,
    NoDjangoFieldRule,
)
from functional_relational_python.outputs.python.plan import (
    ModelsAndMappingsInOneModule,
    WithoutMappings,
)
from functional_relational_python.outputs.python.symbols import ModulePath
from functional_relational_python.pipeline import steps
from functional_relational_python.pipeline.payloads import Payload, PayloadName
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
    FileGeneratedTwice,
    ModuleShadowsPackage,
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


def by_kind(compiled: compiler.CompiledView | compiler.CompiledPayload) -> ModulePath:
    match compiled:
        case compiler.CompiledView():
            return ModulePath("app.cards")
        case compiler.CompiledPayload():
            return ModulePath("app.payloads")


catalog = Catalog(
    entities=[Company],
    id_base=IdBase.fake(),
    one_to_one_marker=OneToOneMarker.fake(),
    views=[View(name=ViewName("CompanyCard"), source=Company, step=steps.pipeline())],
    payloads=[
        Payload(
            name=PayloadName("CompanyCreate"),
            source=Company,
            step=steps.omit(FieldName("id")),
        )
    ],
    layout=by_kind,
    mappings=ModulePath("app.mappings"),
    django=DjangoModels.fake(),
)

without_mappings_catalog = catalog.model_copy(update={"mappings": WithoutMappings()})


def mappings_into(module: ModulePath) -> Catalog:
    return catalog.model_copy(update={"mappings": module})


mappings_into_a_view_module_catalog = mappings_into(ModulePath("app.cards"))
mappings_into_a_payload_module_catalog = mappings_into(ModulePath("app.payloads"))
mappings_into_the_django_module_catalog = mappings_into(DjangoModels.fake().module)
mappings_over_a_package_catalog = mappings_into(ModulePath("app"))

invalid_catalog = catalog.model_copy(
    update={
        "views": [
            View(name=ViewName("not valid"), source=Company, step=steps.pipeline())
        ]
    }
)

shared_name_catalog = catalog.model_copy(
    update={
        "payloads": [
            Payload(
                name=PayloadName(view.name.root), source=Company, step=steps.pipeline()
            )
            for view in catalog.views
        ]
    }
)

django_into_a_python_module_catalog = catalog.model_copy(
    update={
        "django": DjangoModels.fake().model_copy(
            update={"module": ModulePath("app.cards")}
        )
    }
)


class Untyped(pydantic.BaseModel):
    id: CompanyId
    labels: frozenset[Text]


invalid_view_and_django_catalog = invalid_catalog.model_copy(
    update={"entities": [Untyped]}
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

    assert written == (
        root.file(ModulePath("app.cards").file_path()),
        root.file(ModulePath("app.payloads").file_path()),
        root.file(ModulePath("app.mappings").file_path()),
        root.file(DjangoModels.fake().module.file_path()),
        root.file(DjangoModels.fake().queries.file_path()),
    )
    assert commands.check(filesystem, root) == Ok(None)


def test_generate_writes_the_mapping_functions_into_the_mappings_module() -> None:
    filesystem, root = project(in_this_module(Text("catalog")))
    _ = commands.generate(filesystem, root).unwrap()

    mappings = filesystem.files[root.file(ModulePath("app.mappings").file_path())]

    assert "def company_card(" in mappings.root


def test_generate_writes_no_mappings_module_without_mappings() -> None:
    filesystem, root = project(in_this_module(Text("without_mappings_catalog")))

    written = commands.generate(filesystem, root).unwrap()

    assert root.file(ModulePath("app.mappings").file_path()) not in written
    assert commands.check(filesystem, root) == Ok(None)


def test_check_reports_a_missing_mappings_module() -> None:
    filesystem, root = project(in_this_module(Text("without_mappings_catalog")))
    _ = commands.generate(filesystem, root).unwrap()
    filesystem.write(
        root.file(ConfigFile.name),
        ProjectConfig.default()
        .model_copy(update={"catalog": in_this_module(Text("catalog"))})
        .to_toml(),
    )

    match commands.check(filesystem, root):
        case Ok(_):
            pytest.fail("expected stale files")
        case Err(error):
            assert isinstance(error, StaleFilesError)
            assert [(file.path, file.staleness) for file in error.stale] == [
                (root.file(ModulePath("app.mappings").file_path()), Staleness.MISSING)
            ]


def test_check_reports_an_edited_file() -> None:
    filesystem, root = project(in_this_module(Text("catalog")))
    _ = commands.generate(filesystem, root).unwrap()
    path = root.file(ModulePath("app.cards").file_path())
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
        (in_this_module(Text("shared_name_catalog")), DiagnosticsError),
        (
            in_this_module(Text("django_into_a_python_module_catalog")),
            DiagnosticsError,
        ),
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


@pytest.mark.parametrize(
    ("attribute", "diagnostic"),
    [
        (
            Text("mappings_into_a_view_module_catalog"),
            ModelsAndMappingsInOneModule(module=ModulePath("app.cards")),
        ),
        (
            Text("mappings_into_a_payload_module_catalog"),
            ModelsAndMappingsInOneModule(module=ModulePath("app.payloads")),
        ),
        (
            Text("mappings_into_the_django_module_catalog"),
            FileGeneratedTwice(path=DjangoModels.fake().module.file_path()),
        ),
        (
            Text("mappings_over_a_package_catalog"),
            ModuleShadowsPackage(path=ModulePath("app").file_path()),
        ),
    ],
)
def test_generate_reports_a_mappings_module_another_output_uses(
    attribute: Text,
    diagnostic: ModelsAndMappingsInOneModule
    | FileGeneratedTwice
    | ModuleShadowsPackage,
) -> None:
    filesystem, root = project(in_this_module(attribute))

    match commands.generate(filesystem, root):
        case Ok(written):
            pytest.fail(f"expected diagnostics, wrote {written}")
        case Err(error):
            assert isinstance(error, DiagnosticsError)
            assert error.diagnostics == (diagnostic,)


def test_generate_reports_the_diagnostics_of_every_output() -> None:
    filesystem, root = project(in_this_module(Text("invalid_view_and_django_catalog")))

    match commands.generate(filesystem, root):
        case Ok(written):
            pytest.fail(f"expected diagnostics, wrote {written}")
        case Err(error):
            assert isinstance(error, DiagnosticsError)
            assert {type(diagnostic) for diagnostic in error.diagnostics} == {
                InvalidModelName,
                NoDjangoFieldRule,
            }
