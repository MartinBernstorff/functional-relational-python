"""A Catalog: everything `frp generate` needs, defined in the user's code."""

import importlib
import sys
from collections.abc import Sequence

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler
from functional_relational_python.entity_graph.entity_graph import EntityGraph
from functional_relational_python.entity_graph.id import IdBase, OneToOneMarker
from functional_relational_python.outputs.django.models import (
    DjangoModels,
    DjangoModelsRenderer,
    WithoutDjangoModels,
)
from functional_relational_python.outputs.django.queries import DjangoQueriesRenderer
from functional_relational_python.outputs.python.plan import (
    ModuleLayout,
    PlacedModels,
    PythonLayout,
    PythonPlan,
    WithoutMappings,
)
from functional_relational_python.outputs.python.render import PythonRenderer
from functional_relational_python.outputs.python.symbols import FirstParty, ModulePath
from functional_relational_python.pipeline.payloads import Payload
from functional_relational_python.pipeline.unions import Union
from functional_relational_python.pipeline.views import View
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticsError,
)
from functional_relational_python.support.filesystem import Directory
from functional_relational_python.support.generated_files import GeneratedFiles
from functional_relational_python.support.identifier import Identifier


class Catalog(pydantic.BaseModel, frozen=True):
    """The Entities, the class their ID types subclass, the Views over them, the
    Payloads, the Unions of Views or of Payloads, the module each View's,
    Payload's or Union's model goes to, and whether and where mapping functions
    and Django models go.

    Views are read models, Payloads are write models. A View and a Payload cannot
    share a name. A Union's Variants are generated as if `views` or `payloads`
    listed them. `layout` receives each compiled View, Payload and Union.
    `mappings` is the one module every View's and Union of Views' mapping
    function goes to, see `MappingsRenderer`.
    It cannot be a module `layout` returns, a Django module, a module defining a
    Source or an Entity a View looks up, or a module named like a package other
    files are generated into (`app` when generating `app.views`). `django`
    generates a Django model per Entity, and Django queries for each View over
    an Entity, see `DjangoModels`. An Entity listed in `DjangoModels.existing`
    gets no generated model, since the application already has one. `id_base` is `IdBase(Id)`, or an application's
    own base, and `one_to_one_marker` is `OneToOneMarker(OneToOne)`, or an
    application's own marker, so that the Entity modules, and with them the
    generated code, do not need frp installed.

    >>> from functional_relational_python.entity_graph.id import (
    ...     Id,
    ...     IdBase,
    ...     OneToOne,
    ...     OneToOneMarker,
    ... )
    >>> catalog = Catalog(
    ...     entities=[Company],
    ...     id_base=IdBase(Id),
    ...     one_to_one_marker=OneToOneMarker(OneToOne),
    ...     views=[View(name=ViewName("CompanySummary"), source=Company, step=...)],
    ...     payloads=[
    ...         Payload(name=PayloadName("CompanyCreate"), source=Company, step=...)
    ...     ],
    ...     unions=[],
    ...     layout=lambda compiled: ModulePath("myapp.models"),
    ...     mappings=ModulePath("myapp.mappings"),
    ...     django=WithoutDjangoModels(),
    ... )
    """

    entities: Sequence[type[pydantic.BaseModel]]
    id_base: IdBase
    one_to_one_marker: OneToOneMarker
    views: Sequence[View]
    payloads: Sequence[Payload]
    unions: Sequence[Union]
    layout: ModuleLayout
    mappings: ModulePath | WithoutMappings
    django: DjangoModels | WithoutDjangoModels

    def generate(
        self, first_party: FirstParty
    ) -> Result[GeneratedFiles, DiagnosticsError[Diagnostic]]:
        """Every generated file, or every diagnostic of every output.

        Generated imports of `first_party` packages, and of the packages frp
        generates, are first party; all others but the standard library are
        third party.
        """
        match EntityGraph.build(self.entities, self.id_base, self.one_to_one_marker):
            case Err(graph_error):
                return Err(graph_error)
            case Ok(graph):
                pass
        parts: list[GeneratedFiles] = []
        reported: list[Diagnostic] = []
        for generated in self._outputs(graph, first_party):
            match generated:
                case Ok(files):
                    parts.append(files)
                case Err(error):
                    reported.extend(error.diagnostics)
        match DiagnosticsError.or_ok(reported, lambda: parts):
            case Err(output_error):
                return Err(output_error)
            case Ok(generated):
                pass
        match GeneratedFiles.merged(generated):
            case Ok(merged):
                return Ok(merged)
            case Err(merge_error):
                return Err(merge_error)

    def _outputs(
        self, graph: EntityGraph, first_party: FirstParty
    ) -> list[Result[GeneratedFiles, DiagnosticsError[Diagnostic]]]:
        """Each output, with compiling and placing the models as outputs too.

        Django models need no compiled Views, so their diagnostics are reported
        even when compiling fails. The layout places every model once, for every
        output that refers to it.
        """
        django_models = self._django_models(graph, first_party)
        match compiler.compile_catalog(graph, self.views, self.payloads, self.unions):
            case Err(compile_error):
                return [Err(compile_error), django_models]
            case Ok(compiled):
                pass
        match PlacedModels.of_catalog(self.layout, compiled):
            case Err(layout_error):
                return [Err(layout_error), django_models]
            case Ok(placed):
                return [
                    self._python(compiled, placed, first_party),
                    django_models,
                    self._django_queries(graph, compiled, placed, first_party),
                ]

    def _python(
        self,
        compiled: compiler.CompiledCatalog,
        placed: PlacedModels,
        first_party: FirstParty,
    ) -> Result[GeneratedFiles, DiagnosticsError[Diagnostic]]:
        match PythonPlan.of(compiled, self._python_layout(placed)):
            case Err(plan_error):
                return Err(plan_error)
            case Ok(plan):
                return Ok(PythonRenderer.render(plan, first_party))

    def _python_layout(self, placed: PlacedModels) -> PythonLayout:
        match self.mappings:
            case WithoutMappings():
                return PythonLayout.models_only(placed.module_of)
            case ModulePath() as module:
                return PythonLayout.mappings_into(placed.module_of, module)

    def _django_models(
        self, graph: EntityGraph, first_party: FirstParty
    ) -> Result[GeneratedFiles, DiagnosticsError[Diagnostic]]:
        match self.django:
            case WithoutDjangoModels():
                return Ok(GeneratedFiles(files=()))
            case DjangoModels() as django:
                pass
        match DjangoModelsRenderer.render(graph, django, first_party):
            case Ok(files):
                return Ok(files)
            case Err(error):
                return Err(error)

    def _django_queries(
        self,
        graph: EntityGraph,
        compiled: compiler.CompiledCatalog,
        placed: PlacedModels,
        first_party: FirstParty,
    ) -> Result[GeneratedFiles, DiagnosticsError[Diagnostic]]:
        match self.django:
            case WithoutDjangoModels():
                return Ok(GeneratedFiles(files=()))
            case DjangoModels() as django:
                pass
        match DjangoQueriesRenderer.render(
            graph,
            django,
            (*compiled.views, *compiled.unions),
            placed,
            first_party=first_party,
        ):
            case Ok(files):
                return Ok(files)
            case Err(error):
                return Err(error)


class ImportPath(pydantic.RootModel[str], frozen=True):
    """Where a Catalog is defined, as `module:attribute`, e.g. `myapp.frp:catalog`."""

    @pydantic.field_validator("root")
    @classmethod
    def _module_and_attribute(cls, root: str) -> str:
        module, separator, attribute = root.partition(":")
        if not separator:
            msg = f"'{root}' is not of the form 'module:attribute'"
            raise ValueError(msg)
        _ = (ModulePath(module), Identifier(attribute))
        return root

    @staticmethod
    def fake() -> ImportPath:
        return ImportPath("views:catalog")

    def module(self) -> ModulePath:
        return ModulePath(self.root.partition(":")[0])

    def attribute(self) -> Identifier:
        return Identifier(self.root.partition(":")[2])

    def load(self, search: Directory) -> Result[Catalog, CatalogLoadError]:
        """Imports the catalog. `search` is added to the module search path first,
        so a catalog next to the project config is found without installing it."""
        if str(search.root) not in sys.path:
            sys.path.insert(0, str(search.root))
        module = self.module().root
        try:
            imported = importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name is not None and f"{module}.".startswith(f"{error.name}."):
                return Err(CatalogModuleNotFoundError(self))
            raise
        match getattr(imported, self.attribute().root, None):
            case Catalog() as catalog:
                return Ok(catalog)
            case None:
                return Err(CatalogAttributeMissingError(self))
            case found:
                return Err(NotACatalogError(self, type(found)))


class CatalogModuleNotFoundError(Exception):
    def __init__(self, path: ImportPath) -> None:
        self.path = path
        super().__init__(
            f"Cannot import module '{path.module().root}' for '{path.root}'."
        )


class CatalogAttributeMissingError(Exception):
    def __init__(self, path: ImportPath) -> None:
        self.path = path
        super().__init__(
            f"Module '{path.module().root}' has no attribute '{path.attribute().root}'."
        )


class NotACatalogError(Exception):
    def __init__(self, path: ImportPath, found: type) -> None:
        self.path = path
        super().__init__(f"'{path.root}' is a {found.__name__}, not a Catalog.")


type CatalogLoadError = (
    CatalogModuleNotFoundError | CatalogAttributeMissingError | NotACatalogError
)
