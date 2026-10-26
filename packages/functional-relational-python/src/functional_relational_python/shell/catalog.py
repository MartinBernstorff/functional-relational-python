"""A Catalog: everything `frp generate` needs, defined in the user's code."""

import importlib
import sys
from collections.abc import Sequence

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler
from functional_relational_python.entity_graph.entity_graph import EntityGraph
from functional_relational_python.outputs.python.plan import (
    ModuleLayout,
    PythonLayout,
    PythonPlan,
)
from functional_relational_python.outputs.python.render import PythonRenderer
from functional_relational_python.outputs.python.symbols import ModulePath
from functional_relational_python.pipeline.payloads import Payload
from functional_relational_python.pipeline.views import View
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticsError,
)
from functional_relational_python.support.filesystem import Directory
from functional_relational_python.support.generated_files import GeneratedFiles
from functional_relational_python.support.identifier import Identifier


class Catalog(pydantic.BaseModel, frozen=True):
    """The Entities, the Views over them, the Payloads, and the module each View's
    or Payload's model goes to.

    Views are read models, Payloads are write models. A View and a Payload cannot
    share a name. `layout` receives each compiled View and Payload.

    >>> catalog = Catalog(
    ...     entities=[Company],
    ...     views=[View(name=ViewName("CompanySummary"), source=Company, step=...)],
    ...     payloads=[
    ...         Payload(name=PayloadName("CompanyCreate"), source=Company, step=...)
    ...     ],
    ...     layout=lambda compiled: ModulePath("myapp.models"),
    ... )
    """

    entities: Sequence[type[pydantic.BaseModel]]
    views: Sequence[View]
    payloads: Sequence[Payload]
    layout: ModuleLayout

    def generate(
        self,
    ) -> Result[GeneratedFiles, DiagnosticsError[Diagnostic]]:
        match EntityGraph.build(self.entities):
            case Err(graph_error):
                return Err(graph_error)
            case Ok(graph):
                pass
        match compiler.compile_catalog(graph, self.views, self.payloads):
            case Err(compile_error):
                return Err(compile_error)
            case Ok(compiled):
                pass
        match PythonPlan.of(compiled, PythonLayout.models_only(self.layout)):
            case Err(plan_error):
                return Err(plan_error)
            case Ok(plan):
                return Ok(PythonRenderer.render(plan))


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
