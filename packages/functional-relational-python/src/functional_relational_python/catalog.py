"""A ViewCatalog: everything `frp generate` needs, defined in the user's code."""

import importlib
import sys
from collections.abc import Sequence

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python import compiler
from functional_relational_python.codegen import (
    Codegen,
    CodegenError,
    Identifier,
    ModuleLayout,
    ModulePath,
)
from functional_relational_python.entity_graph import EntityGraph, EntityGraphError
from functional_relational_python.filesystem import Directory
from functional_relational_python.generated_files import GeneratedFiles
from functional_relational_python.views import View


class ViewCatalog(pydantic.BaseModel, frozen=True):
    """The Entities, the Views over them, and the module each View model goes to.

    >>> catalog = ViewCatalog(
    ...     entities=[Company],
    ...     views=[View(name=ViewName("CompanySummary"), source=Company, step=...)],
    ...     layout=lambda view: ModulePath("myapp.views"),
    ... )
    """

    entities: Sequence[type[pydantic.BaseModel]]
    views: Sequence[View]
    layout: ModuleLayout

    def generate(
        self,
    ) -> Result[
        GeneratedFiles, EntityGraphError | compiler.CompileError | CodegenError
    ]:
        match EntityGraph.build(self.entities):
            case Err(graph_error):
                return Err(graph_error)
            case Ok(graph):
                pass
        match compiler.compile_views(graph, self.views):
            case Err(compile_error):
                return Err(compile_error)
            case Ok(compiled):
                return Codegen.generate(compiled, self.layout)


class ImportPath(pydantic.RootModel[str], frozen=True):
    """Where a ViewCatalog is defined, as `module:attribute`, e.g. `myapp.frp:catalog`."""

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

    def load(self, search: Directory) -> Result[ViewCatalog, CatalogLoadError]:
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
            case ViewCatalog() as catalog:
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
        super().__init__(f"'{path.root}' is a {found.__name__}, not a ViewCatalog.")


type CatalogLoadError = (
    CatalogModuleNotFoundError | CatalogAttributeMissingError | NotACatalogError
)
