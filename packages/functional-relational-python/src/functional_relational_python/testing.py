"""The InMemoryViewEngine, a fake for tests that read Views."""

from safe_result import Err, Ok, Result

from functional_relational_python.compiler import compiler
from functional_relational_python.entity_graph.entity_graph import EntityGraph
from functional_relational_python.outputs.engine import (
    DanglingReference,
    InMemoryViewEngine,
    NonEntityFromModelError,
    NotAnEntityError,
    ReadDiagnostic,
    SharedOneToOneReference,
    UnknownUnionError,
    UnknownViewError,
    UnmetRequirement,
    UnmetReverseRequirement,
    VariantRow,
    ViewRow,
)
from functional_relational_python.shell.catalog import Catalog
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticsError,
)

__all__ = [
    "DanglingReference",
    "InMemoryViewEngine",
    "NonEntityFromModelError",
    "NotAnEntityError",
    "ReadDiagnostic",
    "SharedOneToOneReference",
    "UnknownUnionError",
    "UnknownViewError",
    "UnmetRequirement",
    "UnmetReverseRequirement",
    "VariantRow",
    "ViewRow",
    "engine_for",
]


def engine_for(
    catalog: Catalog,
) -> Result[InMemoryViewEngine, DiagnosticsError[Diagnostic]]:
    """An empty engine for the Entities, Views and Unions of Views of `catalog`, or
    every diagnostic of its Entities, or else of its Views, WriteModels and
    Unions."""
    match EntityGraph.build(
        catalog.entities, catalog.id_base, catalog.one_to_one_marker
    ):
        case Err(graph_error):
            return Err(graph_error)
        case Ok(graph):
            pass
    match compiler.compile_catalog(
        graph, catalog.views, catalog.write_models, catalog.unions
    ):
        case Err(compile_error):
            return Err(compile_error)
        case Ok(compiled):
            views = compiler.CompiledViews(views=compiled.views, unions=compiled.unions)
            return Ok(InMemoryViewEngine(graph, views))
