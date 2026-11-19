"""Generate Pydantic view models from a relational algebra over your entities.

The names needed to define a Catalog. The pipeline vocabulary is in `steps` and
`conditions`, the Django output config in `django`, and the in-memory engine in
`testing`.

>>> import functional_relational_python as frp
>>> from functional_relational_python import conditions as c, steps as s
"""

from functional_relational_python.compiler.compiler import (
    CompiledUnion,
    CompiledView,
    CompiledWriteModel,
    CompiledWriteModelUnion,
)
from functional_relational_python.entity_graph.entity_graph import FieldName
from functional_relational_python.entity_graph.id import (
    Id,
    IdBase,
    OneToOne,
    OneToOneMarker,
)
from functional_relational_python.outputs.python.plan import WithoutMappings
from functional_relational_python.outputs.python.symbols import ModulePath
from functional_relational_python.pipeline.steps import Query, from_
from functional_relational_python.pipeline.unions import Union, UnionName
from functional_relational_python.pipeline.views import View, ViewName
from functional_relational_python.pipeline.write_models import (
    WriteModel,
    WriteModelName,
)
from functional_relational_python.shell.catalog import Catalog

__all__ = [
    "Catalog",
    "CompiledUnion",
    "CompiledView",
    "CompiledWriteModel",
    "CompiledWriteModelUnion",
    "FieldName",
    "Id",
    "IdBase",
    "ModulePath",
    "OneToOne",
    "OneToOneMarker",
    "Query",
    "Union",
    "UnionName",
    "View",
    "ViewName",
    "WithoutMappings",
    "WriteModel",
    "WriteModelName",
    "from_",
]
