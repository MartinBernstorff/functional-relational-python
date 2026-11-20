"""Configuration for the Django output: `Catalog(django=DjangoModels(...))`.

>>> from django.db import models
>>> from functional_relational_python import django as dj
>>> dj.DjangoModels(
...     module=ModulePath("app.models"),
...     queries=ModulePath("app.queries"),
...     app_label=dj.AppLabel("app"),
...     every_model=dj.ModelOptions(ordering=(dj.OrderingTerm("-created_at"),)),
...     entity_configs={
...         Person: dj.GeneratedModel(
...             bases=(Stamped,),
...             fields={
...                 FieldName("employer_id"): dj.FieldConfig(
...                     on_delete=dj.OnDelete.CASCADE,
...                     related_name=dj.RelatedName("employees"),
...                 ),
...             },
...             constraints=(dj.Check(...),),
...         ),
...         Company: dj.ExistingDjangoModel(
...             module=ModulePath("crm.models"), name=Identifier("Company")
...         ),
...     },
... )

Bases, decorators, field classes and default values are given as the objects
themselves, e.g. `models.Model` or `uuid.uuid4`, or as a `Symbol` for one the
Catalog cannot import, e.g. an abstract Django model. Other values in generated
source are written as a `ValueExpression`, e.g. `ValueExpression("254")`.
"""

from functional_relational_python.outputs.django.models import (
    AppLabel,
    AutoNow,
    AutoNowAdd,
    BareDecorator,
    CalledDecorator,
    DatabaseDefault,
    DbTable,
    Default,
    DjangoField,
    DjangoIndex,
    DjangoModelName,
    DjangoModels,
    ExistingDjangoModel,
    FieldConfig,
    FieldTypes,
    ForeignKeyConstraint,
    GeneratedModel,
    IndexField,
    IndexName,
    ModelGenericBase,
    ModelOptions,
    OnDelete,
    OrderingTerm,
    PlainBase,
    ReferencePrimaryKey,
    RelatedName,
    Unset,
    WithoutDjangoModels,
)
from functional_relational_python.outputs.python.symbols import (
    Symbol,
    ValueExpression,
)
from functional_relational_python.pipeline.constraints import (
    Check,
    ConstraintName,
    EveryRow,
    Unique,
)
from functional_relational_python.support.identifier import Identifier

__all__ = [
    "AppLabel",
    "AutoNow",
    "AutoNowAdd",
    "BareDecorator",
    "CalledDecorator",
    "Check",
    "ConstraintName",
    "DatabaseDefault",
    "DbTable",
    "Default",
    "DjangoField",
    "DjangoIndex",
    "DjangoModelName",
    "DjangoModels",
    "EveryRow",
    "ExistingDjangoModel",
    "FieldConfig",
    "FieldTypes",
    "ForeignKeyConstraint",
    "GeneratedModel",
    "Identifier",
    "IndexField",
    "IndexName",
    "ModelGenericBase",
    "ModelOptions",
    "OnDelete",
    "OrderingTerm",
    "PlainBase",
    "ReferencePrimaryKey",
    "RelatedName",
    "Symbol",
    "Unique",
    "Unset",
    "ValueExpression",
    "WithoutDjangoModels",
]
