# functional-relational-python

Generate Pydantic view models from a relational algebra over your domain entities.

You declare your domain as Pydantic models (Entities), then define each read model
(View) and write model (WriteModel) as a query over them. `frp generate` writes
the Pydantic models, typed mapping functions from Entities to Views, and,
optionally, Django models and querysets. `frp check` fails when the generated code
is stale.

Work in progress: the API changes between minor versions.

## Install

```sh
uv add functional-relational-python
```

Requires Python 3.14.

## Imports

The public API is in five modules. Anything else may change without notice.

| Module | Holds |
| --- | --- |
| `functional_relational_python` | What defining a Catalog needs: `Id`, `View`, `WriteModel`, `Union`, `Catalog`, `from_`, the name types, … |
| `functional_relational_python.steps` | The steps a Query applies |
| `functional_relational_python.conditions` | Conditions for `where` and for database constraints |
| `functional_relational_python.django` | Configuration for the Django output |
| `functional_relational_python.testing` | The `InMemoryViewEngine` |

The examples below import them as:

```python
import functional_relational_python as frp
from functional_relational_python import conditions as c, steps as s
```

## Example

### 1. Entities

An Entity is a Pydantic model whose `id` field is an ID type, a subclass of `Id`.
Any other field typed with an ID type is a Reference to the Entity with that ID.

```python
# myapp/entities.py
import enum

import pydantic

import functional_relational_python as frp


class DepartmentId(frp.Id[int]):
    pass


class EmployeeId(frp.Id[int]):
    pass


class Status(enum.StrEnum):
    ACTIVE = "active"
    LEFT = "left"


class Department(pydantic.BaseModel):
    id: DepartmentId
    name: str


class Employee(pydantic.BaseModel):
    id: EmployeeId
    name: str
    email: str | None
    status: Status
    department_id: DepartmentId  # a Reference to Department
```

### 2. Views, WriteModels and a Catalog

A `Query` is a model and the steps applied to it: `from_(model, *steps)`. A
`View` or `WriteModel` is a name and a Query. The `Catalog` lists everything to
generate, and where it goes.

```python
# myapp/frp_catalog.py
import functional_relational_python as frp
from functional_relational_python import conditions as c, steps as s
from functional_relational_python.django import WithoutDjangoModels

from myapp.entities import Department, Employee, Status

active_employees = frp.from_(
    Employee,
    s.where(c.field(frp.FieldName("status")).eq(Status.ACTIVE)),
    s.omit(frp.FieldName("status")),
)

employee_card = frp.View(
    name=frp.ViewName("EmployeeCard"),
    query=active_employees.then(
        s.require(frp.FieldName("email")),
        s.resolve(
            s.references(),
            s.omit(frp.FieldName("id")),
            s.rename(frp.FieldName("name"), frp.FieldName("department_name")),
        ),
        s.flatten(frp.FieldName("department")),
    ),
)

employee_create = frp.WriteModel(
    name=frp.WriteModelName("EmployeeCreate"),
    query=frp.from_(Employee, s.omit(frp.FieldName("id"))),
)

catalog = frp.Catalog(
    entities=[Employee, Department],
    views=[employee_card],
    write_models=[employee_create],
    unions=[],
    layout=lambda compiled: frp.ModulePath("myapp.generated.models"),
    mappings=frp.ModulePath("myapp.generated.mappings"),
    django=WithoutDjangoModels(),
)
```

### 3. Config

`frp.toml`, in the working directory or a parent, up to the repository root:

```toml
catalog = "myapp.frp_catalog:catalog"
output_dir = "."
known_first_party = ["myapp"]
```

`frp config init` writes one with defaults, and `frp config list` describes each
option.

### 4. Generate

```sh
frp generate   # write the generated modules
frp check      # fail if they differ from what generate would write, e.g. in CI
```

`myapp/generated/models.py`:

```python
class EmployeeCard(BaseModel, frozen=True):
    id: EmployeeId
    name: str
    email: str  # narrowed from `str | None` by `require`
    department_name: str  # from the resolved, flattened Department


class EmployeeCreate(BaseModel, frozen=True):
    name: str
    email: str | None
    status: Status
    department_id: DepartmentId
```

`myapp/generated/mappings.py` has one function per View. It takes the from_model
row and a lookup per Reference it resolves. It returns `None` for a row the View's
`where` excludes, and raises `ValueError` for a row that does not meet a
`require`:

```python
def employee_card(
    source: myapp.entities.Employee,
    *,
    department_by_id: collections.abc.Mapping[
        myapp.entities.DepartmentId,
        myapp.entities.Department,
    ],
) -> myapp.generated.models.EmployeeCard | None: ...
```

## Steps

All in `functional_relational_python.steps`. A field is named with
`frp.FieldName("...")`. A step that takes `fields` accepts a single `FieldName` or a
`select(predicate)`, which acts on every field the predicate accepts.
`select.all()` selects every field.

| Step | Does |
| --- | --- |
| `omit(fields)` | Removes fields. |
| `pick(names)` | Keeps only `names`, in the from_model's order. |
| `rename(field, new)` | Renames a field. With a `select`, `new` maps each old name to its new name. |
| `where(condition)` | Keeps only the rows for which `condition` is true. Views only, not in a nested pipeline. |
| `require(fields)` | Narrows `X \| None` to `X`. A row whose field is `None` is an error, not dropped. |
| `resolve(fields, *then)` | Replaces each Reference with the Entity it points to, `department_id` becoming `department`. `then` is a pipeline applied to that Entity. `fields` is a `select` or `references()`. |
| `flatten(fields)` | Replaces a resolved field with the fields of its Entity. |
| `reverse(entity, *then, name=None, via_field=ViaField.OMIT)` | Adds a field holding what references this row by `entity`'s one Reference to it: for a one-to-one Reference the `entity` or `None`, for a many-to-one Reference every such `entity`, in a `SortedSequence` by `id`. The Reference back to this row is dropped unless `via_field=ViaField.KEEP`. |
| `reverse_via(entity, via, *then, ...)` | `reverse`, for an `entity` with several References to this row. |
| `add(name, annotation, default)` | Declares a field with no value from the from_model, e.g. a union tag. A View's added field needs a default. |
| `default(fields, value)` | Sets the default of fields in the generated model. |
| `retype(fields, transform)` / `retype_to(fields, annotation)` | Changes field types. WriteModels only. |
| `pipeline(*steps)` | Composes steps into one step. Define your own steps this way, typed `Step`. |

`Query.then(*steps)` extends a query, so Views can share a base query, and an
update WriteModel can extend the create WriteModel's.

### Conditions

`where`, and the database constraints below, take conditions from
`functional_relational_python.conditions`:

```python
status, email = c.field(frp.FieldName("status")), c.field(frp.FieldName("email"))

status.eq(Status.ACTIVE) & ~email.is_null()
```

A field reference's methods are the comparisons `eq`, `ne`, `lt`, `le`, `gt` and
`ge`, named as in Python's `operator` module, and `is_null()`. A comparison's
right-hand side is another field reference or a literal. Combine conditions with
`&`, `|` and `~`. `None == x` is false, as in SQL.

The functions `c.eq(status, Status.ACTIVE)`, `c.and_(...)`, `c.or_(...)`,
`c.not_(...)` and so on build the same conditions.

A `where` that compares a field with `eq` narrows its type in the View: after
`s.where(c.field(frp.FieldName("status")).eq(Status.ACTIVE))` the field is
`Literal[Status.ACTIVE]`.

## Unions

A `Union` is a tagged union of Views, or of WriteModels, generated as
`Annotated[A | B, Discriminator(...)]`:

```python
from typing import Literal

employee_row = frp.Union(
    name=frp.UnionName("EmployeeRow"),
    discriminator=frp.FieldName("kind"),
    variants=(
        frp.View(
            name=frp.ViewName("CurrentEmployee"),
            query=frp.from_(
                Employee,
                s.where(c.field(frp.FieldName("status")).eq(Status.ACTIVE)),
                s.add(frp.FieldName("kind"), Literal["current"], "current"),
            ),
        ),
        frp.View(
            name=frp.ViewName("FormerEmployee"),
            query=frp.from_(
                Employee,
                s.where(c.field(frp.FieldName("status")).ne(Status.ACTIVE)),
                s.add(frp.FieldName("kind"), Literal["former"], "former"),
            ),
        ),
    ),
)
```

Pass it in `Catalog(unions=[employee_row])`. Each row gives the result of the first
Variant whose `where` keeps it. A Variant's tag is either a field `add` declares or
a field its `where` narrows with `eq`.

## Catalog options

| Field | Is |
| --- | --- |
| `entities` | Every Entity. |
| `views`, `write_models`, `unions` | What to generate. |
| `layout` | A function from each compiled View, WriteModel or Union (`frp.CompiledView`, `frp.CompiledWriteModel`, `frp.CompiledUnion`, `frp.CompiledWriteModelUnion`) to the `ModulePath` its model goes to. |
| `mappings` | The `ModulePath` for every mapping function, or `frp.WithoutMappings()`. |
| `django` | `DjangoModels(...)` to generate a Django model per Entity and a QuerySet per View, or `WithoutDjangoModels()`. |
| `id_base` | The class ID types subclass. Defaults to `IdBase(Id)`. |
| `one_to_one_marker` | The class whose instances mark a Reference one-to-one, as in `Annotated[DepartmentId, OneToOne()]`. Defaults to `OneToOneMarker(OneToOne)`. |

With the defaults, Entity modules and generated code import frp at runtime. To
avoid that, define your own ID base (a `pydantic.RootModel` subclass) and
one-to-one marker (a frozen dataclass), and pass them as `IdBase(AppId)` and
`OneToOneMarker(AppOneToOne)`.

A `View` and `WriteModel` also take a `base`, the class their generated model
subclasses. Defaults to `pydantic.BaseModel`.

### Django

`functional_relational_python.django` holds its configuration. `DjangoModels` generates a Django model per Entity into `module`, and a QuerySet
and Manager per Entity into `queries`, with a query for each View. Only `module`,
`queries` and `app_label` are required:

```python
from django.db import models
from functional_relational_python import django as dj

django = dj.DjangoModels(
    module=frp.ModulePath("myapp.models"),
    queries=frp.ModulePath("myapp.queries"),
    app_label=dj.AppLabel("myapp"),
    every_model=dj.ModelOptions(ordering=(dj.OrderingTerm("-id"),)),
    entity_configs={
        Employee: dj.GeneratedModel(
            fields={
                frp.FieldName("department_id"): dj.FieldConfig(
                    on_delete=dj.OnDelete.CASCADE,
                    related_name=dj.RelatedName("employees"),
                ),
            },
            constraints=(dj.Unique(...),),
        ),
    },
)
```

`every_model` sets what every generated model shares: bases, decorators,
ordering, per-field options, and QuerySet and Manager bases. `entity_configs`
holds an Entity's own: a `GeneratedModel`, whose settings replace those of
`every_model` and whose `fields` combine with its, or an `ExistingDjangoModel`
for an Entity the application already has a Django model for. A `FieldConfig`
holds a field's `default`, and for a Reference its `on_delete` and
`related_name`. Bases, decorators and field classes are the objects themselves,
e.g. `models.Model`, or a `Symbol` for one the Catalog cannot import. Database
constraints are `Check(name, condition)` and `Unique(name, fields, where)`.
Migrations are yours to make, e.g. with `makemigrations`.

## Testing with the InMemoryViewEngine

`InMemoryViewEngine` runs Views over Entities held in memory. It defines what a
View means, and works as a fake in your own tests. `engine_for` builds an empty
one for a Catalog:

```python
from functional_relational_python.testing import engine_for

from myapp.frp_catalog import catalog

engine = engine_for(catalog).unwrap()
engine.put(Department(id=DepartmentId(1), name="Sales")).unwrap()
engine.put(
    Employee(
        id=EmployeeId(1),
        name="Ada",
        email="ada@example.com",
        status=Status.ACTIVE,
        department_id=DepartmentId(1),
    )
).unwrap()

engine.read(frp.ViewName("EmployeeCard"))
# Ok((ViewRow({FieldName("id"): EmployeeId(1), FieldName("name"): "Ada",
#              FieldName("email"): "ada@example.com",
#              FieldName("department_name"): "Sales"}),))
```

`read` returns every dangling Reference and unmet `require` as an error, rather
than raising at the first. `read_union` reads a Union.

## Errors

frp checks the whole Catalog before writing anything: unknown fields, name clashes,
`require` on a field that is not optional, a `where` in the wrong place, and so on.
`frp generate` reports every diagnostic at once and writes no files.
