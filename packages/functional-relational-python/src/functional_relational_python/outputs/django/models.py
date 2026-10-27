"""Renders a Django model per Entity, so a Django project can store Entities.

Each model's manager `objects` is its Entity's Manager, which `queries`
generates with a method per View over the Entity.

A one-to-one Reference is a `OneToOneField` with a reverse accessor on the
referenced Entity's model, `<entity>_by_<foreign key>`, e.g.
`company_profile_by_company` for `CompanyProfile.company_id`, so that a query can
follow it backwards. A many-to-one Reference has no reverse accessor, so several
never clash. A related name in the Entity's `DjangoModelOptions` names either's.

`DjangoModelOptions` also give a model its base classes, class decorators,
`Meta` options, field defaults, whether the database enforces its foreign keys,
and the bases of its QuerySet and Manager.

Everything here is pure: it builds file contents and touches no filesystem, and
it never imports Django. Migrations are the user's to make.
"""

import datetime
import enum
import keyword
import typing
import uuid
from collections.abc import Iterable, Iterator, Mapping

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import model_names
from functional_relational_python.entity_graph.entity_graph import (
    Cardinality,
    EntityGraph,
    EntityNode,
    FieldName,
    Optionality,
    Reference,
    RuntimeAnnotation,
)
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    LineLength,
    ModulePath,
    Symbol,
    TypeExpression,
    ValueExpression,
)
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)
from functional_relational_python.support.filesystem import FileText
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)
from functional_relational_python.support.identifier import Identifier


class DjangoField(pydantic.BaseModel, frozen=True):
    """A Django model field: its class and keyword arguments, e.g.
    `models.CharField(max_length=254)`.

    A field class in `django.db.models` is written as `models.<name>`; any other
    is imported by name.
    """

    models_module: typing.ClassVar[ModulePath] = ModulePath("django.db.models")

    field_class: Symbol
    arguments: Mapping[Identifier, ValueExpression]

    @staticmethod
    def models_symbol(name: Identifier) -> Symbol:
        """`name` of `django.db.models`."""
        return Symbol(module=DjangoField.models_module, qualname=(name,))

    @staticmethod
    def written(symbol: Symbol) -> TypeExpression:
        """`symbol` where the generated module refers to it: `models.<name>` in
        `django.db.models`, else by its imported name."""
        if symbol.module == DjangoField.models_module:
            return TypeExpression(f"models.{symbol.reference().root}")
        return symbol.reference()

    @staticmethod
    def builtin(
        name: Identifier, arguments: Mapping[Identifier, ValueExpression]
    ) -> DjangoField:
        """The field class `name` of `django.db.models`."""
        return DjangoField(
            field_class=DjangoField.models_symbol(name), arguments=arguments
        )

    @staticmethod
    def fake() -> DjangoField:
        return DjangoField.builtin(Identifier("TextField"), {})

    auto_increment: typing.ClassVar[frozenset[Identifier]] = frozenset(
        {
            Identifier("AutoField"),
            Identifier("BigAutoField"),
            Identifier("SmallAutoField"),
        }
    )

    def is_builtin(self) -> bool:
        return self.field_class.module == DjangoField.models_module

    def is_auto_increment(self) -> bool:
        """Whether this is one of Django's auto-incrementing fields, which only a
        primary key may be."""
        match self.field_class.qualname:
            case (name,) if self.is_builtin():
                return name in DjangoField.auto_increment
            case _:
                return False

    def reference(self) -> TypeExpression:
        """The field class, where the generated module refers to it."""
        return DjangoField.written(self.field_class)


class OnDelete(enum.StrEnum):
    """What deleting a referenced Entity does to the Entities referencing it.

    `SET_DEFAULT` is left out: nothing checks that the Reference has a default.
    """

    CASCADE = "CASCADE"
    PROTECT = "PROTECT"
    RESTRICT = "RESTRICT"
    SET_NULL = "SET_NULL"
    DO_NOTHING = "DO_NOTHING"


class ReferenceKey(pydantic.BaseModel, frozen=True):
    """A Reference, by the Entity declaring it and its field."""

    entity: type[pydantic.BaseModel]
    field: FieldName


class AppLabel(pydantic.RootModel[str], frozen=True):
    """A Django app label, e.g. `app`. It is a Python identifier."""

    @pydantic.field_validator("root")
    @classmethod
    def _identifier(cls, root: str) -> str:
        _ = Identifier(root)
        return root

    @staticmethod
    def fake() -> AppLabel:
        return AppLabel("app")


class _DjangoName(pydantic.RootModel[str], frozen=True):
    """A name Django accepts for a field or a related name: an identifier, not a
    keyword, neither ending in `_` nor holding `__`."""

    @pydantic.field_validator("root")
    @classmethod
    def _accepted(cls, root: str) -> str:
        _ = Identifier(root)
        if root.endswith("_") or "__" in root:
            msg = f"'{root}' ends in '_' or holds '__', which Django rejects"
            raise ValueError(msg)
        return root


class RelatedName(_DjangoName, frozen=True):
    """The attribute of a referenced Entity's Django model holding the models
    referencing it through one Reference, e.g. `ad_sets`: Django's
    `related_name`."""

    @staticmethod
    def fake() -> RelatedName:
        return RelatedName("referencing")

    def field_name(self) -> FieldName:
        return FieldName(self.root)


class ForeignKeyConstraint(enum.StrEnum):
    """Whether the database enforces a foreign key: `NOT_ENFORCED` is Django's
    `db_constraint=False`, for a table whose referenced rows may be missing or
    elsewhere."""

    ENFORCED = "enforced"
    NOT_ENFORCED = "not enforced"


type DjangoValue = Symbol | ValueExpression
"""A value in generated source: a Symbol is imported and referred to by name,
e.g. `uuid.uuid4` or the enum member `Kind.INSTANCE`, and an expression is
written as it is."""


class Default(pydantic.BaseModel, frozen=True):
    """`default=<value>`: the value Django gives a new model's field. A callable,
    e.g. `uuid.uuid4`, is called for each new model."""

    value: DjangoValue


class DatabaseDefault(pydantic.BaseModel, frozen=True):
    """`db_default=<value>`: the value the database gives a new row's column."""

    value: DjangoValue


class AutoNow(pydantic.BaseModel, frozen=True):
    """`auto_now=True`: Django sets the field to now on every save."""


class AutoNowAdd(pydantic.BaseModel, frozen=True):
    """`auto_now_add=True`: Django sets the field to now when the model is
    created."""


type FieldDefault = Default | DatabaseDefault | AutoNow | AutoNowAdd


class BareDecorator(pydantic.BaseModel, frozen=True):
    """`@<function>` on a model class."""

    function: Symbol


class CalledDecorator(pydantic.BaseModel, frozen=True):
    """`@<function>(<arguments>)` on a model class, e.g.
    `@register(follow=['channel_campaign'])`."""

    function: Symbol
    arguments: Mapping[Identifier, DjangoValue]


type ClassDecorator = BareDecorator | CalledDecorator


class DbTable(pydantic.RootModel[str], frozen=True):
    """The table a Django model is stored in, e.g. `djank_djangochannelcampaign`,
    to map a model onto an existing table."""

    @pydantic.field_validator("root")
    @classmethod
    def _not_empty(cls, root: str) -> str:
        if not root:
            msg = "a table name cannot be empty"
            raise ValueError(msg)
        return root


class DefaultDbTable(pydantic.BaseModel, frozen=True):
    """The table Django names, `<app label>_<model name>`."""


class OrderingTerm(pydantic.RootModel[str], frozen=True):
    """A field a Django model's rows are ordered by, e.g. `-created_at` for
    descending, or a field of a related model, e.g. `company__name`."""

    @pydantic.field_validator("root")
    @classmethod
    def _field(cls, root: str) -> str:
        _ = Identifier(root.removeprefix("-"))
        return root


class IndexField(pydantic.RootModel[str], frozen=True):
    """A field an index covers, e.g. `created_at`, or `-created_at` for
    descending."""

    @pydantic.field_validator("root")
    @classmethod
    def _field(cls, root: str) -> str:
        _ = _DjangoName(root.removeprefix("-"))
        return root


class IndexName(pydantic.RootModel[str], frozen=True):
    """The name of an index: at most 30 characters, starting with a letter, as
    Django requires."""

    longest: typing.ClassVar[int] = 30

    @pydantic.field_validator("root")
    @classmethod
    def _accepted(cls, root: str) -> str:
        _ = Identifier(root)
        if len(root) > IndexName.longest or not root[0].isalpha():
            msg = (
                f"'{root}' is longer than {IndexName.longest} characters or does "
                "not start with a letter"
            )
            raise ValueError(msg)
        return root


class DjangoIndex(pydantic.BaseModel, frozen=True):
    """`models.Index(fields=[...], name=...)` in a model's `Meta.indexes`."""

    fields: typing.Annotated[tuple[IndexField, ...], pydantic.Field(min_length=1)]
    name: IndexName


class DjangoMeta(pydantic.BaseModel, frozen=True):
    """A Django model's options, besides its app label.

    Ordering terms and index fields are not checked against the Entity's fields,
    since a base class may declare more; Django's system checks report one the
    model lacks.
    """

    db_table: DbTable | DefaultDbTable
    ordering: tuple[OrderingTerm, ...]
    indexes: tuple[DjangoIndex, ...]

    @staticmethod
    def default() -> DjangoMeta:
        return DjangoMeta(db_table=DefaultDbTable(), ordering=(), indexes=())


class ModelGenericBase(pydantic.BaseModel, frozen=True):
    """A base class generic in the Django model, subscripted with the Entity's,
    e.g. `SoftDeleteManager["app.models.Company"]`."""

    origin: Symbol


class PlainBase(pydantic.BaseModel, frozen=True):
    """A base class used as it is, e.g. one written for a single Entity."""

    origin: Symbol


type DjangoClassBase = ModelGenericBase | PlainBase


class DjangoModelOptions(pydantic.BaseModel, frozen=True):
    """How an Entity's Django model is declared, besides its fields' types.

    `bases` are the model's base classes, e.g. abstract models and mixins, in
    order. `decorators` decorate the model class, outermost first. The model's
    `Meta` does not subclass a base's `Meta`, so `meta` holds every option it
    needs, e.g. an `ordering` an abstract base declares.

    `defaults` and `related_names` are by the Entity's field, a Reference by
    its own name, e.g. `company_id`. A Reference with a related name gets a
    reverse accessor of that name on the referenced Entity's model. A one-to-one
    Reference has one anyway, see `DjangoModels.one_to_one_accessor`.
    `foreign_key_constraint` applies to every Reference of the Entity.

    The Entity's QuerySet subclasses `query_set_base`, and its Manager
    `manager_base`, see `DjangoQueriesRenderer`. A Manager base's
    `get_queryset()` must build its QuerySet with Django's, e.g. by filtering
    `super().get_queryset()`, so that it is the Entity's QuerySet.
    """

    bases: typing.Annotated[tuple[Symbol, ...], pydantic.Field(min_length=1)]
    decorators: tuple[ClassDecorator, ...]
    meta: DjangoMeta
    defaults: Mapping[FieldName, FieldDefault]
    related_names: Mapping[FieldName, RelatedName]
    foreign_key_constraint: ForeignKeyConstraint
    query_set_base: DjangoClassBase
    manager_base: DjangoClassBase

    @staticmethod
    def default() -> DjangoModelOptions:
        """Django's: a plain `models.Model`, with Django's QuerySet and Manager."""
        return DjangoModelOptions(
            bases=(DjangoField.models_symbol(Identifier("Model")),),
            decorators=(),
            meta=DjangoMeta.default(),
            defaults={},
            related_names={},
            foreign_key_constraint=ForeignKeyConstraint.ENFORCED,
            query_set_base=ModelGenericBase(
                origin=DjangoField.models_symbol(Identifier("QuerySet"))
            ),
            manager_base=ModelGenericBase(
                origin=DjangoField.models_symbol(Identifier("Manager"))
            ),
        )

    @staticmethod
    def fake() -> DjangoModelOptions:
        return DjangoModelOptions.default()


class DjangoModels(pydantic.BaseModel, frozen=True):
    """Generate a Django model per Entity, all into `module` with `app_label`, and
    the Django queries for Views into `queries`.

    `field_types` overrides the built-in rule for a type, see `FieldTypes`.
    `on_delete` sets what deleting the referenced Entity does, per Reference; a
    Reference not in it is `PROTECT`. `queries` holds a QuerySet and a Manager per
    Entity, which its model's manager is, see `DjangoQueriesRenderer`.
    `model_options` declares every Entity's model, except one in
    `entity_model_options`, which replaces them for that Entity.
    """

    module: ModulePath
    app_label: AppLabel
    field_types: Mapping[RuntimeAnnotation, DjangoField]
    on_delete: Mapping[ReferenceKey, OnDelete]
    queries: ModulePath
    model_options: DjangoModelOptions
    entity_model_options: Mapping[type[pydantic.BaseModel], DjangoModelOptions]

    @staticmethod
    def fake() -> DjangoModels:
        return DjangoModels(
            module=ModulePath("app.models"),
            app_label=AppLabel.fake(),
            field_types={},
            on_delete={},
            queries=ModulePath("app.queries"),
            model_options=DjangoModelOptions.fake(),
            entity_model_options={},
        )

    def options_of(self, entity: type[pydantic.BaseModel]) -> DjangoModelOptions:
        """How `entity`'s Django model is declared."""
        return self.entity_model_options.get(entity, self.model_options)

    def one_to_one_accessor(
        self, entity: type[pydantic.BaseModel], reference: FieldName
    ) -> FieldName:
        """The attribute of the referenced Entity's Django model holding the
        `entity` whose one-to-one Reference `reference` points to it: its related
        name, else e.g. `company_profile_by_company` for
        `CompanyProfile.company_id`."""
        match self.options_of(entity).related_names.get(reference):
            case RelatedName() as related_name:
                return related_name.field_name()
            case None:
                snake = model_names.PascalCase(entity.__name__).snake_case().root
                foreign_key = DjangoModelsRenderer.foreign_key_name(reference)
                return FieldName(f"{snake}_by_{foreign_key.root}")

    def model_of(self, entity: type[pydantic.BaseModel]) -> Symbol:
        """The Django model of `entity`, named like it."""
        return Symbol(module=self.module, qualname=(Identifier(entity.__name__),))

    def query_set_of(self, entity: type[pydantic.BaseModel]) -> Symbol:
        """The QuerySet of `entity`'s Django model, e.g. `CompanyQuerySet`."""
        return Symbol(
            module=self.queries,
            qualname=(Identifier(f"{entity.__name__}QuerySet"),),
        )

    def manager_of(self, entity: type[pydantic.BaseModel]) -> Symbol:
        """The Manager of `entity`'s Django model, e.g. `CompanyManager`."""
        return Symbol(
            module=self.queries,
            qualname=(Identifier(f"{entity.__name__}Manager"),),
        )


class WithoutDjangoModels(pydantic.BaseModel, frozen=True):
    """Generate no Django models."""


class ModelOptionsForNonEntity(pydantic.BaseModel, Diagnostic, frozen=True):
    """`entity_model_options` names a class that is not an Entity."""

    entity: type[pydantic.BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}: DjangoModels.entity_model_options names "
            "it, but it is not an Entity."
        )


class DefaultForNonField(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity's model options give a default to a field it does not have."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: its Django model "
            "options give it a default, but the Entity has no such field."
        )


class RelatedNameForNonReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity's model options give a related name to a field that is not a
    Reference of it."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: its Django model "
            "options give it a related name, but it is not a Reference."
        )


class NoDjangoFieldRule(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity field's type, and every type it wraps, has no Django field rule."""

    entity: type[pydantic.BaseModel]
    field: FieldName
    annotation: AnnotationText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: no Django field for "
            f"'{self.annotation.root}'. Add one to DjangoModels.field_types."
        )


class OnDeleteForNonReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`on_delete` names a field that is not a Reference of an Entity."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: DjangoModels.on_delete "
            "names it, but it is not a Reference of an Entity."
        )


class SetNullOnRequiredReference(pydantic.BaseModel, Diagnostic, frozen=True):
    """`on_delete` is `SET_NULL` for a Reference that cannot be `None`."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: on_delete is SET_NULL, "
            "but the Reference is not optional."
        )


class DjangoFieldNameClash(pydantic.BaseModel, Diagnostic, frozen=True):
    """Several fields of an Entity would give its Django model one attribute.

    A Reference `x_id` becomes the foreign key `x`, whose ID attribute is `x_id`.
    """

    entity: type[pydantic.BaseModel]
    name: FieldName
    fields: tuple[FieldName, ...]

    def describe(self) -> DiagnosticMessage:
        fields = ", ".join(f"'{field.root}'" for field in self.fields)
        return DiagnosticMessage(
            f"{self.entity.__qualname__}: the fields {fields} would each be the "
            f"Django model attribute '{self.name.root}'."
        )


class DjangoModelNameClash(pydantic.BaseModel, Diagnostic, frozen=True):
    """Several Entities, or an Entity and an import, have one name in the module
    their Django models are generated into."""

    name: Identifier

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"Django models: several Entities or imports are named '{self.name.root}'."
        )


class InvalidDjangoFieldName(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity field would be a Django model field with a name Django or the
    generated module cannot use: not an identifier, `pk`, `Meta`, ending in `_`,
    containing `__`, `objects`, which is the manager, or `models` or an imported
    field class or Manager, which the class body would shadow.

    A Reference `x_id` is the foreign key `x`, so `name` may differ from `field`.
    """

    entity: type[pydantic.BaseModel]
    field: FieldName
    name: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: '{self.name.root}' "
            "cannot be the name of a field of its Django model."
        )


class ReverseAccessorClash(pydantic.BaseModel, Diagnostic, frozen=True):
    """The reverse accessor of a Reference, one-to-one or with a related name,
    would have a name the referenced Entity's Django model binds otherwise: a
    field, an ID attribute, a name the model reserves, or the reverse accessor
    of another Reference.

    `references` are the References whose reverse accessor it is.
    """

    entity: type[pydantic.BaseModel]
    name: FieldName
    references: tuple[ReferenceKey, ...]

    def describe(self) -> DiagnosticMessage:
        references = ", ".join(
            f"{reference.entity.__qualname__}.{reference.field.root}"
            for reference in self.references
        )
        return DiagnosticMessage(
            f"{self.entity.__qualname__}: '{self.name.root}', the reverse accessor "
            f"of {references}, is bound more than once on its Django model."
        )


class InvalidReverseAccessorName(pydantic.BaseModel, Diagnostic, frozen=True):
    """The reverse accessor of a one-to-one Reference would have a name Django
    rejects as a related name: one ending in `_` or holding `__`, e.g. for an
    Entity named with `__`."""

    name: FieldName
    reference: ReferenceKey

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.reference.entity.__qualname__}.{self.reference.field.root}: its "
            f"reverse accessor '{self.name.root}' cannot be a Django related name, "
            "since Django rejects one ending in '_' or holding '__'. Rename the "
            "Entity or the field."
        )


class AutoIncrementOnNonPrimaryKey(pydantic.BaseModel, Diagnostic, frozen=True):
    """A type override gives a field other than the primary key an
    auto-incrementing Django field, e.g. an override keyed on `int`."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: only a primary key can "
            "auto-increment. Key the override on the Entity's ID type instead."
        )


type _DeclarationDiagnostic = NoDjangoFieldRule | AutoIncrementOnNonPrimaryKey


type DjangoDiagnostic = (
    NoDjangoFieldRule
    | InvalidDjangoFieldName
    | AutoIncrementOnNonPrimaryKey
    | OnDeleteForNonReference
    | SetNullOnRequiredReference
    | DjangoFieldNameClash
    | DjangoModelNameClash
    | ReverseAccessorClash
    | InvalidReverseAccessorName
    | ModelOptionsForNonEntity
    | DefaultForNonField
    | RelatedNameForNonReference
)


class FieldRole(enum.StrEnum):
    """What an Entity field is in its Django model."""

    PRIMARY_KEY = "primary key"
    FOREIGN_KEY = "foreign key"
    PLAIN = "plain"


class NoFieldRuleError(Exception):
    def __init__(self, annotation: RuntimeAnnotation) -> None:
        self.annotation = annotation
        super().__init__(f"No Django field for {annotation!r}")


class FieldTypes(pydantic.BaseModel, frozen=True):
    """The Django field for an Entity field's type.

    The type is unwrapped one RootModel at a time, e.g. `Email` to `Text` to
    `str`. The first type with a rule, overridden or built in, gives the field.
    The field is nullable when any of those types admits `None`.

    A primary key has no built-in rule for `str`: `TextField` cannot be a primary
    key on every database, and a `CharField` needs a `max_length` only the user
    knows. Foreign keys to it take its type.
    """

    builtin: typing.ClassVar[Mapping[RuntimeAnnotation, DjangoField]] = {
        str: DjangoField.builtin(Identifier("TextField"), {}),
        int: DjangoField.builtin(Identifier("BigIntegerField"), {}),
        bool: DjangoField.builtin(Identifier("BooleanField"), {}),
        float: DjangoField.builtin(Identifier("FloatField"), {}),
        bytes: DjangoField.builtin(Identifier("BinaryField"), {}),
        datetime.datetime: DjangoField.builtin(Identifier("DateTimeField"), {}),
        datetime.date: DjangoField.builtin(Identifier("DateField"), {}),
        datetime.time: DjangoField.builtin(Identifier("TimeField"), {}),
        datetime.timedelta: DjangoField.builtin(Identifier("DurationField"), {}),
        uuid.UUID: DjangoField.builtin(Identifier("UUIDField"), {}),
    }

    overrides: Mapping[RuntimeAnnotation, DjangoField]

    def field_of(
        self, annotation: RuntimeAnnotation, role: FieldRole
    ) -> Result[_TypedField, NoFieldRuleError]:
        optionality = Optionality.REQUIRED
        current = annotation
        while True:
            split = Optionality.of_annotation(current)
            if split.optionality == Optionality.OPTIONAL:
                optionality = Optionality.OPTIONAL
            current = split.required
            if current in self.overrides:
                return Ok(
                    _TypedField(field=self.overrides[current], optionality=optionality)
                )
            if isinstance(current, type) and issubclass(current, pydantic.RootModel):
                current = current.model_fields["root"].annotation
                continue
            if current in FieldTypes.builtin and not (
                role == FieldRole.PRIMARY_KEY and current is str
            ):
                return Ok(
                    _TypedField(
                        field=FieldTypes.builtin[current], optionality=optionality
                    )
                )
            return Err(NoFieldRuleError(annotation))


class _TypedField(pydantic.BaseModel, frozen=True):
    field: DjangoField
    optionality: Optionality


class _Declaration(pydantic.BaseModel, frozen=True):
    """One attribute of a Django model, e.g. `name = models.TextField()`.

    `name` is not yet checked to be one Django accepts. `source` is the Entity
    field it holds.
    """

    name: FieldName
    field: DjangoField
    positional: tuple[ValueExpression, ...]
    source: FieldName
    role: FieldRole
    imported: tuple[Symbol, ...]
    """What its arguments refer to by name, e.g. the callable of a default."""

    def defaulted(self, default: FieldDefault) -> _Declaration:
        """With `default` as a keyword argument."""
        match default:
            case Default(value=value):
                argument, written = Identifier("default"), value
            case DatabaseDefault(value=value):
                argument, written = Identifier("db_default"), value
            case AutoNow():
                argument, written = Identifier("auto_now"), ValueExpression("True")
            case AutoNowAdd():
                argument, written = Identifier("auto_now_add"), ValueExpression("True")
        expression = DjangoModelsRenderer.value(written)
        return self.model_copy(
            update={
                "field": self.field.model_copy(
                    update={"arguments": {**self.field.arguments, argument: expression}}
                ),
                "imported": (
                    *self.imported,
                    *((written,) if isinstance(written, Symbol) else ()),
                ),
            }
        )

    def attributes(self) -> tuple[FieldName, ...]:
        """The names it binds on the model: a foreign key `x` also binds `x_id`,
        its column and ID attribute."""
        match self.role:
            case FieldRole.FOREIGN_KEY:
                return (self.name, DjangoModelsRenderer.id_attribute(self.name))
            case FieldRole.PRIMARY_KEY | FieldRole.PLAIN:
                return (self.name,)

    def has_valid_name(self, imported: frozenset[Identifier]) -> bool:
        """Whether Django accepts the name, and the class body does not shadow
        `django.db.models`, the options class `Meta`, the manager or an
        `imported` name."""
        root = self.name.root
        return not (
            not root.isidentifier()
            or keyword.iskeyword(root)
            or self.name in DjangoModelsRenderer.reserved
            or root.endswith("_")
            or "__" in root
            or Identifier(root) in imported
        )

    def line(self) -> FileText:
        arguments = ", ".join(
            [
                *(value.root for value in self.positional),
                *(
                    f"{name.root}={value.root}"
                    for name, value in self.field.arguments.items()
                ),
            ]
        )
        return FileText(
            f"    {self.name.root} = {self.field.reference().root}({arguments})\n"
        )


class _NoReverseAccessor(pydantic.BaseModel, frozen=True):
    """A Reference whose referenced Entity's model has no attribute for it:
    `related_name='+'`."""


class _PlannedModel(pydantic.BaseModel, frozen=True):
    """`manager` is the class of its manager `objects`."""

    entity: type[pydantic.BaseModel]
    name: Identifier
    declarations: tuple[_Declaration, ...]
    manager: Symbol
    options: DjangoModelOptions

    def referred(self) -> Iterator[Symbol]:
        """Every Symbol the model refers to by name, `django.db.models` ones
        too."""
        for declaration in self.declarations:
            yield declaration.field.field_class
            yield from declaration.imported
        yield self.manager
        yield from self.options.bases
        for decorator in self.options.decorators:
            yield decorator.function
            match decorator:
                case CalledDecorator(arguments=arguments):
                    yield from (
                        value
                        for value in arguments.values()
                        if isinstance(value, Symbol)
                    )
                case BareDecorator():
                    pass

    def statement(self, app_label: AppLabel) -> FileText:
        """The class statement, decorators first."""
        decorators = "".join(
            f"@{DjangoModelsRenderer.decorator(decorator).root}\n"
            for decorator in self.options.decorators
        )
        bases = ", ".join(DjangoField.written(base).root for base in self.options.bases)
        meta = self.options.meta
        options = [f"app_label = {app_label.root!r}"]
        match meta.db_table:
            case DbTable(root=table):
                options.append(f"db_table = {table!r}")
            case DefaultDbTable():
                pass
        if meta.ordering:
            options.append(f"ordering = {[term.root for term in meta.ordering]!r}")
        if meta.indexes:
            indexes = ", ".join(
                f"models.Index(fields={[field.root for field in index.fields]!r}, "
                f"name={index.name.root!r})"
                for index in meta.indexes
            )
            options.append(f"indexes = [{indexes}]")
        return FileText(
            f"{decorators}class {self.name.root}({bases}):\n"
            f"    {DjangoModelsRenderer.manager.root} = "
            f"{self.manager.bound_name().root}()\n"
            + "".join(declaration.line().root for declaration in self.declarations)
            + "\n    class Meta:\n"
            + "".join(f"        {option}\n" for option in options)
        )


class DjangoModelsRenderer:
    """Writes the Django model of every Entity into one module."""

    header: typing.ClassVar[FileText] = FileText(
        f"{GeneratedFiles.header.root}"
        "# Regenerate with `frp generate`; `frp check` fails when this is stale.\n"
        "# Migrations are yours to make, e.g. with `makemigrations`.\n"
    )
    models_import: typing.ClassVar[Identifier] = Identifier("models")
    manager: typing.ClassVar[FieldName] = FieldName("objects")
    reserved: typing.ClassVar[frozenset[FieldName]] = frozenset(
        {FieldName("pk"), FieldName("Meta"), FieldName("models"), manager}
    )
    """Names a field cannot have: Django's alias for the primary key, the
    model's options class, the imported `django.db.models`, and the manager."""

    @staticmethod
    def render(
        graph: EntityGraph, config: DjangoModels
    ) -> Result[GeneratedFiles, DiagnosticsError[DjangoDiagnostic]]:
        """The module of Django models, or every diagnostic.

        Models are in name order, and refer to each other by name, so the order of
        Entities does not matter.
        """
        types = FieldTypes(overrides=config.field_types)
        reported: list[DjangoDiagnostic] = []
        planned: list[_PlannedModel] = []
        references = {
            ReferenceKey(entity=model, field=reference.field): reference
            for model, node in graph.entities.items()
            for reference in node.references
        }
        for key, on_delete in config.on_delete.items():
            match references.get(key):
                case None:
                    reported.append(
                        OnDeleteForNonReference(entity=key.entity, field=key.field)
                    )
                case Reference(optionality=Optionality.REQUIRED) if (
                    on_delete == OnDelete.SET_NULL
                ):
                    reported.append(
                        SetNullOnRequiredReference(entity=key.entity, field=key.field)
                    )
                case Reference():
                    pass
        reported.extend(
            ModelOptionsForNonEntity(entity=entity)
            for entity in config.entity_model_options
            if entity not in graph.entities
        )
        for model, node in sorted(
            graph.entities.items(), key=lambda item: item[0].__name__
        ):
            options = config.options_of(model)
            reported.extend(
                DjangoModelsRenderer._misplaced_options(model, node, options)
            )
            declarations: list[_Declaration] = []
            for declared in DjangoModelsRenderer._declarations(
                model, node, types, config
            ):
                match declared:
                    case _Declaration():
                        declarations.append(declared)
                    case NoDjangoFieldRule() | AutoIncrementOnNonPrimaryKey():
                        reported.append(declared)
            reported.extend(DjangoModelsRenderer._field_clashes(model, declarations))
            reported.extend(
                DjangoModelsRenderer._accessor_clashes(
                    graph, config, model, declarations
                )
            )
            planned.append(
                _PlannedModel(
                    entity=model,
                    name=Identifier(model.__name__),
                    declarations=tuple(declarations),
                    manager=config.manager_of(model),
                    options=options,
                )
            )
        imports = frozenset(
            symbol.imported()
            for model in planned
            for symbol in model.referred()
            if symbol.module != DjangoField.models_module
        )
        imported_names = frozenset(symbol.bound_name() for symbol in imports)
        reported.extend(
            InvalidDjangoFieldName(
                entity=model.entity, field=declaration.source, name=declaration.name
            )
            for model in planned
            for declaration in model.declarations
            if not declaration.has_valid_name(imported_names)
        )
        reported.extend(DjangoModelsRenderer._name_clashes(planned, imports))
        return DiagnosticsError.or_ok(
            sorted(reported, key=lambda diagnostic: diagnostic.describe().root),
            lambda: GeneratedFiles(
                files=(
                    GeneratedFile(
                        path=config.module.file_path(),
                        text=DjangoModelsRenderer._module(
                            config.app_label, planned, imports
                        ),
                    ),
                )
            ),
        )

    @staticmethod
    def _misplaced_options(
        model: type[pydantic.BaseModel],
        node: EntityNode,
        options: DjangoModelOptions,
    ) -> Iterator[DefaultForNonField | RelatedNameForNonReference]:
        """Defaults for fields the Entity lacks, and related names for fields
        that are not its References."""
        yield from (
            DefaultForNonField(entity=model, field=field)
            for field in options.defaults
            if field.root not in model.model_fields
        )
        references = {reference.field for reference in node.references}
        yield from (
            RelatedNameForNonReference(entity=model, field=field)
            for field in options.related_names
            if field not in references
        )

    @staticmethod
    def decorator(decorator: ClassDecorator) -> ValueExpression:
        """The expression after `@`."""
        function = DjangoField.written(decorator.function).root
        match decorator:
            case BareDecorator():
                return ValueExpression(function)
            case CalledDecorator(arguments=arguments):
                written = ", ".join(
                    f"{name.root}={DjangoModelsRenderer.value(value).root}"
                    for name, value in arguments.items()
                )
                return ValueExpression(f"{function}({written})")

    @staticmethod
    def value(value: DjangoValue) -> ValueExpression:
        match value:
            case Symbol():
                return ValueExpression(DjangoField.written(value).root)
            case ValueExpression():
                return value

    @staticmethod
    def _declarations(
        model: type[pydantic.BaseModel],
        node: EntityNode,
        types: FieldTypes,
        config: DjangoModels,
    ) -> Iterator[_Declaration | _DeclarationDiagnostic]:
        """Each field's, with its default."""
        defaults = config.options_of(model).defaults
        for name in model.model_fields:
            declared = DjangoModelsRenderer._declaration(
                model, node, types, config, FieldName(name)
            )
            default = defaults.get(FieldName(name))
            match declared:
                case _Declaration() if default is not None:
                    yield declared.defaulted(default)
                case (
                    _Declaration()
                    | NoDjangoFieldRule()
                    | AutoIncrementOnNonPrimaryKey()
                ):
                    yield declared

    @staticmethod
    def _declaration(
        model: type[pydantic.BaseModel],
        node: EntityNode,
        types: FieldTypes,
        config: DjangoModels,
        field: FieldName,
    ) -> _Declaration | _DeclarationDiagnostic:
        for reference in node.references:
            if reference.field == field:
                return DjangoModelsRenderer._foreign_key(model, reference, config)
        info = model.model_fields[field.root]
        role = (
            FieldRole.PRIMARY_KEY
            if field == EntityNode.primary_key
            else FieldRole.PLAIN
        )
        match types.field_of(info.annotation, role):
            case Ok(typed):
                pass
            case Err():
                return NoDjangoFieldRule(
                    entity=model,
                    field=field,
                    annotation=AnnotationText.of(info.annotation),
                )
        if role == FieldRole.PLAIN and typed.field.is_auto_increment():
            return AutoIncrementOnNonPrimaryKey(entity=model, field=field)
        arguments = dict(typed.field.arguments)
        if typed.optionality == Optionality.OPTIONAL:
            arguments[Identifier("null")] = ValueExpression("True")
        if role == FieldRole.PRIMARY_KEY:
            arguments[Identifier("primary_key")] = ValueExpression("True")
        return _Declaration(
            name=field,
            field=typed.field.model_copy(update={"arguments": arguments}),
            positional=(),
            source=field,
            role=role,
            imported=(),
        )

    @staticmethod
    def foreign_key_name(reference: FieldName) -> FieldName:
        """The foreign key holding the Reference `reference`: its name without
        `_id`, e.g. `company_id` is `company`."""
        return FieldName(reference.root.removesuffix("_id"))

    @staticmethod
    def id_attribute(foreign_key: FieldName) -> FieldName:
        """The attribute holding the ID a foreign key points to, e.g. `company_id`
        for `company`. It is also the foreign key's column."""
        return FieldName(f"{foreign_key.root}_id")

    @staticmethod
    def attribute_of(node: EntityNode, field: FieldName) -> FieldName:
        """The Django model attribute holding the value of the Entity's `field`:
        the ID attribute of the foreign key for a Reference, else the field."""
        if any(reference.field == field for reference in node.references):
            return DjangoModelsRenderer.id_attribute(
                DjangoModelsRenderer.foreign_key_name(field)
            )
        return field

    @staticmethod
    def column_optionality(
        node: EntityNode, field: FieldName, field_types: FieldTypes
    ) -> Optionality:
        """Whether the column holding the Entity's `field` may be NULL: an optional
        Reference's, or one whose type, or a type it wraps, admits `None`. A field
        without a Django field rule may be."""
        for reference in node.references:
            if reference.field == field:
                return reference.optionality
        annotation = node.model.model_fields[field.root].annotation
        match field_types.field_of(annotation, FieldRole.PLAIN):
            case Ok(typed):
                return typed.optionality
            case Err():
                return Optionality.OPTIONAL

    @staticmethod
    def _accessor_clashes(
        graph: EntityGraph,
        config: DjangoModels,
        model: type[pydantic.BaseModel],
        declarations: Iterable[_Declaration],
    ) -> Iterator[ReverseAccessorClash | InvalidReverseAccessorName]:
        """Reverse accessors on `model`'s Django model named like another one, or
        like a name the model binds or reserves, or named as Django forbids.
        References are in Entity name and field order."""
        bound = {
            *DjangoModelsRenderer.reserved,
            *(
                attribute
                for declaration in declarations
                for attribute in declaration.attributes()
            ),
        }
        by_name: dict[FieldName, list[ReferenceKey]] = {}
        for entity, node in sorted(
            graph.entities.items(), key=lambda item: item[0].__name__
        ):
            for reference in node.references:
                if reference.target is not model:
                    continue
                match DjangoModelsRenderer._related_name(config, entity, reference):
                    case FieldName() as name:
                        by_name.setdefault(name, []).append(
                            ReferenceKey(entity=entity, field=reference.field)
                        )
                    case _NoReverseAccessor():
                        pass
        for name, references in by_name.items():
            if len(references) > 1 or name in bound:
                yield ReverseAccessorClash(
                    entity=model, name=name, references=tuple(references)
                )
            # It is an identifier, made of an Entity's and a field's name, and no
            # keyword, since it holds `_by_`; or a RelatedName, which is valid.
            if name.root.endswith("_") or "__" in name.root:
                yield from (
                    InvalidReverseAccessorName(name=name, reference=reference)
                    for reference in references
                )

    @staticmethod
    def _related_name(
        config: DjangoModels, model: type[pydantic.BaseModel], reference: Reference
    ) -> FieldName | _NoReverseAccessor:
        """The reverse accessor of `model`'s `reference`: a one-to-one
        Reference's, or the related name the model's options give a many-to-one
        one."""
        match reference.cardinality:
            case Cardinality.ONE_TO_ONE:
                return config.one_to_one_accessor(model, reference.field)
            case Cardinality.MANY_TO_ONE:
                pass
        match config.options_of(model).related_names.get(reference.field):
            case RelatedName() as related_name:
                return related_name.field_name()
            case None:
                return _NoReverseAccessor()

    @staticmethod
    def _foreign_key(
        model: type[pydantic.BaseModel], reference: Reference, config: DjangoModels
    ) -> _Declaration:
        """A foreign key to the referenced Entity's model, named without `_id`: a
        `OneToOneField` for a one-to-one Reference, else a `ForeignKey`. Without a
        reverse accessor, its related name is `'+'`."""
        deleted = config.on_delete.get(
            ReferenceKey(entity=model, field=reference.field), OnDelete.PROTECT
        )
        match reference.cardinality:
            case Cardinality.MANY_TO_ONE:
                field_class = Identifier("ForeignKey")
            case Cardinality.ONE_TO_ONE:
                field_class = Identifier("OneToOneField")
        match DjangoModelsRenderer._related_name(config, model, reference):
            case FieldName(root=related_name):
                pass
            case _NoReverseAccessor():
                related_name = "+"
        arguments = {
            Identifier("on_delete"): ValueExpression(f"models.{deleted.value}"),
            Identifier("related_name"): ValueExpression(repr(related_name)),
        }
        if reference.optionality == Optionality.OPTIONAL:
            arguments[Identifier("null")] = ValueExpression("True")
        match config.options_of(model).foreign_key_constraint:
            case ForeignKeyConstraint.ENFORCED:
                pass
            case ForeignKeyConstraint.NOT_ENFORCED:
                arguments[Identifier("db_constraint")] = ValueExpression("False")
        return _Declaration(
            name=DjangoModelsRenderer.foreign_key_name(reference.field),
            field=DjangoField.builtin(field_class, arguments),
            positional=(ValueExpression(repr(reference.target.__name__)),),
            source=reference.field,
            role=FieldRole.FOREIGN_KEY,
            imported=(),
        )

    @staticmethod
    def _field_clashes(
        model: type[pydantic.BaseModel], declarations: Iterable[_Declaration]
    ) -> Iterator[DjangoFieldNameClash]:
        bound: dict[FieldName, list[FieldName]] = {}
        for declaration in declarations:
            for attribute in declaration.attributes():
                bound.setdefault(attribute, []).append(declaration.source)
        for name, fields in bound.items():
            if len(fields) > 1:
                yield DjangoFieldNameClash(
                    entity=model,
                    name=name,
                    fields=tuple(sorted(fields, key=lambda field: field.root)),
                )

    @staticmethod
    def _name_clashes(
        planned: list[_PlannedModel], imports: frozenset[Symbol]
    ) -> Iterator[DjangoModelNameClash]:
        """`imports` are the Symbols the module imports by name: field classes,
        Managers, base classes, decorators and defaults."""
        bound: dict[Identifier, list[Symbol | Identifier]] = {
            DjangoModelsRenderer.models_import: [DjangoModelsRenderer.models_import]
        }
        for symbol in imports:
            bound.setdefault(symbol.bound_name(), []).append(symbol)
        for model in planned:
            bound.setdefault(model.name, []).append(model.name)
        for name, origins in bound.items():
            if len(origins) > 1:
                yield DjangoModelNameClash(name=name)

    @staticmethod
    def _module(
        app_label: AppLabel, planned: list[_PlannedModel], imports: frozenset[Symbol]
    ) -> FileText:
        by_module: dict[ModulePath, set[Identifier]] = {}
        for symbol in imports:
            by_module.setdefault(symbol.module, set()).add(symbol.bound_name())
        lines = ["from django.db import models"]
        for module, bound in by_module.items():
            names = sorted(name.root for name in bound)
            line = f"from {module.root} import {', '.join(names)}"
            if len(line) > LineLength.default().root:
                listed = "".join(f"    {name},\n" for name in names)
                line = f"from {module.root} import (\n{listed})"
            lines.append(line)
        imported = "\n".join(sorted(lines))
        classes = "\n\n".join(model.statement(app_label).root for model in planned)
        return FileText(
            f"{DjangoModelsRenderer.header.root}\n{imported}\n\n\n{classes}"
        )
