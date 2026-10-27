"""Renders a Django model per Entity, so a Django project can store Entities.

Each model's manager `objects` is its Entity's Manager, which `queries`
generates with a method per View over the Entity.

A one-to-one Reference is a `OneToOneField` with a reverse accessor on the
referenced Entity's model, `<entity>_by_<foreign key>`, e.g.
`company_profile_by_company` for `CompanyProfile.company_id`, so that a query can
follow it backwards. A many-to-one Reference has no reverse accessor, so several
never clash.

An Entity the application already has a Django model for, e.g. a hand-written
`Company`, gets none: a Reference to it is a foreign key to that model, which
the module imports. See `ExistingDjangoModel`.

Everything here is pure: it builds file contents and touches no filesystem, and
it never imports Django. Migrations are the user's to make.
"""

import datetime
import enum
import keyword
import math
import operator
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

    An argument is written as source (`ValueExpression("254")`), refers to a
    Symbol (`choices=Channel.choices`), or is another field
    (`ArrayField(base_field=models.TextField())`). A Symbol, or a field class, in
    `django.db.models` is written as `models.<name>`; any other is imported by
    name.
    """

    models_module: typing.ClassVar[ModulePath] = ModulePath("django.db.models")

    field_class: Symbol
    arguments: Mapping[Identifier, ValueExpression | Symbol | DjangoField]

    @staticmethod
    def builtin(
        name: Identifier, arguments: Mapping[Identifier, ValueExpression]
    ) -> DjangoField:
        """The field class `name` of `django.db.models`."""
        return DjangoField(
            field_class=Symbol(module=DjangoField.models_module, qualname=(name,)),
            arguments=arguments,
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

    @staticmethod
    def is_in_models_module(symbol: Symbol) -> bool:
        """Whether `symbol` is in `django.db.models`, which the generated module
        imports as `models`."""
        return symbol.module == DjangoField.models_module

    def is_auto_increment(self) -> bool:
        """Whether this is one of Django's auto-incrementing fields, which only a
        primary key may be."""
        match self.field_class.qualname:
            case (name,) if DjangoField.is_in_models_module(self.field_class):
                return name in DjangoField.auto_increment
            case _:
                return False

    def nested_fields(self) -> Iterator[DjangoField]:
        """The fields among its arguments, and theirs, e.g. an `ArrayField`'s
        `base_field`."""
        for argument in self.arguments.values():
            match argument:
                case DjangoField():
                    yield argument
                    yield from argument.nested_fields()
                case ValueExpression() | Symbol():
                    pass

    def with_arguments(
        self, arguments: Mapping[Identifier, ValueExpression | Symbol | DjangoField]
    ) -> DjangoField:
        """This field with `arguments` added, after its own."""
        return self.model_copy(update={"arguments": {**self.arguments, **arguments}})

    def to_expression(self, positional: Iterable[ValueExpression]) -> ValueExpression:
        """The field constructed with `positional` and its arguments, e.g.
        `models.ForeignKey('Company', on_delete=models.PROTECT)`."""
        arguments = ", ".join(
            [
                *(value.root for value in positional),
                *(
                    f"{name.root}={DjangoField._written(argument).root}"
                    for name, argument in self.arguments.items()
                ),
            ]
        )
        return ValueExpression(
            f"{DjangoField._referred(self.field_class).root}({arguments})"
        )

    def imports(self) -> Iterator[Symbol]:
        """What the generated module imports by name to construct the field: its
        class and the Symbols its arguments refer to, outside `django.db.models`."""
        yield from DjangoField._imported(self.field_class)
        for argument in self.arguments.values():
            match argument:
                case ValueExpression():
                    pass
                case Symbol():
                    yield from DjangoField._imported(argument)
                case DjangoField():
                    yield from argument.imports()

    @staticmethod
    def _written(argument: ValueExpression | Symbol | DjangoField) -> ValueExpression:
        match argument:
            case ValueExpression():
                return argument
            case Symbol():
                return ValueExpression(DjangoField._referred(argument).root)
            case DjangoField():
                return argument.to_expression(())

    @staticmethod
    def _referred(symbol: Symbol) -> TypeExpression:
        """`symbol`, where the generated module refers to it."""
        if DjangoField.is_in_models_module(symbol):
            return TypeExpression(f"models.{symbol.reference().root}")
        return symbol.reference()

    @staticmethod
    def _imported(symbol: Symbol) -> Iterator[Symbol]:
        """What the generated module imports to refer to `symbol`: nothing for
        `django.db.models`, which it imports as `models`."""
        if not DjangoField.is_in_models_module(symbol):
            yield symbol.imported()


class OnDelete(enum.StrEnum):
    """What deleting a referenced Entity does to the Entities referencing it.

    `SET_DEFAULT` is left out, because generated fields have no default.
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


class ExistingDjangoModel(pydantic.BaseModel, frozen=True):
    """A Django model the application already has for an Entity: the class `name`
    in `module`, e.g. flowbase's `DjangoCompany` in `djank.models.company`.

    frp generates no model for the Entity. The generated models import this one,
    and their foreign keys point to the class, so that type checkers see the
    foreign key's ID attribute. `module` must therefore not import the generated
    models while it is being imported. Django queries import it only when type
    checking.

    The model holds each field of the Entity under the attribute a generated
    model would, so that a query reads it: a field under its name, and a
    Reference `x_id` as the foreign key `x`. Its manager is its own.
    `reverse_accessors` is the `related_name` its model gives each one-to-one
    Reference of the Entity, e.g. `meta_detail` for `channel_campaign_id`, which a
    query follows backwards. frp checks none of this against the model.
    """

    module: ModulePath
    name: Identifier
    reverse_accessors: Mapping[FieldName, FieldName]

    @staticmethod
    def fake() -> ExistingDjangoModel:
        return ExistingDjangoModel(
            module=ModulePath("crm.models"),
            name=Identifier("Company"),
            reverse_accessors={},
        )

    def symbol(self) -> Symbol:
        return Symbol(module=self.module, qualname=(self.name,))


class DjangoModels(pydantic.BaseModel, frozen=True):
    """Generate a Django model per Entity, all into `module` with `app_label`, and
    the Django queries for Views into `queries`.

    `field_types` overrides the built-in rule for a type, see `FieldTypes`.
    `on_delete` sets what deleting the referenced Entity does, per Reference; a
    Reference not in it is `PROTECT`. `queries` holds a QuerySet and a Manager per
    Entity, which its model's manager is, see `DjangoQueriesRenderer`. `existing`
    names the Django model the application already has for an Entity, which then
    gets no generated model, QuerySet or Manager, see `ExistingDjangoModel`.
    """

    module: ModulePath
    app_label: AppLabel
    field_types: Mapping[RuntimeAnnotation, DjangoField]
    on_delete: Mapping[ReferenceKey, OnDelete]
    queries: ModulePath
    existing: Mapping[type[pydantic.BaseModel], ExistingDjangoModel]

    @staticmethod
    def fake() -> DjangoModels:
        return DjangoModels(
            module=ModulePath("app.models"),
            app_label=AppLabel.fake(),
            field_types={},
            on_delete={},
            queries=ModulePath("app.queries"),
            existing={},
        )

    def model_of(self, entity: type[pydantic.BaseModel]) -> Symbol:
        """The Django model of `entity`: the existing one, or the generated one,
        named like it."""
        match self.existing.get(entity):
            case ExistingDjangoModel() as existing:
                return existing.symbol()
            case None:
                return Symbol(
                    module=self.module, qualname=(Identifier(entity.__name__),)
                )

    def generates(self, entity: type[pydantic.BaseModel]) -> bool:
        """Whether frp generates `entity`'s Django model, QuerySet and Manager:
        unless the application has its own model for it."""
        return entity not in self.existing

    def reverse_accessor_of(
        self, entity: type[pydantic.BaseModel], reference: FieldName
    ) -> FieldName:
        """The attribute of the referenced Entity's Django model holding the
        `entity` whose one-to-one Reference `reference` points to it: the one an
        existing model declares, else the one frp generates, see
        `DjangoModelsRenderer.reverse_accessor`. An existing model declaring none
        is a diagnostic of the Django models."""
        match self.existing.get(entity):
            case ExistingDjangoModel(reverse_accessors=declared) if (
                reference in declared
            ):
                return declared[reference]
            case ExistingDjangoModel() | None:
                return DjangoModelsRenderer.reverse_accessor(entity, reference)

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


type StoredValue = object
"""A value a column stores, e.g. `'meta'` for the enum member `Channel.META`."""


class NoDjangoFieldRule(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity field's type, and every type it wraps, has no Django field rule.

    `unruled` is the type the rules stopped at: the field's type unwrapped, or a
    type inside it, e.g. the items of a list or the values of an enum.
    """

    entity: type[pydantic.BaseModel]
    field: FieldName
    annotation: AnnotationText
    unruled: AnnotationText

    def describe(self) -> DiagnosticMessage:
        inside = (
            ""
            if self.unruled == self.annotation
            else f", since there is none for '{self.unruled.root}' in it"
        )
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: no Django field for "
            f"'{self.annotation.root}'{inside}. Add one to DjangoModels.field_types."
        )


class UnwritableDjangoChoices(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity field's type is, or wraps, an enum or `Literal` whose values the
    generated module cannot write out as `choices`: only booleans, integers,
    finite floats, strings and bytes can be, not e.g. a `date` or a `UUID`.

    `values` are the values it cannot write.
    """

    entity: type[pydantic.BaseModel]
    field: FieldName
    values_of: AnnotationText
    values: tuple[StoredValue, ...]

    def describe(self) -> DiagnosticMessage:
        values = ", ".join(repr(value) for value in self.values)
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: the choices of "
            f"'{self.values_of.root}' cannot be written out, since they hold "
            f"{values}. Add a rule to DjangoModels.field_types."
        )


class UnimportableDjangoChoices(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity field's type is, or wraps, a Django `Choices` enum the generated
    module cannot import, e.g. one defined in a function, so its field cannot
    refer to the enum's `choices`."""

    entity: type[pydantic.BaseModel]
    field: FieldName
    enum_class: type[enum.Enum]

    def describe(self) -> DiagnosticMessage:
        name = self.enum_class.__qualname__
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: the Django field's "
            f"choices are '{name}.choices', but '{name}' cannot be imported by its "
            "module and name. Define it at module level, or add a rule to "
            "DjangoModels.field_types."
        )


class MixedDjangoChoiceTypes(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity field's type is, or wraps, an enum or `Literal` whose values have
    several types, e.g. `Literal["a", 1]`, so no one Django field holds them.

    `values_of` is the enum or `Literal`, and `value_types` are in name order.
    """

    entity: type[pydantic.BaseModel]
    field: FieldName
    values_of: AnnotationText
    value_types: tuple[type, ...]

    def describe(self) -> DiagnosticMessage:
        types = ", ".join(
            f"'{value_type.__qualname__}'" for value_type in self.value_types
        )
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: the values of "
            f"'{self.values_of.root}' are of the types {types}, but a Django field "
            "holds one. Add a rule to DjangoModels.field_types."
        )


class ExistingModelOfNonEntity(pydantic.BaseModel, Diagnostic, frozen=True):
    """`existing` names a Django model for a class that is not an Entity."""

    model: type[pydantic.BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.model.__qualname__}: DjangoModels.existing names a Django model "
            "for it, but it is not an Entity."
        )


class ExistingModelInGeneratedModule(pydantic.BaseModel, Diagnostic, frozen=True):
    """An existing Django model is in the module frp generates the Django models
    or queries into, which would overwrite it."""

    entity: type[pydantic.BaseModel]
    module: ModulePath

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}: its existing Django model is in "
            f"'{self.module.root}', which frp generates."
        )


class UndeclaredReverseAccessor(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity with an existing Django model has a one-to-one Reference whose
    reverse accessor its `ExistingDjangoModel.reverse_accessors` does not
    declare."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: a one-to-one Reference "
            "of an existing Django model, but its reverse_accessors does not name "
            "the related_name."
        )


class ReverseAccessorOfNonOneToOne(pydantic.BaseModel, Diagnostic, frozen=True):
    """`ExistingDjangoModel.reverse_accessors` names a field that is not a
    one-to-one Reference of its Entity."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: its existing Django "
            "model's reverse_accessors names it, but it is not a one-to-one "
            "Reference."
        )


class OnDeleteOnExistingModel(pydantic.BaseModel, Diagnostic, frozen=True):
    """`on_delete` names a Reference of an Entity whose Django model exists, so
    frp does not generate its foreign key."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: DjangoModels.on_delete "
            "names it, but its Entity's Django model is in DjangoModels.existing, "
            "so frp does not generate its foreign key."
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
    """The reverse accessor of a one-to-one Reference would have a name the
    referenced Entity's Django model binds otherwise: a field, an ID attribute, a
    name the model reserves, or the reverse accessor of another one-to-one
    Reference.

    `references` are the one-to-one References whose reverse accessor it is.
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
    auto-incrementing Django field, e.g. an override keyed on `int`, or nests one
    in its field, e.g. as the `base_field` of a list's `ArrayField`."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: only a primary key can "
            "auto-increment. Key the override on the Entity's ID type instead."
        )


type _DeclarationDiagnostic = (
    NoDjangoFieldRule
    | UnimportableDjangoChoices
    | UnwritableDjangoChoices
    | MixedDjangoChoiceTypes
    | AutoIncrementOnNonPrimaryKey
)


type DjangoDiagnostic = (
    NoDjangoFieldRule
    | UnimportableDjangoChoices
    | UnwritableDjangoChoices
    | MixedDjangoChoiceTypes
    | InvalidDjangoFieldName
    | AutoIncrementOnNonPrimaryKey
    | OnDeleteForNonReference
    | OnDeleteOnExistingModel
    | SetNullOnRequiredReference
    | ExistingModelOfNonEntity
    | ExistingModelInGeneratedModule
    | UndeclaredReverseAccessor
    | ReverseAccessorOfNonOneToOne
    | DjangoFieldNameClash
    | DjangoModelNameClash
    | ReverseAccessorClash
    | InvalidReverseAccessorName
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


class UnimportableChoicesError(Exception):
    def __init__(self, enum_class: type[enum.Enum]) -> None:
        self.enum_class = enum_class
        super().__init__(f"{enum_class!r} cannot be imported by its module and name")


class MixedValueTypesError(Exception):
    def __init__(
        self, annotation: RuntimeAnnotation, value_types: tuple[type, ...]
    ) -> None:
        self.annotation = annotation
        self.value_types = value_types
        super().__init__(f"The values of {annotation!r} have several types")


class UnwritableChoicesError(Exception):
    def __init__(
        self, values_of: RuntimeAnnotation, values: tuple[StoredValue, ...]
    ) -> None:
        self.values_of = values_of
        self.values = values
        super().__init__(f"The choices of {values_of!r} hold {values!r}")


type FieldRuleError = (
    NoFieldRuleError
    | UnimportableChoicesError
    | MixedValueTypesError
    | UnwritableChoicesError
)


class FieldTypes(pydantic.BaseModel, frozen=True):
    """The Django field for an Entity field's type.

    The type is unwrapped one layer at a time: a RootModel to its root, an
    `Annotated` to the type it annotates (its metadata is left out), and a `type`
    alias to its value, e.g. `Email` to `Text` to `str`. The first type with a
    rule, overridden or built in, gives the field. The field is nullable when any
    of those types admits `None`. An `Annotated` type is as pydantic records it,
    so an override for `PositiveInt` applies, but one for
    `Annotated[str, Field(min_length=1)]` does not: pydantic records it as
    `Annotated[str, MinLen(1)]`.

    Besides the rules for single types, there are built-in rules that look into a
    type:

    - An enum is the field of its values' type, with `choices`. A Django `Choices`
      enum's are `<enum>.choices`, which keeps its labels, so the generated module
      imports the enum; with `__empty__`, the field is nullable. Any other enum's
      are its values, each its own label. A Flag enum has no `choices`.
    - A `Literal` is the field of its values' type, with them as `choices`. An enum
      member's value is its `value`, and `None` makes the field nullable.
    - `list[X]` is a Postgres `ArrayField` whose `base_field` is the field of `X`.
      Override it, per list type, on another database.

    A primary key has no built-in rule for `str`: `TextField` cannot be a primary
    key on every database, and a `CharField` needs a `max_length` only the user
    knows. Foreign keys to it take its type.
    """

    django_choices: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("django.db.models.enums"), qualname=(Identifier("Choices"),)
    )
    array_field: typing.ClassVar[Symbol] = Symbol(
        module=ModulePath("django.contrib.postgres.fields"),
        qualname=(Identifier("ArrayField"),),
    )

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
    ) -> Result[_TypedField, FieldRuleError]:
        return self._field_of(annotation, role, frozenset())

    def _field_of(
        self,
        annotation: RuntimeAnnotation,
        role: FieldRole,
        expanding: frozenset[RuntimeAnnotation],
    ) -> Result[_TypedField, FieldRuleError]:
        """`expanding` are the `type` aliases and RootModels being unwrapped, so
        that a recursive one, e.g. `type Tree = list[Tree]`, has no rule."""
        optionality = Optionality.REQUIRED
        current = annotation
        while True:
            split = Optionality.of_annotation(current)
            if split.optionality == Optionality.OPTIONAL:
                optionality = Optionality.OPTIONAL
            current = split.required
            if FieldTypes._is_hashable(current) and current in self.overrides:
                return Ok(
                    _TypedField(field=self.overrides[current], optionality=optionality)
                )
            if typing.get_origin(current) is typing.Annotated:
                current = typing.get_args(current)[0]
                continue
            if isinstance(current, typing.TypeAliasType) or (
                isinstance(current, type) and issubclass(current, pydantic.RootModel)
            ):
                if current in expanding:
                    return Err(NoFieldRuleError(current))
                expanding = expanding | {current}
                current = (
                    current.__value__
                    if isinstance(current, typing.TypeAliasType)
                    else current.model_fields["root"].rebuild_annotation()
                )
                continue
            match self._builtin_of(current, role, expanding):
                case Ok(typed):
                    return Ok(typed.made_optional_by(optionality))
                case Err() as error:
                    return error

    def _builtin_of(
        self,
        annotation: RuntimeAnnotation,
        role: FieldRole,
        expanding: frozenset[RuntimeAnnotation],
    ) -> Result[_TypedField, FieldRuleError]:
        """The built-in rule for `annotation`, which no override names."""
        if (
            FieldTypes._is_hashable(annotation)
            and annotation in FieldTypes.builtin
            and not (role == FieldRole.PRIMARY_KEY and annotation is str)
        ):
            return Ok(
                _TypedField(
                    field=FieldTypes.builtin[annotation],
                    optionality=Optionality.REQUIRED,
                )
            )
        if typing.get_origin(annotation) is typing.Literal:
            return self._literal_of(annotation, role, expanding)
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return self._enum_of(annotation, role, expanding)
        if typing.get_origin(annotation) is list:
            (item,) = typing.get_args(annotation)
            match self._field_of(item, FieldRole.PLAIN, expanding):
                case Ok(typed):
                    return Ok(
                        _TypedField(
                            field=DjangoField(
                                field_class=FieldTypes.array_field,
                                arguments={Identifier("base_field"): typed.with_null()},
                            ),
                            optionality=Optionality.REQUIRED,
                        )
                    )
                case Err() as error:
                    return error
        return Err(NoFieldRuleError(annotation))

    def _literal_of(
        self,
        literal: RuntimeAnnotation,
        role: FieldRole,
        expanding: frozenset[RuntimeAnnotation],
    ) -> Result[_TypedField, FieldRuleError]:
        args = typing.get_args(literal)
        values = _ChoiceValues(
            values_of=literal,
            stored=tuple(
                _ChoiceValues.stored_of(value) for value in args if value is not None
            ),
            optionality=Optionality.OPTIONAL if None in args else Optionality.REQUIRED,
        )
        match values.listed_choices():
            case Ok(choices):
                return self._of_values(
                    values, {Identifier("choices"): choices}, role, expanding
                )
            case Err() as error:
                return error

    def _enum_of(
        self,
        enum_class: type[enum.Enum],
        role: FieldRole,
        expanding: frozenset[RuntimeAnnotation],
    ) -> Result[_TypedField, FieldRuleError]:
        """A Flag enum has no `choices`, since a value may combine members. A
        Django `Choices` enum with `__empty__` has a choice for `None`, so its
        field is nullable."""
        values = _ChoiceValues(
            values_of=enum_class,
            stored=tuple(_ChoiceValues.stored_of(member) for member in enum_class),
            optionality=Optionality.REQUIRED,
        )
        if issubclass(enum_class, enum.Flag):
            return self._of_values(values, {}, role, expanding)
        if not any(
            f"{base.__module__}.{base.__qualname__}"
            == FieldTypes.django_choices.describe().root
            for base in enum_class.__mro__
        ):
            match values.listed_choices():
                case Ok(choices):
                    return self._of_values(
                        values, {Identifier("choices"): choices}, role, expanding
                    )
                case Err() as error:
                    return error
        match Symbol.of(enum_class):
            case Ok(symbol):
                choices = Symbol(
                    module=symbol.module,
                    qualname=(*symbol.qualname, Identifier("choices")),
                )
                if hasattr(enum_class, "__empty__"):
                    values = values.model_copy(
                        update={"optionality": Optionality.OPTIONAL}
                    )
                return self._of_values(
                    values, {Identifier("choices"): choices}, role, expanding
                )
            case Err():
                return Err(UnimportableChoicesError(enum_class))

    def _of_values(
        self,
        values: _ChoiceValues,
        arguments: Mapping[Identifier, ValueExpression | Symbol],
        role: FieldRole,
        expanding: frozenset[RuntimeAnnotation],
    ) -> Result[_TypedField, FieldRuleError]:
        """The field of the type every value has, with `arguments`."""
        match tuple(
            sorted(
                {type(value) for value in values.stored},
                key=lambda value_type: value_type.__qualname__,
            )
        ):
            case (value_type,):
                match self._field_of(value_type, role, expanding):
                    case Ok(typed):
                        return Ok(
                            _TypedField(
                                field=typed.field.with_arguments(arguments),
                                optionality=typed.optionality,
                            ).made_optional_by(values.optionality)
                        )
                    case Err() as error:
                        return error
            case ():
                return Err(NoFieldRuleError(values.values_of))
            case value_types:
                return Err(MixedValueTypesError(values.values_of, value_types))

    @staticmethod
    def _is_hashable(annotation: RuntimeAnnotation) -> bool:
        """Whether `annotation` can be looked up in a rule mapping: an `Annotated`
        type with unhashable metadata cannot."""
        try:
            _ = hash(annotation)
        except TypeError:
            return False
        return True


class _ChoiceValues(pydantic.BaseModel, frozen=True):
    """The values an enum's or a `Literal`'s column stores, and whether it may also
    be NULL, e.g. for `Literal["a", None]`."""

    values_of: RuntimeAnnotation
    stored: tuple[StoredValue, ...]
    optionality: Optionality

    @staticmethod
    def stored_of(value: StoredValue) -> StoredValue:
        """What a column stores for `value`: an enum member's value, else
        `value`."""
        if isinstance(value, enum.Enum):
            # Type checkers type `Enum.value` as an implicit `Any`.
            stored: StoredValue = operator.attrgetter("value")(value)
            return stored
        return value

    def listed_choices(self) -> Result[ValueExpression, UnwritableChoicesError]:
        """The values as Django choices, each its own label, written as their
        `repr`. Only a boolean, integer, finite float, string or bytes is written
        that way."""
        unwritable = tuple(
            value
            for value in self.stored
            if not (
                type(value) in {bool, int, str, bytes}
                or (
                    isinstance(value, float)
                    and type(value) is float
                    and math.isfinite(value)
                )
            )
        )
        if unwritable:
            return Err(UnwritableChoicesError(self.values_of, unwritable))
        return Ok(ValueExpression(repr([(value, value) for value in self.stored])))


class _TypedField(pydantic.BaseModel, frozen=True):
    field: DjangoField
    optionality: Optionality

    def made_optional_by(self, optionality: Optionality) -> _TypedField:
        """This field, made optional by an `optionality` it is wrapped in."""
        match optionality:
            case Optionality.OPTIONAL:
                return self.model_copy(update={"optionality": optionality})
            case Optionality.REQUIRED:
                return self

    def with_null(self) -> DjangoField:
        """The field, with `null=True` when it is optional."""
        match self.optionality:
            case Optionality.OPTIONAL:
                return self.field.with_arguments(
                    {Identifier("null"): ValueExpression("True")}
                )
            case Optionality.REQUIRED:
                return self.field


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
        return FileText(
            f"    {self.name.root} = {self.field.to_expression(self.positional).root}\n"
        )


class _PlannedModel(pydantic.BaseModel, frozen=True):
    """`manager` is the class of its manager `objects`."""

    entity: type[pydantic.BaseModel]
    name: Identifier
    declarations: tuple[_Declaration, ...]
    manager: Symbol


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
        reported.extend(DjangoModelsRenderer._existing_misconfigured(graph, config))
        for key, on_delete in config.on_delete.items():
            match references.get(key):
                case None:
                    reported.append(
                        OnDeleteForNonReference(entity=key.entity, field=key.field)
                    )
                case Reference() if not config.generates(key.entity):
                    reported.append(
                        OnDeleteOnExistingModel(entity=key.entity, field=key.field)
                    )
                case Reference(optionality=Optionality.REQUIRED) if (
                    on_delete == OnDelete.SET_NULL
                ):
                    reported.append(
                        SetNullOnRequiredReference(entity=key.entity, field=key.field)
                    )
                case Reference():
                    pass
        for model, node in sorted(
            graph.entities.items(), key=lambda item: item[0].__name__
        ):
            if not config.generates(model):
                # Its fields are unknown, but reverse accessors may still clash.
                reported.extend(
                    DjangoModelsRenderer._accessor_clashes(graph, config, model, ())
                )
                continue
            declarations: list[_Declaration] = []
            for declared in DjangoModelsRenderer._declarations(
                model, node, types, config
            ):
                match declared:
                    case _Declaration():
                        declarations.append(declared)
                    case (
                        NoDjangoFieldRule()
                        | UnimportableDjangoChoices()
                        | MixedDjangoChoiceTypes()
                        | UnwritableDjangoChoices()
                        | AutoIncrementOnNonPrimaryKey()
                    ):
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
                )
            )
        imports = frozenset(
            {
                *(
                    symbol
                    for model in planned
                    for declaration in model.declarations
                    for symbol in declaration.field.imports()
                ),
                *(model.manager for model in planned),
                *(
                    config.model_of(reference.target)
                    for model in planned
                    for reference in graph.entities[model.entity].references
                    if not config.generates(reference.target)
                ),
            }
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
    def _declarations(
        model: type[pydantic.BaseModel],
        node: EntityNode,
        types: FieldTypes,
        config: DjangoModels,
    ) -> Iterator[_Declaration | _DeclarationDiagnostic]:
        references = {reference.field: reference for reference in node.references}
        for name, info in model.model_fields.items():
            field = FieldName(name)
            match references.get(field):
                case Reference() as reference:
                    yield DjangoModelsRenderer._foreign_key(model, reference, config)
                    continue
                case None:
                    pass
            role = (
                FieldRole.PRIMARY_KEY
                if field == EntityNode.primary_key
                else FieldRole.PLAIN
            )
            match types.field_of(info.rebuild_annotation(), role):
                case Ok(typed):
                    pass
                case Err(error):
                    match error:
                        case NoFieldRuleError():
                            yield NoDjangoFieldRule(
                                entity=model,
                                field=field,
                                annotation=AnnotationText.of(info.annotation),
                                unruled=AnnotationText.of(error.annotation),
                            )
                        case UnwritableChoicesError():
                            yield UnwritableDjangoChoices(
                                entity=model,
                                field=field,
                                values_of=AnnotationText.of(error.values_of),
                                values=error.values,
                            )
                        case UnimportableChoicesError():
                            yield UnimportableDjangoChoices(
                                entity=model, field=field, enum_class=error.enum_class
                            )
                        case MixedValueTypesError():
                            yield MixedDjangoChoiceTypes(
                                entity=model,
                                field=field,
                                values_of=AnnotationText.of(error.annotation),
                                value_types=error.value_types,
                            )
                    continue
            if any(
                nested.is_auto_increment() for nested in typed.field.nested_fields()
            ) or (role == FieldRole.PLAIN and typed.field.is_auto_increment()):
                yield AutoIncrementOnNonPrimaryKey(entity=model, field=field)
                continue
            django_field = typed.with_null()
            if role == FieldRole.PRIMARY_KEY:
                django_field = django_field.with_arguments(
                    {Identifier("primary_key"): ValueExpression("True")}
                )
            yield _Declaration(
                name=field,
                field=django_field,
                positional=(),
                source=field,
                role=role,
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
    def reverse_accessor(
        entity: type[pydantic.BaseModel], reference: FieldName
    ) -> FieldName:
        """The attribute of the referenced Entity's Django model holding the
        `entity` whose one-to-one Reference `reference` points to it, e.g.
        `company_profile_by_company` for `CompanyProfile.company_id`."""
        snake = model_names.PascalCase(entity.__name__).snake_case().root
        foreign_key = DjangoModelsRenderer.foreign_key_name(reference)
        return FieldName(f"{snake}_by_{foreign_key.root}")

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
        annotation = node.model.model_fields[field.root].rebuild_annotation()
        match field_types.field_of(annotation, FieldRole.PLAIN):
            case Ok(typed):
                return typed.optionality
            case Err():
                return Optionality.OPTIONAL

    @staticmethod
    def _existing_misconfigured(
        graph: EntityGraph, config: DjangoModels
    ) -> Iterator[
        ExistingModelOfNonEntity
        | ExistingModelInGeneratedModule
        | UndeclaredReverseAccessor
        | ReverseAccessorOfNonOneToOne
    ]:
        """Existing Django models named for a non-Entity, or in a generated
        module, or declaring other reverse accessors than their Entity's
        one-to-one References have."""
        for entity, existing in config.existing.items():
            if existing.module in (config.module, config.queries):
                yield ExistingModelInGeneratedModule(
                    entity=entity, module=existing.module
                )
            match graph.entities.get(entity):
                case None:
                    yield ExistingModelOfNonEntity(model=entity)
                    continue
                case EntityNode() as node:
                    pass
            one_to_one = [
                reference.field
                for reference in node.references
                if reference.cardinality == Cardinality.ONE_TO_ONE
            ]
            yield from (
                UndeclaredReverseAccessor(entity=entity, field=field)
                for field in one_to_one
                if field not in existing.reverse_accessors
            )
            yield from (
                ReverseAccessorOfNonOneToOne(entity=entity, field=field)
                for field in existing.reverse_accessors
                if field not in one_to_one
            )

    @staticmethod
    def _accessor_clashes(
        graph: EntityGraph,
        config: DjangoModels,
        model: type[pydantic.BaseModel],
        declarations: Iterable[_Declaration],
    ) -> Iterator[ReverseAccessorClash | InvalidReverseAccessorName]:
        """Reverse accessors on `model`'s Django model named like another one, or
        like a name the model binds or reserves, or, for one frp generates, named
        as Django forbids. An existing model declares its own. References are in
        Entity name and field order."""
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
                if (
                    reference.target is model
                    and reference.cardinality == Cardinality.ONE_TO_ONE
                ):
                    by_name.setdefault(
                        config.reverse_accessor_of(entity, reference.field), []
                    ).append(ReferenceKey(entity=entity, field=reference.field))
        for name, references in by_name.items():
            if len(references) > 1 or name in bound:
                yield ReverseAccessorClash(
                    entity=model, name=name, references=tuple(references)
                )
            # It is an identifier, made of an Entity's and a field's name, and no
            # keyword, since it holds `_by_`.
            if name.root.endswith("_") or "__" in name.root:
                yield from (
                    InvalidReverseAccessorName(name=name, reference=reference)
                    for reference in references
                    if config.generates(reference.entity)
                )

    @staticmethod
    def _foreign_key(
        model: type[pydantic.BaseModel],
        reference: Reference,
        config: DjangoModels,
    ) -> _Declaration:
        """A foreign key to the referenced Entity's model, named without `_id`: a
        `OneToOneField` with a reverse accessor for a one-to-one Reference, else a
        `ForeignKey` without one (`related_name='+'`). A generated model is named
        by a string, since it may be defined further down, an existing one by
        its imported class."""
        bound = config.model_of(reference.target).bound_name().root
        target = ValueExpression(
            repr(bound) if config.generates(reference.target) else bound
        )
        deleted = config.on_delete.get(
            ReferenceKey(entity=model, field=reference.field), OnDelete.PROTECT
        )
        match reference.cardinality:
            case Cardinality.MANY_TO_ONE:
                field_class = Identifier("ForeignKey")
                related_name = "+"
            case Cardinality.ONE_TO_ONE:
                field_class = Identifier("OneToOneField")
                related_name = config.reverse_accessor_of(model, reference.field).root
        arguments = {
            Identifier("on_delete"): ValueExpression(f"models.{deleted.value}"),
            Identifier("related_name"): ValueExpression(repr(related_name)),
        }
        if reference.optionality == Optionality.OPTIONAL:
            arguments[Identifier("null")] = ValueExpression("True")
        return _Declaration(
            name=DjangoModelsRenderer.foreign_key_name(reference.field),
            field=DjangoField.builtin(field_class, arguments),
            positional=(target,),
            source=reference.field,
            role=FieldRole.FOREIGN_KEY,
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
        """`imports` are the field classes, Managers and existing Django models
        the module imports by name."""
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
        classes = "\n\n".join(
            f"class {model.name.root}(models.Model):\n"
            f"    {DjangoModelsRenderer.manager.root} = "
            f"{model.manager.bound_name().root}()\n"
            + "".join(declaration.line().root for declaration in model.declarations)
            + "\n    class Meta:\n"
            + f"        app_label = {app_label.root!r}\n"
            for model in planned
        )
        return FileText(
            f"{DjangoModelsRenderer.header.root}\n{imported}\n\n\n{classes}"
        )
