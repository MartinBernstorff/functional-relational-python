"""Renders a Django model per Entity, so a Django project can store Entities.

Each model's manager `objects` is its Entity's Manager, which `queries`
generates with a method per View over the Entity.

A one-to-one Reference is a `OneToOneField` with a reverse accessor on the
referenced Entity's model, `<entity>_by_<foreign key>`, e.g.
`company_profile_by_company` for `CompanyProfile.company_id`, so that a query can
follow it backwards. A many-to-one Reference has no reverse accessor, so several
never clash.

Everything here is pure: it builds file contents and touches no filesystem, and
it never imports Django. Migrations are the user's to make.
"""

import datetime
import enum
import keyword
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

    def with_arguments(
        self, arguments: Mapping[Identifier, ValueExpression | Symbol | DjangoField]
    ) -> DjangoField:
        """This field with `arguments` added, after its own."""
        return self.model_copy(update={"arguments": {**self.arguments, **arguments}})

    def call(self, positional: Iterable[ValueExpression]) -> ValueExpression:
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
                return argument.call(())

    @staticmethod
    def _referred(symbol: Symbol) -> TypeExpression:
        """`symbol`, where the generated module refers to it."""
        if symbol.module == DjangoField.models_module:
            return TypeExpression(f"models.{symbol.reference().root}")
        return symbol.reference()

    @staticmethod
    def _imported(symbol: Symbol) -> Iterator[Symbol]:
        """What the generated module imports to refer to `symbol`: nothing for
        `django.db.models`, which it imports as `models`."""
        if symbol.module != DjangoField.models_module:
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


class DjangoModels(pydantic.BaseModel, frozen=True):
    """Generate a Django model per Entity, all into `module` with `app_label`, and
    the Django queries for Views into `queries`.

    `field_types` overrides the built-in rule for a type, see `FieldTypes`.
    `on_delete` sets what deleting the referenced Entity does, per Reference; a
    Reference not in it is `PROTECT`. `queries` holds a QuerySet and a Manager per
    Entity, which its model's manager is, see `DjangoQueriesRenderer`.
    """

    module: ModulePath
    app_label: AppLabel
    field_types: Mapping[RuntimeAnnotation, DjangoField]
    on_delete: Mapping[ReferenceKey, OnDelete]
    queries: ModulePath

    @staticmethod
    def fake() -> DjangoModels:
        return DjangoModels(
            module=ModulePath("app.models"),
            app_label=AppLabel.fake(),
            field_types={},
            on_delete={},
            queries=ModulePath("app.queries"),
        )

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


class UnimportableDjangoChoices(pydantic.BaseModel, Diagnostic, frozen=True):
    """An Entity field's type is, or wraps, a Django `Choices` enum the generated
    module cannot import, e.g. one defined in a function, so its field cannot
    refer to the enum's `choices`."""

    entity: type[pydantic.BaseModel]
    field: FieldName
    choices: type[enum.Enum]

    def describe(self) -> DiagnosticMessage:
        name = self.choices.__qualname__
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: the Django field's "
            f"choices are '{name}.choices', but '{name}' cannot be imported by its "
            "module and name. Define it at module level, or add a rule to "
            "DjangoModels.field_types."
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
    auto-incrementing Django field, e.g. an override keyed on `int`."""

    entity: type[pydantic.BaseModel]
    field: FieldName

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.entity.__qualname__}.{self.field.root}: only a primary key can "
            "auto-increment. Key the override on the Entity's ID type instead."
        )


type _DeclarationDiagnostic = (
    NoDjangoFieldRule | UnimportableDjangoChoices | AutoIncrementOnNonPrimaryKey
)


type DjangoDiagnostic = (
    NoDjangoFieldRule
    | UnimportableDjangoChoices
    | InvalidDjangoFieldName
    | AutoIncrementOnNonPrimaryKey
    | OnDeleteForNonReference
    | SetNullOnRequiredReference
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


type StoredValue = object
"""A value a column stores, e.g. `'meta'` for the enum member `Channel.META`."""


class UnimportableChoicesError(Exception):
    def __init__(self, choices: type[enum.Enum]) -> None:
        self.choices = choices
        super().__init__(f"{choices!r} cannot be imported by its module and name")


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
      imports the enum. Any other enum's are its values, each its own label.
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
    ) -> Result[_TypedField, NoFieldRuleError | UnimportableChoicesError]:
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
            if isinstance(current, typing.TypeAliasType):
                current = current.__value__
                continue
            if typing.get_origin(current) is typing.Annotated:
                current = typing.get_args(current)[0]
                continue
            if isinstance(current, type) and issubclass(current, pydantic.RootModel):
                current = current.model_fields["root"].rebuild_annotation()
                continue
            match self._builtin_of(current, role):
                case Ok(typed):
                    return Ok(typed.or_optional(optionality))
                case Err() as error:
                    return error

    def _builtin_of(
        self, annotation: RuntimeAnnotation, role: FieldRole
    ) -> Result[_TypedField, NoFieldRuleError | UnimportableChoicesError]:
        """The built-in rule for `annotation`, which no override names."""
        if annotation in FieldTypes.builtin and not (
            role == FieldRole.PRIMARY_KEY and annotation is str
        ):
            return Ok(
                _TypedField(
                    field=FieldTypes.builtin[annotation],
                    optionality=Optionality.REQUIRED,
                )
            )
        if typing.get_origin(annotation) is typing.Literal:
            values = typing.get_args(annotation)
            stored = tuple(
                FieldTypes._stored(value) for value in values if value is not None
            )
            return self._with_choices(
                annotation,
                stored,
                FieldTypes._listed_choices(stored),
                role,
                Optionality.OPTIONAL if None in values else Optionality.REQUIRED,
            )
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return self._enum_of(annotation, role)
        if typing.get_origin(annotation) is list:
            (item,) = typing.get_args(annotation)
            match self.field_of(item, FieldRole.PLAIN):
                case Ok(typed):
                    return Ok(
                        _TypedField(
                            field=DjangoField(
                                field_class=FieldTypes.array_field,
                                arguments={Identifier("base_field"): typed.declared()},
                            ),
                            optionality=Optionality.REQUIRED,
                        )
                    )
                case Err() as error:
                    return error
        return Err(NoFieldRuleError(annotation))

    def _enum_of(
        self, enum_class: type[enum.Enum], role: FieldRole
    ) -> Result[_TypedField, NoFieldRuleError | UnimportableChoicesError]:
        stored = tuple(FieldTypes._stored(member) for member in enum_class)
        if not any(
            f"{base.__module__}.{base.__qualname__}"
            == FieldTypes.django_choices.describe().root
            for base in enum_class.__mro__
        ):
            return self._with_choices(
                enum_class,
                stored,
                FieldTypes._listed_choices(stored),
                role,
                Optionality.REQUIRED,
            )
        match Symbol.of(enum_class):
            case Ok(symbol):
                choices = Symbol(
                    module=symbol.module,
                    qualname=(*symbol.qualname, Identifier("choices")),
                )
                return self._with_choices(
                    enum_class, stored, choices, role, Optionality.REQUIRED
                )
            case Err():
                return Err(UnimportableChoicesError(enum_class))

    def _with_choices(
        self,
        annotation: RuntimeAnnotation,
        stored: tuple[StoredValue, ...],
        choices: ValueExpression | Symbol,
        role: FieldRole,
        optionality: Optionality,
    ) -> Result[_TypedField, NoFieldRuleError | UnimportableChoicesError]:
        """The field of the type every `stored` value has, with `choices`, made
        optional by `optionality`."""
        match tuple({type(value) for value in stored}):
            case (value_type,):
                match self.field_of(value_type, role):
                    case Ok(typed):
                        return Ok(
                            _TypedField(
                                field=typed.field.with_arguments(
                                    {Identifier("choices"): choices}
                                ),
                                optionality=typed.optionality,
                            ).or_optional(optionality)
                        )
                    case Err() as error:
                        return error
            case _:
                return Err(NoFieldRuleError(annotation))

    @staticmethod
    def _stored(value: StoredValue) -> StoredValue:
        """What a column stores for `value`: an enum member's value, else
        `value`."""
        if isinstance(value, enum.Enum):
            # Type checkers type `Enum.value` as an implicit `Any`.
            stored: StoredValue = operator.attrgetter("value")(value)
            return stored
        return value

    @staticmethod
    def _listed_choices(stored: tuple[StoredValue, ...]) -> ValueExpression:
        """`stored` as Django choices, each value its own label."""
        return ValueExpression(repr([(value, value) for value in stored]))


class _TypedField(pydantic.BaseModel, frozen=True):
    field: DjangoField
    optionality: Optionality

    def or_optional(self, optionality: Optionality) -> _TypedField:
        """This field, made optional by an `optionality` it is wrapped in."""
        match optionality:
            case Optionality.OPTIONAL:
                return self.model_copy(update={"optionality": optionality})
            case Optionality.REQUIRED:
                return self

    def declared(self) -> DjangoField:
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
            f"    {self.name.root} = {self.field.call(self.positional).root}\n"
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
        for model, node in sorted(
            graph.entities.items(), key=lambda item: item[0].__name__
        ):
            declarations: list[_Declaration] = []
            for declared in DjangoModelsRenderer._declarations(
                model, node, types, config.on_delete
            ):
                match declared:
                    case _Declaration():
                        declarations.append(declared)
                    case (
                        NoDjangoFieldRule()
                        | UnimportableDjangoChoices()
                        | AutoIncrementOnNonPrimaryKey()
                    ):
                        reported.append(declared)
            reported.extend(DjangoModelsRenderer._field_clashes(model, declarations))
            reported.extend(
                DjangoModelsRenderer._accessor_clashes(graph, model, declarations)
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
        on_delete: Mapping[ReferenceKey, OnDelete],
    ) -> Iterator[_Declaration | _DeclarationDiagnostic]:
        references = {reference.field: reference for reference in node.references}
        for name, info in model.model_fields.items():
            field = FieldName(name)
            match references.get(field):
                case Reference() as reference:
                    yield DjangoModelsRenderer._foreign_key(model, reference, on_delete)
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
                            )
                        case UnimportableChoicesError():
                            yield UnimportableDjangoChoices(
                                entity=model, field=field, choices=error.choices
                            )
                    continue
            if role == FieldRole.PLAIN and typed.field.is_auto_increment():
                yield AutoIncrementOnNonPrimaryKey(entity=model, field=field)
                continue
            declared = typed.declared()
            if role == FieldRole.PRIMARY_KEY:
                declared = declared.with_arguments(
                    {Identifier("primary_key"): ValueExpression("True")}
                )
            yield _Declaration(
                name=field,
                field=declared,
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
    def _accessor_clashes(
        graph: EntityGraph,
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
                if (
                    reference.target is model
                    and reference.cardinality == Cardinality.ONE_TO_ONE
                ):
                    by_name.setdefault(
                        DjangoModelsRenderer.reverse_accessor(entity, reference.field),
                        [],
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
                )

    @staticmethod
    def _foreign_key(
        model: type[pydantic.BaseModel],
        reference: Reference,
        on_delete: Mapping[ReferenceKey, OnDelete],
    ) -> _Declaration:
        """A foreign key to the referenced Entity's model, named without `_id`: a
        `OneToOneField` with a reverse accessor for a one-to-one Reference, else a
        `ForeignKey` without one (`related_name='+'`)."""
        deleted = on_delete.get(
            ReferenceKey(entity=model, field=reference.field), OnDelete.PROTECT
        )
        match reference.cardinality:
            case Cardinality.MANY_TO_ONE:
                field_class = Identifier("ForeignKey")
                related_name = "+"
            case Cardinality.ONE_TO_ONE:
                field_class = Identifier("OneToOneField")
                related_name = DjangoModelsRenderer.reverse_accessor(
                    model, reference.field
                ).root
        arguments = {
            Identifier("on_delete"): ValueExpression(f"models.{deleted.value}"),
            Identifier("related_name"): ValueExpression(repr(related_name)),
        }
        if reference.optionality == Optionality.OPTIONAL:
            arguments[Identifier("null")] = ValueExpression("True")
        return _Declaration(
            name=DjangoModelsRenderer.foreign_key_name(reference.field),
            field=DjangoField.builtin(field_class, arguments),
            positional=(ValueExpression(repr(reference.target.__name__)),),
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
        """`imports` are the field classes and Managers the module imports by
        name."""
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
