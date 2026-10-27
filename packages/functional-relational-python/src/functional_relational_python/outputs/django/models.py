"""Renders a Django model per Entity, so a Django project can store Entities.

Each model's manager `objects` is its Entity's Manager, which `queries`
generates with a method per View over the Entity.

A one-to-one Reference is a `OneToOneField` with a reverse accessor on the
referenced Entity's model, `<entity>_by_<foreign key>`, e.g.
`company_profile_by_company` for `CompanyProfile.company_id`, so that a query can
follow it backwards. A many-to-one Reference has no reverse accessor, so several
never clash.

An Entity's constraints, see `pipeline.constraints`, are in its model's
`Meta.constraints`, with a check per numeric bound its fields' types set, e.g.
`campaign_lifetime_budget_gt` for `lifetime_budget: PositiveInt`. A Condition is
written with the semantics it has in `where`, see `conditions`; so a row violates
a Check exactly where the Check is false in Python.

Everything here is pure: it builds file contents and touches no filesystem, and
it never imports Django. Migrations are the user's to make.
"""

import datetime
import enum
import keyword
import math
import typing
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence

import annotated_types
import pydantic
from pydantic.fields import FieldInfo
from safe_result import Err, Ok, Result

from functional_relational_python.compiler import (
    condition_checker,
    diagnostics,
    model_names,
)
from functional_relational_python.entity_graph.entity_graph import (
    Cardinality,
    EntityGraph,
    EntityNode,
    FieldName,
    Optionality,
    Reference,
    RuntimeAnnotation,
)
from functional_relational_python.outputs.django import conditions
from functional_relational_python.outputs.python.mappings import Expression, Indent
from functional_relational_python.outputs.python.symbols import (
    AnnotationText,
    LineLength,
    ModulePath,
    Symbol,
    TypeExpression,
    ValueExpression,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.pipeline.constraints import (
    Check,
    Constraint,
    ConstraintName,
    EveryRow,
    Unique,
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

    def reference(self) -> TypeExpression:
        """The field class, where the generated module refers to it."""
        if self.is_builtin():
            return TypeExpression(f"models.{self.field_class.reference().root}")
        return self.field_class.reference()


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
    `constraints` are each Entity's constraints, which its model's `Meta` lists
    with those its field types imply, see `DjangoModelsRenderer`.
    """

    module: ModulePath
    app_label: AppLabel
    field_types: Mapping[RuntimeAnnotation, DjangoField]
    on_delete: Mapping[ReferenceKey, OnDelete]
    queries: ModulePath
    constraints: Mapping[type[pydantic.BaseModel], Sequence[Constraint]]

    @staticmethod
    def fake() -> DjangoModels:
        return DjangoModels(
            module=ModulePath("app.models"),
            app_label=AppLabel.fake(),
            field_types={},
            on_delete={},
            queries=ModulePath("app.queries"),
            constraints={},
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


class ConstraintsForNonEntity(pydantic.BaseModel, Diagnostic, frozen=True):
    """`DjangoModels.constraints` has constraints for a model that is not an
    Entity, so no Django model holds them."""

    model: type[pydantic.BaseModel]

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.model.__qualname__}: DjangoModels.constraints has constraints "
            "for it, but it is not an Entity."
        )


class ConstraintNameClash(pydantic.BaseModel, Diagnostic, frozen=True):
    """Several constraints, declared or implied by a field's type, have one name.
    The database needs each name once. `entities` have those constraints."""

    name: ConstraintName
    entities: tuple[type[pydantic.BaseModel], ...]

    def describe(self) -> DiagnosticMessage:
        entities = ", ".join(entity.__qualname__ for entity in self.entities)
        return DiagnosticMessage(
            f"Django models: several constraints are named '{self.name.root}', "
            f"of {entities}."
        )


class UnimportableConstraintEnum(pydantic.BaseModel, Diagnostic, frozen=True):
    """A constraint's Condition compares with a member of an enum the Django
    models module cannot import by its module and name."""

    constraint: ConstraintName
    enum_class: AnnotationText

    def describe(self) -> DiagnosticMessage:
        return DiagnosticMessage(
            f"{self.constraint.root}: the Django models cannot import "
            f"{self.enum_class.root}, which its condition compares with."
        )


type _DeclarationDiagnostic = NoDjangoFieldRule | AutoIncrementOnNonPrimaryKey


type _ConstraintDiagnostic = (
    diagnostics.CompileDiagnostic
    | ConstraintsForNonEntity
    | ConstraintNameClash
    | UnimportableConstraintEnum
)


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
    | _ConstraintDiagnostic
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


class _CheckedCheck(pydantic.BaseModel, frozen=True):
    """A Check whose Condition holds up against its Entity."""

    name: ConstraintName
    kept: ir.Filter


class _CheckedUnique(pydantic.BaseModel, frozen=True):
    """A Unique over the Django model fields `fields`, whose Condition holds up
    against its Entity."""

    name: ConstraintName
    fields: tuple[FieldName, ...]
    where: ir.Filter | EveryRow


class _ImpliedBound(pydantic.BaseModel, frozen=True):
    """A bound a field's type sets, e.g. `Gt(0)`: the field, where not NULL,
    compares with `bound` by `operator`."""

    name: ConstraintName
    field: ir.SourceField
    operator: ir.ComparisonOperator
    bound: ValueExpression


type _PlannedConstraint = _CheckedCheck | _CheckedUnique | _ImpliedBound


class _ModelConstraints(pydantic.BaseModel, frozen=True):
    """An Entity's constraints, and why some of those declared are defective."""

    planned: tuple[_PlannedConstraint, ...]
    diagnostics: tuple[_ConstraintDiagnostic, ...]


class _ImportedEnum(pydantic.BaseModel, frozen=True):
    """An enum a constraint refers to, and how the module imports it."""

    enum_class: type[enum.Enum]
    symbol: Symbol


class _PlannedModel(pydantic.BaseModel, frozen=True):
    """`manager` is the class of its manager `objects`."""

    entity: type[pydantic.BaseModel]
    name: Identifier
    declarations: tuple[_Declaration, ...]
    manager: Symbol
    constraints: tuple[_PlannedConstraint, ...]


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
            ConstraintsForNonEntity(model=model)
            for model in config.constraints
            if model not in graph.entities
        )
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
                    case NoDjangoFieldRule() | AutoIncrementOnNonPrimaryKey():
                        reported.append(declared)
            reported.extend(DjangoModelsRenderer._field_clashes(model, declarations))
            reported.extend(
                DjangoModelsRenderer._accessor_clashes(graph, model, declarations)
            )
            constraints = DjangoModelsRenderer._constraints(
                model, node, config.constraints.get(model, ())
            )
            reported.extend(constraints.diagnostics)
            planned.append(
                _PlannedModel(
                    entity=model,
                    name=Identifier(model.__name__),
                    declarations=tuple(declarations),
                    manager=config.manager_of(model),
                    constraints=constraints.planned,
                )
            )
        reported.extend(DjangoModelsRenderer._constraint_name_clashes(planned))
        enums: dict[type[enum.Enum], Symbol] = {}
        for imported in DjangoModelsRenderer._constraint_enums(planned):
            match imported:
                case _ImportedEnum(enum_class=enum_class, symbol=symbol):
                    enums[enum_class] = symbol
                case UnimportableConstraintEnum():
                    reported.append(imported)
        imports = frozenset(
            {
                *(
                    declaration.field.field_class.imported()
                    for model in planned
                    for declaration in model.declarations
                    if not declaration.field.is_builtin()
                ),
                *(model.manager for model in planned),
                *(symbol.imported() for symbol in enums.values()),
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
                            config.app_label,
                            planned,
                            imports,
                            conditions.ConditionWriter(
                                columns=DjangoModelsRenderer.columns(graph, types),
                                models=Expression(
                                    DjangoModelsRenderer.models_import.root
                                ),
                                enums={
                                    used: symbol.reference()
                                    for used, symbol in enums.items()
                                },
                            ),
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
            match types.field_of(info.annotation, role):
                case Ok(typed):
                    pass
                case Err():
                    yield NoDjangoFieldRule(
                        entity=model,
                        field=field,
                        annotation=AnnotationText.of(info.annotation),
                    )
                    continue
            if role == FieldRole.PLAIN and typed.field.is_auto_increment():
                yield AutoIncrementOnNonPrimaryKey(entity=model, field=field)
                continue
            arguments = dict(typed.field.arguments)
            if typed.optionality == Optionality.OPTIONAL:
                arguments[Identifier("null")] = ValueExpression("True")
            if role == FieldRole.PRIMARY_KEY:
                arguments[Identifier("primary_key")] = ValueExpression("True")
            yield _Declaration(
                name=field,
                field=typed.field.model_copy(update={"arguments": arguments}),
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
        annotation = node.model.model_fields[field.root].annotation
        match field_types.field_of(annotation, FieldRole.PLAIN):
            case Ok(typed):
                return typed.optionality
            case Err():
                return Optionality.OPTIONAL

    @staticmethod
    def columns(
        graph: EntityGraph, field_types: FieldTypes
    ) -> conditions.DjangoColumns:
        """The column holding each field of each Entity."""
        return conditions.DjangoColumns(
            of_entity={
                model: {
                    field: conditions.DjangoColumn(
                        attribute=DjangoModelsRenderer.attribute_of(node, field),
                        optionality=DjangoModelsRenderer.column_optionality(
                            node, field, field_types
                        ),
                    )
                    for field in map(FieldName, model.model_fields)
                }
                for model, node in graph.entities.items()
            }
        )

    @staticmethod
    def _constraints(
        model: type[pydantic.BaseModel],
        node: EntityNode,
        declared: Iterable[Constraint],
    ) -> _ModelConstraints:
        """The constraints `declared` for `model`, checked against it, then those
        its fields' types imply."""
        planned: list[_PlannedConstraint] = []
        reported: list[_ConstraintDiagnostic] = []
        relation = ir.Relation.of(ir.FromSource(source=model))
        for constraint in declared:
            checker = condition_checker.ConditionChecker(
                view=constraint.name, relation=relation, subject=model
            )
            match constraint:
                case Check(condition=condition):
                    match checker.check(condition):
                        case condition_checker.Defects(diagnostics=defects):
                            reported.extend(defects)
                        case kept:
                            planned.append(
                                _CheckedCheck(name=constraint.name, kept=kept)
                            )
                case Unique(fields=fields, where=where):
                    unknown = [
                        diagnostics.UnknownField(
                            view=constraint.name,
                            source=relation.source,
                            field=field,
                            valid=relation.field_names(),
                        )
                        for field in fields
                        if field not in relation.field_names()
                    ]
                    match where:
                        case EveryRow():
                            checked: (
                                ir.Filter | EveryRow | condition_checker.Defects
                            ) = where
                        case condition:
                            checked = checker.check(condition)
                    match checked:
                        case condition_checker.Defects(diagnostics=defects):
                            reported.extend((*unknown, *defects))
                        case _ if unknown:
                            reported.extend(unknown)
                        case _:
                            planned.append(
                                _CheckedUnique(
                                    name=constraint.name,
                                    fields=tuple(
                                        DjangoModelsRenderer._django_field_name(
                                            node, field
                                        )
                                        for field in fields
                                    ),
                                    where=checked,
                                )
                            )
        planned.extend(dict.fromkeys(DjangoModelsRenderer._implied_bounds(model, node)))
        return _ModelConstraints(planned=tuple(planned), diagnostics=tuple(reported))

    @staticmethod
    def _implied_bounds(
        model: type[pydantic.BaseModel], node: EntityNode
    ) -> Iterator[_ImpliedBound]:
        """A Check per numeric bound on a field other than the primary key and the
        References, e.g. `Gt(0)` of `PositiveInt`: on the field, inside its `None`,
        or on a RootModel it is or wraps. A bound applies where the field is not
        `None`, as a SQL check passes where the column is NULL.

        A bound that is not a finite `int` or `float`, e.g. a `Decimal` or a
        `datetime`, implies none, nor do other metadata, e.g. `MinLen`.
        """
        snake = model_names.PascalCase(model.__name__).snake_case().root
        references = {reference.field for reference in node.references}
        for name, info in model.model_fields.items():
            field = FieldName(name)
            if field == EntityNode.primary_key or field in references:
                continue
            metadata: list[object] = [*info.metadata]
            current: FieldInfo = info
            annotation = current.annotation
            while True:
                split = Optionality.of_annotation(annotation)
                metadata.extend(split.metadata)
                annotation = split.required
                if not (
                    isinstance(annotation, type)
                    and issubclass(annotation, pydantic.RootModel)
                ):
                    break
                current = annotation.model_fields["root"]
                metadata.extend(current.metadata)
                annotation = current.annotation
            for item in metadata:
                for bound in (
                    item
                    if isinstance(item, annotated_types.GroupedMetadata)
                    else (item,)
                ):
                    match bound:
                        case annotated_types.Gt(gt=value):
                            operator = ir.ComparisonOperator.GT
                        case annotated_types.Ge(ge=value):
                            operator = ir.ComparisonOperator.GE
                        case annotated_types.Lt(lt=value):
                            operator = ir.ComparisonOperator.LT
                        case annotated_types.Le(le=value):
                            operator = ir.ComparisonOperator.LE
                        case _:
                            continue
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int | float)
                        or not math.isfinite(value)
                    ):
                        continue
                    suffix = conditions.ConditionWriter.suffix(operator).root
                    yield _ImpliedBound(
                        name=ConstraintName(f"{snake}_{name}_{suffix}"),
                        field=ir.SourceField(field=field),
                        operator=operator,
                        bound=ValueExpression(repr(value)),
                    )

    @staticmethod
    def _constraint_name_clashes(
        planned: Iterable[_PlannedModel],
    ) -> Iterator[ConstraintNameClash]:
        by_name: dict[ConstraintName, list[type[pydantic.BaseModel]]] = {}
        for model in planned:
            for constraint in model.constraints:
                by_name.setdefault(constraint.name, []).append(model.entity)
        for name, entities in by_name.items():
            if len(entities) > 1:
                yield ConstraintNameClash(
                    name=name, entities=tuple(dict.fromkeys(entities))
                )

    @staticmethod
    def _django_field_name(node: EntityNode, field: FieldName) -> FieldName:
        """The Django model field holding the Entity's `field`: the foreign key
        for a Reference, else the field."""
        if any(reference.field == field for reference in node.references):
            return DjangoModelsRenderer.foreign_key_name(field)
        return field

    @staticmethod
    def _constraint_enums(
        planned: Iterable[_PlannedModel],
    ) -> Iterator[_ImportedEnum | UnimportableConstraintEnum]:
        """The enums the written constraints refer to, each by how the module
        imports it."""
        for model in planned:
            for constraint in model.constraints:
                match constraint:
                    case _CheckedCheck(kept=kept) | _CheckedUnique(where=kept) if (
                        not isinstance(kept, EveryRow)
                    ):
                        used = conditions.ConditionWriter.enums_in(kept)
                    case _CheckedCheck() | _CheckedUnique() | _ImpliedBound():
                        used = iter(())
                for enum_class in used:
                    match Symbol.of(enum_class):
                        case Ok(symbol):
                            yield _ImportedEnum(enum_class=enum_class, symbol=symbol)
                        case Err():
                            yield UnimportableConstraintEnum(
                                constraint=constraint.name,
                                enum_class=AnnotationText.of(enum_class),
                            )

    @staticmethod
    def _constraints_option(
        writer: conditions.ConditionWriter, model: _PlannedModel
    ) -> FileText:
        """The `constraints` line of the model's `Meta`, if it has any."""
        if not model.constraints:
            return FileText("")
        listed = Indent(8).exploded(
            Expression("constraints = ["),
            (
                DjangoModelsRenderer._constraint(writer, model.entity, constraint)
                for constraint in model.constraints
            ),
            Expression("]"),
        )
        return FileText(f"        {listed.root}\n")

    @staticmethod
    def _constraint(
        writer: conditions.ConditionWriter,
        entity: type[pydantic.BaseModel],
        constraint: _PlannedConstraint,
    ) -> Expression:
        """`constraint` as a Django constraint, on a line at the indentation of
        an item of `Meta.constraints`.

        A Check is written as the negation of the rows it excludes: SQL passes a
        check that is NULL, and that is exactly NULL or false where the Check
        holds, so a row violates it exactly where the Check is false in Python.
        """
        models = DjangoModelsRenderer.models_import.root
        indent = Indent(12)
        name = Expression(f"name={constraint.name.root!r}")
        match constraint:
            case _CheckedCheck(kept=kept):
                excluded = writer.excluded(kept, entity)
                condition = writer.written_negation(excluded, indent.deeper())
                return indent.exploded(
                    Expression(f"{models}.CheckConstraint("),
                    (Expression(f"condition={condition.root}"), name),
                    Expression(")"),
                )
            case _ImpliedBound(field=field, operator=operator, bound=bound):
                column = writer.columns.of(entity, field.field).attribute.root
                suffix = conditions.ConditionWriter.suffix(operator).root
                compared = conditions.QLookup(
                    entries=(
                        conditions.QEntry(
                            lookup=Expression(f"{column}__{suffix}"),
                            value=Expression(bound.root),
                        ),
                    )
                )
                condition = writer.written(compared, indent.deeper())
                return indent.exploded(
                    Expression(f"{models}.CheckConstraint("),
                    (Expression(f"condition={condition.root}"), name),
                    Expression(")"),
                )
            case _CheckedUnique(fields=fields, where=where):
                listed = ", ".join(repr(field.root) for field in fields)
                match where:
                    case EveryRow():
                        kept: tuple[Expression, ...] = ()
                    case filtered:
                        written = writer.written(
                            writer.kept(filtered, entity), indent.deeper()
                        )
                        kept = (Expression(f"condition={written.root}"),)
                return indent.exploded(
                    Expression(f"{models}.UniqueConstraint("),
                    (Expression(f"fields=[{listed}]"), *kept, name),
                    Expression(")"),
                )

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
        app_label: AppLabel,
        planned: list[_PlannedModel],
        imports: frozenset[Symbol],
        writer: conditions.ConditionWriter,
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
            + DjangoModelsRenderer._constraints_option(writer, model).root
            for model in planned
        )
        return FileText(
            f"{DjangoModelsRenderer.header.root}\n{imported}\n\n\n{classes}"
        )
