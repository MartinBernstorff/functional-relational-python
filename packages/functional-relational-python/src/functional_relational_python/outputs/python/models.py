"""Renders the planned models: one module of Pydantic models, and of the `type`
aliases of Union models, per module.

Everything here is pure: it builds file contents and touches no filesystem.
`generated_files` writes and checks them.
"""

import typing
from collections.abc import Sequence

import pydantic

from functional_relational_python.outputs.python.plan import (
    Inherited,
    ModelField,
    ModelPlan,
    PythonPlan,
    UnionPlan,
)
from functional_relational_python.outputs.python.symbols import (
    ImportBlock,
    LineLength,
    ModulePath,
    PythonString,
    TypeExpression,
    ValueExpression,
)
from functional_relational_python.pipeline import ir
from functional_relational_python.support.filesystem import FileText
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)


class FieldDeclaration(pydantic.RootModel[str], frozen=True):
    """One field of a model's class body, e.g. `name: Text | None = None`."""


class AliasStatement(pydantic.RootModel[str], frozen=True):
    """A `type` statement declaring a Union model, as ruff formats it."""


class ModelsRenderer:
    """Writes the source of every planned model."""

    header: typing.ClassVar[FileText] = FileText(
        f"{GeneratedFiles.header.root}"
        "# Regenerate with `frp generate`; `frp check` fails when this is stale.\n"
    )

    @staticmethod
    def files(plan: PythonPlan) -> tuple[GeneratedFile, ...]:
        """One file per module, in module order.

        `render.PythonRenderer` writes them together with the mapping functions.
        """
        return tuple(
            GeneratedFile(
                path=module.file_path(),
                text=ModelsRenderer._module(
                    module,
                    tuple(
                        model
                        for view in plan.models.get(module, ())
                        for model in view.with_nested()
                    ),
                    plan.unions.get(module, ()),
                ),
            )
            for module in plan.model_modules()
        )

    @staticmethod
    def _module(
        module: ModulePath,
        models: Sequence[ModelPlan],
        unions: Sequence[UnionPlan],
    ) -> FileText:
        """The classes of `models`, then the `type` statements of `unions`."""
        imports = (
            ImportBlock(
                modules=frozenset(),
                symbols=frozenset(
                    symbol
                    for symbol in (
                        *(symbol for model in models for symbol in model.uses),
                        *(symbol for union in unions for symbol in union.uses()),
                    )
                    if symbol.module.root != "builtins" and symbol.module != module
                ),
            )
            .text()
            .root
        )

        classes = "\n\n".join(
            f"class {model.symbol.bound_name().root}({model.base.root}, frozen=True):\n"
            + (
                "".join(
                    f"    {declaration.root}\n"
                    for field in model.fields
                    for declaration in ModelsRenderer._declarations(field)
                )
                or "    pass\n"
            )
            for model in models
        )
        aliases = "".join(f"{ModelsRenderer._alias(union).root}\n" for union in unions)
        body = "\n\n".join(part for part in (classes, aliases) if part)
        return FileText(f"{ModelsRenderer.header.root}\n{imports}\n\n{body}")

    @staticmethod
    def _alias(union: UnionPlan) -> AliasStatement:
        """`type <Union> = Annotated[<A> | <B>, Discriminator("<field>")]`, on one
        line if it fits, else split inside the brackets with a trailing comma,
        and the union split at its `|`s if it does not fit on a line either."""
        variants = [variant.reference().root for variant in union.variants]
        head = (
            f"type {union.symbol.bound_name().root} = "
            f"{UnionPlan.annotated.reference().root}["
        )
        discriminator = (
            f"{UnionPlan.discriminator_class.reference().root}("
            f"{PythonString(union.discriminator.root).literal().root})"
        )
        line = f"{head}{' | '.join(variants)}, {discriminator}]"
        if len(line) <= LineLength.default().root:
            return AliasStatement(line)
        joined = f"    {' | '.join(variants)},"
        if len(joined) <= LineLength.default().root:
            alternatives = f"{joined}\n"
        else:
            first, *rest = variants
            alternatives = f"    {first}\n" + "".join(
                f"    | {variant}{',' if index == len(rest) - 1 else ''}\n"
                for index, variant in enumerate(rest)
            )
        return AliasStatement(f"{head}\n{alternatives}    {discriminator},\n]")

    @staticmethod
    def _declarations(field: ModelField) -> tuple[FieldDeclaration, ...]:
        """The line declaring `field`, or none when the model inherits it."""
        match field.type:
            case TypeExpression() as type_expression:
                annotated = f"{field.field.name.root}: {type_expression.root}"
            case Inherited():
                return ()
        match field.default:
            case ValueExpression() as default:
                return (FieldDeclaration(f"{annotated} = {default.root}"),)
            case ir.NoDefault():
                return (FieldDeclaration(annotated),)
