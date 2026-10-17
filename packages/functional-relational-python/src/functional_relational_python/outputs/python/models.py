"""Renders the planned View models: one module of Pydantic models per module.

Everything here is pure: it builds file contents and touches no filesystem.
`generated_files` writes and checks them.
"""

import typing
from collections.abc import Sequence

from functional_relational_python.outputs.python.plan import ModelPlan, PythonPlan
from functional_relational_python.outputs.python.symbols import (
    LineLength,
    ModulePath,
)
from functional_relational_python.support.filesystem import FileText
from functional_relational_python.support.generated_files import (
    GeneratedFile,
    GeneratedFiles,
)
from functional_relational_python.support.identifier import Identifier


class ModelsRenderer:
    """Writes the source of every planned View model."""

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
                        for view in plan.models[module]
                        for model in view.with_nested()
                    ),
                ),
            )
            for module in sorted(plan.models, key=lambda module: module.root)
        )

    @staticmethod
    def _module(module: ModulePath, models: Sequence[ModelPlan]) -> FileText:
        imported: dict[ModulePath, set[Identifier]] = {}
        for model in models:
            for symbol in model.uses:
                if symbol.module.root != "builtins" and symbol.module != module:
                    imported.setdefault(symbol.module, set()).add(symbol.bound_name())

        sections: list[list[str]] = []
        for section in ModulePath.import_sections(imported):
            lines: list[str] = []
            for source in section:
                names = sorted(name.root for name in imported[source])
                line = f"from {source.root} import {', '.join(names)}"
                if len(line) > LineLength.default().root:
                    listed = "".join(f"    {name},\n" for name in names)
                    line = f"from {source.root} import (\n{listed})"
                lines.append(line)
            sections.append(lines)
        imports = "\n\n".join("\n".join(lines) for lines in sections)

        base = ModelPlan.base.reference().root
        classes = "\n\n".join(
            f"class {model.symbol.bound_name().root}({base}, frozen=True):\n"
            + (
                "".join(
                    f"    {field.field.name.root}: {field.type.root}\n"
                    for field in model.fields
                )
                or "    pass\n"
            )
            for model in models
        )
        return FileText(f"{ModelsRenderer.header.root}\n{imports}\n\n\n{classes}")
