import subprocess
import sys

import pytest

from functional_relational_python.outputs.python.symbols import (
    ImportBlock,
    ImportSection,
    ImportText,
    ModulePath,
    SitePackages,
    Symbol,
)
from functional_relational_python.support.filesystem import FileText
from functional_relational_python.support.identifier import Identifier


@pytest.mark.parametrize(
    ("module", "section"),
    [
        (ModulePath("collections.abc"), ImportSection.STANDARD_LIBRARY),
        (ModulePath("pydantic.types"), ImportSection.THIRD_PARTY),
        (
            ModulePath("functional_relational_python.outputs"),
            ImportSection.FIRST_PARTY,
        ),
        (ModulePath("app.views"), ImportSection.FIRST_PARTY),
    ],
    ids=["standard-library", "installed", "project", "not-importable"],
)
def test_a_module_is_in_the_section_of_where_it_is_installed(
    module: ModulePath, section: ImportSection
) -> None:
    assert ImportSection.of(module, SitePackages.current()) == section


def mixed_block() -> ImportBlock:
    """Imports from every section, with names of every kind ruff orders."""
    return ImportBlock(
        modules=frozenset(
            map(ModulePath, ("collections.abc", "pydantic", "app.views", "App2"))
        ),
        symbols=frozenset(
            Symbol(module=ModulePath(module), qualname=(Identifier(name),))
            for module, names in (
                ("typing", ("Annotated", "TYPE_CHECKING", "cast")),
                ("pydantic", ("MISSING", "BaseModel")),
                ("pydantic.types", ("AwareDatetime",)),
                ("app.views", ("Company", "company_id", "v1", "v001", "v01")),
                ("app.views10", ("View",)),
                ("app.views2", ("View",)),
                ("app.views02", ("View",)),
            )
            for name in names
        ),
    )


def test_an_import_block_puts_each_section_apart() -> None:
    assert mixed_block().text() == ImportText(
        "import collections.abc\n"
        "from typing import TYPE_CHECKING, Annotated, cast\n"
        "\n"
        "import pydantic\n"
        "from pydantic import MISSING, BaseModel\n"
        "from pydantic.types import AwareDatetime\n"
        "\n"
        "import app.views\n"
        "import App2\n"
        "from app.views import Company, company_id, v001, v01, v1\n"
        "from app.views02 import View\n"
        "from app.views2 import View\n"
        "from app.views10 import View\n"
    )


def test_a_long_import_from_is_split_as_ruff_splits_it() -> None:
    names = [f"Name{index:02}xxxxxxxxxx" for index in range(4)]
    block = ImportBlock(
        modules=frozenset(),
        symbols=frozenset(
            Symbol(module=ModulePath.fake(), qualname=(Identifier(name),))
            for name in names
        ),
    )

    assert block.text() == ImportText(
        f"from {ModulePath.fake().root} import (\n"
        + "".join(f"    {name},\n" for name in names)
        + ")\n"
    )


def test_an_import_block_is_as_ruff_sorts_it() -> None:
    assert ruff_import_diff(mixed_block().text()) == FileText("")


def test_an_empty_import_block_has_no_text() -> None:
    assert ImportBlock(modules=frozenset(), symbols=frozenset()).text() == ImportText(
        ""
    )


def ruff_import_diff(text: FileText | ImportText) -> FileText:
    """How ruff would sort the imports of `text`: empty when they are sorted
    already. The packages generated source here imports from site-packages are
    third party and the rest first party, which is how `ImportSection.of` has
    them in this workspace."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            "I",
            "--diff",
            "--config",
            'lint.isort.default-section = "first-party"',
            "--config",
            'lint.isort.known-third-party = ["annotated_types", "django", "pydantic"]',
            "--stdin-filename",
            "generated.py",
            "-",
        ],
        input=text.root,
        capture_output=True,
        text=True,
        check=False,
    )
    return FileText(result.stdout + result.stderr)
