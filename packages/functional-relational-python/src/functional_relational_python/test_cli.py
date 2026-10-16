import importlib.metadata

from typer.testing import CliRunner

from functional_relational_python import cli
from functional_relational_python.project_config import ProjectConfig


def test_version_prints_the_installed_version() -> None:
    expected = importlib.metadata.version("functional-relational-python")

    result = CliRunner().invoke(cli.app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == expected


def test_config_list_shows_every_option() -> None:
    result = CliRunner().invoke(cli.app, ["config", "list"])

    assert result.exit_code == 0
    assert [
        line.split(" ")[0] for line in result.stdout.splitlines() if line[:1] != " "
    ] == [option.name.root for option in ProjectConfig.options()]
