import importlib.metadata

from typer.testing import CliRunner

from functional_relational_python import cli


def test_version_prints_the_installed_version() -> None:
    expected = importlib.metadata.version("functional-relational-python")

    result = CliRunner().invoke(cli.app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == expected
