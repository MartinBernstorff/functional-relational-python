import importlib.metadata
import logging
from pathlib import Path
from typing import Annotated

import typer
from safe_result import Err, Ok

from functional_relational_python import commands
from functional_relational_python.filesystem import Directory, LocalFileSystem
from functional_relational_python.project_config import ConfigFile, ProjectConfig

app = typer.Typer(
    no_args_is_help=True,
    help="Generate Pydantic view models from a relational algebra over your entities.",
)
config_app = typer.Typer(
    no_args_is_help=True,
    help=f"Manage the project config, {ConfigFile.name.root}.",
)
app.add_typer(config_app, name="config")


@app.callback()
def configure_logging(
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Only log warnings and errors.")
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO, format="%(message)s"
    )


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(importlib.metadata.version("functional-relational-python"))


@app.command()
def generate() -> None:
    """Write the View model modules of the project's ViewCatalog."""
    match commands.generate(LocalFileSystem(), Directory(Path.cwd())):
        case Ok(written):
            for path in written:
                logging.getLogger(__name__).info("Wrote %s", path.root)
        case Err(error):
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1)


@app.command()
def check() -> None:
    """Fail when a View model module differs from what `generate` would write."""
    match commands.check(LocalFileSystem(), Directory(Path.cwd())):
        case Ok(_):
            logging.getLogger(__name__).info("Generated View models are up to date.")
        case Err(error):
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1)


@config_app.command("init")
def config_init() -> None:
    """Write a config with default values to the working directory."""
    match ConfigFile.init(LocalFileSystem(), Directory(Path.cwd())):
        case Ok(path):
            logging.getLogger(__name__).info("Wrote %s", path.root)
        case Err(error):
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1)


@config_app.command("list")
def config_list() -> None:
    """Show every config option, with its default."""
    for option in ProjectConfig.options():
        typer.echo(f"{option.name.root} (default: {option.default.root})")
        typer.echo(f"    {option.description.root}")


@config_app.command("show")
def config_show() -> None:
    """Show where the config was found, and its values."""
    match ConfigFile.find(LocalFileSystem(), Directory(Path.cwd())):
        case Ok(config_file):
            typer.echo(f"# {config_file.path.root}")
            typer.echo(config_file.config.to_toml().root, nl=False)
        case Err(error):
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1)
