import importlib.metadata
import logging
from typing import Annotated

import typer

app = typer.Typer(
    no_args_is_help=True,
    help="Generate Pydantic view models from a relational algebra over your entities.",
)


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
