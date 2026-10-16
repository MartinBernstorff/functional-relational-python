import logging
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ConfigDict, RootModel
from safe_result import Err, Ok, Result

from frp_dev.docs_link import (
    Checkout,
    DocsLinker,
    DocsLinkOccupiedError,
    DocsSource,
    DocsSourceMissingError,
    HomeDir,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Development commands for the functional-relational-python repo.",
)


class ShellCommand(RootModel[tuple[str, ...]]):
    model_config = ConfigDict(frozen=True)

    @staticmethod
    def fake() -> ShellCommand:
        return ShellCommand(("true",))

    def run(self, checkout: Checkout) -> Result[None, ShellCommandFailedError]:
        logging.getLogger(__name__).info("$ %s", " ".join(self.root))
        completed = subprocess.run(self.root, cwd=checkout.root, check=False)
        if completed.returncode != 0:
            return Err(ShellCommandFailedError(self, ExitCode(completed.returncode)))
        return Ok(None)


class ExitCode(RootModel[int]):
    model_config = ConfigDict(frozen=True)

    @staticmethod
    def fake() -> ExitCode:
        return ExitCode(1)


class ShellCommandFailedError(Exception):
    def __init__(self, command: ShellCommand, exit_code: ExitCode) -> None:
        super().__init__(f"`{' '.join(command.root)}` exited with {exit_code.root}")
        self.command = command


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
def setup() -> None:
    """Make the current checkout ready to work in. Safe to re-run.

    Installs the pinned toolchain (moon, Python, uv) with proto, syncs the
    virtualenv, installs the lefthook git hooks, and symlinks the docs folder in
    as `docs/`. The docs folder is read from $FRP_DOCS_SOURCE, falling back to
    ~/Git/functional-relational-python-docs/functional-relational-python-docs.
    """
    logger = logging.getLogger(__name__)
    checkout = Checkout(Path.cwd())
    for command in (
        ShellCommand(("proto", "install")),
        ShellCommand(("uv", "sync", "--locked")),
        ShellCommand(("uv", "run", "lefthook", "install")),
    ):
        match command.run(checkout):
            case Ok():
                pass
            case Err(error):
                logger.error("%s", error)
                raise typer.Exit(code=1)

    env_source = os.environ.get("FRP_DOCS_SOURCE")
    source = (
        DocsSource(Path(env_source))
        if env_source
        else DocsSource.from_home(HomeDir(Path.home()))
    )
    match DocsLinker.link(source, checkout):
        case Ok(outcome):
            logger.info("docs/: %s", outcome.value)
        case Err(error):
            match error:
                case DocsSourceMissingError():
                    # Contributors without the docs folder can still work in the repo.
                    logger.info("%s", error)
                case DocsLinkOccupiedError():
                    logger.error("%s", error)
                    raise typer.Exit(code=1)


@app.command()
def teardown() -> None:
    """Clean up before the current checkout is deleted. Safe to re-run.

    Removes the `docs/` symlink. The virtualenv and task cache live inside the
    checkout and go with it.
    """
    outcome = DocsLinker.unlink(Checkout(Path.cwd()))
    logging.getLogger(__name__).info("docs/: %s", outcome.value)
