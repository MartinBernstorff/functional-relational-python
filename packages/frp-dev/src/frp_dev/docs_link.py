import enum
from pathlib import Path

from pydantic import ConfigDict, RootModel
from safe_result import Err, Ok, Result


class DocsSource(RootModel[Path]):
    """The folder holding the project docs, outside the repository."""

    model_config = ConfigDict(frozen=True)

    @staticmethod
    def from_home(home: HomeDir) -> DocsSource:
        return DocsSource(
            home.root
            / "Git"
            / "personal-projects-docs"
            / "functional-relational-python"
        )

    @staticmethod
    def fake() -> DocsSource:
        return DocsSource(Path("/tmp/frp-docs"))


class HomeDir(RootModel[Path]):
    model_config = ConfigDict(frozen=True)

    @staticmethod
    def fake() -> HomeDir:
        return HomeDir(Path("/tmp/home"))


class Checkout(RootModel[Path]):
    """The root of a clone or worktree of this repository."""

    model_config = ConfigDict(frozen=True)

    def docs_link(self) -> Path:
        return self.root / "docs"

    @staticmethod
    def fake() -> Checkout:
        return Checkout(Path("/tmp/checkout"))


class DocsLinked(enum.Enum):
    CREATED = "created"
    ALREADY_PRESENT = "already present"


class DocsUnlinked(enum.Enum):
    REMOVED = "removed"
    NOT_PRESENT = "not present"


class DocsSourceMissingError(Exception):
    def __init__(self, source: DocsSource) -> None:
        super().__init__(
            f"Docs folder not found at {source.root}; skipped linking docs/"
        )
        self.source = source


class DocsLinkOccupiedError(Exception):
    def __init__(self, checkout: Checkout) -> None:
        super().__init__(
            f"{checkout.docs_link()} exists and is not a symlink to the docs; "
            "move it away and re-run setup"
        )
        self.checkout = checkout


type DocsLinkError = DocsSourceMissingError | DocsLinkOccupiedError


class DocsLinker:
    """Symlinks the docs folder into a checkout as `docs/`."""

    @staticmethod
    def link(
        source: DocsSource, checkout: Checkout
    ) -> Result[DocsLinked, DocsLinkError]:
        link = checkout.docs_link()
        if link.is_symlink() and link.resolve() == source.root.resolve():
            return Ok(DocsLinked.ALREADY_PRESENT)
        if link.exists() or link.is_symlink():
            return Err(DocsLinkOccupiedError(checkout))
        if not source.root.is_dir():
            return Err(DocsSourceMissingError(source))
        link.symlink_to(source.root, target_is_directory=True)
        return Ok(DocsLinked.CREATED)

    @staticmethod
    def unlink(checkout: Checkout) -> DocsUnlinked:
        link = checkout.docs_link()
        if not link.is_symlink():
            return DocsUnlinked.NOT_PRESENT
        link.unlink()
        return DocsUnlinked.REMOVED
