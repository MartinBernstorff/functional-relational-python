"""Generated files: writing them, and checking that what is on disk is up to date."""

import enum

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.filesystem import (
    Directory,
    FilePath,
    FileSystem,
    FileText,
    RelativePath,
)


class GeneratedFile(pydantic.BaseModel, frozen=True):
    path: RelativePath
    text: FileText


class Staleness(enum.StrEnum):
    MISSING = "missing"
    OUTDATED = "outdated"


class StaleFile(pydantic.BaseModel, frozen=True):
    path: FilePath
    staleness: Staleness


class StaleFilesError(Exception):
    def __init__(self, stale: tuple[StaleFile, ...]) -> None:
        self.stale = stale
        lines = (f"{file.path.root} is {file.staleness}" for file in stale)
        super().__init__(
            "\n".join(
                [
                    "Generated files are stale:",
                    *lines,
                    "Run `frp generate` to update them.",
                ]
            )
        )


class GeneratedFiles(pydantic.BaseModel, frozen=True):
    files: tuple[GeneratedFile, ...]

    def write(self, filesystem: FileSystem, output: Directory) -> None:
        for file in self.files:
            filesystem.write(output.file(file.path), file.text)

    def check(
        self, filesystem: FileSystem, output: Directory
    ) -> Result[None, StaleFilesError]:
        """Ok when every file on disk matches what would be generated."""
        stale: list[StaleFile] = []
        for file in self.files:
            path = output.file(file.path)
            match filesystem.read(path):
                case Ok(text) if text == file.text:
                    pass
                case Ok(_):
                    stale.append(StaleFile(path=path, staleness=Staleness.OUTDATED))
                case Err(_):
                    stale.append(StaleFile(path=path, staleness=Staleness.MISSING))
        if stale:
            return Err(StaleFilesError(tuple(stale)))
        return Ok(None)
