"""The filesystem, as a port, so code that reads and writes files can be tested."""

import abc
from pathlib import Path, PurePosixPath

import pydantic
from safe_result import Err, Ok, Result


class FilePath(pydantic.RootModel[Path], frozen=True):
    @staticmethod
    def fake() -> FilePath:
        return FilePath(Path("/project/app/views.py"))


class RelativePath(pydantic.RootModel[PurePosixPath], frozen=True):
    """A path relative to some directory, e.g. `app/views.py`."""

    @staticmethod
    def fake() -> RelativePath:
        return RelativePath(PurePosixPath("app/views.py"))


class Directory(pydantic.RootModel[Path], frozen=True):
    @staticmethod
    def fake() -> Directory:
        return Directory(Path("/project"))

    def file(self, path: RelativePath) -> FilePath:
        return FilePath(self.root / path.root)

    def ancestry(self) -> tuple[Directory, ...]:
        """This directory, then each of its parents up to the filesystem root."""
        return tuple(Directory(path) for path in (self.root, *self.root.parents))


class FileText(pydantic.RootModel[str], frozen=True):
    @staticmethod
    def fake() -> FileText:
        return FileText("x = 1\n")


class FileMissingError(Exception):
    def __init__(self, path: FilePath) -> None:
        self.path = path
        super().__init__(f"{path.root} does not exist")


class FileSystem(abc.ABC):
    @abc.abstractmethod
    def read(self, path: FilePath) -> Result[FileText, FileMissingError]: ...

    @abc.abstractmethod
    def write(self, path: FilePath, text: FileText) -> None:
        """Writes `text`, creating missing parent directories."""

    @abc.abstractmethod
    def exists(self, path: FilePath) -> bool:
        """Whether a file or directory exists at `path`."""

    @abc.abstractmethod
    def delete(self, path: FilePath) -> None:
        """Deletes the file at `path`, if there is one."""

    @abc.abstractmethod
    def python_files(self, directory: Directory) -> tuple[FilePath, ...]:
        """Every `.py` file under `directory`, in path order.

        Hidden directories, such as `.venv` and `.git`, are skipped.
        """


class LocalFileSystem(FileSystem):
    def read(self, path: FilePath) -> Result[FileText, FileMissingError]:
        try:
            return Ok(FileText(path.root.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return Err(FileMissingError(path))

    def write(self, path: FilePath, text: FileText) -> None:
        path.root.parent.mkdir(parents=True, exist_ok=True)
        _ = path.root.write_text(text.root, encoding="utf-8")

    def exists(self, path: FilePath) -> bool:
        return path.root.exists()

    def delete(self, path: FilePath) -> None:
        path.root.unlink(missing_ok=True)

    def python_files(self, directory: Directory) -> tuple[FilePath, ...]:
        found: list[FilePath] = []
        for root, directories, files in directory.root.walk():
            directories[:] = [name for name in directories if not name.startswith(".")]
            found.extend(
                FilePath(root / name) for name in files if name.endswith(".py")
            )
        return tuple(sorted(found, key=lambda path: path.root))


class InMemoryFileSystem(FileSystem):
    """A stateful fake: `files` is what has been written."""

    def __init__(self) -> None:
        self.files: dict[FilePath, FileText] = {}

    def read(self, path: FilePath) -> Result[FileText, FileMissingError]:
        if path in self.files:
            return Ok(self.files[path])
        return Err(FileMissingError(path))

    def write(self, path: FilePath, text: FileText) -> None:
        self.files[path] = text

    def exists(self, path: FilePath) -> bool:
        return path in self.files or any(
            path.root in file.root.parents for file in self.files
        )

    def delete(self, path: FilePath) -> None:
        _ = self.files.pop(path, None)

    def python_files(self, directory: Directory) -> tuple[FilePath, ...]:
        return tuple(
            sorted(
                (
                    path
                    for path in self.files
                    if path.root.is_relative_to(directory.root)
                    and path.root.suffix == ".py"
                    and not any(
                        part.startswith(".")
                        for part in path.root.relative_to(directory.root).parent.parts
                    )
                ),
                key=lambda path: path.root,
            )
        )
