from pathlib import Path, PurePosixPath

import pytest
from safe_result import Err

from functional_relational_python.support.filesystem import (
    Directory,
    FileSystem,
    FileText,
    InMemoryFileSystem,
    LocalFileSystem,
    RelativePath,
)


@pytest.fixture(params=["in-memory", "local"])
def filesystem(request: pytest.FixtureRequest) -> FileSystem:
    """Each implementation, so the fake is held to what the real one does."""
    match request.param:
        case "in-memory":
            return InMemoryFileSystem()
        case _:
            return LocalFileSystem()


def test_python_files_are_found_in_subdirectories_but_not_hidden_ones(
    filesystem: FileSystem, tmp_path: Path
) -> None:
    root = Directory(tmp_path)
    output = Directory(tmp_path / "output")
    found = [
        RelativePath(PurePosixPath("a.py")),
        RelativePath(PurePosixPath("sub/b.py")),
    ]
    skipped = [
        RelativePath(PurePosixPath("notes.txt")),
        RelativePath(PurePosixPath(".venv/c.py")),
    ]
    for path in [*found, *skipped]:
        filesystem.write(output.file(path), FileText.fake())
    filesystem.write(
        root.file(RelativePath(PurePosixPath("outside.py"))), FileText.fake()
    )

    assert filesystem.python_files(output) == tuple(map(output.file, found))


def test_a_deleted_file_is_gone(filesystem: FileSystem, tmp_path: Path) -> None:
    path = Directory(tmp_path).file(RelativePath(PurePosixPath("a.py")))
    filesystem.write(path, FileText.fake())

    filesystem.delete(path)

    assert isinstance(filesystem.read(path), Err)
    assert not filesystem.exists(path)
