from pathlib import PurePosixPath

import pytest
from safe_result import Err, Ok

from functional_relational_python.filesystem import (
    Directory,
    FileText,
    InMemoryFileSystem,
    RelativePath,
)
from functional_relational_python.generated_files import (
    GeneratedFile,
    GeneratedFiles,
    StaleFile,
    Staleness,
)


def two_files() -> GeneratedFiles:
    return GeneratedFiles(
        files=(
            GeneratedFile(path=RelativePath.fake(), text=FileText.fake()),
            GeneratedFile(
                path=RelativePath(PurePosixPath("app/other.py")),
                text=FileText.fake(),
            ),
        )
    )


def test_written_files_are_up_to_date() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()
    files.write(filesystem, output)

    assert files.check(filesystem, output) == Ok(None)


def test_write_puts_each_file_under_the_output_directory() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()

    files.write(filesystem, output)

    assert filesystem.files == {
        output.file(file.path): file.text for file in files.files
    }


def test_missing_and_outdated_files_are_all_stale() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()
    outdated, missing = files.files
    filesystem.write(
        output.file(outdated.path), FileText(f"{outdated.text.root}# edited\n")
    )

    match files.check(filesystem, output):
        case Ok(_):
            pytest.fail("expected stale files")
        case Err(error):
            assert error.stale == (
                StaleFile(
                    path=output.file(outdated.path), staleness=Staleness.OUTDATED
                ),
                StaleFile(path=output.file(missing.path), staleness=Staleness.MISSING),
            )
