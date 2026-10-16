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


def generated_text() -> FileText:
    return FileText(f"{GeneratedFiles.header.root}{FileText.fake().root}")


def two_files() -> GeneratedFiles:
    return GeneratedFiles(
        files=(
            GeneratedFile(path=RelativePath.fake(), text=generated_text()),
            GeneratedFile(
                path=RelativePath(PurePosixPath("app/other.py")),
                text=generated_text(),
            ),
        )
    )


def test_written_files_are_up_to_date() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()
    files.write(filesystem, output).unwrap()

    assert files.check(filesystem, output) == Ok(None)


def test_write_puts_each_file_under_the_output_directory() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()

    files.write(filesystem, output).unwrap()

    assert filesystem.files == {
        output.file(file.path): file.text for file in files.files
    }


def test_write_replaces_a_file_it_generated_before() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()
    first, _ = files.files
    filesystem.write(
        output.file(first.path), FileText(f"{first.text.root}# outdated\n")
    )

    files.write(filesystem, output).unwrap()

    assert filesystem.files[output.file(first.path)] == first.text


def test_write_refuses_to_overwrite_files_it_did_not_generate() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()
    for file in files.files:
        filesystem.write(output.file(file.path), FileText.fake())
    before = dict(filesystem.files)

    match files.write(filesystem, output):
        case Ok(_):
            pytest.fail("expected NotGeneratedFilesError")
        case Err(error):
            assert error.paths == tuple(output.file(file.path) for file in files.files)
    assert filesystem.files == before


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


def test_check_reports_a_generated_file_that_is_no_longer_generated() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()
    files.write(filesystem, output).unwrap()
    kept, removed = files.files

    match GeneratedFiles(files=(kept,)).check(filesystem, output):
        case Ok(_):
            pytest.fail("expected stale files")
        case Err(error):
            assert error.stale == (
                StaleFile(path=output.file(removed.path), staleness=Staleness.LEFTOVER),
            )


def test_write_deletes_generated_files_that_are_no_longer_generated() -> None:
    filesystem, output, files = InMemoryFileSystem(), Directory.fake(), two_files()
    files.write(filesystem, output).unwrap()
    users_own = output.file(RelativePath(PurePosixPath("app/models.py")))
    filesystem.write(users_own, FileText.fake())
    kept, _ = files.files

    GeneratedFiles(files=(kept,)).write(filesystem, output).unwrap()

    assert set(filesystem.files) == {output.file(kept.path), users_own}
