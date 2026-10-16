from pathlib import PurePosixPath

import pytest
from safe_result import Err, Ok

from functional_relational_python.shell.project_config import (
    ConfigExistsError,
    ConfigFile,
    ConfigLoadError,
    ConfigNotFoundError,
    ConfigSyntaxError,
    InvalidConfigError,
    ProjectConfig,
)
from functional_relational_python.support.filesystem import (
    Directory,
    FileText,
    InMemoryFileSystem,
    RelativePath,
)


def repository(filesystem: InMemoryFileSystem, root: Directory) -> None:
    filesystem.write(
        root.file(RelativePath(PurePosixPath(".git/HEAD"))), FileText.fake()
    )


def subdirectory(directory: Directory) -> Directory:
    return Directory(directory.root / "sub" / "dir")


def find_error(filesystem: InMemoryFileSystem, start: Directory) -> ConfigLoadError:
    match ConfigFile.find(filesystem, start):
        case Ok(found):
            pytest.fail(f"expected an error, got {found}")
        case Err(error):
            return error


def test_init_writes_the_default_config_which_find_reads_back() -> None:
    filesystem, root = InMemoryFileSystem(), Directory.fake()

    _ = ConfigFile.init(filesystem, root).unwrap()

    assert ConfigFile.find(filesystem, root) == Ok(
        ConfigFile(path=root.file(ConfigFile.name), config=ProjectConfig.default())
    )


def test_init_does_not_overwrite_a_config() -> None:
    filesystem, root = InMemoryFileSystem(), Directory.fake()
    _ = ConfigFile.init(filesystem, root).unwrap()

    match ConfigFile.init(filesystem, root):
        case Ok(path):
            pytest.fail(f"expected an error, wrote {path}")
        case Err(error):
            assert isinstance(error, ConfigExistsError)
            assert error.path == root.file(ConfigFile.name)


def test_find_searches_parent_directories() -> None:
    filesystem, root = InMemoryFileSystem(), Directory.fake()
    repository(filesystem, root)
    _ = ConfigFile.init(filesystem, root).unwrap()

    found = ConfigFile.find(filesystem, subdirectory(root)).unwrap()

    assert found.directory() == root


def test_find_stops_at_the_repository_root() -> None:
    filesystem, outside = InMemoryFileSystem(), Directory.fake()
    _ = ConfigFile.init(filesystem, outside).unwrap()
    root = Directory(outside.root / "repository")
    repository(filesystem, root)

    error = find_error(filesystem, subdirectory(root))

    assert isinstance(error, ConfigNotFoundError)
    assert error.searched == subdirectory(root).ancestry()[:3]


@pytest.mark.parametrize(
    ("text", "error_type"),
    [
        (FileText("catalog = "), ConfigSyntaxError),
        (FileText('catalog = "no-colon"\noutput_dir = "."\n'), InvalidConfigError),
        (FileText('catalog = "a:class"\noutput_dir = "."\n'), InvalidConfigError),
        (FileText('catalog = "a:b"\n'), InvalidConfigError),
        (
            FileText('catalog = "a:b"\noutput_dir = "."\nunknown = 1\n'),
            InvalidConfigError,
        ),
    ],
)
def test_a_bad_config_is_an_error(
    text: FileText, error_type: type[ConfigLoadError]
) -> None:
    filesystem, root = InMemoryFileSystem(), Directory.fake()
    filesystem.write(root.file(ConfigFile.name), text)

    assert isinstance(find_error(filesystem, root), error_type)


def test_output_dir_is_relative_to_the_config_file() -> None:
    root = Directory.fake()
    output_dir = RelativePath(PurePosixPath("src"))

    config_file = ConfigFile(
        path=root.file(ConfigFile.name),
        config=ProjectConfig.default().model_copy(update={"output_dir": output_dir}),
    )

    assert config_file.output_dir() == Directory(root.root / output_dir.root)
