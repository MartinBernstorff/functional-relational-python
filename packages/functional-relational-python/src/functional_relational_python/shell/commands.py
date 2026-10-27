"""What `frp generate` and `frp check` do, with the filesystem injected."""

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.outputs.python.symbols import FirstParty
from functional_relational_python.shell.catalog import CatalogLoadError
from functional_relational_python.shell.project_config import (
    ConfigFile,
    ConfigLoadError,
)
from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticsError,
)
from functional_relational_python.support.filesystem import (
    Directory,
    FilePath,
    FileSystem,
)
from functional_relational_python.support.generated_files import (
    GeneratedFiles,
    NotGeneratedFilesError,
    StaleFilesError,
)

type GenerateError = ConfigLoadError | CatalogLoadError | DiagnosticsError[Diagnostic]


class Generated(pydantic.BaseModel, frozen=True):
    files: GeneratedFiles
    output: Directory


def _generated(
    filesystem: FileSystem, start: Directory
) -> Result[Generated, GenerateError]:
    """The files the project's Catalog generates, and where they belong."""
    match ConfigFile.find(filesystem, start):
        case Err(config_error):
            return Err(config_error)
        case Ok(config_file):
            pass
    match config_file.config.catalog.load(config_file.directory()):
        case Err(load_error):
            return Err(load_error)
        case Ok(catalog):
            pass
    match catalog.generate(FirstParty(frozenset(config_file.config.known_first_party))):
        case Err(generate_error):
            return Err(generate_error)
        case Ok(files):
            return Ok(Generated(files=files, output=config_file.output_dir()))


def generate(
    filesystem: FileSystem, start: Directory
) -> Result[tuple[FilePath, ...], GenerateError | NotGeneratedFilesError]:
    """Writes every generated module, and returns the paths written."""
    match _generated(filesystem, start):
        case Err(error):
            return Err(error)
        case Ok(generated):
            pass
    match generated.files.write(filesystem, generated.output):
        case Err(write_error):
            return Err(write_error)
        case Ok(_):
            return Ok(
                tuple(
                    generated.output.file(file.path) for file in generated.files.files
                )
            )


def check(
    filesystem: FileSystem, start: Directory
) -> Result[None, GenerateError | StaleFilesError]:
    """Ok when every generated module on disk is what `generate` would write."""
    match _generated(filesystem, start):
        case Err(error):
            return Err(error)
        case Ok(generated):
            return generated.files.check(filesystem, generated.output)
