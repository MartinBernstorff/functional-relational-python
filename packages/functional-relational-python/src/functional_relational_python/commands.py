"""What `frp generate` and `frp check` do, with the filesystem injected."""

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python import compiler
from functional_relational_python.catalog import CatalogLoadError
from functional_relational_python.codegen import CodegenError
from functional_relational_python.entity_graph import EntityGraphError
from functional_relational_python.filesystem import Directory, FilePath, FileSystem
from functional_relational_python.generated_files import (
    GeneratedFiles,
    StaleFilesError,
)
from functional_relational_python.project_config import ConfigFile, ConfigLoadError

type GenerateError = (
    ConfigLoadError
    | CatalogLoadError
    | EntityGraphError
    | compiler.CompileError
    | CodegenError
)


class Generated(pydantic.BaseModel, frozen=True):
    files: GeneratedFiles
    output: Directory


def _generated(
    filesystem: FileSystem, start: Directory
) -> Result[Generated, GenerateError]:
    """The files the project's ViewCatalog generates, and where they belong."""
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
    match catalog.generate():
        case Err(generate_error):
            return Err(generate_error)
        case Ok(files):
            return Ok(Generated(files=files, output=config_file.output_dir()))


def generate(
    filesystem: FileSystem, start: Directory
) -> Result[tuple[FilePath, ...], GenerateError]:
    """Writes every View model module, and returns the paths written."""
    match _generated(filesystem, start):
        case Err(error):
            return Err(error)
        case Ok(generated):
            generated.files.write(filesystem, generated.output)
            return Ok(
                tuple(
                    generated.output.file(file.path) for file in generated.files.files
                )
            )


def check(
    filesystem: FileSystem, start: Directory
) -> Result[None, GenerateError | StaleFilesError]:
    """Ok when every View model module on disk is what `generate` would write."""
    match _generated(filesystem, start):
        case Err(error):
            return Err(error)
        case Ok(generated):
            return generated.files.check(filesystem, generated.output)
