"""Project config: `frp.toml`, found in the working directory or a parent.

The search stops at the directory containing `.git`, so a config outside the
repository is never picked up.
"""

import json
import tomllib
import typing
from pathlib import PurePosixPath

import pydantic
from safe_result import Err, Ok, Result

from functional_relational_python.shell.catalog import ImportPath
from functional_relational_python.support.filesystem import (
    Directory,
    FilePath,
    FileSystem,
    FileText,
    RelativePath,
)


class OptionName(pydantic.RootModel[str], frozen=True):
    pass


class OptionDescription(pydantic.RootModel[str], frozen=True):
    pass


class TomlValue(pydantic.RootModel[str], frozen=True):
    """A value as written in TOML, e.g. `"src"` with its quotes."""


class Option(pydantic.BaseModel, frozen=True):
    name: OptionName
    description: OptionDescription
    default: TomlValue


class ProjectConfig(pydantic.BaseModel, frozen=True, extra="forbid"):
    catalog: ImportPath = pydantic.Field(
        description="Where the Catalog is defined, as 'module:attribute'. "
        "The module is imported with this file's directory on the search path."
    )
    output_dir: RelativePath = pydantic.Field(
        description="The directory View model modules are written under, relative "
        "to this file. A model in module 'a.b' is written to '<output_dir>/a/b.py'."
    )

    @staticmethod
    def default() -> ProjectConfig:
        return ProjectConfig(
            catalog=ImportPath("views:catalog"),
            output_dir=RelativePath(PurePosixPath(".")),
        )

    @staticmethod
    def options() -> tuple[Option, ...]:
        """Every option, with what `config init` writes for it."""
        default = ProjectConfig.default().model_dump(mode="json")
        return tuple(
            Option(
                name=OptionName(name),
                description=OptionDescription(field.description or ""),
                default=TomlValue(json.dumps(default[name])),
            )
            for name, field in ProjectConfig.model_fields.items()
        )

    def to_toml(self) -> FileText:
        values = self.model_dump(mode="json")
        return FileText(
            "\n".join(
                f"# {option.description.root}\n"
                f"{option.name.root} = {json.dumps(values[option.name.root])}\n"
                for option in ProjectConfig.options()
            )
        )


class ConfigNotFoundError(Exception):
    def __init__(self, searched: tuple[Directory, ...]) -> None:
        self.searched = searched
        directories = ", ".join(str(directory.root) for directory in searched)
        super().__init__(
            f"No {ConfigFile.name.root} found in {directories}. "
            "Create one with `frp config init`."
        )


class ConfigSyntaxError(Exception):
    def __init__(self, path: FilePath, error: tomllib.TOMLDecodeError) -> None:
        self.path = path
        super().__init__(f"{path.root} is not valid TOML: {error}")


class InvalidConfigError(Exception):
    def __init__(self, path: FilePath, error: pydantic.ValidationError) -> None:
        self.path = path
        super().__init__(f"{path.root} is not a valid config: {error}")


class ConfigExistsError(Exception):
    def __init__(self, path: FilePath) -> None:
        self.path = path
        super().__init__(f"{path.root} already exists.")


type ConfigLoadError = ConfigNotFoundError | ConfigSyntaxError | InvalidConfigError


class ConfigFile(pydantic.BaseModel, frozen=True):
    """A project config, and the file it was read from."""

    name: typing.ClassVar[RelativePath] = RelativePath(PurePosixPath("frp.toml"))
    repository_marker: typing.ClassVar[RelativePath] = RelativePath(
        PurePosixPath(".git")
    )

    path: FilePath
    config: ProjectConfig

    def directory(self) -> Directory:
        return Directory(self.path.root.parent)

    def output_dir(self) -> Directory:
        return Directory(self.path.root.parent / self.config.output_dir.root)

    @staticmethod
    def find(
        filesystem: FileSystem, start: Directory
    ) -> Result[ConfigFile, ConfigLoadError]:
        searched: list[Directory] = []
        for directory in start.ancestry():
            searched.append(directory)
            path = directory.file(ConfigFile.name)
            match filesystem.read(path):
                case Ok(text):
                    return ConfigFile.parse(path, text)
                case Err(_):
                    pass
            if filesystem.exists(directory.file(ConfigFile.repository_marker)):
                break
        return Err(ConfigNotFoundError(tuple(searched)))

    @staticmethod
    def parse(path: FilePath, text: FileText) -> Result[ConfigFile, ConfigLoadError]:
        try:
            values = tomllib.loads(text.root)
        except tomllib.TOMLDecodeError as error:
            return Err(ConfigSyntaxError(path, error))
        try:
            config = ProjectConfig.model_validate(values)
        except pydantic.ValidationError as error:
            return Err(InvalidConfigError(path, error))
        return Ok(ConfigFile(path=path, config=config))

    @staticmethod
    def init(
        filesystem: FileSystem, directory: Directory
    ) -> Result[FilePath, ConfigExistsError]:
        """Writes the default config into `directory`, unless one is already there."""
        path = directory.file(ConfigFile.name)
        if filesystem.exists(path):
            return Err(ConfigExistsError(path))
        filesystem.write(path, ProjectConfig.default().to_toml())
        return Ok(path)
