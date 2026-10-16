import keyword

import pydantic
from safe_result import Err, Ok, Result


class InvalidIdentifierError(Exception):
    def __init__(self, name: pydantic.RootModel[str]) -> None:
        self.name = name
        super().__init__(f"'{name.root}' is not a Python identifier")


class Identifier(pydantic.RootModel[str], frozen=True):
    """A Python name that is not a keyword, e.g. a class or module name."""

    @pydantic.field_validator("root")
    @classmethod
    def _not_keyword(cls, root: str) -> str:
        if not root.isidentifier() or keyword.iskeyword(root):
            msg = f"'{root}' is not a Python identifier"
            raise ValueError(msg)
        return root

    @staticmethod
    def fake() -> Identifier:
        return Identifier("CompanySummary")

    @staticmethod
    def of(name: pydantic.RootModel[str]) -> Result[Identifier, InvalidIdentifierError]:
        """`name` as an Identifier, e.g. a View name as a class name."""
        try:
            return Ok(Identifier(name.root))
        except pydantic.ValidationError:
            return Err(InvalidIdentifierError(name))
