from typing import Annotated, TypeIs

from pydantic import ConfigDict, RootModel, SkipValidation, field_validator


class Id[T](RootModel[T]):
    """frp's own ID base: `IdBase(Id)` makes every `Id` subclass an ID type.

    Generated code imports the ID types, so with this base it needs frp installed.
    An application that must run without frp names its own base, see `IdBase`.

    >>> class DepartmentId(Id[int]):
    ...     pass
    """

    model_config = ConfigDict(frozen=True)


# Pydantic would check against the parametrised class `RootModel[object]`, which no
# ID type subclasses. Every IdType comes from an `IdBase.includes` check, so skip it.
type IdType = Annotated[type[RootModel[object]], SkipValidation]
"""Any ID type, i.e. any subclass of the catalog's `IdBase`."""

# Likewise, no ID is an instance of `RootModel[object]`, and validating would
# convert it.
type IdValue = Annotated[RootModel[object], SkipValidation]
"""Any ID, i.e. an instance of any ID type."""


class IdBase(RootModel[IdType], frozen=True):
    """The class every ID type subclasses, named by the catalog.

    A model whose `id` field is an ID type is an Entity, and any other field typed
    with an ID type is a Reference to the Entity keyed by that type. The base is
    a strict `RootModel` subclass, e.g. frp's `Id`, or an application's own:

    >>> class AppId(RootModel[UUID], frozen=True):
    ...     pass
    >>> IdBase(AppId)

    `RootModel` itself is no base, since every newtype would be an ID type.
    """

    @field_validator("root")
    @classmethod
    def _strict_root_model_subclass(cls, root: object) -> object:
        match root:
            case type() if issubclass(root, RootModel) and root is not RootModel:
                return root
            case _:
                msg = f"{root!r} is not a strict subclass of RootModel"
                raise ValueError(msg)

    @staticmethod
    def fake() -> IdBase:
        return IdBase(Id)

    def includes(self, annotation: object) -> TypeIs[type[RootModel[object]]]:
        """Whether `annotation` is an ID type: a strict subclass of this base."""
        return (
            isinstance(annotation, type)
            and issubclass(annotation, self.root)
            and annotation is not self.root
        )
