from typing import Annotated

from pydantic import ConfigDict, RootModel, SkipValidation


class Id[T](RootModel[T]):
    """The base of every ID type.

    A model whose `id` field is an `Id` subclass is an Entity, and any other field
    typed with an `Id` subclass is a Reference to the Entity keyed by that type.

    >>> class DepartmentId(Id[int]):
    ...     pass
    """

    model_config = ConfigDict(frozen=True)


# Pydantic would check against the parametrised class `Id[object]`, which no ID type
# subclasses. Every IdType comes from an `issubclass(_, Id)` check, so skip it.
type IdType = Annotated[type[Id[object]], SkipValidation]
"""Any ID type, i.e. any subclass of `Id`."""

# Likewise, no ID is an instance of `Id[object]`, and validating would convert it.
type IdValue = Annotated[Id[object], SkipValidation]
"""Any ID, i.e. an instance of any subclass of `Id`."""
