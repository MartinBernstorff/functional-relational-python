import typing
from typing import Annotated, LiteralString

import pydantic

from functional_relational_python.entity_graph.entity_graph import SpecialForm

if typing.TYPE_CHECKING:
    from pydantic_core import CoreSchema


class SortedSequence[T, K: LiteralString](tuple[T, ...]):
    """The Entities a View holds for a reversed many-to-one Reference, ascending by
    the field `K` of the Entity each element was built from, e.g.
    `SortedSequence[Ad, Literal["id"]]`.

    `K` names a field of that Entity, not of the element: a nested pipeline may
    omit or rename it. The type documents the order and checks nothing; each
    output orders the elements itself. Django orders them in SQL, a mapping
    function's caller in the query that fills its parameter, and the
    InMemoryViewEngine in Python, by `id.root`. A string key may order differently
    under a database's collation than in Python.

    A model field typed with it validates from a list or a tuple of its elements,
    and dumps as a JSON array.
    """

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: object, handler: pydantic.GetCoreSchemaHandler
    ) -> CoreSchema:
        """A tuple of the elements' type, made a SortedSequence once validated."""
        match typing.get_args(source):
            case (element, *_):
                elements: object = element
            case _:
                elements = object
        # The element type is known at run time only, so the types are
        # subscripted as special forms.
        items = typing.cast("SpecialForm", tuple)[elements, ...]
        return handler.generate_schema(
            typing.cast("SpecialForm", Annotated)[items, pydantic.AfterValidator(cls)]
        )
