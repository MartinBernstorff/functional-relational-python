from typing import Annotated

import pydantic

from functional_relational_python.entity_graph.entity_graph import FieldName
from functional_relational_python.pipeline.payloads import Payload
from functional_relational_python.pipeline.views import View


class UnionName(pydantic.RootModel[str], frozen=True):
    @staticmethod
    def fake() -> UnionName:
        return UnionName("MetaCampaign")


class Union(pydantic.BaseModel, frozen=True):
    """A tagged union of Views: a name, a Discriminator and the Variants.

    Its model is `Annotated[A | B, Discriminator("<discriminator>")]`, named
    `name`. Each Variant is a View, which generates its own model and mapping
    function as if the Catalog listed it, or a nested Union, which generates its
    own model and mapping function too.

    Every Variant reads one Source. A Source row gives the result of the first
    Variant, in order, whose Filters keep it; a row no Variant keeps gives no
    result. Each Variant's model has the field `discriminator`, typed by one Tag
    (`Literal[<tag>]`): a Source column a Variant's `where` narrows with `==`, or
    a field `add` declares. Two Variants may not share a Tag, and a nested Union's
    Variants all have the same Tag for `discriminator`, its own Tag here.

    Payloads are Variants only in Unions of Payloads, which are not supported yet.
    """

    name: UnionName
    discriminator: FieldName
    variants: Annotated[
        tuple[View | Payload | Union, ...], pydantic.Field(min_length=2)
    ]

    def unions(self) -> tuple[Union, ...]:
        """This Union and every Union nested in it, each after those nested in
        it, once each."""
        nested = (
            union
            for variant in self.variants
            if isinstance(variant, Union)
            for union in variant.unions()
        )
        return tuple(dict.fromkeys((*nested, self)))

    def views(self) -> tuple[View, ...]:
        """The Views among this Union's Variants and those of the Unions nested in
        it, in Variant order, once each."""
        found: list[View] = []
        for variant in self.variants:
            match variant:
                case View():
                    found.append(variant)
                case Union():
                    found.extend(variant.views())
                case Payload():
                    pass
        return tuple(dict.fromkeys(found))
