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
    """A tagged union of Views, or of Payloads: a name, a Discriminator and the
    Variants.

    Its model is `Annotated[A | B, Discriminator("<discriminator>")]`, named
    `name`. Each Variant's model has the field `discriminator`, typed by one Tag
    (`Literal[<tag>]`). Two Variants may not share a Tag, and a nested Union's
    Variants all have the same Tag for `discriminator`, its own Tag here.

    A Union of Views has Views and nested Unions of Views as Variants. Each View
    generates its own model and mapping function as if the Catalog listed it, and
    so does each nested Union. Every Variant reads one Source. A Source row gives
    the result of the first Variant, in order, whose Filters keep it; a row no
    Variant keeps gives no result. A Variant's Tag is a Source column its `where`
    narrows with `==`, or a field `add` declares.

    A Union of Payloads has Payloads and nested Unions of Payloads as Variants,
    which may read different Sources. Each Payload generates its own model as if
    the Catalog listed it. A Union of Payloads is a write model, so it generates
    only its model, no mapping function. A Payload has no Filters, so its Tag is
    a field `retype_to` types, or `add` declares, as a `Literal` of one value.
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
        return tuple(
            dict.fromkeys(leaf for leaf in self.leaves() if isinstance(leaf, View))
        )

    def payloads(self) -> tuple[Payload, ...]:
        """The Payloads among this Union's Variants and those of the Unions nested
        in it, in Variant order, once each."""
        return tuple(
            dict.fromkeys(leaf for leaf in self.leaves() if isinstance(leaf, Payload))
        )

    def leaves(self) -> tuple[View | Payload, ...]:
        """The Views and Payloads among this Union's Variants and those of the
        Unions nested in it, in Variant order."""
        return tuple(
            leaf
            for variant in self.variants
            for leaf in (variant.leaves() if isinstance(variant, Union) else (variant,))
        )
