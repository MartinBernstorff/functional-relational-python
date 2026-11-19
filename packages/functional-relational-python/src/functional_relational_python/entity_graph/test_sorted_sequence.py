from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import pydantic
import pytest

from functional_relational_python.entity_graph.id import Id
from functional_relational_python.entity_graph.sorted_sequence import SortedSequence


class AdId(Id[int]):
    pass


class Ad(pydantic.BaseModel, frozen=True):
    id: AdId


class Campaign(pydantic.BaseModel, frozen=True):
    ads: SortedSequence[Ad, Literal["id"]]


@pytest.mark.parametrize("given", [list, tuple], ids=["list", "tuple"])
def test_a_field_validates_from_a_sequence_into_a_sorted_sequence_as_given(
    given: Callable[[Iterable[object]], Sequence[object]],
) -> None:
    ads = (Ad(id=AdId(2)), Ad(id=AdId(1)))

    campaign = Campaign.model_validate({"ads": given(ad.model_dump() for ad in ads)})

    assert (type(campaign.ads), campaign.ads) == (SortedSequence, ads)


def test_a_field_dumps_as_a_json_array() -> None:
    campaign = Campaign(ads=SortedSequence((Ad(id=AdId(1)),)))

    assert campaign.model_dump_json() == '{"ads":[{"id":1}]}'


def test_a_json_array_validates_back_into_the_field() -> None:
    campaign = Campaign(ads=SortedSequence((Ad(id=AdId(1)), Ad(id=AdId(2)))))

    assert Campaign.model_validate_json(campaign.model_dump_json()) == campaign
