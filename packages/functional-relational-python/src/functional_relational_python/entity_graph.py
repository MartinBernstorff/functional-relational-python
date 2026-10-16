"""Placeholder until MB-177 lands the real EntityGraph."""

import pydantic


class EntityGraph(pydantic.BaseModel, frozen=True):
    entities: tuple[type[pydantic.BaseModel], ...]

    @staticmethod
    def fake() -> EntityGraph:
        return EntityGraph(entities=())
