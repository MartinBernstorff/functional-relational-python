import pydantic

from functional_relational_python import steps


class ViewName(pydantic.RootModel[str], frozen=True):
    @staticmethod
    def fake() -> ViewName:
        return ViewName("CompanySummary")


class View(pydantic.BaseModel, frozen=True):
    name: ViewName
    source: type[pydantic.BaseModel]
    step: steps.Step
