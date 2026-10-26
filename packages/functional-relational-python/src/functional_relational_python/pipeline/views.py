import pydantic

from functional_relational_python.pipeline import steps


class ViewName(pydantic.RootModel[str], frozen=True):
    @staticmethod
    def fake() -> ViewName:
        return ViewName("CompanySummary")


class View(pydantic.BaseModel, frozen=True):
    """A read model: a name, a Source and a pipeline of steps over it.

    `base` is the class its generated model subclasses. It may be generic, e.g.
    `PutModel[CompanyId]`. A kept field the base declares is not re-declared.
    """

    name: ViewName
    source: type[pydantic.BaseModel]
    step: steps.Step
    base: type[pydantic.BaseModel] = pydantic.BaseModel
