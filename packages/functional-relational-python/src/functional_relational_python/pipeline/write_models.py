import pydantic

from functional_relational_python.pipeline import steps
from functional_relational_python.pipeline.views import ViewName


class WriteModelName(pydantic.RootModel[str], frozen=True):
    @staticmethod
    def fake() -> WriteModelName:
        return WriteModelName("CompanyCreate")


class WriteModel(pydantic.BaseModel, frozen=True):
    """A write model, such as the body of a create or an update request.

    A sibling of View: a name and a Query of the same steps. A WriteModel
    generates only its model, no mapping function. An update write model is built by
    extending the create query with `Query.then`.

    `base` is the class its generated model subclasses, as for a View.
    """

    name: WriteModelName
    query: steps.Query
    base: type[pydantic.BaseModel] = pydantic.BaseModel


type DefinitionName = ViewName | WriteModelName
"""The name of a View or a WriteModel, which is also the name of its model."""
