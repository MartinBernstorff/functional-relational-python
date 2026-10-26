import pydantic

from functional_relational_python.pipeline import steps
from functional_relational_python.pipeline.views import ViewName


class PayloadName(pydantic.RootModel[str], frozen=True):
    @staticmethod
    def fake() -> PayloadName:
        return PayloadName("CompanyCreate")


class Payload(pydantic.BaseModel, frozen=True):
    """A write model, such as the body of a create or an update request.

    A sibling of View: a name, a Source and a pipeline of the same steps. A Payload
    generates only its model, no mapping function. An update payload is built by
    composing steps, e.g. the create pipeline followed by more steps.

    `base` is the class its generated model subclasses, as for a View.
    """

    name: PayloadName
    source: type[pydantic.BaseModel]
    step: steps.Step
    base: type[pydantic.BaseModel] = pydantic.BaseModel


type DefinitionName = ViewName | PayloadName
"""The name of a View or a Payload, which is also the name of its model."""
