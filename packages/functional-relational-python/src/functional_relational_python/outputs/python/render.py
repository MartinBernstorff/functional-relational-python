"""Renders a whole PythonPlan: its View models and, when planned, its mapping
functions, as one set of generated files.

The files are written and checked together, so writing them never deletes
another part of the same plan as no longer generated.
"""

from functional_relational_python.outputs.python.mappings import MappingsRenderer
from functional_relational_python.outputs.python.models import ModelsRenderer
from functional_relational_python.outputs.python.plan import PythonPlan
from functional_relational_python.outputs.python.symbols import FirstParty
from functional_relational_python.support.generated_files import GeneratedFiles


class PythonRenderer:
    """Writes the source of everything a PythonPlan holds."""

    @staticmethod
    def render(plan: PythonPlan, first_party: FirstParty) -> GeneratedFiles:
        """The model modules, then the mapping function modules. Their imports
        of `first_party` packages, and of the packages generated here, are first
        party."""
        generated = first_party.with_generated(
            (*plan.model_modules(), *plan.function_modules())
        )
        return GeneratedFiles(
            files=(
                *ModelsRenderer.files(plan, generated),
                *MappingsRenderer.files(plan, generated),
            )
        )
