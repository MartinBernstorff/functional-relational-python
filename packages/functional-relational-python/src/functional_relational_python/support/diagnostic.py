"""Diagnostics: the defects a stage finds in the user's Entities and Views.

Every stage reports its defects the same way: one `DiagnosticsError` carrying every
Diagnostic it found, so the shell renders them all with one code path.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import pydantic
from safe_result import Err, Ok, Result


class DiagnosticMessage(pydantic.RootModel[str], frozen=True):
    """What a Diagnostic tells the user, as one line."""

    @staticmethod
    def fake() -> DiagnosticMessage:
        return DiagnosticMessage("CompanySummary: Company has no field 'nmae'.")


class Diagnostic(ABC):
    """A defect in the user's Entities or Views.

    An ABC rather than a Protocol, so Pydantic models can subclass it explicitly.
    """

    @abstractmethod
    def describe(self) -> DiagnosticMessage: ...


class DiagnosticsError[D: Diagnostic](Exception):
    """Every Diagnostic a stage found. There is at least one."""

    def __init__(self, diagnostics: tuple[D, *tuple[D, ...]]) -> None:
        self._diagnostics = diagnostics
        super().__init__(
            "\n".join(diagnostic.describe().root for diagnostic in diagnostics)
        )

    @property
    def diagnostics(self) -> tuple[D, *tuple[D, ...]]:
        return self._diagnostics

    @staticmethod
    def or_ok[E: Diagnostic, T](
        diagnostics: Iterable[E], value: Callable[[], T]
    ) -> Result[T, DiagnosticsError[E]]:
        """An error with every diagnostic, or `value()` when there are none.

        `value` is only called when there are no diagnostics, so it may rely on the
        checks that produced them having passed.
        """
        match tuple(diagnostics):
            case (first, *rest):
                return Err(DiagnosticsError((first, *rest)))
            case _:
                return Ok(value())
