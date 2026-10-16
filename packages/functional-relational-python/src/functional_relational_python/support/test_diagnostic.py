import pydantic
import pytest
from safe_result import Err, Ok

from functional_relational_python.support.diagnostic import (
    Diagnostic,
    DiagnosticMessage,
    DiagnosticsError,
)


class Described(pydantic.BaseModel, Diagnostic, frozen=True):
    message: DiagnosticMessage

    def describe(self) -> DiagnosticMessage:
        return self.message


class Value(pydantic.RootModel[str], frozen=True):
    pass


def test_the_error_message_has_one_line_per_diagnostic() -> None:
    first = Described(message=DiagnosticMessage.fake())
    second = Described(message=DiagnosticMessage(f"{first.message.root} again"))

    error = DiagnosticsError((first, second))

    assert str(error).splitlines() == [first.message.root, second.message.root]


def test_or_ok_returns_every_diagnostic_and_does_not_compute_the_value() -> None:
    diagnostics = (
        Described(message=DiagnosticMessage.fake()),
        Described(message=DiagnosticMessage(f"{DiagnosticMessage.fake().root}!")),
    )

    def value() -> Value:
        pytest.fail("value must not be computed when there are diagnostics")

    match DiagnosticsError.or_ok(diagnostics, value):
        case Err(error):
            assert error.diagnostics == diagnostics
        case Ok(computed):
            pytest.fail(f"expected diagnostics, got {computed}")


def test_or_ok_returns_the_value_when_there_are_no_diagnostics() -> None:
    value = Value("generated")

    assert DiagnosticsError.or_ok((), lambda: value) == Ok(value)
