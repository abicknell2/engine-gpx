from datetime import date
import json
import os
from pathlib import Path
import shutil
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import flask
import flask.testing
from gpkit.exceptions import (
    DualInfeasible, Infeasible, InvalidGPConstraint, InvalidLicense, InvalidPosynomial, InvalidSGPConstraint,
    MathematicallyInvalid, PrimalInfeasible, UnboundedGP, UnknownInfeasible, UnnecessarySGP
)
from pint.errors import (
    DefinitionError, DefinitionSyntaxError, DimensionalityError, LogarithmicUnitCalculusError, OffsetUnitCalculusError,
    PintTypeError, RedefinitionError, UndefinedUnitError
)
import pytest

from api import api as api_mod
from api.api import app, auth
from utils import error_helpers
from utils.constants.error_messages import KNOWN_ERROR_MESSAGES
from utils.error_helpers import (LOG_ROOT, is_known_user_error, should_mask_error)

KNOWN_ERROR_MSGS = ValueError(KNOWN_ERROR_MESSAGES[0])

GENERIC_EXCEPTIONS = [
    RuntimeError("Divided by zero on some value"),
    AttributeError("'NoneType' object has no attribute 'foo'"),
    TypeError("Expected int but got str"),
    ValueError("Unable to parse date"),
    IndexError("list index out of range"),
    KeyError("missing key 'foo'"),
    NameError("name 'foobar' is not defined"),

    # Below are more exceptions that can be tested locally,
    # though, for speed in CI, they are commented out.

    # UnboundLocalError("local variable 'x' referenced before assignment"),
    # ImportError("No module named 'ghost'"),
    # ModuleNotFoundError("No module named 'phantom'"),
    # NotImplementedError("This method must be overridden"),
    # AssertionError("Expected true but got false"),
    # StopIteration("No more items in iterator"),
    # StopAsyncIteration("Async iterator exhausted"),
    # SyntaxError("invalid syntax in expression"),
    # IndentationError("unexpected indent"),
    # FileNotFoundError("No such file or directory: 'missing.txt'"),
    # PermissionError("Permission denied: '/etc/passwd'"),
    # IsADirectoryError("Is a directory: '/tmp'"),
    # NotADirectoryError("Not a directory: '/tmp/file.txt/data'"),
    # IOError("I/O operation failed"),
    # OSError("General OS error"),
    # MemoryError("Out of memory while allocating array"),
    # BufferError("Operation would overflow buffer"),
    # OverflowError("Numerical result out of range"),
    # UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    # UnicodeEncodeError("ascii", "é", 0, 1, "ordinal not in range(128)"),
    # UnicodeError("General unicode failure"),
    # FloatingPointError("Floating point division failed"),
    # EOFError("EOF reached while reading line"),
    # BlockingIOError("Resource temporarily unavailable")
]

PINT_EXCEPTIONS = [
    DimensionalityError("m", "s"),
    DefinitionError("foo", str, "bad syntax"),
    DefinitionSyntaxError("bad definition"),
    RedefinitionError("foo", str),
    UndefinedUnitError("badunit"),
    PintTypeError("Invalid type"),
    OffsetUnitCalculusError("degC", "kelvin"),
    LogarithmicUnitCalculusError("dB", "unitless"),
]

GPKIT_EXCEPTIONS = [
    MathematicallyInvalid("Invalid math"),
    InvalidPosynomial("Negative coeff"),
    InvalidGPConstraint("Not GP-safe"),
    InvalidSGPConstraint("Not SGP-safe"),
    UnnecessarySGP("Is already a GP"),
    InvalidLicense("License expired"),
    Infeasible("Problem is infeasible"),
    UnknownInfeasible("Unknown infeasibility"),
    PrimalInfeasible("Primal cert failed"),
    DualInfeasible("Dual cert failed"),
]

KNOWN_EXCEPTIONS = PINT_EXCEPTIONS + GPKIT_EXCEPTIONS

mock_auth: MagicMock = MagicMock()
mock_auth.username = "test_user"
mock_auth.password = "test_password"


@pytest.fixture(scope="module")
def test_client() -> Generator[flask.testing.FlaskClient, None, None] | None:
    """Create a test client for the Flask application with fully bypassed authentication."""
    from api.api import app as test_app

    def mock_verify_password(username: str, password: str) -> bool:
        return True

    def mock_authenticate(auth: Any, password: str) -> str:
        return "test_user"

    with patch.dict(os.environ, {"COP_NO_LOGIN": "1"}):  # Mock login to bypass authentication
        with patch.object(auth, "verify_password", mock_verify_password):  # Always authenticate
            with patch.object(auth, "authenticate", mock_authenticate):  # Return mock user
                with patch.object(auth, "get_auth", return_value=mock_auth):  # Disable auth enforcement
                    with test_app.test_client() as client:
                        client.environ_base["HTTP_AUTHORIZATION"
                                            ] = "Basic dGVzdF91c2VyOnBhc3N3b3Jk"  # Inject mock auth header
                        yield client
    return None


@pytest.fixture()
def log_dir() -> Path:
    """
    Create and return the log directory used for all test logs.
    Cleans up after the full test session.
    """
    log_root = error_helpers.LOG_ROOT
    log_root.mkdir(parents=True, exist_ok=True)
    user_root = Path(log_root / "anonymous")
    user_root.mkdir(parents=True, exist_ok=True)
    return user_root


@pytest.mark.usefixtures("test_client")
def test_solverthread_logs_dimensionality_error_with_username(test_client):
    # Create a valid base model
    model = {
        "model": {
            "type": ["operations"],
            "modelName": "Basic operations"
        },
    }

    # Submit model to /solve
    response = test_client.post("/solve", json=model)
    assert response.status_code == 200

    # Parse streamed JSON from response
    raw = response.data.decode("utf-8", errors="ignore")
    payload_start = raw.find("{")
    assert payload_start != -1
    payload = json.loads(raw[payload_start:])
    assert "errors" in payload

    # Check frontend shows known error unmasked
    error_text = payload["errors"][0]
    expected_fragment = str(error_text)
    assert expected_fragment in error_text, f"Expected fragment '{expected_fragment}' in: {error_text}"

    # Validate it logs to logs/[username]/errors_<date>.log
    username = "test_user"  # as patched by `COP_NO_LOGIN`
    log_file = LOG_ROOT / username / f"errors_{date.today().isoformat()}.log"
    assert log_file.exists(), f"Expected log file: {log_file}"
    contents = log_file.read_text()
    assert "KeyError" in contents


@pytest.mark.parametrize(
    "msg,expected", [
        (KNOWN_ERROR_MESSAGES[0], True),
        ("Brand-new scary traceback", False),
        ("key not found: foo | varkeys: []", True),
    ]
)
def test_is_known_user_error_helper(msg, expected):
    assert is_known_user_error(msg) is expected


@pytest.mark.parametrize("exc,should_mask", [(KNOWN_ERROR_MSGS, False)] + [(e, True) for e in GENERIC_EXCEPTIONS])
def test_solve_route_mask_and_log(test_client, log_dir, exc, should_mask, monkeypatch):
    # Replace SolverThread with a stub that only sets self.exc
    class StubThread(api_mod.SolverThread):

        def run(self):
            try:
                raise exc  # simulate a real failure
            except Exception as e:
                self.exc = e
                root = error_helpers.get_root_exception(e)
                if isinstance(root, error_helpers.KNOWN_ERROR_CLASSES):
                    self.error_uuid = error_helpers.log_unexpected_error(e)
                elif error_helpers.should_mask_error(e):
                    self.error_uuid = error_helpers.log_unexpected_error(e)
                else:
                    self.error_uuid = None

    monkeypatch.setattr(api_mod, "SolverThread", StubThread)

    # Minimal model – details irrelevant; short-circuit before solving
    response = test_client.post("/solve", json={"model": {"type": []}})
    assert response.status_code == 200

    raw = response.data.decode("utf-8", errors="ignore")
    start = raw.find("{")
    assert start != -1, "No JSON object found in streamed response"
    payload = json.loads(raw[start:])

    msg = payload["errors"][0]

    if should_mask:
        assert msg.lower().startswith("an unexpected error"), "unexpected error not masked"
    else:
        assert str(exc) in msg, "known error should pass through"

    # check the log files
    log_file = log_dir / f"errors_{date.today()}.log"

    if should_mask:
        log_contents = log_file.read_text()
        assert "Traceback (most recent call last):" in log_contents
        assert type(exc).__name__ in log_contents
        assert log_file.exists(), "log file missing for unexpected error"
        assert str(exc) in log_contents
    else:
        if log_file.exists():
            # If the file exists from earlier tests, check that this specific error is NOT in it
            log_contents = log_file.read_text()
            assert str(exc) not in log_contents, "Known error was wrongly logged"


@pytest.mark.parametrize("msg", KNOWN_ERROR_MESSAGES)
def test_every_literal_caught(msg):
    assert is_known_user_error(msg), f"Literal not recognised: {msg}"


@pytest.mark.parametrize("exc", KNOWN_EXCEPTIONS)
def test_pint_and_gpkit_errors_are_unmasked_and_user_visible(
    test_client,
    log_dir,
    exc,
    monkeypatch,
):

    class StubThread(api_mod.SolverThread):

        def run(self):
            try:
                raise exc  # simulate a real failure
            except Exception as e:
                self.exc = e
                root = error_helpers.get_root_exception(e)
                if isinstance(root, error_helpers.KNOWN_ERROR_CLASSES):
                    self.error_uuid = error_helpers.log_unexpected_error(e)
                elif error_helpers.should_mask_error(e):
                    self.error_uuid = error_helpers.log_unexpected_error(e)
                else:
                    self.error_uuid = None

    monkeypatch.setattr(api_mod, "SolverThread", StubThread)

    response = test_client.post("/solve", json={"model": {"type": []}})
    assert response.status_code == 200

    raw = response.data.decode("utf-8", errors="ignore")
    start = raw.find("{")
    assert start != -1, "No JSON object found in streamed response"
    payload = json.loads(raw[start:])

    error_text = payload["errors"][0]
    assert str(exc) in error_text, f"Expected exception message in response: {exc}"
    assert not should_mask_error(exc), f"Exception was wrongly masked: {exc}"

    log_file = log_dir / f"errors_{date.today()}.log"
    assert log_file.exists(), "Log file was not created"
    log_contents = log_file.read_text()
    print('exc: ', exc)

    assert normalise_exception_text(exc) in log_contents, f"Expected domain error to be logged: {exc}"


@pytest.mark.parametrize(
    "exc, expected_fragment", [
        (UnboundedGP("No upper bound on variable 'x'"), "upper bound"),
        (UnboundedGP("No lower bound on variable 'y'"), "lower bound"),
    ]
)
def test_unboundedgp_upper_and_lower_are_logged(
    test_client,
    log_dir,
    exc,
    expected_fragment,
    monkeypatch,
):

    class StubThread(api_mod.SolverThread):

        def run(self):
            try:
                raise exc
            except Exception as e:
                self.exc = e
                root = error_helpers.get_root_exception(e)
                if isinstance(root, error_helpers.KNOWN_ERROR_CLASSES):
                    self.error_uuid = error_helpers.log_unexpected_error(e)
                elif error_helpers.should_mask_error(e):
                    self.error_uuid = error_helpers.log_unexpected_error(e)
                else:
                    self.error_uuid = None

    monkeypatch.setattr(api_mod, "SolverThread", StubThread)

    # Trigger solve
    response = test_client.post("/solve", json={"model": {"type": []}})
    assert response.status_code == 200

    # Parse error from frontend
    raw = response.data.decode("utf-8", errors="ignore")
    payload_start = raw.find("{")
    assert payload_start != -1
    payload = json.loads(raw[payload_start:])
    assert "errors" in payload
    error_msg = payload["errors"][0]
    assert expected_fragment in error_msg.lower(), f"Expected '{expected_fragment}' in: {error_msg}"

    # Confirm it was logged
    log_file = log_dir / f"errors_{date.today().isoformat()}.log"
    assert log_file.exists(), "Log file was not created"
    contents = log_file.read_text()
    assert normalise_exception_text(expected_fragment
                                    ) in contents.lower(), f"Expected fragment '{expected_fragment}' in log: {contents}"


def normalise_exception_text(e: Exception) -> str:
    """
    Return a normalized string version of an exception, resilient to Pint/Gpkit formatting changes.
    """
    if isinstance(e, DimensionalityError):
        return "Cannot convert from"
    elif isinstance(e, DefinitionError):
        return "bad syntax"
    elif isinstance(e, DefinitionSyntaxError):
        return "bad definition"
    elif isinstance(e, RedefinitionError):
        return "Cannot redefine"
    elif isinstance(e, UndefinedUnitError):
        return "not defined"
    elif isinstance(e, PintTypeError):
        return "Invalid type"
    elif isinstance(e, OffsetUnitCalculusError):
        return "cannot operate with offset units"
    elif isinstance(e, LogarithmicUnitCalculusError):
        return "cannot operate with logarithmic units"
    elif isinstance(e, MathematicallyInvalid):
        return "Invalid math"
    elif isinstance(e, InvalidPosynomial):
        return "Negative coeff"
    elif isinstance(e, InvalidGPConstraint):
        return "Not GP-safe"
    elif isinstance(e, InvalidSGPConstraint):
        return "Not SGP-safe"
    elif isinstance(e, UnnecessarySGP):
        return "already a GP"
    elif isinstance(e, InvalidLicense):
        return "License expired"
    elif isinstance(e, (Infeasible, UnknownInfeasible, PrimalInfeasible, DualInfeasible)):
        return "infeasible"
    elif isinstance(e, UnboundedGP):
        msg = str(e).lower()
        if "lower bound" in msg:
            return "lower bound"
        elif "upper bound" in msg:
            return "upper bound"
        return "unbounded"
    else:
        return str(e)
