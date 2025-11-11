# type: ignore
import json
import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on

from test.shared_functions import (
    discover_test_cases,
    ensure_data_directory_exists,
    run_model_analysis,
)

current_dir = os.path.abspath(os.path.dirname(__file__))
engine_root = os.path.dirname(current_dir)
repo_root = os.path.dirname(engine_root)

MODEL_DIRECTORIES = [
    os.path.join(engine_root, "test", "data", "models"),
    os.path.join(repo_root, "models to test"),
]

MODEL_DIRECTORIES = [
    os.path.abspath(path) for path in MODEL_DIRECTORIES if os.path.isdir(path)
]


@pytest.fixture(scope="function", autouse=True)
def setup_data_directories() -> Generator[None, None, None]:
    """Ensure all model directories exist before tests."""
    for model_dir in MODEL_DIRECTORIES:
        ensure_data_directory_exists(model_dir)
    yield


REQUIRED_KEYS = [
    "processStacks",
    "cellResults",
    "processResults",
    "lineSummary",
    "variableCosts",
    "recurringCosts",
    "totalCost",
    "timeInCellDetail",
    "timeInCellDecomp",
    "productSummary",
]


def _last_nonempty_line(text: str) -> str:
    if not text:
        return ""
    lines = [ln.rstrip() for ln in str(text).splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _last_logged_message(mock_logger: MagicMock) -> str:
    """
    Prefer the last of error/exception/critical, whichever has calls.
    """
    for meth in ("error", "exception", "critical"):
        func = getattr(mock_logger, meth, None)
        if not func or not getattr(func, "call_args_list", None):
            continue
        last_call = func.call_args_list[-1]
        try:
            return " ".join(str(arg) for arg in (last_call.args or []))
        except Exception:
            return "(failed to parse solver error log)"
    return ""


def _case_id(param) -> str:
    # param is the (input_json, _expected_json) tuple produced by discover_test_cases
    try:
        input_json, _ = param
    except Exception:
        return str(param)
    # A compact, readable id: just the model file name without extension
    base = os.path.basename(input_json)
    return os.path.splitext(base)[0]


def _collect_test_cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for model_dir in MODEL_DIRECTORIES:
        cases.extend(discover_test_cases(model_dir))
    return cases


@pytest.mark.parametrize(
    ("input_json", "_expected_json"),
    _collect_test_cases(),
    ids=_case_id,
)
def test_models_solve_only(
    input_json: str,
    _expected_json: str,
    record_property,  # provided by pytest
) -> None:
    """
    Each model is its own test (pytest counts failures individually).
    On failure:
      - show missing required keys
      - show solver detail
      - record a per-model summary (model + last error line) via record_property
        so conftest can print a pretty end-of-run table.
    """
    with open(input_json, "r") as f:
        input_model = json.load(f)

    with patch("utils.logger._get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        raised_exc = None
        try:
            output = run_model_analysis(input_model) or {}
        except Exception as e:  # defensive
            output = {}
            raised_exc = e

        missing = [k for k in REQUIRED_KEYS if k not in output]
        if not missing:
            return  # solved

        # Build "missing from output" bullets
        missing_lines = "\n  - ".join(f"[{k}] missing from output" for k in missing)

        # Prefer actual exception; else last logged error/exception/critical
        if raised_exc is not None:
            solver_detail = f"{raised_exc.__class__.__name__}: {raised_exc}"
        else:
            last_msg = _last_logged_message(mock_logger)
            solver_detail = last_msg or "(no solver error captured)"

        model_name = os.path.basename(input_json)

        # Store summary data on the test report
        last_line = _last_nonempty_line(solver_detail) or "(no solver error captured)"
        record_property("model_name", model_name)
        record_property("last_line", last_line)

        pytest.fail(
            f"❌ Model did not solve for {model_name}.\n\n"
            f"Expected keys missing:\n  - {missing_lines}\n\n"
            f"Solver error/exception:\n{solver_detail}"
        )
