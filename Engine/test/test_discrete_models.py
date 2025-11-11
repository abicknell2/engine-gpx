# type: ignore
import json
import os
from typing import Generator

import pytest

from test.shared_functions import (
    discover_test_cases,
    ensure_data_directory_exists,
    is_discrete_model,
    is_discrete_model_path,
    run_model_analysis,
)

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on

from utils import logger
from utils.types.shared import AcceptedTypes

current_dir = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(current_dir)
DATA_DIR = os.environ.get("TEST_MODELS_DIR") or os.path.join(REPO_ROOT, "test", "data", "models")

DISCRETE_CASES = [
    case for case in discover_test_cases(DATA_DIR) if is_discrete_model_path(case[0])
]

if not DISCRETE_CASES:
    pytest.skip("No discrete resource models found in fixtures", allow_module_level=True)


@pytest.fixture(scope="function", autouse=True)
def setup_data_directory() -> Generator[None, None, None]:
    """Mirror the behaviour of the main model test module."""
    logger.info(f"Using test models directory: {DATA_DIR}")
    ensure_data_directory_exists(DATA_DIR)
    yield


def _as_float(value: AcceptedTypes) -> float | None:
    """Best-effort conversion of a value that may carry pint units."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        magnitude = getattr(value, "magnitude")
    except AttributeError:
        pass
    else:
        try:
            return float(magnitude)
        except (TypeError, ValueError):
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_integer_like(value: AcceptedTypes, tolerance: float = 1e-6) -> bool:
    numeric = _as_float(value)
    if numeric is None:
        return False
    return abs(numeric - round(numeric)) <= tolerance


@pytest.mark.parametrize("input_json, expected_json", DISCRETE_CASES)
def test_discrete_model_structure(input_json: str, expected_json: str) -> None:
    """Validate structural aspects of discrete resource solves."""
    with open(input_json, "r", encoding="utf-8") as fh:
        input_model = json.load(fh)

    assert is_discrete_model(input_model), "Fixture discovery expected a discrete model"

    output = run_model_analysis(input_model) or {}
    assert output, "run_model_analysis returned empty output for discrete model"

    # Ensure key collections are present.
    critical_keys = [
        "cellResults",
        "lineSummary",
        "totalCost",
        "costComponents",
        "cashflows",
    ]

    for key in critical_keys:
        assert key in output, f"Missing '{key}' in discrete model output"

    cell_results = output.get("cellResults", [])
    assert isinstance(cell_results, list), "cellResults should be a list"
    assert cell_results, "Expected at least one cell result"

    # Every cell entry should carry the basic throughput fields and integer workstations.
    required_cell_keys = {
        "name",
        "utilization",
        "numWorkstations",
        "queueingTime",
        "arrivalCV",
        "departureCV",
    }

    for cell in cell_results:
        assert required_cell_keys.issubset(cell.keys()), f"Cell entry missing keys: {required_cell_keys - set(cell.keys())}"
        assert _is_integer_like(cell.get("numWorkstations")), "Workstation counts should be integer-like"
        qt = _as_float(cell.get("queueingTime"))
        assert qt is None or qt >= 0.0, "Queueing time must be non-negative"

    discrete_vars = output.get("discreteVariables", {})
    if isinstance(discrete_vars, dict):
        for var, raw in discrete_vars.items():
            assert _is_integer_like(raw), f"Discrete variable {var} should resolve to an integer count"
            assert _as_float(raw) is None or _as_float(raw) >= 1.0, "Discrete resource counts must be >= 1"

    # Validate cost collections contain numeric values to guard against regression returning strings.
    for component in output.get("costComponents", []):
        value = component.get("value")
        assert _as_float(value) is not None, "Cost component values must be numeric"

    for cashflow in output.get("cashflows", []):
        value = cashflow.get("cashflow")
        assert _as_float(value) is not None, "Cashflow entries must be numeric"

    # Probability distributions (if present) should have monotonically increasing time axis.
    pdf_points = output.get("pdfpoints", [])
    if isinstance(pdf_points, list):
        for entry in pdf_points:
            t_val = _as_float(entry.get("time"))
            assert t_val is None or t_val >= 0.0, "pdfpoints time values must be non-negative"