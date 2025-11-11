# type: ignore
import json
import math
import os
from statistics import mean
from test.shared_functions import (
    discover_test_cases,
    ensure_data_directory_exists,
    is_discrete_model,
    run_model_analysis,
)
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on
from utils import logger
from utils.types.shared import AcceptedTypes

# Get the absolute path of the current script's directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Move up one level to set the root folder
REPO_ROOT = os.path.dirname(current_dir)

# Define the data directory containing JSON files
DATA_DIR = os.environ.get("TEST_MODELS_DIR") or os.path.join(REPO_ROOT, "test", "data", "models")

@pytest.fixture(scope="function", autouse=True)
def setup_data_directory() -> Generator[None, None, None]:
    """Fixture to ensure DATA_DIR exists before tests."""
    logger.info(f"Using test models directory: {DATA_DIR}")
    ensure_data_directory_exists(DATA_DIR)
    yield  # Run the test


def compare_values(
    output_val: AcceptedTypes, expected_val: AcceptedTypes, rel_tol: float = 1e-5, abs_tol: float = 1e-4
) -> bool:
    """
    Recursively compare two Python values/structures:
      - Floats are compared with math.isclose() using the given tolerances.
      - Dictionaries are compared key-by-key (recursively).
      - Lists/Tuples are compared element-by-element (recursively).
      - All other types are compared with direct equality (==).

    :param output_val: Value produced by your code under test.
    :param expected_val: Expected/reference value to compare against.
    :param rel_tol: Relative tolerance for floating-point comparisons.
    :param abs_tol: Absolute tolerance for floating-point comparisons.
    :return: True if the two values match within the rules above, False otherwise.

    Examples:
        >>> compare_values(3.1415926535, 3.1415926536, rel_tol=1e-7)
        True
        >>> compare_values({'a': [1, 2, 3.000000001]}, {'a': [1, 2, 3.0]})
        True
        >>> compare_values((1, 2), [1, 2])
        False
    """

    # 1) Floats
    if isinstance(output_val, float) and isinstance(expected_val, float):
        return math.isclose(output_val, expected_val, rel_tol=rel_tol, abs_tol=abs_tol)

    # 2) Dictionaries
    if isinstance(output_val, dict) and isinstance(expected_val, dict):
        # Check if both dicts have the same set of keys
        if set(output_val.keys()) != set(expected_val.keys()):
            return False
        # Recursively compare each value
        for key in output_val:
            if not compare_values(output_val[key], expected_val[key], rel_tol, abs_tol):
                return False
        return True

    # 3) Lists or Tuples
    if isinstance(output_val, (list, tuple)) and isinstance(expected_val, (list, tuple)):
        # Must be the same length
        if len(output_val) != len(expected_val):
            return False
        # Compare element by element
        for o_item, e_item in zip(output_val, expected_val):
            if not compare_values(o_item, e_item, rel_tol, abs_tol):
                return False
        return True

    # 4) Fallback: direct equality
    return output_val == expected_val


@pytest.mark.parametrize("input_json, expected_json", discover_test_cases(DATA_DIR))
def test_correct_results(input_json: str, expected_json: str) -> None:
    """
    Validate model output keys and all nested values as separate 'subtests' and report all mismatches.
    """
    with open(input_json) as f:
        input_model = json.load(f)
    if is_discrete_model(input_model):
        pytest.skip(
            "Discrete resource solves yield non-deterministic allocations; "
            "see test_discrete_solve.py for structural validation."
        )
    with open(expected_json) as f:
        expected_output = json.load(f)

    output = run_model_analysis(input_model) or {}

    if output is None:
        pytest.fail("run_model_analysis returned None unexpectedly.")

    required_keys = [
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

    errors = []

    for key in required_keys:
        if key not in output:
            errors.append(f"[{key}] missing from output")
            continue
        if key not in expected_output:
            errors.append(f"[{key}] missing from expected output")
            continue

        out_val = output[key]
        exp_val = expected_output[key]

        if not out_val and not exp_val:
            continue  # skip empty dict/list

        # Dict comparison
        if isinstance(out_val, dict) and isinstance(exp_val, dict):
            for sub_key in exp_val:
                if sub_key not in out_val:
                    errors.append(f"[{key}][{sub_key}] missing from output")
                    continue
                tol_args = {"rel_tol": 5e-2, "abs_tol": 1e-3} if sub_key == "cellVariationSensitivity" else {}
                if not compare_values(out_val[sub_key], exp_val[sub_key], **tol_args):
                    errors.append(f"[{key}][{sub_key}] mismatch: expected {exp_val[sub_key]}, got {out_val[sub_key]}")

        # List of dicts comparison
        elif isinstance(out_val, list) and isinstance(exp_val, list):
            if len(out_val) != len(exp_val):
                errors.append(f"[{key}] length mismatch: expected {len(exp_val)}, got {len(out_val)}")
                continue
            for i, (o, e) in enumerate(zip(out_val, exp_val)):
                if not isinstance(o, dict) or not isinstance(e, dict):
                    errors.append(f"[{key}][{i}] expected dict, got {type(o)} and {type(e)}")
                    continue
                for sub_key in e:
                    if sub_key not in o:
                        errors.append(f"[{key}][{i}][{sub_key}] missing from output")
                        continue
                    if sub_key in {"queueingTime", "cellFlowTime"}:
                        if not math.isclose(o[sub_key], e[sub_key], rel_tol=1e-5, abs_tol=1e-2):
                            errors.append(
                                f"[{key}][{i}][{sub_key}] mismatch: expected approx {e[sub_key]}, got {o[sub_key]}"
                            )
                    else:
                        tol_args = {"rel_tol": 5e-2, "abs_tol": 1e-3} if sub_key == "cellVariationSensitivity" else {}
                        if not compare_values(o[sub_key], e[sub_key], **tol_args):
                            errors.append(f"[{key}][{i}][{sub_key}] mismatch: expected {e[sub_key]}, got {o[sub_key]}")
        else:
            if not compare_values(out_val, exp_val):
                errors.append(f"[{key}] mismatch: expected {exp_val}, got {out_val}")

    if errors:
        error_report = "\n  - ".join(errors)
        raise AssertionError(f"❌ Mismatches found in {os.path.basename(input_json)}:\n  - {error_report}")

    logger.info(f"✅ All checks passed for: {input_json}")


@pytest.mark.parametrize("input_json,expected_json", discover_test_cases(DATA_DIR))
def test_optional_keys(input_json: str, expected_json: str) -> None:
    """
    Validate optional keys (unitCostBreakout, tooling, etc.) if they exist in the output.
    Each comparison is treated independently with collected mismatches.
    """
    with open(input_json, "r") as f:
        input_model = json.load(f)
    if is_discrete_model(input_model):
        pytest.skip(
            "Discrete resource solves yield non-deterministic allocations; "
            "see test_discrete_solve.py for structural validation."
        )
    with open(expected_json, "r") as f:
        expected_output = json.load(f)

    output = run_model_analysis(input_model) or {}

    if output is None:
        pytest.fail("run_model_analysis returned None unexpectedly.")

    optional_keys = ["unitCostBreakout", "tooling", "feederLines", "cashflows"]
    errors = []

    for key in optional_keys:
        if key not in output:
            continue  # optional, skip silently
        if key not in expected_output:
            errors.append(f"[{key}] present in output but missing from expected output")
            continue

        out_val = output[key]
        exp_val = expected_output[key]

        # Handle list-of-dicts
        if isinstance(out_val, list) and isinstance(exp_val, list):
            if len(out_val) != len(exp_val):
                errors.append(f"[{key}] length mismatch: expected {len(exp_val)}, got {len(out_val)}")
                continue
            for i, (o, e) in enumerate(zip(out_val, exp_val)):
                if not isinstance(o, dict) or not isinstance(e, dict):
                    errors.append(f"[{key}][{i}] expected dicts, got {type(o)} and {type(e)}")
                    continue
                for sub_key in e:
                    if sub_key not in o:
                        errors.append(f"[{key}][{i}][{sub_key}] missing from output")
                        continue
                    out_item = o[sub_key]
                    exp_item = e[sub_key]
                    if isinstance(out_item, float) and isinstance(exp_item, float):
                        if not math.isclose(out_item, exp_item, rel_tol=1e-5, abs_tol=1e-4):
                            errors.append(f"[{key}][{i}][{sub_key}] mismatch: expected {exp_item}, got {out_item}")
                    else:
                        if out_item != exp_item:
                            errors.append(f"[{key}][{i}][{sub_key}] mismatch: expected {exp_item}, got {out_item}")

        # Handle float keys
        elif isinstance(out_val, float) and isinstance(exp_val, float):
            if not math.isclose(out_val, exp_val, rel_tol=1e-5, abs_tol=1e-4):
                errors.append(f"[{key}] float mismatch: expected {exp_val}, got {out_val}")

        # Handle anything else
        else:
            if out_val != exp_val:
                errors.append(f"[{key}] mismatch: expected {exp_val}, got {out_val}")

    if errors:
        raise AssertionError(
            f"❌ Mismatches found in optional keys for {os.path.basename(input_json)}:\n  - " + "\n  - ".join(errors)
        )

    logger.info(f"✅ Optional keys test passed for: {input_json}")


@pytest.mark.parametrize("input_json,expected_json", discover_test_cases(DATA_DIR))
def test_probabilities(input_json: str, expected_json: str) -> None:
    """
    Test probabilities key within the JSON output against expected values.
    Collects all mismatches and reports them as structured failures.
    """
    with open(input_json, "r") as f:
        input_model = json.load(f)
    if is_discrete_model(input_model):
        pytest.skip(
            "Discrete resource solves yield non-deterministic allocations; "
            "see test_discrete_solve.py for structural validation."
        )
    with open(expected_json, "r") as f:
        expected_output = json.load(f)

    output = run_model_analysis(input_model) or {}

    if output is None:
        pytest.fail("run_model_analysis returned None unexpectedly.")

    errors = []

    if "probabilities" not in output:
        if "probabilities" in expected_output:
            errors.append("'probabilities' expected in output but missing.")
        else:
            return  # optional, skip
    else:
        flow_out = output["probabilities"].get("flowTime", {})
        flow_exp = expected_output["probabilities"].get("flowTime", {})

        # Check for key presence
        missing_keys = set(flow_exp.keys()) - set(flow_out.keys())
        extra_keys = set(flow_out.keys()) - set(flow_exp.keys())
        if missing_keys:
            errors.append(f"Missing keys in flowTime: {missing_keys}")
        if extra_keys:
            errors.append(f"Unexpected keys in flowTime: {extra_keys}")

        # Quantile values
        for key in ["low", "q1", "median", "q3", "high"]:
            if key in flow_out and key in flow_exp:
                if not math.isclose(flow_out[key], flow_exp[key], rel_tol=1e-5, abs_tol=1e-6):
                    errors.append(f"Mismatch at [{key}]: expected {flow_exp[key]}, got {flow_out[key]}")

        # boxPlot
        if "boxPlot" in flow_out and "boxPlot" in flow_exp:
            if len(flow_out["boxPlot"]) != len(flow_exp["boxPlot"]):
                errors.append(
                    f"boxPlot length mismatch: expected {len(flow_exp['boxPlot'])}, got {len(flow_out['boxPlot'])}"
                )
            else:
                for i, (box_o, box_e) in enumerate(zip(flow_out["boxPlot"], flow_exp["boxPlot"])):
                    if not isinstance(box_o, dict):
                        errors.append(f"boxPlot[{i}] should be a dict, got {type(box_o)}")
                        continue
                    if "time" in box_o and "time" in box_e:
                        if len(box_o["time"]) != len(box_e["time"]):
                            errors.append(f"boxPlot[{i}]['time'] length mismatch")
                        else:
                            for j, (t_o, t_e) in enumerate(zip(box_o["time"], box_e["time"])):
                                if not math.isclose(t_o, t_e, rel_tol=1e-5, abs_tol=1e-6):
                                    errors.append(f"boxPlot[{i}]['time'][{j}] mismatch: expected {t_e}, got {t_o}")
                    elif "time" in box_e:
                        errors.append(f"boxPlot[{i}] missing 'time' key in output")

        # Unit
        if flow_out.get("unit") != flow_exp.get("unit"):
            errors.append(f"Mismatch in 'unit': expected {flow_exp.get('unit')}, got {flow_out.get('unit')}")

    if errors:
        raise AssertionError(
            f"❌ Mismatches found in probabilities for {os.path.basename(input_json)}:\n  - " + "\n  - ".join(errors)
        )

    logger.info(f"✅ Test passed for probabilities in: {input_json}")


@pytest.mark.parametrize("input_json,expected_json", discover_test_cases(DATA_DIR))
def test_costcomponents(input_json: str, expected_json: str) -> None:
    """
    Test costComponents with numerical precision handling.
    Collects and reports all mismatches across the list of components.
    """
    with open(input_json, "r") as f:
        input_model = json.load(f)
    if is_discrete_model(input_model):
        pytest.skip(
            "Discrete resource solves yield non-deterministic allocations; "
            "see test_discrete_solve.py for structural validation."
        )
    with open(expected_json, "r") as f:
        expected_output = json.load(f)

    output = run_model_analysis(input_model) or {}
    errors = []

    output_costcomponents = output.get("costComponents", [])
    expected_costcomponents = expected_output.get("costComponents", [])

    if len(output_costcomponents) != len(expected_costcomponents):
        errors.append(
            f"costComponents length mismatch: expected {len(expected_costcomponents)}, got {len(output_costcomponents)}"
        )

    for i, (cost_output, cost_expected) in enumerate(zip(output_costcomponents, expected_costcomponents)):
        if cost_output.get("name") != cost_expected.get("name"):
            errors.append(
                f"costComponents[{i}]['name'] mismatch: expected '{cost_expected.get('name')}', got '{cost_output.get('name')}'"
            )

        if not math.isclose(
                cost_output.get("value", 0),
                cost_expected.get("value", 0),
                rel_tol=1e-6,
                abs_tol=2.0,
        ):
            errors.append(
                f"costComponents[{i}]['value'] mismatch: expected {cost_expected.get('value')}, got {cost_output.get('value')}"
            )

    if errors:
        raise AssertionError(
            f"❌ Mismatches found in costComponents for {os.path.basename(input_json)}:\n  - " + "\n  - ".join(errors)
        )

    logger.info(f"✅ Test passed for costComponents in: {input_json}")


@pytest.mark.parametrize("input_json,expected_json", discover_test_cases(DATA_DIR))
def test_pdfpoints(input_json: str, expected_json: str) -> None:
    """
    Test pdfPoints output.
    Reports average absolute error for each mismatched key (e.g. 'probability', 'time').
    """
    with open(input_json, "r") as f:
        input_model = json.load(f)
    if is_discrete_model(input_model):
        pytest.skip(
            "Discrete resource solves yield non-deterministic allocations; "
            "see test_discrete_solve.py for structural validation."
        )
    with open(expected_json, "r") as f:
        expected_output = json.load(f)

    output = run_model_analysis(input_model) or {}

    output_pdfpoints = output.get("pdfpoints", [])
    expected_pdfpoints = expected_output.get("pdfpoints", [])

    key_diffs = {
        "probability": [],
        "time": [],
    }

    if len(output_pdfpoints) != len(expected_pdfpoints):
        raise AssertionError(
            f"❌ pdfPoints length mismatch in {os.path.basename(input_json)}: "
            f"expected {len(expected_pdfpoints)}, got {len(output_pdfpoints)}"
        )

    for pdf_out, pdf_exp in zip(output_pdfpoints, expected_pdfpoints):
        # probability
        p_out = pdf_out.get("probability", 0)
        p_exp = pdf_exp.get("probability", 0)
        if isinstance(p_out, float) and isinstance(p_exp, float):
            if not math.isclose(p_out, p_exp, rel_tol=1e-5, abs_tol=1e-9):
                key_diffs["probability"].append(abs(p_out - p_exp))

        # time
        t_out = pdf_out.get("time", 0)
        t_exp = pdf_exp.get("time", 0)
        if isinstance(t_out, float) and isinstance(t_exp, float):
            if not math.isclose(t_out, t_exp, rel_tol=1e-4, abs_tol=1e-2):
                key_diffs["time"].append(abs(t_out - t_exp))

        # unit
        u_out = pdf_out.get("unit")
        u_exp = pdf_exp.get("unit")
        if u_out != u_exp:
            if not key_diffs.get("unit"):
                key_diffs["unit"] = []
            key_diffs["unit"].append((u_out, u_exp))

    summary_lines = []
    for key, diffs in key_diffs.items():
        if key == "unit":
            if diffs:
                mismatches = len(diffs)
                mismatched_units = {f"{exp} != {out}" for out, exp in diffs}
                summary_lines.append(f"• unit: {mismatches} mismatches ({'; '.join(mismatched_units)})")
        elif diffs:
            summary_lines.append(f"• {key}: {mean(diffs):.10f} average error ({len(diffs)} mismatches)")

    if summary_lines:
        raise AssertionError(
            f"❌ Average mismatches in pdfPoints for {os.path.basename(input_json)}:\n  " + "\n  ".join(summary_lines)
        )

    logger.info(f"✅ Test passed for pdfPoints in: {input_json}")


def test_missing_data_dir() -> None:
    """Test if FileNotFoundError is raised when DATA_DIR does not exist."""
    temp_path: str | None = None

    if os.path.exists(DATA_DIR):  # pragma: no cover
        temp_path = DATA_DIR + "_backup"
        os.rename(DATA_DIR, temp_path)

    try:
        with pytest.raises(FileNotFoundError, match="Could not find the data directory"):
            ensure_data_directory_exists(DATA_DIR)
    finally:
        if temp_path:  # pragma: no cover
            os.rename(temp_path, DATA_DIR)


def test_run_model_analysis_exception_handling() -> None:
    """Test if run_model_analysis() correctly handles exceptions and logs them."""
    broken_input = {"model": {"type": "this_will_cause_an_error"}}

    # Mock logger.exception to track calls
    with patch("utils.logger._get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        output: AcceptedTypes = run_model_analysis(broken_input)

        # Ensure the function returns {} when an exception occurs
        assert output == {}, "Expected function to return {} on exception"

        # Verify that logger.error was called
        mock_logger.error.assert_called()

        # Extract all arguments passed to logger.error
        log_args = mock_logger.error.call_args[0]

        # Convert all arguments to strings for comparison
        logged_messages = " ".join(str(arg) for arg in log_args)
        print('logged_messages: ', logged_messages)

        # Check if the expected message is present in the logs
        assert "An exception occurred: Missing required key: manufacturing." in logged_messages, "Expected exception message not logged"