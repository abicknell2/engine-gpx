# type: ignore
import json
import math
import os
from statistics import mean
from test.shared_functions import (discover_test_cases, ensure_data_directory_exists, run_model_analysis)
from typing import Generator, Union

import pytest

from utils import logger
from utils.types.shared import AcceptedTypes

# Get the absolute path of the current script's directory
current_dir: str = os.path.abspath(os.path.dirname(__file__))

# Move up one level to set the root folder
REPO_ROOT: str = os.path.dirname(current_dir)

# Define the data directory containing JSON files
DATA_DIR: str = os.path.join(REPO_ROOT, "test", "data", "rate_ramps")

# Define a type alias for a plot entry.
PlotDict = dict[str, Union[float, list[float], list[dict[str, Union[float, str]]]]]


@pytest.fixture(scope="function", autouse=True)
def setup_data_directory() -> Generator[None, None, None]:
    """Fixture to ensure DATA_DIR exists before tests."""
    ensure_data_directory_exists(DATA_DIR)
    yield  # Run the test


@pytest.mark.parametrize("input_json,expected_json", discover_test_cases(DATA_DIR))
def test_rate_ramp_results(input_json: str, expected_json: str) -> None:
    """
    Validate rate ramp results against expected outputs.
    Reports average error for mismatches in 'range' entries; others are reported in detail.
    """
    with open(input_json, "r") as f:
        input_model: AcceptedTypes = json.load(f)
    with open(expected_json, "r") as f:
        expected_output: AcceptedTypes = json.load(f)

    output: AcceptedTypes = run_model_analysis(input_model)
    if output is None:
        pytest.fail("run_model_analysis returned None unexpectedly.")

    errors = []
    range_diffs = []

    output_plots = output.get("plot", [])
    expected_plots = expected_output.get("plot", [])

    if not isinstance(output_plots, list) or not isinstance(expected_plots, list):
        pytest.fail("'plot' key missing or not a list in one of the outputs.")

    MAX_ALLOWED_DIFF = 50
    if abs(len(output_plots) - len(expected_plots)) > MAX_ALLOWED_DIFF:
        raise AssertionError(
            f"❌ plot length mismatch in {os.path.basename(input_json)}: "
            f"expected {len(expected_plots)}, got {len(output_plots)} (allowed diff: ±{MAX_ALLOWED_DIFF})"
        )

    for i, (out_plot, exp_plot) in enumerate(zip(output_plots, expected_plots)):
        # rate
        if not math.isclose(float(out_plot.get("rate", 0)), float(exp_plot.get("rate", 0)), rel_tol=1e-6):
            errors.append(f"plot[{i}]['rate'] mismatch: expected {exp_plot.get('rate')}, got {out_plot.get('rate')}")

        # range
        if "range" in exp_plot:
            if "range" not in out_plot:
                errors.append(f"plot[{i}]['range'] missing from output")
            elif len(out_plot["range"]) != len(exp_plot["range"]):
                errors.append(
                    f"plot[{i}]['range'] length mismatch: expected {len(exp_plot['range'])}, got {len(out_plot['range'])}"
                )
            else:
                for j, (o_val, e_val) in enumerate(zip(out_plot["range"], exp_plot["range"])):
                    if not math.isclose(float(o_val), float(e_val), rel_tol=1e-6):
                        range_diffs.append(abs(float(o_val) - float(e_val)))

        # resources
        out_resources = out_plot.get("resources", [])
        exp_resources = exp_plot.get("resources", [])

        if len(out_resources) != len(exp_resources):
            errors.append(
                f"plot[{i}]['resources'] length mismatch: expected {len(exp_resources)}, got {len(out_resources)}"
            )
        else:
            for k, (o_res, e_res) in enumerate(zip(out_resources, exp_resources)):
                for key in e_res:
                    if key not in o_res:
                        errors.append(f"plot[{i}]['resources'][{k}]['{key}'] missing in output")
                        continue
                    o_val = o_res[key]
                    e_val = e_res[key]
                    if isinstance(e_val, float):
                        if not math.isclose(float(o_val), float(e_val), rel_tol=1e-6):
                            errors.append(
                                f"plot[{i}]['resources'][{k}]['{key}'] mismatch: expected {e_val}, got {o_val}"
                            )
                    else:
                        if o_val != e_val:
                            errors.append(
                                f"plot[{i}]['resources'][{k}]['{key}'] mismatch: expected {e_val}, got {o_val}"
                            )

        # Total Unit Cost
        o_cost = out_plot.get("Total Unit Cost", 0)
        e_cost = exp_plot.get("Total Unit Cost", 0)
        if not math.isclose(float(o_cost), float(e_cost), rel_tol=1e-4):
            errors.append(f"plot[{i}]['Total Unit Cost'] mismatch: expected {e_cost}, got {o_cost}")

    if errors:
        raise AssertionError(
            f"❌ Mismatches found in rate ramp results for {os.path.basename(input_json)}:\n  - "
            + "\n  - ".join(errors)
        )

    if range_diffs:
        avg_range_error = mean(range_diffs)
        raise AssertionError(
            f"❌ Average absolute error in plot[*]['range'] for {os.path.basename(input_json)}: "
            f"{avg_range_error:.6f} (based on {len(range_diffs)} mismatches)"
        )

    logger.info(f"✅ Test passed for: {input_json}")


def test_missing_data_dir() -> None:
    """Test if FileNotFoundError is raised when DATA_DIR does not exist."""
    temp_path: str | None = None

    if os.path.exists(DATA_DIR):
        temp_path = DATA_DIR + "_backup"
        os.rename(DATA_DIR, temp_path)

    try:
        with pytest.raises(FileNotFoundError, match="Could not find the data directory"):
            ensure_data_directory_exists(DATA_DIR)
    finally:
        if temp_path:
            os.rename(temp_path, DATA_DIR)
