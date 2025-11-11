# type: ignore
import json
import os

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on
from test.shared_functions import (ensure_data_directory_exists, run_model_analysis)
from typing import Generator

from gpkit import ureg
import pytest

from utils import logger

# Get the absolute path of the current script's directory
current_dir = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(current_dir))
DATA_DIR = os.path.join(REPO_ROOT, "test", "data", "math_cases")

# Define the specific JSON file for these tests
INPUT_JSON_PATH = os.path.join(DATA_DIR, "first_second_process_time_dynamic_constraint.json")


@pytest.fixture(scope="function", autouse=True)
def setup_data_directory() -> Generator[None, None, None]:
    """Fixture to ensure DATA_DIR exists before tests."""
    ensure_data_directory_exists(DATA_DIR)
    yield  # Run the test


def test_first_process_time_alternate() -> None:
    with open(INPUT_JSON_PATH, "r") as f:
        input_model = json.load(f)

    output = run_model_analysis(input_model)

    # Locate the specific variable in the list of allVariables.
    first_process = None
    for variable in output.get("allVariables", []):
        if variable.get("name") == "First process Time":
            first_process = variable
            break

    assert first_process is not None, ("Variable 'First process Time' was not found in the output.")

    # Define our parameters with their units
    fp_rate = 1
    fr_dimension = 100
    fp_rate_unit = "min/pct"  # 1 minute per percent
    fr_dimension_unit = "pct"  # percentage unit
    first_process_expected_unit = "minute"

    # Define quantities with units using gpkit's ureg
    fp_rate_qty = ureg.Quantity(fp_rate, fp_rate_unit)
    fr_dimension_qty = ureg.Quantity(fr_dimension, fr_dimension_unit)

    # Calculate the result: fp_rate_qty * fr_dimension_qty.
    # Since fp_rate_qty is in min/% and fr_dimension_qty is in %,
    # the percentage units cancel out, leaving a result in minutes.
    result = fp_rate_qty * fr_dimension_qty

    expected_value = result.magnitude  # should be 100.0 (1 min/% * 100% = 100 min)
    expected_unit = first_process_expected_unit  # expecting "minute"

    # Verify the computed value and unit.
    assert first_process["value"] == expected_value, (
        f"Expected value {expected_value} but got {first_process['value']}."
    )
    assert first_process["unit"] == expected_unit, (f"Expected unit '{expected_unit}' but got {first_process['unit']}.")

    logger.info(f"✅ Test passed for: {INPUT_JSON_PATH}")


def test_second_process_time() -> None:
    with open(INPUT_JSON_PATH, "r") as f:
        input_model = json.load(f)

    output = run_model_analysis(input_model)

    # Locate the specific variable in the list of allVariables.
    second_process = None
    for variable in output.get("allVariables", []):
        if variable.get("name") == "Second Process Cell Process Time":
            second_process = variable
            break

    assert second_process is not None, ("Variable 'Second Process Cell Process Time' was not found in the output.")

    # Define our parameters with their units
    sp_rate = 40
    fr_dimension = 100
    sp_rate_unit = "pct/hour"  # equivalent to "%/hr"
    fr_dimension_unit = "pct"  # equivalent to "%"
    second_process_expected_unit = "minute"

    # Define quantities with units using gpkit's ureg
    sp_rate_qty = ureg.Quantity(sp_rate, sp_rate_unit)
    fr_dimension_qty = ureg.Quantity(fr_dimension, fr_dimension_unit)

    # Calculate the result: (1 / sp_rate_qty) * fr_dimension_qty.
    # This yields a time quantity in hours because:
    #   1 / (pct/hour) -> hour/pct, and multiplying by pct cancels the percent units.
    result = (1 / sp_rate_qty) * fr_dimension_qty

    # Convert the result from hours to minutes
    result_in_min = result.to("minute")

    expected_value = result_in_min.magnitude  # should be 150.0
    expected_unit = second_process_expected_unit  # expecting "minute"

    # Verify the computed value and unit.
    assert second_process["value"] == expected_value, (
        f"Expected value {expected_value} but got {second_process['value']}."
    )
    assert second_process["unit"] == expected_unit, (
        f"Expected unit '{expected_unit}' but got {second_process['unit']}."
    )

    logger.info(f"✅ Test passed for: {INPUT_JSON_PATH}")


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
