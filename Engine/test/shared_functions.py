# type: ignore
import json
import os

import api.oneshot as oneshot

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on
from utils import logger
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import replace_units_to_process


def discover_test_cases(data_dir: str) -> list[tuple[str, str]]:
    """Find valid input JSON and expected _results.json files across subfolders."""
    test_cases = []
    for root, _, files in os.walk(data_dir):
        json_files = set(f for f in files if f.endswith(".json"))

        for file in json_files:
            if file.endswith("_results.json"):
                input_file = file.replace("_results.json", ".json")
                if input_file in json_files:
                    test_cases.append((
                        os.path.join(root, input_file),
                        os.path.join(root, file),
                    ))
    return test_cases


def ensure_data_directory_exists(data_dir: str) -> None:
    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Could not find the data directory: {data_dir}. Ensure the project structure is correct.",
        )


def is_discrete_model(model: AcceptedTypes) -> bool:
    """Return True when the supplied model input enables discrete resource solves."""
    if not isinstance(model, dict):
        return False

    candidate_sections: list[dict[str, AcceptedTypes] | None] = []

    # Primary location for finance data inside test fixtures.
    inner_model = model.get("model") if isinstance(model.get("model"), dict) else None
    if isinstance(inner_model, dict):
        candidate_sections.append(inner_model.get("finance"))

    # Some callers may already pass the inner model dictionary.
    candidate_sections.append(model.get("finance") if isinstance(model.get("finance"), dict) else None)

    for section in candidate_sections:
        if isinstance(section, dict) and section.get("discretizedResources"):
            return True

    return False


def is_discrete_model_path(path: str) -> bool:
    """Convenience helper that inspects a JSON file on disk for discrete solves."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False

    return is_discrete_model(data)


def run_model_analysis(input_model: AcceptedTypes) -> AcceptedTypes:
    input_model = replace_units_to_process(input_model)
    if not isinstance(input_model, dict):
        return {}

    output = {}
    try:
        model_type: list[str] = input_model.get("model", {}).get("type", [])

        if "rateHike" in input_model:
            logger.info("Run Ramp-Up Analysis")
            output = oneshot.ramp_up_analysis(input_model)
        elif "bestworst" in model_type:
            logger.info("Solving best/worst")
            output = oneshot.new_best_worst(input_model)
        elif 'system' in model_type or 'variants' in model_type:
            logger.info("Solving a system")
            output = oneshot.multi_product(input_model)
        else:
            logger.info("Solving simple model")
            output = oneshot.run_one_shot(input_model)
    except Exception as e:
        logger.exception(e)
        return output

    return output