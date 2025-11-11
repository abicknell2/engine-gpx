import json
import math
import os
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import flask
from flask import Response
import flask.testing
import pytest

from api.api import app, auth  # Flask app / auth objects
from gpx.custom_units import refresh_fx_rates
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import smart_round

CURRENCY_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "data",
    "currency_conversion",
)

symbol_to_iso = {"$": "USD", "£": "GBP", "€": "EUR"}
iso_set = {"USD", "GBP", "EUR"}


@pytest.fixture(scope="module")
def test_client() -> Generator[flask.testing.FlaskClient, None, None] | None:
    mock_auth: MagicMock = MagicMock()
    mock_auth.username = "test_user"
    mock_auth.password = "test_password"

    def mock_verify_password(username: str, password: str) -> bool:
        return True

    def mock_authenticate(auth: Any, password: str) -> str:
        return "test_user"

    def mock_authorize(role: str, user: str, auth: Any) -> bool:
        return True

    with patch.dict(os.environ, {"COP_NO_LOGIN": "1", "SAVE_MODEL_DATA_FOR_TESTS": "1"}):
        with patch.object(auth, "verify_password", mock_verify_password):
            with patch.object(auth, "authenticate", mock_authenticate):
                with patch.object(auth, "authorize", mock_authorize):
                    with patch.object(auth, "get_auth", return_value=mock_auth):
                        with patch.object(auth, "current_user", return_value="test_user"):
                            with patch("flask_httpauth.HTTPBasicAuth.login_required", lambda f: f):
                                with app.test_client() as client:
                                    client.environ_base["HTTP_AUTHORIZATION"] = ("Basic dGVzdF91c2VyOnBhc3N3b3Jk")
                                    yield client
    return None


def _extract_currency(unit_str: str) -> tuple[str | None, str]:
    """
    Return detected currency (ISO) and the remainder of the unit string.
    Examples
    -------
    '$/hr'      -> ('USD', '/hr')
    'USD/count' -> ('USD', '/count')
    'GBP'       -> ('GBP', '')
    """
    unit_str = unit_str.strip()
    # symbol first
    if unit_str and unit_str[0] in symbol_to_iso:
        return symbol_to_iso[unit_str[0]], unit_str[1:].lstrip()
    # iso code as leading token
    for iso in iso_set:
        if unit_str.upper().startswith(iso):
            return iso, unit_str[len(iso):]
    return None, unit_str  # no currency found


@pytest.mark.parametrize(
    "file_name",
    [
        "basic_operations_currency_conversions.json",
        "basic_model_currency_conversion_w_uncertain_var.json",
        "basic_operations_currency_conversion_w_currency_per_unit.json",
        "basic_operations_currency_conversion_w_currency_per_unit_and_uncertainty.json",
        "basic_operations_currency_conversions_w_multiple.json",
        "basic_operations_currency_conversions_w_uncertainty.json",
        "basic_operations_currency_conversions_w_two_point_uncertainty.json",
        "basic_operations_currency_conversions_w_multiple_rate_uncertainty.json",
        "basic_operations_currency_conversions_w_multiple_mixed.json",
    ],
)
def test_currency_cases_api(test_client: flask.testing.FlaskClient, file_name: str) -> None:
    path = os.path.join(CURRENCY_DIR, file_name)
    with open(path, "r") as f:
        input_model: dict[str, AcceptedTypes] = json.load(f)

    response: Response = test_client.post("/solve?save_model_data_for_tests=true", json=input_model)
    assert response.status_code == 200, f"Non‑200 response for {file_name}"

    resp_text: str = response.data.decode("utf-8").strip()
    json_start: int = resp_text.find("{")
    assert json_start != -1, f"No JSON body in response for {file_name}"
    output: dict[str, AcceptedTypes] = json.loads(resp_text[json_start:])

    # Basic success checks
    assert "errors" not in output or output["errors"] == [], f"Unexpected errors for {file_name}"

    # Basic success checks
    assert "errors" not in output or output["errors"] == [], f"Unexpected errors for {file_name}"

    # Check for presence of core output sections
    assert "resultsIndex" in output and isinstance(output["resultsIndex"],
                                                   list), (f"'resultsIndex' key missing or not a list for {file_name}")
    assert "cellResults" in output and isinstance(output["cellResults"],
                                                  list), (f"'cellResults' key missing or not a list for {file_name}")
    assert len(output["cellResults"]) > 0, f"No cell results found in output for {file_name}"

    # Optionally verify total cost or flow time data exists
    assert "totalCost" in output and isinstance(output["totalCost"],
                                                list), (f"'totalCost' key missing or not a list for {file_name}")
    assert any("value" in cost and "unit" in cost
               for cost in output["totalCost"]), (f"No valid cost entries in 'totalCost' for {file_name}")
    assert "pdfpoints" in output and isinstance(output["pdfpoints"],
                                                list), (f"'pdfpoints' key missing or not a list for {file_name}")

    # base currency comes from either modules[0]['customUnits']['unitCurrency']
    # or simply the 'to' field of the first fx row.
    model = input_model["model"] if "model" in input_model else input_model
    base_cur = model["finance"]["conversionRates"][0]["to"].upper()

    fx_table: dict[str, float | dict] = {r["from"].upper(): r["rate"] for r in model["finance"]["conversionRates"]}

    # flatten uncertain rates to nominal/likely for these deterministic tests
    for k, v in list(fx_table.items()):
        if isinstance(v, dict):
            fx_table[k] = (v.get("likely") or (v["min"] + v["max"]) / 2)

    # locate converted variables in response once, keyed by original 'key'
    #  → name matches the 'name' field in the input variable dict
    converted_lookup = {var["name"]: var for var in output.get("allVariables", []) if isinstance(var, dict)}

    # walk every variable in the input & validate those needing conversion
    variables = model["modules"][0]["manufacturing"]["variables"]
    for v in variables:
        cur, remainder = _extract_currency(v["unit"])
        if cur is None or cur == base_cur:
            continue  # already base currency or not a currency at all
        assert cur in fx_table, f"No FX rate from {cur}→{base_cur} in {file_name}"
        rate = fx_table[cur]
        assert isinstance(rate, (int, float)), "Rate must be numeric for this test"

        inp_value = v["value"]
        expected_value = inp_value * rate
        expected_unit = f"{base_cur}{remainder}"

        # look up converted variable
        out_var = converted_lookup.get(v["name"])
        assert out_var is not None, \
            f"Converted var '{v['name']}' missing in API output ({file_name})"

        out_val = float(out_var["value"])
        out_unit = out_var["unit"].replace(" ", "")  # normalise spacing

        # numeric comparison (absolute tolerance because magnitudes are reasonable)
        assert math.isclose(out_val, expected_value, rel_tol=0, abs_tol=1e-6), \
            (f"{v['name']} mismatch in {file_name}: "
             f"expected {expected_value} {expected_unit}, "
             f"got {out_val} {out_unit}")

        # unit string - tolerate case differences
        assert expected_unit.lower() == out_unit.lower(), \
            (f"{v['name']} unit mismatch in {file_name}: "
             f"expected '{expected_unit}', got '{out_unit}'")


@pytest.mark.parametrize(
    "file_name",
    [
        "basic_operations_currency_conversions_using_fallback.json",
    ],
)
def test_currency_conversion_fallback_only(test_client: flask.testing.FlaskClient, file_name: str) -> None:
    path = os.path.join(CURRENCY_DIR, file_name)
    with open(path, "r") as f:
        input_model: dict[str, AcceptedTypes] = json.load(f)

    response: Response = test_client.post("/solve?save_model_data_for_tests=true", json=input_model)
    assert response.status_code == 200, f"Non‑200 response for {file_name}"

    response_data: str = response.data.decode("utf-8").strip()
    json_start = response_data.find("{")
    assert json_start != -1, f"No JSON body in response for {file_name}"
    output: dict[str, AcceptedTypes] = json.loads(response_data[json_start:])

    assert "errors" not in output or output["errors"] == [], f"Unexpected errors in {file_name}"

    # Get base currency from unitCurrency
    model = input_model["model"]
    base_currency = model["modules"][0]["customUnits"]["unitCurrency"].upper()

    # Get variables to check
    variables = model["modules"][0]["manufacturing"]["variables"]
    converted = {v["name"]: v for v in output.get("allVariables", []) if isinstance(v, dict)}

    for v in variables:
        original_unit = v.get("unit", "")
        value = v["value"]
        name = v["name"]

        # Only process if it's a foreign currency (e.g., $)
        cur, suffix = _extract_currency(original_unit)
        live_rates = refresh_fx_rates(base_currency)

        if not cur or cur.upper() == base_currency:
            continue

        cur = cur.upper()
        assert cur in live_rates, f"Missing fallback rate for {cur} → {base_currency}"

        fx_rate = live_rates[cur]
        expected_value = value * fx_rate
        expected_unit = f"{base_currency}{suffix}"
        # round the expected value to match output from cost rounding in the solution generation
        expected_value = smart_round(expected_value)

        # Look up converted value
        out_var = converted.get(name)
        assert out_var is not None, f"Missing converted var '{name}' in response"

        actual_val = float(out_var["value"])
        actual_unit = out_var["unit"].replace(" ", "")

        assert math.isclose(
            actual_val, expected_value, rel_tol=1e-3
        ), (f"{name} value mismatch: expected {expected_value}, got {actual_val}")

        assert expected_unit.lower() == actual_unit.lower(
        ), (f"{name} unit mismatch: expected '{expected_unit}', got '{actual_unit}'")
