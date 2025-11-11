import json
import logging
import math
import os
from typing import Any, Generator

from flask import Response
from flask.testing import FlaskClient
import pint
import pytest

# isort: off
import gpx.custom_units  # noqa: F401
# isort: on
from unittest.mock import MagicMock, patch

from gpkit import ureg

from api.api import SolverThread, app, auth
import gpx
from utils.unit_helpers import setup_currency_conversion_variables

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data", "currency_conversion")

logging.basicConfig(level=logging.DEBUG)
currency_map = {"$": "USD", "£": "GBP", "€": "EUR"}


def load_model_data(file_name: str) -> dict:
    path = os.path.join(DATA_DIR, file_name)
    logging.info(f"🔍 Loading model from: {path}")
    with open(path, "r") as f:
        return json.load(f)["model"]


# helper - pull the same rates that were loaded into the module (for independent checks)
def fetch_exchange_rates(base: str = "USD") -> dict[str, float]:
    rates = gpx.custom_units.fetch_exchange_rates(base)
    rates = dict(rates)
    rates[base] = 1.0
    return rates


# currency unit strings
@pytest.mark.parametrize("unit", ["USD", "GBP", "EUR"])
def test_currency_unit_string(unit):
    q = 10 * ureg(unit)
    assert q.magnitude == 10
    assert str(q.units) == unit


def safe_make_fx_context(base_to_code_rates: dict[str, float], base: str = "USD") -> pint.Context:
    """
    Build a safe FX context from BASE->CODE rates, defining relations:
      1 {code} = (1 / rate) {base}
    Notes:
      - Expects *unflipped* rates (Frankfurter style).
      - Skips redefining USD to avoid 'plain -> derived' errors.
    """
    ctx = pint.Context("FX")
    for code, rate in base_to_code_rates.items():
        if code not in gpx.custom_units.ALLOWED_CURRENCIES or code == base:
            continue
        # rate should be numeric; math.isnan requires float
        rate_f = float(rate)
        if math.isnan(rate_f):
            continue
        if code == "USD":  # Don't redefine USD!
            continue
        ctx.redefine(f"{code} = {1.0 / rate_f} {base}")
    return ctx


def test_usd_to_gbp_to_eur_back_to_usd_round_trip():
    """
    Use the library's get_rates() (flipped: code -> base/code) and invert to BASE->CODE
    for safe_make_fx_context, then verify round-trip USD→GBP→EUR→USD is identity.
    """
    # Get flipped (USD base) rates from the module; safe=False ensures availability
    flipped = gpx.custom_units.get_rates(base="USD", safe=False)  # {'GBP': USD/GBP, 'EUR': USD/EUR, ...}
    # Invert to base->code for the helper
    base_to_code = {code: 1.0 / val for code, val in flipped.items() if val}
    base_to_code["USD"] = 1.0

    # Patch FX context safely
    safe_ctx = safe_make_fx_context(base_to_code, base="USD")
    gpx.custom_units.ureg._contexts["FX"] = safe_ctx  # overwrite safely

    # Now proceed with the round-trip
    start_value = 123.45
    usd_start = start_value * ureg("USD")

    gbp = gpx.custom_units.convert_currency(usd_start, "USD", "GBP")
    eur = gpx.custom_units.convert_currency(gbp, "GBP", "EUR")
    usd_end = gpx.custom_units.convert_currency(eur, "EUR", "USD")

    print(
        f"USD → GBP → EUR → USD round-trip: "
        f"{usd_start.magnitude} → {gbp.magnitude} → {eur.magnitude} → {usd_end.magnitude}"
    )

    assert math.isclose(usd_end.magnitude, start_value, rel_tol=0, abs_tol=1e-9)
    assert str(usd_end.units) == "USD"


@pytest.fixture(scope="module")
def test_client() -> Generator[FlaskClient, None, None] | None:
    """Create a test client for the Flask application with fully bypassed authentication."""
    mock_auth: MagicMock = MagicMock()
    mock_auth.username = "test_user"
    mock_auth.password = "test_password"

    def mock_verify_password(username: str, password: str) -> bool:
        return True

    def mock_authenticate(auth: Any, password: str) -> str:
        return "test_user"

    def mock_authorize(role: str, user: str, auth: Any) -> bool:
        return True

    with patch.dict(os.environ, {"COP_NO_LOGIN": "1"}):  # Mock login to bypass authentication
        with patch.object(auth, "verify_password", mock_verify_password):  # Always authenticate
            with patch.object(auth, "authenticate", mock_authenticate):  # Return mock user
                with patch.object(auth, "authorize", mock_authorize):  # Always authorize
                    with patch.object(auth, "get_auth", return_value=mock_auth):  # Disable auth enforcement
                        with patch.object(auth, "current_user", return_value="test_user"):  # Mock user context
                            with patch(
                                    "flask_httpauth.HTTPBasicAuth.login_required",
                                    lambda f: f,
                            ):  # Fully bypass login
                                with app.test_client() as client:
                                    client.environ_base["HTTP_AUTHORIZATION"
                                                        ] = "Basic dGVzdF91c2VyOnBhc3N3b3Jk"  # Inject mock auth header
                                    yield client
    return None


def test_custom_unit_symbol_conversion(test_client):
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "data", "custom_unit_cases", "basic_operations_custom_units.json")

    with open(model_path, "r") as f:
        input_model = json.load(f)

    captured_model = {}

    def mock_run(self):
        nonlocal captured_model
        captured_model = self.model
        self.ret = {"status": "ok"}
        self.exc = None

    with patch.object(SolverThread, "run", mock_run):
        response: Response = test_client.post("/solve", json=input_model)
        assert response.status_code == 200

        vs = captured_model["model"]["modules"][0]["manufacturing"]["variablesSimple"]
        vmap = {v["name"]: v["unit"] for v in vs if "unit" in v}

        missing_units = [v["name"] for v in vs if "unit" not in v]
        print("Variables missing 'unit':", missing_units)

        assert vmap["First process Cell Non-Recurring Cost"] == "USD"
        assert vmap["Second Process Cell Non-Recurring Cost"] == "GBP"
        assert vmap["Tooling Non-Recurring Cost"] == "EUR"

        full_vars = captured_model["model"]["modules"][0]["manufacturing"]["variables"]

        percent_units = [v["unit"] for v in full_vars if v["unit"] in ("pct", "percent")]
        assert percent_units, "Expected at least one '%' unit to be converted to 'pct' or 'percent'"


def test_fx_rate_consistency():
    """
    Within the FX context, 1 GBP → USD should equal the flipped rate
    returned by get_rates(base='USD')['GBP'].
    """
    with ureg.context("FX"):
        rate_gbp_usd = (1 * ureg.GBP).to("USD").magnitude

    expected = gpx.custom_units.get_rates(base="USD", safe=False)["GBP"]
    logging.debug(f"1 GBP → USD (via Pint): {rate_gbp_usd}")
    logging.debug(f"1 GBP → USD (expected flipped): {expected}")

    assert math.isclose(rate_gbp_usd, expected, rel_tol=1e-9), (
        f"Pint rate {rate_gbp_usd} vs expected {expected}"
    )


@pytest.mark.parametrize(
    "from_unit,to_unit",
    [("GBP", "USD"), ("EUR", "USD"), ("GBP", "EUR"), ("EUR", "GBP")],
)
def test_currency_requires_context(from_unit, to_unit):
    val = 5 * ureg(from_unit)
    try:
        result = val.to(to_unit).magnitude
        logging.debug(f"Conversion {from_unit} → {to_unit} outside context: {result}")
        assert math.isnan(result)
    except Exception as e:
        logging.debug(f"Conversion {from_unit} → {to_unit} failed as expected with exception: {e}")
        assert True


@pytest.mark.parametrize(
    "from_unit,to_unit",
    [("USD", "GBP"), ("GBP", "USD"), ("USD", "EUR"), ("EUR", "USD"), ("GBP", "EUR"), ("EUR", "GBP")],
)
def test_currency_conversion_accuracy(from_unit, to_unit):
    rates = fetch_exchange_rates(base=from_unit)
    amount = 25000
    val = amount * ureg(from_unit)
    pint_result = gpx.custom_units.convert_currency(val, from_unit, to_unit).magnitude

    with ureg.context("FX"):
        expected_rate = (1 * ureg(from_unit)).to(to_unit).magnitude

    expected = amount * expected_rate

    logging.debug("\n=== Currency Conversion Accuracy ===")
    logging.debug(f"API rates: {rates}")
    logging.debug(f"1 {from_unit} → {to_unit} (reciprocal): {1 / rates[from_unit] if from_unit != 'USD' else 1}")
    logging.debug(f"Pint conversion {from_unit}->{to_unit}: {pint_result} vs manual {expected}")
    logging.debug(f"Expected Amount: {expected}")
    logging.debug(f"Actual Amount: {pint_result}")
    logging.debug(f"Rate From {from_unit}: {rates[from_unit]}")
    logging.debug(f"Rate To {to_unit}: {rates[to_unit]}")

    assert math.isclose(
        pint_result, expected, rel_tol=1e-5, abs_tol=1e-9
    ), f"Mismatch {from_unit}->{to_unit}: expected {expected}, got {pint_result}"


@pytest.mark.parametrize(
    "amount, from_unit, to_unit",
    [(25000, "USD", "GBP"), (25000, "GBP", "USD"), (25000, "USD", "EUR"), (25000, "EUR", "USD"), (25000, "GBP", "EUR"),
     (25000, "EUR", "GBP")],
)
def test_currency_conversion_accuracy_with_amount(amount, from_unit, to_unit):
    rates = fetch_exchange_rates(base=from_unit)
    pint_result = gpx.custom_units.convert_currency(amount, from_unit, to_unit).magnitude

    with ureg.context("FX"):
        expected_rate = (1 * ureg(from_unit)).to(to_unit).magnitude

    expected = amount * expected_rate

    logging.debug("\n=== Currency Conversion Accuracy ===")
    logging.debug(f"API rates: {rates}")
    logging.debug(f"1 {from_unit} → {to_unit} (reciprocal): {1 / rates[from_unit] if from_unit != 'USD' else 1}")
    logging.debug(f"Pint conversion {from_unit}->{to_unit}: {pint_result} vs manual {expected}")
    logging.debug(f"Expected Amount: {expected}")
    logging.debug(f"Actual Amount: {pint_result}")
    logging.debug(f"Rate From {from_unit}: {rates[from_unit]}")
    logging.debug(f"Rate To {to_unit}: {rates[to_unit]}")

    assert math.isclose(
        pint_result, expected, rel_tol=1e-5, abs_tol=1e-9
    ), f"Mismatch {from_unit}->{to_unit}: expected {expected}, got {pint_result}"


def extract_variable_by_key(variables, key):
    """Helper to find a variable dict by its key."""
    for var in variables:
        if var.get("key") == key:
            return var
    raise KeyError(f"Variable with key '{key}' not found.")


def _prepare(file_name: str, allow_fallback: bool):
    """
    Return
        • val         numeric amount in the *source* currency
        • fx_pv       the ParametricVariable created by
                      setup_currency_conversion_variables that converts
                      source → base
        • rates       finance.conversionRates list from the JSON
        • src_cur     source currency (USD,…)
        • base_cur    base/target currency (GBP,…)
    Works with the new ParametricVariable-based implementation.
    """
    logging.debug(f"\n📘 Loading model from: {file_name}")
    model_data = load_model_data(file_name)
    model = model_data["model"] if "model" in model_data else model_data

    variables = model["modules"][0]["manufacturing"]["variables"]
    var_dict = extract_variable_by_key(
        variables,
        "First process Cell Non-Recurring Cost // First Module",
    )
    logging.debug(f"  ➤ Original variable dict: {var_dict}")

    src_cur = currency_map[var_dict["unit"]]
    val = var_dict["value"]

    rates = model["finance"].get("conversionRates", [])
    base_cur = rates[0]["to"]
    assert any(r["from"] == src_cur for r in rates)

    fx_vars = setup_currency_conversion_variables(model, base=base_cur, allow_dynamic_fallback=allow_fallback)
    assert fx_vars, "No FX variables returned"

    # Pick the FX PV that converts this source→base
    wanted = f"{src_cur} to {base_cur}"
    fx_pv = next(pv for pv in fx_vars if pv.name == wanted)

    logging.debug(f"  ➤ Matched PV  {fx_pv.name}  qty={fx_pv.qty}  unit={fx_pv.unit}")
    return val, fx_pv, rates, src_cur, base_cur


@pytest.mark.parametrize(
    "file_name, expected_units, allow_fallback",
    [
        ("basic_operations_currency_conversions.json", "USD/GBP", False),
        ("basic_operations_currency_conversions_w_multiple.json", "USD/GBP", False),
    ],
)
def test_currency_usd_to_gbp_static_rates(file_name, expected_units, allow_fallback):
    logging.debug(f"\n🔍 STATIC-rate test: {file_name}")
    val, fx_pv, rates, src_cur, _ = _prepare(file_name, allow_fallback)

    exp_rate = next(r["rate"] for r in rates if r["from"] == src_cur)
    assert isinstance(exp_rate, (int, float))

    # numeric result from ParametricVariable
    numeric_conv = val * fx_pv.qty.magnitude
    expected_val = val * exp_rate

    logging.debug(f"  • Base amount:      {val}")
    logging.debug(f"  • Static rate used: {exp_rate}")
    logging.debug(f"  • Expected value:   {expected_val}")
    logging.debug(f"  • Numeric result:   {numeric_conv}")
    logging.debug(f"  • Units:            {(val * fx_pv.qty).units}")

    assert math.isclose(numeric_conv, expected_val, rel_tol=1e-9)

    # Unit check
    units_str = str((val * fx_pv.qty).units).lower()
    for tok in ("usd", "gbp"):
        assert tok in units_str
