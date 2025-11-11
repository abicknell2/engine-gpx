from collections import OrderedDict
import copy
import json
import logging
import math
import re
from typing import Optional

from gpkit import Variable
from gpkit.units import ureg
import pint

from gpx.custom_units import INITIALISED_FX_BASES, get_rates, refresh_fx_rates
from gpx.dag.parametric import ParametricVariable
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


def apply_replacements(
    data: AcceptedTypes,
    replacements: OrderedDict[str, str],
    count: int | None = None,
    replaced: list[tuple[str, str, str]] | None = None,
    skip_keys: set[str] | None = None,
    parent: dict[str, AcceptedTypes] | None = None
) -> AcceptedTypes:
    """Generic recursive replacement utility with optional count limit. Optionally collects replaced strings and skips keys."""
    if replaced is None:
        replaced = []
    if skip_keys is None:
        skip_keys = set()

    if isinstance(data, str):
        original_data = data
        for pattern, replacement in replacements.items():
            if re.search(pattern, data):
                if count is not None:
                    data = re.sub(pattern, replacement, data, count=count)
                else:
                    data = re.sub(pattern, replacement, data)
        if original_data != data:
            context_name = "<unknown>"
            if parent and isinstance(parent, dict):
                context_name = parent.get("name") or parent.get("property") or context_name
            replaced.append((context_name, original_data, data))
        if data == "":
            logging.warning(f"Replaced '{original_data}' -> Empty String!")
        return data

    elif isinstance(data, dict):
        return {
            k: (v if k in skip_keys else apply_replacements(v, replacements, count, replaced, skip_keys, parent=data))
            for k, v in data.items()
        }

    elif isinstance(data, list):
        return [apply_replacements(item, replacements, count, replaced, skip_keys, parent=parent) for item in data]

    else:
        return data


def log_replacements(title: str, replaced: list[tuple[str, str, str]]) -> None:
    """Logs a formatted summary of the string replacements made, including the context name and before/after values."""
    if replaced:
        logging.debug(f"\n{title}")
        max_orig_len = max(len(orig) for _, orig, _ in replaced)
        for name, orig, new in replaced:
            logging.debug(f"  {orig.ljust(max_orig_len)}  ->  {new}    (name/property: {name})")


def percent_unit_replacements_to_process(data: AcceptedTypes) -> tuple[AcceptedTypes, list[tuple[str, str, str]]]:
    """Standardizes percent units in data to internal format (e.g., 'pct'). Logs and returns replacements made."""
    replacements = OrderedDict([
        (r"/%", "/pct"),
        (r"%/", "pct/"),
        (r"^%$", "pct"),
        (r"%", "pct"),
    ])
    replaced: list[tuple[str, str, str]] = []
    result = apply_replacements(data, replacements, count=1, replaced=replaced, skip_keys={"property", "name"})
    log_replacements("[PERCENT UNIT REPLACEMENTS - TO PROCESS]", replaced)
    return result


def percent_unit_display_text(data: AcceptedTypes) -> AcceptedTypes:
    """Converts internal percent unit format (e.g., 'pct') back to user-friendly display text using '%'. Logs replacements made."""
    replacements = OrderedDict([
        (r"/pct", "/%"),
        (r"/percent", "/%"),
        (r"pct/", "%/"),
        (r"percent/", "%/"),
        (r"^pct$", "%"),
        (r"^percent$", "%"),
        (r"pct", "%"),
        (r"percent", "%"),
    ])
    replaced: list[tuple[str, str, str]] = []
    result = apply_replacements(data, replacements, count=1, replaced=replaced, skip_keys={"property"})
    log_replacements("[PERCENT UNIT REPLACEMENTS - DISPLAY TEXT]", replaced)
    return result


def hour_unit_replacements_to_process(data: AcceptedTypes) -> AcceptedTypes:
    """Standardizes hour units in data to internal format (e.g., 'hr'). Logs and returns replacements made."""
    replacements = OrderedDict([
        (r"\bh/", "hr/"),
        (r"/h\b", "/hr"),
        (r"\bh\b", "hr"),
    ])
    replaced: list[tuple[str, str, str]] = []
    result = apply_replacements(data, replacements, count=1, replaced=replaced)
    log_replacements("[HOUR UNIT REPLACEMENTS - TO PROCESS]", replaced)
    return result


def hour_unit_display_text(data: AcceptedTypes) -> AcceptedTypes:
    """Converts internal hour unit format (e.g., 'hr') back to user-friendly display text (e.g., 'h'). Logs replacements made."""
    replacements = OrderedDict([
        (r"\bh/", "hr/"),
        (r"/h\b", "/hr"),
        (r"\bh\b", "hr"),
    ])

    replaced: list[tuple[str, str, str]] = []
    result = apply_replacements(data, replacements, count=1, replaced=replaced)
    log_replacements("[HOUR UNIT REPLACEMENTS - DISPLAY TEXT]", replaced)
    return result


def currency_unit_replacements_to_process(data: AcceptedTypes) -> AcceptedTypes:
    """Standardizes currency symbols in data (e.g., $, £, €) to internal 3-letter ISO codes (e.g., USD, GBP, EUR). Logs replacements made."""
    replacements = OrderedDict([
        (r"/\$", "/USD"),
        (r"\$/", "USD/"),
        (r"(?<!\s)\$(?!\s)", "USD"),
        (r"/£", "/GBP"),
        (r"£/", "GBP/"),
        (r"(?<!\s)£(?!\s)", "GBP"),
        (r"/€", "/EUR"),
        (r"€/", "EUR/"),
        (r"(?<!\s)€(?!\s)", "EUR"),
    ])
    replaced: list[tuple[str, str, str]] = []
    result = apply_replacements(data, replacements, replaced=replaced)
    log_replacements("[CURRENCY UNIT REPLACEMENTS - TO PROCESS]", replaced)
    return result


def currency_unit_display_text(data: AcceptedTypes, settings: Settings) -> AcceptedTypes:
    """Converts internal currency unit codes (e.g., 'USD') back to currency symbols (e.g., '$') for display,
    unless the default currency already uses those symbols. Logs replacements made if any occur."""
    currency_iso_to_symbol = {
        "USD": "$",
        "GBP": "£",
        "EUR": "€",
    }

    default = settings.default_currency.upper().strip()

    if default not in currency_iso_to_symbol:
        replacements = OrderedDict([
            (r"/USD", "/$"),
            (r"USD/", "$/"),
            (r"(?<!\s)USD(?!\s)", "$"),
            (r"/GBP", "/£"),
            (r"GBP/", "£/"),
            (r"(?<!\s)GBP(?!\s)", "£"),
            (r"/EUR", "/€"),
            (r"EUR/", "€/"),
            (r"(?<!\s)EUR(?!\s)", "€"),
        ])
        replaced: list[tuple[str, str, str]] = []
        data = apply_replacements(data, replacements, replaced=replaced)
        log_replacements("[CURRENCY UNIT REPLACEMENTS - DISPLAY TEXT]", replaced)

    return data


def replace_units_to_process(data: AcceptedTypes) -> AcceptedTypes:
    """Runs all unit normalization replacements (percent, hour, and currency) for internal processing."""
    data = percent_unit_replacements_to_process(data)
    data = hour_unit_replacements_to_process(data)
    data = currency_unit_replacements_to_process(data)
    return data


def replace_unit_display_text(interaction: AcceptedTypes) -> AcceptedTypes:
    """Converts internal unit representations in solution output back to display-friendly format (%, h, $, etc.)."""
    data = interaction.solutions
    data = percent_unit_display_text(data)
    data = hour_unit_display_text(data)
    data = currency_unit_display_text(data, interaction.settings)
    return data


def store_original_currency_data(model: "AcceptedTypes", settings: Settings) -> None:
    """
    Populate `settings.original_currency_data` with a subtree that
    contains **only** those dicts whose *unit/rateUnit/durationUnit*
    mention a currency *different* from `default_currency`.
    """

    result: Optional[dict] = None

    # basic currency helpers kept inline (no separate defs)
    cur_map = {"$": "USD", "£": "GBP", "€": "EUR"}
    known_iso = {"USD", "GBP", "EUR"}
    def_to_iso = cur_map.get(settings.default_currency, settings.default_currency).upper()

    # depth-first walk using an explicit stack (to avoid nested defs)
    stack: list = [(model, ["model"], model)]  # (node, path, parent)

    while stack:
        node, path, parent = stack.pop()

        # dict branch
        if isinstance(node, dict):
            for key, val in node.items():

                is_unit_key = key in {"unit", "units", "rateUnit", "durationUnit"}
                if is_unit_key and isinstance(val, str):

                    unit_clean = val.replace("(", "").replace(")", "").replace(" ", "")
                    tokens = [tok for tok in unit_clean.split("/") if tok]

                    # does *any* token hold a non-default currency?
                    diff_currency = False
                    for tok in tokens:
                        iso = cur_map.get(tok, tok).upper()
                        if iso in known_iso and iso != def_to_iso:
                            diff_currency = True
                            break

                    if diff_currency:
                        # ── mirror the parent object into `result` following `path`
                        result = {}
                        cur = result
                        for i, seg in enumerate(path[:-1]):
                            nxt = path[i + 1]
                            if isinstance(seg, int):  # list index
                                while len(cur) <= seg:
                                    cur.append({})
                                cur = cur[seg]
                            else:  # dict key
                                if seg not in cur:
                                    cur[seg] = [] if isinstance(nxt, int) else {}
                                cur = cur[seg]

                        last = path[-1]
                        if isinstance(last, int):
                            while len(cur) <= last:
                                cur.append({})
                            cur[last] = copy.deepcopy(parent)
                        else:
                            cur[last] = copy.deepcopy(parent)
                        # no need to push deeper into this branch
                        continue

                # descend
                stack.append((val, path + [key], val if isinstance(val, dict) else parent))

        # list branch
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                stack.append((item, path + [idx], item))

    # prune empty {} / [] that may have been created during mirroring
    prune_stack = [result]
    while prune_stack:
        current = prune_stack.pop()
        if isinstance(current, dict):
            keys_to_delete = [k for k, v in current.items() if v in ({}, [], None)]
            for k in keys_to_delete:
                del current[k]
            for v in current.values():
                prune_stack.append(v)
        elif isinstance(current, list):
            # iterate *backwards* so popping doesn’t mess up indices
            for i in range(len(current) - 1, -1, -1):
                if current[i] in ({}, [], None):
                    current.pop(i)
                else:
                    prune_stack.append(current[i])

    settings.original_currency_data = result


def include_original_currency_data(model: AcceptedTypes, settings: Settings) -> AcceptedTypes:
    """
    Walk the model once, and for every variable that matches one of the
    entries cached in `settings.original_currency_data`, attach an
    `originalCurrencyData` block containing its original value + unit.

    The function **does not** edit or normalise any of the model’s
    existing unit strings.
    """
    updated_model = copy.deepcopy(model)

    # helper: pull every dict that has a 'unit' and a name-like field
    # from the snapshot we saved *before* unit normalisation.
    def gather_currency_references(data: AcceptedTypes) -> list[dict[str, AcceptedTypes]]:
        refs: list[dict[str, AcceptedTypes]] = []

        def walk(node: AcceptedTypes) -> None:
            if isinstance(node, dict):
                if "unit" in node and ("name" in node or "property" in node):
                    refs.append(node)
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for itm in node:
                    walk(itm)

        walk(data)
        return refs

    original_refs = gather_currency_references(settings.original_currency_data)

    # helper: if variable matches a reference, attach original data
    def inject_original_currency_data(variable: dict[str, AcceptedTypes], reference: dict[str, AcceptedTypes]) -> None:
        var_keys = {variable.get("name", "").lower(), variable.get("property", "").lower()}
        ref_keys = {reference.get("name", "").lower(), reference.get("property", "").lower()}

        if var_keys & ref_keys:  # intersection ≠ ∅  → we have a match
            variable["originalCurrencyData"] = {}
            if "value" in reference:
                variable["originalCurrencyData"]["value"] = reference["value"]
            if "unit" in reference:
                variable["originalCurrencyData"]["unit"] = reference["unit"]

    # scan *only* the allVariables array in the results
    if isinstance(updated_model.get("allVariables"), list):
        for var in updated_model["allVariables"]:
            for ref in original_refs:
                inject_original_currency_data(var, ref)

    return updated_model


def setup_currency_conversion_variables(
    model_dict: dict[str, AcceptedTypes],
    base: str = "USD",
    allow_dynamic_fallback: bool = True
) -> list[ParametricVariable]:
    """
    Build ParametricVariables + Constraints for all FX conversions in the model.

    Returns:
        - List of ParametricVariables (with .varkey)
        - Dict of constraints keyed by param.name (used for acyclic_constraints)
    """
    if base != "USD" and base not in INITIALISED_FX_BASES:
        refresh_fx_rates(base)
        INITIALISED_FX_BASES.add(base)

    finance = model_dict.get("finance", {})
    conversion_rates = finance.get("conversionRates", [])

    if not conversion_rates and not allow_dynamic_fallback:
        raise ValueError("No conversionRates found and fallback is disabled.")

    fx_params: list[ParametricVariable] = []

    if not conversion_rates and allow_dynamic_fallback:
        base_currency = (model_dict.get("modules", [{}])[0].get("customUnits", {}).get("unitCurrency", "USD").upper())
        foreign_currencies = detect_foreign_currencies(model_dict, base_currency)

        # Ensure FX rates are available
        rates = refresh_fx_rates(base_currency)

        # Build fallback conversion rates for all non-base currencies
        conversion_rates = []
        for from_currency in foreign_currencies:
            fx_rate = rates.get(from_currency)
            if fx_rate is not None:
                conversion_rates.append({"from": from_currency, "to": base_currency, "rate": fx_rate})

    for rate_entry in conversion_rates:
        from_currency = rate_entry.get("from")
        to_currency = rate_entry.get("to")
        rate = rate_entry.get("rate")

        if not from_currency or not to_currency:
            continue

        if to_currency != base:
            raise ValueError(f"Target currency '{to_currency}' must match base currency '{base}'")

        unit_str = f"{to_currency}/{from_currency}"
        param_name = f"{from_currency} to {to_currency}"

        # Handle static, uncertain, or fallback rate
        if rate is None:
            if not allow_dynamic_fallback:
                raise ValueError(f"Missing rate for {param_name} and fallback disabled")
            rate = get_rates(base, safe=True).get(from_currency)
            if rate is None:
                raise ValueError(f"No live FX rate available for {from_currency}")
            value = rate
            uncertainty = None
        elif isinstance(rate, dict):
            value = rate.get("likely", None)
            uncertainty = {
                "min": rate["min"],
                "value":
                value,  # Set to none will successfully fail on isinstance(var.value, Number) in uncertainty_helpers.py.
                "max": rate["max"],
            }
            if value is None:
                value = (rate["min"] + rate["max"]) / 2  # Use to create initial PV qty
        else:
            value = rate
            uncertainty = None

        qty = value * ureg(unit_str)

        fx_param = ParametricVariable(
            name=param_name, varkey=Variable(param_name, unit_str), qty=qty, unit=unit_str, is_input=True
        )
        fx_param.update_value(quantity=qty)

        # stash uncertainty information for the uncertainty solver (harmless otherwise)
        if uncertainty is not None:
            fx_param.uncertainty_bounds = uncertainty  # type: ignore[attr-defined]

        fx_params.append(fx_param)

    return fx_params


def detect_foreign_currencies(model_dict: dict[str, AcceptedTypes], base_currency: str = "USD") -> set[str]:
    """
    Scans the entire model_dict as a string for any foreign currency indicators
    (symbols or ISO codes) that are not the base currency.

    Args:
        model_dict: The model data as a dictionary.
        base_currency: The base currency to exclude from results.

    Returns:
        A set of detected foreign currency codes (e.g., {"EUR", "GBP"}).
    """
    base_currency = base_currency.upper()
    all_currencies = {"USD", "GBP", "EUR"}
    symbols_to_currency = {"$": "USD", "£": "GBP", "€": "EUR"}

    model_str = json.dumps(model_dict)

    detected = set()
    for symbol, iso in symbols_to_currency.items():
        if symbol in model_str and iso != base_currency:
            detected.add(iso)

    for iso in all_currencies:
        if iso != base_currency and iso in model_str:
            detected.add(iso)

    return detected


def is_continuous_unit(unit):
    '''Check if the unit has dimensionality of mass, volume, etc.'''
    try:
        return any(unit.check(dim) for dim in ('[mass]', '[volume]', '[energy]', '[length]'))
    except Exception:
        return False


def _canonical_q_hr_and_hours_per(qty: pint.Quantity, settings: Settings) -> tuple[pint.Quantity, float]:
    """
    Return the quantity in <currency>/hr plus the number of hours
    that correspond to the chosen calendar bucket.

    If *target_unit* is None we just echo the input and 1 h.
    """
    if settings.rate_unit is None:
        return qty, 1.0

    alias_map = {
        "hour": "hr",
        "hours": "hr",
        "hrs": "hr",
    }
    target_unit = alias_map.get(settings.rate_unit.lower(), settings.rate_unit)

    q_hr = qty.to(f"{settings.default_currency_iso}/hr")

    hrs_per = {
        "hr": 1,
        "day": settings.hrs_per_shift * settings.shifts_per_day,
        "week": settings.hrs_per_shift * settings.shifts_per_day * settings.days_per_week,
        "month":
        settings.hrs_per_shift * settings.shifts_per_day * settings.days_per_week * settings.weeks_per_year / 12.0,
        "year": settings.hrs_per_shift * settings.shifts_per_day * settings.days_per_week * settings.weeks_per_year,
    }.get(target_unit)

    if hrs_per is None:
        raise ValueError(f"Unsupported time unit '{target_unit}'")

    return q_hr, hrs_per


def convert_item_recurring_cost(
    qty: pint.Quantity,
    settings,
) -> pint.Quantity:
    """
    Scale the magnitude from USD/hr to USD/(chosen period) **but keep the unit USD/hr**.
    Suitable for line-items where you override the `unit` field yourself.
    """
    q_hr, hrs = _canonical_q_hr_and_hours_per(qty, settings)
    return q_hr * hrs


def convert_total_recurring_cost(
    qty: pint.Quantity,
    settings,
) -> pint.Quantity:
    """
    Fully convert to USD/<chosen period>; used for *Total Recurring Cost*.
    """
    q_hr, hrs = _canonical_q_hr_and_hours_per(qty, settings)
    scalar = q_hr.magnitude * hrs
    return scalar * ureg(f"{settings.default_currency_iso}/{settings.rate_unit}")


def update_settings_finance_units(settings: Settings, finance: dict[str, AcceptedTypes]):
    if finance is not None:
        settings.hrs_per_shift = float(finance.get("hrsPerShift", settings.hrs_per_shift))
        settings.shifts_per_day = float(finance.get("shiftsPerDay", settings.shifts_per_day))
        settings.days_per_week = float(finance.get("daysPerWeek", settings.days_per_week))
        settings.weeks_per_year = float(finance.get("weeksPerYear", settings.weeks_per_year))
        settings.rate_unit = str(finance.get("rateUnit", settings.rate_unit)).lower()
        settings.duration_unit = str(finance.get("durationUnit", settings.duration_unit)).lower()
    return settings


def reduce_currency_values_to_sig_figs(data: AcceptedTypes, sig_figs: int = 2) -> AcceptedTypes:
    """
    Recursively walk through the data, updating numeric currency-like values
    to the specified number of significant figures.

    We look for:
      - Keys named 'marginalCost' or 'cashflow' (always currency).
      - 'value' keys that appear to be costs (if 'name' or 'unit' indicates cost or '$').
      - Known cost-like labels in 'name' ('Cost', 'Non-Recurring', 'Average Unit Cost', etc.).
      - Also check 'totalCost' dict for 'Total Non-Recurring Cost' or anything else that is obviously currency.

    We do NOT alter the 'unit' text; only the numeric value is changed.
    """

    # Helper function to format a float to `sig_figs` significant figures,
    # returning a float (not a string) for convenience.
    def to_sigfigs(num: float, sf: int) -> float:
        # Edge cases: 0.0 or very small
        if num == 0:
            return 0.0

        # Use format approach to preserve magnitude
        return float(f"{num:.{sf}g}")

    # Checks if a dictionary entry looks like currency based on name/unit
    def is_currency_like(d: dict) -> bool:
        # if "unit" explicitly has '$'
        unit = d.get("unit", "")
        if "$" in str(unit):
            return True

        # if "name" includes words like "Cost" or "Non-Recurring" or "Average Unit Cost"
        name = d.get("name", "")
        name_lower = name.lower()
        if ("cost" in name_lower) or ("non-recurring" in name_lower) or ("average unit cost" in name_lower):
            return True

        return False

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # 1) If key is obviously a currency key:
                if key in ("cashflow", "marginalCost"):
                    data[key] = to_sigfigs(float(value), sig_figs)
                # 2) If key == 'value' but the dictionary indicates currency in 'name' or 'unit'
                elif key == "value" and is_currency_like(data):
                    data[key] = to_sigfigs(float(value), sig_figs)
                # 3) If this is the 'totalCost' block with "Total Non-Recurring Cost"
                elif key == "Total Non-Recurring Cost":
                    data[key] = to_sigfigs(float(value), sig_figs)

            # Recurse into substructures
            elif isinstance(value, dict) or isinstance(value, list):
                data[key] = reduce_currency_values_to_sig_figs(value, sig_figs)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = reduce_currency_values_to_sig_figs(item, sig_figs)

    return data


def round_currency_values(data: AcceptedTypes, decimals: int = 2) -> AcceptedTypes:
    """
    Recursively walk through the JSON-like data structure, rounding any
    currency-related values to the specified number of decimal places.

    We do NOT reduce them to “significant figures,” but rather to 'decimals'
    places. For example, decimals=2 means:
      1326719.0   -> 1326719.00
      1234.5678   -> 1234.57
      999.9999    -> 1000.00

    You can adjust 'decimals' to 2 for typical currency usage (cents).
    """

    def is_currency_like(d: dict) -> bool:
        """
        Decide if a dict's 'value' key is currency. For instance:
          - the 'name' or 'unit' field might contain '$' or 'Cost'
          - known currency keys such as 'Non-Recurring Cost', 'marginalCost', etc.
        """
        # if there's a 'unit' with '$'
        unit = str(d.get("unit", "")).lower()
        if "$" in unit:
            return True

        # if 'name' has "Cost" or "Non-Recurring" or something similar
        name = str(d.get("name", "")).lower()
        if ("cost" in name) or ("non-recurring" in name) or ("average unit cost" in name):
            return True

        return False

    def round_to_decimals(num: float, d_places: int) -> float:
        """Round 'num' to 'd_places' decimal places, returning float."""
        return smart_round(num, sigfigs=3)
        # return round(num, d_places)

    if isinstance(data, dict):
        for key, value in data.items():
            # 1. If the value is numeric, decide if we should round it
            if isinstance(value, (int, float)):
                # Cases that are definitely currency
                #  a) key names "cashflow", "marginalCost", or something like "Total Non-Recurring Cost"
                #  b) key == "value" but the dictionary suggests currency
                if key in ("cashflow", "marginalCost", "Total Non-Recurring Cost"):
                    data[key] = round_to_decimals(value, decimals)
                elif key == "value" and is_currency_like(data):
                    data[key] = round_to_decimals(value, decimals)

            # 2. If the value is another dict or list, recurse
            elif isinstance(value, (dict, list)):
                data[key] = round_currency_values(value, decimals)

    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = round_currency_values(item, decimals)

    return data


def smart_round(x: float, sigfigs: int = 3) -> float:
    """
    Round *x* such that:

    * If |x| < 100, round to *sigfigs* significant figures (default 3).
    * Otherwise (|x| >= 100), round to the nearest integer.

    Parameters
    ----------
    x : float
        The number to round.
    sigfigs : int, optional
        Number of significant figures to keep when |x| < 100 (default 3).
        Must be a positive integer.

    Returns
    -------
    float | int
        The rounded value.

    Raises
    ------
    ValueError
        If *sigfigs* is not a positive integer.
    """
    if sigfigs <= 0:
        raise ValueError("sigfigs must be a positive integer")

    absx = abs(x)

    # Special-case zero (log10 is undefined for 0)
    if absx == 0:
        return 0.0

    if absx < 100:
        # sigfigs significant figures ⇒ decimals = sigfigs − 1 − floor(log10(|x|))
        decimals = sigfigs - 1 - int(math.floor(math.log10(absx)))
        return round(x, max(decimals, 0))
    else:
        # Nearest integer
        return round(x)
