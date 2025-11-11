import logging
import math
from functools import lru_cache

from gpkit.units import ureg
import pint
import requests

INITIALISED_FX_BASES: set[str] = set()

# Simple units
for unit_name in ["percent", "pct", "parts_per_million", "ppm", "hour", "h", "hr"]:
    if unit_name in ureg._units:
        del ureg._units[unit_name]

ureg.define("percent = 1e-2 = pct")
ureg.define("parts_per_million = 1e-6 = ppm")
ureg.define("hour = 3600 * second = h = hr")

# Currency units
for currency_unit in ["GBP", "£", "EUR", "€", "USD", "$"]:
    if currency_unit in ureg._units:
        del ureg._units[currency_unit]

ureg.define("USD = [money] = $")
ureg.define("GBP = 1 * USD = £")
ureg.define("EUR = 1 * USD = €")

pint.set_application_registry(ureg)

ALLOWED_CURRENCIES = {"USD", "EUR", "GBP"}

# Last-installed flipped rates per base currency:
# e.g. for base='USD', {'GBP': USD/GBP, 'EUR': USD/EUR, ...}
CURRENT_RATES_BY_BASE: dict[str, dict[str, float]] = {}

# Conservative fallback rates when online refresh fails. Values are base -> code.
_OFFLINE_USD_TO_EUR = 0.92
_OFFLINE_USD_TO_GBP = 0.78
OFFLINE_RATES_BASE_TO_CODE: dict[str, dict[str, float]] = {
    "USD": {"EUR": _OFFLINE_USD_TO_EUR, "GBP": _OFFLINE_USD_TO_GBP},
    "EUR": {
        "USD": 1.0 / _OFFLINE_USD_TO_EUR,
        "GBP": _OFFLINE_USD_TO_GBP / _OFFLINE_USD_TO_EUR,
    },
    "GBP": {
        "USD": 1.0 / _OFFLINE_USD_TO_GBP,
        "EUR": _OFFLINE_USD_TO_EUR / _OFFLINE_USD_TO_GBP,
    },
}


@lru_cache(maxsize=None)
def fetch_exchange_rates(base: str = "USD") -> dict[str, float]:
    """
    Download FX rates for a base currency. Results are cached per 'base'
    for the lifetime of the Python process.

    Returns:
        Mapping base->code (e.g., {'GBP': 0.77}) meaning 1 {base} = rate {code}.
    """
    url = f"https://api.frankfurter.app/latest?base={base}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return dict(r.json()["rates"])  # base->code
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        fallback = OFFLINE_RATES_BASE_TO_CODE.get(base)
        if not fallback:
            raise
        logging.getLogger(__name__).warning(
            "Falling back to offline FX rates for %s due to %s", base, exc.__class__.__name__
        )
        return dict(fallback)


def make_fx_context(flipped_rates: dict[str, float], base: str = "USD") -> pint.Context:
    """
    Build a pint FX context from flipped rates:
      flipped_rates: code -> (to_base factor), e.g. {'GBP': USD/GBP}

    Defines: 1 {code} = flipped {base}
    """
    ctx = pint.Context("FX")
    for code, flipped in flipped_rates.items():
        if code not in ALLOWED_CURRENCIES or code == base or flipped is None or math.isnan(flipped):
            continue
        ctx.redefine(f"{code} = {flipped} {base}")
    return ctx


def refresh_fx_rates(base: str = "USD") -> dict[str, float]:
    """
    Fetch (cached) base->code rates, flip to code->(to_base), install FX context,
    store and return the flipped mapping.

    Example (base='USD'): returns {'GBP': USD/GBP, 'EUR': USD/EUR, ...}
    """
    base_to_code = fetch_exchange_rates(base)  # cached per base
    flipped = {
        code: (1 / rate)
        for code, rate in base_to_code.items()
        if rate and not math.isnan(rate)
    }
    ureg._contexts.pop("FX", None)  # remove stale context
    ureg.add_context(make_fx_context(flipped, base))
    CURRENT_RATES_BY_BASE[base] = dict(flipped)  # store a copy
    return flipped


def get_rates(base: str = "USD", *, safe: bool = True) -> dict[str, float]:
    """
    Return last-known flipped rates for `base` (code -> base/code).

    Args:
        base: Base currency for the installed FX context.
        safe: If True, never hits the network; returns {} if uncached.
              If False, ensures a refresh (may hit network) and returns rates.

    Returns:
        A copy of the flipped mapping for `base`.
    """
    if base in CURRENT_RATES_BY_BASE:
        return dict(CURRENT_RATES_BY_BASE[base])
    return {} if safe else refresh_fx_rates(base)


def convert_currency(amount, from_currency: str, to_currency: str, *, base: str = "USD") -> pint.Quantity:
    """
    Convert *amount* between currencies using the current FX context.
    Ensures the FX context for `base` is initialized on first use.
    """
    if from_currency not in ALLOWED_CURRENCIES or to_currency not in ALLOWED_CURRENCIES:
        raise ValueError("Currency not supported by this registry")

    # Lazily initialize the FX context for this base (including USD).
    if base not in INITIALISED_FX_BASES:
        # If we already have cached rates for this base, avoid network I/O
        cached = CURRENT_RATES_BY_BASE.get(base)
        if cached:
            ureg._contexts.pop("FX", None)
            ureg.add_context(make_fx_context(cached, base))
        else:
            refresh_fx_rates(base)
        INITIALISED_FX_BASES.add(base)

    q = amount if isinstance(amount, pint.Quantity) else amount * ureg(from_currency)
    if from_currency == to_currency:
        return q

    with ureg.context("FX"):
        return q.to(to_currency)
