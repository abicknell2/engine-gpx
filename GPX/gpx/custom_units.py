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


DEFAULT_FLIPPED_FX_RATES: dict[str, dict[str, float]] = {
    # We only support three currencies, so a symmetric 1:1 fallback is fine
    # when the live rates endpoint is unreachable.
    "USD": {"EUR": 1.0, "GBP": 1.0},
    "EUR": {"USD": 1.0, "GBP": 1.0},
    "GBP": {"USD": 1.0, "EUR": 1.0},
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
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return dict(r.json()["rates"])  # base->code


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
    flipped: dict[str, float] | None = None
    fetch_error: Exception | None = None
    try:
        base_to_code = fetch_exchange_rates(base)  # cached per base
        flipped = {
            code: (1 / rate)
            for code, rate in base_to_code.items()
            if rate and not math.isnan(rate)
        }
    except (requests.RequestException, ValueError, KeyError) as exc:
        fetch_error = exc
        # Fall back to last-known or static rates so model solves are not blocked
        # by transient network issues. The fallback is already in "flipped"
        # form: code -> base/code.
        flipped = CURRENT_RATES_BY_BASE.get(base) or DEFAULT_FLIPPED_FX_RATES.get(base)

    if not flipped:
        # If we have absolutely nothing, propagate the original exception so the
        # caller can decide how to handle it.
        if fetch_error:
            raise fetch_error
        raise RuntimeError("No FX rates available for context initialization")

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
