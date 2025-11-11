from typing import Callable, Sequence

import numpy as np

import gpx  # GPkit‑XPRESS model objects

ROUND_FXN: dict[str, Callable[[float], float]] = {
    "floor": np.floor,
    "ceil": np.ceil,
    "round": round,
}


def round_value(val: float, scheme: str = "round") -> float:
    "return *val* rounded with *scheme* ('round', 'floor', or 'ceil')"
    return ROUND_FXN.get(scheme, np.floor)(val)


def build_uikey_gpx_map(
    active_modules: dict[str, object],
    resource_keys: Sequence[str],
) -> dict[str, gpx.Variable]:
    """
    Walk every active module and build a `{ui_key: gpx_var}` map
    for the supplied *resource_keys*.
    """
    uikey_map: dict[str, gpx.Variable] = {}
    for mod in active_modules.values():
        if hasattr(mod, "gpx_variables"):
            uikey_map.update({key: mod.gpx_variables[key] for key in resource_keys if key in mod.gpx_variables}, )
    return uikey_map


def select_highest_sensitive_var(
    capvars: Sequence[gpx.Variable],
    sensitivities: dict[gpx.Variable, float],
) -> gpx.Variable:
    """
    Return the capacity variable with the *lowest* (i.e. most
    rate‑limiting) sensitivity.
    """
    # get the sensitivities by varkey but only if in the list of rampable varkeys
    # sort by the sensitivity
    ordered = sorted(
        [(vk, sensitivities[vk]) for vk in capvars],
        key=lambda x: x[1],
        reverse=False,
    )
    # increase the most sensitive
    return ordered[0][0]


def uncertain_collapse_resources(
    best_recs: list[dict[str, str | np.float64]],
    likely_recs: list[dict[str, str | np.float64]],
    worst_recs: list[dict[str, str | np.float64]],
) -> list[dict[str, str | np.float64]]:
    "collapse the resources for the different cases into one list of resources"
    newrecs: list[dict[str, str | np.float64]] = []

    recs: dict[str, list[dict[str, str | np.float64]]] = {"min": best_recs, "max": worst_recs}

    resources: dict[str, dict[str, str | np.float64]] = {
        case: {
            str(rr["key"]): rr["value"] for rr in r
        } for case, r in recs.items()
    }

    for r in likely_recs:
        rkey = r["key"]
        if not all([v[str(rkey)] == r["value"] for v in resources.values()]):
            r.update({name: value[str(rkey)] for name, value in resources.items()})
        newrecs.append(r)

    return newrecs
