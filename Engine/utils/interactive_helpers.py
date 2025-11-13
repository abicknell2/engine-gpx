import json
from typing import TYPE_CHECKING

import numpy as np

import gpx
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import replace_unit_display_text

if TYPE_CHECKING:
    from api.interactive import InteractiveModel


def _fix_numpy_types(obj: AcceptedTypes) -> AcceptedTypes:
    """
    Recursively convert numpy int/float types in a nested structure
    to native Python int/float so that json.dumps can handle them.
    """
    if isinstance(obj, dict):
        return {k: _fix_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_numpy_types(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def sanitise_solution(interaction: dict[str, AcceptedTypes]) -> dict[str, AcceptedTypes]:
    solstr = json.dumps(interaction.solutions)
    interaction.solutions = json.loads(solstr)
    interaction.solutions = _fix_numpy_types(interaction.solutions)
    interaction.solutions = replace_unit_display_text(interaction)
    return interaction.solutions


def retrieve_round(
    varname: str,
    solution: object,
    decimals: int | None = None,
    soldict: str = "variables",
    not_found: float = 0.0,
    **kwargs: AcceptedTypes,
) -> float | None:
    """retrieves and rounds a variable from a solution by name

    If the variable is not found, returns the variable not_found value

    #TODO: look for magnitude on the variable

    Arguments
    ---------
    varname : string
        the name of the variable
    solution : gpx.Model.solution
        the solved

    Keyword Arguments
    -----------------
    decimals : int (Default=None)
        The number of decimals to round to
        If None, does not round the variable

    Returns
    -------
    float
        The rounded version of the variable if found

    """
    # TODO: finish implementation
    pass


def discretize_resources(interaction: "InteractiveModel", discvar, discval, **kwargs: AcceptedTypes) -> gpx.Model:
    # get the resources from the modules
    rescs: list[gpx.Variable] = []
    for m in interaction.active_modules.values():
        if hasattr(m, "get_production_resources"):
            rescs.extend(m.get_production_resources())

    # check original solve
    if "solve_orig" in kwargs:
        solve_orig = kwargs["solve_orig"]
    else:
        solve_orig = True

    system_module = interaction.active_modules.get("system") if hasattr(interaction, "active_modules") else None
    target_dict: dict[object, float | int | None] = {}

    relaxation_vars: set[gpx.Variable] = set()

    if system_module and getattr(system_module, "by_split", True) is False:
        mcclasses = getattr(system_module, "mcclasses", {})
        if isinstance(mcclasses, dict):
            for mclass in mcclasses.values():
                rate_var = getattr(mclass, "lam", None)
                if rate_var is None:
                    continue
                target_val = interaction.gpx_model.substitutions.get(rate_var)
                if target_val is None:
                    vkey = getattr(rate_var, "key", None)
                    if vkey is not None:
                        target_val = interaction.gpx_model.substitutions.get(vkey)
                    if target_val is None and vkey is not None:
                        target_val = interaction.gpx_model.substitutions.get(str(vkey))
                if target_val is not None:
                    target_dict[rate_var] = target_val

                for linked_lam in getattr(mclass, "all_lams", []) or []:
                    if linked_lam is not None:
                        relaxation_vars.add(linked_lam)

                line_obj = getattr(mclass, "line", None)
                line_lam = getattr(line_obj, "lam", None)
                if line_lam is not None:
                    relaxation_vars.add(line_lam)

        gpx_obj = getattr(system_module, "gpxObject", None)
        if isinstance(gpx_obj, dict):
            system_obj = gpx_obj.get("system")
        else:
            system_obj = getattr(gpx_obj, "system", None)
        system_rate_var = getattr(system_obj, "lam", None) if system_obj else None
        if system_rate_var is not None:
            sys_rate_val = interaction.gpx_model.substitutions.get(system_rate_var)
            if sys_rate_val is None:
                vkey = getattr(system_rate_var, "key", None)
                if vkey is not None:
                    sys_rate_val = interaction.gpx_model.substitutions.get(vkey)
                if sys_rate_val is None and vkey is not None:
                    sys_rate_val = interaction.gpx_model.substitutions.get(str(vkey))
            if sys_rate_val is not None:
                target_dict[system_rate_var] = sys_rate_val

        if isinstance(gpx_obj, dict):
            total_cost_obj = gpx_obj.get("totalCost")
        else:
            total_cost_obj = getattr(gpx_obj, "totalCost", None)

        num_units_var = getattr(total_cost_obj, "numUnits", None)
        if num_units_var is not None:
            relaxation_vars.add(num_units_var)

        feeders = getattr(system_module, "feeders", None) or []
        for feeder in feeders:
            feeder_lam = getattr(feeder, "lam", None)
            if feeder_lam is not None:
                relaxation_vars.add(feeder_lam)

    if target_dict:
        return interaction.gpx_model.by_rate_discretesolve(
            discrete_resources=rescs,
            target_dict=target_dict,
            solve_orig=solve_orig,
            relax_targets_on_infeasible=True,
            extra_relaxation_vars=relaxation_vars or None,
            restore_relaxations=False,
        )

    # set up the discrete solve
    return interaction.gpx_model.discretesolve(
        discrete_resources=rescs,
        target_variable=discvar,
        target_value=discval,
        solve_orig=solve_orig,
    )