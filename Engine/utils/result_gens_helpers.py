import collections
import logging
import numbers
from typing import TYPE_CHECKING, cast

import gpkit
import numpy as np

from api import module_plugins
from api.constants import (COST_ROUND_DEC, MAX_MINUTES_PROCESS, OTHER_ROUND_DEC, SMART_SIGFIGS, TIME_ROUND_DEC)
from api.result_generators.result_gens import ResultGenerator
from api.small_posts import decompose_process_time
from api.small_scripts import (gamma_box_points, gamma_pdf_points, get_gamma_shapescale, welch_approx)
import gpx
import gpx.manufacturing
import gpx.mfgcosts
from gpx.multiclass.mccell import MCell
import gpx.primitives
from utils.settings import Settings
from utils.types.result_gens import costResult
from utils.types.shared import (AllResults, CollectVars, Results, ResultsDict, ResultsIndex, ResultsList)
from utils.unit_helpers import convert_item_recurring_cost, smart_round

if TYPE_CHECKING:
    from api.module_types.manufacturing import Manufacturing
    from api.module_types.production_finance import ProductionFinance
    from api.result_generators.product_summary import ProductSummary


def index_entry(dispname: str, resultname: str) -> dict[str, str]:
    return {
        "name": dispname,
        "value": resultname,
    }


def var_from_attr(
    sol: gpx.Model,
    gpxobj: gpkit.Model,
    varattr: dict[str, str],
    collectvars: dict[str, gpx.Variable],
    results: Results,
    rounding: int | None = None,
    checkzero: bool = True,
    addunit: bool = True
) -> None:
    """
    Extracts and collects variables from a GPkit object based on a mapping of attribute names,
    then appends their values (optionally rounded and with units) to a results list.

    Parameters
    ----------
    sol : gpx.Model
        The solved model object containing variable values.
    gpxobj : gpkit.Model
        The GPkit object from which variables are retrieved using attribute names.
    varattr : dict[str, str]
        A dictionary mapping variable descriptions (output names) to attributes in `gpxobj`.
        Format: {'Description': 'attribute_name'}
    collectvars : dict[str, gpx.Variable]
        A dictionary where extracted GPkit variable references will be collected.
        Keys match the descriptions from `varattr`.
    results : Results
        A list-like object where variable value dictionaries will be appended. Each entry includes
        the name, value, and optionally unit of the variable.
    rounding : int, optional
        Number of decimal places to round values to. If None, no rounding is applied.
    checkzero : bool, optional (default=True)
        If True, variables with value zero will be skipped.
    addunit : bool, optional (default=True)
        If True, the variable's unit string is included in the results.

    Returns
    -------
    None
        The function updates `collectvars` and `results` in-place.
    """
    for name, attr in varattr.items():
        try:
            varkey = getattr(gpxobj, attr)
        except AttributeError:
            # catch exception where the object does not have an attribute
            logging.info(
                f"Attribute {attr} with name {name} not found in object: {gpxobj.__class__.__name__} while converting object to results"
            )
            break

        if checkzero and not varkey != 0:
            break
        collectvars[name] = varkey
        var_detail = {
            "name": name,
            'value': sol[varkey] if rounding is None else np.around(float(sol[varkey]), decimals=rounding),
        }

        if addunit:
            # add the units if requested
            var_detail["unit"] = varkey.unitstr()

        results.append(var_detail)


def combine_results(*gens: list["ResultGenerator"]) -> AllResults:
    """combine the results of multiple results creators

    Arguments
    ---------
    gens : generators
    """
    results: Results = {}
    results_index: ResultsIndex = []
    collect_vars: CollectVars = {}

    for g in gens:
        append_results(g, results=cast(ResultsList, results), results_index=results_index, collect_vars=collect_vars)
    return results, results_index, collect_vars


def append_results(
    resgen: list["ResultGenerator"],
    results: Results,
    results_index: ResultsIndex,
    collect_vars: CollectVars,
) -> None:
    """add a single result"""

    results.update(cast(ResultsDict, resgen[0]))  # update the dictionary
    results_index.append(cast(dict[str, str], resgen[1]))  # add to the list
    collect_vars.update(cast(CollectVars, resgen[2]))  # update the dictionary with new variables


def gen_cashflows(
    sol: gpkit.SolutionArray,
    finmod: "ProductionFinance",
    lamvar: gpx.Variable,
    costobj: gpx.primitives.Cost,
) -> list[dict[str, float | str | np.float16 | costResult]]:
    "creates the cashflow entries"
    # monthly production quantity
    rate = sol(lamvar).to("count/hr").magnitude
    mo_ops_hrs = finmod.get_hourly_duration(
        duration=1,
        ops_time=True,
        durationUnit="months",
        returnasmon=False,
    )  # monthly operational hours
    mtot_qty = rate * mo_ops_hrs
    # monthly variable costs
    if costobj.variableCost != 0:
        # there is a non zero variable cost. find in solution
        mtot_var_costs = mtot_qty * sol(costobj.variableCost).magnitude
    else:
        mtot_var_costs = 0.0

    # TODO:  check to see if there is a rate ramp
    rate_ramp = finmod.get_plugins(module_plugins.RateRamp)
    # tot_ramp_dur = 0  # Unused var
    ramp_sols = []

    if len(rate_ramp) > 1:
        # raise issue
        raise ValueError("Too many rate rate_ramp to process")
    elif len(rate_ramp) > 0:
        rate_ramp_instance = cast(module_plugins.RateRamp, rate_ramp[0])
        # update the ramp duration
        # tot_ramp_dur = rate_ramp.ramp_tot_duration  # Unused var
        # get the ramp solutions
        ramp_sols = rate_ramp_instance.get_ramp_cost_solutions(sol)

    # monthly recurring costs
    if costobj.recurringCost != 0:
        # non-zero recurring cost. find in solution
        mtot_recur = sol(costobj.recurringCost).magnitude * mo_ops_hrs
    else:
        mtot_recur = 0.0

    # duration in number of months
    # the duration already includes the difference to the ramps
    num_months = finmod.duration if finmod.durationUnit == "months" else 12 * finmod.duration
    if not isinstance(num_months, int):
        # months count is not an int. round to int
        logging.info(f"RESULT-GENS NPV| rounding number of months from {num_months}")
        num_months = round(num_months)

    # cashflow amounts
    # all the nonrecurring costs
    cf_nrec = sol(costobj.nonrecurringCost).magnitude  # assume there has to be nonrecurring cost
    # steady state
    cf_ss = mtot_var_costs + mtot_recur

    # cashflows objects
    # for all steady-state months
    cfobj_ss = {"cashflow": np.round(cf_ss, decimals=COST_ROUND_DEC), "output": np.round(mtot_qty)}

    # make the list
    cfs = []
    # add the ramps
    for rs in ramp_sols:
        # create the object and add to the cashflows
        cfs.extend([{
            "cashflow": np.round(rs["monthCost"], decimals=COST_ROUND_DEC),
            "output": np.round(rs["monthQuantity"]),
        }] * round(rs["duration"]))

    # add the remaining months of cashflows after the ramp
    cfs.extend([cfobj_ss] * int(num_months))

    # copy the first object
    cfs[0] = cfs[0].copy()
    # add the cashflow at zero for all the non-recurring cost
    cfs[0]["cashflow"] = np.round(cfs[0]["cashflow"] + cf_nrec)

    # TODO:  cumulative output across the months

    return cfs


def gen_entry(
    sol: gpkit.SolutionArray,
    var: gpx.Variable,
    entryname: str,
    round: int | None = None,
    withunits: bool = False
) -> dict[str, object]:
    """generate an entry for the solution dictionary
    Use with a list `extend` command

    Arguments
    ---------
    sol : gpx.Model.solution
        the solution to reference
    var : gpx.Variable
        the variable to reference
    entryname : str
        the name of the entry to use
    round : int or None (default=None)
        number of decimals to round the solution to
        does not round if None
    withunits : boolean
        create an extra entry using entryname with unist
        format <entryName>Unit

    """
    ad = {}  # dictionary to return

    if round is not None:
        ad[entryname] = np.around(sol[var], decimals=round)
    else:
        ad[entryname] = sol[var]

    if withunits:
        unitname = entryname + "Unit"  # define the key for the units
        try:
            ad[unitname] = var.units
        except AttributeError:
            # variable not specified with units say it's unitless
            ad[unitname] = "unitless"

    return ad


def combine_results(  # noqa: F811:
    sol: gpkit.SolutionArray, restocombine: list["ResultGenerator"], settings: Settings
) -> "ResultGenerator":  # TODO: Find correct type.  Function is redefined here from line 2033. 2033 is newer though, but code requires this function for tests to pass.
    """combine multiple result generators

    Arguments
    ---------
    sol : gpx.Model.Solution
        the common input model
    restocombine : list
        list of the rsults to combine

    Returns
    -------
    ResultGenerator
        single result generator with the combination
    """

    rg = ResultGenerator(sol, settings)

    for rtc in restocombine:
        # combine the results
        for rname, r in rtc.results.items():
            # go through the results
            if rname in rg.results:
                # if the result name is already in the results, append to the list
                # TODO: Refactor and tidy up this code
                if isinstance(rg.results[rname], list):
                    if isinstance(r, list):
                        rg.results[rname].extend(r)
                    elif isinstance(r, dict):
                        rg.results[rname].update(r)
                    else:
                        rg.results[rname].append(cast(dict[str, float | np.float64 | int | str], r))
                elif isinstance(rg.results[rname], dict):
                    if isinstance(r, list):
                        rg.results[rname].extend(r)
                    elif isinstance(r, dict):
                        rg.results[rname].update(r)
                    else:
                        rg.results[rname].append(cast(dict[str, float | np.float64 | int | str], r))
            else:
                # if the result name is new, add it to the results
                rg.results[rname] = r

        # create the results index
        for ri in rtc.results_index:
            if ri["name"] not in [rii["name"] for rii in rg.results_index]:
                # only add the name of the index if it is not already there
                rg.results_index.append(ri)

        # combine the collect_vars
        rg.collect_vars.update(rtc.collect_vars)

        # combine the aux variables
        rg.aux_vars.extend(rtc.aux_vars)

    return rg


def obj_from_var(
    sol: gpx.Model,
    var: gpx.Variable,
    descr: str | None = None,
    units: str | None = None,
    value: float | None = None,
    round_dec: int = 2
) -> dict[str, gpx.Variable]:
    """_summary_

    Parameters
    ----------
    sol : gpkit.SolutionArray
        _description_
    var : gpx.Variable
        variable to convert to dictionary
    descr : str, optional
        overrides the default description of the variable, by default None
    units : str, optional
        overrides default variable, by default None
    value : float, optional
        overrides a lookup in the gpkit solution object
    round_dec : int, optional
        number of decimals to round

    Returns
    -------
    dict[str, Variable]
        the dictionary representation of the variable
    """
    if value is None:
        # check to make sure var in solution
        try:
            value = sol(var).magnitude
        except KeyError:
            # variable not found in the solution. return empty
            if isinstance(var, numbers.Number):
                # if the variable is just a number, set it to that
                # make sure to convert to float for json serialization
                value = np.float64(var)
            else:
                value = 0.0

    if not units:
        # get the units from the solution
        try:
            units = var.unitstr()
        except AttributeError:
            # if can't get the units, assume it's unitless
            units = "-"

    var_as_dict = {
        "name": var.label if not descr else descr,
        "value": float(np.round(value, decimals=round_dec)),
        "unit": units,
    }

    return var_as_dict


def process_varcosts(gpxsol: gpx.Model, varcosts: dict, round_dec: int):
    """
    Processes primary variable costs.

    Returns:
      entries: list of dicts to assign to self.results[...]
      cost_objs: list of costResult objects.
    """
    entries = []
    cost_objs = []
    for name, rc in varcosts.items():
        if rc.variableCost != 0:
            # Extract value from the GP solution
            val = gpxsol["variables"][rc.variableCost.key]
            val = float(smart_round(val, sigfigs=SMART_SIGFIGS))
            entries.append({"name": name, "value": val})
            cost_objs.append(costResult(name, val, rc.variableCost))
    return entries, cost_objs


def process_labcosts(gpxsol: gpx.Model, labcosts: list, round_dec: int):
    """
    Processes labor costs.

    Returns:
      entries: list of dicts for labor costs (with " Labor Cost" appended to names)
      cost_objs: list of costResult objects with the resource set to the labor cell's name.
    """
    entries = []
    cost_objs = []
    for labobj in labcosts:
        for name, lcost in labobj.cellLabor.items():
            val = gpxsol["variables"][lcost]
            val = float(smart_round(val, sigfigs=SMART_SIGFIGS))
            full_name = f"{name} Labor Cost"
            entries.append({"name": full_name, "value": val})
            cost_objs.append(costResult(full_name, val, lcost, resource=name))
    return entries, cost_objs


def process_invholdcost(gpxsol: gpx.Model, invholdcost, round_dec: int):
    """
    Processes the inventory holding cost.

    Returns:
      entry: a dict for self.results
      cost_obj: a costResult object
      totinvcost: the numeric value extracted (for use in summaries).
    """
    totinvcost = gpxsol["variables"].get(invholdcost.variableCost, 0)
    val = float(smart_round(totinvcost, sigfigs=SMART_SIGFIGS))
    entry = {"name": "Inventory Holding Cost", "value": val}
    cost_obj = costResult("Inventory Holding Cost", val, invholdcost.variableCost)
    return entry, cost_obj, totinvcost


def process_recurcosts(gpxsol: gpx.Model, recur_costs: dict, round_dec: int, settings: Settings):
    """
    Processes primary recurring costs.

    Returns:
      entries: list of dicts (with unit "$/hr")
      cost_objs: list of costResult objects.
    """
    entries = []
    cost_objs = []
    for name, rc in recur_costs.items():
        if rc.recurringCost != 0:
            var = rc.recurringCost
            disp = convert_item_recurring_cost(gpxsol(var), settings)
            val = float(np.around(disp.magnitude, round_dec))
            entries.append({
                "name": name,
                "value": val,
                "unit": f"{settings.default_currency}/{settings.rate_unit}",
            })
            cost_objs.append(costResult(name, val, var))
    return entries, cost_objs


def process_cell_costs(gpxsol: gpx.Model, cell_costs: dict, round_dec: int, settings: Settings):
    """
    Processes cell recurring costs.

    Returns:
      entries: list of dicts (with unit "$/hr")
      cost_objs: list of costResult objects where the name is augmented with " Recurring Cost"
                and resource is set to the key.
    """
    entries = []
    cost_objs = []
    for name, cc in cell_costs.items():
        if cc.recurringCost != 0:
            var = cc.recurringCost
            disp = convert_item_recurring_cost(gpxsol(var), settings)
            val = float(np.around(disp.magnitude, decimals=round_dec))
            entries.append({
                "name": name,
                "value": val,
                "unit": f"{settings.default_currency}/{settings.rate_unit}",
            })
            cost_objs.append(costResult(f"{name} Recurring Cost", val, var, resource=name))
    return entries, cost_objs


def process_fs_cost(gpxsol: gpx.Model, fs_cost, round_dec: int, settings: Settings):
    """
    Process the floorspace recurring cost from a GPkit model solution.

    Parameters
    ----------
    gpxsol : gpx.Model
        A solved GPkit model from which variable values are extracted.
    fs_cost : cost object
        A cost object for the total floorspace.
    round_dec : int
        Number of decimal places to round the cost to.

    Returns
    -------
    tuple of (dict or None, costResult or None)
        A tuple containing:
        - `entry`: a dictionary for results display, or None if not applicable.
        - `cost_obj`: a `costResult` object with metadata, or None.
    """
    if fs_cost and hasattr(fs_cost, "recurringCost"):
        var = fs_cost.recurringCost
        # conditional check this way avoids accidentally creating a constraint
        if var != 0:
            disp = convert_item_recurring_cost(gpxsol(var), settings)
            val = float(np.around(disp.magnitude, decimals=round_dec))
            if val == 0:
                return None, None

            entry = {
                "name": "Floorspace Recurring Cost",
                "value": val,
                "unit": f"{settings.default_currency}/{settings.rate_unit}",
            }
            cost_obj = costResult("Floorspace Recurring Cost", val, var, resource="Floorspace Recurring Cost")
            return entry, cost_obj

    return None, None


def process_nr_cell_costs(gpxsol: gpx.Model, module: "Manufacturing", round_dec: int):
    """
    Processes the cell cost entries for non-recurring costs.
    
    Returns:
      - cell_entries: list of dictionaries to add to results.
      - cell_cost_objs: list of costResult objects.
      - cell_collect: dict of variable mappings for collect_vars.
    """
    sol = gpxsol["variables"]
    cell_entries = []
    cell_cost_objs = []
    cell_collect = {}
    for _, name in module.get_cells_in_order():
        try:
            cell_costs = module.gpxObject["cellCosts"]
            if isinstance(cell_costs, dict):
                c = cell_costs.get(name)
            elif isinstance(cell_costs, list):
                c = next((item for item in cell_costs if isinstance(item, dict) and item.get("name") == name), None)
            else:
                c = None

            if c is None:
                logging.warn(f"RESULT-GENS NRCostResults| ERROR {name} skipped because cell cost not found")
                continue

            var = c.nonrecurringCost
            value = np.around(sol[var], decimals=round_dec)
            cell_entries.append({"name": name, "value": float(value)})
            cell_collect[f"{name} Subtotal Cell Cost"] = var
            cell_cost_objs.append(costResult(f"{name} Non-Recurring Cost", float(value), var, resource=name))
        except KeyError:
            logging.warn(f"RESULT-GENS NRCostResults| ERROR {name} skipped due to KeyError processing results")
            continue
    return cell_entries, cell_cost_objs, cell_collect


def process_nr_results_dict(
    gpxsol: gpx.Model, gpxobj: dict, solvar: str = "variables", decimals: int = OTHER_ROUND_DEC
):
    """
    Processes a dictionary of non-recurring cost results (e.g. toolCosts or nonRecurringCost)
    and returns a list of result dictionaries.
    """
    sol = gpxsol[solvar]
    entries = []
    for name, k in gpxobj.items():
        try:
            value = 0 if isinstance(k.nonrecurringCost, int) and k.nonrecurringCost == 0 else np.around(
                sol[k.nonrecurringCost.key], decimals=decimals
            )
            entries.append({"name": name, "value": float(value)})
        except KeyError:
            continue
    return entries


def build_cost_breakout_list(cost_list: list[costResult], cost_type: str, transform_func=lambda x: x):
    """
    Processes a list of costResult objects to produce a breakout list.
    The transform_func lets you adjust the cost value (for example, amortization).
    """
    breakout = []
    for c in cost_list:
        breakout.append({
            "name": c.displayname,
            "value": float(np.round(transform_func(c.value), decimals=COST_ROUND_DEC)),
            "costType": cost_type,
            "resource": c.resource,
        })
    return breakout


def build_labor_cost_entries(gpxsol: gpx.Model, labcosts: list, settings: Settings) -> list[dict]:
    """
    For every labor cost item (from each labor object’s cellLabor dictionary),
    builds a result dictionary with the cell name, its cost value (using gpxsol["variables"].get),
    and a unit of "$".
    """
    entries = []
    for labobj in labcosts:
        for name, lcost in labobj.cellLabor.items():
            # Use .get() so that if a cost is absent we get a default 0.
            value = gpxsol["variables"].get(lcost, 0)
            entries.append({"name": name, "value": value, "unit": settings.default_currency_iso})
    return entries


def collect_labor_cost_vars(labcosts: list) -> dict:
    """
    Returns a dictionary mapping each cell’s labor cost variable to a key in the form:
    "<cellname> Labor Cost".
    """
    collect = {}
    for labobj in labcosts:
        for cellname, lcost in labobj.cellLabor.items():
            collect[str(cellname) + " Labor Cost"] = lcost
    return collect


def sort_labor_entries(entries: list[dict], cellorderdict: dict[str, int]) -> list[dict]:
    """
    Sorts the labor cost entries according to the ordering defined in cellorderdict.
    """
    return sorted(entries, key=lambda c: cellorderdict.get(str(c["name"]), 0))


def compute_cell_totals(gpxsol: gpx.Model, labcosts: list, round_dec: int) -> dict:
    """
    For each cell (using the first labor cost object's cells_headcount), compute:
      - Total Headcount as gpxsol(c[0].m) * gpxsol(c[1])
      - Total Labor Hours as np.round(gpxsol(c[0].tnu).to("hr") * gpxsol(c[1]), decimals=round_dec)
    Returns a dictionary mapping the cell name to these totals.
    """
    cell_totals = {}
    # Assume labcosts[0].cells_headcount exists and is a dictionary
    for name, c in labcosts[0].cells_headcount.items():
        headcount = gpxsol(c[0].m) * gpxsol(c[1])
        labor_hours = np.round(gpxsol(c[0].tnu).to("hr") * gpxsol(c[1]), decimals=round_dec)
        cell_totals[name] = {"Total Headcount": headcount, "Total Labor Hours": labor_hours}
    return cell_totals


def build_cell_auxiliary(cell_totals: dict, round_dec: int) -> list[dict]:
    """
    Converts cell_totals into a list of auxiliary entries for each cell.
    For each cell and each total (headcount and labor hours), it produces a dictionary
    with the cell name and key, the rounded value (using the .magnitude of the quantity),
    an appropriate unit ("count" for headcount, "hr" for hours), and other metadata.
    """
    aux_entries = []
    for name, totals in cell_totals.items():
        for key, val in totals.items():
            aux_entries.append({
                "name": f"{name} {key}",
                "value": np.round(val.magnitude, decimals=round_dec),
                "unit": "count" if "headcount" in key.lower() else "hr",
                "sensitivity": 0.0,
                "source": "Calculated Value",
                "category": [],
            })
    return aux_entries


def compute_system_totals(cell_totals: dict) -> tuple[float, float]:
    """
    Computes overall system totals:
      - Sum of headcounts (using the raw totals)
      - Sum of labor hours (using the .magnitude values)
    Returns a tuple (total_headcount, total_hours).
    """
    total_headcount = np.sum([c["Total Headcount"] for c in cell_totals.values()])
    total_hours = np.sum([c["Total Labor Hours"].magnitude for c in cell_totals.values()])
    return total_headcount, total_hours


def build_system_auxiliary(
    total_headcount: float,
    total_hours: float,
    tot_labcost: float,
    include_headcount: bool,
    round_dec: int,
    settings: Settings,
) -> list[dict]:
    """
    Builds auxiliary entries for system totals. Always includes "Unit Total Labor Hours"
    and "Unit Total Labor Cost". If include_headcount is True, also includes "Line Total Headcount".
    """
    aux = []
    items = {
        "Unit Total Labor Hours": (total_hours, "hr"),
        "Unit Total Labor Cost": (tot_labcost, settings.default_currency_iso),
    }
    if include_headcount:
        items["Line Total Headcount"] = (total_headcount, "count")
    for name, (val, unit_str) in items.items():
        aux.append({
            "name": name,
            "value": np.round(val, decimals=round_dec),
            "unit": unit_str,
            "sensitivity": 0.0,
            "source": "Calculated Value",
            "category": [],
        })
    return aux


def compute_qna_wip(sol: gpx.Model, cell) -> tuple:
    """
    Computes the work-in-process values for a QNA cell.
    Returns a tuple (wip, wipQueue, wipStation).
    """
    # Total WIP: throughput * total time (converted to hours)
    wip = sol(cell.lam) * sol(cell.W).to("hr")
    if cell.queue_at_cell:
        wipq = sol(cell.lam) * sol(cell.Wq).to("hr")
        # Use cell.k if defined among the variables, else use a simpler formula.
        if cell.k in sol["variables"]:
            wips = sol(cell.rho) * sol(cell.m) * sol(cell.k)
        else:
            wips = sol(cell.rho) * sol(cell.m)
    else:
        wipq = 0 * gpkit.units('count')
        wips = wip
    return wip, wipq, wips


def build_qna_cell_result_dict(
    sol: gpx.Model, cell, name: str, cell_number: int, other_round_dec: int, sens_round_dec: int
) -> dict:
    """
    Builds a result dictionary for a QNA cell using the full model sol.
    """
    wip, wipq, wips = compute_qna_wip(sol, cell)

    cell_w = sol['variables'][cell.W.key] if not cell.secondary_cell else 0.0

    result = {
        "name":
        name,
        "cellNumber":
        cell_number,
        "numWorkstations":
        float(np.round(sol["variables"][cell.m.key], decimals=other_round_dec)),
        "wip":
        np.round(float(sol['variables'][cell.lam.key] * cell_w / 60.0), decimals=other_round_dec),
        "wipQueue": (np.round(wipq.magnitude, decimals=other_round_dec) if hasattr(wipq, "magnitude") else wipq),
        "wipStation":
        np.round(wips.magnitude, decimals=other_round_dec),
        "queueingTime":
        np.around(sol["variables"][cell.Wq.key], decimals=other_round_dec),
        "queueingTimeUnit":
        "".join(str(cell.Wq.units).split()[1:]),
        "cellTimeSensitivity":
        np.around(sol['sensitivities']['variables'].get(cell.nu.key, 0.0), decimals=sens_round_dec),
        "cellVariationSensitivity":
        np.around(sol['sensitivities']['variables'].get(cell.chicv.key, 0.0), decimals=sens_round_dec),
        "utilization":
        np.around(sol["variables"][cell.rho], decimals=other_round_dec),
        "processTime":
        np.around(sol["variables"][cell.tnu], decimals=other_round_dec),
        "cellFlowTime":
        np.around(cell_w, decimals=OTHER_ROUND_DEC),
        "cellCVD":
        np.around(np.sqrt(sol["variables"][cell.c2d]), decimals=other_round_dec),
        "queueingBeforeCell":
        str(cell.queue_at_cell),
    }
    return result


def update_qna_collect_vars(cell, name: str) -> dict:
    """
    Returns a dictionary with the key variables for a QNA cell.
    """
    return {
        f"{name} Workstations": cell.m,
        f"{name} Queueing Time": cell.Wq,
        f"{name} Process Time": cell.tnu,
        f"{name} Total Flow Time": cell.W,
        f"{name} Utilization": cell.rho,
    }


def compute_accumulated_flow(cell_results: list[dict], other_round_dec: int) -> None:
    """
    Computes accumulated flow time for a sorted list of cell results.
    Updates each dictionary in place by adding the key "accumFlow".
    """
    if cell_results:
        cell_results[0]["accumFlow"] = cell_results[0]["cellFlowTime"]
        for i in range(1, len(cell_results)):
            cell_results[i]["accumFlow"] = np.around(
                float(cell_results[i - 1]["accumFlow"]) + float(cell_results[i]["cellFlowTime"]),
                decimals=other_round_dec,
            )


def compute_mcell_wip(sol: dict, mc) -> float:
    """
    Computes the aggregated work-in-process (WIP) for an MCell by summing
    the WIP values of each subcell. Expects sol to be the variables dictionary.
    """
    wip_vals = []
    for subcell in mc.cells:

        Wkey = getattr(subcell.W, "key", None)
        if Wkey is None:
            continue

        wip_vals.append(sol[Wkey] * sol[subcell.lam.key] / 60.0)
    return float(np.sum(wip_vals)) if wip_vals else 0.0


def build_mcell_cell_result_dict(
    sol: dict, sens: dict, mc, name: str, other_round_dec: int, sens_round_dec: int
) -> dict:
    """
    Builds and returns a result dictionary for an MCell using the provided variables dictionary
    (sol) and the sensitivities dictionary (sens).

    Parameters:
      - sol: A dictionary of variables (e.g. self.gpxsol["variables"])
      - sens: A dictionary of sensitivity values (e.g. self.gpxsol["sensitivities"]["variables"])
      - mc: An MCell object.
      - name: The multiproduct cell's name.
      - other_round_dec: Decimal rounding for most quantities.
      - sens_round_dec: Decimal rounding for sensitivity quantities.

    Returns:
      A dictionary with aggregated metrics such as utilization, number of workstations, queueing time,
      and sensitivityTime.
    """
    wip_sum = compute_mcell_wip(sol, mc)
    return {
        "name": name,
        "utilization": np.around(sol[mc.rho_bar.key], decimals=other_round_dec),
        "numWorkstations": np.around(sol[mc.m.key], decimals=other_round_dec),
        "queueingTime": np.around(sol[mc.Wq_bar.key], decimals=other_round_dec),
        "queueingTimeUnit": "".join(str(mc.Wq_bar.units).split()[1:]),
        "arrivalCV": np.around(np.sqrt(sol[mc.c2a_bar.key]), decimals=other_round_dec),
        "departureCV": np.around(np.sqrt(sol[mc.c2d_bar.key]), decimals=other_round_dec),
        "wip": np.around(wip_sum, decimals=other_round_dec),
        "sensitivityTime": np.around(sens[mc.eta_t], decimals=sens_round_dec),
    }


def update_mcell_collect_vars(mc, name: str) -> dict:
    """
    Returns a dictionary with the key variables for an MCell.
    """
    return {
        f"{name} Workstations": mc.m,
        f"{name} Queueing Time": mc.Wq_bar,
        f"{name} Total Utilization": mc.rho_bar,
    }


def build_product_summary(resultslist: list[ResultGenerator]) -> dict:
    """
    Aggregates the summary results from a list of ResultGenerator instances.

    Each generator is assumed to have a dictionary called `summary_res`.
    Returns an aggregated dictionary.
    """
    summary = {}
    for rg in resultslist:
        # Merge each result's summary_res into a single dictionary.
        # If keys collide later, later keys will override earlier ones.
        summary.update(rg.summary_res)
    return summary


def flatten_product_summaries(prod_sums: list["ProductSummary"]) -> dict:
    """
    Flattens the product summaries for a system by prepending the product name to each key.

    Each ProductSummary is assumed to have its results under the key "productSummary"
    as a list containing a single summary dictionary. One key is assumed to be "productName".

    For each summary (other than the productName), the new key is "productName | originalKey"
    and the value is extracted (if the value has an attribute `value`, its value is used;
    otherwise, if it is an instance of a NumPy floating value it is converted to float; otherwise,
    the value is used as-is).
    """
    flattened = {}
    for psum in prod_sums:
        # Get the list of summary dictionaries from the product summary results.
        product_summary = psum.results.get("productSummary", [])
        if product_summary:  # Ensure there is at least one summary dictionary.
            ps_dict = product_summary[0]
            prodname = ps_dict.get("productName", "Unknown Product")
            for key, val in ps_dict.items():
                if key != "productName":
                    new_key = f"{prodname} | {key}"
                    if hasattr(val, "value"):
                        new_val = val.value
                    elif isinstance(val, np.floating):
                        new_val = float(val)
                    else:
                        new_val = val
                    flattened[new_key] = new_val
    return flattened


def compute_tooling_metrics(gpxsol: gpx.Model, tool: gpx.manufacturing.ConwipTooling) -> tuple[float, float, float]:
    """
    Computes key metrics for a tooling object:
      - tot_W: total tool flow time (in minutes) aggregated via tool.W_stop
      - first_Wq: queueing time (in minutes) from the first cell (tool.cell_start.Wq)
      - tool_util: utilization defined as (tot_W - first_Wq) / tot_W
    """
    tot_W = np.sum(gpxsol(tool.W_stop)).to("min").magnitude
    first_Wq = gpxsol(tool.cell_start.Wq).to("min").magnitude
    tool_util = (tot_W - first_Wq) / tot_W if tot_W != 0 else 0.0
    return tot_W, first_Wq, tool_util


def build_tooling_result(
    tool: gpx.manufacturing.ConwipTooling, gpxsol: gpx.Model, tool_name=None, tool_costs=None
) -> dict:
    """
    Builds a dictionary of tooling results for a given tool.
    Uses the GP solution’s variables (i.e. gpxsol["variables"]) to extract the cost and tool count.
    """
    sol_vars = gpxsol["variables"]
    name = tool.name if tool_name is None else tool_name
    nonrecurringCost = tool.nonrecurringCost if tool_costs is None else tool_costs['nonrecurringCost']
    tot_W, first_Wq, tool_util = compute_tooling_metrics(gpxsol, tool)
    return {
        "name": name,  # assuming the tool has an attribute 'name'
        "totalCost": np.round(sol_vars(nonrecurringCost).magnitude, decimals=1),  # TODO: Should this be being changed?
        "toolCount": np.round(sol_vars(tool.L).magnitude, decimals=1),  # TODO: Should this be being changed?
        "utilization": 0.0,  # TODO: Should this be 0.0?
        "toolFlowTime": np.round(tot_W, decimals=OTHER_ROUND_DEC),
    }


def update_tooling_collect(tool: gpx.manufacturing.ConwipTooling, tool_name=None, tool_costs=None) -> dict:
    """
    Returns a dictionary of key cost variables for the tool
    for updating the overall collect_vars.
    """
    name = tool.name if tool_name is None else tool_name
    nonrecurringCost = tool.nonrecurringCost if tool_costs is None else tool_costs['nonrecurringCost']
    return {
        f"{name} Subtotal Tool Cost": nonrecurringCost,
        f"{name} Count of Tools": tool.L,
    }


def build_tooling_aux(tool: gpx.manufacturing.ConwipTooling, gpxsol: gpx.Model, tool_name=None) -> dict:
    """
    Builds an auxiliary variable entry for a tool’s utilization.
    """
    tot_W, first_Wq, tool_util = compute_tooling_metrics(gpxsol, tool)
    name = tool.name if tool_name is None else tool_name
    return {
        "name": f"{name} Utilization",
        "value": np.round(tool_util * 100.0, decimals=COST_ROUND_DEC),
        "unit": "%",
        "sensitivity": 0,
        "source": "Calculated Value",
        "category": [],
    }


def process_total_cost(unitcost: gpx.primitives.TotalCost, sol_vars: dict, prod_costs: dict) -> tuple[dict, dict]:
    """
    Processes the cost types within the TotalCost object.
    For now we use a named tuple for nonrecurringCost.
    
    Returns a tuple:
      (result_dict, collect_dict)
    where result_dict maps the display name (e.g., "Total Non-Recurring Cost")
    to the rounded value, and collect_dict maps that display name to the corresponding variable.
    """
    # Add the Total non-recurring cost
    totnr_cost_var = unitcost.nonrecurringCost
    totnr_cost = sol_vars(totnr_cost_var).magnitude

    # find the sum of all the product variable costs
    totprodvarcost = np.sum([
        getattr(sol_vars(tt.nonrecurringCost), 'magnitude', sol_vars(tt.nonrecurringCost))
        if tt.nonrecurringCost != 0 else 0 for tt in prod_costs.values()
    ])

    # correct the cost results
    totnr_cost -= totprodvarcost

    return np.round(totnr_cost, decimals=COST_ROUND_DEC), totnr_cost


def compute_offshift_points(sol: gpx.Model, c: gpx.Variable, cname: str, other_round_dec: int) -> list[dict]:
    """
    Given an offshift variable c for a cell, computes intermediate quantities and returns a list
    of dictionaries representing time–quantity points.
    
    It computes:
      - lambar = sol(c.lam) / sol(c.timeratio)
      - ybar = sol(c.lam) * (sol(c.W).to("hr"))
      - toff = sol(c.toff)
      - ss = (2*ybar - toff*lambar) / 3
      - ystar = ss + toff*lambar
      - ton = sol(c.ton)
      - endtime = (ton + toff).magnitude
    Then returns points at time 0, at ton, at endtime (and 24 if needed),
    with the cell name appended.
    """
    lambar = sol(c.lam) / sol(c.timeratio)
    ybar = sol(c.lam) * sol(c.W).to("hr")
    toff = sol(c.toff)
    ss = (2.0 * ybar - toff * lambar) / 3.0
    ystar = ss + toff * lambar
    ton = sol(c.ton)
    endtime = (ton + toff).magnitude

    pts = [
        {
            "time": np.float64(0.0),
            "quantity": np.around(ss.magnitude, decimals=other_round_dec)
        },
        {
            "time": np.float64(ton.magnitude),
            "quantity": np.around(ystar.magnitude, decimals=other_round_dec)
        },
        {
            "time": np.float64(endtime),
            "quantity": np.around(ss.magnitude, decimals=other_round_dec)
        },
    ]
    if endtime < 24:
        pts.append({"time": np.float64(24), "quantity": np.around(ss.magnitude, decimals=other_round_dec)})
    # Append the cell name to each dictionary.
    pts = [{**pt, "cell": cname} for pt in pts]
    return pts


def update_offshift_collect(c: gpx.Variable, cname: str) -> dict:
    """
    Returns a dictionary to update collect_vars with the offshift time for a given cell.
    """
    return {f"{cname} Off-Shift Time": c.ton}


def build_process_result_entry(
    p: gpx.primitives.Process, name: str, sol: gpx.Model, target_units: str, time_round_dec: int
) -> dict:
    """
    Builds a result dictionary for a single process.
    
    If the process object p has a pre-processing time (tpre) it is used; otherwise,
    p.t is used. Also computes the standard deviation (stdev) in the target unit and
    uses an external function decompose_process_time() to produce a breakdown.
    
    Parameters:
      - p: Process object.
      - name: The process name.
      - sol: The GP solution (used as a callable to get variable values and
             sol["variables"] for key lookup).
      - target_units: The target unit as a string (e.g., "hr" or "min").
      - time_round_dec: Decimal places for rounding.
    
    Returns:
      A dictionary with keys "type", "name", "processTime", "stdev", "unit", and "breakdown".
    """
    # Determine process time using tpre if defined; otherwise, use t.
    if hasattr(p, "tpre"):
        proc_time = float(np.around(sol(p.tpre).to(target_units).magnitude, decimals=time_round_dec))
    else:
        proc_time = float(np.around(sol(p.t).to(target_units).magnitude, decimals=time_round_dec))
    stdev_val = float(np.around(sol(p.stdev).to(target_units).magnitude, decimals=time_round_dec))
    # decompose_process_time is assumed to be defined elsewhere.
    breakdown = decompose_process_time(name, p, sol)
    return {
        "type": name,
        "name": name,
        "processTime": proc_time,
        "stdev": stdev_val,
        "unit": target_units,
        "breakdown": breakdown,
    }


def update_process_collect(p: gpx.primitives.Process, sol: gpx.Model) -> dict:
    """
    Returns a dictionary for updating collect_vars with the process variable.
    """
    return {str(p.t.key): p.t}


def build_line_summary(
    sol_vars: dict, fabline: gpx.manufacturing.FabLine, var_attr: dict[str, str], rounding: int, collectvars: dict
) -> list:
    """
    Extracts line summary attributes (like lam, W, L) from fabline and updates collectvars.
    Returns a list of summary dictionaries.
    """
    summary = []
    var_from_attr(
        sol=sol_vars, gpxobj=fabline, varattr=var_attr, collectvars=collectvars, results=summary, rounding=rounding
    )
    return summary


def compute_flow_time(sol_vars: dict, fabline: gpx.manufacturing.FabLine, max_minutes_flow: float,
                      rounding: int) -> tuple[float, str]:
    """
    Computes the overall flow time from fabline.W using sol_vars.
    Converts to minutes; if too large, converts to hours.
    
    Returns:
      (flow_time_value, flow_units)
    """
    flow_time_obj = sol_vars(fabline.W)
    flow_time_obj = flow_time_obj.to("min")
    if flow_time_obj.magnitude > max_minutes_flow:
        flow_time_obj = flow_time_obj.to("hr")
    flow_units = str(flow_time_obj.units)
    flow_time = flow_time_obj.magnitude
    return flow_time, flow_units, flow_time_obj


def compute_flow_distribution(sol_vars: dict, fabline: gpx.manufacturing.FabLine,
                              time_round_dec: int) -> tuple[dict, list]:
    """
    Computes the gamma distribution for the flow time in fabline.
    For each cell in fabline.cells, it computes gamma parameters via small_scripts.get_gamma_shapescale.
    Then approximates the overall distribution using small_scripts.welch_approx.
    Returns a tuple (probs, pdf_points), where probs is the dictionary returned by gamma_box_points
    and pdf_points is the raw list from gamma_pdf_points.
    """
    dists = [get_gamma_shapescale(mean=sol_vars[c.W] / 60.0, cv2=sol_vars[c.c2d]) for c in fabline.cells]
    dist = welch_approx(dists)
    probs = gamma_box_points(dist)
    pdf_points = gamma_pdf_points(dist, points=100, upper=4, round=time_round_dec)
    return probs, pdf_points


def assign_flow_distribution(probs: dict, pdf_points: list, flow_units: str) -> dict:
    """
    Filters and updates the flow distribution data.
    
    - Assigns the given flow_units to probs under the key "unit".
    - Filters out any PDF points with infinite probability.
    - Assigns the filtered list to probs under the key "pdfPoints".
    
    Returns a dictionary of the form {"flowTime": probs}.
    """
    probs["unit"] = flow_units
    filtered_pdf = []
    for pp in pdf_points:
        if pp["probability"] != np.inf:
            new_item = {**pp, "unit": flow_units}
            filtered_pdf.append(new_item)
    probs["pdfPoints"] = filtered_pdf
    return {"flowTime": probs}


def collect_cell_floor_space(fscosts: gpx.mfgcosts.FloorspaceCost) -> dict[str, gpx.Variable]:
    """
    Returns a dictionary mapping each cell name to its floor space variable.
    """
    return {cname: var for cname, var in fscosts.cell_floor_space.items()}


def collect_line_floor_space(fscosts: gpx.mfgcosts.FloorspaceCost) -> dict[str, gpx.Variable]:
    """
    Returns a dictionary mapping line-level floor space cost descriptors to variables.
    For each descriptor (e.g. "Non-Recurring" and "Recurring") if the corresponding attribute is nonzero,
    it creates a key "Total Floor Space {desc} Cost".
    Also returns the total floor space variable under the key "Total Floor Space".
    """
    collect = {}
    collects = {
        "Non-Recurring": "nonrecurringCost",
        "Recurring": "recurringCost",
    }
    for desc, attrname in collects.items():
        fscvar = getattr(fscosts, attrname, 0)
        if fscvar != 0:
            key = f"Total Floor Space {desc} Cost"
            collect[key] = fscvar
    # Total area
    collect["Total Floor Space"] = fscosts.totalFloorSpace
    return collect


def build_floor_space_summary(gpxsol: gpx.Model, total_floor_var, rounding: int) -> dict:
    """
    Computes the total floor space value and unit from the GP solution.
    Returns a dictionary for summary results.
    """
    total_area = gpxsol(total_floor_var)
    summary = {
        "Total Floor Space": np.round(total_area.magnitude, decimals=rounding),
        "Total Floor Space Units": total_area.units.__str__().replace("**", "^"),
    }
    return summary


def update_cellresults_with_floor_space(
    gpxsol: gpx.Model, cell_results: list[dict], fscosts: gpx.mfgcosts.FloorspaceCost
) -> None:
    """
    Updates each cell's result dictionary in cell_results with its floor space data,
    if available in fscosts.cell_floor_space.
    """
    for cr in cell_results:
        name = cr.get("name")
        if name in fscosts.cell_floor_space:
            cfs = gpxsol(fscosts.cell_floor_space[name])
            cr["totalFloorSpace"] = np.round(cfs.magnitude, decimals=OTHER_ROUND_DEC)
            cr["totalFloorSpaceUnits"] = cfs.units.__str__().replace("**", "^")


def compute_tool_utilization(gpxsol, tool):
    """
    Computes the total tool flow time (tot_W) from tool.W_stop (in minutes),
    the first cell queueing time from tool.cell_start.Wq (in minutes), and the tool
    utilization as (tot_W - first_Wq) / tot_W.
    Returns a tuple: (tot_W, first_Wq, utilization)
    """
    tot_W = np.sum(gpxsol(tool.W_stop)).to("min").magnitude
    first_Wq = gpxsol(tool.cell_start.Wq).to("min").magnitude
    utilization = (tot_W - first_Wq) / tot_W if tot_W != 0 else 0.0
    return tot_W, first_Wq, utilization


def build_tool_result_entry(name, t, gpx_tool, gpxsol):
    """
    Builds a dictionary of tooling results for one tool.

    Uses the GP solution's variable dictionary (gpxsol["variables"]) to obtain cost and count values.
    """
    sol_vars = gpxsol["variables"]
    tot_W, first_Wq, util = compute_tool_utilization(gpxsol, gpx_tool)
    return {
        "name":
        name,
        "totalCost":
        0 if isinstance(t.nonrecurringCost, int) and t.nonrecurringCost == 0 else
        np.around(sol_vars[t.nonrecurringCost.key], decimals=COST_ROUND_DEC),
        "toolCount":
        np.around(sol_vars[t.L.key], decimals=OTHER_ROUND_DEC),
        'utilization':
        np.round(util * 100.0, decimals=OTHER_ROUND_DEC),
        "toolFlowTime":
        np.round(tot_W, decimals=OTHER_ROUND_DEC),
    }


def build_tool_aux_entry(name, gpxsol, gpx_tool):
    """
    Builds an auxiliary variable entry for the tool's utilization.
    """
    _, _, util = compute_tool_utilization(gpxsol, gpx_tool)
    return {
        "name": f"{name} Utilization",
        "value": np.round(util * 100.0, decimals=COST_ROUND_DEC),
        "unit": "%",
        "sensitivity": 0,
        "source": "Calculated Value",
        "category": [],
    }


def update_tool_collect(name, t):
    """
    Returns a dict for updating collect_vars for the tool.
    """
    return {
        f"{name} Subtotal Tool Cost": t.nonrecurringCost,
        f"{name} Count of Tools": t.L,
    }


def get_process_location(process_flow: list, feeder_processes: list, secondary_processes: list = None) -> dict:
    """
    Returns a mapping of process IDs to a ProcessLocation tuple.
    ProcessLocation is represented as (is_main, feeder), where:
      - For p in process_flow or secondary_processes: is_main=True, feeder=None
      - For p in feeder_processes: is_main=False, feeder is p["feederLine"]
    """
    ProcessLocation = lambda is_main, feeder: (is_main, feeder)

    proc_loc = {p["id"]: ProcessLocation(True, None) for p in process_flow}

    if secondary_processes:
        proc_loc.update({p["id"]: ProcessLocation(False, None) for p in secondary_processes})

    proc_loc.update({p["id"]: ProcessLocation(False, p["feederLine"]) for p in feeder_processes})

    return proc_loc


def get_feeders_feed_locs(feeder_lines: list) -> dict:
    """
    Returns a dict mapping feeder line id to the process location (tuple) of its 'to' field.
    """
    # Assume each feeder line dict has an "id" and a "to" field.
    # Process location for a feeder is looked up later from the process location mapping.
    return {fl["id"]: fl["to"] for fl in feeder_lines}


def build_feeder_downstreams(feedersfeedlocs: dict, proc_loc: dict) -> dict:
    """
    For each feeder line id (key from feedersfeedlocs), recursively follow its feeder until
    a main process is reached, collecting downstream feeder ids.
    Returns a dict mapping each feeder line id to a list of its downstream feeder ids.
    """
    downstream = {}
    for fl in feedersfeedlocs.keys():
        ds = []
        cur = fl
        # Loop until the process location for the current feeder indicates it is main.
        while True:
            # Look up the process id from feedersfeedlocs: this is the 'to' field.
            pid = feedersfeedlocs[cur]
            # Get the ProcessLocation tuple from proc_loc.
            loc = proc_loc.get(pid, (True, None))
            if loc[0]:
                break
            cur = loc[1]  # the feeder id from the tuple
            ds.append(cur)
        downstream[fl] = ds
    return downstream


def compute_feeder_flows(gpxsol: any, gpx_feeder_lines: dict, feeder_lines: list) -> tuple:
    """
    Computes and returns three dictionaries:
      - flprocflow: mapping feeder line id to its process flow list (from feeder_processes).
      - fflowtimes: mapping feeder line id to its total flow time (summed from its cells, in target unit).
      - fendqueues: mapping feeder line id to its batch (queue) time.
    Uses the full solution object (gpxsol) and expects target unit to be "hr".
    """
    target_unit = "hr"  # as per original code
    flprocflow = {}
    fflowtimes = {}
    fendqueues = {}
    # Assume that feeder_processes is available in the calling scope.
    for fl in feeder_lines:
        flname = fl["id"]
        # Process flow for this feeder line: filter feeder_processes by feederLine == flname.
        pflow = [p for p in fl.get("processes", []) if p.get("feederLine") == flname]
        # If not provided in the feeder line dict, you may rely on external data.
        flprocflow[flname] = pflow

        # Get the GP feeder line.
        gpxfl = gpx_feeder_lines.get(flname)
        # if there is no gpx line (e.g. cost only) skip results
        if not gpxfl:
            continue

        cell_times = [gpxsol(c.W).to(target_unit).magnitude for c in gpxfl.cells]
        feeder_flow_time = np.sum(cell_times)
        fflowtimes[flname] = feeder_flow_time

        # Determine feeder quantity.
        if isinstance(gpxfl.batch_qty, numbers.Number):
            flqty = int(gpxfl.batch_qty)
        else:
            flqty = gpxsol(gpxfl.batch_qty).magnitude

        # For downstream quantities, here we assume none (or later compute them).
        ds_qty = []  # if no downstream, use product=1
        if ds_qty:
            flqty = flqty * int(np.prod(np.array(ds_qty)))
        # Compute batching time if flqty is not 1.
        if flqty != 1:
            # queue delay is (batch_qty / lam) * (flqty - 1)
            int_deptime = gpxsol(gpxfl.cuml_batch_qty / gpxfl.lam).to(target_unit).magnitude
            int_deptime = 1 / int_deptime if int_deptime != 0 else 0.0
            batch_time = int_deptime * (flqty - 1.0)
            fendqueues[flname] = batch_time
    return flprocflow, fflowtimes, fendqueues


def compute_local_offsets(
    gpxsol: any, process_flow: list, gpx_processes: dict, gpx_cells: dict, gpx_feeder_cells: dict,
    feedersfeedlocs: dict, flprocflow: dict, fflowtimes: dict, fendqueues: dict, target_unit: str, feeder_lines: list
) -> dict:
    """
    Computes a dict mapping each feeder line id to its local offset.
    For each feeder line, sums process times (from gpx_processes via process_flow or flprocflow)
    and queueing times (from visited cells in gpx_cells or gpx_feeder_cells) until reaching
    the target process (fl["to"]), then computes:
      offset = total lead time - flow time - end queue time.
    """
    localoffset = {}
    for fl in feeder_lines:
        flname = fl["id"]
        processtime = 0.0
        visitedcells = []
        if feedersfeedlocs[flname][0]:  # if the feeder joins the mainline
            for i, p in enumerate(process_flow):
                if p["id"] == fl["to"]:
                    break
                gpxprocess = gpx_processes[p["type"]]
                processtime += gpxsol(gpxprocess.t).to(target_unit).magnitude
                if i == 0:
                    visitedcells.append(gpx_cells[p["cell"]])
                elif p["cell"] != visitedcells[-1]:
                    visitedcells.append(gpx_cells[p["cell"]])
        else:
            # feeder feeds another cell
            target_line = feedersfeedlocs[flname][1]
            for i, p in enumerate(flprocflow.get(target_line, [])):
                if p["id"] == fl["to"]:
                    break
                gpxprocess = gpx_processes[p["type"]]
                processtime += gpxsol(gpxprocess.t).to(target_unit).magnitude
                if i == 0:
                    visitedcells.append(gpx_feeder_cells[p["cell"]])
                elif p["cell"] != visitedcells[-1]:
                    visitedcells.append(gpx_feeder_cells[p["cell"]])
        queuetime_list = [gpxsol(c.Wq).to(target_unit).magnitude for c in visitedcells]
        queuetime = float(np.sum(queuetime_list))
        total_lead_time = float(np.sum([queuetime, processtime]))
        offset = total_lead_time - fflowtimes.get(flname, 0) - fendqueues.get(flname, 0)
        localoffset[flname] = offset
    return localoffset


def compute_total_offsets(feederdownstreams: dict, localoffset: dict) -> dict:
    """
    Computes total offset for each feeder line as:
      total_offset = local_offset[fl] + sum(local_offset for all downstream feeder lines)
    """
    return {
        flname: np.sum([localoffset[fd]
                        for fd in feederdownstreams.get(flname, [])]) + localoffset[flname]
        for flname in feederdownstreams.keys()
    }


def build_feeder_line_results(
    feeder_lines: list, fflowtimes: dict, fendqueues: dict, totaloffset: dict, target_unit: str
) -> list:
    """
    Builds and returns a list of feeder line result dictionaries.
    
    Each result contains flowTime, startOffset, endQueueTime, timeUnit, and name.
    """
    results = []
    for fl in feeder_lines:
        flname = fl["id"]
        result = {
            "flowTime": np.round(fflowtimes.get(flname, 0.0), decimals=OTHER_ROUND_DEC),
            "startOffset": np.round(totaloffset.get(flname, 0), decimals=OTHER_ROUND_DEC),
            "endQueueTime": np.round(fendqueues.get(flname, 0.0), decimals=OTHER_ROUND_DEC),
            "timeUnit": target_unit,
            "name": flname,
        }
        results.append(result)
    return results


def build_mcclass_result_dict(by_split: bool, sol: dict, mc: MCell, name: str) -> dict:
    resdict = {}
    # add name and rate
    resdict.update({'name': name, 'hourlyRate': np.around(sol['variables'][mc.line.lam.key], decimals=OTHER_ROUND_DEC)})
    if by_split:
        # if defined by split include the sensitivity to the class split
        resdict['rateSensitivity'] = np.around(sol['sensitivities']['variables'][mc.X], decimals=OTHER_ROUND_DEC)
    else:
        # find the sensitivity of the input rate
        resdict['rateSensitivity'] = np.around(
            sol['sensitivities']['variables'][mc.line.lam.key], decimals=OTHER_ROUND_DEC
        )
    return resdict
