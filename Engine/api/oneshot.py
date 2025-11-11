"""single call to create and solve model based on json
to be used with easy api
"""

import json
import logging
import os
import time
from typing import cast

from api.constants import RECORD_UPLOAD
from api.context.ramp import rampContext
from api.context.uncertain import Uncertain
import api.interactive.model as interactive
from api.interactive_generators import createModelFromDict
from api.multiproduct import (
    create_from_dict, create_from_variants_dict, multi_product_risk_results, update_variant_results,
    update_variants_solution
)
import gpx
from gpx import Model
from utils.constraints_helpers import update_module_constraints_bestworst
import utils.logger as logger
from utils.type_helpers import is_float
from utils.types.shared import AcceptedTypes


def run_one_shot(inputdict: dict[str, AcceptedTypes]) -> dict[str, Model]:
    """Runs a one-shot analysis

    Arguments
    ---------
    inputdict : string
        A json describing the model to be solved

    Returns
    -------
    string
        JSON formatted string to send to the front-end
    """
    print("oneshot")

    if not isinstance(inputdict, dict):
        raise ValueError("Input must be a dictionary")

    # create the processes
    interaction = createModelFromDict(inputdict=inputdict["model"])

    # check for a trade-study
    interaction.trade_study = "trade" in inputdict

    if interaction.trade_study:
        interaction.trade_params = cast(dict[str, AcceptedTypes], inputdict["trade"])

    # GPX Operations
    interaction.generate_gpx(interaction.settings)
    interaction.solve()
    interaction.create_results()

    return interaction.solutions


def new_best_worst(inputdict: dict[str, AcceptedTypes]) -> dict[str, Model]:
    # create the interactive from the input model
    interaction: interactive.InteractiveModel = createModelFromDict(inputdict=inputdict["model"])

    # check for a trade-study
    # if 'trade' in inputdict:
    #     raise ValueError('Does not perform solve trades with uncertain inputs')

    # check for a trade-study
    interaction.trade_study = "trade" in inputdict
    if interaction.trade_study:
        interaction.trade_params = cast(dict[str, AcceptedTypes], inputdict["trade"])

    # add the context for uncertainty
    interaction.context = Uncertain(interaction)
    interaction.is_uncertain = True

    # TODO:  set the target rate in the context

    # GPX Operations
    # DEBUG: only use the gpx gen in the uncertain product solve
    # interaction.generate_gpx(default_currency=interaction.settings.default_currency_iso)
    interaction.solve()
    interaction.create_results()

    # return the solutions
    return interaction.solutions

    # TODO:  if there is an `inf` in the results, should raise exception


def custom_variants(inputjson):
    'custom parts with variants'

    # convert variants to multi-products
    interaction = None

    # check for uncertain context
    if interaction.has_uncertainty():
        interaction.context = Uncertain(interaction)

    # solve the model
    interaction.generate_gpx()
    interaction.solve()
    interaction.create_results()
    return interaction.solutions


def multi_product(inputdict: dict[str, AcceptedTypes]) -> dict[str, Model]:
    """performs analysis for multi-product contexts"""
    # determine if native system model or variants
    from_variants = 'variants' in inputdict['model']['type']

    # create the interaction
    indict = inputdict['model']
    if from_variants:
        interaction = create_from_variants_dict(inputdict=indict)
    else:
        interaction = create_from_dict(inputdict=indict)

    # check for the uncertain context
    if interaction.has_uncertainty():
        # add uncertainty context if needed
        interaction.context = Uncertain(interaction)
        interaction.is_uncertain = True

    interaction.trade_study = "trade" in inputdict
    if interaction.trade_study:
        interaction.trade_params = cast(dict[str, AcceptedTypes], inputdict["trade"])

    finance = indict.get('finance')
    if finance is not None:
        interaction.discretized_resources = finance.get("discretizedResources", False)

    # if 'trade' in inputdict:
    #     raise ValueError('Solve Trade not available for System')
    # interaction.modules[0].type = 'multiproduct'

    # generating the model here is conditional on if it has uncertainty
    if not interaction.has_uncertainty():
        # only generate the gpx if there is not already uncertainty
        interaction.generate_gpx(interaction.settings)

    duration_value = inputdict['model'].get('duration', None)
    duration_unit = finance.get('durationUnit', None)

    if duration_value and duration_unit:
        duration = {'value': duration_value, 'unit': duration_unit}
    else:
        duration = None

    interaction.solve(duration=duration)

    # update the solution if it is from variants
    interaction.create_results()

    gpx_model = getattr(interaction, "gpx_model", None)
    solution = getattr(gpx_model, "solution", None) if gpx_model is not None else None
    if solution is not None:
        try:
            print("GPX solution table:")
            print(solution.table())
        except Exception:  # pragma: no cover - diagnostic aid only
            logging.exception("Failed to print GPX solution table")

    if interaction.trade_study:
        return interaction.solutions

    # update the product information for risks
    multi_product_risk_results(interaction)

    if from_variants:
        interaction.solutions = update_variant_results(interaction.solutions)


    return interaction.solutions


def ramp_up_analysis(inputdict: dict[str, AcceptedTypes]) -> dict[str, Model]:
    "TEMPORARY performs a ramp up analysis"
    logging.info("running rate ramp from one shot")

    # get the mfg inputs
    model = inputdict.get("model", {})

    if not isinstance(model, dict) or model is None:
        raise ValueError("Model must be a dictionary")

    mfg_model = model.get("manufacturing", {})

    # check if there is a complete rate-hike defined
    if "rateHike" in inputdict:
        max_rate = int(float(cast(int, inputdict["rateHike"].get("maxRate", 1))))
        resources: list[dict[str, str]] = cast(
            list[dict[str, str]], inputdict["rateHike"].get("resources", [])
        )  # give an empty list if there are no resources

        # get the starting point
        min_rate = inputdict["rateHike"].get("minRate", None)
        min_rate = float(min_rate) if is_float(min_rate) else None

        # check to make sure model is in "duration" amortization
        try:
            if isinstance(inputdict["model"], dict) and isinstance(
                    inputdict["model"].get("manufacturing"),
                    dict) and inputdict["model"]["manufacturing"].get("amortization") != "duration":
                # raise error. duration must be specified for a rate hike
                raise ValueError("Duration must be specified for a rate hike")
        except KeyError as e:
            logging.error(f"failed to check if model is in duration mode: {e}")
            raise ValueError("Error running rate hike analysis")
        except Exception as e:
            raise e

        # if there are resources in the rate hike, add them to the module before
        # generating gpx
        # dictupdate = {v["key"]: v for v in resources}  # Unused variable

        model = inputdict.get("model", {})
        if not isinstance(model, dict):
            raise ValueError("Model must be a dictionary")

        modules = model.get("modules", [])
        if not isinstance(modules, list):
            raise ValueError("Modules must be a list")

        for m in modules:
            if not isinstance(m, dict):
                raise ValueError("M must be a dict")

            if "manufacturing" in m:
                if isinstance(m["manufacturing"], dict) and "variables" in m["manufacturing"]:
                    mvars: list[dict[str, gpx.Variable]
                                ] = cast(list[dict[str, gpx.Variable]], m["manufacturing"]["variables"])
                else:
                    raise ValueError("Manufacturing variables must be a dictionary")

                if isinstance(m["manufacturing"], dict) and not m["manufacturing"].get("exposedVariables"):
                    # make sure there is at least an empty list here
                    m["manufacturing"]["exposedVariables"] = []

                expvars: list[dict[str, gpx.Variable]
                              ] = cast(list[dict[str, gpx.Variable]], m["manufacturing"]["exposedVariables"])
                props_to_match = ["category", "property", "type"]
                # go through the resources to see if any are already in exposed
                # variables
                if not isinstance(resources, list):
                    raise ValueError("Resources must be a list")

                for i, r in enumerate(resources):
                    # find if there is already a variable
                    if isinstance(expvars, list) and all(isinstance(v, dict) for v in expvars):
                        ext_var: list[dict[str, gpx.Variable]] = [
                            v for v in expvars
                            if all(v[p] == cast(dict[str, gpx.Variable], r)[p] for p in props_to_match)
                        ]
                    else:
                        ext_var = []

                    if ext_var:
                        # update the resource with the key of the existing var
                        evar = ext_var[0]
                        r["name"] = evar["name"]
                        if isinstance(r, dict):
                            r["key"] = evar.get("key", evar["name"])  # if no key, just use the name
                        resources[i] = r

                    if not ext_var:
                        # there is not an existing variable add directly to the model
                        expvars.append(r)
                        mvars.append(r)

    # make sure input model does not have a rate
    if mfg_model.get("rate", None) is not None:
        # clear out the rate if there is an input
        logger.debug("rate found in manufacturing model input. overriding")
        if isinstance(mfg_model, dict):
            mfg_model["rate"] = None

    # create the interaction
    interaction = createModelFromDict(inputdict=inputdict["model"])

    # if the model has unccertainty, make sure all variables get substitutions
    # for likely
    if interaction.has_uncertainty():
        # make sure all values have a likely substitution
        for mod in interaction.active_modules.values():
            if hasattr(mod, "variables"):
                # find variables missing the likely input (mean of min and max)
                update_module_constraints_bestworst(mod, mod.variables)
        #  create the uncertain context
        interaction.context = Uncertain(interaction)
        interaction.is_uncertain = True

    # create the context for the rate ramp
    interaction.context = rampContext(
        interaction,
        max_rate=max_rate,
        min_rate=float(min_rate) if isinstance(min_rate, (int, float, str)) else None,
        ramp_resources=resources,
    )

    # reflect if the uncertinaty should be evaluated
    interaction.context.eval_uncertainty = bool(inputdict["rateHike"].get("bestWorst", False))

    # generate and solve the model
    if not interaction.is_uncertain:
        interaction.generate_gpx(interaction.settings)  # only generate if there is not uncertainty

    # try to solve
    errs = []
    try:
        interaction.solve()
    except Exception as e:
        # find any errors that came up
        if hasattr(e, "message"):
            errs.append(e.message)
        else:
            errs.append(str(e))

        # delete the last, failed solve
        # failed_solution = interaction.context.ramp_sols.pop()

    # try and generate results in spite of errors
    interaction.create_results()

    # get the solution
    # similar to how we are doing the solve-trade

    res = interaction.solutions

    if not isinstance(res, dict):
        # convert the result to a dictionary
        res = {"plot": res}

    # append if there are any errors
    if errs:
        # get the last step
        lstp = interaction.context._discrete_steps[-1]
        lstp_str = f"---Last step before infeasbility: {lstp[0]}  {lstp[1]} -> {lstp[2]} ---"

        res["errors"] = [lstp_str, *errs]

    return res


def stream_ramp(inputdict: AcceptedTypes) -> None:
    "makes a rate ramp with streaming intermediate results"
    pass


def record_input(inputjson: AcceptedTypes, module: str) -> None:
    if RECORD_UPLOAD:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestr}_{module}.json"
        with open(os.path.join("logs", filename), "w") as writefile:
            json.dump(inputjson, writefile)
