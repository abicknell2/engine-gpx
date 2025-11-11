import logging
from numbers import Number
import time
from typing import Union

import numpy as np

from api.module_types.module_type import ModuleType
from api.small_scripts import cdf_boxplot, check_magnitude
import api.uncertainty_generator as uncertainty_generator
import gpx
from gpx.uncertainty.distributions import (ThreePointVariable, UniformDistribtuion)
from gpx.uncertainty.uncertain_inputs import UncertainInput
from utils.types.data import Parameter


def cost_distribution(
    sol_likely: gpx.Model,
    sol_best: gpx.Model,
    sol_worst: gpx.Model,
    uvars: list[UncertainInput],
    points: int = 10000,
    *,
    results_dictionary: bool = False,
) -> (list[dict[str, np.float64]]
      | dict[str, list[dict[str, np.float64]] | dict[str, np.float64]]):
    "estimate the cost distribution"

    # record start time
    start_time: float = time.time()

    new_costs: list[float] = uncertainty_generator.estimate_cost_dist(
        sol_likely,  # generate cost distribution from the likely scenario
        uvars,
        num_samples=points,
        interpsens=True,
        multchange=True,
        sol_best=sol_best,
        sol_worst=sol_worst,
    )

    logging.info(f"cost distribution generation at {time.time() - start_time} [sec]")

    cost_cases: list[float] = [
        check_magnitude(sol_best["cost"]),
        check_magnitude(sol_worst["cost"]),
    ]

    # estimate a histogram
    y_hist, x_hist = uncertainty_generator.get_histogram(
        new_costs, minval=float(min(cost_cases)), maxval=float(max(cost_cases))
    )

    # get a relative y
    y_rel = np.divide(np.array(y_hist), np.sum(y_hist))
    y_rel = np.divide(y_rel, np.ptp(x_hist))

    # replace inf and nan
    y_rel = np.nan_to_num(y_rel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # align the x vector
    x_aligned = x_hist[:-1]

    cost_dist: list[dict[str, np.float64]] = [{
        "x": np.float64(x_val),
        "y": np.float64(y_val)
    } for x_val, y_val in zip(x_aligned, y_rel)]

    logging.info(f"PDF created at {time.time() - start_time} [sec]")

    # find the CDF only if there is a cost_dist
    cost_cdf: list[dict[str, np.float64]] = []
    if cost_dist:
        # cumulative sum the probs
        y_cuml = np.cumsum(y_rel)
        # find the scale to 1
        y_cuml = y_cuml / y_cuml[-1]
        # replace inf and nan
        y_cuml = np.nan_to_num(y_cuml, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        cost_cdf = [{
            "x": np.float64(x_val),
            "y": np.float64(y_val),
            "y_percent": np.float64(y_val * 100.0)
        } for x_val, y_val in zip(x_aligned, y_cuml)]

        # find the box plot
        box_pts: dict[str, np.float64] = cdf_boxplot(list(zip(x_aligned, y_cuml)))

        if results_dictionary:
            return {
                "costDist": cost_dist,
                "costPDF": cost_cdf,
                "costBoxplot": box_pts,
            }
    else:
        return {"costDist": cost_dist}

    print(f"cost distribution complete at {time.time()} [sec]")

    return cost_dist


def modvars_to_dists(module: "ModuleType") -> dict[str, Union[ThreePointVariable, UniformDistribtuion]]:
    """converts module variables to uncertain variables

    Arguements
    ---------
    module :
        the input module
    """

    vars_to_convert: dict[str, Parameter] = {}  # the variables to convert

    if hasattr(module, "variables"):
        # check to make sure the module has variables
        for varname, var in module.variables.items():
            if hasattr(var, "min") and hasattr(var, "max"):
                # check to see if this captures
                if isinstance(var.min, Number) and isinstance(var.max, Number):
                    vars_to_convert[varname] = var

    # convert to ditributions
    udists: dict[str, Union[ThreePointVariable, UniformDistribtuion]] = {}
    for varname, var in vars_to_convert.items():
        # create the uncertainty distributions
        if isinstance(var.value, Number):
            # if there is a likely value that is a number, use a three point dist
            # check to see the values are increasing
            if var.min <= var.value and var.value <= var.max:
                # create the distribution
                udists[varname] = ThreePointVariable([var.min, var.value, var.max])
            else:
                # raise an error that the inputs are not increasing
                raise ValueError(
                    '"{vname}" must have strictly increasing uncertain input parameters: [{varmin}, {varval}, {varmax}]'
                    .format(vname=varname, varmin=var.min, varval=var.value, varmax=var.max),
                )
        else:
            # check to make sure inputs are strictly increasing
            if var.min is not None and var.max is not None and var.min < var.max:
                # treat as a uniform distribution
                udists[varname] = UniformDistribtuion([var.min, var.max])
            else:
                # raise an error that the inputs are not increasing
                raise ValueError(
                    '"{vname}" must have strictly increasing uncertain input parameters: [{varmin}, {varmax}]'.format(
                        vname=varname,
                        varmin=var.min,
                        varmax=var.max,
                    ),
                )

    return udists
