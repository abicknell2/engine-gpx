"small helper scripts"

from itertools import groupby
from numbers import Number
from typing import Union, cast

from gpkit import units
import numpy as np
import scipy
from scipy.stats._distn_infrastructure import rv_frozen

from api.constants import OTHER_ROUND_DEC, SENS_ROUND_DEC
import gpx
from utils.settings import Settings
from utils.types.data import Parameter
from utils.types.shared import Probabilties


def update_disp_varname(varname: str) -> str:
    "updates the display Varname to format <product> | <variable name> [<module name>]"
    sp = varname.split(" :: ")
    prefix = ""
    if len(sp) > 1:
        prefix = f"{sp[0]} |"

    rem = sp[-1]
    sp = rem.split(" // ")
    # post = ""
    # if len(sp) > 1:
    #     post = "[%s]" % sp[-1] # Unused variable
    rem = sp[0]
    # dispname = '%s %s %s' %(prefix, rem, post)
    dispname = f"{prefix} {rem}"  # getting rid of the module name from the brackets
    dispname = dispname.strip()  # strip leading and trailing whitespace

    return dispname


def get_sols_units(sol: gpx.Model, var: "gpx.Variable") -> Number:
    "returns the solution with units"
    try:
        return sol[var] * units(var.units)
    except AttributeError:
        return sol[var]


def match_gamma(mean: float, stdev: float) -> rv_frozen:
    """create a moment-matched gamma function

    Arguments
    ---------
    mean : float
        mean of the distribution to match

    stdev : float
        standard deviation to match
    """
    cv2 = float(stdev**2) / float(mean**2)

    # note how this differs slightly from other definitions of the shape parameter
    shape = 1.0 / cv2
    scale = float(mean) / shape
    return scipy.stats.gamma(shape, scale=scale)


def get_gamma_shapescale(mean: float, stdev: float | None = None, cv2: float | None = None) -> tuple[float, float]:
    "find the shape (k) and scale (theta) of a gamma distribution that moment-matches the input statistics"

    # make sure only one is specified
    if stdev and cv2:
        raise ValueError("Specify only one of stdev or cv2")
    if stdev is None and cv2 is None:
        raise ValueError("Specify at least one of stdev or cv2")

    # calculate using std
    if stdev:
        # calculate the cv2 from stdev
        cv2 = float(stdev**2) / float(mean)

    shape = 1.0 / cv2 if cv2 is not None else 0.0
    scale = float(mean) / shape

    # return the parametmers
    return shape, scale


def welch_approx(dists: list[tuple[float, float]]) -> rv_frozen:
    "convert from list of tuples of [shape(k), scale(theta)] to a resulting distribution"

    distsmultsum = np.sum([d[0] * d[1] for d in dists])

    ksum_num = distsmultsum**2
    ksum_denom = np.sum([d[1]**2 * d[0] for d in dists])
    ksum = ksum_num / ksum_denom

    thetasum = distsmultsum / ksum

    return scipy.stats.gamma(ksum, scale=thetasum)


def match_gamma_cv2(mean: float, cv2: float) -> rv_frozen:
    """create a moment-matched gamma function

    Arguments
    ---------
    mean : float
        mean to moment-match
    cvs : float
        squared coefficient of variation

    """
    # note how this differs slightly from other definitions of the shape parameter
    shape = 1.0 / cv2
    scale = float(mean) / shape
    return scipy.stats.gamma(shape, scale=scale)


def gamma_box_points(dist: rv_frozen, nstd: int = 1, ci: float = 0.95, quartiles: bool = True) -> Probabilties:
    """Produce an array of points on a pdf

    Args
    ----
    dist : scipy.stats
        Distribution to plot

    points : int
        number of points to sample

    upper : int
        sets the upper limit from upper*stdev

    """
    if quartiles:
        box_points = {
            "low": dist.ppf(1 - ci),
            "q1": dist.ppf(0.25),
            "median": dist.ppf(0.5),
            "q3": dist.ppf(0.75),
            "high": dist.ppf(ci),
        }
    else:
        box_points = {
            "low": dist.ppf(1 - ci),
            "q1": (dist.mean() - float(nstd) * dist.std()),
            "median": dist.mean(),
            "q3": (dist.mean() + float(nstd) * dist.std()),
            "high": dist.ppf(ci),
        }
    box_points["boxPlot"] = [{}, {}]
    box_points["boxPlot"][0]["time"] = [box_points[key] for key in ["low", "q1", "median", "q3", "high"]]

    return box_points


def gamma_pdf_points(
    dist: rv_frozen,
    points: int = 200,
    upper: int = 3,
    xname: str = "time",
    yname: str = "probability",
    round: int | None = None
) -> list[dict[str, Union[np.float64, str]]]:
    """Produce an array of points on a pdf

    Args
    ----
    dist : scipy.stats
        Distribution to plot

    points : int
        number of points to sample

    upper : int
        sets the upper limit from upper*stdev

    yname : string
        the name for the y axis

    round : None or int (default=None)
        rounds the x values to the number of decimal places

    """
    x = np.linspace(0, dist.mean() + upper * dist.std(), points)
    prob = dist.pdf(x)

    if round is not None:
        # round the x vector
        x = np.around(x, decimals=round)

    return [{xname: val[0], yname: val[1]} for val in zip(x, prob)]


def cdf_boxplot(cdf: list[tuple[float, float]],
                nstd: int = 1,
                ci: float = 0.95,
                quartiles: bool = True) -> dict[str, np.float64]:
    "generates the points of a boxplot"  # AI generated

    # Extract the cumulative probabilities from cdf
    cum_probs = np.array([p[1] for p in cdf])

    # Calculate the quartiles and IQR based on the cumulative probabilities
    vals = np.array([p[0] for p in cdf])

    # Calculate the upper and lower bounds based on the cumulative probabilities
    # upper = np.interp(1.5, cum_probs, vals)  # Unused variable
    # lower = np.interp(-0.5, cum_probs, vals)  # Unused variable

    # create a function to wrap the interpolation
    def getpt(v: float) -> np.float64:
        return np.float64(np.interp(v, cum_probs, vals))

    box_points = {
        "low": getpt((1 - ci)),
        "q1": getpt(0.25),
        "median": getpt(0.5),
        "q3": getpt(0.75),
        "high": getpt(ci),
    }

    return box_points


def remove_repeated_entries(inputlist: list[int]) -> list[int]:
    """removes repeated entries from a list

    Example Usage
    -------------
    list = [1,1,2,3,4,4,4,1]
    remove_sequential_duplicates(list) ==> [1,2,3,4,1]
    """
    return [k for k, g in groupby(inputlist)]


def check_magnitude(obj: float) -> float:
    "checks to see if the object has a magnitude or just returns the object"
    if hasattr(obj, "magnitude"):
        return float(obj.magnitude)

    return float(obj)


def filter_unit_text(units: str) -> str:
    """filters the text of the units

    Arguments
    ---------
    units : string
        the units to filter
    """
    if "**" in units:
        # replace the exponent with a carrot
        units = units.replace("**", "^")

    return units


def make_all_vars(
    settings: Settings,
    collected_vars: dict[str, "gpx.Variable"],
    sol: gpx.Model,
    tags_dict: dict[str, Union[list[str], Parameter]] | None = None,
) -> tuple[list[dict[str, np.float64 | list | str]], dict[str, "gpx.Variable"]]:  # type: ignore
    "make the all vars"
    all_vars = []
    all_gpxvars = {}
    varnames = []

    sens_threshold = 1e-7  # threshold over which to calculate the cost driver
    margin_threshold = 0.01  # threshold for reporting the margin

    descriptive_vars = collect_descriptive_vars(collected_vars)

    for var in collected_vars.values():
        # list all variables in format
        # get the most descriptive name of the variable

        if isinstance(var, int) and var == 0:
            continue

        name = descriptive_vars[var.key]
        # TODO:  access variable with units using `m.solution(var)`
        if var in sol["variables"]:
            varname = update_disp_varname(str(name))
            if varname not in varnames:
                # prevent duplicates
                varnames.append(varname)

                # find if the variable is an input
                var_isinput = var in sol["constants"]

                # pull out values
                var_unit = filter_unit_text(
                    "".join(var.units.__str__().split()[1:]),
                )  # pull out the units of the variable
                var_val = sol["variables"][var.key]
                var_sens = sol["sensitivities"]["variables"].get(var.key, 0.0)

                varresultdict = {
                    "name": varname,
                    "value": np.around(float(var_val), decimals=OTHER_ROUND_DEC),
                    "unit": var_unit,
                    "source": "From Assumption" if var_isinput else "Calculated Value",
                    "tags": [],
                }

                # if there are tags
                if tags_dict:
                    vartags = tags_dict.get(varname)
                    if vartags:
                        # if there are tags update in the result
                        varresultdict.update({"tags": vartags})

                # add the sensitivity only if the variable is an input
                if var_isinput:
                    varresultdict["sensitivity"] = np.around(var_sens, decimals=SENS_ROUND_DEC)

                # find the unit cost drivers
                # if var_isinput:
                if np.abs(var_sens) > sens_threshold:
                    # only add the marginal costs if the variable is an input
                    costdriver = sol["cost"] * (((var_val + 1) / var_val)**var_sens - 1.0)

                    margin, pow10 = driver_ring_up(
                        threshold=cast(Number, margin_threshold),
                        sens=var_sens,
                        cost=sol["cost"],
                        val=var_val,
                    )

                    # check to see if the perturbation is reasonable
                    pow10_threshold = 5  # max at 100k
                    if cast(float, pow10) <= pow10_threshold:
                        # add the marginal costs to the results
                        costdriver = margin  # set the driver to the margin
                        unitmult = str(10**cast(float, pow10)) + " " if pow10 != 0 else ""  # find the unit multiplier

                        if len(var_unit) > 0:
                            # create units only if the variable has units
                            costdriver_units = f"{settings.default_currency_iso} / {unitmult}{var_unit}"  # units for the cost drivers
                        else:
                            if pow10 == 0:
                                unitmult = "1"
                            costdriver_units = f"{settings.default_currency_iso} / {unitmult}"

                        varresultdict.update({
                            "marginalCost": np.around(float(costdriver), decimals=OTHER_ROUND_DEC),
                            "marginalCostUnit": costdriver_units,
                        })

                all_vars.append(varresultdict)
                all_gpxvars[varname] = var

    return all_vars, all_gpxvars


def collect_descriptive_vars(collected_vars: dict[str, "gpx.Variable"]) -> dict[str, str]:
    "get only the most descriptive variables"
    descriptive_collect_vars = {}  # the varkey and the descriptive name
    for name, var in collected_vars.items():

        if isinstance(var, int) and var == 0:
            continue

        if var.key not in descriptive_collect_vars:
            # if it's not already in the dict, add it
            descriptive_collect_vars[var.key] = name
        else:
            # varkey already in the dict
            if len(name) > len(descriptive_collect_vars[var.key]):
                # if the new name is longer (more descriptive) replace it
                descriptive_collect_vars[var.key] = name

    return descriptive_collect_vars


def driver_ring_up(threshold: Number, sens: Number, cost: Number, val: Number) -> tuple[Number, Number]:
    """_summary_

    Parameters
    ----------
    threshold : Number
        _description_
    sens : Number
        _description_
    cost : Number
        _description_
    val : Number
        _description_

    Returns
    -------
    Tuple[Number, Number]
        [calculated marginal cost,
        power of 10 needed to meet the threshold]
    """
    "finds the minimum perturbation to exceed the measurement threshold"
    # get the abs of the sensitivity
    sens = np.abs(cast(float, sens))

    w = np.log((cast(float, threshold) / cast(float, cost) + 1)) / sens + np.log(cast(float, val))
    x = np.exp(w) - val
    pow10 = np.ceil((np.log(x) / np.log(10)))

    # make the margin at least 1
    if pow10 < 0:
        pow10 = 0

    # now calculate the margin with the new value
    margin = (((val + 10**pow10) / val)**sens - 1) * cost

    return margin, pow10
