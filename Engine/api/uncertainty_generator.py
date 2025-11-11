"generates uncertainty analysis"

import logging
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

import gpx
import gpx.uncertainty.distributions as gpxu_dist
import gpx.uncertainty.uncertain_inputs as gpxu
import utils.logger as logger
from utils.types.shared import AcceptedTypes, NumericArray

if TYPE_CHECKING:
    from api.interactive.model import InteractiveModel
# from constants import OTHER_ROUND_DEC, SENS_ROUND_DEC
# from small_scripts import check_magnitude

logging_preamble = "Copernicus-Engine uncertainty-generator|"

# def uncertainty_from_dict(interactive, inputdict):
#     '''create the uncertainty analysis

#     Arguments
#     ---------
#     interactive : interactive.InteractiveModel
#         the iteractive instance with the
#     inputdict : dict
#         model objects

#     Returns
#     -------
#         - all bootstrapped variables
#         - the risks for all of the uncertain inputs
#     '''
#     # map the location of the variables to the active modules

#     modelmap = [
#         ('design', inputdict['design']['selectedModule']) if 'design' in inputdict else ('design', None),
#         ('manufacturing', inputdict['manufacturing']['selectedModule'])
#     ]

#     modelmap = {name : location for location, name in modelmap if name is not None}

#     # get the distributions for inputs from each module
#     udists = {}
#     varlocs = {}
#     # udists, varlocs = dict_to_dists(inputdict['inputs'], modelmap)

#     for mod in list(interactive.active_modules.values()):
#         vars_to_update = list(filter(lambda x: x['name'] == mod.name, inputdict['modules']))
#         vars_to_update = vars_to_update[0]
#         vars_to_update = vars_to_update[vars_to_update['type']]['variables']  # use the `type` of the module to find the variables
#         add_udists, add_varlocs = dict_to_dists(vars_to_update, modelmap[mod.name])
#         udists.update(add_udists)
#         varlocs.update(add_varlocs)

#     #TODO: retrieve variables from the modules directly

#     # if there are not any distributions, just return the solution
#     if len(udists) == 0:
#         return interactive.solutions

#     # get the variables from the gpx model
#     uvars = gen_uvars(varlocs, udists, interactive)

#     # create uncertainty model
#     M = gpxu.UncertainModel(
#         interactive.gpx_model,
#         *uvars,
#         base_senss=interactive.gpx_model.solution['sensitivities']['variables'],
#         bootstrap=False
#     )

#     # set the base case for the interactive model

#     # get the list of variables
#     allvarkeys = list(interactive.collected_variables.values())
#     allvarkeysbyvar = {varkey : name for name, varkey in interactive.collected_variables.items()}

#     #DEBUG
#     try:
#         sol_best = M.get_best_case()
#     except RuntimeWarning:
#         print('Failed Best Case')
#         input('>>>')
#     try:
#         sol_likely = M.get_likely_case()
#     except RuntimeWarning:
#         print('Failed Likely Case')
#         input('>>>')
#     try:
#         sol_worst = M.get_worst_case()
#     except RuntimeWarning:
#         print('Failed Worst Case')
#         input('>>>')

#     try:
#         allvars = M.boostrap_variable(*allvarkeys, gethilo=True)
#     except RuntimeWarning:
#         print(M.gpmodel.program.solver_out)

#     # filter out entries where there is no change with tolerance
#     if M.is_boostrapped:
#         tolchange = 1e-5
#         abschangetol = 0.05
#         allvars = {var : res for var,res in allvars.items() if np.ptp(res)/np.mean(res) >= tolchange and np.ptp(res) >= abschangetol}

#     risks = M.risk_eval(sum_risk=True, include_points=True, measure='net')

#     # get solutions
#     sol_best = M.get_best_case()
#     sol_likely = M.get_likely_case()
#     sol_worst = M.get_worst_case()

#     # find combined risk and sensitivity data
#     cost = np.float64(check_magnitude(sol_likely['cost']))
#     risks_sens = []
#     for name, risk in risks.items():
#         risk_sum = np.float64(risk['sumRisk'])
#         var_key = interactive.collected_variables[name]
#         val_best = sol_best['variables'][var_key]
#         val_worst = sol_worst['variables'][var_key]
#         value = sol_likely['variables'][var_key]
#         sens = sol_likely['sensitivities']['variables'][var_key]
#         risk_profile = np.abs(risk_sum/cost/sens)

#         risks_sens.append({
#             'name' : name,
#             'sumRisk' : np.around(risk_sum, decimals=OTHER_ROUND_DEC),
#             'riskProfile' : np.around(np.float64(risk_profile), decimals=OTHER_ROUND_DEC),
#             'value' : np.around(np.float64(value), decimals=OTHER_ROUND_DEC),
#             # 'propRisk' : np.float64(risk_profile/value),
#             'propRisk' : np.around(np.float64(np.abs((val_best-val_worst)/value/2.0)), decimals=OTHER_ROUND_DEC),
#             'sensitivity' : np.around(sens, decimals=SENS_ROUND_DEC),
#             'absSensitivity' : np.around(np.abs(sens), decimals=SENS_ROUND_DEC),
#             'costedSens' : np.around(cost*sens, decimals=OTHER_ROUND_DEC),
#             'absCostedSens' : np.abs(np.around(cost*sens, decimals=OTHER_ROUND_DEC)),
#             'costedSensUnit': '[$ change / /%percent increase]'
#         })

#     # sort risk sensitivities alphabetically
#     risks_sens.sort(key=lambda x: x['name'])

#     # put all risks together
#     risks = [{'name' : name, **val} for name, val in risks.items()]

#     # sort risks by value
#     risks.sort(key = lambda x: x['sumRisk'], reverse=True)

#     # format range_vars
#     allvars = [{'name' : allvarkeysbyvar[var],
#                 # 'name' : str(var.key),
#                 'unit' : '$' if 'USD' in str(var.units) else ''.join(str(var.units).split()[1:]),
#                 'best' : np.around(float(sol_best['variables'][var]), decimals=OTHER_ROUND_DEC),
#                 'worst' : np.around(float(sol_worst['variables'][var]), decimals=OTHER_ROUND_DEC),
#                 'likely' : np.around(float(sol_likely['variables'][var]), decimals=OTHER_ROUND_DEC),
#                 'value' : np.round(np.array(vals),2).tolist() if M.is_boostrapped else [],}
# for var, vals in allvars.items() if 'Sensitivity' not in
# allvarkeysbyvar[var]]

#     # substitute the uncertainty likely case for the
#     interactive.gpx_solution = M.get_likely_case()
#     interactive.create_results()

#     # estimate cost distribution
#     num_samples=10000
#     # num_samples=5
#     new_costs = estimate_cost_dist(sol_likely, uvars, num_samples=num_samples, interpsens=True, multchange=True, sol_best=sol_best, sol_worst=sol_worst)

#     cost_cases = [check_magnitude(sol_best['cost']), check_magnitude(sol_worst['cost'])]

#     # estimate a histogram
#     # y,x = np.histogram(new_costs, bins=int(num_samples/10))
#     y,x = get_histogram(new_costs, minval=min(cost_cases), maxval=max(cost_cases))

#     # get relative y
#     # y_rel = float_(y)/np.sum(y)
#     y_rel = np.divide(np.array(y), np.sum(y))
#     y_rel = np.divide(y_rel, np.ptp(x))

#     x = x[:-1]

#     # cost_dist needs to be a list of dicts
#     # cost_dist = [{'x' : np.around(x, decimals=OTHER_ROUND_DEC),
#     #               'y' : np.around(y, decimals=OTHER_ROUND_DEC)}
#     #              for x,y in zip(x,y_rel)]

#     cost_dist = [{'x' : x,
#                   'y' : y}
#                  for x,y in zip(x,y_rel)]

#     # cost_dist = {
#     #     # 'points' : float_(list(zip(x,y))),
#     #     'x' : np.around(x, decimals=OTHER_ROUND_DEC).tolist(),
#     #     'y' : np.around(y, decimals=OTHER_ROUND_DEC).tolist(),
#     #     # 'y' : list(y),
#     # }

#     sens_error = []

#     for uv in uvars:
#         sens_bestworst = [sol_best['sensitivities']['variables'][uv.var.key],
#                           sol_worst['sensitivities']['variables'][uv.var.key]]

#         sens_error.append({
#             'name' : uv.name,
#             'sensitivity' : np.around(sol_likely['sensitivities']['variables'][uv.var.key], decimals=SENS_ROUND_DEC),
#             'upperSens' : np.around(max(sens_bestworst), decimals=SENS_ROUND_DEC),
#             'lowerSens' : np.around(min(sens_bestworst), decimals=SENS_ROUND_DEC)
#         })

#     # sort sens_error by absolute value of sensitivity
#     sens_error.sort(key=lambda x: np.abs(x['sensitivity']))

#     # update the results index
#     interactive.solutions['resultsIndex'].append(
#         {'name': 'Uncertain Variable Characteristics', 'value': 'riskSens'}
#     )

#     return {
#         'risks' : risks,
#         'rangeVars' : allvars,    # the uncertain variables
#         'riskSens' : risks_sens,  # combined risk and sensitivity data
#         'costDist' : cost_dist,
#         'sensU' : sens_error,
#         **interactive.solutions, # returning the base case.
#         # **likely_case,  #TODO: return the likely case instead
#     }


def dict_to_dists(
    inputdict: AcceptedTypes, module_location: str
) -> tuple[dict[str, gpxu_dist.ThreePointVariable | gpxu_dist.UniformDistribtuion], dict[str, str]]:
    """converts the min max to an uncertain variable

    Arguments
    ---------
    inputdict : dict
        the uncertainty represented as a variable
    module_location : string
        the name of the module where the gpx variable is located

    Returns
    -------
    gpx.uncertainty.UncertainVariable
    """

    # filter out variables which do not have min and max cases
    vars_to_process: list[dict[str, AcceptedTypes]
                          ] = [var for var in inputdict if isinstance(var, dict) and "min" in var and "max" in var]

    # for all the variables that are uncertain, create uncertain variables
    udists: dict[str, gpxu_dist.ThreePointVariable | gpxu_dist.UniformDistribtuion] = {}
    varlocs: dict[str, str] = {}

    for var in vars_to_process:
        if var["min"] is not None and var["max"] is not None:
            # check to make sure the points are actually defined
            logger.debug(f"{logging_preamble}adding uncertainty from: {var['name']}")
            if "value" in var and var["value"] != "" and var["value"] != "null" and var["value"] is not None:
                # three point distribution
                if isinstance(var["name"], str):
                    udists[var["name"]] = gpxu_dist.ThreePointVariable([var["min"], var["value"], var["max"]])
                else:
                    logging.warning(f"{logging_preamble}variable name is not a string: {var['name']}")

                # DEBUG: save old format
                # udists[var['name']] = gpxu_dist.ThreePointVariable(
                #     var['best'],
                #     var['worst'],
                #     mode=var['likely']
                # )

            else:
                if isinstance(var["name"], str):
                    udists[var["name"]] = gpxu_dist.UniformDistribtuion([var["min"], var["max"]])
                else:
                    logging.warning(f"{logging_preamble}variable name is not a string: {var['name']}")
                # treat as uniform

                # DEBUG: keep old format around
                # udists[var['name']] = gpxu_dist.UniformDistribtuion(
                #     var['best'],
                #     var['worst']
                # )

            # where is the variable located
            if isinstance(var["name"], str):
                varlocs[var["name"]] = module_location
            else:
                logging.warning(f"{logging_preamble}variable name is not a string: {var['name']}")

    return udists, varlocs


def gen_uvars(
    varlocs: dict[str, str], udists: dict[str, Union[gpxu_dist.ThreePointVariable, gpxu_dist.UniformDistribtuion]],
    imodel: "InteractiveModel"
) -> list[gpxu.UncertainInput]:
    """generate the uncertain variables

    Args:
        varlocs (dict): Mapping of variable names to module locations.
        udists (dict): Mapping of variable names to uncertain distributions.
        imodel (Any): The model object.

    Returns:
        list: A list of uncertain variables.
    """
    uvars: list[gpx.Variable] = []

    for name, dist in udists.items():
        if name in imodel.gpx_model.varkeys:
            logger.debug(f"{logging_preamble}adding uvar: {name}")
            loc: str = varlocs[name]
            gpxvar = imodel.active_modules[loc].gpx_variables[name]
            uvars.append(gpxu.UncertainInput(gpxvar, dist, name))
        else:
            # if the uncertain variable is not in the model, ignore it
            logging.warning(f"{logging_preamble}uvar not in model: {name}")

    return uvars


def is_uncertain(variable: gpx.Variable) -> bool:
    """determines if a variable is uncertain

    Args:
        variable (dict): The variable to check.

    Returns:
        bool: True if the variable is uncertain, otherwise False.
    """
    if "best" in variable and "worst" in variable:
        if variable["best"] is not None and variable["worst"] is not None:
            return True
    return False


def estimate_cost_dist(
    sol: gpx.Model,
    uvars: list[gpx.Variable],
    num_samples: int = 100,
    is_smoothed: bool = True,
    multchange: bool = True,
    interpsens: bool = False,
    **kwargs: AcceptedTypes,
) -> list[float]:
    """estimate the distribution of the costs

    Optional Arguments
    ------------------
    sol_best : solution
        best-case solution
    sol_worst : solution
        worst-case solution
    interpsens : bool
        interpolate the sensitivities
    multchange : bool
        use the multiplicative version of the change

    """

    # if there are no uncertain variables just send single cost
    if len(uvars) == 0:
        return 1000 * [sol["cost"]]

    # get a vector of the sensitivities
    sens = np.array([sol["sensitivities"]["variables"][uv.var.key] for uv in uvars])

    # get a vector of the likely values from the solution
    vals: NDArray[np.float64] = np.array([sol["variables"][uv.var.key] for uv in uvars])
    # tile the vals and transpose
    vals = np.tile(vals, (num_samples, 1)).transpose()

    # sample the variables
    samples_list: list[NDArray[np.float64]] = [uv.uncertainty.sample(size=num_samples) for uv in uvars]

    # form samples into a matrix
    # rows are different vars
    # cols are different samples
    samples: NDArray[np.float64] = np.array(samples_list)

    # interpolate sensitivities based on the sampled values of the variables
    if interpsens and kwargs["sol_best"] is not None and kwargs["sol_worst"] is not None:
        interp_sens_list = []
        sol_worst = kwargs["sol_worst"]
        sol_best = kwargs["sol_best"]
        sol_likely = sol

        sol_best_variables = sol_best.get("variables", {})
        sol_likely_variables = sol_likely.get("variables", {})
        sol_worst_variables = sol_worst.get("variables", {})

        sol_best_sensitivities = sol_best.get("sensitivities", {})
        sol_likely_sensitivities = sol_likely.get("sensitivities", {})
        sol_worst_sensitivities = sol_worst.get("sensitivities", {})

        sol_best_sensitivities_variables = sol_best_sensitivities.get("variables", {})
        sol_likely_sensitivities_variables = sol_likely_sensitivities.get("variables", {})
        sol_worst_sensitivities_variables = sol_worst_sensitivities.get("variables", {})

        for i in range(len(uvars)):
            var = uvars[i].var
            xp = [
                sol_best_variables.get(var, None),
                sol_likely_variables.get(var, None),
                sol_worst_variables.get(var, None),
            ]
            # may need to retrieve magnitude
            fp = [
                sol_best_sensitivities_variables.get(var, None),
                sol_likely_sensitivities_variables.get(var, None),
                sol_worst_sensitivities_variables.get(var, None),
            ]

            interp_sens_list.append([np.interp(s, xp, fp) for s in samples[i]])

        interp_sens: NDArray[np.float64] = np.array(interp_sens_list)

    # using multiplicative percents
    if multchange and len(samples) > 0:
        # find the u0 vector
        u0 = vals

        try:
            s: NDArray[np.float64] = interp_sens
        except BaseException:
            s = np.tile(sens, (num_samples, 1)).transpose()

        # the ustar matrix is the samples
        ustar = samples

        # ustar/u0
        try:
            change = np.divide(ustar, u0)
            change = np.power(change, s)
            change = np.prod(change, axis=0)

            # starting cost
            cstar = np.multiply(change, sol["cost"])
        except ValueError as e:
            logging.info(f"UNCERTAINTY-GENERATOR | {e}")
            raise ValueError("Cannot estimate cost distribution")

        return list(cstar)

    # find percent diff
    try:
        diff = np.divide(samples - vals, vals)
    except ValueError:
        raise ValueError("Could not estimate costs from uncertainties")

    if interpsens:
        perchange = np.multiply(diff, interp_sens)
        perchange = np.sum(perchange, axis=0)
    else:
        # calculate expected change percents
        perchange = np.matmul(diff.transpose(), sens)

    cost = sol["cost"]
    new_costs = cost * (1 + perchange)

    return list(new_costs)


def get_histogram(
    s: list[float],
    minval: float | None = None,
    maxval: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    "get a histogram"
    num_samps = len(s)
    win_length = 100

    if minval is not None and maxval is not None:
        # try and extend the histogram to the limits
        y, x = np.histogram(s, bins=int(num_samps / 30), range=(minval, maxval))
    else:
        y, x = np.histogram(s, bins=int(num_samps / 30))

    y_smooth = smooth(y, window_len=win_length + 1)
    y_smooth = y_smooth[int(win_length / 2):-int(win_length / 2)]
    # y_smooth[0] = 0
    # y_smooth[-1] = 0

    return y_smooth.astype(np.float64), x.astype(np.float64)


def smooth(x: NumericArray, window_len: int = 11, window: str = "hanning") -> NumericArray:
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    source:
        https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        # instead of failing just don't smooth
        logging.info("UNCERTAINTY GENERATOR :: SMOOTH | Input vector smaller than window size. Skipping smoothing")
        return x

    if window_len < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = getattr(np, window)(window_len)

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y
