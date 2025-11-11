from itertools import chain
import logging
from typing import TYPE_CHECKING, Callable, Optional, Union, cast

from gpkit.exceptions import Infeasible, UnknownInfeasible
import numpy as np

from api.constants import OTHER_ROUND_DEC, SENS_ROUND_DEC
from api.context.solution_context import SolutionContext
from api.small_scripts import check_magnitude, make_all_vars
import gpx
from gpx.uncertainty.distributions import (ThreePointVariable, UniformDistribtuion)
from gpx.uncertainty.uncertain_inputs import UncertainInput, UncertainModel
from utils.constraints_helpers import update_module_constraints_bestworst
from utils.settings import Settings
from utils.types.shared import AcceptedTypes, RiskSensitivity, RisksRow
from utils.uncertainty_helpers import cost_distribution, modvars_to_dists

if TYPE_CHECKING:
    from api.interactive.model import InteractiveModel


class Uncertain(SolutionContext):
    """a solve context for uncertain inputs

    Attributes
    ----------
    basis_context : SolutionContext
        the context for the certain case
    udists : dict
        uncertainty distributions
    uvars : list
        complete gpx uncertain variables
    baselinesol : dict
        the GPX solution to the baseline model
    usols : dict
        uncertain solutions
    M : Optional[UncertainModel]
        the uncertain model
    """

    def __init__(self, interactive: "InteractiveModel", **kwargs: AcceptedTypes) -> None:
        "initialize from existing context"
        self.M: Optional[UncertainModel] = None
        self.udists: dict[str, Union[ThreePointVariable, UniformDistribtuion]] = {}
        self.uvars: list[UncertainInput] = []
        self.usols: dict[str, gpx.Model] = {}

        # set the solution scenario
        self.sol_scenario: Optional[AcceptedTypes] = interactive.uncertainty_scenario

        # get the baseline context from the parent
        self.basis_context: SolutionContext = interactive.context

        # track success on different scenarios in solve
        self.all_scenario_success: bool = False  # all solves have been successful
        self.scenario_status: dict[str, bool] = {}
        self.scenario_status_message: dict[str, str] = {}
        self.settings: Settings = None

        # call the parent constructor
        super().__init__(interactive, **kwargs)

    def get_gpx_model(self, settings: Settings, **kwargs: AcceptedTypes) -> gpx.Model:
        "get the model of the basis context"
        self.settings = settings
        return self.basis_context.get_gpx_model(settings, **kwargs)

    def get_cost(self) -> float:
        "get the cost function from the certain baseline model"
        return self.basis_context.get_cost()

    def get_substitutions(self, for_assembly: bool = False, **kwargs: AcceptedTypes) -> dict[str, float]:
        "just return the basis context substitutions"
        self._substitutions_ = self.basis_context.get_substitutions(for_assembly)
        return self._substitutions_

    # def make_solutions(self, **kwargs):
    #     'make the solutions'
    #     super().make_solutions(**kwargs)
    #     #TODO: incorporate the solution returns from the basis context

    def solve(self, max_rate: float | None = None, **kwargs: AcceptedTypes) -> None:
        """solve the uncertain context

        Solves the baseline with necessary substitutions for the `likely` values.
        Then solves the uncertain cases
        """
        # check to see if there is discrete resources
        discrete_resources: bool = self.interaction.discretized_resources
        if discrete_resources:
            # temporarily disable resource discretization
            self.interaction.discretized_resources = False

        # initalize
        self._initial_solve(**kwargs)

        for case, func in self.solve_scenarios_dict.items():
            try:
                self.usols[case] = func(**kwargs)
                self.scenario_status[case] = True
            except Exception as err:
                if isinstance(err, (UnknownInfeasible, Infeasible, IndexError)):
                    self.scenario_status[case] = False
                    self.scenario_status_message[case] = "Model is infeasible. Check for conflicting constraints."
                else:
                    raise err

        # check if all scenarios were successful
        self.all_scenario_success = all(self.scenario_status.values())

        # if resources need discretization
        if discrete_resources:
            # re-enable the discretization and solve
            self.interaction.discretized_resources = True
            # get the baseline from the case
            sol: gpx.Model = self.usols[str(self.sol_scenario)]
            # put the substituions back
            self.interaction.gpx_model.substitutions.update(sol["constants"])
            # substitute solution
            self.interaction.gpx_model.solution = sol
            # solve the baseline

            # put the new discrete solution
            self.usols[str(self.sol_scenario)] = self.basis_context.solve(solve_orig=True)

    def _initial_solve(self, **kwargs: AcceptedTypes) -> None:
        "make the substituions for the likely context"
        # make the substitutions for the likely context
        for mod in list(self.interaction.active_modules.values()):
            if hasattr(mod, "variables"):
                update_module_constraints_bestworst(mod, mod.variables)

        # GPX Baseline Solve
        # generate using the interaction
        # this is not done in the call to `oneshot`
        self.interaction.generate_gpx(self.interaction.settings)

        # update substitutions from the basis context
        # also should update if there were missing `value` entries in any of the variables
        # this will create the "likely" scenario
        self.interaction.gpx_model.substitutions.update(self.basis_context.get_substitutions())

        # solve using the baseline context

        max_rate = kwargs.pop("max_rate", None)
        self.basis_context.solve(max_rate=max_rate if isinstance(max_rate, float) else None, **kwargs)
        # save the baseline solution
        self.baselinesol: gpx.Model = self.interaction.gpx_model.solution

        # SOLVE WITH UNCERTAIN CALCULATIONS
        # generate uncertain inputs
        self._make_uncertain(**kwargs)  # generate uncertain

        # solve the uncertain model scenarios
        self.solve_scenarios_dict: dict[str, Callable[..., gpx.Model]] = {  # type: ignore
            "best": self.M.get_best_case,  # type: ignore
            "likely": self.M.get_likely_case,  # type: ignore
            "worst": self.M.get_worst_case,  # type: ignore
        }

    def make_solutions(self, **kwargs: AcceptedTypes) -> gpx.Model:
        "make the solutions"
        result_warnings: list[str] = []  # hold any warning while creating the results

        # if the selected scenario is None, default to likely
        if self.sol_scenario is None:
            self.sol_scenario = "likely"

        # check to make sure the input is in the solutions
        if self.sol_scenario not in self.scenario_status:
            raise ValueError(f'Selected Scenario "{self.sol_scenario}" is not valid')

        # check to make sure the selected scenario is solved
        if not self.scenario_status[str(self.sol_scenario)]:
            raise ValueError(
                f"{self.sol_scenario.upper()} CASE scenario does not have a solution. Choose another scenario for Summarized Results.",
            )

        # change basis reference to current I
        self.interaction.gpx_solution = self.usols[str(self.sol_scenario)]
        self.basis_context.interaction = self.interaction

        # call the basis context to get all its solutions
        collect_variables: dict[str, gpx.Variable] = self.basis_context.make_solutions(**kwargs)

        # EVALUATE RISKS AND COST DISTRIBUTION
        # only evaluate if all scenarios solved successfully
        if self.all_scenario_success:

            # risks and uncertainties
            risks: dict[str, RisksRow] = self.M.risk_eval(
                sum_risk=True,
                include_points=False,
                measure="net",
            )
            # TODO:  this is getting the cost to base the risk from the selected
            # scenario. Is that correct?
            cost = np.float64(check_magnitude(self.usols[self.sol_scenario]["cost"]))
            # use the likely case as the basis for the risk assessment
            risk_sens: list[RiskSensitivity] = []

            for name, risk in risks.items():
                risk_sum = np.float64(risk["sumRisk"])
                name = name.replace("::", "|")
                # create a stripped version of the name for comp since the naming strips
                # the leading
                compname: str = name.strip()
                if compname not in self.interaction.collected_variables:
                    # if the variable cannot be found skip adding
                    # look if there is a vesion where the module name is not included
                    tryname: str = compname.split(" // ")[0].strip()
                    if tryname in self.interaction.collected_variables:
                        compname = tryname
                        logging.info(f"Uncertain Solution Gen | Variable {compname} found excluding module")
                    else:
                        logging.warning(f"Uncertain Solution Gen | Collected Variables Skipped for {name}")
                        # add to the warnings
                        result_warnings.append(f"Could not generate risk data for variable '{name}'")
                        continue

                var_key: str = self.interaction.collected_variables[compname]
                val_best: float = self.usols["best"]["variables"][var_key]
                val_worst: float = self.usols["worst"]["variables"][var_key]
                val_likely: float = self.usols["likely"]["variables"][var_key]
                sens: float = self.usols["likely"]["sensitivities"]["variables"][var_key]
                # risk_profile = np.abs(risk_sum/cost/sens)

                risk_sens.append({
                    "name":
                    name,
                    "sumRisk":
                    np.around(risk_sum, decimals=OTHER_ROUND_DEC),
                    # 'riskProfile' : np.around(np.float64(risk_profile), decimals=OTHER_ROUND_DEC),
                    "best":
                    np.float64(val_best),
                    "worst":
                    np.float64(val_worst),
                    "value":
                    np.around(np.float64(val_likely), decimals=OTHER_ROUND_DEC),
                    # 'propRisk' : np.float64(risk_profile/val_likely),
                    "propRisk":
                    np.around(
                        np.float64(np.abs((val_best - val_worst) / val_likely / 2.0)),
                        decimals=OTHER_ROUND_DEC,
                    ),
                    "sensitivity":
                    np.around(sens, decimals=SENS_ROUND_DEC),
                    "absSensitivity":
                    np.around(np.abs(sens), decimals=SENS_ROUND_DEC),
                    "costedSens":
                    np.around(cost * sens, decimals=OTHER_ROUND_DEC),
                    "absCostedSens":
                    np.abs(np.around(cost * sens, decimals=OTHER_ROUND_DEC)),
                    "costedSensUnit":
                    f"[{self.settings.default_currency_iso} change / /%percent increase]",
                })

            # sort risks alphabetically
            risk_sens.sort(key=lambda x: x["name"])

            # put all risks together
            risks_list: list[RisksRow] = [{"name": name, **val} for name, val in risks.items()]
            risks_list.sort(key=lambda x: x["sumRisk"], reverse=True)

            # add risks to results
            self.interaction.solutions["riskSens"] = risk_sens
            self.interaction.solutions["resultsIndex"].append({
                "name": "Uncertain Variable Characteristics",
                "value": "riskSens",
            })

            # COST DISTRIBUTIONS (delegated to helper)
            dist_dicts = cost_distribution(
                self.usols["likely"],
                self.usols["best"],
                self.usols["worst"],
                self.uvars,
                results_dictionary=True,
            )
            # self.interaction.solutions['costDist'] = self._cost_dist()
            self.interaction.solutions.update(dist_dicts)
            self.interaction.solutions["resultsIndex"].extend([
                {
                    "name": "Cost Probability Distribution",
                    "value": "costDist",
                },
                {
                    "name": "Cost Cumulative Distribution",
                    "value": "costPDF",
                },
                {
                    "name": "Cost Probability Box Plot",
                    "value": "costBoxplot",
                },
            ])

            # make the box plots into an array
            box_plot_names: list[str] = ["low", "q1", "median", "q3", "high"]

            # update the 'probabilities' object
            # boxplot

            boxplot: list[dict[str, list[np.float64]]] = [{"x": [np.float64(0)] * 5}]
            if "costBoxplot" in dist_dicts:
                cost_boxplot: dict[str, np.float64] = cast(dict[str, np.float64], dist_dicts.get("costBoxplot", {}))
                boxplot = [{"x": [cost_boxplot[k] for k in box_plot_names]}]
                # only return if there is boxplot data
                self.interaction.solutions.setdefault("probabilities", {})
                self.interaction.solutions["probabilities"]["costDist"] = {
                    "pdfPoints": dist_dicts["costDist"],
                    "boxPlot": boxplot,
                    # 'unit'      : '$',    #TODO:  pull from the module unit property
                }

            # ADD OTHER SCENARIOS AS COLUMNS TO THE ALL VARIABLES
            # take the collect variables
            setofscenarios: set[str] = set(["best", "likely", "worst"])
            setofscenarios.remove(str(self.sol_scenario))

            # TODO:  use list comprehension for this
            # other_scenarios = {}
            # for scen in setofscenarios:
            #     other_scenarios[scen] = {
            #         v['name'] : v['values']
            #         for v in make_all_vars(collect_variables, self.usols[scen])[0]
            #     }
            other_scenarios: dict[str, dict[str, np.float64]] = {
                scen: {
                    str(v["name"]): cast(np.float64, v["value"]) for v in
                    make_all_vars(collected_vars=collect_variables, sol=self.usols[scen], settings=self.settings)[0]
                } for scen in setofscenarios
            }

            # TODO:  need to get the aux variables as well
            # get the manufacturing module rg and get the aux vars
            for sc in setofscenarios:
                aux_vars: list[gpx.Variable] = []
                # get all the results generators
                resgens = [
                    mod.get_results(sol=self.usols[sc], settings=self.interaction.settings)
                    for mod in self.interaction.active_modules.values()
                    if hasattr(mod, "get_results")
                ]
                for rg in chain.from_iterable(resgens):  # loop over the flattened result generators
                    aux_vars.extend(rg.aux_vars)
                # reformat aux_vars
                aux_vars_dict: dict[str, gpx.Variable] = {v["name"]: v["value"] for v in aux_vars}
                # merge the aux_vars into the other scenarios
                other_scenarios[sc].update(aux_vars_dict)

            # merge into the columns of the result
            for var in self.interaction.solutions["allVariables"]:
                var.update({
                    scenname: scenresult.get(var["name"], "") for scenname, scenresult in other_scenarios.items()
                })

        # return scenario failure information
        if len(self.scenario_status_message) > 0 or result_warnings:
            errs: list[str] = []

            if len(self.scenario_status_message) > 0:
                # add scenario warnings
                errs.extend([
                    "{} Case {}".format(case.title(), err) for case, err in self.scenario_status_message.items()
                ])

            if result_warnings:
                # add skipped result warnings
                errs.extend(result_warnings)

            self.interaction.solutions["errors"] = errs

        return collect_variables

    def _make_uncertain(self, **kwargs: AcceptedTypes) -> None:
        "makes the uncertain model"
        # TODO: look at `uncertainty_from_dict` for inspiration

        # get the distibutions for the uncertain variables
        for mod in self.interaction.active_modules.values():
            # get the uncertain distributions from the variables in each module
            self.udists.update(modvars_to_dists(mod))

        # generate the uncertain variables
        self._make_uvars()

        # filter out any variables that are not actually in the solution
        self.uvars = [uv for uv in self.uvars if uv.var in self.baselinesol["constants"]]

        # generate the uncertain model
        self.M = UncertainModel(
            self.interaction.gpx_model,
            *self.uvars,
            base_senss=self.baselinesol["sensitivities"]["variables"],
            bootstrap=False,
        )

    # if there are no uncertainties, just return the baseline model
    def _make_uvars(self) -> None:
        "creates the uncertain variables"
        gpxvars: dict[str, gpx.Variable] = {}
        for m in self.interaction.active_modules.values():
            if hasattr(m, "gpx_variables"):
                gpxvars.update(m.gpx_variables)
        for varname, dist in self.udists.items():
            # create the gpx uncertainty variable for each uncertain distribution
            if varname in gpxvars:
                # if the variable is actually in the model
                self.uvars.append(UncertainInput(gpxvars[varname], dist, varname))
