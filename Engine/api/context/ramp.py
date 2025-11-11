"rate ramp analysis"

from collections import namedtuple
import logging
from typing import TYPE_CHECKING, Optional, Union, cast

import gpkit
from gpkit import SolutionArray, Variable
import numpy as np
import pandas as pd

from api.constants import OTHER_ROUND_DEC
from api.context import MinUnitCost, SolutionContext
from api.context.uncertain import Uncertain
from api.module_types.manufacturing import Manufacturing
from api.module_types.production_finance import ProductionFinance
import gpx
import utils.logger as logger
from utils.ramp_helpers import (
    build_uikey_gpx_map, round_value, select_highest_sensitive_var, uncertain_collapse_resources
)
from utils.settings import Settings
from utils.types.shared import (AcceptedTypes, ResourceDict, ResourceStep, ResourceStepList, ResourceSteps)

if TYPE_CHECKING:
    from api.interactive.model import InteractiveModel

OFFSET_STEP = 1.0001

logging.getLogger().setLevel(logging.DEBUG)

colNameType = namedtuple("colNameType", "case coltype colname")


class rampContext(SolutionContext):
    "rate ramp context"
    "inspired by the uncertainty context"

    def __init__(
        self,
        imodel: "InteractiveModel",
        max_rate: int = 1,
        ramp_resources: list[dict[str, str]] = [],
        min_rate: Optional[float] = None,
        preserve_basis_context: bool = True,
        **kwargs: AcceptedTypes,
    ) -> None:
        "initialize from existing context"
        # set up variables
        # solutions
        self.ramp_sols: SolutionArray = []

        # capacity variables
        self.capvars: list[gpx.Variable] = []
        # resources to ramp over
        self.ramp_resources: list[dict[str, str]] = ramp_resources
        # create a dict for all the UI-key, gpxvarkey paris
        self._uikey_gpxkey_map: dict[str, gpx.Variable] = {}
        # the solved resrouces at different steps
        self._resource_steps: ResourceSteps = {}
        self._discrete_steps: list[tuple[gpx.Variable, float, float]] = []  # the discrete steps in resources

        if imodel.is_uncertain and not isinstance(imodel.context, Uncertain):
            imodel.context = Uncertain(imodel)

        # store the basis context
        self.basis_context: Union[SolutionContext, Uncertain, MinUnitCost] = imodel.context

        # get the max rate
        self.max_rate: float = float(max_rate)
        # get the min rate
        self.min_rate: Optional[float] = min_rate
        # the max rates of each ramp
        self.ramp_rates: Union[list[float], dict[str, list[float]]] = []

        # add the context to the interaction
        imodel.context = cast(SolutionContext, self)

        # validate the context with the model
        self._validate_context()

        # number of result points
        self.result_points: int = 100

        # create a named tuple for the column types
        self.colNameType = colNameType

        # should the uncertainty be evaluated
        self.eval_uncertainty = False

        self._uncertain_rate_at_worst = None

        # call the parent constructor
        super().__init__(imodel, **kwargs)

    def ramp_initialize(self, rounding_func: str = "round") -> None:
        "set the ramp variables to 1 and get the initial solution"
        # get the resource keys from all the ramp resources
        resc_keys: list[str] = [r["key"] for r in self.ramp_resources]
        self._uikey_gpxkey_map = build_uikey_gpx_map(self.interaction.active_modules, resc_keys)
        # create the capacity vars from this new dict
        self.capvars = list(self._uikey_gpxkey_map.values())

        if self.min_rate:
            # find the starting point by converting to hourly rate
            minhrrate = self._finance_module.get_hourly_rate(self.min_rate)
            # get the production rate
            # get the gpx model
            m = self.interaction.gpx_model
            # substitue the starting point
            m.substitutions[self._ratevar] = minhrrate
            # delete substitutions for resources
            for vk in self.capvars:
                if vk in m.substitutions:
                    del m.substitutions[vk]
            # solve at the starting point
            s = m.solve()
            # substitute rounding
            # make sure resource count not less than 1
            m.substitutions.update({vk: max(round_value(s(vk), rounding_func), 1.0) for vk in self.capvars})
            # delete rate constraint
            del m.substitutions[self._ratevar]
        else:
            # put substitutions for all the ramp variables
            self.interaction.gpx_model.substitutions.update({vk: 1 for vk in self.capvars})
            # get the list of capacity variables from the modules

        if False:
            # if the list is empty generate
            for mod in self.interaction.active_modules.values():
                if hasattr(mod, "ramp_get_resource_int_vars"):
                    self.capvars.extend(mod.ramp_get_resource_int_vars())

    @property
    def _finance_module(self) -> ProductionFinance:
        "returns the active finance module"
        return cast(ProductionFinance, self.interaction.active_modules["finance"])

    @property
    def _mfg_module(self) -> Manufacturing:
        "return the active manufacturing module"
        return cast(Manufacturing, self.interaction.active_modules["manufacturing"])

    @property
    def _ratevar(self) -> dict[str, gpx.Variable]:
        "gets the rate variable"
        mfgm = self._mfg_module
        return mfgm.gpxObject["fabLine"].lam

    def solve(self, max_rate: float | None = None, **kwargs: AcceptedTypes) -> None:
        "solve the rate ramp"

        # TODO:  check to see if there is a starting point

        if not max_rate:
            max_rate = self.max_rate

        # TODO:  convert the input max rate into hourly
        # self.interaction.active_modules['finance'].get_hourly_rate(max_rate)

        if self.eval_uncertainty:
            logger.debug("solving uncertainty loops")
            # perform the rate hike under uncertainty
            # do all three scenarios sequentially
            self.ramp_sols = {}
            self.ramp_rates = {}

            # disable resource discretization
            self.interaction.discretized_resources = False

            if not isinstance(self.basis_context, Uncertain):
                raise RuntimeError(
                    "eval_uncertainty=True but basis_context is "
                    f"{type(self.basis_context).__name__}; expected Uncertain"
                )

            # solve the initial model and convert to uncertain
            self.basis_context._initial_solve()

            # ordered so worst is solved first. This helps with interp
            cases = ["worst", "likely", "best"]
            sols_at_worst: dict[str, gpx.Model] = {}
            self._uncertain_sols_at_worst = sols_at_worst

            for case in cases:
                logger.debug(f"starting {case} case")
                # make a container for the solutions specific to the case
                func = self.basis_context.solve_scenarios_dict[case]
                sols = self.ramp_sols[case] = []
                rates = self.ramp_rates[case] = []

                # initialize the model for the ramp (round-down)
                self.ramp_initialize()

                # make the substituions for the case
                func()  # TODO: make sure this is getting the correct gpx model as a result

                # initialize the model for the ramp (round-down)
                self.ramp_initialize()

                # get the current rate
                cur_rate = self._solve_current(sols=sols)
                rates.append(cur_rate)

                if case == "worst":
                    # save the value
                    s = sols[-1]
                    sols_at_worst[case] = s
                    self._uncertain_rate_at_worst = s(self._get_ratevarkey())
                    # rate_at_worst = _uncertain_rate_at_worst # Unused var
                else:
                    "solve the other cases at worst"
                    logger.debug(f"match {case} to worst value")
                    sols_at_worst[case] = sols_at_worst["worst"]

                    # try:
                    #     sols_at_worst[case] = self._uncertain_base_solve_value(rate_at_worst)
                    # except Exception as err:
                    #     # could not solve the 1-all case at the worst point
                    #     logging.warn(f'solving at worst-case point failed for {case} case with error: {err}')
                    #     # just use the worst point as the substitute
                    #     sols_at_worst[case] = sols_at_worst['worst']

                # loop up to rate
                while cur_rate < max_rate:
                    # step the model and solve
                    self._step_ramp(sols[-1])
                    # solve at point
                    cur_rate = self._solve_current(sols=sols, rates=rates)
                    print(cur_rate)

                # add the resources steps
                self._resource_steps[case] = [self._resources_at_point(s) for s in sols]

        # solve just the likely case
        else:
            # if the underlying problem actually is uncertain, get the likely case
            if self.interaction.is_uncertain:
                if not isinstance(self.basis_context, Uncertain):
                    raise RuntimeError(
                        "eval_uncertainty=True but basis_context is "
                        f"{type(self.basis_context).__name__}; expected Uncertain"
                    )
                # temporarily disable the discrete solve to initialize
                self.interaction.discretized_resources = False
                self.basis_context._initial_solve()
                self.ramp_initialize()
                self.basis_context.M.get_likely_case()

            # initialize the problem
            self.ramp_initialize()

            # the current solved rate
            cur_rate = self._solve_current()
            self.ramp_rates.append(cur_rate)

            # iterate on the ramp while the current rate is less than the
            try:
                while cur_rate < max_rate:
                    # ramp up the inputs
                    self._step_ramp(self.ramp_sols[-1])  # send the most recent solution

                    # update the current rate
                    cur_rate = self._solve_current()
                    self.ramp_rates.append(cur_rate)
                    print(cur_rate)
            except Exception as e:
                # still make the resource steps
                self._resource_steps['likely'] = [self._resources_at_point(s) for s in self.ramp_sols]

                # raise the error
                raise e

            # # after solving process all the solutions into resrouces steps
            self._resource_steps['likely'] = [self._resources_at_point(s) for s in self.ramp_sols]

        print(f"Final rate reached {cur_rate}")
        print(f"Used {len(self.ramp_rates)}")

    def _uncertain_base_solve_value(self, rate: float) -> gpx.Model:
        "solve the 1-all case at a specific rate"
        m: gpx.Model = self.interaction.gpx_model
        # update the rate substitution to the input rate
        m.substitutions[self._get_ratevarkey()] = rate

        # solve at the point
        sol = m.solve()

        # delete the rate subsitution
        del m.substitutions[self._get_ratevarkey()]

        return sol

    def _solve_current(
        self,
        sols: Optional[list[gpx.Model]] = None,
        rates: Optional[list[float]] = None,
        **kwargs: AcceptedTypes
    ) -> float:
        "solve for the current step in the ramp"

        # solve at the current point
        self.interaction.gpx_model.solve()

        # get the current solution
        sol = self.interaction.gpx_model.solution
        # append the new solution to context
        # want to be explicit with the None since is often initialized to an empty list
        if sols is not None:
            sols.append(sol)
        else:
            self.ramp_sols.append(sol)

        # get the rate from the model
        ratevar = self._get_ratevarkey()
        cur_rate = sol["variables"][ratevar]

        # convert to aggregate rate
        finm = self._get_finmod()
        cur_rate, _, rate_unit = finm.get_aggregate_rate(cur_rate, 1)

        # if a rates list is provided, append
        if rates is not None:
            rates.append(cur_rate)

        # return the rate
        return cur_rate

    def _get_ratevarkey(self) -> gpx.Variable:
        "get the varkey for the production rate"
        if "manufacturing" in self.interaction.active_modules:
            mfgm = self.interaction.active_modules["manufacturing"]
            ratevar = mfgm.gpxObject["fabLine"].lam  # get the rate varkey from
            return ratevar
        else:
            raise ValueError("The production rate could not be found in the selected manufacturing module")

    def _get_finmod(self) -> ProductionFinance | Manufacturing:
        "gets the finance module"
        try:
            return self.interaction.active_modules["finance"]
        except KeyError:
            raise ValueError("Could not locate a finance module")

    def _step_ramp(self, sol: gpx.Model, step_size: int = 1) -> None:
        "step up the ramp by finding the capacity variable"
        # get the sensitivites from the solution
        solsens = sol["sensitivities"]["variables"]
        stepvar = select_highest_sensitive_var(self.capvars, solsens)
        # save to the list of steps
        old_val = self.interaction.gpx_model.substitutions[stepvar]
        self._discrete_steps.append((stepvar, old_val, old_val + step_size))
        self.interaction.gpx_model.substitutions[stepvar] = old_val + step_size

    def _validate_context(self) -> None:
        "validates the context for the current interaction"
        pass

    def get_rate_resources(self) -> None:
        "get the resources for the rate hike"
        pass

    def make_solutions(self, **kwargs: AcceptedTypes) -> dict[str, Variable]:
        "make the solutions from the ramp"
        # at a givine rate, there are two points:
        #      - the upper-throughput-limit of the lower cost solution
        #       - the lower-throughput-limit of the higher cost solution
        # list of the solution objecet
        sols: list[gpx.Model] = []
        if self.eval_uncertainty:
            dfs: dict[str, pd.DataFrame] = {}  # dictionary of dataframes for each of the individual solutions
            res_cols: list[str] = []  # resource columns
            cost_cols: list[str] = []  # cost columns

            # TODO:  maked named tuples for the columns

            # coltypes: list[str] = ["Total Unit Cost", "resources", "total variable cost", "total fixed costs"]  # Unused variable coltypes

            cases: list[str] = ["worst", "likely", "best"]

            # make all the columns based on case and column type
            # cols: list[rampContext.colNameType] = [
            #     self.colNameType(case=case, coltype=coltype, colname=f"{case} {coltype}")
            #     for case in cases
            #     for coltype in coltypes
            # ]  # Unused variable cols

            finm: ProductionFinance = cast(ProductionFinance, self._get_finmod())

            # Incompatible types in assignment (expression has type "tuple[float, float, str]", variable has type "float")

            rate_at_worst: tuple[float, float, str] = finm.get_aggregate_rate(self._uncertain_rate_at_worst, 1)
            rate_quantity = rate_at_worst[0] * gpkit.ureg(rate_at_worst[2])
            rate_at_worst = rate_quantity.magnitude
            # get the magnitude of the worst rate rate
            # unit_count_at_worst: int = finm.get_total_units  # Unused variable unit_count_at_worst

            # loop over the cases
            for case, ratesol in self.ramp_rates.items():
                # loop over rates
                colname: str = f"{case} Total Unit Cost"
                rescol: str = f"{case} resources"
                sol: gpkit.SolutionArray = self._get_solution_points(
                    sols=self.ramp_sols[case],
                    rates=ratesol,
                    resource_steps=self._resource_steps[case],
                    cost_name=colname,
                    resource_name=rescol,
                    case=case,
                )
                # append to the cost columns names
                cost_cols.append(colname)
                res_cols.append(rescol)

                # if not the worst case, add the solution at the worst point
                if case == "worst":
                    # set the rate to the worst rate
                    # sol[0]['rate'] = rate_at_worst
                    rate_at_worst = sol[0]["rate"]

                # make the dataframe
                df: pd.DataFrame = pd.DataFrame.from_records(sol, index="rate")
                dfs[case] = df

            # join the dataframes for the different cases
            alldfs: pd.DataFrame = pd.concat(dfs.values())
            # sort by index
            alldfs.sort_index(inplace=True)

            # ## Cut-off lower than starting point
            # # find the first point that excceds the starting rate
            # first_point = alldfs.reset_index()[alldfs.index > self.min_rate].index[0]
            # # move one row up
            # first_point -= 1
            # first_point = max(first_point, 0)
            # # down-size the dataframe
            # alldfs = alldfs.iloc[first_point:]

            # compute the average duration constant based on rate and unit counts
            avg_dur: float = (alldfs["total unit count"] / alldfs.index).mean()

            # sample the rate space to capture the curves
            # newdf = np.linspace(min(alldfs.index), max(alldfs.index), self.result_points)
            # TODO:  create the new df based on the min and max
            min_rate = self.min_rate if self.min_rate is not None else 0.0
            max_rate = self.max_rate if self.max_rate is not None else 1.0
            newdf: np.NDArray[np.float64] = np.linspace(min_rate, max_rate, self.result_points, dtype=np.float64)
            # newdf_df = pd.DataFrame(index=newdf)  # Unused variable newdf_df
            # rebuild the dataframe
            newdf_df = pd.DataFrame(index=newdf)
            alldfs = alldfs.combine_first(newdf_df)
            # fill empty unit counts
            missing_counts: pd.Series[bool] = alldfs["total unit count"].isna()
            alldfs.loc[missing_counts, "total unit count"] = alldfs.loc[missing_counts].index.to_numpy() * avg_dur

            # add in records for the best case and likely case at the worst point
            # addldf = self._get_results_at_worst(rate_at_worst, colnames=cost_cols, resnames=res_cols)  # TODO: Unused function?
            # alldfs = alldfs.combine_first(addldf)

            for case in cases:
                unitcostcol: str = f"{case} Total Unit Cost"
                capital_cost_col: str = f"{case} Total Capital Cost"
                fixedcostcol: str = f"{case} total fixed costs"
                varcostcol: str = f"{case} total variable costs"
                # forward-fill costs down
                alldfs[[fixedcostcol, varcostcol,
                        capital_cost_col]] = alldfs[[fixedcostcol, varcostcol, capital_cost_col]].ffill()
                # back-fill costs
                alldfs[[fixedcostcol, varcostcol,
                        capital_cost_col]] = alldfs[[fixedcostcol, varcostcol, capital_cost_col]].bfill()

                # find missing unit costs rows
                missing_costs: pd.Series[bool] = alldfs[unitcostcol].isna()
                missinglocs: pd.DataFrame = alldfs.loc[missing_costs]

                # calculate the missing cost and update
                alldfs.loc[missing_costs, unitcostcol] = (
                    missinglocs[varcostcol] + missinglocs[fixedcostcol] / missinglocs["total unit count"]
                )

            # fill the resource column down
            alldfs[res_cols] = alldfs[res_cols].ffill()

            # back-fill up any resource columns
            alldfs[res_cols] = alldfs[res_cols].bfill()

            # truncate the dataframe to min and max
            alldfs = alldfs.loc[alldfs.index >= self.min_rate]
            alldfs = alldfs.loc[alldfs.index <= self.max_rate]
            # TODO:  combine into a single statement

            # make total unit cost column
            alldfs["Total Unit Cost"] = alldfs["likely Total Unit Cost"]
            alldfs["Total Capital Cost"] = alldfs["likely Total Capital Cost"]

            # collapse resources into one column
            alldfs["resources"] = alldfs.apply(
                lambda x: self._uncertain_collapse_resources(
                    best_recs=x["best resources"],
                    likely_recs=x["likely resources"],
                    worst_recs=x["worst resources"],
                ),
                axis=1,
            )
            alldfs["range"] = alldfs.apply(lambda x: [x["best Total Unit Cost"], x["worst Total Unit Cost"]], axis=1)
            alldfs["rate"] = alldfs.index
            sols_dict = alldfs[["rate", "range", "resources", "Total Unit Cost",
                                "Total Capital Cost"]].to_dict(orient="records")
        # if there is no uncertainty
        else:
            # get the solution points
            sols = self._get_solution_points(
                sols=self.ramp_sols,
                # Argument "rates" to "_get_solution_points" of "rampContext" has incompatible type "list[float] | dict[str, list[float]]"; expected "list[float]"
                rates=cast(list[float], self.ramp_rates),
                resource_steps=self._resource_steps['likely'],
            )
            # convert to a dataframe
            df = pd.DataFrame.from_records(sols, index="rate")

            # compute the average duration
            unitcountcol: str = "total unit count"
            avg_dur = (df[unitcountcol] / df.index).mean()

            DEFAULT_MIN_RATE_FRACTION = 0.1
            if self.min_rate is None:
                base_sol = self.ramp_sols[0]

                rate_var_key = self._get_ratevarkey()
                base_vars = base_sol.get("variables", {})
                if rate_var_key not in base_vars:
                    raise ValueError(f"Rate variable {rate_var_key} not found in base solution variables")

                base_rate = base_vars[rate_var_key]
                rate_at_all_one, _, _ = self._get_finmod().get_aggregate_rate(base_rate, 1)

                self.min_rate = DEFAULT_MIN_RATE_FRACTION * rate_at_all_one

            # TODO:  check the input rate
            # TODO:  truncate the dataframe

            # re-sample the rate
            if self.result_points == 1 or self.min_rate == self.max_rate:
                newdf = np.array([self.max_rate])  # degenerate ramp
            else:
                newdf = np.linspace(self.min_rate, self.max_rate, self.result_points)

            new_df = pd.DataFrame(index=newdf)
            # merge the dfs
            df = df.combine_first(new_df)

            # fill missing unit counts
            missing_counts_2: pd.Series[bool] = df[unitcountcol].isna()
            df.loc[missing_counts_2, unitcountcol] = df.loc[missing_counts_2].index.to_numpy() * avg_dur

            # fill resources & costs
            varcostcol = "total variable costs"
            fixedcostcol = "total fixed costs"
            capital_costs_col = "Total Capital Cost"
            fillcols: list[str] = [varcostcol, fixedcostcol, capital_costs_col, "resources"]
            # forward fill
            df[fillcols] = df[fillcols].ffill()
            # back fill if needed
            df[fillcols] = df[fillcols].bfill()

            # calculate missing unit costs
            costcol: str = "Total Unit Cost"
            missing_counts_2 = df[costcol].isna()
            missinglocs = df.loc[missing_counts_2]

            df.loc[missing_counts_2, costcol] = (
                missinglocs[varcostcol] + missinglocs[fixedcostcol] / missinglocs[unitcountcol]
            )

            # truncate the dataframe to the min and max
            df = df.loc[df.index >= self.min_rate]
            df = df.loc[df.index <= self.max_rate]
            # TODO:  figure out how to make this one statement

            # add the index back as a column
            df["rate"] = df.index
            # return as resords dict
            sols_dict = df[["rate", "resources", "Total Unit Cost", "Total Capital Cost"]].to_dict(orient="records")

        # Incompatible types in assignment (expression has type "list[dict[Hashable, Any]]", variable has type "dict[str, Any]")
        self.interaction.solutions = sols_dict  # TODO: Find correct type, df.to_dict returns a list of dicts
        return sols_dict

    def _resources_at_point(self, sol: gpx.Model) -> ResourceStep:
        "return a list of resources at the particular solution point"
        reslist: ResourceStep = []
        for res in self.ramp_resources:
            resdict: ResourceDict = res.copy()
            key_value = resdict.get("key")
            if isinstance(key_value, str):
                gpxvarkey = self._uikey_gpxkey_map[key_value]
            else:
                raise TypeError(f"Expected key to be str, but got {type(key_value)}")
            gpxvarkey = self._uikey_gpxkey_map[resdict["key"]]
            solval = sol["variables"][gpxvarkey]
            resdict["value"] = solval
            reslist.append(resdict)
        return reslist

    def _uncertain_collapse_resources(
        self,
        best_recs: list[dict[str, str | np.float64]],
        likely_recs: list[dict[str, str | np.float64]],
        worst_recs: list[dict[str, str | np.float64]],
    ) -> list[dict[str, str | np.float64]]:
        "collapse the resources for the different cases into one list of resources"
        return uncertain_collapse_resources(best_recs, likely_recs, worst_recs)

    def _get_solution_points(
        self,
        sols: list[gpx.Model],
        rates: list[float],
        resource_steps: ResourceStepList,
        cost_name: str = "Total Unit Cost",
        resource_name: str = "resources",
        case: Optional[str] = None,
    ) -> gpkit.SolutionArray:
        # list of results records
        results: gpkit.SolutionArray = []
        unit_count_name: str = "total unit count"
        case = case + " " if case else ""

        pt_obj: dict[str, ResourceStep | np.float64] = {}

        for i, sol in enumerate(sols):
            # non-recurring cost component of the result
            costmod = self.interaction.active_modules["manufacturing"].gpxObject["unitCost"]

            # cnrvar: Any = costmod.nonrecurringCost  # varkey  # Unused variable cnrvar

            # total number of units
            prodtot = sol["variables"][costmod.numUnits]
            trough_numunits = prodtot  # number of units produced at the trough

            # TODO:  record total recurring and non-recurring cost
            capcost: float = sol["variables"].get(costmod.nonrecurringCost, 0)
            fixedcost: float = 0
            fixedcost += sol["variables"].get(costmod.recurringCost, 0)
            fixedcost += sol["variables"].get(costmod.nonrecurringCost, 0)

            # TODO:  get the variable cost
            variablecost: float = sol["variables"].get(costmod.variableCost, 0)

            # get the unit cost (trough)
            # trough_unitcost: float = variablecost + fixedcost / trough_numunits  # Unused variable trough_unitcost

            # fixed and variable costs to add (will be the same for each peak and trough)
            costsdict: dict[str, np.float64] = {
                f"{case}total variable costs": np.float64(variablecost),
                f"{case}total fixed costs": np.float64(fixedcost),
                f"{case}Total Capital Cost": np.float64(capcost),
            }

            # generate a point at the previous peak
            if i > 0:
                # generate the point only if it is not the first point

                # get the number of units at the previous point
                peak_numunits: float = sols[i - 1]["variables"][costmod.numUnits]

                # calulate new unit cost
                peak_unitcost: float = variablecost + fixedcost / peak_numunits

                # calculate the change in amortized non-recurring costs (over fewer units)
                pt_obj = {
                    # slightly offset the rate
                    "rate": np.round(rates[i - 1], decimals=2) * OFFSET_STEP,
                    # add the difference in the non-recurring costs back at the point
                    cost_name: np.round(peak_unitcost, decimals=OTHER_ROUND_DEC),
                    unit_count_name: np.float64(peak_numunits),
                }
                # add the updated resources
                pt_obj[resource_name] = resource_steps[i]
                # update with cost information
                pt_obj.update(costsdict)
                results.append(pt_obj)

            # this is the point at the trough
            pt_obj = {
                "rate": np.round(rates[i], decimals=2),
                cost_name: np.round(sol["cost"], decimals=OTHER_ROUND_DEC),
                unit_count_name: trough_numunits,
            }
            # update with cost information
            pt_obj.update(costsdict)
            # add the updated resources
            pt_obj[resource_name] = resource_steps[i]

            # add the pt_obj to the list
            results.append(pt_obj)

        return results

    def _get_results_at_worst(
        self,
        worst_rate: float,
        colnames: list[str],
        resnames: list[str],
    ) -> pd.DataFrame:
        "get the result to merge in"
        cases = ["best", "likely"]
        # create the record
        sol = {"rate": worst_rate}
        for c in cases:
            # get the solution
            s = self._uncertain_sols_at_worst[c]
            # get the rate
            rate, _, _ = self._get_finmod().get_aggregate_rate(s["variables"][self._get_ratevarkey()], 1)
            # generate the solution point
            pt = self._get_solution_points(
                sols=[s],
                rates=[worst_rate],
                resource_steps=[
                    self._resources_at_point(s)
                ],  # ? Added to a list to match everywhere else this is used, front end and tests pass as expected foing rate hike with best/worst cases.
                resource_name=f"{c} resources",
                cost_name=f"{c} Total Unit Cost",
                case=c,
            )
            sol.update(cast(dict[str, np.float64], pt[0]))

            # update the rate to worst (to get around the rounding)
            sol["rate"] = worst_rate

        df = pd.DataFrame.from_records([sol], index="rate")

        return df

    def get_gpx_model(self, settings: Settings, **kwargs: AcceptedTypes) -> gpx.Model:
        "use call from basis context"
        return self.basis_context.get_gpx_model(settings, **kwargs)

    def get_cost(self) -> float:
        "use call from basis context"
        return self.basis_context.get_cost()

    def get_substitutions(
        self,
        for_assembly: bool = False,
        **kwargs: AcceptedTypes,
    ) -> dict[gpx.Variable, float]:
        "use call from basis context"
        return self.basis_context.get_substitutions(for_assembly=False, **kwargs)

    def _get_col_dicts(
        self,
        cols: list[colNameType],
        cases: list[str],
        coltypes: list[str],
        grouping: str = "cases",
    ) -> dict[str, list[str]]:
        if grouping == "cases":
            looper: list[str] = cases
            loopattr: str = "case"
        if grouping == "coltypes":
            looper = coltypes
            loopattr = "coltype"
        return {loop: [c.colname for c in filter(lambda x: getattr(x, loopattr) == loop, cols)] for loop in looper}
