import numpy as np
import numpy_financial as npf

from api.constants import COST_ROUND_DEC
from api.module_types.production_finance import ProductionFinance
from api.result_generators.non_recurring_cost_results import NRCostResults
from api.result_generators.recurring_cost_results import RecurrCostResults
from api.result_generators.result_gens import ResultGenerator
from api.result_generators.var_cost_results import VarCostResults
import gpx
import gpx.manufacturing
import gpx.primitives
from utils.result_gens_helpers import (build_cost_breakout_list, gen_cashflows, obj_from_var)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import convert_total_recurring_cost


class TotalCostResults(ResultGenerator):
    """total cost

    Index Contributions
    --------------------
    Total Costs | totalCost

    """

    def __init__(
        self,
        gpxsol: gpx.Model,
        settings: Settings,
        unitcost: gpx.primitives.UnitCost,
        varCost: VarCostResults,
        recCost: RecurrCostResults,
        nonrecCost: NRCostResults,
        nonrecurCosts: NRCostResults | None = None,
        parts_per: int = 1,
        kit_name: str = "kit",
        **kwargs: AcceptedTypes,
    ) -> None:
        """
        Arguments
        ---------
        unitcost :
        """
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        sol = self.gpxsol["variables"]
        self.sol = sol
        self.unitcost = unitcost

        self.results["totalCost"] = []

        # Total Unit Cost
        unit_cost_var = unitcost.unitCost
        self.collect_vars["Total Unit Cost"] = unit_cost_var
        self.results["totalCost"].append(obj_from_var(sol, var=unit_cost_var, descr="Total Unit Cost"))

        # Total Non-Recurring Cost
        nonrecurring_cost_var = unitcost.nonrecurringCost
        if nonrecurring_cost_var != 0:
            self.collect_vars["Total Non-Recurring Cost"] = nonrecurring_cost_var
            self.results["totalCost"].append(
                obj_from_var(sol, var=nonrecurring_cost_var, descr="Total Non-Recurring Cost")
            )

        # Total Recurring Cost
        recurring_cost_var = unitcost.recurringCost
        if recurring_cost_var != 0:
            self.collect_vars["Total Recurring Cost"] = recurring_cost_var

            # native $/hr quantity from the solution
            qty_native = self.gpxsol(recurring_cost_var)

            # convert $/hr -> $/<rate_unit>  (month, week, …)
            qty_disp = convert_total_recurring_cost(qty_native, settings)

            # push into results via obj_from_var
            self.results["totalCost"].append(
                obj_from_var(
                    sol=self.gpxsol["variables"],
                    var=recurring_cost_var,
                    descr="Total Recurring Cost",
                    units=f"{settings.default_currency_iso}/{settings.rate_unit}",
                    value=qty_disp.magnitude,
                    round_dec=COST_ROUND_DEC
                )
            )

        # Total Variable Cost
        variable_cost_var = unitcost.variableCost
        if variable_cost_var != 0:
            self.collect_vars["Total Variable Cost"] = variable_cost_var
            self.results["totalCost"].append(obj_from_var(sol, var=variable_cost_var, descr="Total Variable Cost"))

        # Total Production Quantity
        prod_qty_var = unitcost.numUnits
        self.collect_vars["Total Production Quantity"] = prod_qty_var
        self.results["totalCost"].append(obj_from_var(sol, var=prod_qty_var, descr="Total Production Quantity"))
        prod_qty = sol(prod_qty_var).magnitude

        # Total production horizon (in hours)
        prod_horiz = float(sol(unitcost.horizon).to("hr").magnitude)

        # Index for total costs
        self.results_index.append({"name": "Total Costs", "value": "totalCost"})

        # Product Summary variables
        self.summary_res.update({
            "Total Unit Cost":
            float(np.around(sol[unit_cost_var], decimals=COST_ROUND_DEC)),
            "Total Unit Cost Units":
            settings.default_currency_iso,
            "Total Capital Cost":
            float(np.around(sol.get(nonrecurring_cost_var, 0), decimals=COST_ROUND_DEC)),
            "Total Capital Cost Units":
            settings.default_currency_iso,
        })

        # Build the unit cost breakdown.
        # (We assume build_cost_breakout_list converts costResult lists to a list of dictionaries.)
        ucost_res = []
        ucost_res.extend(build_cost_breakout_list(varCost.allcosts, "Variable"))
        ucost_res.extend(
            build_cost_breakout_list(recCost.allcosts, "Recurring", transform_func=lambda x: x * prod_horiz / prod_qty)
        )
        ucost_res.extend(
            build_cost_breakout_list(nonrecCost.allcosts, "Non-Recurring Cost", transform_func=lambda x: x / prod_qty)
        )
        self.results["unitCostBreakout"] = ucost_res

        # If there are multiple parts per kit, roll up total costs accordingly.
        if parts_per and parts_per != 1:

            kit_total_cost = sol[unit_cost_var] * parts_per
            self.results["totalCost"].append({
                "name": f"Total {kit_name} Unit Cost",
                "value": float(np.around(kit_total_cost, decimals=COST_ROUND_DEC)),
                "unit": settings.default_currency_iso,
            })

            # add to auxiliary variables
            self.aux_vars.append({
                "name": f"Total {kit_name} Unit Cost",
                "value": kit_total_cost,
                "unit": settings.default_currency_iso,
                "sensitivity": 0,
                "source": "Calculated Value",
                "category": [],
            })

            if variable_cost_var != 0:
                kit_var_cost = sol[variable_cost_var] * parts_per
                kit_var_cost_units = f"{settings.default_currency_iso}/{kit_name}"
                self.results["totalCost"].append({
                    "name": f"Total {kit_name} Variable Cost",
                    "value": float(np.around(kit_var_cost, decimals=COST_ROUND_DEC)),
                    "unit": kit_var_cost_units,
                })
                # add to auxiliary variables
                self.aux_vars.append({
                    "name": f"Total {kit_name} Variable Cost",
                    "value": kit_var_cost,
                    "unit": kit_var_cost_units,
                    "sensitivity": 0,
                    "source": "Calculated Value",
                    "category": [],
                })

            self.results["totalCost"].append({
                "name": f"Total {kit_name} Production Quantity",
                "value": float(np.around(sol[prod_qty_var] / parts_per)),
                "unit": f"{kit_name}s",
            })
            kitcost_resname = "kitCostBreakout"
            self.results_index.append({"name": f"{kit_name} Part Cost Breakout", "value": kitcost_resname})
            kit_breakout = []
            for c in ucost_res:
                entry = c.copy()
                entry["value"] = float(np.around(entry["value"] * parts_per, decimals=COST_ROUND_DEC))
                kit_breakout.append(entry)
            self.results[kitcost_resname] = kit_breakout

    def add_npv_results(
        self,
        settings: Settings,
        finmod: "ProductionFinance",
        fabline: gpx.manufacturing.FabLine = None,
    ) -> None:
        """adds npv results to the result gens
        patterned after the use of the rate aggregation
        gets called from a solve context in `context.make_solutions`

        Parameters
        ----------
        finmod : ProductionFinance
            finance module to reference for NPV calculation

        Returns
        -------
        [type]
            [description]
        """
        # NPV of all costs
        # numpy has some nice financial helper functions here: https://numpy.org/doc/1.18/reference/routines.financial.html
        # NOTE:  when numpy is upgraded, the financial functions are moved to a separate numpy-financial library
        #       https://pypi.org/project/numpy-financial/
        # npv = {} # Unused var
        # TODO:  can we get access to the model finance module here?
        #       - total units
        #       - total rate
        #       - total horizon

        # get the discount rate from the finance module
        disc_rate = finmod.costOfCapital  # annual discount rate

        # get the number of years (periods)
        num_years = finmod.duration
        if finmod.durationUnit == "months":
            num_years = num_years / 12.0

        # round the number of years up
        num_years = np.ceil(num_years)

        # Recurring Costs
        if self.unitcost.recurringCost != 0:
            # get the number of recurring periods per year
            hrsperyear = finmod.get_hourly_duration(duration=1, durationUnit="years", returnasmon=False)
            hrly_recurrcost = self.sol(self.unitcost.recurringCost).to(f"{settings.default_currency_iso}/hr").magnitude
            # calculate the recurring cost per year
            recurrcostperyear = hrsperyear * hrly_recurrcost
        else:
            # otherwise the recurring costs are just 0
            recurrcostperyear = 0

        # Variable costs
        if self.unitcost.variableCost != 0:
            # the number of units produced annually
            rate = self.sol(self.unitcost.lam)
            annual_production = finmod.get_qty(rate.magnitude, duration=1, duration_units="years")
            # variable cost per unit
            varcost = self.sol(self.unitcost.variableCost).magnitude
            # variable cost cash flow per year
            varcostperyear = varcost * annual_production

        # Calculate NPV values
        # set the target NPV
        target_npv = {"NPV Recurring Cost": recurrcostperyear, "NPV Variable Cost": varcostperyear}
        # get the NPVs
        npv_dict = {name: npf.npv(disc_rate, [val] * int(num_years)) for name, val in target_npv.items()}

        # sum all the NPV
        npv_tot = np.sum(list(npv_dict.values())) + self.sol(self.unitcost.nonrecurringCost).magnitude
        npv_dict["Total NPV Cost"] = npv_tot

        # add to the aux variables
        for name, val in npv_dict.items():
            self.aux_vars.append({
                "name": name,
                "value": np.round(val, decimals=COST_ROUND_DEC),
                "unit": settings.default_currency_iso,
                "sensitivity": 0,
                "source": "Calculated Value",
                "category": [],
            })

            # add to the total cost results
            # self.results['totalCost'][name] = np.round(val, decimals=COST_ROUND_DEC)
            self.results["totalCost"].append(
                obj_from_var(self.sol, var=None, descr=name, value=val, units=settings.default_currency_iso)
            )

        # monthly_total of variable costs
        # monthly production quantity
        # lamvar = finmod.mfg_module.gpxObject['fabLine'].lam
        # rate = self.sol(lamvar).to('count/hr').magnitude
        # # convert the rate to monthly
        # mtot_qty = rate*finmod.get_hourly_duration(duration=1, ops_time=True, durationUnit='months', returnasmon=False)
        # mtot_variable_cost = mtot_qty*self.sol(self.unitcost.variableCost).magnitude

        # # monthly recurring costs
        # #TODO:  aggreagate the recurring costs to a monthly basis
        # mtot_recur = 1000.0

        # # get the number of months
        # num_months = finmod.duration if finmod.durationUnit == 'months' else 12*finmod.duration
        # if not isinstance(num_months, int):
        #     # months count is not an int. round to int
        #     logging.info('RESULT-GENS NPV| rounding number of months from {}'.format(num_months))
        #     num_months = round(num_months)

        # # cashflow for all recurring months
        # cf_tn = {'cashflow' : np.round(mtot_variable_cost + mtot_recur),
        #          'output' : mtot_qty}
        # # cashflow at t0
        # cf_t0 = {'cashflow' : np.round(cf_tn['cashflow'] + self.sol(self.unitcost.nonrecurringCost).magnitude),
        #          'output' : mtot_qty}

        # # make the cashflow list
        # self.results['cashflows'] = [cf_t0]
        # #TODO:  add ramp up & learning curve
        # self.results['cashflows'].extend([cf_tn]*(num_months-1))

        lamvar = finmod.mfg_module.gpxObject["fabLine"].lam  # use the rate of the main line
        costobj = self.unitcost  # use the unit cost
        self.results["cashflows"] = gen_cashflows(sol=self.sol, finmod=finmod, lamvar=lamvar, costobj=costobj)

        # save the total NPV cost as an instance attribute
        self.npv_total = npv_tot
