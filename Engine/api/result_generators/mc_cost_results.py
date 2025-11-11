from typing import TYPE_CHECKING

import numpy as np

from api.constants import COST_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.primitives
from utils.result_gens_helpers import gen_cashflows, process_total_cost
from utils.settings import Settings
from utils.types.shared import AcceptedTypes

if TYPE_CHECKING:
    from api.module_types.production_finance import ProductionFinance


class MCCostResults(ResultGenerator):
    """Gets the cost results from the multiproduct system.

    (Inspired by TotalCostResults.)
    """

    def __init__(
        self, gpxsol: gpx.Model, settings: Settings, prod_costs: dict, unitcost: gpx.primitives.TotalCost,
        **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol, settings, **kwargs)
        sol_vars = self.sol = self.gpxsol["variables"]
        self.unitcost = unitcost

        resultkey = "totalCost"
        self.results[resultkey] = {}

        total_costs, totnr_cost = process_total_cost(unitcost, sol_vars, prod_costs)

        # Add to the results
        self.results[resultkey]['Total Non-Recurring Cost'] = total_costs

        # Add to the auxiliary variables
        self.aux_vars.append({
            'name': 'Total Non-Recurring Cost',
            'value': np.round(totnr_cost, decimals=COST_ROUND_DEC),
            'unit': '$',
            'sensitivity': 0.0,
            'source': 'Calculated Value',
            'category': [],
        })

        # TODO:  re-implement this with a more sophisticed version of TotalCost

        # # create named tuples to handle the different variable names
        # cost_type = collections.namedtuple('cost_type', ('varname', 'dispname', 'collect'))

        # # create a list of the different types of costs to process
        # cost_types = [
        #     cost_type('nonrecurringCost', 'Total Non-Recurring Cost', True) # Non-recurring costs
        # ]

        # # loop over the different types of costs and compile results
        # for ct in cost_types:
        #     # try and get the varkey
        #     cvar = getattr(unitcost, ct.varname)
        #     if cvar and cvar != 0:
        #         # append to the results
        #         self.results[resultkey][ct.dispname] = np.round(sol[cvar], decimals=COST_ROUND_DEC)
        #         # if to be added to the collect variables
        #         if ct.collect:
        #             self.collect_vars[ct.dispname] = cvar

        self.results_index.append({
            "name": "System Total Costs",
            "value": resultkey,
        })

    def add_npv_results(self, settings: Settings, finmod: "ProductionFinance") -> None:
        # Calculate the recurring cashflow at each timestep.
        lamvar = finmod.mfg_module.gpxObject["system"].lam
        costobj = self.unitcost
        self.results["cashflows"] = gen_cashflows(sol=self.sol, finmod=finmod, lamvar=lamvar, costobj=costobj)
