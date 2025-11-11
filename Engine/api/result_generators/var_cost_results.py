from typing import TYPE_CHECKING

import numpy as np

from api.constants import COST_ROUND_DEC, OTHER_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.primitives
import gpx.recurring_cost
from utils.result_gens_helpers import (process_invholdcost, process_labcosts, process_varcosts)
from utils.settings import Settings
from utils.types.result_gens import costResult
from utils.types.shared import AcceptedTypes


class VarCostResults(ResultGenerator):
    """Variable cost results

    Index Contributions
    --------------------
    Recurring Cost Components | variableCosts
    """

    def __init__(
        self,
        gpxsol: gpx.Model,
        settings: Settings,
        varcosts: dict[str, gpx.primitives.Cost],
        labcosts: list[gpx.recurring_cost.LaborCost] = [],
        invholdcost: gpx.primitives.Cost | None = None,
        **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        self.allcosts: list[costResult] = []
        resname = "variableCosts"

        rounding_digits = 4

        # Process primary variable costs
        var_entries, var_costs = process_varcosts(self.gpxsol, varcosts, rounding_digits)
        self.results[resname] = var_entries
        self.allcosts.extend(var_costs)

        # Process labor costs
        if labcosts:
            lab_entries, lab_costs = process_labcosts(self.gpxsol, labcosts, rounding_digits)
            self.results[resname].extend(lab_entries)
            self.allcosts.extend(lab_costs)

        self.results_index.append({"name": "Variable Cost Components", "value": resname})

        # Total labor cost summary
        if labcosts:
            totlab = np.sum([
                self.gpxsol["variables"][lcost] for labobj in labcosts for lcost in labobj.cellLabor.values()
            ])
            self.summary_res.update({
                "Total Labor Cost": float(np.around(totlab, decimals=COST_ROUND_DEC)),
                "Total Labor Cost Units": settings.default_currency_iso
            })

        # Process inventory holding cost
        if invholdcost:
            inv_entry, inv_cost, totinvcost = process_invholdcost(self.gpxsol, invholdcost, rounding_digits)
            self.results[resname].append(inv_entry)
            self.allcosts.append(inv_cost)
            self.summary_res.update({
                "Total Invenotry Holding Cost":
                float(np.around(totinvcost, decimals=OTHER_ROUND_DEC)),
                "Total Invenotry Holding Cost Units":
                settings.default_currency_iso,
            })
            self.collect_vars["Total Inventory Holding Cost"] = invholdcost.variableCost
