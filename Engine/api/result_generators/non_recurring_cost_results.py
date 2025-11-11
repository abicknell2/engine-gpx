from typing import TYPE_CHECKING, Optional

import numpy as np

from api.constants import COST_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
from utils.result_gens_helpers import (process_nr_cell_costs, process_nr_results_dict)
from utils.settings import Settings
from utils.types.result_gens import costResult
from utils.types.shared import AcceptedTypes, CellResults

if TYPE_CHECKING:
    from api.module_types.manufacturing import Manufacturing


class NRCostResults(ResultGenerator):
    """Non-recurring cost

    Index Contributions
    -------------------
    Capital Cost Components | costComponents
    """

    def __init__(
        self,
        gpxsol: gpx.Model,
        module: "Manufacturing",
        settings: Settings,
        cellresults: Optional["CellResults"] = None,
        **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        self.allcosts: list[costResult] = []
        self.resultsname = "costComponents"
        self.results[self.resultsname] = []

        # Process cell costs
        cell_entries, cell_cost_objs, cell_collect = process_nr_cell_costs(self.gpxsol, module, COST_ROUND_DEC)
        self.results[self.resultsname].extend(cell_entries)
        self.allcosts.extend(cell_cost_objs)
        # Update collect_vars for each cell cost found.
        self.collect_vars.update(cell_collect)

        # Process additional cost groups from gpxObject if they are defined as dictionaries.
        if isinstance(module.gpxObject.get("toolCosts"), dict):
            tool_entries = process_nr_results_dict(self.gpxsol, module.gpxObject["toolCosts"])
            self.results[self.resultsname].extend(tool_entries)

        if isinstance(module.gpxObject.get("nonRecurringCost"), dict):
            nr_entries = process_nr_results_dict(self.gpxsol, module.gpxObject["nonRecurringCost"])
            self.results[self.resultsname].extend(nr_entries)

        # TODO: Add floorspace costs if needed

        self.results_index.append({"name": "Capital Cost Components", "value": self.resultsname})

        # Update cellresults with each cell’s capital cost, if provided.
        if cellresults:
            sol = self.gpxsol["variables"]
            for cr in cellresults.cell_results:
                c = module.gpxObject["cellCosts"].get(str(cr["name"]))
                if c:
                    cr["capitalCost"] = (
                        np.round(sol[c.nonrecurringCost], decimals=COST_ROUND_DEC) if c.nonrecurringCost != 0 else 0
                    )
