from api.constants import COST_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.mfgcosts
import gpx.non_recurring_costs
import gpx.recurring_cost
from utils.result_gens_helpers import (process_cell_costs, process_fs_cost, process_recurcosts)
from utils.settings import Settings
from utils.types.result_gens import costResult
from utils.types.shared import AcceptedTypes


class RecurrCostResults(ResultGenerator):

    def __init__(
        self,
        gpxsol: gpx.Model,
        recur_costs: gpx.recurring_cost.RecurringCosts,
        settings: Settings,
        cell_costs: dict[str, gpx.non_recurring_costs.CellCost] | None = None,
        fs_cost: gpx.mfgcosts.FloorspaceCost | None = None,
        tool_costs=None,
        **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        self.allcosts: list[costResult] = []
        resname = "recurringCosts"
        self.results[resname] = []

        r_cost_objs = recur_costs
        if tool_costs:
            r_cost_objs.update(tool_costs)

        # Process main recurring costs
        recur_entries, recur_costs_list = process_recurcosts(
            self.gpxsol, recur_costs, COST_ROUND_DEC, settings=settings
        )
        self.results[resname].extend(recur_entries)
        self.allcosts.extend(recur_costs_list)

        # Process cell recurring costs
        if cell_costs:
            cell_entries, cell_costs_list = process_cell_costs(
                self.gpxsol, cell_costs, COST_ROUND_DEC, settings=settings
            )
            self.results[resname].extend(cell_entries)
            self.allcosts.extend(cell_costs_list)

        # Process floorspace recurring cost
        if fs_cost:
            fs_entry, fs_cost_obj = process_fs_cost(
                gpxsol=self.gpxsol, settings=settings, fs_cost=fs_cost, round_dec=COST_ROUND_DEC
            )
            if fs_entry is not None:
                self.results[resname].append(fs_entry)
                self.allcosts.append(fs_cost_obj)

        self.results_index.append({"name": "Recurring Cost Components", "value": resname})
