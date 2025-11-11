from typing import Union

import numpy as np

from api.constants import COST_ROUND_DEC, OTHER_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.recurring_cost
from utils.result_gens_helpers import (
    build_cell_auxiliary, build_labor_cost_entries, build_system_auxiliary, collect_labor_cost_vars,
    compute_cell_totals, compute_system_totals, sort_labor_entries
)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes, CellResults


class LaborCost(ResultGenerator):
    "Direct labor cost"

    def __init__(
        self,
        gpxsol: gpx.Model,
        settings: Settings,
        labcosts: list[gpx.recurring_cost.LaborCost],
        cellresults: Union["CellResults"] = None,
        cellorderdict: dict[str, int] | None = None,
        include_headcount: bool = True,
        **kwargs: AcceptedTypes,
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        self.labcosts = labcosts

        if labcosts:
            # Build labor cost result entries for each cell.
            self.results["laborCosts"] = build_labor_cost_entries(gpxsol=gpxsol, labcosts=labcosts, settings=settings)

            # Compute the total labor cost from the entries.
            tot_labcost = np.sum([c["value"] for c in self.results["laborCosts"] if isinstance(c, dict)])

            # Update collect_vars with the labor cost variables.
            self.collect_vars.update(collect_labor_cost_vars(labcosts))

            # If a cell order dictionary is provided, sort the entries accordingly.
            if cellorderdict:
                self.results["laborCosts"] = sort_labor_entries(self.results["laborCosts"], cellorderdict)

            # Register the labor costs results in the index.
            self.results_index.append({"name": "Labor Cost Results", "value": "laborCosts"})

            # Compute totals by cell based on the first labor cost object's cells_headcount.
            self.cell_totals = compute_cell_totals(gpxsol, labcosts, OTHER_ROUND_DEC)
            # Append each per‑cell auxiliary entry to aux_vars.
            for aux_entry in build_cell_auxiliary(self.cell_totals, OTHER_ROUND_DEC):
                self.aux_vars.append(aux_entry)

            # Compute overall system totals.
            total_headcount, total_hours = compute_system_totals(self.cell_totals)
            # Build system-level auxiliary entries.
            for aux_entry in build_system_auxiliary(total_headcount, total_hours, tot_labcost, include_headcount,
                                                    OTHER_ROUND_DEC, settings=settings):
                self.aux_vars.append(aux_entry)

            # Update the summary with system totals.
            self.summary_res.update({"Total Labor Hours": total_hours})
            if include_headcount:
                self.summary_res.update({"Total Headcount": total_headcount})

            # Automatically update the cell results if provided.
            if cellresults:
                self.update_cell_results(cellresults=cellresults)

    def update_cell_results(self, cellresults: "CellResults") -> None:
        "Add labor results to the cell results"
        # Build a dictionary of the cell labor cost variables.
        lcosts = {name: cvar for labobj in self.labcosts for name, cvar in labobj.cellLabor.items()}
        for c in cellresults.cell_results:
            if c["name"] in lcosts:
                c["laborCost"] = np.round(self.gpxsol["variables"][lcosts[c["name"]]], decimals=COST_ROUND_DEC)
                if c["name"] in self.cell_totals:
                    # Add additional per-cell results if available.
                    c["laborHours"] = np.round(
                        self.cell_totals[c["name"]]["Total Labor Hours"].magnitude,
                        decimals=OTHER_ROUND_DEC,
                    )
                    c["headCount"] = np.round(
                        self.cell_totals[c["name"]]["Total Headcount"].magnitude,
                        decimals=OTHER_ROUND_DEC,
                    )
