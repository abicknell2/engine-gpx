from typing import TYPE_CHECKING

import numpy as np

from api.constants import (MAX_MINUTES_FLOW, OTHER_ROUND_DEC, TAT_ROUND_DAYS, TIME_ROUND_DEC)
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.manufacturing
from utils.result_gens_helpers import (
    assign_flow_distribution, build_line_summary, compute_flow_distribution, compute_flow_time
)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes

if TYPE_CHECKING:
    from api.module_types.production_finance import ProductionFinance


class LineResults(ResultGenerator):
    """Results for production line"""

    def __init__(
        self, gpxsol: gpx.Model, fabline: gpx.manufacturing.FabLine, settings: Settings, **kwargs: AcceptedTypes
    ) -> None:
        """
        Parameters:
          - gpxsol: The GP solution.
          - fabline: A FabLine object.
        """
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        # Build line summary.
        var_attr = {
            "Average Production Rate": "lam",
            "Average Mainline Total Flow Time": "W",
            "Total Mainline WIP Inventory": "L",
        }
        sol_vars = self.gpxsol["variables"]
        self.results["lineSummary"] = build_line_summary(
            sol_vars, fabline, var_attr, OTHER_ROUND_DEC, self.collect_vars
        )
        self.results_index.append({
            "name": "Line Summary",
            "value": "lineSummary",
        })
        self.fabline = fabline

        # Compute flow time.
        self.flow_time, self.flow_units, self.flow_time_obj = compute_flow_time(
            sol_vars, fabline, MAX_MINUTES_FLOW, OTHER_ROUND_DEC
        )

        # Save variability from the last cell.
        last_cell = fabline.cells[-1]
        self.flow_cv2 = sol_vars[last_cell.c2d]

        # Compute the flow time distribution.
        probs, pdf_points = compute_flow_distribution(sol_vars, fabline, TIME_ROUND_DEC)
        # Filter and assign units to the distribution.
        flow_distribution = assign_flow_distribution(probs, pdf_points, self.flow_units)

        # Save probabilities (even if not shown in the index).
        self.results["probabilities"] = flow_distribution
        # Save the pdf points separately.
        self.results["pdfpoints"] = flow_distribution["flowTime"]["pdfPoints"]
        self.results_index.append({
            "name": "Flow Time Distribution",
            "value": "pdfpoints",
        })

        # Update product summary.
        self.summary_res.update({
            "Part Flow Time": np.round(self.flow_time, decimals=OTHER_ROUND_DEC),
            "Part Flow Time Units": self.flow_units,
        })

    def add_agg_rate(self, finmod: "ProductionFinance") -> None:
        """Add the aggregate rate to the results."""
        sol_vars = self.gpxsol["variables"]
        rate = sol_vars[self.fabline.lam]
        agg_rate, agg_std, agg = finmod.get_aggregate_rate(rate, np.sqrt(self.flow_cv2))
        agg_unit = f"count/{agg}"
        self.results["lineSummary"].append({
            "name": "Aggregate Production Rate",
            "value": np.around(agg_rate, decimals=OTHER_ROUND_DEC),
            "unit": agg_unit,
        })
        self.results["lineSummary"].append({
            "name": "Standard Deviation of Aggregate Production Rate",
            "value": np.around(agg_std, decimals=OTHER_ROUND_DEC),
            "unit": agg_unit,
        })
        self.aux_vars.append({
            "name": "Aggregate Production Rate",
            "value": np.round(agg_rate, decimals=OTHER_ROUND_DEC),
            "unit": agg_unit,
            "sensitivity": 0,
            "source": "Calculated Value",
            "category": [],
        })
        self.aux_vars.append({
            "name": "Standard Deviation of Aggregate Production Rate",
            "value": np.round(agg_std, decimals=OTHER_ROUND_DEC),
            "unit": agg_unit,
            "sensitivity": 0,
            "source": "Calculated Value",
            "category": [],
        })
        self.summary_res.update({
            "Aggregate Rate": np.round(agg_rate, decimals=OTHER_ROUND_DEC),
            "Aggregate Rate Units": agg_unit,
        })

        # Turnaround time
        ft_in_hr = self.flow_time_obj.to('hr').magnitude
        tat_val = ft_in_hr / finmod.hrsPerDay
        if tat_val > 1:
            # if it covers more than one day of production
            tat_unit = "days"
        else:
            tat_val = self.flow_time
            tat_unit = self.flow_units

        self.aux_vars.append({
            "name": "Average Turnaround Time",
            "value": np.round(tat_val, decimals=TAT_ROUND_DAYS),
            "unit": tat_unit,
            "sensitivity": 0,
            "source": "Calculated Value",
            "category": [],
        })
        self.summary_res.update({
            "Average Turnaround Time": np.round(tat_val, decimals=TAT_ROUND_DAYS),
            "Average Turnaround Time Units": tat_unit,
        })
