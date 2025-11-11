from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.manufacturing
import gpx.non_recurring_costs
from utils.result_gens_helpers import (build_tool_aux_entry, build_tool_result_entry, index_entry, update_tool_collect)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


class ToolResults(ResultGenerator):
    """Tooling results

    Index Contributions
    -------------------
    Tooling Results | tooling
    """

    def __init__(
        self, gpxsol: gpx.Model, toolcosts: dict[str, gpx.non_recurring_costs.ConwipToolingCost],
        tools: dict[str, gpx.manufacturing.ConwipTooling], settings: Settings, **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        self.resultsname = "tooling"
        self.results[self.resultsname] = []
        for name, t in toolcosts.items():
            # Get the corresponding GP tool object.
            gpx_tool = tools[name]
            entry = build_tool_result_entry(name, t, gpx_tool, self.gpxsol)
            self.results[self.resultsname].append(entry)
            self.collect_vars.update(update_tool_collect(name, t))
            self.aux_vars.append(build_tool_aux_entry(name, self.gpxsol, gpx_tool))

        self.results_index.append(index_entry("Shared Tooling", resultname=self.resultsname))
