from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.manufacturing
from utils.result_gens_helpers import (build_tooling_aux, build_tooling_result, index_entry, update_tooling_collect)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


class MCToolResults(ResultGenerator):  # TODO: Determine if this should be being called anywhere, it currently isn't?

    def __init__(
        self,
        gpxsol: gpx.Model,
        tools: list[gpx.manufacturing.ConwipTooling],
        tool_costs: dict[str, gpx.non_recurring_costs.ConwipToolingCost],
        settings: Settings,
        **kwargs: AcceptedTypes,
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        resultname = "tooling"
        results = []

        # Loop over each tool and build result entries.
        for mctool_name, mctool in tools.items():
            for tool in mctool.tools:

                mctool_costs = {}
                mctool_costs['nonrecurringCost'
                             ] = tool_costs[mctool_name].nonrecurringCost if mctool_name in tool_costs else None

                results.append(build_tooling_result(tool, self.gpxsol, mctool_name, mctool_costs))
                self.collect_vars.update(update_tooling_collect(tool, mctool_name, mctool_costs))
                # Append the auxiliary variable entry for this tool’s utilization.
                self.aux_vars.append(build_tooling_aux(tool, self.gpxsol, mctool_name))

        self.results[resultname] = results
        self.results_index.append(index_entry("Shared Tooling", resultname=resultname))
