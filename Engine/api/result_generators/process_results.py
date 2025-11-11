import numpy as np

from api.constants import MAX_MINUTES_PROCESS, TIME_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.primitives
from utils.result_gens_helpers import (build_process_result_entry, update_process_collect)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


class ProcessResults(ResultGenerator):
    """Results for processes"""

    def __init__(
        self, gpxsol: gpx.Model, processchain: list[dict[str, gpx.primitives.Process]],
        processes: dict[str, gpx.primitives.Process], settings: Settings, **kwargs: AcceptedTypes
    ) -> None:
        """
        Parameters:
          - gpxsol: The GP solution.
          - processchain: List of dictionaries (each with a "type" key) specifying the processing order.
          - processes: Dictionary of Process objects.
        """
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        sol = self.gpxsol

        process_results = []
        stacks = []  # If needed for process stacks (not fully implemented here)
        # Extract a list of process names in order from the process chain.
        all_process_names = [p["type"] for p in processchain]

        # Determine target unit for process time (if average processing time in minutes > MAX_MINUTES_PROCESS, use 'hr')
        avg_proc_time = np.average([sol(p.t).to("min").magnitude for p in processes.values()])
        target_units = "hr" if avg_proc_time > MAX_MINUTES_PROCESS else "min"

        # For each process name in the chain, build a result entry.
        for name in all_process_names:
            p = processes[name]
            entry = build_process_result_entry(p, name, sol, target_units, TIME_ROUND_DEC)
            process_results.append(entry)
            # Update collect_vars with the process variable.
            self.collect_vars.update(update_process_collect(p, sol))
            # Optionally, you could also extend "stacks" via decompose_process_time(..., splitstacks=True)
            # For now, we leave stacks empty or handle it as needed.

        self.results["processResults"] = process_results
        self.results["processStacks"] = stacks
        self.results_index.extend([
            {
                "name": "Manufacturing Processes",
                "value": "processResults"
            },
            {
                "name": "Process Time Stack-Ups",
                "value": "processStacks"
            },
        ])
