from api.result_generators.process_locator import ProcessLocator
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.feeder
import gpx.manufacturing
import gpx.primitives
from utils.result_gens_helpers import (
    build_feeder_line_results, compute_feeder_flows, compute_local_offsets, compute_total_offsets, index_entry
)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


class FeederFlowtimes(ResultGenerator):
    "Create flowtimes for the feederlines"

    def __init__(
        self,
        gpxsol: gpx.Model,
        gpx_feeder_lines: dict[str, gpx.feeder.FeederLine],
        feeder_lines: list[dict[str, gpx.feeder.FeederLine]],
        feeder_processes: list[dict[str, gpx.primitives.Process]],
        secondary_processes: list[dict[str, gpx.primitives.Process]],
        process_flow: list[dict[str, gpx.primitives.Process]],
        gpx_processes: dict[str, gpx.primitives.Process],
        gpx_cells: dict[str, gpx.manufacturing.QNACell],
        gpx_feeder_cells: dict[str, gpx.manufacturing.QNACell],
        settings: Settings,
        **kwargs: AcceptedTypes,
    ) -> None:

        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        target_unit = "hr"
        flres = self.results["feederLines"] = []
        self.results_index.append(index_entry("Feeder Line Flow Times", resultname="feederLines"))

        # Build process location mapping.
        proc_loc = ProcessLocator(process_flow, feeder_processes, secondary_processes)

        # Build a mapping from each feeder line's 'id' to its 'to' field's process location.
        # Here we assume the "to" field in each feeder line gives a process id.
        feedersfeedlocs = {fl["id"]: proc_loc[fl["to"]] for fl in feeder_lines}

        # Build downstream mapping for each feeder line.# Build downstream mapping for each feeder line.
        feederdownstreams = {}
        for fl in feeder_lines:
            fl_id = fl["id"]
            cur_proc = fl["to"]

            ds = []
            while True:
                if proc_loc.is_main(cur_proc):
                    break

                feeder_line = proc_loc.downstream_feeder(cur_proc)
                if feeder_line is None:
                    break

                ds.append(feeder_line)
                cur_proc = next((f["to"] for f in feeder_lines if f["id"] == feeder_line), None)

            feederdownstreams[fl_id] = ds

        valid_feeder_lines = [fl for fl in feeder_lines if not proc_loc.is_secondary(fl["to"])]

        # Compute feeder flows: process flows, flow times, and end queue times.
        flprocflow, fflowtimes, fendqueues = compute_feeder_flows(gpxsol, gpx_feeder_lines, valid_feeder_lines)

        # Compute local offsets for each feeder line. (Pass in fflowtimes and fendqueues.)
        localoffset = compute_local_offsets(
            gpxsol, process_flow, gpx_processes, gpx_cells, gpx_feeder_cells, feedersfeedlocs, flprocflow, fflowtimes,
            fendqueues, target_unit, valid_feeder_lines
        )

        # Compute total offsets.
        totaloffset = compute_total_offsets(feederdownstreams, localoffset)

        # Build feeder line results.
        flresults = build_feeder_line_results(feeder_lines, fflowtimes, fendqueues, totaloffset, target_unit)
        for res in flresults:
            flres.append(res)
