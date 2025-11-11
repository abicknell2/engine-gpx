"result generators for modules"

import logging
from typing import TYPE_CHECKING, Optional, cast

import numpy as np

from api.constants import MAX_MINUTES_PROCESS, OTHER_ROUND_DEC, TIME_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.manufacturing
from utils.settings import Settings
from utils.types.shared import AcceptedTypes

if TYPE_CHECKING:
    from api.result_generators.cell_results import CellResults


class CellTimes(ResultGenerator):
    """results for the times in cell"""

    def __init__(
        self,
        cells: dict[str, gpx.manufacturing.QNACell],
        gpxsol: gpx.Model,
        settings: Settings,
        cells_in_order: list[tuple[int, str]] | None = None,
        cell_results: Optional["CellResults"] = None,
        **kwargs: AcceptedTypes
    ) -> None:
        """
        Arguments
        ---------
        cells           : dict of cells
        gpxsol          : gpx.Solution
        cells_in_order  : list of the cell names in order
        cell_results    : CellResults object to optionally update
        """
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        sol = self.gpxsol["variables"]
        res = []
        decomp = []

        # yapf: disable
        rate_unit = (
            "hr" if np.average([
                gpxsol(c.t).to("min").magnitude for c in cells.values()
            ]) > MAX_MINUTES_PROCESS else "min"
        )
        # yapf: enable

        for cname, cell in cells.items():
            # loop through the cells
            resdict = {}

            # get the process time
            process_time = self.gpxsol(cell.t).to(rate_unit).magnitude

            # gets the auxiliary time if any
            try:
                aux_time = self.gpxsol(cell.taux).to(rate_unit).magnitude
            except ValueError as e:
                # check to see if this value error is based on a KeyError
                if cell.taux not in sol:
                    # assume this is a key error
                    aux_time = 0.0
                else:
                    # if not, raise the error again
                    raise e

            # if the cell does not have queueing do not add
            if cell.queue_at_cell:
                queueing_time = self.gpxsol(cell.Wq).to(rate_unit).magnitude
            else:
                queueing_time = 0.0

            total_time = process_time + aux_time + queueing_time

            resdict = {
                "Waiting for Next Cell": aux_time,
                "Queueing Before Cell": queueing_time,
                "Processing Time": process_time,
            }

            # convert all the nums to floats and round
            resdict = {name: float(np.around(val, decimals=TIME_ROUND_DEC)) for name, val in resdict.items()}

            # create the decomposition as well
            for vname, val in resdict.items():
                decomp.append({"Cell Name": cname, "time": val, "type": str(vname), "unit": rate_unit})

            # calculate productivity after the decomp
            resdict["Productivity"] = np.around(
                process_time / total_time * 100.0,
                decimals=OTHER_ROUND_DEC,
            )  # [percent] process time to total time

            resdict["Cell Name"] = cname  # add cell name
            resdict["unit"] = rate_unit

            # append the dictionary to the result
            res.append(resdict)

        # sort the results into the order specified by cells_in_order
        if cells_in_order is not None:
            order_in_cells = {name: num for num, name in cells_in_order}
            res.sort(key=lambda x: order_in_cells[x["Cell Name"]])
            decomp.sort(key=lambda x: order_in_cells[x["Cell Name"]])

        self.results["timeInCellDetail"] = res
        self.results["timeInCellDecomp"] = decomp
        self.results_index.extend([
            {
                "name": "Time in Cell Detail",
                "value": "timeInCellDetail",
            },
            {
                "name": "Time in Cell Decomposed",
                "value": "timeInCellDecomp",
            },
        ])

        try:
            if cell_results:
                # if there are cell results as input, update
                self.update_cellresults(cell_results)
        except Exception as e:
            # skip adding the results and log the error
            logging.info(f"RESULT GENS:CELL TIMES | failed to update cell results: {e}")

    def update_cellresults(self, cell_results: "CellResults") -> None:
        "update the cell results with wait for next cell"
        celltimes: dict[str, dict[str, float | np.float64 | str]] = {
            str(v["Cell Name"]): v for v in self.results["timeInCellDetail"] if isinstance(v, dict)
        }
        for c in cell_results.cell_results:
            celldata: dict[str, float | np.float64
                           | str] = cast(dict[str, float | np.float64 | str], celltimes.get(str(c["name"])))
            if celldata:
                # check the units
                crunit = str(c["queueingTimeUnit"])
                ctunit = str(celldata["unit"])
                waittime = float(celldata["Waiting for Next Cell"])

                if "min" in crunit and "hr" in ctunit:
                    waittime = waittime * 60.0
                elif "hr" in crunit and "min" in ctunit:
                    waittime = waittime / 60.0

                # add to cellresults
                c["waitForNextCell"] = waittime
                c["waitForNextCellUnit"] = crunit