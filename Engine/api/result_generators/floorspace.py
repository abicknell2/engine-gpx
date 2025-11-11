from typing import TYPE_CHECKING, Optional

from api.constants import OTHER_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.mfgcosts
from utils.result_gens_helpers import (
    build_floor_space_summary, collect_cell_floor_space, collect_line_floor_space, update_cellresults_with_floor_space
)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes

if TYPE_CHECKING:
    from api.result_generators.cell_results import CellResults


class FloorSpace(ResultGenerator):
    "Floor space of the factory"

    def __init__(
        self,
        gpxsol: gpx.Model,
        fscosts: gpx.mfgcosts.FloorspaceCost,
        settings: Settings,
        cellresults: Optional["CellResults"] = None,
        **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        # Ensure fscosts is an instance of FloorspaceCost.
        if not isinstance(fscosts, gpx.mfgcosts.FloorspaceCost):
            return

        # Update collect_vars with cell-level floor space variables.
        self.collect_vars.update(collect_cell_floor_space(fscosts))
        # Update collect_vars with line-level floor space variables.
        line_collect = collect_line_floor_space(fscosts)
        self.collect_vars.update(line_collect)

        # Build summary results for total floor space.
        self.summary_res.update(build_floor_space_summary(gpxsol, fscosts.totalFloorSpace, OTHER_ROUND_DEC))

        # Update cell results with floor space info if provided.
        if cellresults:
            update_cellresults_with_floor_space(gpxsol, cellresults.cell_results, fscosts)
