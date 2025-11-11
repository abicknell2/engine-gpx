from api.constants import OTHER_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
from utils.result_gens_helpers import (compute_offshift_points, index_entry, update_offshift_collect)
from utils.types.shared import AcceptedTypes


class OffShiftResults(ResultGenerator):

    def __init__(self, gpxsol: gpx.Model, offshiftdict: dict[str, gpx.Variable], **kwargs: AcceptedTypes) -> None:
        super().__init__(gpxsol, **kwargs)

        if offshiftdict:
            sol = self.gpxsol  # use the full model for function calls
            all_pts = []
            # Loop over each offshift variable.
            for cname, c in offshiftdict.items():
                pts = compute_offshift_points(sol, c, cname, OTHER_ROUND_DEC)
                all_pts.extend(pts)
                # Update the collect_vars.
                self.collect_vars.update(update_offshift_collect(c, cname))
            rname = "offshiftResults"
            self.results[rname] = all_pts
            self.results_index.append(index_entry("Off-Shift Cell Results", resultname=rname))
