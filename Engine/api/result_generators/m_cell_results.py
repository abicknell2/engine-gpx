import collections

import numpy as np

from api.constants import OTHER_ROUND_DEC, SENS_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.multiclass
from utils.result_gens_helpers import (build_mcell_cell_result_dict, update_mcell_collect_vars)
from utils.types.shared import AcceptedTypes


class MCellResults(ResultGenerator):
    """Results for multiproduct cells"""

    def __init__(
        self,
        gpxsol: gpx.Model,
        mcells: dict[str, gpx.multiclass.MCell],
        cellnames: list[str],
        settings: str,
        from_variants=False,
        **kwargs: AcceptedTypes,
    ) -> None:
        """
        Parameters:
          - gpxsol: The GP model solution.
          - mcells: A dictionary of multiproduct (MCell) objects.
          - cellnames: A list of cell names (if needed).
        """
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        # For MCells, use the variables dictionary...
        var_sol = self.gpxsol["variables"]
        # ...and get the sensitivities dictionary.
        sens_vars = self.gpxsol["sensitivities"]["variables"]

        results: list[dict] = []
        for name, mc in mcells.items():
            # Build the result dictionary by passing both var_sol and sens_vars.
            result = build_mcell_cell_result_dict(var_sol, sens_vars, mc, name, OTHER_ROUND_DEC, SENS_ROUND_DEC)
            result["xSplits"] = {
                c.product_name: round(float(self.gpxsol(c.x)), OTHER_ROUND_DEC) for c in mc.cells if hasattr(c, "x")
            }

            # if it is variants, define some additional parameters as an average
            if from_variants:
                pass

            else:
                # sum queue WIP by number of parts (not number of batches) in the queue
                wip_count = collections.namedtuple('wip_count', ['queue', 'station', 'total'])
                wips = []
                for c in mc.cells:
                    if c.secondary_cell:
                        continue
                    c_lam = gpxsol(c.lam).to('count/hr')
                    lq = gpxsol(c.Wq).to('hr') * c_lam
                    ltot = gpxsol(c.W).to('hr') * c_lam
                    ls = ltot - lq
                    wips.append(wip_count(queue=(lq), station=(ls), total=(ltot)))

                wip_result_names = {
                    'wip': np.sum([w.total.magnitude for w in wips]),
                    'wipQueue': np.sum([w.queue.magnitude for w in wips]),
                    'wipStation': np.sum([w.station.magnitude for w in wips])
                }

                result.update({k: np.round(v, decimals=OTHER_ROUND_DEC) for k, v in wip_result_names.items()})

            # Add the cell to the results
            results.append(result)
            self.collect_vars.update(update_mcell_collect_vars(mc, name))
        self.results["cellResults"] = results
        self.results_index.append({
            "name": "Multi-Product Cells",
            "value": "cellResults",
        })
