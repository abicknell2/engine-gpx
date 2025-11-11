import numpy as np

from api.constants import OTHER_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
from utils.result_gens_helpers import build_mcclass_result_dict, index_entry
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


class MCClassResults(ResultGenerator):

    def __init__(
        self,
        gpxsol: gpx.Model,
        settings: Settings,
        classes: dict[str, gpx.multiclass.MClass],
        labor_costs: list[gpx.recurring_cost.LaborCost] | None = None,
        by_split: bool = False,
        **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        sol = self.gpxsol  # full solution object

        #TODO:  Ideas for results
        #           fraction of total cost
        #           fraction of production time
        #           Total class WIP

        resultname = 'productResults'
        results = []

        for name, mc in classes.items():
            results.append(build_mcclass_result_dict(by_split, sol, mc, name))

        self.results[resultname] = results
        self.results_index.append(index_entry('Product Results', resultname=resultname))
