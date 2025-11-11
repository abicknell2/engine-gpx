from api.constants import OTHER_ROUND_DEC, SENS_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
import gpx
import gpx.manufacturing
from utils.result_gens_helpers import (build_qna_cell_result_dict, compute_accumulated_flow, update_qna_collect_vars)
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


class CellResults(ResultGenerator):
    """Cell results"""

    def __init__(
        self,
        gpxsol: gpx.Model,
        settings: Settings,
        cells: dict[str, gpx.manufacturing.QNACell],
        cells_in_order: list[tuple[int, str]],
        **kwargs: AcceptedTypes,
    ) -> None:
        """
        Parameters:
          - gpxsol: The GP model solution.
          - cells: A dictionary of QNA cells.
          - cells_in_order: A list of tuples (cell number, cell name) indicating order.
        """
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        # For QNA cells use the full model so that functions may be called (e.g., sol(cell.lam))
        sol = self.gpxsol

        cell_results = []
        for num, name in cells_in_order:
            cell = cells.get(name)
            if cell:
                # Build the result dictionary via our helper.
                result = build_qna_cell_result_dict(sol, cell, name, num, OTHER_ROUND_DEC, SENS_ROUND_DEC)
                cell_results.append(result)
                # Update collect_vars using our helper.
                self.collect_vars.update(update_qna_collect_vars(cell, name))

                if not cell.secondary_cell:
                    # add the flow time if it is not a secondary cell
                    self.collect_vars[f'{name} Total Flow Time'] = cell.W

        # Sort results by cellNumber.
        cell_results.sort(key=lambda x: float(x["cellNumber"]))
        # Compute accumulated flow.
        compute_accumulated_flow(cell_results, OTHER_ROUND_DEC)

        self.cell_results = cell_results
        self.results["cellResults"] = cell_results
        self.results_index.append({
            "name": "Manufacturing Cells",
            "value": "cellResults",
        })
