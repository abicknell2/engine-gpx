from api.result_generators.product_summary import ProductSummary
from api.result_generators.result_gens import ResultGenerator
import gpx
from utils.result_gens_helpers import flatten_product_summaries
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


class MCProductSummary(ResultGenerator):
    """Flatten the product summaries in a system"""

    def __init__(
        self, gpxsol: gpx.Model, settings: Settings, prod_sums: list[ProductSummary], **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        # Use the helper to flatten the product summaries into one dictionary.
        flat_summary = flatten_product_summaries(prod_sums)
        resname = "summaryByProduct"
        self.results[resname] = flat_summary
        self.results_index.append({
            "name": "System Summary By Product",
            "value": resname,
        })
