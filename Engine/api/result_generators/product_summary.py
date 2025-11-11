from api.result_generators.result_gens import ResultGenerator
import gpx
from utils.result_gens_helpers import build_product_summary
from utils.settings import Settings
from utils.types.shared import AcceptedTypes


class ProductSummary(ResultGenerator):
    """A summary result for the entire product

    Summary Results
    ---------------
    Total Unit Cost  
    Total Non-Recurring Cost  
    Total Labor Costs  
    Aggregate Production Rate  
    Total Flow Time  
    """

    def __init__(
        self, gpxsol: gpx.Model, resultslist: list[ResultGenerator], settings: Settings, **kwargs: AcceptedTypes
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)
        self.resultslist = resultslist
        self.resname = "productSummary"
        # Initialize the product summary with an empty dictionary inside a list.
        self.results[self.resname] = [{}]
        # Update the summary (via the helper).
        self.update_summary()
        self.results_index.append({
            "name": "Product Summary",
            "value": self.resname,
        })

    def update_summary(self) -> None:
        """Update the product summary by merging the summary_res dictionaries from all result generators."""
        # Retrieve the aggregated summary using the helper.
        aggregated = build_product_summary(self.resultslist)
        # The product summary is stored as a list with a single dictionary.
        self.results[self.resname][0].update(aggregated)
