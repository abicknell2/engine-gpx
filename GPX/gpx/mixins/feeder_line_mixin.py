from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from gpkit import Monomial, Variable

if TYPE_CHECKING:
    from gpx.feeder import FeederLine


@dataclass
class FeederLineMixin:
    """Data mixin for the different types of feederlines"""
    target_line: "FeederLine"
    batch_qty: Union[int, Variable]
    cuml_batch_qty: Union[int, Variable, Monomial]
    CUML_QTY_NAME: str = "cuml_batch_qty"

    def _calculate_batch_accumulation(self):
        "calculates the cumulative batching quantity for the feederline"
        # just get the monomial, don't return the constraint yet
        self.cuml_batch_qty = self.batch_qty * getattr(self.target_line, self.CUML_QTY_NAME, 1)

    def update_target_line(self, target_line: "FeederLine", update_cuml_qty: bool = True):
        "updates the target line"
        # set the target line
        self.target_line = target_line
        if update_cuml_qty:
            self._calculate_batch_accumulation()
