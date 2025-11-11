from typing import cast

from gpkit import units
import numpy as np

from api.context.solution_context import SolutionContext
from gpx import Variable
from utils.types.shared import AcceptedTypes


class MinUnitCost(SolutionContext):
    """context to minimize unit cost"""

    def get_cost(self) -> float:
        "minimize unit cost"
        return self.interaction.active_modules["manufacturing"].gpxObject["unitCost"].unitCost

    def get_substitutions(
        self,
        for_assembly: bool = False,
        **kwargs: AcceptedTypes,
    ) -> dict[Variable, float]:
        "get the substitions for the rate"
        mfg_model = self.interaction.active_modules["manufacturing"].gpxObject
        finmod = self.interaction.active_modules["finance"]

        # look to see if this is for assembly purposes
        for_assembly = cast(bool, kwargs.get("for_assembly", False))

        # use finmod inputs
        if finmod.rate is not None:
            # if the rate is specified as normal

            # get the target rate
            rate = np.float64(finmod.get_hourly_rate(finmod.rate))
            lam = mfg_model["fabLine"].lam
            num_units = finmod.get_qty(rate)
            # update rate substitution
            self._substitutions_[lam] = rate * units("count/hr")

            # substitute the quantity or the horizon
            if finmod.amortization == "quantity":
                # subsitute quantity
                num_units -= finmod.rampqty
                num_units = np.float64(num_units)
                # update the substitutions
                self._substitutions_[mfg_model["unitCost"].numUnits] = num_units

            # specify the production horizon
            elif finmod.amortization == "duration":
                # substitute the horizon (in hours) from the finance module
                self._substitutions_[mfg_model["unitCost"].horizon] = finmod.get_hourly_duration()

        # if there is no rate specified
        elif finmod.rate is None and finmod.amortization == "duration":
            # can try and min unit cost
            self._substitutions_.update({
                mfg_model["unitCost"].horizon: finmod.get_hourly_duration(),
            })

        elif not for_assembly:
            # otherwise the problem is not appropriately constrained so raise an error
            # only if not being called for the purposes of an assembly model
            raise ValueError("Need to specify a production rate")

        # update the cost of capital
        self._substitutions_[mfg_model["invHolding"].holdingRate] = finmod.costOfCapital

        # old subsitutions
        # self._substitutions_.update({
        #     mfg_model['fabLine'].lam : float(self.interaction.rate)/20.7/21.0*units('count/hr'),
        #     mfg_model['unitCost'].numUnits : float(self.interaction.quantity)*units('count'),
        #     mfg_model['invHolding'].holdingRate : 0.1,
        # })

        return self._substitutions_

    def get_disc_target_var(self) -> Variable:
        "gets the target variable for a discrete solve"
        # get the rate from the manufacturing module
        mfgm = self.interaction.active_modules["manufacturing"]
        return getattr(mfgm.gpxObject["fabLine"], "lam")

    def get_disc_target_val(self) -> float:
        "gets the target value for the discrete"
        ratevar = self.get_disc_target_var()
        rateval = self.interaction.gpx_model.substitutions.get(ratevar)
        return rateval
