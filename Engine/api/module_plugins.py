"plugins for copernics modules"

import logging
from typing import TYPE_CHECKING

import gpkit
from gpkit import Variable, VectorVariable, units
import numpy as np

from gpx.non_recurring_costs import NonRecurringCost
from gpx.primitives import Cost
from utils.types.shared import AcceptedTypes

if TYPE_CHECKING:
    from api.module_types.module_type import ModuleType
    from api.module_types.production_finance import ProductionFinance


class Plugin:
    """plugins go to modules to provide additional analysis

    generally they proivde some abstraction of a set a behaviors that are particular to a certain module
    - logistics, trucking, train, distance travelled, etc
    - material handling
    - agv
    see https://gitlab.com/advancedanalytics/refactor/copernicus-engine/-/issues/94

    Attributes
    ----------
    parent_module : ModuleType
        the parent module to which the plug in is attached
    """

    def __init__(self, parent: 'ModuleType | None') -> None:
        """_summary_

        Parameters
        ----------
        parent : ModuleType
            _description_
        """
        self.parent_module: 'ModuleType | None' = None
        # update the parent
        self.update_parent(parent=parent)

    def update_parent(self, parent: 'ModuleType | None' = None) -> None:
        "update the parent reference"
        if parent:
            # add the parent to the
            self.parent_module = parent
            # add itself to the plugin list of the parent
            self.parent_module.plugins.append(self)


class RateRamp(Plugin):
    """_summary_

    Attributes
    ----------
    ramps : list[]
        each of the ramp steps
    learning : numbers.Number
        any learning processes
    constraints : gpkit.ConstraintSet
    """

    def __init__(
        self, ramps: list[dict[str, float]], learning: float = 0.0, finmod: 'ProductionFinance | None' = None
    ) -> None:
        """_summary_

        Parameters
        ----------
        ramps : list
            _description_
        learning : float, optional
            _description_, by default 0.0
        finmod : ProductionFinance, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        e
            _description_
        """
        # call to super to update the parent module
        super().__init__(parent=finmod)

        # set up ramps
        self.ramps: list[dict[str, float]] = ramps
        self.learning: float = learning

        self.constr: list[gpkit.ConstraintSet] = []

        self._rcosts_nl: list[Variable] = []  # recurring non-labor costs
        self._vcosts_nl: list[Variable] = []  # variable non-balor costs
        self._vcosts_l: list[Variable] = []  # costs that are labor related

        if finmod:
            self.finmod = finmod

        # calculate the
        # create a non-recurring cost for

        # create missing rate or qty for all ramps
        new_ramps: list[dict[str, float]] = []
        for r in self.ramps:
            try:
                if len(r) == 0:
                    # this is an empty row
                    break
                if "rate" not in r or r["rate"] == "":
                    raise ValueError("All points in a rate ramp must have a rate specified")

                # if 'duration' in r:
                #     r['quantity'] = finmod.get_qty(
                #         finmod.get_hourly_rate(r['rate']),
                #         duration=r['duration']
                #     )
                # elif 'quantity' in r:
                #     r['duration'] = r['quantity'] / r['rate']
                # else:
                #     # create the durations
                #     # should just be created in the dur units so no need to do anything fancy
                #     raise ValueError('Specify rate ramp with duration')

                try:
                    r["productivity"] = 1 if r["productivity"] == "" else r["productivity"] / 100.0
                except KeyError:
                    raise ValueError("Labor productivity must be defined")

                if r["productivity"] > 1:
                    logging.warn("Rate ramp productivity exceeds 100%")

            except KeyError as e:
                raise e

            new_ramps.append(r)

        self.ramps = new_ramps

        self.laborcostsvars: VectorVariable = VectorVariable(len(self.ramps), "")

    @property
    def tot_num_units(self) -> float:
        "the total number of units produced"
        main_rate = self.finmod.get_hourly_rate(self.finmod.rate)

        return self.finmod.get_qty(main_rate) + self.ramp_num_units

    @property
    def ramp_num_units(self) -> float:
        "the number of units produced throughout the ramp"
        return float(np.sum([r["quantity"] for r in self.ramps]))

    @property
    def ramp_tot_duration(self) -> float:
        "total duration of the ramp in duration units"
        return float(np.sum([r["duration"] for r in self.ramps]))

    @property
    def gpx_constraints(self, **kwargs: AcceptedTypes) -> list[gpkit.ConstraintSet]:
        "return the gpx constraints as a list"

        return self.constr  # No self.const but there is self.constr?

    def update_finmod(self, finmod: 'ProductionFinance | None' = None) -> None:
        if finmod:
            self.finmod = finmod
            self.finmod.plugins.append(self)

        self.finmod.rampqty = self.ramp_num_units

        if self.finmod.duration <= 0.08:
            raise ValueError(
                "Total program duration must be greater than the total rate ramp duration by at least one month",
            )

        self.finmod.rampduration = self.ramp_tot_duration

    def update_rates(self) -> None:
        new_ramps: list[dict[str, float]] = []
        for r in self.ramps:
            try:
                if "duration" in r:
                    r["quantity"] = self.finmod.get_qty(self.finmod.get_hourly_rate(r["rate"]), duration=r["duration"])
                elif r.get("quantity") and str(r.get("quantity")).strip() != "":
                    r["duration"] = r["quantity"] / r["rate"]
                else:
                    raise ValueError("Specify rate ramp with duration or quantity")
            except KeyError as e:
                logging.error(f"Plugins: Rate Ramp | could not create ramps due to {e}")
                raise e

            new_ramps.append(r)

        self.ramps = new_ramps

    def add_costs(self, *costs: Cost) -> None:
        "adds non-labor costs"

        # filter out any empty costs
        costs = tuple(filter(lambda c: len(c) > 0, costs))

        # take only the variable and non-recurring costs for each element

        for c in costs:
            self._rcosts_nl.append(c.recurringCost)
            self._vcosts_nl.append(c.variableCost)

    def add_labor_costs(self, *lcosts: Cost) -> None:
        "adds labor related costs"
        for lc in lcosts:
            self._vcosts_l.append(lc.variableCost)

    def get_aux_cost(self) -> NonRecurringCost:
        "get the non-recurring cost equivalent over the ramp period"

        # get only the recurring and variable costs for the ramp period

        rcosts = np.sum(self._rcosts_nl) * self.finmod.get_hourly_duration(duration=self.ramp_tot_duration)
        vcosts = np.sum(self._vcosts_nl) * self.ramp_num_units * units("count")

        # loop over all variable costs and all
        lcosts = np.sum([c * r["quantity"] / r["productivity"] for c in self._vcosts_l for r in self.ramps])

        # check to make sure not all the costs are 0
        nrcost = NonRecurringCost(rcosts, vcosts, lcosts, default_currency=self.default_currency)
        # TODO:  if all the inputs are 0, return a 0 nonrecurring cost (or None)
        return nrcost

    def get_ramp_cost_solutions(self, sol: gpkit.SolutionArray, basis: str = "monthly") -> list[dict[str, float]]:
        "get the ramps and their respective costs"
        ramp_costs: list[dict[str, float]] = []
        for r in self.ramps:
            # get all of the existing properties
            ramp_obj: dict[str, float] = {key: val for key, val in r.items()}
            # convert the duration to monthly
            mo_dur = self.finmod.get_horizon_duration(r["duration"], durationUnit="months")
            ramp_obj["duration"] = float(mo_dur)  # has to be an int
            # find the monthly quantity
            mo_qty = r["quantity"] / mo_dur
            ramp_obj["monthQuantity"] = float(mo_qty)
            # add the labor variable cost
            ramp_obj["lVarCost"] = float(
                np.sum([sol(c).magnitude for c in self._vcosts_l if c in sol]) * mo_qty / r["productivity"],
            )

            # TODO:  add other variable and recurring costs
            # yapf: disable
            ramp_obj["recurCost"] = float(
                np.sum([sol(c).magnitude for c in self._rcosts_nl if c in sol])
                * self.finmod.get_hourly_duration(duration=1.0, durationUnit="months", returnasmon=False),
            )
            # yapf: enable

            ramp_obj["varCost"] = float(np.sum([sol(c).magnitude for c in self._vcosts_nl if c in sol]) * mo_qty, )

            # TODO:  find the monthly cost
            ramp_obj["monthCost"] = float(np.sum([ramp_obj[cvar] for cvar in ["lVarCost", "recurCost", "varCost"]]))

            ramp_costs.append(ramp_obj)

        return ramp_costs

    @property
    def ss_addl_costs(self) -> None:
        "steady-state auxiliary cost"

    @property
    def _gpx_constraints(self) -> None:
        "get the constraints from the pluging"
        pass

    @property
    def get_constraints(self) -> list[gpkit.ConstraintSet]:
        return [self.get_aux_cost, self.constr]  # "RateRamp" has no attribute "cosntr"; maybe "constr"?
