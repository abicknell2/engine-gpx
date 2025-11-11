import numbers
from typing import TYPE_CHECKING, Optional, Union, cast

from gpkit import units
import numpy as np

from api.module_types.module_type import ModuleType
import gpx
from utils.types.shared import AcceptedTypes

if TYPE_CHECKING:
    from api.module_types.manufacturing import Manufacturing
    from api.multiproduct import Multiproduct


class ProductionFinance(ModuleType):
    """implements a production finance module
    this module handles all of the rate and shift conversions for a production scenario

    Attributes
    ----------
    shiftsPerDay : int
        number of available shifts per day
    hrsPerShift :
        number of production hours available per shift
    daysPerWeek : int < 7
        number of production days per week
    weeksPerYear : int < 52
        number of production weeks per year

    rateUnit : str
        the units of the input production rate
    durationUnit : str
        the units of the duration

    daysPerMonth :
        average production days per month
    amortiziation : str (default='quantity')
        sets the type of amortization of non-recurring cost
        'quantity' : uses the total number of pieces produced
        'duration' : uses a production horizon

    """

    def __init__(self, mfg_module: "Optional[Manufacturing]" = None, **kwargs: dict[str, AcceptedTypes]) -> None:
        self.shiftsPerDay: int = 2
        self.hrsPerShift: int = 7
        self.daysPerWeek: int = 5
        self.weeksPerYear: int = 50
        self.rateUnit: str = "month"
        self.durationUnit: str = "years"
        self.duration: float = 1
        self.rate: float = 1
        self.quantity: float = 1
        self.costOfCapital: float = 0.1
        self.amortization: str = "quantity"  # set the default amortization method to quantity
        self.inventoryPieceCost: float = 1.0  # other costs per piece of inventory

        # ramp qtys
        self.rampqty: float = 0
        self.rampduration: float = 0

        # coupled manufacturing module
        self.mfg_module: Optional[Union[Multiproduct, ProductionFinance, Manufacturing, Design,
                                        "AssemblySystem"]] = mfg_module

        super().__init__(**kwargs)

        self.hrsPerDay: int = self.shiftsPerDay * self.hrsPerShift
        self.daysPerYear: int = self.daysPerWeek * self.weeksPerYear
        self.daysPerMonth: float = self.daysPerYear / 12.0

        # check that all the inputs are correct
        if self.hrsPerDay > 24:
            raise ValueError("Exceeded 24 hrs per day of production time")
        if self.daysPerWeek > 7:
            raise ValueError("Exceeded 7 days per week")
        if self.daysPerYear > 366:
            raise ValueError("Exceeded days per year")

    def _dicttoobject(self, inputdict: AcceptedTypes, **kwargs: dict[str, AcceptedTypes]) -> None:
        if not isinstance(inputdict, dict):
            raise ValueError("inputdict must be a dictionary")

        finance = inputdict.get("finance", {})

        if not isinstance(finance, dict):
            raise ValueError("finance must be a dictionary")

        prodfin: dict[str, AcceptedTypes] = cast(dict[str, AcceptedTypes], finance)
        if not isinstance(prodfin, dict):
            prodfin = {}

        for v in vars(self).keys():
            if prodfin.get(v, None):
                # only update if the inputdict value is not None
                setattr(self, v, prodfin[v])

        if "costOfCapital" in prodfin:
            # convert input dict cost of capital to percent
            if not isinstance(self.costOfCapital, numbers.Number):
                # check to make sure the cost of capital is a number
                raise ValueError('Input for "Cost of capital" is required')

            self.costOfCapital = self.costOfCapital / 100.0

    def get_total_units(self) -> float:
        return self.get_qty(self.get_hourly_rate(self.rate))

    def get_hourly_rate(self, rate: float | None, **kwargs: AcceptedTypes) -> float | None:
        """converts the input rate to hourly

        Arguments
        ---------
        rate
            the production rate to scale
        rate_unit : str (Optional)
            the units of the input rate
            if not defined, defaults to module rate units
            `['hr', 'day', 'week', 'month', 'year']``

        """
        # check for None rate
        if rate is None:
            return None

        # get the units
        try:
            rate_unit: str = str(kwargs["rate_unit"]).lower()
        except KeyError:
            rate_unit = self.rateUnit

        if rate_unit == "day":
            return rate / self.hrsPerShift / self.shiftsPerDay
        if rate_unit == "week":
            return self.get_hourly_rate(rate / self.daysPerWeek, rate_unit="day")
        if rate_unit == "month":
            return self.get_hourly_rate(rate / self.daysPerMonth, rate_unit="day")
        if rate_unit == "year":
            return self.get_hourly_rate(rate / self.daysPerYear, rate_unit="day")
        if rate_unit == "shift":
            return rate / self.hrsPerShift
        if rate_unit == "hour":
            return rate
        if rate_unit == "min":
            return rate * 60.0

        # TODO:  if not one of the specified units, raise an error
        raise ValueError(f'Could not convert finance module rate to "{rate_unit}"')

    def get_qty(self, hr_rate: float | None, duration: float | None = None, duration_units: str | None = None) -> float:
        "get the quantity at a specific duration as a monomial"

        if self.amortization.lower() == "quantity":
            # just use the quantity for the amortization
            return self.quantity

        if duration_units is None:
            duration_units = self.durationUnit

        if duration is None:
            # duration = self.duration - self.rampduration
            duration = self.duration

        if duration_units.lower() == "years":
            if hasattr(hr_rate, "units"):
                # if the rate has units, need to return the monomial
                if hr_rate is not None:
                    try:
                        return float(hr_rate * self.hrsPerDay * self.daysPerYear * duration * units("hr"))
                    except TypeError:
                        return hr_rate * self.hrsPerDay * self.daysPerYear * duration * units("hr")
                return 0.0
            return float(hr_rate or 0) * self.hrsPerDay * self.daysPerYear * duration

        elif duration_units.lower() == "months":
            if hasattr(hr_rate, "units"):
                # if the rate has units, need to return the monomial
                if hr_rate is not None:
                    try:
                        return float(hr_rate or 0.0 * self.hrsPerDay * self.daysPerYear / 12.0 * duration * units("hr"))
                    except TypeError:
                        return hr_rate * self.hrsPerDay * self.daysPerYear / 12.0 * duration * units("hr")
                return 0.0
            return float(hr_rate or 0) * self.hrsPerDay * self.daysPerYear / 12.0 * duration

        else:
            raise ValueError("Duration units not allowed for", duration_units)

    def get_horizon_duration(self, horizon: float | None = None, durationUnit: str | None = None) -> float:
        "converts the units of a horizon"
        if not horizon:
            horizon = self.duration

        if durationUnit == self.durationUnit:
            # no need to convert
            return horizon
        elif durationUnit == "months" and self.durationUnit == "years":
            return horizon * 12.0
        elif durationUnit == "yeaars" and self.durationUnit == "months":
            return horizon / 12.0
        else:
            raise Exception(f"cannot convert horizon to units {durationUnit}")

    def get_hourly_duration(
        self,
        duration: float | None = None,
        ops_time: bool = True,
        durationUnit: str | None = None,
        returnasmon: bool = True,
    ) -> float:
        "gets the total duration in hours"
        # durcount = None     # number of hourly durations per input duration

        if not durationUnit:
            durationUnit = self.durationUnit

        if not duration:
            # duration = self.duration - self.rampduration
            duration = self.duration

        durcount: float | None = None
        if ops_time:
            if durationUnit == "years":
                # return self.duration*self.hrsPerDay*self.daysPerYear*units('hr')
                durcount = float(self.hrsPerDay * self.daysPerYear)
            elif durationUnit == "months":
                durcount = float(self.hrsPerDay * self.daysPerMonth)
            elif durationUnit == "weeks":
                durcount = float(self.hrsPerDay * self.daysPerWeek)
        else:
            # not operations time but total time
            if durationUnit == "years":
                durcount = 365.0 * 24.0
            elif durationUnit == "months":
                durcount = 365.0 / 12.0 * 24.0
            elif durationUnit == "weeks":
                durcount = 7.0 * 24.0

        if durcount is not None:
            if returnasmon:
                # return the calculation as a monomial
                return duration * durcount
            # otherwise, just return as a scalar
            return duration * durcount

        raise ValueError(f'Duration not entered in the correct units "{self.durationUnit}"')

    def get_aggregate_rate(self, hr_rate: float, hr_cv: float, agg: str | None = None) -> tuple[float, float, str]:
        """gets the aggregated production rate

        Arguments
        ---------
        hr_rate : float
            the hourly production rate
        hr_cv   : float
            the coefficient of variation of the hourly production rate
        agg     : str or `None` (default=None)
            the level at which to aggregate the production rate
            if `None`, uses the `rateUnit` property of the module

        Returns
        -------
        rate, standard deviation, aggregate units

        """

        if agg is None:
            agg = self.rateUnit

        hourly_rate = self.get_hourly_rate(1, rate_unit=agg)
        if hourly_rate is None:
            raise ValueError("Hourly rate cannot be None")
        num_per = 1.0 / float(hourly_rate)

        agg_rate = hr_rate * num_per
        agg_std = np.sqrt(num_per) * hr_cv * hr_rate

        return agg_rate, agg_std, agg

    def get_ramp_qty(self) -> float:
        "get the quantity from the ramp"
        if not hasattr(self, "rates") or self.rates is None or len(self.rates) == 0:
            return 0.0
        return 0.0

    @property
    def num_months(self) -> float:
        return float(self.get_hourly_duration(durationUnit="months", returnasmon=False))

    def change_recurring_basis(self, recur_cost: gpx.primitives.Cost) -> None:
        "shift the input recurring cost to a monthly basis"
        pass
