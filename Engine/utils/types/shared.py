from typing import TYPE_CHECKING, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray

import gpx
import gpx.recurring_cost
from types import TracebackType
from utils.types.data import Parameter

if TYPE_CHECKING:
    from api.module_types.production_finance import ProductionFinance
    from gpx import Variable
    from gpx.feeder import FeederLine
    from gpx.non_recurring_costs import NonRecurringCost
    from gpx.recurring_cost import RecurringCosts, VariableCosts
    from utils.types.result_gens import costResult  # noqa: F401

AcceptedTypes = Union[str, int, float, bool, "ProductionFinance", "VariableCosts", "RecurringCosts", "NonRecurringCost",
                      "FeederLine", "Variable", None, tuple["AcceptedTypes"], list["AcceptedTypes"],
                      dict[str, "AcceptedTypes"]]
ExcInfoType = tuple[Type[BaseException], BaseException, Optional[TracebackType]]
RiskPoint = dict[str, float]
RisksRow = dict[str, Union[float, str, list[float], list[RiskPoint]]]
RiskSensitivity = dict[str, float | str]
NumericArray = NDArray[Union[np.float64, np.float32, np.int32, np.int64]]
ResourceDict = dict[str, str]
ResourceStep = list[ResourceDict]
ResourceStepList = list[ResourceStep]
ResourceSteps = dict[str, list[list[dict[str, str]]]]
ResourceStepsList = list[ResourceSteps]
SolutionPointResults = list[dict[str, ResourceStep | np.float64]]

CellResults = list[dict[str, float | np.float64 | int | str]]
ResultsDict = dict[str, Union[list[dict[str, Union[float, str, np.float16, int,
                                                   "costResult"]]], gpx.recurring_cost.Cost, "CellResults",
                              dict[str, float | str], list[Union[dict[str, Union[dict[str, AcceptedTypes], str,
                                                                                 int]]]], dict[str, AcceptedTypes]]]
ResultsList = list[Union[list[dict[str, Union[float, str, np.float16, int, "costResult"]]], gpx.recurring_cost.Cost,
                         "CellResults", dict[str, float | str], list[Union[dict[str, dict[str, AcceptedTypes]],
                                                                           dict[str, AcceptedTypes]]]]]
Results = Union[ResultsDict, ResultsList]
ResultsIndex = list[dict[str, str]]
CollectVars = dict[str, gpx.Variable]
AllResults = dict[str, Union["Results", "ResultsIndex", "CollectVars"]] | tuple["Results", "ResultsIndex",
                                                                                "CollectVars"]
Probabilties = dict[str, Union[np.float64, dict[str, float], list[dict[str, float | str]], list[dict[str,
                                                                                                     list[np.float64]]],
                               list[dict[str, np.float64]], list[dict[str, Union[np.float64, str]]], str]]
ConstraintsList = list[dict[str, Union[int, str, float, bool, np.float64]]]
Substitutions = dict[str, tuple[float, str]]
LHS = dict[str, Parameter]
SubsConstraintsLHSTuple = tuple[Substitutions, ConstraintsList, LHS]
Modules = list[dict[str,
                    bool | dict[str, bool | dict[str, list[dict[str, int | str], dict[str, str]] | list[dict[str, str]]]
                                | list[AcceptedTypes] | list[dict[str, bool | int | str], dict[str, float | str]]
                                | list[dict[str, bool | str], dict[str, str]]
                                | list[dict[str, float | str], dict[str, int | str]] | list[dict[str, str]] | str]
                    | float | list[dict[str, str]] | str],
               dict[str, bool | dict[str, list[AcceptedTypes] | list[dict[str, bool | int | str], dict[str, int | str]]]
                    | float | list[dict[str, str]] | str]]
ComponentResults = list[dict[str,
                             dict[str, dict[str, dict[str, np.float64 | list[dict, dict[str, list[np.float64]]]
                                                      | list[dict[str, np.float64 | str]] | str]]
                                  | dict[str, list | str] | list | list[dict[str, dict[str, np.float64] | float | str]]
                                  | list[dict[str, float | np.float64 | int | str]]
                                  | list[dict[str, float | np.float64 | str]] | list[dict[str, float | str]]
                                  | list[dict[str, np.float64 | list | str]] | list[dict[str, np.float64 | str]]] | int
                             | str]]
