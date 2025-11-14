"""creates an interacive model to be used with Copernicus

Use example
-----------
- create interactive object
    - contains all input elements
    - contains the structural information of the models
- convert to GPX model. Solve. Retrieve the solution
- convert solution to JSON object

"""
import json
import logging
import re
from typing import TYPE_CHECKING, Optional, Union, cast

import gpkit
from gpkit import Variable
from gpkit.exceptions import (DualInfeasible, Infeasible, UnboundedGP, UnknownInfeasible)
from gpkit.tools.autosweep import autosweep_1d
import numpy as np

from api.context.min_unit_cost import MinUnitCost
from api.context.ramp import rampContext
from api.context.solution_context import SolutionContext
from api.context.uncertain import Uncertain
from api.errors import ApiModelError, translate_key
from api.factories.context_factory import make_context
from api.interactive.gpx_builder import GpxModelBuilder
from api.interactive.solve_strategies import BasicSolveStrategy, SolveStrategy
from api.module_types.design import Design
from api.module_types.manufacturing import Manufacturing
from api.module_types.production_finance import ProductionFinance
from api.modules import ModelModules
import gpx
from utils.interactive_helpers import sanitise_solution
from utils.settings import Settings
import utils.types.data as data
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import (include_original_currency_data, replace_unit_display_text, round_currency_values)

if TYPE_CHECKING:
    from api.assembly import AssemblySystem
    from api.multiproduct import Multiproduct, MultiproductContext


class InteractiveModel:
    """An interactive model object

    Adds dynamic capabilities to the front-end model object

    Order to Construct the Model
    ----------------------------
    Modules
    Design
    Processes
    Cells or workzones
    Primary Flows
    Secondary Flows
    Cost Models

    GPX Translation
    ---------------
    Variables
    Models
    Constraints
    Substitutions
    Solve

    Items
    -----
    partName : string
        descriptive name of the part
    partType: string
        the type of part design template to reference

    process : string
        selects the manufacturing process chain from the modules
    quantity : int
        production quantity for amortization
    rate : float
        required monthly production rate
    line : string
        selects the type of production line

    context : context.SolutionContext
        the solution context for the interation
    solution : results.Solution
        holds the solution

    design : data_types.Design
        describes the design
    manufacturing : data_types.Manufacturing
        describes the manufacturing system
    model : data_types.Model
    modules : modules.ModelModules
        the different modules in the model
    active_modules : dict
        the selected modules which are active
        'design' or 'manufacturing' : ModuleType

    GPX Items
    ---------
    gpx_constraints : list
        list of all the constraints
    gpx_model : gpx.Model
        the actual model

    gpx_variables : list
        a list of all of the gpx variables

    _substitutions_ : dict
        a dictionary of parameters that have variables which will require substitution
        before solving
    gpx.Variable : (value to substitute, units)
    """

    def __init__(self, inputdict: AcceptedTypes, settings: Optional[Settings] = None) -> None:
        "Initialize"
        self.partName: str = ""
        self.partType: str = ""
        self.design: Design | None = None
        self.manufacturing: data.Manufacturing | None = None
        self.modules: Optional[Union[list[Union["Multiproduct", ProductionFinance, Manufacturing, Design,
                                                "AssemblySystem"]], ModelModules]] = None
        self.active_modules: (
            dict[str, Union["Multiproduct", ProductionFinance, Manufacturing, Design, "AssemblySystem"]] | ModelModules
        ) = {}

        # context is created via factory to keep original default
        self.context: Union[
            SolutionContext,
            "MultiproductContext",
            Uncertain,
            rampContext,
            MinUnitCost,
        ] = make_context("min_cost", self)

        self.settings = settings or Settings()
        self.gpx_model: gpx.Model | None = None
        self.gpx_solution = None
        self.gpx_variables: dict[str, Variable] = {}

        self.solutions: dict[str, gpx.Model] = {}
        self.collected_variables: dict[str, Variable] = {}

        self.trade_study: bool = False
        self.trade_params: dict[str, AcceptedTypes] = {}
        self.trade_results: AcceptedTypes = None

        self.acyclic_constraints: list[gpkit.ConstraintSet] = []  # constraints for the acyclic models
        self.acyclic_params: dict[str, AcceptedTypes] = {}  # parameters for the acyclic models

        # misc flags
        self.uncertainty_scenario: None | AcceptedTypes = None
        self.quantity: int = 0
        self.rate: float = 0.0
        self.rate_resources: list[str] = []

        # controls for solving context info
        self.discretized_resources: bool = False
        self.is_uncertain: bool = False

        # helpers
        self._gpx_builder = GpxModelBuilder(self, inputdict)
        self._solve_strategy: SolveStrategy = BasicSolveStrategy()

    def set_active_modules(self, amods: dict[str, Union["ProductionFinance", "Manufacturing", "Design"]]) -> None:
        """set the active modules

        Arguments
        ---------
        amods : dict
        """
        self.active_modules.update({name: mod for name, mod in amods.items()})

    def solve(self, max_rate: Optional[float | None] = None, **kwargs: AcceptedTypes) -> None:
        "Solve the model"
        # call the context to solve

        # #TODO: if the interactive is a sweep, create the sweep
        # if self.trade_study:
        #     self._trade_sweep()
        #     return

        try:
            # setting verbosity to 0 will prevent cvxopt from going into debug
            max_rate = max_rate if max_rate is not None else cast(float, kwargs.pop("max_rate", None))
            self.gpx_solution = self._solve_strategy.solve(
                context=self.context,
                max_rate=max_rate,
                **kwargs,
            )
        except Exception as error:
            # check for the different types of errors

            # infeasibility errors
            if isinstance(
                    error,
                (UnknownInfeasible, IndexError)):  # TODO:  gpkit fails creating an error. Fix this with gpkit update
                raise ApiModelError(
                    "Model is infeasible. Check rate requirement and constraints.",
                    status_code=560 if isinstance(error, IndexError) else 559,
                ) from error
            if isinstance(error, DualInfeasible):
                raise ApiModelError("Model is infeasible. Costs trending to 0.", status_code=558) from error
            if isinstance(error, Infeasible):
                raise ApiModelError(
                    f"Model is infeasible. Check for conflicting constraints: {error}",
                    status_code=557,
                ) from error
            if isinstance(error, UnboundedGP):
                # get the name of the variable
                etext = str(error).split("\n")
                logging.error(f"solve error 555: {error}")

                # collect the potential errors
                candidates = [t for t in etext if "QNA" in t or "//" in t]

                if len(etext) > 1:
                    # if the unboundedness is already a user-defined variable return
                    # directly
                    if "//" in etext[0]:
                        raise ApiModelError(etext[0]) from error  # TODO:  test this functionality
                elif "." in etext:
                    # this is a background variable unbounded
                    raise ApiModelError("Could not solve. One or more unbounded variables.", status_code=554) from error

                # bound = "upper" if "upper" in str(error) else "lower"

                errs = [self._find_cell_bound_errors(ctxt) for ctxt in candidates]
                # strip empty errors
                errs = [e for e in errs if e]
                # trim down the errors
                max_error_length = 10
                if len(errs) > max_error_length:
                    errs = errs[:max_error_length]
                if not errs:
                    # log the error interally
                    # don't want to return an empty error
                    raise ValueError('One or more internal variable is missing a bound')

                raise ApiModelError(". ".join(errs), status_code=555) from error

            # if is already a ValueError
            if isinstance(error, ValueError):
                # just return it directly
                raise error
            # general model solver errors
            if isinstance(error, RuntimeWarning) or isinstance(error, ValueError):
                if self.has_uncertainty():
                    # could arise from inside the context. Raise directly
                    logging.error(f"Interactive Uncertainty Model Error | {error}")
                    raise (error)
                else:
                    logging.error(f"INTERACTIVE UNCAUGHT ERROR | {error}")
                raise ApiModelError("Optimal solution not found.", status_code=554) from error

            # but don't throw if an issue arising from the context
            raise error

        # check to see if the model was debugged and update sensitivities
        if self.gpx_solution is not None and "boundedness" in self.gpx_solution:
            # reform the sensitivities so only the
            senss = self.gpx_solution["sensitivities"]["constants"]
            new_sens = {}
            for v in self.gpx_variables.values():
                if senss.get(v.key.__str__(), None):
                    # if the variable is in the solution, replace with only the max
                    # value
                    svars = senss[v.key.__str__()]
                    # find the largest magnitude
                    max_entry = max(svars.items(), key=lambda x: np.abs(x[1]))
                    # create a new entry with the variable key
                    new_sens[v.key] = max_entry[1]

            # update the results with the new sensitivities
            self.gpx_solution["sensitivities"]["constants"].update(new_sens)
            self.gpx_solution["sensitivities"]["variables"].update(new_sens)

    def _trade_sweep(self) -> None:
        "generates a sweep"
        m = self.gpx_model
        xvar = translate_key(str(self.trade_params["xPara"]))
        yvar = translate_key(str(self.trade_params["yPara"]))

        try:
            xvar = self.collected_variables[xvar]
        except KeyError:
            raise ValueError(f"Variable not found: {xvar}")

        if yvar == "Total Unit Cost":
            yvar = m.cost
        else:
            try:
                yvar = self.collected_variables[yvar]
            except KeyError:
                raise ValueError(f"Variable not found: {yvar}")

        lower = float(self.trade_params["xMin"]) if isinstance(self.trade_params["xMin"], (str, int, float)) else 0.0
        upper = float(self.trade_params["xMax"]) if isinstance(self.trade_params["xMax"], (str, int, float)) else 0.0
        self._tradey = yvar

        # get the solve time and estimate points

        solvetime = self.gpx_solution["soltime"]
        maxtime = 60.0 * 3.0  # seconds
        points = maxtime / solvetime
        maxpoints = 100
        points = int(np.min([points, maxpoints]))

        self._trade_pts = np.linspace(lower, upper, points)

        if True:
            logging.info("Starting AUTOSWEEP")
            try:
                self.gpx_sweep_sol = autosweep_1d(m, 1e-4, xvar, [lower, upper])
                # pull the first solution from the bst
                # so can generate the results
                self.gpx_solution = self.gpx_sweep_sol.sols[0]
            except (Infeasible, DualInfeasible, UnknownInfeasible) as e:
                # check to see if there are warnings and elevate them.
                failtext = "Solve Trade Fails. Check to make ensure feasbility for all inputs"
                if self.solutions.get("warnings", None):
                    # include any warnings
                    failtext += "\n===WARNINGS===\n"
                    warnings = self.solutions.get("warnings")
                    if isinstance(warnings, list):
                        failtext += "\n".join(map(str, warnings))
                raise ValueError(failtext) from e
        else:
            # manually sweep
            self.gpx_sweep_sol = m.sweep({xvar: self._trade_pts})

    def generate_gpx(self, settings: str, for_assembly=False, **kwargs: AcceptedTypes) -> None:
        self._gpx_builder.build(settings=settings, for_assembly=for_assembly, **kwargs)

    def create_results(self, **kwargs: AcceptedTypes) -> AcceptedTypes:
        "create the results in context"
        # generate the context solution
        # helps to create the collect_variables
        try:
            sol = self.context.make_solutions(**kwargs)
        except Exception as e:
            logging.error(f"INTERACTIVE CREATE RESULTS| result generation failed: {e}")
            if isinstance(e, ValueError):
                # raise directly from the error
                raise ApiModelError(str(e), status_code=599)
            else:
                raise ValueError("Could not generate results") from e

        # run the trade study
        if self.trade_study:
            self._trade_sweep()
            self._trade_results()
            # return None  # _trade_results only ever returned None

        # TODO: move the uncertainty to here?

        # convert currencies / units exactly as in original helper
        if self.settings.original_currency_data is not None:
            self.solutions = include_original_currency_data(self.solutions, settings=self.settings)

        self.solutions = self.fix_numpy_types(self.solutions)
        self.solutions = sanitise_solution(self)
        self.process_currency_values()
        return sol

    def fix_numpy_types(self, obj):
        """
        Recursively convert numpy int/float types in a nested structure
        to native Python int/float so that json.dumps can handle them.
        """
        if isinstance(obj, dict):
            return {k: self.fix_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.fix_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    def convert_units_for_display(self) -> None:
        self.solutions = replace_unit_display_text(self.solutions)

    def process_currency_values(self) -> None:
        self.solutions = round_currency_values(self.solutions)

    def _trade_results(self) -> None:
        "create the trade study results"

        resx = self._trade_pts
        resy = self.gpx_sweep_sol.sample_at(resx)
        resy = resy[self._tradey]
        if hasattr(resy, "magnitude"):
            resy = resy.magnitude

        res: AcceptedTypes = [{
            str(self.trade_params["xPara"]): float(x),
            str(self.trade_params["yPara"]): float(y),
        } for x, y in zip(resx, resy)]
        # TODO:  put into an object for threading-compatability
        self.solutions["plot"] = res

    def has_uncertainty(self) -> bool:
        "finds if the model has any uncertainty"
        for mod in self.active_modules.values():
            try:
                # loop through the modules to see if there are any uncertain variables
                for var in mod.variables.values():
                    # an uncertain variable has a min and max that are both not none
                    if var.min is not None and var.max is not None:
                        return True
            except AttributeError:
                # ducktyping for modules
                pass

        return False  # if there are no uncertain variables, then there is no uncertainty in the interaction

    def get_rate_resource_vars(self, resources: list[str] | None = None) -> None:
        """get the varkeys of the production resources

        Used for rate hike and discrete resource analysis
        """

        # if there are resources specified
        if resources:
            pass

        if not self.rate_resources or len(self.rate_resources):
            raise ValueError("there must be ")
        pass

    def _find_cell_bound_errors(self, err_text: str) -> str:
        cell_props = {
            "c2a": "arrival flow",
            "c2d": "departure flow",
            "c2s": "process variation",
            "m": "workstation count",
            "Wq": "queueing time"
        }

        # format for the qna cell names
        if "QNACell" in err_text.split(":")[0]:
            if 'manufacturing' in self.active_modules:
                mfg = self.active_modules['manufacturing']
                split_text = err_text.split('.')  # split on the variable
                cell_name = mfg.get_cell_by_qna_name(split_text[0])

                if not cell_name:
                    cell_name = 'An unknown cell'

                # get the unbounded property of the cell
                cprop = ''
                for pn, cp in cell_props.items():
                    if pn in split_text[1]:
                        cprop = cp
                # find the bound
                bound = 'lower' if 'lower' in err_text else 'upper'
                bound = bound + ' '
                # construct the string
                err_format = '{} {} has no {}bound'.format(cell_name, cprop, bound)
                return err_format
            return None
        return err_text.split(", but would gain it from any of these sets:")[0]

    def _get_cellname(self, error_text: str) -> str:
        match_text = r"\bQNACell\\d*"
        r = re.findall(match_text, error_text)
        mfg: Optional[Manufacturing] = cast(
            Manufacturing, self.active_modules["manufacturing"] if isinstance(self.active_modules, dict) else None
        )

        # get the cellname from the manufacturing module
        return mfg.get_cell_by_qna_name(r[0])

    @property
    def module_map(self) -> dict[str, str]:
        """maps from the name of the module to the type of module

        Returns
        -------
        dict
        """
        manufacturing_name: str = str(self.active_modules["manufacturing"].name
                                      ) if isinstance(self.active_modules, dict) else ""
        design_name: str = str(self.active_modules["design"].name) if isinstance(self.active_modules, dict) else ""
        # TODO: What would be better to have as a manufacturing and design name if the modules are not a dict are therefore fail?
        return {
            manufacturing_name: "manufacturing",
            design_name: "design",
        }
