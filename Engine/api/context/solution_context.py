from typing import TYPE_CHECKING, Optional, Union, cast

import gpkit

from api.module_types.production_finance import ProductionFinance
from api.result_generators.line_results import LineResults
from api.result_generators.product_summary import ProductSummary
import api.small_scripts as small_scripts
import gpx
from gpx import Model, Variable
from utils.interactive_helpers import discretize_resources
from utils.settings import Settings
from utils.types.data import Parameter
from utils.types.shared import AcceptedTypes

if TYPE_CHECKING:
    import api.interactive.model as interactive


class SolutionContext:
    """solution context

    - builds the proper constraints
    - provides the correct solver for the interactive to call
    - creates the results

    Attributes
    ----------
    interaction :
        interactive object
    constraints : list
    _substitutions_ : dict
    required_modules : list
        a list of modules which are required to use the context

    Methods
    -------
    make_solutions : make results from the contextualized solution
    make_gpx_constraints : make the GPX constraints specific to the context
    get_constraints : return the additional constraints from the context
    get_cost :
    get_substituions : get the dict of substitions from the context (if any)
    get_gpx_model : gets the gpx model
    solve : calls a solution method
    
    """

    def __init__(self, interaction: "interactive.InteractiveModel") -> None:
        self.interaction: "interactive.InteractiveModel" = interaction
        self.gpx_constraints: Optional[list[object]] = None
        self._substitutions_: dict[Variable, gpkit.nomials.math.Posynomial] = {}
        self.settings = None

    def make_gpx_constraints(self, **kwargs: AcceptedTypes) -> None:
        "make the gpx constraints"
        self.gpx_constraints = []

    def get_gpx_model(self, settings: Settings, **kwargs: AcceptedTypes) -> Model:
        "get the gpx model"
        self.settings = settings
        self.make_gpx_constraints()

        constr = [
            *(self.interaction.gpx_constraints if isinstance(self.interaction.gpx_constraints, list) else []),
            *(self.gpx_constraints if isinstance(self.gpx_constraints, list) else []),
        ]

        # pull in the acyclic parametric models
        return Model(
            self.get_cost(),
            constr,
            acyclic_constraints=self.interaction.acyclic_constraints,
            # self.interaction.gpx_constraints
        )

    def get_cost(self) -> float:
        "cost is context-specific and must be overloaded"
        return 1.0

    def get_substitutions(self, for_assembly: bool = False, **kwargs: AcceptedTypes) -> dict[Variable, float]:
        return self._substitutions_

    def update_substitutions(self, **kwargs: AcceptedTypes) -> None:
        "update the substtions of the interaction"
        self.interaction._substitutions_.update(self._substitutions_)

    def get_disc_target_var(self) -> Optional[Variable]:
        "gets the target variable for a discrete solve"
        return None

    def get_disc_target_val(self) -> Optional[float]:
        "gets the target value for the discrete"
        return None

    def solve(self, max_rate: float | None = None, **kwargs: AcceptedTypes) -> gpx.Model:
        "solve the model"
        # check for discrete solve
        if self.interaction.discretized_resources:
            return discretize_resources(
                interaction=self.interaction,
                discvar=self.get_disc_target_var(),
                discval=self.get_disc_target_val(),
                **kwargs,
            )

        return self.interaction.gpx_model.solve(**kwargs)

    def make_solutions(self, **kwargs: AcceptedTypes) -> dict[str, gpx.Variable]:
        "make the solutions"
        result_gens = []  # this will be a list of dicts

        sol = self.interaction.gpx_solution

        # results containers
        self.interaction.solutions = {}
        self.interaction.solutions["resultsIndex"] = []
        collect_variables: dict[str, Variable] = {}

        # collect the tags
        tags_dict: dict[str, Union[list[str], Parameter]] = {}

        # first collect all the results from the modules
        for m in self.interaction.active_modules.values():
            if hasattr(m, "get_results"):
                result_gens.extend(m.get_results(sol=sol, settings=self.interaction.settings))
                collect_variables.update(m.gpx_variables)
                tags_dict.update(m.variables)

        # update the tags to a dict
        tags_dict = {k: v.tags for k, v in tags_dict.items() if v.tags}

        # get the aggregate production rate
        finmod: ProductionFinance = cast(
            ProductionFinance, self.interaction.active_modules["finance"]
        )  # get the finance module
        linesums = [rg for rg in result_gens if isinstance(rg, LineResults)]  # find the line results
        for ls in linesums:
            # create the aggregate results
            ls.add_agg_rate(finmod=finmod)

        # generate NPV results for any result gens
        for rg in result_gens:
            if hasattr(rg, "add_npv_results"):
                rg.add_npv_results(finmod=finmod, settings=self.settings)

        # get all the summary variables
        for rg in result_gens:
            if isinstance(rg, ProductSummary):
                rg.update_summary()

        aux_vars: list[dict] = []

        # put all the results in proper containers
        for rg in result_gens:
            self.interaction.solutions.update(rg.results)
            self.interaction.solutions["resultsIndex"].extend(rg.results_index)
            collect_variables.update(rg.collect_vars)
            # get the aux variables
            aux_vars.extend(rg.aux_vars)

        # ALL VARIABLES
        # put all the variables in results
        all_vars: list[dict[str, Variable]] = []
        all_gpxvars: dict[str, Variable] = {}
        all_vars, all_gpxvars = small_scripts.make_all_vars(
            collected_vars=collect_variables, sol=sol, tags_dict=tags_dict, settings=self.settings
        )

        # add the aux variables to the all variables
        all_vars.extend(aux_vars)

        # sort the all variables
        all_vars.sort(key=lambda x: str(x["name"]))

        for i, var in enumerate(all_vars):
            if var["unit"] == "hour":
                if isinstance(var["value"], (int, float)):
                    all_vars[i]["value"] = 60.0 * var["value"]
                all_vars[i]["unit"] = "minutes"

        self.interaction.solutions["allVariables"] = cast(AcceptedTypes, all_vars)
        self.interaction.solutions["resultsIndex"].append({"name": "All Variables", "value": "allVariables"})

        # set the attribute for collected variables
        # self.interaction.collected_variables = {name : var for name, var in collect_variables.items() if var in sol['variables']}
        self.interaction.collected_variables = all_gpxvars

        return collect_variables
