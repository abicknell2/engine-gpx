from collections import namedtuple
import logging
from typing import TYPE_CHECKING, Optional, cast

from gpkit import Variable
import pint

from api.acyclic_interpreter import make_parametric_constraint
import api.module_plugins as module_plugins
from api.objects import CopernicusObject
from api.result_generators.result_gens import ResultGenerator
import gpx
from gpx.dag.parametric import ParametricConstraint, ParametricVariable
import utils.logger as logger
from utils.settings import Settings
from utils.types.data import Parameter

if TYPE_CHECKING:
    from utils.types.shared import LHS, AcceptedTypes, ConstraintsList


class ModuleType(CopernicusObject):
    """
    Items
    -----
    name : string
        the name of the module
    type : string
        the type of the module
    modelInputs : list
        those variables which are not constrained by a posynomial
        mostly used in the internals in the front end
        allows the user to change inputs on the dashboard
        also makes the variables available for the uncertainty analysis
    lhs : dict
        dict of variables on the lhs
        list of variables which appear on the left hand side of constriaints
        useful for constructing the gpx_variables dict
        name : data_type.Parameter
    constraints : list
        the list of posynomial constraints as dicts
    results : dict


    _original_constraints : list
        the original input constraints

    _active_constraints : list
        dict of active constraints

    _del_subs_ : list
        list of substituions to delete from the module

    gpx_variables : dict
        dict of the gpx variables for the module
        name (as used in the constraint) : gpx.Variable
    gpx_constraints : list
        a list of all of the constraints defined in the model
    gpx_substitutions : dict
        dict of varkey : substitution as tuple (<value>, <units>)

    Inherited Items
    ---------------
    infoLabels : dict
        a dictionary of additional keys and values to provide augmented labeling
        for example for implementing trl thermometer.

    gpxObject : (various)
        the related gpx object representation
        this is assigned during the InteractiveModel translation step
            'processes'
            'cells'
            'fabLine'
            'cellCosts'
            'toolCost'
            'unitCost'

    displayParams : list
        the list of attributes which should be revealed externally

    variables : dict
        the variables which are defined in the model with the following entries
        name : data_types.Parameter : the name referred to in the models and the object

    substitutions : dict
        variables which have substitutions which must be performed
        name : substitutions
            the name should be the same as in the variables dict


    """

    def __init__(self, **kwargs: "AcceptedTypes") -> None:
        self.constraints: "ConstraintsList" = []
        self.name: str = ""
        self.type: str = ""
        self.gpx_variables: dict[str, Variable] = {}
        self.lhs: "LHS" = {}
        self._active_constraints: list[dict[str, "AcceptedTypes"]] = []  # TODO: Find correct type
        self._original_constraints: "ConstraintsList" = []
        self._del_subs_: list[str] = []
        self.gpxObject: dict[str, "AcceptedTypes"] = {}  # TODO: Find correct type
        self.gpx_substitutions: dict[str, tuple[float, str]] = {}
        self.id: str | None = None
        self.plugins: list[module_plugins.Plugin] = []
        self.suppress_results: bool = False  # suppress the automatic generation of results

        self.var_sub_names: dict[str, str] = {}  # variable substitutions
        # th variables that are substituted and will need to be replacced in the
        # solution
        self.gpx_var_subs: dict[Variable, Variable] = {}
        # self.results = result_gens.ResultGenerator()
        super().__init__(**kwargs)

        # reset_module_constraints(self)  # initialize active constraints to the
        # constraint list

    def _dicttoobject(self, inputdict: "AcceptedTypes") -> None:
        """construct the model from an input dictionary
        Overloads the inherited function
        """

        # format to match API
        try:
            input = cast(dict[str, list["AcceptedTypes"]], inputdict)
            self.modelInputs: list["AcceptedTypes"] = input["modelInputs"]
            # create variables from model inputs
            self.variables: dict[str, Parameter] = {
                str(model_input["name"]): Parameter(construct_from_dict=model_input)
                for model_input in self.modelInputs
                if isinstance(model_input, dict) and isinstance(model_input.get("name"), str)
            }
        except KeyError:
            logging.info("No model inputs for module")
            self.variables = {}
            self.modelInputs = []
        # self.modelInputs = {iput['name'] : iput for iput in inputdict['modelInputs']}

        # add variables which are not input but still appear in the constraints
        # self.variables.update({
        #     name : vardict
        #     for name, vardict in self.lhs.items() if name not in self.variables.keys()
        # })

        # lhs gets created in the modules
        if isinstance(self.lhs, dict):
            # Handle as a dictionary
            for name, param in self.lhs.items():
                if name not in self.variables:
                    self.variables[name] = param
        elif isinstance(self.lhs, tuple):
            # TODO: Find out why self.lhs is also getting an empty tuple of ({}, [], {})
            pass

        self._subs_from_params()

        # call the super function to assign the rest of the variables
        super()._dicttoobject(inputdict)

    def make_gpx_vars(
        self, settings: Settings, bykey: bool = True, byname: bool = True, **kwargs: "AcceptedTypes"
    ) -> None:
        """generate the gpx variables for a module

        Arguments
        ---------
        module : ModuleType
        bykey : boolean
            (default=True)
            add variables to the dict by key

        """
        for name, param in self.variables.items():
            # TODO:  replace with the generator function from Parameter
            units_ = str(param.unit)

            descr = str(param.name)
            key = str(param.key)
            try:
                newvar = Variable(key, units_, descr)
            except pint.UndefinedUnitError as e:
                raise ValueError(f"Unit error on variable {name}: [{e.args[0]}] is not a unit") from e
            except Exception as e:
                # any other exception
                raise e

            # add variables with param.name
            if bykey:
                self.gpx_variables[name] = newvar
            # add variables with param.key as key
            if byname:
                self.gpx_variables[param.key] = newvar
            # put the varkey back to the parameter
            param.gpx_varkey = newvar.key  # TODO: Find correct type, gpkit varkey?

        # create the gpx substitutions
        self.gpx_substitutions = {}

    def _subs_from_params(self) -> None:
        "create the substitutions from the variables"
        if isinstance(self.substitutions, tuple):
            # TODO: Find out why subs gets an empty tuple too ({}, [], {})
            pass

    # @property
    def gpx_constraints(self, **kwargs: "AcceptedTypes") -> list["AcceptedTypes"]:
        "returns the constraints"
        return []

    def gpx_translate(self, **kwargs: dict[str, "AcceptedTypes"]) -> None:
        "translate the object to gpx. put results in gpxObjects"
        pass

    def get_disp_cost(self, costtype: str) -> "AcceptedTypes":
        "get the display version of the costs"
        if costtype == "varCosts":
            return self.gpxObject["varCosts"]
        if costtype == "laborCosts":
            return self.gpxObject["laborCosts"]
        return []

    def get_results(
        self,
        sol: gpx.Model,
        settings: Settings,
        suppressres_override: bool = False,
        **kwargs: dict[str, "AcceptedTypes"],
    ) -> list["ResultGenerator"]:
        """return the results

        Args
        ----
        sol : gpx.Solution
            the solution

        Returns
        -------


        """
        return [ResultGenerator(sol, settings=settings)]

    def get_acyclic_constraints(
        self,
        parametric_variables: dict[str, ParametricVariable],
        substitutions: dict[str, tuple["AcceptedTypes", "AcceptedTypes"]] | None = None,  # TODO: Find correct type
        additional_variables: dict[str, Variable] = {},
        **kwargs: "AcceptedTypes",
    ) -> list[ParametricConstraint]:
        "create the acyclic constraints"
        # update the parametric_variables from the parent interactive
        # small_scripts.create_parametric_variable()
        aconstr: list[ParametricConstraint] = []

        # create a larger list of variables
        gpxvars = {**self.gpx_variables, **additional_variables}

        for c in self.constraints:
            if c.get("acyclic", False):
                # create the constraint
                aconstr.append(
                    make_parametric_constraint(
                        inputconstraint=c,
                        variables=gpxvars,
                        parametric_vars=parametric_variables,
                        substitutions=substitutions,  # TODO: Find correct type
                    ),
                )

        return aconstr

    def get_plugins(self, plugin_type: type[module_plugins.Plugin]) -> list[module_plugins.Plugin]:
        "gets a list of the plugins of a certain type"
        return list(filter(lambda x: isinstance(x, plugin_type), self.plugins))

    def find_variables(
        self,
        filter_category: str | list[str] | None = None,
        filter_property: str | list[str] | None = None,
        filter_type: str | list[str] | None = None,
        return_data: str = "keys",
        vars_attr: str = "variables",
        emptyisnone: bool = False,
    ) -> list[Variable] | dict[str, Variable] | None:
        """finds the variables according to category and property

        returns the union of all filters

        Arguments
        ---------
        filter_category : str (default=None)
            the category to filter
        filter_property : str (default=None)
            the property string to filter
        filter_type : str (default=None)
            the type to filter
        return_data : str (default='keys')
            the data type to return:
                'keys' : just the keys of the variables
                'list' : just the parameter objects themselves
                'dict' : the keys and parameter objects (just like filtering the self.variables)
        vars_attr : str (default='variables')
            the attribute of self to get the variables
        emptyisnone : bool
            if the list is empty, return None instead
        """

        # create a named tuple to go through the different filters
        filterInput = namedtuple("filterInput", ["inputData", "property", "inputDataList"])

        # make a list of the different options
        input_types = {
            "filter_category": filterInput(inputData=filter_category, property="category", inputDataList=[]),
            "filter_property": filterInput(inputData=filter_property, property="property", inputDataList=[]),
            "filter_type": filterInput(inputData=filter_type, property="type", inputDataList=[]),
        }

        # convert non-iterable inputs to lists
        for i in input_types.values():
            if not isinstance(i.inputData, list):
                # make into a list either with value or None
                i.inputDataList.append(i.inputData)
            else:
                # already a list. Extend the existing list
                i.inputDataList.extend(i.inputData)

        vardict = getattr(self, vars_attr)

        varlist = list(vardict.items())

        # Loop through the data and down-filter
        for i in input_types.values():
            if i.inputData:  # make sure not NONE
                # update the list filtering over multiple values if relevant
                varlist = list(filter(lambda x: getattr(x[1], i.property) in i.inputDataList, varlist))

        if emptyisnone and not varlist:
            return None

        # return the list of keys
        if return_data == "keys":
            return [v[0] for v in varlist]
        elif return_data == "list":
            return [v[1] for v in varlist]
        elif return_data == "dict":
            return dict(varlist)

        return None

    def find_gpx_variable(
        self,
        filter_category: str | list[str] | None = None,
        filter_property: str | list[str] | None = None,
        filter_type: str | list[str] | None = None,
        idx: Optional[int] = 0,
        emptyisnone: bool = False,
        gpx_vars_attr: str = "gpx_variables",
        vars_attr: str = "variables",
        replacevals: list[str] | None = None,
        fail_on_none: bool = False,
    ) -> Variable | list[Variable] | None:
        """returns the gpx variables according to filters

        Arguments
        ---------
        gpx_vars_attr : str (default='gpx_variables')
            the attribute of self which contains the gpx_variables
        replacevals : list
            list of values to replace instead of the whole variable
        """

        # get the varkeys
        varkeys = self.find_variables(
            filter_category=filter_category,
            filter_property=filter_property,
            filter_type=filter_type,
            vars_attr=vars_attr,
            return_data="keys",
        )
        if not varkeys and fail_on_none:
            raise ValueError(
                f"no matching keys found for filters category: {filter_category} | property: {filter_property} | type: {filter_type}",
            )

        gpxvars: list[Variable] = []
        try:
            gpxvars = [self.gpx_variables[vk] for vk in varkeys]
        except KeyError as e:
            # list the varkeys
            logger.debug(f"list of varkeys: {*cast(list[Variable], varkeys), }")
            logger.debug(f"length of gpx variable list: {len(self.gpx_variables)}")
            # log the error
            logging.error(f"error on a missing variable {e}")
            # print more information about the variable
            logging.error(f"getting key name {e.args[0]}")

            # try and access with key string stored in the variable
            logger.debug("trying to access with the keystring that is stored with the variable")
            for vk in varkeys:
                # get the variable key directly from the parameter
                param_var = self.variables[vk]
                logger.debug(f"varkey from parameter: {param_var.key}")
                # try and access it with this varkey
                logger.debug(f"access gpx with the varkey: {param_var.key in self.gpx_variables}")

                # try and get the gpkit varkey directly
                if hasattr(param_var, "gpx_varkey"):
                    logger.debug(f"parameter gpx varkey: {param_var.gpx_varkey}")

            # look at the gpkit varkey in the variable

            # raise the original error
            # TODO:  make sure to re-enable
            raise ValueError(f"key not found: {e} | varkeys: {varkeys}")
            raise e

        if emptyisnone and not gpxvars:
            return None

        if idx is None:
            return gpxvars

        try:
            return gpxvars[idx]
        except IndexError:
            return gpxvars
