"assembly-system"

import logging
from typing import Dict, List

# gpkit imports
from gpkit import ConstraintSet, Variable
from gpkit.nomials.math import Monomial, PosynomialInequality
import numpy as np

import api.interactive_generators as interactive_generators
from api.module_types.module_type import ModuleType
from api.multiproduct import create_from_dict, create_from_variants_dict
from api.objects import CopernicusObject
# copernicus engine imports
from api.result_generators.non_recurring_cost_results import NRCostResults
from api.result_generators.recurring_cost_results import RecurrCostResults
from api.result_generators.result_gens import ResultGenerator
from api.result_generators.var_cost_results import VarCostResults
from api.small_scripts import make_all_vars
from gpx.recurring_cost import VariableCosts
from utils import logger
from utils.result_gens_helpers import index_entry
from utils.settings import Settings
from utils.types.shared import ComponentResults

CONTR_ATTRIBUTES = ["substitutions", "constraints", "lhs"]


class AssemblySystem(ModuleType):
    """[summary]

    Attributes
    ----------
    substitutions : dict
        all of the substituions for the model
    gpx_variables : dict
        the gpx variables contained in the model

    Methods
    -------
    gpx_constraints
        returns all of the gpx constriants
    """

    def __init__(self, default_currency: str, **kwargs):
        self.components = []
        self.modelName = ""
        self.default_currency = default_currency
        self.model_feat_prohib = ["system", "variants"]  # types of models to prohibit in an assembly

        super().__init__(**kwargs)

        # create the components
        compdict = {}
        prohibited_models = {}
        for c in self.components:
            # create a component object
            compname = c["source"].get("modelName")

            if not compname or compname in compdict:
                # raise error on repeated component names
                raise ValueError(f'Cannot repeat component name "{compname}"')

            # get the types from the model
            props = c["source"].get("type", [])
            # check to make sure not a prohibited model types
            prohibited_props = [prop for prop in props if prop in self.model_feat_prohib]
            if prohibited_props:
                # track the prohibited models
                prohibited_models[compname] = prohibited_props
            else:
                # Add the product as usual
                compdict[compname] = AssemblyComponent(construct_from_dict=c)

        # Raise a custom exception if there are problematic models
        if prohibited_models:
            prohib_models_str = {k: " or ".join(v) for k, v in prohibited_models.items()}
            logging.error(f"ASSEMBLY | Prohibited models {str(prohib_models_str)}")
            model_errors = [f"{k} cannot be {v}" for k, v in prohib_models_str.items()]
            error_text = "Models are not allowed in an assembly model: " + ", ".join(model_errors)
            raise ValueError(error_text)

        self.componentdict = compdict
        # TODO:  put this as a single-line comprehension

        # pull the attributes up from the components
        for c in self.componentdict.values():
            for ca in CONTR_ATTRIBUTES:
                if hasattr(c, ca):
                    # update the dict for the component
                    try:
                        getattr(self, ca).update(getattr(c, ca))
                    except AttributeError:
                        # try it as a list
                        getattr(self, ca).extend(getattr(c, ca))

        # append to the variables list from each of the components recursively
        # TODO:  make this a property function
        for c in self.componentdict.values():
            if hasattr(c, "variables"):
                new_vars = {cv.key: cv for cv in c.variables.values()}
                self.variables.update(new_vars)

    def _dicttoobject(self, inputdict):
        # return super()._dicttoobject(inputdict)
        CopernicusObject._dicttoobject(self, inputdict)

    def gpx_translate(self, settings: Settings) -> None:
        "create the gpx constraints"
        # generate on the interactives
        for c in self.componentdict.values():
            c.gpx_translate(settings=settings)
            # update the `self.substitutions`
            self.substitutions.update(c.substitutions)
            self.gpx_variables.update(c.gpx_variables)

    # @property
    # def variables(self) -> list:
    #     'recursive call to all the sub models'
    #     vars = {}
    #     for c in self.componentdict.values():
    #         if hasattr(c, 'variables'):
    #             # loop through each of the components
    #             cvars = {cv['key'] : cv for cv in c.variables}
    #             vars.update(cvars)

    #     return vars

    def make_gpx_vars(self, settings: Settings) -> None:
        "puts variables into `self.gpx_variables`"
        logging.info("in assembly, gpx variables are already generated at the components")
        # update gpx variables from all components
        for c in self.componentdict.values():
            c.make_gpx_vars(settings)
            self.gpx_variables.update(c.gpx_variables)

    def gpx_constraints(self, **kwargs) -> List[PosynomialInequality]:
        "put the list of constraints. add any module-specific constraints"
        constr = []
        for c in self.componentdict.values():
            constr.extend(c.gpx_constraints(**kwargs))

        return constr

    def link_mfg(self, mfg_module: ModuleType) -> None:
        "link to the manufacturing system"
        self.mfg_module = mfg_module
        # update the link based on the link to the manufacturing module
        self._update_mfg_link()

    def _update_mfg_link(self) -> None:
        "update the link from the manufacturing module to this module"
        self.mfg_module.assembly_module = self

    # functions to update the manufacturing module
    def get_variable_cost(self, asdict=False) -> List[VariableCosts]:
        "get the posynomial to add to the destination component cost as a variable cost"
        varcostdict = {f"{cname} Unit Cost": c.unitcost_to_variablecost() for cname, c in self.componentdict.items()}
        if asdict:
            # return the variable costs as a dictionary
            return varcostdict
        # otherwise return as a list
        return list(varcostdict.values())

    def get_separated_costs(self, asdict=False):
        "get the separate costs"
        costs = {}

        return costs

    def get_rate_constraintset(self) -> List[Monomial]:
        "get the constraint set for all the additional rates"
        # TODO:  also get the influence at the cell-level with variability
        constr = []
        for c in self.componentdict.values():
            constr.append(c.gpxvar_linerate / c.gpxvar_assemqty)
        # loop through component objects
        # get a constraint for each with the rate
        return constr

    def get_finance_constraints(self, thoriz: Variable, rate: Variable) -> ConstraintSet:
        "get the equality constraints for the horizon and hlding rate"
        constr = []
        for c in self.componentdict.values():
            constr.extend([
                c.gpxvar_thoriz == thoriz,
                c.gpxvar_rhold >= rate  # put this as a lower bound to prevent infeasability
            ])
        return ConstraintSet(constr)

    def get_results(self, sol, settings: Settings) -> List[ResultGenerator]:
        "get the results"
        # collect all the variables from the modules
        return [AssemComponentResult(sol, self.componentdict, assem_module=self, settings=settings)]

    def get_hierarchy(self) -> Dict:
        "get the hierarchy of components"
        hierarchy = {
            "componentName": self.modelName,
            "childComponents": [c.get_child_components() for c in self.componentdict.values()],
        }
        return hierarchy


class AssemblyComponent(ModuleType):
    "the class for each component in the assembly"

    def __init__(self, **kwargs) -> None:
        self.source = {}
        self.quantity = 0
        self.to = ""
        self.modelName = ""
        self.type = []

        super().__init__(**kwargs)
        # remove any rate input from the input dict since in an assembly
        self.source["manufacturing"].pop("rate")

        # rename all of the variables to include compnent name
        # this is done before creating the interactive so that names get pulled up
        var_update_list = ["variables", "variablesSimple"]
        for m in self.source.get("modules", []):
            v = m[m["type"]]
            for utype in var_update_list:
                if utype in v:
                    self._update_var_names(v[utype], updatekey=True)  # should be able to only update the key

        # create an interaction
        if 'variants' in self.type:
            # use the variants generator
            self.interaction = create_from_variants_dict(self.source)
            # TODO: future, maybe have the interactive_generators automatically return the right generator based on the type of the model
        else:
            # create a regular model
            self.interaction = interactive_generators.createModelFromDict(self.source)

        # create the variables
        for m in self.interaction.active_modules.values():
            # update the self variables with all the variables from each module in the
            # component
            if m:
                self.variables.update(m.variables)
            # implementation to rename. should be using names from the beginning, though
            # self.variables.update({
            #     '[{}] {}'.format(str(self.modelName), str(name)) : param    # add the component name
            #     for name, param in m.variables.items()
            # })

        # create the constraints and substitutions
        # TODO:  determine if this ever gets used
        #       may get used when the
        # constr_attr = ['substitutions', 'constraints', 'lhs']
        constr_attr = CONTR_ATTRIBUTES
        for m in self.interaction.active_modules.values():
            if m:
                for ca in constr_attr:
                    if hasattr(m, ca):
                        # update the dict for the component
                        try:
                            getattr(self, ca).update(getattr(m, ca))
                        except AttributeError:
                            # try it as a list
                            getattr(self, ca).extend(getattr(m, ca))

    def _dicttoobject(self, inputdict) -> None:
        CopernicusObject._dicttoobject(self, inputdict)
        self.modelName = inputdict["source"].get("modelName", "<unknown component>")
        self.type = inputdict['source'].get('type', [])

    def make_gpx_vars(self, settings: Settings) -> None:
        "put the gpx vars in the the object"
        for m in self.interaction.active_modules.values():
            if hasattr(m, "make_gpx_vars"):
                m.make_gpx_vars(settings)
                self.gpx_variables.update(m.gpx_variables)

        # create a variable for the assembly quantity
        self.gpxvar_assemqty = Variable(
            f"{self.modelName} Component Assembly Quantity",
            self.quantity,
            "count",
            "Assembly Quantity",
        )
        # add to list of gpx variables
        self.gpx_variables[self.gpxvar_assemqty.key] = self.gpxvar_assemqty

    def gpx_translate(self, **kwargs) -> None:
        # TODO:  create a for required quantity
        #       constrain the rate
        self.interaction.generate_gpx(for_assembly=True, settings=self.interaction.settings)

        for sname, svar in self.interaction.gpx_model.substitutions.items():
            # get the substitutions from the model
            # if sname in self.interaction.gpx_model:
            try:
                u = self.interaction.gpx_model[sname].units
                u = "" if not u else u.units.__str__()
                subtuple = (svar, u)
                self.substitutions[str(sname)] = subtuple
            except KeyError as e:
                logger.debug(f"AssemblyComponent| not in model {e}")

        # for m in self.interaction.active_modules.values():
        #     self.substitutions.update(m.substitutions)
        #     # or should this be self.interaction._substitutions?
        #     # self.substitutions.update(self.interaction.gpx_model.substitutions)
        #     # self.interaction.gpx_model[varname].units.units.__str__()

        # update substitutions with the produdciton quantity
        self.substitutions[self.gpxvar_assemqty.key] = (self.quantity, "count")

        # set an attribute with the GPX variable for the line rate
        if 'variants' in self.type:
            # get the rate from the system
            sys_mod = self.interaction.active_modules['system']
            self.gpxvar_linerate = sys_mod.get_total_ratevar()

            # expose the finance and holding rate
            self.gpxvar_thoriz = sys_mod.gpxObject['totalCost'].horizon
            ## inventory holding rate is the tha same for all the components of a system
            ## the holding rates should all be the same variable so can just get the first one
            self.gpxvar_rhold = next(iter(sys_mod.gpxObject['invHolding'].values())).holdingRate

            # update variables with the substituions from the cells
            cell_nus = [c.nu for c in sys_mod.gpxObject['system'].flat_cells]  # get all of the variables
            cell_etas = {
                cnu.key.str_without(): cnu for cnu in cell_nus
            }  # make into dict by varname and the actual variable
            self.gpx_variables.update(cell_etas)  # update the components variables
        else:
            # get the rate from the cost
            self.gpxvar_linerate = self.interaction.active_modules["manufacturing"].gpxObject["fabLine"].lam

        # expose finance and holding rate
        mfgmod = self.interaction.active_modules.get("manufacturing")
        mfgmod = mfgmod.gpxObject
        self.gpxvar_thoriz = mfgmod["unitCost"].horizon
        self.gpxvar_rhold = mfgmod["invHolding"].holdingRate

        # set a variable for the unit cost
        # TODO:  should this be the average unit cost or the cost for all of the shipset for variants?
        #       could multiply the average cost by the quantity for the component
        self.gpxvar_unitcost = self.interaction.gpx_model.cost

        # update gpx_variables with any new ones from the interactive
        self.gpx_variables.update(self.interaction.gpx_variables)

    def get_results(self, sol) -> List[ResultGenerator]:
        "get the results from all the modules"
        # TODO:  add the option to flatten
        resgens = []
        for m in self.interaction.active_modules.values():
            if m:  # skipping None modules
                resgens.extend(m.get_results(sol=sol, settings=self.interaction.settings))

        return resgens

    def unitcost_to_variablecost(self) -> VariableCosts:
        "translate the unit cost to a variable cost with quantity"
        return VariableCosts(
            self.gpxvar_assemqty * self.gpxvar_unitcost,
            default_currency=self.interaction.settings.default_currency_iso
        )

    def gpx_constraints(self, **kwargs) -> ConstraintSet:
        "get all the constraints from modules"
        constr = []
        for m in self.interaction.active_modules.values():
            if m:
                constr.extend(m.gpx_constraints(**kwargs))

        # if the product is also a variant, need to get the additional constraints generated by the context
        if 'variants' in self.type:
            # add constraints from the base context
            constr.extend(self.interaction.context.gpx_constraints)

        return ConstraintSet(constr)
        # return ConstraintSet(constr)

    def get_child_components(self) -> Dict:
        "get the components"
        # find if there are any assembly modules in the model
        assymod = self.interaction.active_modules.get("assembly")
        # get any components from the assembly
        childcomps: dict[str, list | str] = assymod.get_hierarchy() if assymod else []
        if len(childcomps) > 0:
            childcomps = childcomps["childComponents"]
        # make the entry
        comps = {
            "componentName": self.modelName,
            "childComponents": childcomps,
        }
        return comps

    def _update_var_names(
        self,
        sourcelist: list[dict[str, bool | int | str], dict[str, float | str]],
        updatename: bool = False,
        updatekey: bool = False,
    ) -> None:
        "inplace update the variables names to include the component"
        for param in sourcelist:
            if updatekey:
                param["key"] = f"{self.modelName} :: {param['key']}"
            # for the name, put the product at the end
            if updatename:
                param["name"] = f"{param['name']} [{self.modelName}]"
            # if updatekey: param['name'] = '{} :: {}'.format(self.modelName, str(param['name']))

        # for param in sourcelist:
        #     if updatekey: param['key'] = '{} [{}]'.format(str(param['key']), self.modelName)
        #     if updatename: param['name'] = '{} [{}]'.format(str(param['name']), self.modelName)


class AssemComponentResult(ResultGenerator):
    "a result generator that puts all of the results into one result generator by component"

    def __init__(
        self,
        gpxsol,
        compdict: Dict[str, AssemblyComponent],
        settings: Settings,
        assem_module: AssemblySystem = None,
        **kwargs,
    ) -> None:
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        # get the results for each component
        component_results = []

        for cname, c in compdict.items():
            # get the result generators from all the modules in the component
            rgs = c.get_results(gpxsol)
            # run through the result generators to get the results
            result_dict = {}
            coll_vars = {}
            for rg in rgs:
                result_dict.update(rg.get_all_results()["results"])
                coll_vars.update(rg.collect_vars)

            # create the all variables for the component
            all_vars, _ = make_all_vars(collected_vars=coll_vars, sol=gpxsol, settings=settings)
            result_dict["allVariables"] = all_vars

            # orient the result as a list of dicts
            component_results.append({
                "componentName": cname,
                "componentResults": result_dict,
                "componentQuantity": c.quantity,
            })

        # re-orient the component results to be a list of objects

        # make the component results as the property for `results`
        resultname = "componentResults"  # dictionary key
        displayname = "Assembly Component Results"
        self.results[resultname] = component_results
        self.results_index.append(index_entry(displayname, resultname))

        # generate the component hierarchy
        if assem_module:
            resultname = "componentHierarchy"
            displayname = "Assembly Component Hierarchy"
            self.results[resultname] = assem_module.get_hierarchy()
            self.results_index.append(index_entry(displayname, resultname))

        # generate the results from the manufacturing module in the assembly
        mfg_resultgens = assem_module.mfg_module.get_results(sol=gpxsol, suppressres_override=True, settings=settings)

        # go through the list of result gens
        # find cost results that need modified. append the rest to this result gen
        for mrg in mfg_resultgens:
            # roll-up the capital costs
            if isinstance(mrg, NRCostResults):
                rollup(
                    component_results=component_results,
                    targetrg=mrg,
                    resultname="costComponents",
                    displayname="Non-Recurring Cost",
                )

            # roll up recurring costs
            elif isinstance(mrg, RecurrCostResults):
                rollup(
                    component_results=component_results,
                    targetrg=mrg,
                    resultname="recurringCosts",
                    displayname="Recurring Cost",
                )

            # roll up variable costs
            elif isinstance(mrg, VarCostResults):
                rollup(
                    component_results=component_results,
                    targetrg=mrg,
                    resultname="variableCosts",
                    displayname="Variable Cost",
                    quantity=True,  # evaluating the assembly quantity on the variable costs
                )

            # add the mrg to this result generator
            # match the properties of the result generator
            self.aux_vars.extend(mrg.aux_vars)
            self.collect_vars.update(mrg.collect_vars)
            self.results.update(mrg.results)
            self.results_index.extend(mrg.results_index)
            self.summary_res.update(mrg.summary_res)


def rollup(
    component_results: ComponentResults,
    targetrg: ResultGenerator,
    resultname: str,
    displayname: str,
    quantity=False,
):
    "rolls up costs in the component results"
    for c in component_results:
        rollup = np.sum([cost["value"] for cost in c["componentResults"][resultname]])

        # check for assembly quantity
        qty = 1
        if quantity:
            qty = c["componentQuantity"]

        # add to the target result generator if not 0
        if rollup != 0.0:
            targetrg.results[resultname].append({
                "name": f"{c['componentName']} Total {displayname}",
                "value": rollup * qty,
            })
