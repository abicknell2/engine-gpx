"multi-product modules"

from collections import OrderedDict as OD
from copy import deepcopy
from itertools import groupby
import logging
import math
from typing import Optional, cast

import gpkit
from gpkit import ConstraintSet, Variable
from gpkit.varkey import VarKey
import numpy as np

from api.context import SolutionContext
from api.interactive.gpx_builder import GpxModelBuilder
from api.interactive.model import InteractiveModel
import api.interactive_generators as interactive_generators
from api.module_plugins import RateRamp
from api.module_translation import associate_cost
from api.module_types.manufacturing import Manufacturing
from api.module_types.module_type import ModuleType
from api.module_types.production_finance import ProductionFinance
from api.result_generators.celltimes import CellTimes
from api.result_generators.labor_costs import LaborCost
from api.result_generators.line_results import LineResults
from api.result_generators.m_cell_results import MCellResults
from api.result_generators.mc_class_results import MCClassResults
from api.result_generators.mc_cost_results import MCCostResults
from api.result_generators.mc_product_summary import MCProductSummary
from api.result_generators.mc_rec_cost import MCNonRecCost, MCRecCost
from api.result_generators.mc_tool_results import MCToolResults
from api.result_generators.non_recurring_cost_results import NRCostResults
from api.result_generators.process_results import ProcessResults
from api.result_generators.product_summary import ProductSummary
from api.result_generators.recurring_cost_results import RecurrCostResults
from api.result_generators.result_gens import ResultGenerator
from api.result_generators.var_cost_results import VarCostResults
from api.constants import OTHER_ROUND_DEC
import gpx
from gpx.dag.parametric import ParametricVariable, XFromSplits
import gpx.manufacturing
from gpx.multiclass import MClass, MCSystem
from gpx.multiclass.costs import ProductVariableCost
from gpx.multiclass.mccell import MCell
from gpx.multiclass.mclass import MClassFeeder
from gpx.multiclass.mctool import MCTool
import gpx.non_recurring_costs
from gpx.non_recurring_costs import CellCost, ConwipToolingCost
from gpx.primitives import Cost, TotalCost
import gpx.recurring_cost
from gpx.recurring_cost import InventoryHolding
from gpx.recurring_cost import LaborCost as GPXLaborCost
from utils.interactive_helpers import discretize_resources
import utils.logger as logger
from utils.multiproduct_helpers import (
    add_x_splits_to_acyclic_constraints, add_rate_links_to_acyclic_constraints, set_floorspace_costs, set_x_rate_values,
    update_w_cap
)
from utils.result_gens_helpers import combine_results
from utils.settings import Settings
from utils.types.data import Parameter
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import update_settings_finance_units


def create_from_dict(inputdict: dict[str, AcceptedTypes], hasvariants: bool = False) -> InteractiveModel:
    """creates a multi-product context from an input dictionary"""
    # check the type of the line
    if isinstance(inputdict, dict) and inputdict.get(
            'line', 'cell') != 'cell':  # default to cell type if cannot get line type from variants
        raise ValueError('Multi Product not available for: %s' % inputdict.get('line', '<unknown line architecture>'))

    # create the model
    settings = Settings()
    settings = update_settings_finance_units(settings=settings, finance=inputdict.get("finance", None))
    interaction = InteractiveModel(inputdict, settings=settings.clone())

    # update the context
    interaction.context = MultiproductContext(interaction)

    # check to see if the model is with variants
    hasvariants = 'variants' in inputdict['type']

    finm: ProductionFinance = ProductionFinance(construct_from_dict=inputdict)
    if hasvariants:
        # TODO:  create the system from variants
        sysm = MultiproductVariants(construct_from_dict=inputdict, finmod=finm, settings=interaction.settings)
    else:
        # create the system in the normal way
        sysm = Multiproduct(construct_from_dict=inputdict, finmod=finm, settings=interaction.settings)

    #TODO:  if its a variants system, create the system using the variants object

    # If the system is producing a kit of parts
    sysm.from_kit = 'partsPer' in inputdict

    # set the finance module inputs
    finm.amortization = str(inputdict["amortization"])
    if sysm.by_split:
        rate_value = inputdict.get("rate", FileNotFoundError)
        if isinstance(rate_value, (str, int, float)):
            finm.rate = float(rate_value)
        else:
            finm.rate = None

    if finm.amortization == "duration":
        duration = inputdict.get("duration")
        if isinstance(duration, (str, int, float)):
            finm.duration = float(duration)
        else:
            raise ValueError(f"Invalid type for duration: {type(duration)}")

    elif finm.amortization == "quantity":
        quantity = inputdict.get("quantity")
        if isinstance(quantity, (str, int, float)):
            finm.quantity = float(quantity)
        else:
            raise ValueError(f"Invalid type for quantity: {type(quantity)}")

    # DEBUG: set the rates manually since not sent from front end
    # inputdict['rates'] = [{'duration': 0.5, 'productivity': 50, 'rate': 10}, {'duration': 0.5, 'productivity': 75, 'rate': 20}]

    # set up the rate ramp
    rates = inputdict.get('rates')
    if rates and len(rates) == 0:
        rates = None

    if rates:
        if not sysm.by_split:
            raise ValueError('Rate Ramp can only be used with system specified by "Split"')

        ramp = RateRamp(rates, finmod=finm)
        sysm.rateramp = ramp  # TODO: Should this have an attribute in the class?
        sysm.rateramp.update_rates()  # TODO: Should this have an attribute in the class?
        # correct the duration from the rate ramps
        finm.duration = finm.duration - sysm.rateramp.ramp_tot_duration  # TODO: Should this have an attribute in the class?

        sysm.rateramp.update_finmod()  # TODO: Should this have an attribute in the class?

    # add the module to the interacive
    interaction.modules = [sysm, finm]

    # also set the active module
    interaction.active_modules = {"system": sysm, "finance": finm}

    # add the system module to the finance module
    finm.mfg_module = sysm

    # process product production rates
    if sysm.by_split:
        # define the system rate
        # if no rate is defined, will be None
        if not isinstance(inputdict["rate"], (str, int, float)):
            sysm.system_rate = None
        else:
            sysm.system_rate = finm.get_hourly_rate(float(inputdict["rate"]))
    else:
        for name, p in zip(list(sysm.products.keys()), sysm.prod_inputs):

            if not isinstance(p, dict):
                raise ValueError(f"Invalid type for rate: {type(p)}")

            manufacturing = p.get("manufacturing", None)
            if manufacturing is None:
                raise ValueError("Manufacturing not defined")

            rate = manufacturing.get("rate", None)
            if rate is None:
                raise ValueError("Rate not defined")

            if not isinstance(rate, (str, int, float)):
                raise ValueError(f"Invalid type for rate: {type(rate)}")

            rate_unit = str(
                p.get("manufacturing", {}).get("rateUnit") or p.get("finance", {}).get("rateUnit") or finm.rateUnit
            ).lower()

            sysm.prod_rates[name] = finm.get_hourly_rate(float(rate), rate_unit=rate_unit)

    # set the duration
    if not isinstance(inputdict.get("duration"), (str, int, float)):
        sysdur = None
    else:
        sysdur = float(inputdict["duration"])

    # TODO:  set the duration by the rate ramp
    if rates:
        # if there is a rate ramp, reduce the total system duration
        sysdur -= ramp.ramp_tot_duration

    # interaction.active_modules["finance"].duration  # TODO: Check if was needed somewhere

    interaction._gpx_builder = GpxModelBuilder(interaction, inputdict)
    return interaction


class Multiproduct(ModuleType):
    """multi-class module

    Attributes
    ----------
    shared_cells : dict
        {<shared cell name> : [(<product>,<cell>)]}
        the cells that are shared among the different products
    rates : OrderedDict
        the requried hourly rates for the different products

    products : OrderedDict of interaction
        the list of the products

    prod_inputs : list of dicts
        the inputs dictionaries for each of the products

    system_rate : float (default=None)
        the total system production rate
        used with split specification of product rates

    by_split : boolean (default=False)
        determines if the rates for the individual products are determined by
        split of the master rate or by their own individual rates

    sys_vars : dict of objects.Parameters
        variables which are defined at the system level

    sys_gpx_vars : dict of gpx.Variables
        gpx variables which are specific to the system

    mixType : str
        ('concurrent', 'phased')
        describes the mix strategy
            'concurrent'  :  multiple product types occupy the line at the same time
            'phased'      :  the line is changed-over between different products. a single product is run at steady state

    """

    def __init__(
        self,
        settings: Settings,
        finmod: Optional[ProductionFinance] = None,
        **kwargs: dict[str, AcceptedTypes]
    ) -> None:
        """initialize the modules

        Keyword Arguments
        -----------------
        products : OrderedDict
            the interacive objects corresponding to the products

        """
        self.products: OD[str, InteractiveModel] = OD()  # pass in the list of interaction that is the products
        self.shared_cells: OD[str, list[str]] = OD()  # OD preserves input order of the shared cells
        self.shared_tools: OD[str, list[str]] = OD()  # OD preserves input order of the shared tools
        self.variables: dict[str, Parameter] = {}
        self.prod_inputs: list[dict[str, AcceptedTypes]] = []
        self.prod_rates: dict[str, float | None] = {}  # product finance data
        self.prod_splits = None
        self.gpxObject: dict[str, AcceptedTypes] = {}
        self.by_split = False  # the system is defined by split
        self.by_kit = False
        self.system_rate: float | None = None  # The production rate of the system (in a split context)
        self.mixType: str | None = None
        self.sys_vars: dict[str, Parameter] = {}
        self.from_kit = False  # is the system producing kits (and should be rolled up)
        self.from_variants = False  # flag if this multiproduct is created from variants
        self.kit_quantity = None  # parts per kit quantity
        self.kit_name = None  # name of the kit
        self.model_feat_prohib = ['assembly']  # any of these keys appearing an the model will cause it to fail
        self.sys_gpx_vars: dict[str, Variable] = {}
        self.settings: Settings = settings

        # any of these keys appearing an the model will cause it to fail
        self.model_feat_prohib = [
            "assembly",
            # "variants",
        ]

        # any property must be explicitly allowed
        self.model_feat_allow = ["custom", "bestworst", "manufacturing"]

        # set the financial module
        self.finmod: Optional[ProductionFinance | None] = finmod

        super().__init__(**kwargs)

        # check for collisions in manufacturing names
        # if there are repeated manufacturing names, append the product name to
        # each manufacturing name to make unique

        mfg_names: list[str] = []  # a lits of the active manufacturing names
        repeated_names: list[str] = []  # a list of the repeated names
        design_names: list[str] = []  # a list of the design module names
        repeated_design_names: list[str] = []
        repeated_prod_names: list[str] = []

        # check to make sure there is at least one product
        if len(self.prod_inputs) == 0:
            raise ValueError("Empty System. Add at least one product to the system.")

        for p in self.prod_inputs:
            # loop over all the input products
            if p["modelName"] in repeated_prod_names:
                # check for non-unique product names
                raise ValueError("Product names must be unique")

            repeated_prod_names.append(str(p["modelName"]))

            mname = get_active_mfg_name(p)
            if mname not in mfg_names:
                # the name is so far unique
                mfg_names.append(str(mname))
            else:
                # there is a conflicting name
                repeated_names.append(str(mname))

            # check for design names
            dname = get_active_design_name(p)
            if dname and dname not in design_names:
                # name is so far unique
                design_names.append(str(dname))
            elif dname and dname in design_names:
                # name is not None and is not unique
                repeated_design_names.append(str(dname))

        # rename variables to avoid collisions with identical manufacturing modules
        if len(repeated_names):
            # if there are repeated names, rename in the input dict
            for p in self.prod_inputs:
                p: dict[str, AcceptedTypes]
                # create a formatted postfix to add throughout
                prodname: str = p["modelName"]
                postfix: str = f" ({prodname})"

                for m in p['modules']:
                    m: dict[str, AcceptedTypes]
                    # loop over the modules and rename if necessary
                    if m['type'] == 'manufacturing' and m['name'] in repeated_names:
                        # update the manufacturing name
                        m['name'] = m['name'] + postfix

                        # go through all variables and update keys
                        for v in m['manufacturing']['variables']:
                            if isinstance(v, dict) and isinstance(v["key"], str):
                                v: dict[str, gpkit.Variable]
                                v['key'] = v['key'] + postfix

                        # go through variablesSimple and update keys
                        for v in m['manufacturing']['variablesSimple']:
                            if isinstance(v, dict) and isinstance(v["key"], str):
                                v: dict[str, gpkit.Variable]
                                v['key'] = v['key'] + postfix

        # check for collisions with design module names
        if len(repeated_design_names):
            # there are repeated design module names
            for p in self.prod_inputs:
                p: dict[str, AcceptedTypes]
                prodname: str = p["modelName"]
                postfix: str = f" ({prodname}) "

                for m in p['modules']:
                    m: dict[str, AcceptedTypes]
                    if m['type'] == 'design' and m['name'] in repeated_design_names:
                        # update the module name
                        m['name'] += postfix

                        # update variable names
                        for v in m['design']['variables']:
                            v: dict[str, AcceptedTypes]
                            v['key'] += postfix

        # import the products
        if 'products' in kwargs:
            self.products = kwargs['products']

        # create the interactions
        self.make_prod_interactions()

        # generate the gpx for all the product-specific interactives
        for name, prod in self.products.items():
            # generate the product interacives
            prod.generate_gpx(settings)

            # add the substitutions
            self.substitutions.update(rename_product_dict(name, prod._substitutions_))

            # pull up all of the variables from the subproducts
            # DEBUG: see if moving this call to `self.passed_variables` messes anything up
            #       might be covered by the call to rename_product_dict
            self.passed_variables = {}
            for mod in prod.active_modules.values():
                if mod:
                    # ignores empty modules (e.g. design)
                    for varname, modvar in mod.variables.items():
                        self.variables[f"{name} :: {varname}"] = modvar
                        self.passed_variables[f"{name} :: {varname}"] = modvar

                # self.variables.update(mod.variables)

    def make_prod_interactions(self) -> None:
        """Create the interactive products from the sources."""
        prohibited_models: dict[str, list[str]] = {}

        for i, product in enumerate(self.prod_inputs):
            # Check for prohibited model features
            props = product.get("type", [])
            prohibited_props = [prop for prop in props if prop in self.model_feat_prohib]
            if prohibited_props:
                prohibited_models[product.get('modelName', '<UNKNOWN MODEL>')] = prohibited_props
            else:
                # Add the product as usual
                prod = interactive_generators.createModelFromDict(product)
                self.products[product['modelName']] = prod

                # TODO:  append the kitSplit information to the product
                #       get directly from the inputs if possible
                if self.by_split and self.kit_quantity:
                    # TODO:  look for the split as part of the product
                    # recover the split from the rate
                    prod.split = self.prod_splits[i]
                    prod.kitSplit = self.prod_splits[i] * self.kit_quantity

                if self.by_split:
                    # add the split to the product
                    prod.split = self.prod_splits[i]

        # Raise a custom exception if there are problematic models
        if prohibited_models:
            prohib_models_str = {k: " or ".join(v) for k, v in prohibited_models.items()}
            logging.error(f"MULTIPRODUCT | Prohibited models {str(prohib_models_str)}")
            model_errors = [f"{k} cannot be {v}" for k, v in prohib_models_str.items()]
            error_text = "Models are not allowed in a multi-product system: " + ", ".join(model_errors)
            raise ValueError(error_text)
            # TODO:  raise a custom exception
            # raise CustomMultiProductException(error_text)

    def _dicttoobject(self, inputdict: AcceptedTypes) -> None:
        "overloads the parent function"

        product_input_list: list = inputdict['products']

        # set condition flags based on model information
        self.by_split = inputdict.get('bySplit', False)  # the production is defined by the split or partsPer
        self.by_kit = inputdict.get('byKit', False)  # the system products have parts per kit definition
        self.from_variants = inputdict.get('fromVariant', False)  # get a flag if coming from variants

        calc_splits_from_partsper = False  # flag to see if need to calculate the splits for each product

        if self.by_split:
            # check that all products have inputs
            if self.by_kit and check_kit_in_variants(product_input_list):
                pass
            elif all(['split' in p for p in product_input_list]):
                pass
            else:
                raise ValueError('All products need kit quantity or split defined.')

            if 'partsPer' in inputdict:
                # overall kit quantity
                self.from_kit = True
                self.kit_quantity = inputdict['partsPer']
            elif self.by_kit:
                self.from_kit = True
                # have to calculate the total kit quantity from each part
                # self.kit_quantity = np.sum([p['source']['manufacturing']['partsPer'] for p in product_input_list])
                calc_splits_from_partsper = True

        # get the mix strategy
        self.mixType = str(inputdict.get("mixType", "")) if inputdict.get("mixType") else None

        # get the sources for each product
        # if the input is a variant, store in a different list
        var_prods = []  # list of sources for variants
        var_prod_names = []  # list of the source names of the variants
        new_prods = []  # list of regular products

        # if importing from variants, will need the type of parameterization
        if self.by_split:
            if self.by_kit:
                system_rate_parameterization = 'kit'
            else:
                system_rate_parameterization = 'split'
        else:
            system_rate_parameterization = 'rate'

        sys_rate = inputdict.get('rate', None)

        for p in inputdict['products']:
            src = p['source']
            # if any of the products are variants, get the converted dict
            if 'variants' in src['type']:
                # prod_source = _sys_dict_from_variant_dict(src, rename_prods=True,
                #   target_parameterization=system_rate_parameterization)
                var_prod_names.append(src['modelName'])

                # if by split, need to find the overall splut for the product

                # convert the smashed variant
                prod_source = smash_variant_to_system_dict(
                    p,
                    target_parameterization=system_rate_parameterization,
                    sys_rate=sys_rate,
                )

                # append the updated product source
                var_prods.append(prod_source)

            # elif:   # if any of the input products are systems
            else:
                # otherwise keep as a product and add to the list
                new_prods.append(p)

        # update the inputdict with plain products
        inputdict['products'] = new_prods

        # remove variant names from usedby
        remove_names_from = ['tools', 'cells']
        for n in remove_names_from:
            for t in inputdict[n]:
                # update to only include names that are not the names of variants
                t['usedBy'] = list(filter(lambda x: x not in var_prod_names, t['usedBy']))

        # update the inputdict with variant products
        for vp in var_prods:
            # tools: update usedby
            update_usedby(target_list=inputdict['tools'], update_from=vp['tools'])
            # cells : update usedby
            update_usedby(target_list=inputdict['cells'], update_from=vp['cells'])
            # check the rate

            # products: append to list
            inputdict['products'].extend(vp['products'])

        if calc_splits_from_partsper:
            self.kit_quantity = np.sum([p['source']['manufacturing']['partsPer'] for p in inputdict['products']])

        # set up the product splits
        self.prod_splits = []
        if self.by_split:
            if self.by_kit:
                # interpret parts per as splits
                for p in inputdict['products']:
                    # find the split
                    split = p['source']['manufacturing']['partsPer'] / self.kit_quantity
                    p['split'] = split * 100.0
                    self.prod_splits.append(split)
            else:
                self.prod_splits = [p['split'] / 100. for p in inputdict['products']]

        if not self.prod_splits:
            rate_splits = []
            for p in inputdict['products']:
                rate_splits.append(p['source']['manufacturing']['rate'])

        # run the usual conversion
        # get the sources for each product
        self.prod_inputs = [
            cast(dict[str, AcceptedTypes], p["source"])
            for p in inputdict.get("products", [])
            if isinstance(p, dict) and "source" in p
        ]

        # add the variables from the system itself
        # create a key for the variable to make uniform to other module variables
        for v in inputdict.get("variables", []):
            # use the get function and if there are no variables just give it an empty
            # array to skip over
            if isinstance(v, dict):
                v["key"] = f"Shared Resource :: {v['name']}"

        # put parameters for the system into system variables
        self.sys_vars = {
            str(v["key"]): Parameter(construct_from_dict=v)
            for v in inputdict.get("variables", [])
            if isinstance(v, dict) and "key" in v
        }

        # update self.variables with the system variables
        self.variables.update(self.sys_vars)

        # update the substitutions
        for param in self.sys_vars.values():
            if param.unit == "" or param.unit is None:
                unit = ""
            else:
                unit = str(param.unit)
            self.substitutions[param.key] = (param.value, unit)

        # if defined by splits also get the rates
        if self.by_split:
            products = inputdict.get("products", [])

            if isinstance(products, list):
                for p in products:
                    if not isinstance(p, dict):
                        raise ValueError(f"Invalid item: expected 'p' to be a dict, got {type(p).__name__} instead.")

                    source = p.get("source")
                    if not isinstance(source, dict):
                        raise ValueError(f"Invalid source: expected dict, got {type(source).__name__} instead. p={p}")

                    model_name = source.get("modelName")
                    if not isinstance(model_name, str):
                        raise ValueError(
                            f"Invalid 'modelName': expected string, got {type(model_name).__name__} instead. p={p}"
                        )

                    split = p.get("split")
                    if not isinstance(split, (int, float)):
                        raise ValueError(
                            f"Invalid 'split': expected number (int or float), got {type(split).__name__} instead."
                        )

                    split_value = split / 100.0
                    self.prod_rates[model_name] = split_value

            # get the system production rate
            self.system_rate = inputdict.get("rate")

        # add shared cell resources
        for sc in inputdict["cells"]:
            # loop through the input cells and get the shared cells
            self.shared_cells[sc["name"]] = sc["usedBy"]

        # add any tooling
        tools = inputdict.get("tools", [])
        for st in tools:
            if isinstance(st, dict):
                self.shared_tools[str(st["name"])] = cast(list[str], st["usedBy"])

    def make_gpx_vars(
        self, settings: Settings, bykey: bool = True, byname: bool = True, **kwargs: AcceptedTypes
    ) -> None:
        "get all of the pgx variables"
        # make all of the variables in the product interactives
        self.gpx_variables = {}

        # generate a gpx variable for each system parameter
        self.sys_gpx_vars = {name: param.gpxObject for name, param in self.sys_vars.items()}

        # update the module gpx variables with the new system variables
        self.gpx_variables.update(self.sys_gpx_vars)

        for name, pi in self.products.items():
            # generate the vars and models for each interaction from the products
            pi.generate_gpx(settings)

            # modify the keys of the varibles to include the product name?
            vs = rename_product_dict(name, pi.gpx_variables)

            # add gpx variables to the module
            self.gpx_variables.update(vs)

    def gpx_translate(self, settings: Settings, **kwargs: AcceptedTypes) -> None:
        "translate and put objects into gpxObjects"

        # create the classes based on the products
        self.mcclasses: dict[str, MClass] = {}
        classes: dict[str, MClass] = {}

        # collect any auxiliary constraints
        aux_constraints = []

        # collect feeder lines and secondary processes
        feeders_by_product: dict[str, ConstraintSet] = {}  # TODO: No longer accessed?
        secondary_by_product: dict[str, ConstraintSet] = {}

        feeder_owners = {}
        secondary_owners = {}
        for i, prod in enumerate(self.products.values()):
            pname = list(self.products.keys())[i]
            mfg = prod.active_modules["manufacturing"]

            # for sname in mfg.gpxObject.get("secondaryCells", {}):
            #     if sname in secondary_owners and secondary_owners[sname] != pname:
            #         raise ValueError(
            #             f'Secondary cell "{sname}" is shared between '
            #             f'{secondary_owners[sname]} and {pname}. '
            #             'Secondary cells cannot be shared across products.'
            #         )
            #     secondary_owners[sname] = pname

            # prate = self.prod_inputs[i]['manufacturing']['rate']

            # collect needed constraints for the feeder cell rates
            # feeders_by_product[pname] = ConstraintSet([])

            # collect extra constraints for the secondary processes
            secondary_by_product[pname] = ConstraintSet([])

            # pull out the feeder lines as a list
            feederlines: list = list(mfg.gpxObject['feederLines'].values())

            # add the class to the dictionary
            if self.by_split:
                mclass = MClass(
                    line=mfg.gpxObject["fabLine"],
                    feeder_lines=feederlines,
                    by_split=True,
                    class_split=self.prod_rates[pname],
                    return_processes=False,
                )

            else:
                mclass = MClass(
                    line=mfg.gpxObject["fabLine"],
                    feeder_lines=feederlines,
                    class_rate=self.prod_rates[pname],
                    return_processes=False,
                )

            # set the class
            classes[pname] = mclass
            self.mcclasses[pname] = mclass

            if not self.by_split:
                base_rate = self.prod_rates.get(pname)
                try:
                    lam_units = getattr(mclass.lam, "units", gpkit.units("count/hr"))
                except Exception:
                    lam_units = gpkit.units("count/hr")

                # Keep each lambda strictly positive without re-imposing the full
                # baseline rate. Scale the bound down aggressively so discrete
                # rounding is free to reduce throughput when capacity tightens.
                rate_floor = None
                try:
                    base_quantity = None
                    if base_rate is not None:
                        if hasattr(base_rate, "to"):
                            base_quantity = base_rate.to(lam_units)
                        else:
                            base_quantity = base_rate * lam_units

                    if base_quantity is not None:
                        try:
                            base_mag = float(getattr(base_quantity, "magnitude", base_quantity))
                        except Exception:
                            base_mag = None
                    else:
                        base_mag = None

                    min_floor_mag = 1e-9
                    scaled_floor_mag = (base_mag * 1e-6) if (base_mag and base_mag > 0) else None
                    floor_mag = max(filter(lambda v: v is not None, [scaled_floor_mag, min_floor_mag]))
                    rate_floor = floor_mag * lam_units
                except Exception:
                    logging.debug("Failed to derive relaxed rate floor for %s", pname, exc_info=True)

                if rate_floor is not None:
                    aux_constraints.append(mclass.lam >= rate_floor)

            # update the rates for the secondary cells
            for c in mfg.gpxObject.get("secondaryCells", {}).values():
                aux_constraints.extend([
                    c.lam >= mclass.lam,  # tie rate to its parent class
                    c.c2a >= 1,  # arrival-SCV at least Poisson
                    c.c2d <= 2 * c.c2s,  # monomial upper bound
                    c.Wq <= mclass.W,  # queue time sits inside class flow-time
                ])

        # create the cellmap a list of lists
        # cells in the same list are assumed to be shared
        cellmap: dict[str, list[object]] = {}
        sharedcell_names: list[str] = []

        for cname, prodlist in self.shared_cells.items():
            shared = []
            for pname in prodlist:
                mfg = self.products[pname].active_modules["manufacturing"]
                cell = mfg._gpx_all_cells[cname]
                cell.product_name = pname
                shared.append(cell)

            cellmap[cname] = shared
            sharedcell_names.append(cname)

        # find the related variables
        sharedcell_vars = {}  # dict by shared resources of related variables
        for cname in cellmap.keys():
            # return dict of gpx variables by keyed by cell by type
            varsforcell = self.find_variables(
                filter_category="cells",
                filter_type=cname,
                vars_attr="sys_vars",
                return_data="dict",
            )

            # find the gpx variable with the keys
            vardict = {v.property: self.sys_gpx_vars[v.key] for v in varsforcell.values()}

            # add to dict under the cell name
            sharedcell_vars[cname] = vardict

        users = {cname: {cell.product_name for cell in cells} for cname, cells in cellmap.items()}

        self.feeder_classes: dict[str, list[MClassFeeder]] = {}
        for pname, mc in self.mcclasses.items():
            for fl in mc.feeder_lines:
                fc = MClassFeeder(feeder_line=fl, parent_class=mc)
                self.feeder_classes.setdefault(pname, []).append(fc)

        # Nice-to-have: keep a flattened copy on the module too
        self.feeders = [fc for wrappers in self.feeder_classes.values() for fc in wrappers]

        system = self.gpxObject["system"] = MCSystem(
            classes=list(classes.values()),
            feeders=self.feeders,
            cellmap=cellmap,
            cellvars=sharedcell_vars,
            by_split=self.by_split,
        )

        # create a dict of the shared cells by their name
        # self.shared_cells_gpx_dict = dict(zip(self.shared_cells.keys(), system.mcells))
        self.shared_cells_gpx_dict = system.mcells

        # relate any shared tooling
        tooling_constraints = []  # list of equivalence constraints  # TODO: No longer accessed?
        tool_dict: dict[str, list[gpx.manufacturing.ConwipTooling]] = {}  # list of the tools by name
        mctools: dict[str, MCTool] = {}

        tool_costs = {}  # dict of tool costs

        # loop through the products and find the tools
        for pname, prod in self.products.items():
            # get the tools for the product
            mfgm = prod.active_modules['manufacturing']
            td = mfgm.gpxObject['tools']

            # pull each and add to tool dict
            for tname, t in td.items():
                tool_dict.setdefault(tname, []).append(t)
                # add the costs
                tool_costs.setdefault(
                    tname,
                    mfgm.find_gpx_variable(
                        filter_category='tools',
                        filter_property='Non-Recurring Cost',
                        filter_type=tname,
                        emptyisnone=True,
                    )
                )

        # create the multiproduct tools
        mctools = {tname: MCTool(tools=tlist) for tname, tlist in tool_dict.items()}

        # add the tooling constraints to the gpx object
        self.gpxObject['tools'] = mctools

        # CALCULATE COSTS

        # Shared Cell costs
        # TODO:  use the variables from the system rather than the first product
        shared_cell_costs = {}
        for sharename, prodnames in self.shared_cells.items():
            # get the variable for cost. always pick first instance of the variable
            module = cast(Manufacturing, self.products[prodnames[0]].active_modules["manufacturing"])
            # get the MCell by its name
            mcell = self.shared_cells_gpx_dict[sharename]

            shared_cell_costs[sharename] = CellCost(
                mcell,
                workstation_cost=associate_cost(
                    module,
                    filter_type=sharename,
                    filter_category='cells',
                    default_currency=settings.default_currency_iso
                ),
                default_currency=self.settings.default_currency_iso,
            )

        self.gpxObject["sharedCellCosts"] = shared_cell_costs  # add the gpx object

        floorspace_costs = set_floorspace_costs(self)

        # Tooling Costs
        # Find the total tool costs baased on the lists of the tool and MCTool objects
        toolcosts = {
            tname:
            ConwipToolingCost(
                conwip_tool=mct,
                tool_cost=tool_costs[tname],
                default_currency=settings.default_currency_iso,
            ) for tname, mct in mctools.items()
        }

        self.gpxObject["toolCosts"] = toolcosts

        # Inventory Holding
        invholding = {}
        self.holdingRate = Variable("holding rate", "", "annual inventory holding rate")
        for pname, prod in self.products.items():
            invholding[pname] = InventoryHolding(
                inv_count=classes[pname].line.L,
                flow_time=classes[pname].W,
                holding_rate=self.holdingRate,
                default_currency=self.settings.default_currency_iso
            )

        self.gpxObject["invHolding"] = invholding

        # Product Varible Costs
        varcosts = {}
        for pname, prod in self.products.items():
            mfg_obj = prod.active_modules["manufacturing"].gpxObject
            pc = ProductVariableCost(
                *list(mfg_obj["varCosts"].values()), default_currency=self.settings.default_currency_iso
            )
            # if empty, do not add
            varcosts[pname] = pc

        self.gpxObject["prodCosts"] = varcosts

        # Labor Costs
        labcosts = {}
        for pname, prod in self.products.items():
            mfg_obj = prod.active_modules["manufacturing"].gpxObject
            laborCosts: list[GPXLaborCost] = cast(list[GPXLaborCost], mfg_obj.get("laborCosts", []))
            if "laborCosts" in mfg_obj and len(laborCosts) > 0:
                # skip adding labor costs if empty list
                if isinstance(mfg_obj["laborCosts"], list):
                    labcosts[pname] = ProductVariableCost(
                        mfg_obj['laborCosts'][0],
                        return_costs=True,
                    )

        self.gpxObject["laborCosts"] = labcosts

        # rate ramps
        rampcosts = []

        if hasattr(self, "rateramp"):
            cost_objs_to_add = [
                shared_cell_costs,
                toolcosts,
                invholding,
                varcosts,
            ]

            costs_to_add = []
            for cobj in cost_objs_to_add:
                costs_to_add.extend(list(cobj.values()))

            # prod_costs_to_add = [    # costs in the product which
            #     'toolCosts',
            #     'cellCosts',
            #     'floorspacecost',
            #     'varCosts',
            #     'recurringCost',
            # ]
            # #TODO:  find the costs from each of the products

            # for prod in self.products.values():
            #     mfg_obj = prod.active_modules['manufacturing'].gpxObject
            #     for cname in prod_costs_to_add:
            #         c = mfg_obj.get(cname)
            #         if hasattr(c, 'values'):
            #             #this is a dict. convert to list
            #             c = list(c.values())

            #         if c : costs_to_add.extend(c)
            #     inv_costs.append(mfg_obj['invHolding'])

            #     lab_costs.extend(mfg_obj['laborCosts'])

            self.rateramp.add_costs(*costs_to_add)
            self.rateramp.add_labor_costs(*list(labcosts.values()))

            rampcosts = [self.rateramp.get_aux_cost()]

        # Total Cost
        totalcost_inputs = [
            *shared_cell_costs.values(),
            *toolcosts.values(),
            *invholding.values(),
            *varcosts.values(),
            *labcosts.values(),
            *rampcosts,
        ]
        if floorspace_costs is not None:
            totalcost_inputs.append(floorspace_costs)

        totalcost = TotalCost(
            *totalcost_inputs,
            return_costs=False,
            calc_units=False,
            default_currency=self.settings.default_currency_iso,
        )

        self.gpxObject["laborCosts"] = labcosts
        self.gpxObject["totalCost"] = totalcost
        self.gpxObject['auxConstraints'] = aux_constraints
        if len(rampcosts) > 0:
            self.gpxObject["rampCost"] = rampcosts

    def gpx_constraints(self, **kwargs: AcceptedTypes) -> ConstraintSet:
        "get the gpx constraints"
        constr = []

        # define which objects from the manufacturing modle should be pulled up to
        # the system
        mfg_include_objects = [
            "processes",
            # "tools",
            "varCosts",
            "serialProcesses",
        ]

        # pull the constraints from the product interacives objects
        for pname, p in self.products.items():
            # get all design constraints
            for modname, m in p.active_modules.items():
                # go through the modules in each product interactive
                # get constraints from objects
                if modname == "manufacturing":
                    # get the dynamic constraints defined in the module
                    constr.extend(m.gpx_constraints(dynamic_only=True, **kwargs))

                    # for the manufacturing we don't want any line or cell constraints
                    for cname, go in m.gpxObject.items():
                        if cname in mfg_include_objects:
                            # skip adding constriaints if ignoring
                            if hasattr(go, "values"):
                                # it's a dict
                                constr.extend(list(go.values()))
                            else:
                                if isinstance(go, list):
                                    constr.extend(go)
                                else:
                                    constr.append(go)
                else:
                    # not a production module, include all constraints
                    if hasattr(m, "gpx_constraints"):
                        constr.extend(m.gpx_constraints(**kwargs))

        # build constriaints in `gpx_translate`
        for obj in self.gpxObject.values():
            if hasattr(obj, "values"):
                # it's a dict
                constr.extend(obj.values())
            else:
                if isinstance(obj, list):
                    constr.extend(obj)
                else:
                    constr.append(obj)

        return constr

    def get_acyclic_constraints(self, **kwargs):
        'get the acyclic constraints from each of the product interactions'
        aconstr = []
        for pname, p in self.products.items():
            aconstr.extend(p.acyclic_constraints)

        aconstr = add_x_splits_to_acyclic_constraints(self, aconstr)
        aconstr = add_rate_links_to_acyclic_constraints(self, aconstr)

        return aconstr

    def get_redundant_substitutions(self) -> list[list[VarKey]]:
        "finds the redundant constriaints, substitutions in a model and returns a list to delete"
        red_varkeys: list[list[VarKey]] = []  # list of the redundant varkeys

        # loop over the shared cells
        for rname, shared in self.shared_cells.items():
            # list for the shared variables
            shared_list = self._vars_from_shared(
                resource_name=rname,
                shared=shared,
                filter_category="cells",
                filter_property="Workstation Count",
            )

            # append the shared variable names to the shared list
            if len(shared_list) > 0:
                # only append if there is something in the list
                red_varkeys.append(shared_list)

        # NOTE:  if we try and remove redundant tooling constraints the solve fails with non-shared tooling
        # loop over the shared tooling
        # for rname, shared in self.shared_tools.items():
        #     tool_list = self._vars_from_shared(
        #         resource_name=rname,
        #         shared=shared,
        #         filter_category='tools',
        #         filter_property='Count'
        #     )
        #     # only add to the list of constraints if the tool is shared
        #     if len(tool_list) > 1:
        #         red_varkeys.append(tool_list)

        # return the list of the redundant varkeys
        return red_varkeys

    def _vars_from_shared(
        self,
        resource_name: str,
        shared: list[str],
        filter_category: str,
        filter_property: str,
    ) -> list[VarKey]:
        "gets the list of variables from all products from a shared resource"
        shared_list: list[VarKey] = []
        # loop over the products in each of the shared cells
        for pname in shared:
            p = self.products[pname]
            mfgm = p.active_modules["manufacturing"]
            vs = mfgm.find_gpx_variable(
                filter_category=filter_category,
                filter_property=filter_property,
                filter_type=resource_name,
                idx=None,  # return the whole list
            )
            # extend the shared list by the varkeys
            shared_list.extend([v.key for v in vs])

        return shared_list

    def get_total_ratevar(self, product_name=None):
        '''get the total production rate of the system'''
        return self.gpxObject['system'].lam

    def get_results(
        self,
        sol: "gpx.Model",
        settings: Settings,
        suppressres_override: bool = False,
        **kwargs: dict[str, AcceptedTypes]
    ) -> list[ResultGenerator]:
        """get the results from the module

        Optional Arguments
        ------------------
        finmod : modules.ProductionFinance
            pass the finance module to get the aggregate rates

        """
        results: list[ResultGenerator] = []
        # SYSTEM RESULTS

        self.product_labor = {}

        # cells
        mcells: dict[str, MCell] = self.gpxObject["system"].mcells
        names = list(self.shared_cells.keys())

        results.append(MCellResults(gpxsol=sol, settings=settings, mcells=mcells, cellnames=names))

        # tools

        # product results
        presults: list[ResultGenerator] = []
        namefield = "productName"

        # product summaries
        psums: list[ProductSummary] = []

        for pname, p in self.products.items():
            # contains product-specific result gens
            ppresults: list[ResultGenerator] = []

            # get the manufacturing module
            mfg = p.active_modules["manufacturing"]

            # get the ordered list of the cells
            cells_in_order = mfg.get_cells_in_order()
            # transform into a dict
            cells_in_order = {cname: order for order, cname in cells_in_order}

            # add the results from the processes
            ppresults.append(
                ProcessResults(
                    gpxsol=sol,
                    settings=settings,
                    processchain=mfg.processChain,
                    processes=cast(dict[str, gpx.manufacturing.Process], mfg.gpxObject["processes"]),
                ),
            )

            # add the line results
            lineresults = LineResults(gpxsol=sol, settings=settings, fabline=mfg.gpxObject["fabLine"])
            del lineresults.results["probabilities"]  # ignore probabilities

            try:
                lineresults.add_agg_rate(finmod=cast(ProductionFinance, self.finmod))
            except KeyError:
                pass

            ppresults.append(lineresults)

            # add the labor costs
            ppresults.append(
                LaborCost(
                    gpxsol=sol,
                    labcosts=cast(GPXLaborCost, mfg.gpxObject["laborCosts"]),
                    cellorderdict=cells_in_order,
                    include_headcount=True,
                    settings=settings
                ),
            )
            self.product_labor[pname] = ppresults[-1]

            # get cell times
            ppresults.append(CellTimes(cells=mfg.get_all_cells(), gpxsol=sol, settings=settings))

            # cell results
            # cellresults = result_gens.CellResults(
            #     gpxsol=sol,
            #     cells=mfg.gpxObject['cells'],
            #     cells_in_order=mfg.get_cells_in_order()
            # )
            # ppresults.append(cellresults) # not actually including the cell results to return

            # variable cost results
            ppresults.append(
                VarCostResults(
                    gpxsol=sol,
                    varcosts=cast(ProductVariableCost, mfg.gpxObject["varCosts"]),
                    labcosts=cast(LaborCost, mfg.gpxObject["laborCosts"]),
                    settings=settings
                ),
            )

            # recurring costs
            ppresults.append(
                RecurrCostResults(
                    gpxsol=sol,
                    recur_costs=cast(gpx.recurring_cost.RecurringCosts, mfg.gpxObject["recurringCosts"]),
                    cell_costs=cast(dict[str, gpx.non_recurring_costs.CellCost], mfg.gpxObject["cellCosts"]),
                    settings=settings
                ),
            )

            # TODO:  get some of the cell results
            #       be careful, not all of the usual variables are present
            #       in the result generator, consider using a `get` method since it is safer
            #       if using `dict.get()` what should it return on fail?

            # # get cell results
            # ppresults.append(
            #     result_gens.CellResults(
            #         gpxsol=sol,
            #         cells=mfg.get_all_cells(),
            #         cells_in_order=mfg.get_cells_in_order(),
            #     )
            # )

            # get the summary results from the product
            # is this doppel-gemoppelt with creating the object and adding it to the
            # list
            ps = ProductSummary(gpxsol=sol, settings=settings, resultslist=ppresults)
            psums.append(ps)
            ppresults.append(ps)

            # put the product name into all the fields
            for ppres in ppresults:
                # add the product field to the result
                for resdict in ppres.results.values():
                    for res in resdict:
                        # add the product name to a special identifier field
                        if isinstance(res, dict):
                            res[namefield] = pname
                            # add the kitSplit data as well if bysplit
                            res['kitSplit'] = p.kitSplit if self.by_split and self.from_kit else 1

                # add the product name to the collected variables
                new_collect_vars = {}
                for vname, v in ppres.collect_vars.items():
                    new_vname = f"{pname} :: {vname}"
                    new_collect_vars[new_vname] = v

                # replace the variable collector with the new names
                ppres.collect_vars = new_collect_vars

                # update aux var names
                if hasattr(ppres, "aux_vars"):
                    for var in ppres.aux_vars:
                        var["name"] = f"{pname} | {var['name']}"

                # remove headcount per product

            presults.extend(ppresults)

        # DEBUG: try to append the objects without first combining
        # results.extend(presults)

        # combine the different results into one generator object
        combined_result_generator: ResultGenerator = combine_results(sol, presults, settings=settings)
        # remove doubled total headcount from aux
        maxhc = 0
        newauxvars = []
        for av in combined_result_generator.aux_vars:
            if "Line Total Headcount" in av["name"]:
                # find the max headcount
                maxhc = max(maxhc, av["value"])
            else:
                # otherwise, keep the aux var as is
                newauxvars.append(av)
        # add the new headcount figure
        newauxvars.append({
            "category": [],
            "name": "Line Total Headcount",
            "sensitivity": 0.0,
            "unit": "count",
            "value": maxhc,
        })
        # swap the new list for the old aux vars
        combined_result_generator.aux_vars = newauxvars

        # strip product name from recurring costs
        # for rc in combined_result.results['recurringCosts']:
        #     del rc[namefield]

        # combine recurring costs with the same name
        new_costs: list[gpx.recurring_cost.Cost] = []
        recurring_costs = combined_result_generator.results.get("recurringCosts", [])
        if isinstance(recurring_costs, list):
            sorted_costs = sorted(
                (c for c in recurring_costs if isinstance(c, dict) and isinstance(c.get("name"), str)),
                key=lambda x: str(x["name"])
            )

            for costname, group in groupby(sorted_costs, lambda x: x["name"]):
                if not isinstance(costname, str):
                    raise ValueError(
                        f"Invalid item: expected 'costname' to be a str, got {type(costname).__name__} instead."
                    )

                total_value = 0.0
                for c in group:
                    value = c.get("value")
                    if isinstance(value, (int, float)):
                        total_value += value

                new_costs.append({
                    "name": costname,
                    "unit": f"{settings.default_currency_iso}/hr",
                    "value": float(total_value)
                })

        # add the reformatted costs to the results
        combined_result_generator.results["recurringCosts"] = new_costs

        # append the combined result to all the results
        results.append(combined_result_generator)

        if len(self.gpxObject['tools']) > 0:
            # Tool Results
            results.append(
                MCToolResults(
                    gpxsol=sol,
                    tools=self.gpxObject['tools'],
                    tool_costs=self.gpxObject['toolCosts'],
                    settings=settings,
                )
            )

        # classes
        results.append(MCClassResults(gpxsol=sol, settings=settings, classes=self.mcclasses, by_split=self.by_split))

        # Get the system capital cost results
        results.append(MCNonRecCost(
            gpxsol=sol,
            module=self,
            settings=settings,
        ))

        # Get system recurring cost results
        results.append(
            MCRecCost(
                gpxsol=sol,
                module=self,
                extant_costs=combined_result_generator.results.pop('recurringCosts', []),
                settings=settings,
            )
        )

        # Get the system capital cost results
        # TODO:  where is the system cost calculated?
        # TODO:  get the cost components for the waterfall
        # results.append(result_gens.NRCostResults)

        # system total costs result generator
        results.append(
            MCCostResults(
                gpxsol=sol,
                settings=settings,
                prod_costs=self.gpxObject["prodCosts"],
                unitcost=self.gpxObject["totalCost"],
            )
        )

        # get the summary results and flatten by the product
        results.append(MCProductSummary(gpxsol=sol, settings=settings, prod_sums=psums))
        # loop over the products
        # DEBUG: this isn't working because not all of the results are available in the products
        #       we don't a lot of the QNA variables, for example
        # for pname, p in self.products.items():
        #     # get the manufacturing module from the product
        #     pmfg = p.active_modules['manufacturing']
        #     # get the results from the manufacturing module
        #     for rg in pmfg.get_results(sol):
        #         # check to see if there are summary rg
        #         if isinstance(rg, result_gens.ProductSummary):
        #             # for every entry in the summary add the product name
        #             # loop over the list of dicts
        #             sum_results.extend([
        #                 {'{} | {}'.format(pname, resname) : res}
        #                 for sumr in rg.results[rg.resname]
        #                 for resname, res in sumr.items()
        #             ])

        # TODO:  get any system summary results

        return results

    def get_production_resources(
        self,
        filter_variables: bool = True,
        return_set: bool = False,
    ) -> list[gpx.Variable] | set[gpx.Variable]:
        """get a list of the production resource count varibales

        This is particularly useful for the discrete resource solves

        Returns
        -------
        list[gpx.Variable]
            _description_
        """
        if not self.gpxObject:
            raise ValueError("manufacturing module must have been gpx translated to get production resource variables")

        # TODO: If feeder not in mcell use QNACell.m

        system = self.gpxObject['system']

        def _add_cell_m(target, cell):
            """Append a cell's workstation variable if available."""
            var = getattr(cell, "m", None)
            if var is not None:
                target.add(var)

        resources: set[gpx.Variable] = set()

        shared_cell_ids: set[int] = set()
        for mcell in system.mcells.values():
            for cell in getattr(mcell, "cells", []):
                shared_cell_ids.add(id(cell))

        for mcell in system.mcells.values():
            _add_cell_m(resources, mcell)

        seen_cells: set[int] = set()

        def _maybe_add_cell(cell) -> None:
            cid = id(cell)
            if cid in shared_cell_ids or cid in seen_cells:
                return
            seen_cells.add(cid)
            _add_cell_m(resources, cell)

        for cls in getattr(system, "classes", []):
            for cell in getattr(cls.line, "cells", []):
                _maybe_add_cell(cell)
            for feeder_cls in getattr(cls, "feeder_lines", []):
                for cell in getattr(feeder_cls, "cells", []):
                    _maybe_add_cell(cell)

        for feeder in getattr(system, "feeders", []):
            for cell in getattr(feeder, "cells", []):
                _maybe_add_cell(cell)

        for cell in getattr(system, "single_cells", []):
            _maybe_add_cell(cell)

        rescs = list(resources)

        for mctool in self.gpxObject['tools'].values():
            rescs.append(mctool.L)

        return set(rescs) if return_set else rescs


def rename_product_dict(
    prodname: str, inputdict: dict[str, tuple[AcceptedTypes, AcceptedTypes]]
) -> dict[str, tuple[AcceptedTypes, AcceptedTypes]]:
    "rename the keys to add products in a dictionary"
    return {f"{prodname} :: {oldkey}": val for oldkey, val in inputdict.items()}


class MultiproductContext(SolutionContext):
    "the multi-product context"

    # `context.get_substitutions`

    def get_cost(self) -> Variable:
        "get the cost as the average unit cost"

        # mcsystem = self.interaction.active_modules['system'].gpxObject

        # cost =  mcsystem['totalCost'].totalCost / mcsystem['totalCost'].numUnits

        # return cost
        return self._gpx_cost_var

    def get_substitutions(self, for_assembly: bool = False, **kwargs: AcceptedTypes) -> dict[Variable, float]:
        "get context substitutions"
        sysmod = self.interaction.active_modules["system"]
        finmod = self.interaction.active_modules["finance"]

        horizon = finmod.duration - finmod.rampduration

        # get the cost of capital from the financial module
        self._substitutions_.update({sysmod.holdingRate: finmod.costOfCapital})

        if finmod.amortization == "quantity":
            self._substitutions_[sysmod.gpxObject["totalCost"].numUnits] = finmod.quantity
        elif finmod.amortization == "duration":
            self._substitutions_[sysmod.gpxObject["totalCost"].horizon] = finmod.get_hourly_duration(duration=horizon)

        if hasattr(sysmod, "rateramp"):
            total_units = finmod.get_qty(finmod.get_hourly_rate(finmod.rate), duration=finmod.duration) - finmod.rampqty
            self._substitutions_[sysmod.gpxObject["totalCost"].numUnits] = total_units

        # pull up the subsitutions for eta from the cells
        for c in sysmod.gpxObject["system"].flat_cells:
            self.interaction.gpx_model.substitutions[c.nu] = 1.0
            pass

        # check for 0 rate input
        if finmod.rate == 0:
            raise ValueError('"Total Rate" for the system cannot be 0')

        # substitute the system production rate
        sys_obj = sysmod.gpxObject["system"]
        sys_rate = sys_obj.lam
        if sysmod.by_split:
            # if determined by split, use the entered system rate
            if sysmod.system_rate is not None:
                self._substitutions_[sys_rate] = sysmod.system_rate
            else:
                if finmod.amortization == "quantity":
                    # this combination is not allowed. horizon is unbounded
                    raise ValueError('A "Quantity" input requires a target rate')
                else:
                    # if system rate not specified, make sure not trying to substitute
                    self._substitutions_.pop(sys_rate, None)

        else:
            # find as sum of the class production rates
            # TODO:  sometimes not finding rate in the product class
            # def sum_rate(m: dict[VarKey, float]) -> gpkit.nomials.math.Posynomial:
            #     return np.sum([m[c.lam] for c in sys_obj.classes])

            # self._substitutions_[sys_rate] = sum_rate
            self._substitutions_.pop(sys_rate, None)

        # self.update_substitutions()

        return super().get_substitutions(for_assembly=False, **kwargs)

    def update_substitutions(self, duration: int = None, **kwargs: AcceptedTypes) -> None:
        "update the substitions"
        super().update_substitutions(**kwargs)

        # Remove the redundant substitutions to help cvxopt
        # make a list of the subs to delete
        sysm = self.interaction.active_modules.get("system", None)

        if not sysm:
            logging.error("MULTIPRODUCT UPDATE-SUBSTITUTIONS: could not find the ")
            raise ValueError("Could not create system model")
        # get the substitutions that describe the same resources
        subs_to_del = sysm.get_redundant_substitutions()  # this should be a list of lists of varkeys

        # get the old list of subsitutions
        newsubs = self.interaction.gpx_model.substitutions

        mcsys = sysm.gpxObject["system"]

        if not sysm.by_split:
            newsubs = set_x_rate_values(self, sysm, mcsys, newsubs)

        # newsubs = apply_feeder_batch_qty_rates(self, sysm, newsubs)
        newsubs = update_w_cap(sysm, newsubs, duration)

        fs_vars = sysm.find_variables(
            filter_category='cells',
            filter_property='Floor Space',
            return_data='dict',
            emptyisnone=True,
        ) or {}

        for param in fs_vars.values():
            vk = param.gpxObject
            newsubs[vk] = param.value * gpkit.units(param.unit or "")
            self._substitutions_[vk] = param.value * gpkit.units(param.unit or "")

        # find the actually redundant substitutions and delete
        if len(subs_to_del) > 0:
            # only look for redundant constraints if there are any exposed variables
            redundant_subs = []
            for sublist in subs_to_del:
                rsubs = [s for s in sublist if s in newsubs]
                if len(rsubs) > 1:
                    # if there are redundant inputs, pop the first and append the rest
                    rsubs.pop(0)
                    redundant_subs.extend(rsubs)

            # pop the subs to delete from the old list
            for sub in redundant_subs:
                if not newsubs.pop(sub, None):
                    # could not find the substituion in the model
                    logger.debug(f"Could not find substitution {sub.key.__str__()} to delete")

        self.interaction.gpx_model.substitutions = newsubs

    def make_gpx_constraints(self, **kwargs: AcceptedTypes) -> None:
        "additional, context constraints"

        constr = []

        sysm = self.interaction.active_modules['system']
        # create cost constraints
        mcsystem = sysm.gpxObject

        # get the cost function

        self._gpx_cost_var = Variable("C_{avg unit}", self.settings.default_currency_iso, "Average Unit Cost")

        if sysm.from_kit:
            kitqty = sysm.kit_quantity if sysm.kit_quantity else 1
            constr.append(
                self._gpx_cost_var >= mcsystem['totalCost'].totalCost / mcsystem['totalCost'].numUnits * kitqty
            )
        else:
            constr.append(self._gpx_cost_var >= mcsystem['totalCost'].totalCost / mcsystem['totalCost'].numUnits)

        sys_obj = sysm.gpxObject["system"]
        if sysm.by_split:
            # enforce that the system rate is truly the sum of the class rates
            constr.append(sys_obj.lam >= sum(c.lam for c in sys_obj.classes))

        # get the unit basis for the total cost

        sysmod = self.interaction.active_modules["system"]
        finmod = self.interaction.active_modules["finance"]
        # sum_units = []
        for pname, ihc in sysmod.gpxObject["invHolding"].items():
            # calculate number of units from the inputs
            constr.append(ihc.inventoryValue >= sysmod.gpxObject["totalCost"].get_basis(basis="unit"))

        # constrain total count of units
        if not hasattr(sysmod, "rateramp"):
            total_duration = sysmod.gpxObject["totalCost"].horizon  # use the same variable from the finance
            unit_constr = sysmod.gpxObject["totalCost"].numUnits == sysmod.gpxObject["system"].lam * total_duration
            constr.append(unit_constr)

        # for each of the product variable costs
        for name, pvc in sysmod.gpxObject["prodCosts"].items():
            if pvc.nonrecurringCost != 0:
                # only add the constriant on the number of units if there is a cost
                constr.append(pvc.numUnits >= finmod.get_qty(sysmod.mcclasses[name].line.lam))

        # also for labor costs
        for name, pvc in sysmod.gpxObject['laborCosts'].items():
            if pvc.nonrecurringCost != 0:
                # only add the constriant on the number of units if there is a cost
                constr.append(pvc.numUnits >= finmod.get_qty(sysmod.mcclasses[name].line.lam))

        if not sysm.by_split:
            mcsys = sysm.gpxObject["system"]
            constr.append(mcsys.lam >= sum(mc.lam for mc in mcsys.classes))

        self.gpx_constraints = constr

    def make_solutions(self, **kwargs: AcceptedTypes) -> dict[str, Variable]:
        """make the solutions
        Override the parent class to process product results

        """
        collect_vars = super().make_solutions(**kwargs)

        # TODO:  manually insert aggregates by variable
        # TODO:  in the future, have this pull from the modules
        #       could be similar to collect variables (but no calls to solution)

        # filter for the aggregated measures
        vars_to_add = filter(
            lambda x: x["name"] == "Aggregate Production Rate",
            self.interaction.solutions["lineSummary"],
        )

        # append to the all variables
        for v in vars_to_add:
            # add to the solutions
            self.interaction.solutions["allVariables"].append({
                "name": f"{v['productName']} | {v['name']}",
                "value": v["value"],
                "unit": v["unit"],
                "sensitivity": 0.0,
                "source": "Calculated Value",
            })

        # add an average unit cost

        collect_vars["Average Unit Cost"] = self._gpx_cost_var
        avg_cost = self.interaction.gpx_solution(self._gpx_cost_var)

        self.interaction.solutions["allVariables"].append({
            "name": "Average Unit Cost",
            "value": np.around(avg_cost.magnitude, decimals=3),
            "unit": str(avg_cost.units),
            "sensitivity": 0.0,
            "source": "Calculated Value",
        })

        self.interaction.collected_variables.update(collect_vars)

        return collect_vars

    def old_get_cost(self) -> float:
        "get cost as the sum of all inventory and cells"

        mcsystem = self.interaction.active_modules["system"].gpxObject["system"]

        mcells = mcsystem.mcells
        single_cells = mcsystem.single_cells
        classes = mcsystem.classes

        # tools
        toollist: list[Variable] = []
        for prod in self.interaction.active_modules["system"].products.values():
            mfg = prod.active_modules["manufacturing"].gpxObject
            toollist.extend([t.L for t in mfg["tools"].values()])

        cost = (
            np.sum([c.m for c in mcells]) + np.sum([c.m for c in single_cells])
            + 1e-3 * np.sum([c.line.L for c in classes]) + 0.5 * np.sum(toollist)
        )

        return float(cost)

    def solve(self, duration: int = None, max_rate: float | None = None, **kwargs: AcceptedTypes) -> gpx.Model:
        """solve the model and replace the sensitivities for the MCell rates"""

        # update the substitutions
        # NOTE:  updating substituions will remove any conflicting inputs on shared resources
        #       and default to the inputs from the first product
        self.update_substitutions(duration)

        if self.interaction.discretized_resources:
            discrete_solution = discretize_resources(
                self.interaction, self.get_disc_target_var(), self.get_disc_target_val(), **kwargs
            )
            self._apply_mcell_splits(discrete_solution)
            return discrete_solution

        try:
            sol = self.interaction.gpx_model.solve(**kwargs)
        except Exception as e:
            # TODO:  improve this error messaging in PR #205
            raise e

        mcsystem = self.interaction.active_modules["system"].gpxObject["system"]

        # replace the sensitivity for each product rate by the sum
        for c in mcsystem.classes:
            sumsens = np.sum([sol["sensitivities"]["variables"][lam.key] for lam in c.all_lams])
            sol["sensitivities"]["variables"][c.line.lam.key] = sumsens

        # TODO: also include influence of lambda in non-mcsystem cells

        self._apply_mcell_splits(sol)
        return sol

    def _apply_mcell_splits(self, sol: gpx.Model) -> None:
        """Ensure each MCell has explicit x-split values recorded in the solution."""

        def _to_float(value) -> float | None:
            if value is None:
                return None
            try:
                return float(value.magnitude)
            except AttributeError:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

        if not sol:
            return

        try:
            variables = sol["variables"]
        except Exception:
            return

        try:
            free_variables = sol["freevariables"]
        except Exception:
            free_variables = None

        try:
            constants = sol["constants"]
        except Exception:
            constants = None

        discrete_vars = None
        if hasattr(sol, "__getitem__"):
            try:
                discrete_vars = sol["discreteVariables"]
            except Exception:
                discrete_vars = None

        def _store_value(var_obj, value) -> None:
            key = getattr(var_obj, "key", var_obj)

            for container in (variables, free_variables):
                if container is None:
                    continue
                try:
                    container[key] = value
                except Exception:
                    pass
                if var_obj is not key:
                    try:
                        container[var_obj] = value
                    except Exception:
                        pass

            if constants is not None:
                for candidate in (key, var_obj):
                    try:
                        if candidate in constants:  # type: ignore[arg-type]
                            del constants[candidate]
                    except Exception:
                        try:
                            constants.pop(candidate, None)  # type: ignore[attr-defined]
                        except Exception:
                            pass

        system = self.interaction.active_modules["system"].gpxObject.get("system")
        if not system:
            return

        for mcell_name, mcell in getattr(system, "mcells", {}).items():
            try:
                total_rate = sol(mcell.lam_bar)
            except Exception:
                total_rate = None

            total_mag = _to_float(
                getattr(total_rate, "to", lambda _: total_rate)(mcell.lam_bar.units) if total_rate else None
            )
            if total_mag is None:
                total_mag = 0.0

            if total_mag <= 0.0:
                base_value = 1.0 / max(len(mcell.cells), 1)
                normalized = [base_value for _ in mcell.cells]
            else:
                raw_values: list[float] = []
                for cell in mcell.cells:
                    try:
                        cell_rate = sol(cell.lam)
                        cell_mag = _to_float(cell_rate.to(mcell.lam_bar.units))
                    except Exception:
                        try:
                            cell_mag = _to_float(sol["variables"][cell.lam.key])
                        except Exception:
                            cell_mag = None
                    raw_values.append(max((cell_mag or 0.0) / max(total_mag, 1e-12), 0.0))

                total_raw = sum(raw_values)
                if total_raw > 0.0:
                    normalized = [val / total_raw for val in raw_values]
                else:
                    base_value = 1.0 / max(len(mcell.cells), 1)
                    normalized = [base_value for _ in mcell.cells]

            rounded_values: list[float] = []
            for idx, value in enumerate(normalized):
                if len(mcell.cells) == 1:
                    rounded_values.append(1.0)
                else:
                    rounded_values.append(round(value, OTHER_ROUND_DEC))

            if len(mcell.cells) > 1:
                total_rounded = sum(rounded_values)
                correction = round(1.0 - total_rounded, OTHER_ROUND_DEC)
                if correction:
                    adjusted = round(rounded_values[-1] + correction, OTHER_ROUND_DEC)
                    rounded_values[-1] = min(max(adjusted, 0.0), 1.0)

            for idx, cell in enumerate(mcell.cells):
                value = rounded_values[idx]
                _store_value(cell.x, value)

                product = getattr(cell, "product_name", f"Cell{idx}")
                alias = VarKey(f"{product} :: {mcell_name} x_{idx}")
                _store_value(alias, value)

                if getattr(mcell, "xitem", None) and idx < len(mcell.xitem):
                    xi = mcell.xitem[idx]
                    _store_value(xi, value)
                    alias_item = VarKey(f"{product} :: {mcell_name} xItem_{idx}")
                    _store_value(alias_item, value)
                if getattr(mcell, "xinv", None) and idx < len(mcell.xinv):
                    xv = mcell.xinv[idx]
                    _store_value(xv, value)
                    alias_inv = VarKey(f"{product} :: {mcell_name} xInv_{idx}")
                    _store_value(alias_inv, value)

        if isinstance(discrete_vars, dict):
            for var, raw in list(discrete_vars.items()):
                scalar = _to_float(raw)
                if scalar is None:
                    continue
                count = max(int(math.floor(scalar + 0.5)), 1)
                if getattr(var, "units", None):
                    quantity = count * var.units
                else:
                    quantity = float(count)

                _store_value(var, quantity)
                discrete_vars[var] = quantity

    def get_disc_target_var(self) -> Variable:
        "gets the target variable for a discrete solve"
        # get the rate from the manufacturing module
        sysm = self.interaction.active_modules["system"]
        attr = sysm.gpxObject["system"]
        return getattr(attr, "lam")

    def get_disc_target_val(self) -> float:
        "gets the target value for the discrete"
        ratevar = self.get_disc_target_var()
        rateval = self.interaction.gpx_model.substitutions.get(ratevar)
        return rateval


def multi_product_risk_results(mpi: "InteractiveModel") -> None:
    "add product labels to the risk results"
    try:
        risksens = mpi.solutions["riskSens"]
    except KeyError:
        # if there is no risk sensitivities,
        # we can also assume there is no uncertainty
        return

    if mpi.has_uncertainty:  # TODO: Function "has_uncertainty" could always be true in boolean context
        # otherwise, positive affirmation that there is uncertainty
        for i, rs in enumerate(risksens):
            # go through each variables and add 'product' field
            namesplit = rs["name"].split(" :: ")
            if len(namesplit) == 1:
                # assume it is a shared resource
                risksens[i]["product"] = "System Input"
            elif len(namesplit) == 2:
                # there is a product specified as part of the variable name
                risksens[i]["product"] = namesplit[0]
                risksens[i]["name"] = namesplit[1]  # change the name to exclude product
            # replace the colon seperator with a pipe
            # risksens[i]['name'] = ' | '.join(namesplit)


def get_active_mfg_name(prod_input: dict[str, AcceptedTypes]) -> str | None:
    manufacturing = prod_input.get("manufacturing", {})
    active_mfg_mod_id = manufacturing.get("selectedModule") if isinstance(manufacturing, dict) else None
    module = get_input_module_by_id(prod_input, active_mfg_mod_id) if active_mfg_mod_id is not None else None
    if not module:
        for m in prod_input.get("modules", []):
            if isinstance(m, dict) and m.get("type") == "manufacturing":
                module = m
                break
    return str(module["name"]) if module else None


def get_active_design_name(prod_input: dict[str, AcceptedTypes]) -> str | None:
    "get the name of the active design module for the product"
    # check to see if there is a design module
    if "design" in prod_input:
        design = prod_input.get("design", {})
        if isinstance(design, dict):
            prod_id = design.get("selectedModule")
        else:
            prod_id = None
        module = get_input_module_by_id(prod_input, prod_id if prod_id else "")
        return str(module["name"]) if module else None
    else:
        return None


def get_input_module_by_id(prod_input: dict[str, AcceptedTypes], id) -> dict[str, AcceptedTypes] | None:
    """gets the module by id from a product input"""
    mods = prod_input.get("modules", [])
    if not isinstance(mods, list):
        mods = []

    sid = str(id)
    for x in mods:
        if not isinstance(x, dict):
            continue
        xid = x.get("id")

        if str(xid) == sid:
            return x

        try:
            if float(xid) == float(id):
                return x
        except (TypeError, ValueError):
            pass
    return None


class MultiproductVariants(Multiproduct):
    'handles variants as multiproduct'

    def __init__(self, **kwargs) -> None:
        self.variants = OD()
        super().__init__(**kwargs)

        # TODO:  update the design inputs to match the variants

        pass

    def _dicttoobject(self, inputdict: dict):
        # call the parent function but do not use the super function
        ModuleType._dicttoobject(self, inputdict)

        # share all the resources
        pass

        # TODO:  overload

        self.variants = OD()

        # set the mix type
        self.mixType = 'concurrent'
        # rate should always be a split
        self.by_split = True

        # store the variants as input (in order)
        for v in inputdict.get('variants', []):
            self.variants[v['name']] = v

        # make a list of all the names of the variants
        self.all_var_list = all_var_list = list(self.variants.keys())

        # values in the inputdict object to skip
        objstoskip = ['variants']
        # make all the products with the base manufacturing and design
        for vname in all_var_list:
            # add all the objects except those to skip
            prod_inputdict = {oname: o for oname, o in inputdict.items() if oname not in objstoskip}
            # change the modelname to the name of the variant
            prod_inputdict['modelName'] = vname
            # add the product name as the name of the manufacturing module
            for m in prod_inputdict['modules']:
                if m['type'] == 'manufacturing':
                    m['name'] = vname
            # add to the instance variable
            self.prod_inputs.append(prod_inputdict)

            # look to see what the format of p['source'] looks like

        # define the production rate splits
        self.prod_rates = {vname: v['split'] for vname, v in self.variants.items()}

        # Add shared resourrces
        # shared cells (assume all are shared)
        # for c in inputdict['']

        # for each of the products, update the design inputs

        #TODO:  find shared feeder cells

        return None

    def gpx_translate(self, **kwargs):

        # run the translation for the multiclass system
        return super().gpx_translate(**kwargs)

    pass

    # create a design module for each variant
    # duplicate the manufacturing module for each product
    # in the collected variables, add up all the sensitivities for manufacturing module inputs


def _sys_dict_from_variant_dict(
    inputdict,
    rename_prods=False,
    target_parameterization=None,
    copy_all_mfg=True,
) -> dict:
    'converts an input dictionary for a variant into a dict for a system'
    # create a new system dict based on the variants
    new_inputs: dict = {}  # a new dict to hold the construction as multi-product

    # create a dict of variants
    variants = {v['name']: v for v in inputdict['variants']}

    # rename the variants if needed
    if rename_prods:
        variants = {f'{inputdict["modelName"]} ({k})': v for k, v in variants.items()}

    # get the input modules
    inputmods = inputdict['modules']

    # get the baseline design module
    seldesignmod = inputdict['design']['selectedModule']
    # find the index of the design module in the modules array
    design_mod_idx = [i for i, m in enumerate(inputmods) if m['id'] == seldesignmod]
    design_mod_idx = design_mod_idx[0]
    designmod = inputmods[design_mod_idx]

    # get the manufacturing module
    mfgmod = list(filter(lambda m: m['type'] == 'manufacturing', inputmods))[0]['manufacturing']

    by_kit = all(['partsPer' in v for v in inputdict['variants']])
    from_kit = by_kit or inputdict['manufacturing'].get('partsPer', 1) != 1

    # determine if model is by split
    by_split = all(['split' in v for v in inputdict['variants']]) or by_kit
    # by_split = 'split' in inputdict['variants'][0]

    # get the list of cells
    # in the variants, all cells are shared
    all_variant_names = list(variants.keys())
    new_cells = [{'name': c['name'], 'usedBy': all_variant_names} for c in mfgmod['cells']]
    # append to new inputs
    new_inputs['cells'] = new_cells

    # assume all individual tools
    # tools = [{'name' : t['name'], 'usedBy' : [v]} for t in mfgmod['tools'] for v in all_variant_names]
    tools = [{'name': t['name'], 'usedBy': all_variant_names} for t in mfgmod['tools']]
    new_inputs['tools'] = tools

    # generate the sources for the products
    products = []
    for n, v in variants.items():
        # loop through and generate products sources
        prodata = {}
        imods = [deepcopy(m) for m in inputmods]

        if copy_all_mfg:
            # make a deep copy of the manufacturing dict
            mfg_dict = deepcopy(inputdict['manufacturing'])
        else:
            mfg_dict = inputdict['manufacturing']

        # calculate the splits
        if by_split and not by_kit:
            prodata['split'] = v['split']
            # remove rate from mfg
            mfg_dict.pop('rate', None)
            # remove parts per
            mfg_dict.pop('partsPer', None)
        elif by_kit:
            mfg_dict['partsPer'] = v['partsPer']
            # remove rate from mfg
            # mfg_dict.pop('rate', None)
        else:
            # add the rate into the product
            mfg_dict['rate'] = v['rate']

        prodata['source'] = {
            'type': 'custom',
            'modelName': n,
            'manufacturing': mfg_dict,
            'design': inputdict['design'],
            'finance': inputdict['finance'],
            'modules': imods,
        }

        # make a list of updated keys
        updatevkeys = [p['key'] for p in v['parameters']]

        # modify the design module based on the variant input
        new_vars = [
            var.copy()
            for var in designmod['design']['variables']
            if var['key'] not in updatevkeys or var.get('math', False)
        ]

        # add the new parameters to the module
        new_vars.extend([p for p in v['parameters']])
        prodata['source']['modules'][design_mod_idx]['design']['variables'] = new_vars

        # append to the list of products
        products.append(prodata)

    # update the inputs with the products
    new_inputs['products'] = products

    # update repeated values in the new inputs
    repeated_inputs = [
        'modelName',
        'scenario',
        'finance',
    ]
    new_inputs.update({ri: inputdict[ri] for ri in repeated_inputs})

    # get the manufacturing inputs
    mfg_inputs = inputdict['manufacturing']
    mfglist = ['duration', 'quantity', 'amortization']  # manufacturing inputs to bring to top level

    if by_split:
        mfglist += ['rate']

    new_inputs.update({m: mfg_inputs.get(m) for m in mfglist})

    if from_kit and not by_kit:
        # add the overall kit quantity to the system
        new_inputs['partsPer'] = mfg_inputs['partsPer']

    # update additional values in the new inputs
    new_inputs.update({
        'type': 'system',
        'bySplit': by_split,
        'byKit': by_kit,
        'mixType': 'concurrent',
    })

    # set a flag that coming from variants
    new_inputs['fromVariant'] = True

    return new_inputs


def create_from_variants_dict(inputdict):
    'create from an input dict of variants'

    # convert the dict from variant to system
    new_inputs = _sys_dict_from_variant_dict(inputdict=inputdict)

    # run the conversion on the modified input dict list
    interaction = create_from_dict(new_inputs)

    # TODO:  keep a list of the common inputs so their sensitivites can be combined

    return interaction


def update_variants_solution(sol, commoninputs):
    'update the variables in the solution to combine common inputs'
    pass


def update_variant_results(sols, partsper=None, kitname='kit'):
    'reformat the results from system back to product'

    # add pdfpoints to probabilities.flowtimes
    # filter to just be he first
    pts = sols['pdfpoints']
    ft_list = list(filter(lambda p: p['productName'] == pts[0]['productName'], pts))

    sols['probabilities'] = {
        'flowTime': {
            'pdfPoints': ft_list,
            'unit': 'hour',
            'low': 0,
            'q1': 0,
            'median': 0,
            'q3': 0,
            'high': 0,
        }
    }
    sols['resultsIndex'].append({'name': 'Probability Data', 'value': 'probabilities'})

    ## Update the all variables sensitivities
    allvars = sols['allVariables']
    newvars = []  # the new list of the results
    # dvars = list(filter(lambda x: x['source'] == 'Calculated Value', allvars))  # get the results
    # for all of the paramemters, if the "root" input name is the same and the input value is the same for all products, the sensitivities should be combined

    # TODO: update the unit cost breakout to match kit quantity

    # TODO: temporarily remove the variable cost waterfall chart
    # starting with version 1.0.44 there should be a tab for the kit variable cost totals

    ## Unit costs should be kitted
    varcosts = sols['variableCosts']
    # put total kit costs into the variables
    addallvars = []
    # make a dict of the costs
    totvarcosts = {}
    for vc in varcosts:
        kitVal = vc['value'] * vc['kitSplit']
        if vc['name'] not in totvarcosts:
            # add the new summary cost
            totvarcosts[vc['name']] = [kitVal]
        else:
            # append to an existing
            totvarcosts[vc['name']].append(kitVal)

    # add to all variables
    for cname, c in totvarcosts.items():
        addallvars.append({
            'name': f'Total Kit {cname}',  # TODO:  use the actual kit name, if provided
            'value': np.sum(c),
            'unit': '',
            'source': 'Calculated Value',
        })

    # put all the costs
    for vc in varcosts:
        vc['name'] = f"{vc['productName']} | {vc['name']}"

    ## Re-build unit cost breakout
    kitqty = partsper if partsper else 1
    # go though the total cost results for each product
    # find the recurring, non-recurring, and variable costs and generate the
    # rc = sols['']
    # recurring and non-recurring costs should be used directly and summed up to the kit quantity
    # for each of the variable costs, use the kitSplit quantity

    ## Line Results
    #TODO:  line results are being pulled just for the first product
    #       expand out to all products
    for r in sols['lineSummary']:
        # add the product name to the beginning of the product
        r['name'] = f"{r['productName']} | {r['name']}"

    ## update all variables
    allvars.extend(addallvars)

    ## Add warning that variants are in beta
    # sols.setdefault('warnings', []).append('NOTE: Product Variants are in BETA.')

    return sols


def update_usedby(target_list: list[dict], update_from: list[dict], key: str = 'name', inplace=True):
    'utility for updating usedby objects'
    # convert the update from list into a dict
    update_dict = {o[key]: o['usedBy'] for o in update_from}

    for t in target_list:
        t['usedBy'].extend(update_dict.get(t[key], []))


def smash_variant_to_system_dict(proddict, target_parameterization='rate', sys_rate=None) -> dict:
    'convert to a system dict to merge with the proper target parameters'

    srcdict = proddict['source']
    # convert the dict
    sys_dict = _sys_dict_from_variant_dict(srcdict, rename_prods=True, copy_all_mfg=True)

    # find the current parameterization
    if sys_dict.get('bySplit', False):
        if sys_dict.get('byKit', False):
            cur_param = 'kit'
            total_kit = np.sum([p['source']['manufacturing']['partsPer'] for p in sys_dict['products']])
        else:
            cur_param = 'split'
    else:
        cur_param = 'rate'

    pname = srcdict['modelName']

    # check to make sure that only one of 'partsPer' or 'split' is in the product
    if any(['split' in p and 'partsPer' in p['source']['manufacturing'] for p in sys_dict['products']]):
        raise ValueError(f'Product {pname} overspcified with "kit quantity" and "by % split"')

    # change the parameterization
    if target_parameterization == 'split':
        if cur_param == 'split':
            # multiply the splits
            for p in sys_dict['products']:
                p['split'] *= proddict['split'] / 100.0
                # update the rate
                if not sys_rate:
                    raise ValueError('System Rate must be specified')
                # p['source']['manufacturing']['rate'] = sys_rate

        elif cur_param == 'rate':
            # rase an error
            raise ValueError(f'Product {pname} cannot be specified "by % split" in a system defined "by rate"')

        elif cur_param == 'kit':
            for p in sys_dict['products']:
                # create a split
                p['split'] = proddict['split'] * p['source']['manufacturing']['partsPer'] / total_kit
                # # update the rate

    elif target_parameterization == 'rate':
        # find the total input rate of the product
        prod_rate = srcdict['manufacturing']['rate']

        if cur_param == 'split':
            # need to multiply the rate and the split through each product
            for p in sys_dict['products']:
                # have to copy the source before we can change the module
                # p['source']['manufacturing']['rate'] *= p['split']/100.0
                p['source']['manufacturing']['rate'] = prod_rate * p['split'] / 100.0

        elif cur_param == 'rate':
            # rate is already there
            pass

        elif cur_param == 'kit':
            # adjust the rate by parts per
            for p in sys_dict['products']:
                p['source']['manufacturing']['rate'] = prod_rate * p['source']['manufacturing']['partsPer']

    elif target_parameterization == 'kit':
        if cur_param == 'split':
            prod_parts_per = srcdict['manufacturing']['partsPer']
            for p in sys_dict['products']:
                # create a split
                p['source']['manufacturing']['partsPer'] = p['split'] * prod_parts_per / 100.0
                del p['split']

        elif cur_param == 'rate':
            raise ValueError(f'{pname} cannot be specified with rates on variants under a system with kitting')

        elif cur_param == 'kit':
            # already in kit, pull up directly
            pass

    return sys_dict


def check_kit_in_variants(product_input_list):
    for p in product_input_list:
        # loop through
        if 'variants' in p['source'].get('type', []):
            # check that all variants have partsPer
            if not all(['partsPer' in v for v in p['source']['variants']]):
                if not all(['split' in v for v in p['source']['variants']]):
                    # at least one variant is missing
                    return False
        else:
            # just check the product
            if 'partsPer' not in p['source']['manufacturing']:
                return False

    return True