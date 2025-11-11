"""generate interactive objects from different sources

Model Types
-----------
manufacturing
service
operations
"""
import json
from typing import Optional, Union, cast

from api.assembly import AssemblySystem
from api.interactive.model import InteractiveModel
from api.module_plugins import RateRamp
from api.module_types.manufacturing import Manufacturing
from api.module_types.production_finance import ProductionFinance
from api.modules import DESIGN_REQUIRED, ModelModules
from utils.settings import Settings
from utils.types.data import Design
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import (store_original_currency_data, update_settings_finance_units)


def createModelFromDict(inputdict: AcceptedTypes) -> InteractiveModel:
    """creates an interactive model from a json file

    Arguments
    ---------
    inputdict : dict
        a dict containing all of the model properties

    Returns
    -------
    interactive : InteractiveModel
        an interactive model object

    """
    settings = Settings()
    settings = update_settings_finance_units(settings=settings, finance=inputdict.get("finance", None))
    interaction: InteractiveModel = InteractiveModel(inputdict, settings=settings.clone())

    # look for string of GBP or £
    # save inputs as string
    inputstr: str = json.dumps(inputdict)

    # define replacement currencies
    currencies: list[str] = ["$", "GBP", "£", "\\u00a3", "€", "\\u20ac"]
    currency_map = {"$": "USD", "€": "EUR", "£": "GBP"}
    # # TODO:  loop through and replace the symbols (lazy)
    for c in currencies:
        if c in inputstr:  # TODO: update this to check for the case currency better
            interaction.default_currency = c
            interaction.settings.default_currency = c
            interaction.settings.default_currency_iso = interaction.settings.default_currency if c not in currency_map else currency_map[
                c]

    store_original_currency_data(inputdict, settings=interaction.settings)

    # cast back to a dict
    inputdict = json.loads(inputstr)

    # TODO: transition these steps to modules (e.g. finance modules)
    # # setup production system
    # interaction.quantity = calc_quantity(inputdict=inputdict)   #TODO: see if this gets used
    # interaction.rate = inputdict['manufacturing']['rate']  # this assumes a monthly rate
    mfg_data = inputdict["manufacturing"]
    if not isinstance(mfg_data, dict):
        raise ValueError("Manufacturing data must be a dict")

    rate_val = mfg_data.get("rate")
    rate = float(rate_val) if isinstance(rate_val, (int, float, str)) else None
    interaction.rate = rate if rate is not None else None  # this assumes a monthly rate

    # check to see if the request is a tradestudy
    # TODO:  implement the trade as part of the interactive

    # check for design module
    has_design: bool = "design" in inputdict
    if has_design:
        interaction.design = Design(construct_from_dict=inputdict["design"])

    # check to see if the module requires a design model
    requires_design: bool = any(typ in DESIGN_REQUIRED for typ in inputdict["type"])

    # check if uncertainty is specified in the model and get the scenario to return
    if "bestworst" in inputdict["type"]:
        # if not bestworst or no scenario is not specified, scenario is None
        interaction.uncertainty_scenario = inputdict.get("scenario", None)
        interaction.is_uncertain = True

    # create the modules
    interaction.modules = ModelModules(construct_from_dict=inputdict["modules"], designrequired=requires_design)

    # set active module
    try:
        design_dict = cast(dict[Union[int, float, str], Design], inputdict["design"]) if has_design else {}
        manufacturing = cast(Manufacturing, interaction.modules.modules_by_id()[mfg_data["selectedModule"]])
        design = cast(
            Design,
            (interaction.modules.modules_by_id()[cast(float, design_dict["selectedModule"])] if has_design else None)
        )
        interaction.set_active_modules({"manufacturing": manufacturing, "design": design})
    except KeyError as e:
        raise ValueError(f"Could not load the selecrted manufacturing or design module. (code {e})")

    # connect the design module to the manufacturing module
    active_modules = cast(
        dict[str, Union[ProductionFinance, Manufacturing, Design, AssemblySystem]], interaction.active_modules
    )
    active_modules["manufacturing"].design = active_modules["design"]
    interaction.active_modules = active_modules

    # create the rate ramp module
    rates = mfg_data.get("rates")
    if rates:
        if not isinstance(rates, list):
            raise ValueError("Rates must be a list of dictionaries")

        for r in rates:
            if not isinstance(r, dict):
                raise ValueError("Each rate must be a dictionary")

        active_modules = interaction.active_modules
        manufacturing = cast(Manufacturing, interaction.active_modules["manufacturing"])
        manufacturing.rateramp = RateRamp(rates)  # TODO: Find correct type - Manufacturing has no attribute rateramp
        interaction.active_modules["manufacturing"] = manufacturing

    # check to see if there is assembly and create
    if "components" in inputdict:
        # create a module
        assym = AssemblySystem(construct_from_dict=inputdict, default_currency=interaction.settings.default_currency)
        # connect the manufacturing module
        mfgmod = interaction.active_modules["manufacturing"]
        assym.link_mfg(cast(Manufacturing, mfgmod))
        # suppress results in the manufacturing module
        mfgmod.suppress_results = True
        # set the assembly module as current in the interaction
        interaction.active_modules["assembly"] = assym

    # add the finance module if in the module
    if "finance" in inputdict:
        finmod = ProductionFinance(
            construct_from_dict=inputdict,
            mfg_module=cast(Optional[Manufacturing], interaction.active_modules.get("manufacturing")),
        )
        # update financial inputs from manufacturing dict
        finmod.rate = mfg_data.get("rate")

        amortization_val = mfg_data.get("amortization")
        if not isinstance(amortization_val, str):
            raise ValueError("amortization must be a string")

        finmod.amortization = amortization_val  # the type of amort

        if finmod.amortization == "duration":
            # assign the duration input
            finmod.duration = float(cast(dict[str, Union[str, int, float]], mfg_data)["duration"])
            # if there is a rate ramp, substract that duration
            if rates:
                finmod.duration = (
                    finmod.duration - interaction.active_modules["manufacturing"].rateramp.ramp_tot_duration
                )
        elif finmod.amortization == "quantity":
            # assign the quantity variable
            quantity_val = mfg_data.get("quantity")

            if not isinstance(quantity_val, (int, float, str)):
                raise ValueError("Add an input for total manufacturing quantity")

            finmod.quantity = float(quantity_val)
        else:
            raise ValueError("No input for duration or quantity")

        # check for discrete resources
        dr = inputdict["finance"].get("discretizedResources")
        if not isinstance(dr, bool):
            dr = bool(dr)

        interaction.discretized_resources = dr

        # set module to active
        interaction.active_modules["finance"] = finmod

    # check to see if there is kitting information
    if mfg_data.get("partsPer"):
        interaction.active_modules["manufacturing"].partsPer = mfg_data.get("partsPer")
    if inputdict["finance"].get("kitName"):
        interaction.active_modules["manufacturing"].kitName = inputdict["finance"].get("kitName")

    # check to see if there's a rate ramp
    # TODO:  move this whole thing to a plugin in the finance module
    #       have to manually assign the rates to the finance module

    if rates:
        if "finance" in inputdict:
            # connect finance module for the purpose of calculating ramp up
            interaction.active_modules["manufacturing"].rateramp.finmod = finmod
            # interaction.active_modules['manufacturing'].rateramp.update_finmod(finmod=finmod)
        else:
            raise ValueError("Rate Ramp requires a Finance Module")
            # interaction.active_modules['manufacturing'].rateramp = RateRamp(rates=rates)

        # update all the rates after adding the finance module
        interaction.active_modules["manufacturing"].rateramp.update_rates()
        interaction.active_modules["manufacturing"].rateramp.update_finmod(finmod=finmod)

    for key, module in interaction.active_modules.items():
        if module is not None:
            for var in module.variables.values():
                var.infoLabels["model"] = str(key).title()

    # check to see if the manufacturing has off shift and pass the shift time
    if interaction.active_modules["manufacturing"].has_offshift:
        # get the shift time from the finance module
        if "finance" not in interaction.active_modules:
            # check that there is a finance module
            raise ValueError("A finance module must be specified for off-shift production")
        interaction.active_modules["manufacturing"]._offshift_hrsPerShift = interaction.active_modules["finance"
                                                                                                       ].hrsPerShift
        interaction.active_modules["manufacturing"]._offshift_shiftsPerDay = interaction.active_modules["finance"
                                                                                                        ].shiftsPerDay
    # add model types
    return interaction


def calc_quantity(inputdict: AcceptedTypes) -> Optional[float]:
    "calculates the amortization quantity depending if specified as horizon or quantity"

    try:
        mfg = inputdict.get("manufacturing")
        if not isinstance(mfg, dict):
            raise KeyError("manufacturing key is not a dict")
        amort = mfg.get("amortization")
        if isinstance(amort, str) and amort.lower() == "duration":
            # monthly rate gets converted to annual count
            rate = mfg.get("rate")
            duration = mfg.get("duration")
            return float(cast(float, rate)) * float(cast(float, duration)) * 12.0
        elif isinstance(amort, str) and amort.lower() == "quantity":
            return float(cast(float, mfg.get("quantity")))
    except KeyError:
        pass

    return None
