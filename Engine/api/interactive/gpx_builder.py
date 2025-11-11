import logging
from typing import TYPE_CHECKING

from gpkit import Variable, VarKey, units

from api.acyclic_interpreter import make_parametric_variable
from api.constants import epsilon
from api.errors import ApiModelError
import gpx
from gpx.dag.parametric import ParametricConstraint, ParametricVariable
import utils.logger as logger
from utils.settings import Settings
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import setup_currency_conversion_variables

if TYPE_CHECKING:
    from api.interactive.model import InteractiveModel


class GpxModelBuilder:

    def __init__(self, interactive: "InteractiveModel", inputdict: AcceptedTypes) -> None:
        self.interactive = interactive
        self.inputdict = inputdict
        self.default_currency = None
        self.settings = None

    def build(
        self, settings: Settings, for_assembly: bool, **kwargs: AcceptedTypes
    ) -> None:  # noqa: C901  (same complexity as before)
        "generate gpx model from context"
        self.interactive.gpx_constraints = []
        self.interactive._substitutions_ = {}
        self.settings = settings
        self.default_currency = settings.default_currency_iso

        # self.all_variables = {}
        gpx_vars: dict[str, Variable] = {}
        fx_subs: dict[VarKey, float] = {}

        # # get all the variables from the model
        # for m in self.active_modules.values():
        #     # self.all_variables.extend(getattr(m, 'variablesSimple'))
        #     self.all_variables.update(getattr(m, 'variables', {}))

        try:
            # first, generate all variables for the modules
            for name, module in self.interactive.active_modules.items():
                if hasattr(module, "make_gpx_vars"):  # this needs to be explicit! NOT ask for forgiveness!!
                    module.make_gpx_vars(settings)
                else:
                    logger.debug(f"Module {name} does not have a `make_gpx_vars` function")

                if hasattr(module, "gpx_variables"):
                    gpx_vars.update(module.gpx_variables)

                if getattr(module, "gpx_variables", None):
                    fx_subs.update(self.apply_currency_conversion_to_module(module, gpx_vars))

            # TODO:  sort the order in which the modules are generated
            sort_order = {
                "assembly": 10,  # assembly has to be made before manufacturing
                "design": 5,
                "manufacturing": 1,
            }
            sorted_mod_list = sorted(
                self.interactive.active_modules.items(),
                key=lambda n: sort_order.get(n[0], 0),  # get the priority based on the type of module or set to end
                reverse=True,  # keying in decreasing order
            )

            # then get constraints amd substitutions from modules
            # for name, module in self.active_modules.items()

            for name, module in sorted_mod_list:
                if hasattr(module, "gpx_translate"):
                    try:
                        module.gpx_translate(settings=self.settings)  # tanslate the module to its gpx constraint form
                    except KeyError as e:
                        logging.error(f"could not perform gpx translation on module {name}: {e}")
                        logger.exception(e)  # try to get a more detailed traceback
                        raise ApiModelError(f"Key Error on variable: {e} in module: {name}")

                    self.interactive.gpx_constraints.extend(module.gpx_constraints(acyclic_input=True))
                    self.interactive._substitutions_.update(module.substitutions)
                    self.interactive.acyclic_constraints.extend(
                        module.get_acyclic_constraints(
                            parametric_variables=self.interactive.acyclic_params,
                            substitutions=self.interactive._substitutions_,
                        ),
                    )

                    # use gpx substitution object instead
                    # self._substitutions_.update(module.gpx_substitutions)
                    gpx_vars.update(module.gpx_variables)

        except KeyError as e:
            # could not find the variable
            # TODO:  this error might be catching too much. Too general
            # log the original error and the traceback
            logging.error(f"error generating gpx from module {name}: {e}")
            logger.exception(e)  # this should log the traceback
            raise ApiModelError(
                f"Variable {e} not found. Check UNCONSTRAINED VARIABLES",
                status_code=551,
            )

        self.interactive.gpx_model = self.interactive.context.get_gpx_model(self.settings)

        if fx_subs:
            self.interactive.gpx_model.substitutions.update(fx_subs)
            logger.debug(f"[FX] Applied {len(fx_subs)} FX numerical substitutions "
                         "to GPX model")

        # GENERATE SUBSTITUION DICT
        newsubs: dict[str, gpx.Variable] = {}
        # TODO:  raise error when a substitution cannot be made

        for name, sub in self.interactive._substitutions_.items():
            # filter through all the substitutions
            if name in gpx_vars:
                # only do the sub if can find the variable
                # TODO: make sure this is catching the subs for chi + eta
                u = str(sub[1])  # units and convert input unit format
                subval = sub[0] if sub[0] not in (0, "0") else epsilon
                # substitute epsilon for 0
                if subval is None:
                    raise ValueError(name, "must have an input value")
                newsubs[gpx_vars[name].key] = subval * units(u) if u != "-" else subval

        # get substitutions from the context which are already in GPX form
        newsubs.update(self.interactive.context.get_substitutions(for_assembly=for_assembly, **kwargs))

        # update all model substitutions
        self.interactive.gpx_model.substitutions.update(newsubs)

        # store all gpx variables in the object
        self.interactive.gpx_variables = gpx_vars

    def apply_currency_conversion_to_module(self, module, gpx_vars: dict[str, Variable]) -> dict[VarKey, float]:
        """
        Walks the substitutions of `module` and injects currency-conversion plumbing
        whenever an input value is expressed in a currency that is not the project
        base currency defined in `self.default_currency`.
        """

        _CURRENCIES = {"USD", "GBP", "EUR"}

        # start of verbose debug output for this module
        logger.debug(f"[FX] ── Currency-scan for module: {getattr(module, 'name', module)}")

        # harvest or create ParametricVariables that represent each explicit FX rate
        fx_subs: dict[VarKey, float] = {}
        finance = self.inputdict.get("finance", {})
        conversion_rates = finance.get("conversionRates", [])
        if conversion_rates:
            try:
                fx_params = setup_currency_conversion_variables(
                    self.inputdict,
                    base=self.default_currency,
                )
                logger.debug(f"[FX] Found explicit FX table with {len(fx_params)} entries")
            except Exception as exc:
                logger.warn(f"[FX] Failed to initialise FX variables: {exc!r}")
                raise exc
        else:
            logger.debug("[FX] No explicit FX table present in model — using fallback rates")
            fx_params = setup_currency_conversion_variables(
                self.inputdict,
                base=self.default_currency,
                allow_dynamic_fallback=True  # Force fallback
            )

        for fx_param in fx_params:
            # add each FX VarKey to the global GPX variable registry
            gpx_vars[fx_param.varkey.key] = fx_param.varkey
            fx_subs[fx_param.varkey] = fx_param.magnitude
            self.interactive.acyclic_params[fx_param.name] = fx_param

            module.gpx_variables[fx_param.name] = fx_param.varkey
            gpx_vars[fx_param.name] = fx_param.varkey

            if hasattr(fx_param, "uncertainty_bounds"):
                if 'min' in fx_param.uncertainty_bounds:
                    fx_param.min = fx_param.uncertainty_bounds['min']

                if 'value' in fx_param.uncertainty_bounds:
                    fx_param.value = fx_param.uncertainty_bounds['value']

                if 'max' in fx_param.uncertainty_bounds:
                    fx_param.max = fx_param.uncertainty_bounds['max']

                fx_param.key = fx_param.name
                fx_param.category = "fx_rate"
                fx_param.type = "FX Rate"
                fx_param.property = "rate"
                fx_param.tags = []

                module.variables[fx_param.name] = fx_param

            logger.debug(f"[FX] Registered FX variable {fx_param.name} to {fx_param.magnitude}")

        # examine every substitution attached to this module
        module_subs: dict[str, tuple[AcceptedTypes, str]] = getattr(module, "substitutions", {}) or {}

        for sub_name, (sub_value_raw, sub_unit_raw) in list(module_subs.items()):
            if sub_name not in gpx_vars:
                logger.debug(f"[FX] Skip {sub_name} because GPX variable not yet registered")
                continue

            unit_string = str(sub_unit_raw)
            matched_currency = next((cur for cur in _CURRENCIES if cur in unit_string), None)

            # early exit if not a currency amount or already base currency
            if matched_currency is None or matched_currency == self.default_currency:
                continue

            fx_key = f"{matched_currency} to {self.default_currency}"
            fx_param = self.interactive.acyclic_params.get(fx_key)
            if fx_param is None:
                logger.debug(f"[FX] No FX rate available for {fx_key}, skipping {sub_name}")
                continue

            logger.debug(f"[FX] Converting {sub_name}: {sub_value_raw} {unit_string} "
                         f"using FX variable {fx_key}")

            # guard against zero which causes infeasible logarithms
            sub_value_safe = sub_value_raw if sub_value_raw not in (0, "0") else epsilon

            original_key_string = gpx_vars[sub_name].key
            original_param_name = f"{original_key_string} (original)"
            converted_unit_string = unit_string.replace(matched_currency, self.default_currency)

            # create ParametricVariable representing the user-supplied original value
            original_pv = make_parametric_variable(
                inputvar=gpx_vars[sub_name],
                name=original_param_name,
                substitution=(sub_value_safe, unit_string),
            )
            original_pv.update_value()
            self.interactive.acyclic_params[original_pv.name] = original_pv
            logger.debug(f"[FX] Created parametric input {original_pv.name}")

            # create VarKey and ParametricVariable for the converted value
            converted_varkey = VarKey(
                name=str(original_key_string),
                units=converted_unit_string,
                label="converted",
                key=str(original_key_string),
            )
            converted_pv = ParametricVariable(
                name=original_key_string,
                varkey=converted_varkey,
                unit=converted_unit_string,
                is_input=False,
            )
            self.interactive.acyclic_params[converted_pv.name] = converted_pv
            logger.debug(f"[FX] Created converted variable {converted_pv.name} "
                         f"with units {converted_unit_string}")

            # add equality constraint linking original, FX rate, and converted
            conversion_constraint = ParametricConstraint(
                constraint_as_list=[original_pv.name, "*", fx_param.name],
                inputvars={
                    original_pv.name: original_pv,
                    fx_param.name: fx_param,
                },
            )
            conversion_constraint.update_output_var(converted_pv)
            self.interactive.acyclic_constraints.append(conversion_constraint)
            logger.debug(f"[FX] Added constraint {converted_pv.name} = "
                         f"{original_pv.name} * {fx_param.name}")

            # replace the GPX variable reference in this module so downstream math uses the converted value
            module.gpx_variables[sub_name] = Variable(converted_varkey)
            gpx_vars[sub_name] = module.gpx_variables[sub_name]

            # move substitution to the “… (original)” entry
            module_subs.pop(sub_name, None)
            module_subs[original_param_name] = (sub_value_safe, unit_string)
            logger.debug(f"[FX] Redirected substitution for {sub_name} → {original_param_name}")

        return fx_subs
