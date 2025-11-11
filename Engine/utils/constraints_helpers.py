import logging
from typing import Optional, cast

import numpy as np

import api.constants as constants
from api.module_types.module_type import ModuleType
import utils.logger as logger
from utils.types.data import Parameter
from utils.types.shared import (LHS, ConstraintsList, SubsConstraintsLHSTuple, Substitutions)

logging.basicConfig(level=constants.LOG_LEVEL)

DESIGN_NOT_REQUIRED: list[str] = ["service", "operations", "manufacturing"]

DESIGN_REQUIRED: list[str] = ["custom"]


def sort_constraints(listofconstraints: ConstraintsList) -> SubsConstraintsLHSTuple:
    """sorts the constraints into substitutions and constraints

    Arguments
    ---------
    listofconstraints : list
        the list of all of the constraints

    Returns
    -------
    substitutions : dict
        the variables that will have to be recorded as substitutions
    constraints : list
        the list of the formatted constraints
    LHS : dict
    """
    substitutions: Substitutions = {}
    constraints: ConstraintsList = []
    lhs: LHS = {}

    # loop through the list of variables
    for constrno, v in enumerate(listofconstraints):
        # add the front end constraint number to the constraint (account for 1-indexing)
        v["constrno"] = constrno + 1

        # TODO:  display errors when creating the constraints and substitutionss

        try:
            # add the variable to the LHS
            lhs[str(v["name"])] = Parameter(construct_from_dict=v)
            # check for units. if units are missing, assume unitless
            inputunits: Optional[str] = str(v["unit"]) if isinstance(v.get("unit"), str) else None
            if not inputunits:
                v["unit"] = ""
            # if 'unit' not in v.keys(): v['unit'] = ''
            # check for percentage as units and correct to decimal
            if inputunits == "%":
                # check to see if the input is a dynamic constraint
                if isinstance(v.get("value"), list):
                    # if isinstance(v['value'], list):
                    # throw error if trying to apply a percentage to a dynamic constraint
                    # ignore if there is not an input value (e.g. in best/worst
                    # uncertainty)
                    raise ValueError('"%" cannot be used with a dynamic constraint. Use "unitless" instead.')

                v["unit"] = ""
                # check to see if there is a value
                value = v.get("value", None)
                if value is not None and isinstance(value, (int, float)):
                    v["value"] = float(value / 100.0)

            if "math" not in v.keys() or not v["math"]:
                # check to see if there is a sign entry. If not, assume substitution
                if "sign" not in v:
                    v["sign"] = "="
                if v["sign"] == "=":
                    # this is a variable which should be substituted directly
                    try:
                        if v["value"] == 0:
                            v["value"] = constants.epsilon
                            logger.debug(f"epsilon substituted for 0 for: {v['name']}")

                        substitutions[str(v["name"])] = (float(v["value"]), str(v["unit"]))

                    except KeyError:
                        logging.info(f"value is either < 0 or undefined for > {v['name']}")

                elif v["sign"] == ">=" or v["sign"] == "<=":
                    # this is a constraint but is a monomial
                    v["value"] = v["value"] if v["value"] != 0 else constants.epsilon
                    constraints.append(v)

                else:
                    logging.warning(f"issue parsing constraint: {str(v)}")

            elif v["math"]:
                # full posynomial constraint
                constraints.append(v)

            else:
                logging.warning(f"issue parsing constraint: {str(v)}")

        except ValueError as e:
            logging.error(f"Constraint Generation| {str(e)}")
            raise ValueError(f"Error creating constraint {v['constrno']}: {str(e)}") from e

    # # remove duplicates from the list of LHS as per: https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists
    # lhs = list(set(lhs))

    return substitutions, constraints, lhs


def update_module_constraints_bestworst(module: ModuleType, variables: dict[str, Parameter]) -> bool:
    """update constraints from a best-worst analysis

    Arguments
    ---------
    module : ModuleType
        the module to update
    variables : dict[str, Parameter]
        the variables to update

    Returns
    -------
    wasUpdated : bool
        if there were no updates made, returns false
    """
    # flag if there were any updates made to any of the variables
    wasUpdated: bool = False

    # lists of variables to update
    # vars_to_update = []  # all of the variables which need substituions
    # updated and constraints deleted
    vars_to_sub: list[str] = []
    subs_to_add: dict[str, tuple[float, str]] = {}  # substituions which can be directly updated

    # TODO:  this was an old method where a constrained variable could be replaced with
    #       an uncertain variable. At the moment, this concept is not in use. However,
    #       we'll leave the code here in case it should be needed again.

    # # find variables which were constrained in the base case but are now inputs
    # for var in variables:
    #     if 'isConstrained' in var and var['isConstrained']:
    #         if 'value' in var and var['value'] is not None:
    #             # has a likely value so can update directly
    #             vars_to_update.append(var['name'])
    #         elif 'min' in var and 'max' in var:
    #             if var['min'] is not None and var['max'] is not None:
    #                 # likely value guessed from the average of min and max
    #                 vars_to_update.append(var['name'])
    #                 logger.debug('adding substitution for approximate likely case for: ' + var['name'])
    #                 subs_to_add[var['name']] = (np.mean([var['min'], var['max']]), var['unit'])

    # logger.debug('[engine| update best worst] preparing to filter constraints by variables: ' + str(vars_to_update))

    # wasUpdated = len(vars_to_update) > 0 or len(subs_to_add) > 0

    # # get a filtered list of constraints to remove stuff on LHS using original constraints as source
    # newconstr = remove_constrtaint_by_lhs(module._original_constraints, *vars_to_update)

    # # re-sort all of the constraints
    # module.substitutions, module.constraints, module.lhs = sort_constraints(newconstr)

    # find variables that do not have a value and create from min, max
    # for var in variables:
    #     if 'value' not in var or var['value'] is None:
    #         # substitute the mean of the min and max for the first solve
    #         var['value'] = np.mean([var['min'], var['max']])

    # if the likely is not equal to the value, append to substitutions
    # vars_to_sub.extend([
    #     var for var in variables
    #     if 'value' in var and var['value'] is not None])
    # logger.debug('updating substitutions for: ' + ', '.join([var['name'] for var in vars_to_sub]))

    # find variables with min-max but no likely and substitute average
    # for var in variables:
    #     if 'value' not in var or var['value'] is None:
    #         # only if there is not a value or value is none
    #         logger.debug('[Engine|update best worst] substitute mean as likely case for: ' + var['name'])
    #         subs_to_add[var['name']] = (np.mean([var['min'], var['max']]), var['unit'])

    # ## Modify the substitutions in the module
    # module.substitutions.update({var['name'] : (float(var['value']), var['unit']) for var in vars_to_sub
    # if 'math' in var and not var['math']})  # filter out if the value is a
    # constraint

    # find the variables which do not have value or value is None and add to
    # substitutions
    for key, v in variables.items():
        if not hasattr(v, 'value') or v.value is None:
            # Ensure v.min and v.max are not None before computing the mean
            if v.min is None and v.max is None:
                # If all values are None, substitute with epsilon
                subs_to_add[key] = (constants.epsilon, v.unit)
            else:
                # Filter out None values before computing the mean
                valid_values = [val for val in [v.min, v.max] if val is not None]
                # If at least one valid value exists, compute the mean
                if valid_values:
                    subs_to_add[key] = (float(np.mean(valid_values)), v.unit)
                else:
                    # Safety fallback (shouldn't trigger, but just in case)
                    subs_to_add[key] = (constants.epsilon, v.unit)

    module.substitutions.update(subs_to_add)

    wasUpdated = wasUpdated or len(vars_to_sub) > 0

    # TODO:  see if this is really needed. Seems to be covered by
    #       functionality in interaction.generate_gpx
    # re-run to change any $ to  in the newly-updated substitutions
    # module._subs_from_params()
    #       functionality above.

    return wasUpdated


def update_module_constraints(module: ModuleType, constraint_updates: ConstraintsList) -> None:
    """update the active constraints in a module

    Arguments
    ---------
    module : ModuleType
        the module where the constraints should be updated
    cosntraint_updates : list of dicts
        new constraints to update
    """
    # reset the constraints
    reset_module_constraints(module)
    varstoremove: list[str] = []
    addconstr: ConstraintsList = []

    # loop through constraints to update
    for cu in constraint_updates:
        # check if the updated constraint is substitution
        if cu["sign"] == "=":
            varstoremove.append(str(cu["name"]))
        # add the constraint to a list
        addconstr.append(cu)

    newconstr: ConstraintsList = remove_constrtaint_by_lhs(module._original_constraints, *varstoremove)
    newconstr.extend(addconstr)
    # re-sort the constraints
    module.substitutions, module.constraints, module.lhs = cast(SubsConstraintsLHSTuple, sort_constraints(newconstr))
    # apply to module


def remove_constrtaint_by_lhs(
    constraints: ConstraintsList,
    *update_vars: str,
    key: str = 'name',
) -> ConstraintsList:
    """filter our constraints where lhs is in the update variable
    Arguments
    ---------
    constraints : list of dicts
        list of the constraints as dictionaries
    update_vars : string
        variables to filter by
    key : string
        the entry in the constraint over which to filter

    Returns
    -------
    list of dicts
        the filtered constraint list
    """
    new_constraints: ConstraintsList = []
    for c in constraints:
        if c[key] not in update_vars:
            new_constraints.append(c)
    return new_constraints


def reset_module_constraints(module: ModuleType) -> None:
    """resets the active constraints to the original"""
    module.substitutions, module.constraints, module.lhs = sort_constraints(module._original_constraints)
    # module._active_constraints = {c['name'] : c for c in module.constraints}
