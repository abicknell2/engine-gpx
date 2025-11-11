"interpret the constraints from the engine"

import contextlib
import io
import logging

import gpkit
from gpkit import units as gp_units
import numpy as np
import pint.errors

import gpx
from gpx.constraint_builder import (build_constraint_rhs, constant_with_units, get_monomials)
import utils.logger as logger
from utils.types.shared import AcceptedTypes

logging.basicConfig(level=logging.DEBUG)


def build_single_constraint(variables: dict[str, gpx.Variable],
                            constraint_as_dict: dict[str, gpkit.ConstraintSet]) -> list[object]:
    """build a single posynomial constraint

    Args
    ----
    variables : dict
        dictionary of variables which may appear in the constraints
        key is the name as used in the constraints
        value is the gpx variable representation

    constraint_as_dict : dict
        the dictionary version of a constraint in the following format
        'name' : string
            the name of the variable in the LHS
        'sign' : string
            the equality or inequality operator
        'math' : boolean
            is the variable constrained by a Posynomial
        'value' : list
            the Posynomial constraint
    """
    logger.debug(f"Constraint interpreter|building single constraint: {constraint_as_dict}")

    unit: str = str(constraint_as_dict['unit'])
    var = variables[str(constraint_as_dict["name"])]
    opr = constraint_as_dict["sign"]

    posynomial_build_error = False  # flag to show if there is an issue creating the posynomial

    # the right-hand-side (rhs) of the constratint
    rhs = None

    # check to see if there is math
    if "math" in constraint_as_dict and constraint_as_dict["math"]:
        try:
            rhs = build_constraint_rhs(variables, constraint_as_dict["value"])
        # except ValueError as e:
        except Exception as e:
            # this can arise if the posynomial is not dimensionally consistent
            if "is not defined" in str(e):
                # handle as though the variable is not defined
                raise e
            if isinstance(e, pint.errors.DimensionalityError):
                # # assume there is a parsing issue
                # logging.info('CONSTRAINT-INTERPRETER | ERROR: ' + str(e))
                # raise ValueError(str(e) + ' in "%s"' % str(constraint_as_dict.get('name')))
                # we are going to try and fix the constraint first
                posynomial_build_error = True
                # save the old error
                olderror = e

            else:
                raise e

        # try and fix the constraint if there is a posynomial issue or if the lhs
        # & rhs are not dimensionally consistent
        # have to check is equal to False since the boolean logic behavior for
        # non-empty constraints is unclear
        if posynomial_build_error or (var == rhs) is False:
            # try and correct the constraint
            try:
                new_rhs = rebuild_constraints(variables, constraint_as_dict, lhs=var, target_unit=unit)
            except pint.errors.DimensionalityError as newe:
                # there is still an issue with the dimensionality of the posynomial
                logging.info(f"CONSTRAINT-INTERPRETER | ERROR: {newe}")
                # raise ValueError(str(newe) + ' in "%s"' % str(constraint_as_dict.get('name')))
                raise ValueError(
                    f"{newe} in constraint #{constraint_as_dict.get('constrno', '')} \"{constraint_as_dict.get('name', '')}\"",
                ) from newe

            if new_rhs:
                # if there is successfully a new rhs, update
                # no update will still throw an error, below
                rhs = new_rhs

        if rhs is None:
            # if rhs is still none, raise the original error
            raise ValueError(
                f"{olderror} in constraint #{constraint_as_dict.get('constrno', '')}: \"{constraint_as_dict.get('name', '')}\"",
            )

        try:
            # try and build the constraint
            c = None

            uniterrbuffer = io.StringIO()
            with contextlib.redirect_stdout(uniterrbuffer):
                if opr == ">=":
                    c = var >= rhs
                    # return [var >= rhs]
                elif opr == "<=":
                    # return [var <= float(rhs)*unit]
                    # try:    # put in the exception if tries to be a posynomial
                    c = var <= rhs
                    # return [var <= rhs]
                elif opr == "=":
                    c = var == rhs
                    # return [var == rhs]
                else:
                    exception = f"operator not recognized : {opr}"
                    raise Exception(exception)

            if c is False:
                # this is a dimensionality error
                raise pint.errors.DimensionalityError(var.units, rhs.units, extra_msg=uniterrbuffer.getvalue())
            else:
                # return the constraint set
                return [c]

        except Exception as e:
            constrno = constraint_as_dict.get("constrno", "")
            if isinstance(e, pint.errors.DimensionalityError):
                # catch pint.errors.DimensionalityError
                errstr = str(e)
                # get rid of the unicode dot
                errstr = errstr.translate({ord("·"): " * "})
                # check if there is an extra message
                if e.extra_msg:
                    errstr = str(e.extra_msg)
                    errstr = errstr.split(":")[1]

                errstr = errstr.replace("//", "of")
                logging.error(f"error creating constraint {constrno}: {errstr}")  # log the error
                logger.exception(e)  # get the stack trace
                raise ValueError(f"Unit conflict at constraint {constrno}: {errstr}") from e
            else:
                raise ValueError(
                    f'Could not create constraint #{constrno} for "{constraint_as_dict.get("name", "<unknown>")}"',
                ) from e
    else:
        # math is either FALSE or doesn't exist
        # assume that it is just some constant
        # check that uncertainty is not trying to be applied to an inequality
        if "min" in constraint_as_dict or "max" in constraint_as_dict:
            raise ValueError(
                f'Cannot have an uncertain inequality for "{constraint_as_dict["name"]}". Try creating an additional uncertain variable.',
            )
        try:
            rhs = constant_with_units(constraint_as_dict["value"], unit)
        except AttributeError as e:
            vname = str(constraint_as_dict["name"])
            val = str(constraint_as_dict["value"])
            logging.error(
                f"CONSTRAINT INTERPRETER| fails on variable {vname}, value {val}, units {unit} with error: {e}",
            )
            raise ValueError(
                f'Issue creating "{vname}" as {val} {unit} in constraint # {constraint_as_dict.get("constrno", "")}',
            )

        if opr == ">=":
            return [var >= rhs]
        elif opr == "<=":
            return [var <= rhs]
    return []


def build_constraint_set(
    variables: dict[str, gpx.Variable],
    constraints_as_list: list[dict[str, int | str]],
    acyclic_input: bool = False,
    **kwargs: AcceptedTypes
) -> gpkit.ConstraintSet:
    """build constraints
    cons
        Args
        ----
        variables : dict
            dictionary of variables
            name : gpx.Variable
                key is the name
                value is the gpx variable representation
                probably a merge between the module variables and the design variables

        constraints_as_list : list
            the list of constraints in dict format

        Returns
        -------
        gpkit.ConstraintSet
    """
    constraints = []
    for constr in constraints_as_list:
        if acyclic_input and constr.get("acyclic", False):
            # skip adding the constraint
            continue

        constraints.extend(build_single_constraint(variables, constr))

        # check to make sure there are no booleans
        if not all([not isinstance(c, bool) for c in constraints]):
            raise ValueError("bad constraint found in building constraint set")

    return constraints


def rebuild_constraints(
    variables: dict[str, gpx.Variable],
    constraint_as_dict: dict[str, object],
    lhs: gpx.Variable,
    target_unit: str = ""
) -> object:
    "rebuild the constriaints with the unit"
    constr = constraint_as_dict["value"]
    newrhs = []

    # make sure there are scalars in the constraint
    mons_with_scalars = get_monomials(variables, constr, incl_hasscalar=True, incl_noscalar=False)
    mons_no_scalars = get_monomials(variables, constr, incl_hasscalar=False, incl_noscalar=True)

    if len(mons_with_scalars) == 0:
        # if there are no sclaras in the constraint there's nothing we can do
        return None

    # try and modify with the target units
    for smon in mons_with_scalars:
        # check to see if this monomial fails
        if (lhs == smon) is False:
            # try and scale with target units
            new_smon = smon * gp_units(target_unit)
            if (lhs == new_smon) is False:
                # if it still doesn't succeed, this cannot be fixed
                return None
            else:
                # try the monomial again
                # if it works, add the modified monomial
                newrhs.append(new_smon)
        else:
            # if this monomial is fine, just add it to the new rhs
            newrhs.append(smon)

    # add all the other (non-scalar) monomial terms
    if len(mons_no_scalars) > 0:
        # if there are monomials without scalars, add them now
        newrhs.extend(mons_no_scalars)

    # return the new rhs
    return np.sum(newrhs)
