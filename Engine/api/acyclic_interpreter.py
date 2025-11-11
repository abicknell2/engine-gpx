"interprets the acyclic constraints"
from typing import Optional, Union

import gpx
from gpx import Variable
from gpx.dag.parametric import (TOKEN_PRECEDENCE, ParametricConstraint, ParametricVariable)


def make_parametric_variable(
    inputvar: Optional[Variable] = None,
    key: Optional[str] = None,
    name: Optional[str] = None,
    substitution: Optional[tuple[float, str]] = None,
    unit: Optional[str] = None,
) -> ParametricVariable:
    "creates a parametric variable from a gpx variable"
    kwargs: dict[str, object] = {}
    kwargs["varkey"] = inputvar.key if inputvar else key
    kwargs["source_var"] = inputvar if inputvar else None

    if substitution:
        # process the substitution
        kwargs["magnitude"] = substitution[0]
        kwargs["unit"] = substitution[1]
        kwargs["is_input"] = True

    if unit:
        kwargs["unit"] = unit

    if name:
        kwargs["name"] = name
    else:
        kwargs["name"] = inputvar.key

    pv = ParametricVariable(**kwargs)
    if substitution:
        pv.update_value()

    return pv


def make_parametric_constraint(
    inputconstraint: dict[str, bool | int | list[str] | str],
    variables: dict[str, Variable],
    parametric_vars: dict[str, ParametricVariable],
    substitutions: dict[str, tuple[float, str]] = {},
) -> ParametricConstraint:
    "creates a parametric constraint from a gpx constraint"
    # go through each of the list items
    new_constr: list[Union[str, float]] = []
    inputvars: dict[str, ParametricVariable] = {}  # collect the input variables to the constraint
    value = inputconstraint.get("value", [])
    if not isinstance(value, list):
        raise TypeError(
            f"Expected list for 'value' in constraint {inputconstraint['constrno']} on {inputconstraint['name']}, got {type(value).__name__}"
        )
    for token in value:
        if token not in TOKEN_PRECEDENCE:
            if isinstance(token, str) and "^" in token:
                new_constr.append("^")
                # set a new token to be the whole term
                token = "".join(token.split("^")[1:])

            # get the varkey
            if token in variables:
                # process as a variable
                if isinstance(token, str):
                    gpvar: Variable = variables[token]
                else:
                    raise TypeError(f"Expected string token, got {type(token).__name__}")

                # if gpvar.key not in parametric_vars or sub:
                if gpvar.key not in parametric_vars:
                    sub_val = substitutions.get(token)
                    if sub_val is None:
                        sub_val = substitutions.get(str(gpvar.key))
                    # create the parametric variable
                    new_param_var: ParametricVariable = make_parametric_variable(
                        inputvar=gpvar,
                        name=token,
                        substitution=sub_val,
                    )
                    parametric_vars[gpvar.key] = new_param_var
                else:
                    new_param_var = parametric_vars[gpvar.key]

                inputvars[gpvar.key] = new_param_var

                # rename the token in the constraint to the varkey
                new_constr.append(str(gpvar.key))
            else:
                # try and cast as a number
                try:
                    token = float(str(token))
                    new_constr.append(token)
                except ValueError:
                    raise ValueError(
                        f"{token} is not a valid number or variable in constraint {inputconstraint['constrno']} on {inputconstraint['name']}",
                    )

        else:
            # add the token directly
            new_constr.append(str(token))

    # find the output variable or create the ParametricVariable for the output
    output_varname: str = str(inputconstraint["key"])
    output_gpvar: Variable = variables[output_varname]
    units: str = str(inputconstraint.get("unit", ""))
    if output_gpvar.key in parametric_vars:
        output_var = parametric_vars[output_gpvar.key]
        # make sure units are updated
        output_var.unit = units
    else:
        # units = output_gpvar.unitsstr() if hasattr(output_gpvar, 'unitsstr') else ''
        output_var = make_parametric_variable(key=str(inputconstraint["key"]), name=output_varname, unit=units)
        parametric_vars[output_gpvar.key] = output_var

    # create the constraint
    pc: ParametricConstraint = ParametricConstraint(constraint_as_list=new_constr, inputvars=inputvars)
    # couple the output variable and the constraint
    pc.update_output_var(output_var)

    return pc
