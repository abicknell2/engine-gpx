'nice helper functions'

import numbers

import gpkit
from gpkit import Variable


def optional_variable(
    argval,
    variablename: str = "",
    units: str = "",
    descr: str = "",
    **kwargs,
):
    '''creates optional variables

    Use Case: creating models wich have optional input variables to the constraints.
    For example, optionally passing a process time to a process model

    Arguments
    ---------
    argval : number, Variable, Monomial
        the input argument
    variablename : string
        the name to give the variable if created
    units : string
        the units to give the variable if created
    descr : string
        the description of the variable
    **kwargs
        optional keyword arguments
    '''
    # Check if the input is a plain number
    if isinstance(argval, numbers.Number):
        # Create a new GPkit Variable with the provided value
        var = Variable(variablename, argval, units, descr, **kwargs)
    # If the input is None, create a symbolic variable (no fixed value yet)
    elif argval is None:
        var = Variable(variablename, units, descr, **kwargs)
    # If the input is already a GPkit Variable or Monomial, use it directly
    elif isinstance(argval, (Variable, gpkit.nomials.math.Monomial)):
        var = argval
    else:
        raise TypeError(f"Not a valid input for a variable: {argval!r}")

    return var