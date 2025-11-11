'''

This module handles the construction og GP constraints from lists
and other serialized objects

'''
import logging
import numbers

import gpkit
import numpy as np


def split_list_on_char(input_list, split_char):
    '''split an input list into a list of lists based on a char

    Args
    ---
    input_list : list
        single list of with
    split_char : char
        the character where to split the list

    Returns
    ------
    list of lists
    '''
    # split into lists by particular value
    # pylint: disable=line-too-long
    # following this trick: https://www.geeksforgeeks.org/python-split-list-into-lists-by-particular-value/
    size = len(input_list)
    idx_list = [idx for idx, val in enumerate(input_list) if val == split_char]
    # if the operation is not found, just return the whole constraint
    if not idx_list:
        return [input_list]

    res = [input_list[i + 1:j] for i, j in zip(
        [-1] + idx_list,
        idx_list + ([size] if idx_list[-1] != size else []),
    )]
    return res


def constant_with_units(constant, units):
    '''create a RHS which is just a constant with units

    Arguments
    ---------
    constant : float
    units : string

    '''
    return float(constant) * gpkit.units(units)


def build_constraint_rhs(variables, constraint_rhs):
    '''build the constraint right-hand side (RHS)

    Args
    ----
    variables : dict or gpkit.Model
        variables which may appear in the RHS referenced by their key

    constraint_rhs : list
        the list representation of a constraint

    Returns
    -------
    gpkit.nomials.math.Posynomial
        a gpkit representation of the constraints
    '''
    error_preamble = 'Error parsing constraint|'
    debug_preamble = 'Constraint parsing|'

    # get the monomials as a list
    accl_add = get_monomials(variables, constraint_rhs)

    return np.sum(accl_add)


def get_monomials(variables, constraint_rhs, incl_hasscalar=True, incl_noscalar=True) -> list[gpkit.Monomial]:
    'get a list of monomials'

    # debugging helpers
    debug_preamble = 'Constraint parsing|'
    error_preamble = 'Error parsing constraint|'

    # split on "+"
    # gets to monomials
    monomials = split_list_on_char(constraint_rhs, '+')

    # accumulators by operations
    accl_add = []

    # for each monomial
    # split on "*"

    logging.debug(debug_preamble + 'Constraint: ' + str(constraint_rhs))

    for monterm in monomials:
        hasscalar = False
        accl_mult = []

        multiplication = split_list_on_char(monterm, '*')
        logging.debug(debug_preamble + 'Monomial: ' + str(monterm))

        # split each multiplication on power
        for multterm in multiplication:
            logging.debug(debug_preamble + 'Multiplication: ' + str(multterm))
            power = split_list_on_char(multterm, "^")
            logging.debug(debug_preamble + 'Power: ' + str(power))

            if isinstance(power[0][0], numbers.Number):
                # if the value is directly a number, use it as a scalar
                base = float(power[0][0])
                hasscalar = True  # set the flag since this monomial has a scalar
            elif isinstance(power[0][0], str):
                if (power[0][0].replace('.', '')).isdigit():  # also checks for decimals
                    base = float(power[0][0])
                    hasscalar = True  # set the flag since this is also a scalar
                else:
                    try:
                        base = variables[str(power[0][0])]
                    except KeyError:
                        # the variable was not found
                        raise ValueError('`%s` is not defined' % str(power[0][0]))
            else:
                raise ValueError(error_preamble + str(power[0][0]) + ' cannot be converted to constraint')
            #
            # try:
            #     base = float(power[0][0])
            # except:
            #     base = variables[str(power[0][0])]
            # finally:
            #     raise Exception(error_preamble + 'cannot convert to number or known variable' + str(monterm))

            if len(power[0]) == 1:
                accl_mult.append(base)
            else:
                exponent = float(power[0][1].replace('^', ''))
                accl_mult.append(base**exponent)
            # else:
            #     raise Exception(error_preamble + 'Power must be two terms only')

        # apply the filters
        if incl_hasscalar and hasscalar:
            # if allowing has scalar and there is a scalar
            accl_add.append(np.prod(accl_mult))
        elif incl_noscalar and not hasscalar:
            # multiply all of the power terms
            accl_add.append(np.prod(accl_mult))

    return accl_add
