'gpx primitives'
import logging
import numbers
from typing import Optional, Union

## transferred to "primitives"
from gpkit import Model, Variable, parse_variables
import numpy as np

from gpx.helpers import optional_variable

FlexibleInput = Union[None, Variable, numbers.Number]


class Process(Model):
    '''process model

    Required Parameters
    -------------------
    t        [min]    process time
    stdev    [min]    standard deviation of process times

    Calculated Variables
    --------------------
    cv    [-]    coefficient of variation

    Cost Variables
    --------------
    capitalcost   [USD]   cell capital costs
    '''

    def setup(
        self,
        time: FlexibleInput = None,
        stdev: FlexibleInput = None,
        cv: FlexibleInput = None,
        fpy: FlexibleInput = 1,
    ):
        '''
        Args
        ----
        time : Variable
            average process time
        stdev : Variable
            process time standard deviation
        constriants : list of constraints
            additional constriants
        fpy : number, variable, monomial
            process first pass yield
        '''
        const = []
        istimeonly = False

        if isinstance(fpy, numbers.Number):
            # check if this is a constant coming in
            if fpy > 1:
                raise ValueError('First Pass Yield greater than 1')
            if fpy == 1:
                # ignoring yield and declare time directly
                istimeonly = True
            else:
                # create the first pass yield variable
                self.fpy = fpy = Variable('Y_{fp}', fpy, '-', 'Process First Pass Yield')

        elif fpy is None:
            # make an empty variable for first pass yield
            self.fpy = fpy = Variable('Y_{fp}', '-', 'Process First Pass Yield')
        else:
            # use the variable directly
            self.fpy = fpy

        if istimeonly:
            # use the time by itself
            if time is None:
                self.t = Variable('t', 'min', 'process time')
            else:
                self.t = time
        else:
            # use the time as a function of the FPY and the expected process time
            if time is None:
                self.tpre = Variable('t', 'min', 'process time')
            else:
                self.tpre = time
            # constrain the new process time with the first pass yield
            self.t = Variable('t', 'min', 'Total process time')
            const.append(self.t >= self.tpre / self.fpy)

        if stdev is None:
            self.stdev = Variable('\\sigma', 'min', 'standard deviation of process time')
        else:
            self.stdev = stdev

        if cv is None:
            cv = self.cv = Variable('cv', '-', 'Coefficient of Variation')
        else:
            self.cv = cv

        const.append(cv == self.stdev / self.t)
        return const


class Timeline(Variable):
    '''a class for handling calender time in a production setting

    '''
    pass


class Cost(Model):
    '''Cost Model Primitive

    Note: if an object does not have a recurring or a non-recurring cost component,
    the variable should be explicitly set to 0 (not as a substitution to the variable)
        Example: `self.recurringCost = 0`

    Variables
    ---------
    nonrecurringCost    [USD]       Non-Recurring Costs
    recurringCost       [USD/hr]    Recurring Costs
    variableCost        [USD/unit]  Variable Costs
    '''

    def setup(
        self,
        default_currency: str,
        nonrecurringCost: Optional[Variable] = None,
        recurringCost: Optional[Variable] = None,
        variableCost: Optional[Variable] = None,
    ):
        # Non‑recurring cost
        self.nonrecurringCost = optional_variable(
            nonrecurringCost,
            variablename="cost_{nonrecurring}",
            units=default_currency,
            descr="Non‑Recurring Costs",
        )

        # Recurring cost
        self.recurringCost = optional_variable(
            recurringCost,
            variablename="cost_{recurring}",
            units=f"{default_currency}/hr",
            descr="Recurring Costs",
        )

        # Variable cost
        self.variableCost = optional_variable(
            variableCost,
            variablename="cost_{variable}",
            units=f"{default_currency}/count",
            descr="Variable Cost",
        )

    def get_total(self, qty, duration):
        '''gets the total cost as a posynomial

        Arguments
        ---------
        qty : Nomial, Variable, numerical
            the quantity to add up the variable costs

        duration : Nomial, Varible, Posynomial, numerical
            the duration over which to sum recurring costs

        Raises
        ------
        Warning

        Returns
        -------
        Posynomial
            represents the sum of all the costs
        '''

        return self.nonrecurringCost + self.recurringCost * duration + self.variableCost * qty

    def get_unit(self, qty, duration):
        '''gets the unit cost

        '''
        pass

    def get_absorbed_recurring(self, rate):
        '''gives only the cost on a recurring basis
        Does not handle nonrecurring cost

        Arguments
        ---------
        rate : Nomial, Variable, numerical
            the production rate

        Raises
        ------
        ValueError

        '''
        pass


class UnitCost(Cost):
    '''Unit Cost Model

    Variables
    ---------
    unitCost   [USD]      The fully absorbed unit cost including variablec costs, absorbed recurring costs and amortized non-recurring costs
    numUnits   [count]    The number of units over which to amortize the non-recurring costs
    horizon    [hrs]      The total production time
    '''

    #
    def setup(
        self,
        *inputcosts,
        num_units=None,
        horizon=None,
        return_costs=True,
        rate=None,
        nonrecurringCost: Optional[Variable] = None,
        recurringCost: Optional[Variable] = None,
        variableCost: Optional[Variable] = None,
        default_currency: str,
    ):
        '''model setup

        Arguments
        ---------
        num_units : gpx.Variable or numerical
            the number of units over which to amortize the costs
            if None, variable automatically created
        horizon : gpx.Variable or numerical
            the production horizon in number of hours
        return_costs : boolean (Default: False)
            should the input cost objects be included in the resulting constraints
        rate :
            production rate
        '''
        #pylint: disable=E1004
        super().setup(
            nonrecurringCost=nonrecurringCost,
            recurringCost=recurringCost,
            variableCost=variableCost,
            default_currency=default_currency,
        )
        self.unitCost = Variable('cost^{total}_{unit}', default_currency, 'Total Unit Cost')

        # Number of units
        self.numUnits = optional_variable(
            num_units,
            variablename="n_{units}",
            units="count",
            descr="Number of units",
        )

        # Production horizon
        self.horizon = optional_variable(
            horizon,
            variablename="t_{horizon}",
            units="hr",
            descr="Production horizon",
        )

        # Production rate
        self.lam = optional_variable(
            rate,
            variablename="\\lambda",
            units="count/hr",
            descr="Production Rate",
        )

        constr = [self.numUnits == self.horizon * self.lam]

        if len(inputcosts) == 1:
            # gpkit creates a "positive c" error if there is only one entry
            logging.debug('unit cost for a single cost object: ' + str(inputcosts))
            constr.append([
                self.unitCost >= inputcosts[0].nonrecurringCost / self.numUnits
                + inputcosts[0].recurringCost * self.horizon / self.numUnits + inputcosts[0].variableCosts
            ])

            return constr

        else:
            cost_types = [
                'nonrecurringCost',
                'recurringCost',
                'variableCost',
            ]

            # sum up all the differnt types of costs
            cost_sums = {
                cost_type: np.sum([getattr(cost, cost_type)
                                   for cost in inputcosts
                                   if isinstance(cost, Cost)])
                for cost_type in cost_types
            }

            # sum_nonrecurr = np.sum([cost.nonrecurringCost for cost in inputcosts])
            # sum_recurr = np.sum([cost.recurringCost for cost in inputcosts])
            # sum_var = np.sum([cost.variableCost for cost in inputcosts])

            for key, sum in cost_sums.items():
                if isinstance(sum, numbers.Number) and sum == 0:
                    cost_sums[key] = None
                    setattr(self, key, 0)
                else:
                    constr.append(getattr(self, key) >= sum)

            if all(np.array(list(cost_sums.values())) == None):
                # if all of the sums are None, there is no unit cost
                self.unitCost = 0
            else:
                #DEBUG: remove the recurring cost for debugging
                print('check constraint')
                print('recurring cost', self.recurringCost)
                print('horizon', self.horizon)
                print('num units', self.numUnits)

                # collect only non-zero terms to sum
                summ = []
                if self.nonrecurringCost != 0:
                    summ.append(self.nonrecurringCost / self.numUnits)

                if self.variableCost != 0:
                    summ.append(self.variableCost)

                if self.recurringCost != 0:
                    summ.append(self.recurringCost * self.horizon / self.numUnits)

                constr.append(self.unitCost >= np.sum(summ))

                #DEBUG: need to feed into
                # constr.append(self.numUnits == self.lam*self.horizon)

            if return_costs:
                constr.extend(inputcosts)

            return constr

    def get_basis(self, cost, basis='unit'):
        '''gets the cost attribute on a particular basis

        Arguments
        ---------
        cost : string ['nonrecurring', 'recurring', 'variable]
            the cost to return in the basis

        basis : string ['unit', 'total', 'hourly'] (Default='unit')
            the basis on which to return the cost

        '''
        cost_types = ['nonrecurring', 'recurring', 'variable']  # The only values for which cost can take on
        basis_types = ['unit', 'total', 'variable']

        if cost not in cost_types:
            raise KeyError(cost, 'not found in cost types', cost_types)

        if cost == 'nonrecurring':
            pass


class TotalCost(Cost):
    #TODO: implement discounted cashflow
    '''calculates the total cost

    Variables
    ---------
    totalCost   [USD]      The fully absorbed unit cost including variablec costs, absorbed recurring costs and amortized non-recurring costs
    numUnits   [count]    The number of units over which to amortize the non-recurring costs
    horizon    [hrs]      The total production time
    '''

    #
    def setup(
        self,
        *inputcosts,
        default_currency: str,
        num_units=None,
        horizon=None,
        return_costs=True,
        calc_units=True,
        rate=None,
    ):
        '''model setup

        Arguments
        ---------
        num_units : gpx.Variable or numerical
            the number of units over which to amortize the costs
            if None, variable automatically created
        calc_units : boolean (default: True)
            return a constriant to calculate the number of units produced
        horizon : gpx.Variable or numerical
            the production horizon in number of hours
        return_costs : boolean (Default: False)
            should the input cost objects be included in the resulting constraints
        rate :
            production rate
        '''
        super().setup(default_currency=default_currency)
        self.totalCost = Variable('cost_{total}', default_currency, 'Total Cost')

        if num_units is None:
            self.numUnits = Variable('n_{units}', 'count', 'Number of units')
        else:
            self.numUnits = num_units

        if horizon is None:
            self.horizon = Variable('t_{horizon}', 'hr', 'Production horizon')
        else:
            self.horizon = horizon

        if rate is None:
            self.lam = Variable('\\lambda', 'count/hr', 'Production Rate')
        else:
            self.lam = rate

        if (horizon == 0 and num_units == 0) or not calc_units:
            constr = []
        else:
            constr = [self.numUnits == self.horizon * self.lam]

        if len(inputcosts) == 1:
            # gpkit creates a "positive c" error if there is only one entry
            logging.debug('unit cost for a single cost object: ' + str(inputcosts))
            constr.append(
                self.totalCost >= inputcosts[0].nonrecurringCost + inputcosts[0].recurringCost * self.horizon
                + inputcosts[0].variableCost
            )
            return constr

        else:
            cost_types = [
                'nonrecurringCost',
                'recurringCost',
                'variableCost',
            ]

            # sum up all the differnt types of costs
            cost_sums = {
                cost_type: np.sum([getattr(cost, cost_type)
                                   for cost in inputcosts
                                   if hasattr(cost, cost_type)])
                for cost_type in cost_types
                if any(hasattr(cost, cost_type) for cost in inputcosts)
            }

            for key, sum in cost_sums.items():
                if isinstance(sum, numbers.Number) and sum == 0:
                    cost_sums[key] = None
                    setattr(self, key, 0)
                else:
                    constr.append(getattr(self, key) >= sum)

            if all(np.array(list(cost_sums.values())) == None):
                # if all of the sums are None, there is no unit cost
                self.totalCost = 0
            else:
                # collect only non-zero terms to sum
                summ = []
                if self.nonrecurringCost != 0:
                    summ.append(self.nonrecurringCost)

                if self.variableCost != 0:
                    summ.append(self.variableCost * self.numUnits)

                if self.recurringCost != 0:
                    summ.append(self.recurringCost * self.horizon)

                constr.append(self.totalCost >= np.sum(summ))

            if return_costs:
                constr.extend(inputcosts)

            return constr

    def get_basis(self, basis='unit', **kwargs):
        'get the total cost in a particular basis'

        if basis == 'unit':
            try:
                return self.totalCost / kwargs['num_units']
            except KeyError:
                # num_units is not defined
                # use from object
                return self.totalCost / self.numUnits
