'non recurring costs'

import logging

from gpkit import Model, Variable
import numpy as np

from gpx.primitives import Cost


class NonRecurringCost(Cost):
    '''Non-Recurring Cost Model

    '''

    def setup(
        self,
        *costs,
        default_currency: str,
        costs_as_dict=None,
    ):
        super().setup(default_currency=default_currency)
        self.recurringCost = 0
        self.variableCost = 0

        constraints = []

        if costs_as_dict is None:
            if len(costs) == 0:
                self.nonrecurringCost = 0
            else:
                constraints.append(self.nonrecurringCost >= np.sum(costs))
        else:
            if len(costs_as_dict) == 0:
                self.nonrecurringCost = 0
            else:
                self.cost_dict = costs_as_dict
                constraints.append(self.nonrecurringCost >= np.sum(list(self.cost_dict.values())))

        return constraints


class ConwipToolingCost(Cost):
    '''in a CONWIP production system, we need to determine how many tools of
    each type are required

    The tooling loop also adds an additional constraint on the arrival process
    at the cell which begins the tooling loop
    '''

    def setup(
        self,
        conwip_tool,
        tool_cost,
        default_currency: str,
        useful_life=None,
        renewal_cost=None,
    ):
        '''
        Arguments 
        ---------
        conwip_tool : gpx.manufacturing.ConwipTooling
            the tool to apply the cost
        tool_cost : Cost | Variable
            pass in the tooling costs
        '''
        super().setup(default_currency=default_currency)

        self.variableCost = 0

        self.useful_life = useful_life
        self.renewal_cost = renewal_cost

        if tool_cost is None:
            self.toolCost = Variable('cost_{tool}', default_currency, 'Cost of a tool')
        else:
            self.toolCost = tool_cost

        self.tool = conwip_tool
        self.L = self.tool.L  # number of tools required

        if not isinstance(self.toolCost, Cost):
            # if the toolCost is not Cost just return the simple non-recurring cost
            self.variableCost = 0
            self.recurringCost = 0
            return [self.nonrecurringCost >= self.L * self.toolCost]

        # optional subtotal variable for inspection
        self.subtot = Variable('\\Sigma cost_{tool}', default_currency, 'subtotal for this tool')

        constraints = []

        if self.toolCost.nonrecurringCost != 0:
            constraints.append(self.nonrecurringCost >= self.L * self.toolCost.nonrecurringCost)
        else:
            self.nonrecurringCost = 0

        if self.toolCost.recurringCost != 0:
            constraints.append(self.recurringCost >= self.L * self.toolCost.recurringCost)
        else:
            self.recurringCost = 0

        return constraints


class CellCost(Cost):
    '''Cell Cost Model
    The cost model for a single manufacturing cell

    '''

    def setup(self, cell, default_currency: str, workstation_cost=None, fixed_cost=None):
        '''model setup

        Arguments
        ---------
        cell : QNACell
            the cell to cost
        workstation_cost : Variable or numerical
            the cost of a workstation
        fixed_cost : gpx.Variable, monomial, posynomial, number
            any fixed cost for a cell which does not scale with number of workstations

        Keyword Arguments
        -----------------
        full_cost : gpx.primitives.Cost
            the full cost object
        '''
        logging.warn(
            'cell cost from non-recurring costs will be phased out in favor of cell cost in manufacturing costs'
        )

        super().setup(default_currency=default_currency)

        if workstation_cost is None:
            self.cellCost = Variable('cost_{workstation}', default_currency, 'Cost of a workstation')
        else:
            self.cellCost = workstation_cost

        if not isinstance(self.cellCost, Cost):
            #DEBUG
            print(isinstance(self.cellCost, Cost))

            # if the cellCost is not Cost just return the simple non-recurring cost
            self.variableCost = 0
            self.recurringCost = 0
            return [self.nonrecurringCost >= cell.m * self.cellCost]

        # if the cost object is the full object, add the other constraints for the costs
        constr = []
        if self.cellCost.nonrecurringCost != 0:
            constr.append(self.nonrecurringCost >= cell.m * self.cellCost.nonrecurringCost)
        else:
            self.nonrecurringCost = 0

        if self.cellCost.recurringCost != 0:
            constr.append(self.recurringCost >= cell.m * self.cellCost.recurringCost)
        else:
            self.recurringCost = 0

        if self.cellCost.variableCost != 0:
            self.variableCost = self.cellCost.variableCost
        else:
            self.variableCost = 0

        return constr


# class CellCosts(Cost):
#     ''' Cell-based line cost model
#
#     Capital costs for a cell-based system
#
#     '''
#     def setup(self, cells, line, tools):
#         '''
#         Arguments
#         ---------
#         cells : list of tuple
#             (gpx.QnaCell : cell object, gpx.Variable : cell cost)
#
#         '''
#         super().setup()
#         self.recurringCost = 0
#
#         self.workstationCosts = Variable('cost_{workstation}', 'USD', 'Total costs of workstations')


class LineCost(Cost):
    '''collect all of the nonrecurring and recurring costs for a line
    '''

    def setup(self, *inputcosts, return_costs=False, default_currency: str):
        '''
        Arguments
        ---------
        inputcosts : gpx.primitives.Cost
            various cost models to consider
        '''
        super().setup(default_currency=default_currency)
        return [
            self.nonrecurringCost >= np.sum([c.nonrecurringCost for c in inputcosts]),
            self.recurringCost >= np.sum([c.recurringCost for c in inputcosts]),
            self.variableCost >= np.sum([c.variableCost for c in inputcosts]),
        ]