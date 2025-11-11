'costs for multi_product'
from collections import namedtuple
from dataclasses import dataclass
from itertools import groupby
from typing import Dict, List, Tuple

from gpkit import ConstraintSet, Model, Variable
from gpkit.constraints.tight import Tight
import numpy as np

import gpx
from gpx import TIGHT_CONSTR_TOL
import gpx.mfgcosts
import gpx.multiclass
import gpx.multiclass.mctool
from gpx.primitives import Cost, TotalCost
import gpx.recurring_cost


class ProductVariableCost(Cost):
    'summarize the costs for a single product'

    def setup(
        self,
        *inputcosts,
        split=1,
        default_currency: str,
        num_units=None,
        return_costs=False,
        separable_costs=False,
        calc_units=False,
        **kwargs
    ):
        super().setup(default_currency=default_currency)

        self.split = split

        # create variable for number of units
        if num_units is None:
            self.numUnits = Variable('n_{units}', 'count', 'total unit count')
        else:
            self.numUnits = num_units

        self.recurringCost = 0
        self.separable_costs = separable_costs
        # constraints
        constr = []

        # for separable costs, maintain the vairable cost
        if self.separable_costs:
            # constrain the total variable cost
            if inputcosts:
                constr.append(self.variableCost >= np.sum([c.variableCost for c in inputcosts]))
            else:
                # if there are no costs
                self.variableCost = 0
            # set non-recurring cost
            self.nonrecurringCost = 0

        else:
            # no variable cost
            self.variableCost = 0

        # create the equivalent non-recurring cost
        if len(inputcosts) > 0:
            constr.append(self.nonrecurringCost >= np.sum([c.variableCost for c in inputcosts]) * self.numUnits)
        else:
            # set the non recurring cost to zero
            self.nonrecurringCost = 0

        if return_costs:
            constr.extend(inputcosts)

        return ConstraintSet(constr)


@dataclass
class SeperableVariableCost:
    prod_name: str
    cost: Cost


class SeperatedSystemCost:
    'a class for handling the separate costs of a system'

    def __init__(self, fixed_costs: list[Cost] = [], variable_costs: list[SeperableVariableCost] = [], **kwargs):
        self.fixedCosts: list[Cost]
        self.variableCosts: dict[str, Cost] = {}

        # create the list of fixed costs ignoring any None inputs
        self.fixedCosts = [fc for fc in fixed_costs if fc]

        # convert the variable cost tuples into a dict keyed by the product name
        for k, g in groupby(sorted(variable_costs, key=lambda x: x.prod_name), lambda x: x.prod_name):
            self.variableCosts[k] = sum([gg.cost.variableCost for gg in g])


class MultiProductCost(Cost):
    '''cost model for a multiproduct system

    Variables
    ---------
    c_0_total    [USD]    Total System Cost

    '''

    def setup(self, *inputcosts, **kwargs):
        '''set up the models for the total cost

        Arguments
        ---------
        system : gpx.multiclass.MCSystem
            the multi-class system to cost

        Optional Keyword Arguments
        --------------------------
        cells : list of
        tools
        '''
        super().setup()

        totalcost = TotalCost(*inputcosts)

        self.c_0_total = totalcost.totalCost

        return totalcost


class MCCellCostABC(Cost):
    'cell costs with ABC computation of costs'

    #FUTURE:    do not rely on comparing the MCClasses

    def setup(
        self,
        cell_costs: List[gpx.mfgcosts.CellCost],
        floorspace_costs=None,
        tool_costs: List[gpx.multiclass.mctool.MCToolCost] = [],
        **kwargs
    ):
        super().setup()
        tight_constr = []

        ClassTotal = namedtuple('ClassTotal', ['mcclass', 'costobj'])
        self.total_class_costs: List[ClassTotal] = []

        # structures for handling the costs
        ClassCost = namedtuple('ClassCost', ['mcclass', 'nrcost', 'rcost'])
        class_costs: List[ClassCost] = []

        # get the ABC costs from the cell
        for c in cell_costs:
            cell_classes = c.cell.cells
            # and apply the ABC
            for cc in cell_classes:
                # get the ITEM split from the cell
                xitem = cc.xitem
                # create monomials for the costs
                class_costs.append(
                    ClassCost(
                        mcclass=cc.parent_mcclass,
                        nrcost=c.nonrecurringCost * xitem,
                        rcost=c.recurringCost * xitem,
                    )
                )

        # ABC from floor space
        fcs = floorspace_costs
        if fcs:
            # get the cells
            for cname, c in fcs.cells.items():
                if cname not in fcs.cell_names:
                    # skip cells that don't have floorspace costs
                    continue
                # get the subtotal for each cell
                subtot: Cost = fcs.separated_costs[cname]
                for cc in c.cells:
                    # loop over each of the QNA cells for each product in the cell
                    xitem = cc.xitem
                    class_costs.append(
                        ClassCost(
                            mcclass=cc.parent_mcclass,
                            nrcost=subtot.nonrecurringCost * xitem,
                            rcost=subtot.recurringCost * xitem,
                        )
                    )

        # ABC from tooling
        # consider the split by L and lot size (self.k). implement xitem on the tool
        if tool_costs:
            for tname, tc in tool_costs.items():
                for t in tc.tool.tools:
                    # loop over each tool and create the class cost object
                    xitem = t.xitem
                    class_costs.append(
                        ClassCost(
                            mcclass=t.parent_mcclass,
                            nrcost=tc.nonrecurringCost * xitem,
                            rcost=tc.recurringCost * xitem,
                        )
                    )

        # create cost objects for each class and group
        sort_func = lambda x: x.mcclass
        for k, g in groupby(sorted(class_costs, key=lambda x: id(x.mcclass)), sort_func):
            # create a new cost object for each class's ABC costs
            tcc = Cost()
            grouped_costs: List[ClassCost] = list(g)
            # total the different costs
            tot_nr_cost = np.sum([gg.nrcost for gg in grouped_costs])
            tot_rec_cost = np.sum([gg.rcost for gg in grouped_costs])

            if tot_nr_cost:
                # append a constraint
                tight_constr.append(tcc.nonrecurringCost >= tot_nr_cost)
            else:
                tcc.nonrecurringCost = 0
            if tot_rec_cost:
                tight_constr.append(tcc.recurringCost >= tot_rec_cost)
            else:
                tcc.recurringCost = 0

            # append to the list
            self.total_class_costs.append(ClassTotal(mcclass=k, costobj=tcc))

        # return the constriants as a tight set
        return Tight(tight_constr, reltol=TIGHT_CONSTR_TOL)

    def get_class_costs(self, mcclass) -> Cost:
        'gets the split cost of each class'
        # filter throught all the costs to get the object corresponding to the class
        classcost = next(filter(lambda x: x.mcclass is mcclass, self.total_class_costs))
        # get the cost object
        return classcost.costobj


class MCInvHoldingValue(Cost):
    'inventory holding costs for a class in a multi-class system'

    def setup(
        self,
        mc_class: gpx.multiclass.MClass,
        *costs: Cost,
        cell_costs: Cost = None,
        inv_hold_obj: gpx.recurring_cost.InventoryHolding = None,
        inv_count=None,
        **kwargs
    ):

        super().setup()

        constr = []
        tight_constr = []

        self.nonrecurringCost = 0
        self.recurringCost = 0

        self.horizon = Variable('horizon', 'hr', 'production horizon')

        # get the rate from the class
        self.class_lam = mc_class.lam

        # create collectors for the difference catgeories of costs
        r_costs = 0  # recurring cost
        nr_costs = 0  # non-recurring cost
        var_costs = 0  # variable costs

        for c in costs:
            r_costs += c.recurringCost
            nr_costs += c.nonrecurringCost
            var_costs += c.variableCost

        # add the costs from the cell cost
        if cell_costs is not None:
            r_costs += cell_costs.recurringCost
            nr_costs += cell_costs.nonrecurringCost

        # total all the costs on a unit basis
        #FUTURE:    may want to keep all the costs separate (good for unit-cost breakout?)
        #           have the constributions of the recurring, non-recurring, variable, etc.
        total_costs = 0

        # variable costs are added directly
        if var_costs:
            total_costs += var_costs

        # recurring costs need the rate
        if r_costs:
            total_costs += r_costs / self.class_lam

        # non-recurring costs get summed and divided by the product rate
        if nr_costs:
            total_costs += nr_costs / self.class_lam / self.horizon

        # add the constraint
        tight_constr.append(self.variableCost >= total_costs)

        if inv_hold_obj:
            # assign the inventory value
            tight_constr.append(inv_hold_obj.inventoryValue >= self.variableCost)

        # return the tight constraint set
        return Tight(tight_constr, reltol=TIGHT_CONSTR_TOL)
