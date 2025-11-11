'recurring part costs'
from collections import OrderedDict as OD
import numbers
import typing

from gpkit import Variable, units
import numpy as np

import gpx.manufacturing
from gpx.primitives import Cost, Process


class InventoryHolding(Cost):
    '''Inventory Holding Cost Model

    Unitizes holding costs based on fraction of a year in the system
    Based on the opportunity cost of capital

    Alternatively: could find the non-recurring cost of the entire program and then amortize it

    Variables
    ---------
    holdingRate       [-]       Holding Rate (annual)
    inventoryValue    [USD]     The value of a single piece of inventory
    L                 [count]   Inventory Count
    W                 [hr]     Flow time of a piece of inventory

    '''

    def setup(
        self,
        default_currency: str,
        inv_count=None,
        holding_rate=None,
        flow_time=None,
        inv_value=None,
    ):
        '''
        Arguments
        ---------
        inv_count : Variable
            the variable for inventory in the system
        holding_rate : Variable
        '''
        super().setup(default_currency=default_currency)
        if holding_rate is None:
            self.holdingRate = Variable('r', 0.1, '', 'Annual discount rate')
        else:
            self.holdingRate = holding_rate
        if inv_count is None:
            self.L = Variable('L_{holding}', 'count', 'WIP Inventory in holding')
        else:
            self.L = inv_count
        if flow_time is None:
            self.W = Variable('W_{holding}', 'hr', 'Flow time of a piece of inventory')
        else:
            self.W = flow_time
        if inv_value is None:
            self.inventoryValue = Variable('d_{inventory}', default_currency, 'Inventory Value')
        else:
            self.inventoryValue = inv_value

        self.nonrecurringCost = 0
        self.recurringCost = 0

        return [
            self.variableCost >= self.holdingRate * self.W.to('hr') / 24. / 365. * units('1/hr') * self.inventoryValue
        ]


class VariableCosts(Cost):
    '''Variable Cost Model

    creates recurring, variable costs

    items
    -----
    varcost_dict : dict
        a dictionary of the variable cost input variables
    '''

    def setup(self, *varcosts, varcosts_as_dict=None, default_currency: str):
        '''setup

        Arguments
        ---------
        *varcosts : gpkit.Variable, gpkit.Monomial, gpkit.Posynomial
            variable costs to add
        varcosts_as_dict : dict
            variable costs with names
            name : constant, gpkit.Variable, gpkit.Monomial, gpkit.Posynomial
        '''
        super().setup(default_currency=default_currency)
        self.nonrecurringCost = 0
        self.recurringCost = 0

        if varcosts_as_dict is None:
            if len(varcosts) == 0:
                self.variableCost = 0
                return []
            else:
                return [self.variableCost >= np.sum(varcosts)]

        else:  # create the variables costs from a dictionary
            if len(varcosts_as_dict) == 0:
                self.variableCost = 0
                return []
            self.varcost_dict = varcosts_as_dict
            return [self.variableCost >= np.sum(list(self.varcost_dict.values()))]


class RecurringCosts(Cost):
    '''Recurring Cost Model

    creates recurring, variable costs

    items
    -----
    varcost_dict : dict
        a dictionary of the variable cost input variables
    '''

    def setup(self, *varcosts, varcosts_as_dict=None, default_currency: str):
        '''setup

        Arguments
        ---------
        *varcosts : gpkit.Variable, gpkit.Monomial, gpkit.Posynomial
            recurring costs to add
        varcosts_as_dict : dict
            recurring costs with names
            name : constant, gpkit.Variable, gpkit.Monomial, gpkit.Posynomial
        '''
        super().setup(default_currency=default_currency)
        self.nonrecurringCost = 0
        self.variableCost = 0

        if varcosts_as_dict is None:
            if len(varcosts) == 0:
                self.recurringCost = 0
                return []
            else:
                return [self.recurringCost >= np.sum(varcosts)]

        else:  # create the variables costs from a dictionary
            if len(varcosts_as_dict) == 0:
                self.recurringCost = 0
                return []
            self.varcost_dict = varcosts_as_dict
            return [self.recurringCost >= np.sum(list(self.varcost_dict.values()))]


#TODO: implement
class LaborCost(Cost):
    '''Labor Cost Model

    Captures the labor costs as recurring costs
    '''

    def setup(
        self,
        default_currency: str,
        cells_headcount: dict[str, tuple[gpx.manufacturing.QNACell, Variable]] = {},
        laborRate=150,
        is_variable_cost=True,
    ):
        '''
        Arguments
        ---------
        cells_headcount : dict of tuples
            maps the cells to headcounts
            cellname : (QNAcell : cell object, headcount variable)
        is_variable_cost : Boolean (default: True)
            defines whether to treat the labor costs as variable costs or recurring costs
        '''

        super().setup(default_currency=default_currency)
        self.nonrecurringCost = 0
        if laborRate is None:
            self.laborRate = Variable('r_{labor}', f'{default_currency}/hr/count', 'Labor rate')
        elif isinstance(laborRate, numbers.Number):  # is the laborRate something numerical
            self.laborRate = Variable('r_{labor}', laborRate, f'{default_currency}/hr/count', 'Labor rate')
        else:
            self.laborRate = laborRate

        self.cellLabor = {
            name: Variable(f"{name} Labor Cost", default_currency, f"{name} Labor Cost")
            for name in cells_headcount.keys()
        }

        # set the cells headcount as an instance variable
        self.cells_headcount = cells_headcount

        if is_variable_cost:
            # treat labor as a variable cost
            self.recurringCost = 0

            constr = [self.cellLabor[name] >= c[0].tnu * c[1] * self.laborRate for name, c in cells_headcount.items()]

            # sum all the labor costs
            constr.append(self.variableCost >= np.sum(list(self.cellLabor.values())))

        elif not is_variable_cost:
            # treat labor as a recurring cost
            self.variableCost = 0

            constr = [self.cellLabor[name] >= c[0].m * c[1] * self.laborRate for name, c in cells_headcount.keys()]

        return constr


class ProcessLaborCost(Cost):
    '''Process Labor Cost Model

    Captures the labor costs as recurring costs
    '''

    def setup(
        self,
        processes_in_cells: dict[str, list[str]],
        all_processes: dict[str, Process],
        all_gpx_cells: dict[str, gpx.manufacturing.QNACell],
        default_currency: str,
        cells_headcount: dict[str, tuple[gpx.manufacturing.QNACell, Variable]] = {},
        process_headcount: dict[str, tuple[Process, Variable]] = {},
        feeder_cell_qty={},
        #   feeder_cell_qty:      dict[str, typing.Union(numbers.Number)]={},
        laborRate=150,
        is_variable_cost=True,
        sum_to_cell=True,
    ):
        '''
        Arguments
        ---------
        cells_headcount : dict of tuples
            maps the cells to headcounts
            cellname : (QNAcell : cell object, headcount variable)
        is_variable_cost : Boolean (default: True)
            defines whether to treat the labor costs as variable costs or recurring costs
        '''

        super().setup(default_currency=default_currency)
        self.nonrecurringCost = 0
        if laborRate is None:
            self.laborRate = Variable('r_{labor}', f'{default_currency}/hr/count', 'Labor rate')
        elif isinstance(laborRate, numbers.Number):  # is the laborRate something numerical
            self.laborRate = Variable('r_{labor}', laborRate, f'{default_currency}/hr/count', 'Labor rate')
        else:
            self.laborRate = laborRate

        self.cellLabor = {
            name: Variable(f"{name} Labor Cost", default_currency, f"{name} Labor Cost")
            for name in cells_headcount.keys()
        }

        # set the cells headcount as an instance variable
        self.cells_headcount = cells_headcount
        self.process_headcount = process_headcount

        if not is_variable_cost:
            raise ValueError('ProcessLaborCost must be calculated as a variable cost')
            # treat labor as a variable cost
            self.variableCost = 0

            constr = [self.cellLabor[name] >= c[0].tnu * c[1] * self.laborRate for name, c in cells_headcount.items()]

            # sum all the labor costs
            constr.append(self.variableCost >= np.sum(list(self.cellLabor.values())))

        elif is_variable_cost:
            # treat labor as a variable cost
            self.recurringCost = 0

            # constraints for the labor costs
            constr = []

            # the list of the constraints for finding the max
            maxconst = []

            # go cell by cell to see if there are headcounts
            for cname, plist in processes_in_cells.items():
                # hold the terms to sum
                hcsums = []

                # get the cell object
                cellobj = all_gpx_cells[cname]

                # check to see if there is headcount for the cell
                celldata = cells_headcount.get(cname, None)

                if celldata:
                    # there are headcounts for the cell
                    for pname in plist:
                        # look for process-specific headcount
                        prochc = process_headcount.get(pname, None)

                        if prochc:
                            # there is a process headcount, use that
                            hcsums.append(prochc[1] * all_processes[pname].t)

                        else:
                            # no count specified for process. use cell headcount
                            # append to the sums
                            hcsums.append(cells_headcount[cname][1] * all_processes[pname].t)

                else:
                    # no headcount input for the cell. check all processes
                    # create a variable for the cell labor
                    self.cellLabor[cname] = Variable('{} Labor Cosst'.format(cname), default_currency)

                    for pname in plist:
                        if pname in process_headcount:
                            hcsums.append(process_headcount[pname][1] * all_processes[pname].t)

                    # if there is not any labor cost, delete the variable os it doesn't get summed
                    if not hcsums:
                        # hcsums is empty
                        del self.cellLabor[cname]

                # find the total labor effort for the cell
                if hcsums:
                    # sum only if there are costs
                    # find any feeder line amounts
                    fqty = feeder_cell_qty.get(cname, 1)
                    constr.append(self.cellLabor[cname] >= np.sum(hcsums) * self.laborRate / cellobj.k * fqty)

        # find the total cost at the cell
        constr.append(self.variableCost >= np.sum(list(self.cellLabor.values())))

        return [*constr, *maxconst]
