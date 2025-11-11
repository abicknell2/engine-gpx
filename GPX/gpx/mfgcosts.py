'costs related to manufacturing cells'

from gpkit import ConstraintSet, Model, Variable
import numpy as np

import gpx.manufacturing
from gpx.primitives import Cost


class CellLineCost(Model):  # TODO: is this used?
    ''' Cell-based line cost model

    Capital costs for a cell-based system

    '''

    def setup(self, cells, line, tools, default_currency: str):
        '''
        Arguments
        ---------
        cells : list of tuple
            (gpx.QnaCell : cell object, gpx.Variable : cell cost)

        '''
        self.capitalCost = Variable('cost_{capital}', default_currency, 'Capital Costs')
        constraints = []

        self.workstationCosts = Variable('cost_{workstation}', default_currency, 'Total costs of workstations')


class CellCost(Cost):  # TODO: is this used?
    '''Cell Cost Model
    The cost model for a single manufacturing cell

    Variables
    ---------
    cellCost       [USD/count]   Cell costs that scale with the number of workstations
    cellFixedCost  [USD]         Cell costs that do not scale with   

    '''

    def setup(self, cell, default_currency: str, workstation_cost=None, cell_fixed=None, simple_cost=True):
        '''model setup

        Arguments
        ---------
        cell : QNACell
            the cell to cost
        workstation_cost : gpx.primitives.Cost or gpx.Variable,
            the cost of a workstation
        cell_fixed : gpx.Variable, monomial, posynomial, number
            any fixed cost for a cell which does not scale with number of workstations
        simple_cost : bool
            treat the inputs not as cost objects but only as non-recurring costs
        '''

        super().setup(default_currency=default_currency)
        if simple_cost:
            self.recurringCost = 0
            self.variableCost = 0

            if workstation_cost is None:
                self.cellCost = Variable('cost_{workstation}', default_currency, 'Cost of a workstation')
            else:
                self.cellCost = workstation_cost

            if cell_fixed is None:
                self.cellFixedCost = 0
            else:
                self.cellFixedCost = cell_fixed

            return [self.nonrecurringCost >= cell.m * self.cellCost + self.cellFixedCost]

        # if not simple_cost then can use the full cost object

        constr = []
        self.variableCost = 0

        if workstation_cost.recurringCost is not None and workstation_cost.recurringCost != 0:
            constr.append(self.recurringCost >= cell.m * workstation_cost.recurringCost)
        else:
            self.recurringCost = 0

        if workstation_cost.nonrecurringCost is not None and workstation_cost.nonrecurringCost != 0:
            constr.append(self.nonrecurringCost >= cell.m * workstation_cost.nonrecurringCost)
        else:
            self.nonrecurringCost = 0

        return constr


class FloorspaceCost(Cost):
    '''Floor Space Cost
    Captures recurring and non-recurring costs of the footprint of the factory
    '''

    def setup(
        self,
        cells: dict[str, gpx.manufacturing.QNACell],
        cell_space: dict[str, Variable],
        default_currency: str,
        cost_nr: Variable = None,
        cost_r: Variable = None,
        separate_costs=False,
    ) -> ConstraintSet:
        '''[summary]

        Parameters
        ----------
        cells : dict[str, gpx.manufacturing.QNACell]
            references to all the cells in the system
        cell_space : dict[str, Variable]
            references to all the floor space inputs for the cells
        cost_nr : Variable, optional (default=None)
            reference to the non-recurring cost variable for the floor space
        cost_r : Variable, optional (default=None)
            reference to the recurring cost variable for the floor space
        '''
        # set up the parent cost object
        super().setup(default_currency=default_currency)
        self.variableCost = 0  # there is never floorspace recurring costs

        # create the constraintset
        constr = []

        # find all the cell names with space inputs that are actually in the cells
        cell_names = [n for n in cell_space.keys() if n in cells]

        # create dictionary of variables for the cell floor space
        self.cell_floor_space = {
            cellname:
            Variable(
                '{} Total Floor Space'.format(cellname),  # give the name of the variable
                'm^2',
                'Total floor space for the cell',
            ) for cellname in cell_names
        }

        # create the constriants for the total floorspace for each cell
        constr.extend([self.cell_floor_space[cname] >= cells[cname].m * cell_space[cname] for cname in cell_names])

        if separate_costs:
            # create sub-cost objects for each cell
            self.separated_costs = {
                cname: Cost(default_currency=default_currency) for cname in self.cell_floor_space.keys()
            }

            if cost_nr:
                # calculate the non-recurring cost for each cell
                constr.extend([
                    cc.nonrecurringCost >= cost_nr * self.cell_floor_space[cname]
                    for cname, cc in self.separated_costs.items()
                ])
                # calculate the total nonrecurring cost
                constr.append(
                    self.nonrecurringCost >= np.sum([cc.nonrecurringCost for cc in self.separated_costs.values()])
                )
            else:
                self.nonrecurringCost = 0

            if cost_r:
                # calculate the recurring cost for each cell
                constr.extend([
                    cc.recurringCost >= cost_r * self.cell_floor_space[cname]
                    for cname, cc in self.separated_costs.items()
                ])
                # calculate the total recurring cost
                constr.append(self.recurringCost >= np.sum([cc.recurringCost for cc in self.separated_costs.values()]))
            else:
                self.recurringCost = 0

            for cc in self.separated_costs.values():
                cc.variableCost = 0
                if not cost_nr:
                    cc.nonrecurringCost = 0
                if not cost_r:
                    cc.recurringCost = 0

            # post-calculate the total floor space for use in result processing
            self.totalFloorSpace = Variable(
                'Line Total Floor Space',
                'm^2',
                'Line Total Floor Space',
                evalfn=lambda v: np.sum([v(c) for c in self.cell_floor_space])
            )

        else:
            # calulate the total floor space
            self.totalFloorSpace = Variable('Line Total Floor Space', 'm^2', 'Line Total Floor Space')
            constr.append(self.totalFloorSpace >= np.sum(list(self.cell_floor_space.values())))

            # create the costs
            if cost_nr:
                # create the non recurring cost
                constr.append(self.nonrecurringCost >= cost_nr * self.totalFloorSpace)
            else:
                # set the non-recurring cost to 0
                self.nonrecurringCost = 0

            if cost_r:
                # create the recurring costs
                constr.append(self.recurringCost >= cost_r * self.totalFloorSpace)
            else:
                # set the recurring cost to 0
                self.recurringCost = 0

        # return the constraints
        return constr
