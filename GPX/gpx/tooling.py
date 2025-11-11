'tooling models'

from dataclasses import dataclass

from gpkit import VectorVariable
import numpy as np

from gpx import Model, Variable
from gpx.manufacturing import QNACell
from gpx.primitives import Process


@dataclass
class ToolStop():
    'tool stop'
    entrance_cell: QNACell
    processes: list[Process]


class ConwipToolingByStop(Model):
    'closes the loop on tooling'

    def setup(self, tool_stops: list[ToolStop], capacity=1, loop_tool=True, couple_variations=True, **kwargs):
        '''model the tooling flow

        Arguments
        ---------
        cells : list of QNACell
            the cells in the production system
        starting_cell : int
            1-indexed starting cell for the tooling loop
        ending_cell : int
            1-indexed ending cell for the tooling loop
        indicies : set
            a set of indicies to return
            default : None
            if defined, will override the starting and ending cells
        loop_tool : bool
            return the tool to the cell at the first index
        idx : int
            starting index
        '''
        constraints = []

        self.tool_stops = tool_stops
        self.num_stops = len(tool_stops)

        self.cell_start = tool_stops[0].entrance_cell
        self.cell_end = tool_stops[-1].entrance_cell

        # TOOL VARIABLES
        L = self.L = Variable('L_{tool}', 'count', 'required tool count')

        ## Vector Variables by stop
        L_stop = self.L_stop = VectorVariable(self.num_stops, 'L_{stop}', 'count', 'tool count by stop')
        W_stop = self.W_stop = VectorVariable(self.num_stops, 'W_{stop}', 'min', 'flow time through stop')

        # Add constriants by stop
        for i, stop in enumerate(tool_stops):
            # find the total flow time through the stop as the sum of the expected queeuing time and any processing time
            constraints.append(
                W_stop[i] >= stop.entrance_cell.Wq + np.sum([p.t for p in stop.processes])
            )  # if there is FPY, does it affect Process.t or rather QNACell.lam?
            # Appy Little's Law at the tool stop using the rate of the cell
            constraints.append(L_stop[i] >= stop.entrance_cell.lam * W_stop[i])

        # Couple the flow variations (if specified)
        if couple_variations:
            for i in range(self.num_stops - 1):
                constraints.append(tool_stops[i + 1].entrance_cell.c2a >= tool_stops[i].entrance_cell.c2d)

        if capacity is None:
            k = self.k = Variable('k_{capacity}', 'count', 'tool carrier capacity')
        else:
            k = self.k = capacity

        if loop_tool:  # loop the tool around from the last cell to the first
            constraints.append(self.cell_start.c2a >= self.cell_end.c2d)

        # Constrain the total tooling amount by each of the stops and the capacity
        constraints.append(L * k >= np.sum(L_stop))

        # # add start and end cells to class attributes
        # self.cell_start = cells[start0]
        # self.cell_end = cells[end0]
        # self.cell_range = [cells[i] for i in cellrange]

        return constraints
