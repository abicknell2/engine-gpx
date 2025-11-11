'implements tooling models for a multiclass system'

from typing import List

from gpkit import ConstraintSet, Model, VectorVariable
from gpkit.constraints.tight import Tight
import numpy as np

from gpx import TIGHT_CONSTR_TOL, Model, Variable
from gpx.manufacturing import ConwipTooling
from gpx.non_recurring_costs import ConwipToolingCost
from gpx.primitives import Cost


class MCTool(Model):

    def setup(self, tools: list[ConwipTooling], return_tool_constraints=True):
        'tooling setup'
        self.tools = tools
        constr = []
        tight_constr = []

        # constraints for the tools
        L = self.L = Variable('L_{Multi-class tool}', 'count', 'tool total count')
        constr.append(L >= np.sum([t.L for t in tools]))

        # Return the constraints for the tools as well if requested
        if return_tool_constraints:
            constr.extend([tc for tc in tools])  # find the tooling share splits
        for i, t in enumerate(tools):
            # add inventory and item splits so each product sees its share
            t.xinv = Variable(f'xInv_{i}', '-', 'lot split values')
            t.xitem = Variable(f'xItem_{i}', '-', 'item split')
            constr.extend([t.xinv == t.L / L, t.xitem == t.xinv / t.k])

        # Return the constraints
        return ConstraintSet([constr, Tight(tight_constr, reltol=TIGHT_CONSTR_TOL)])


class MCToolCost(Cost):
    'a separable tooling cost to apply ABC to shared tooling'

    def setup(
        self,
        conwip_tool: MCTool,
        tool_cost=None,
        separate_costs=False,
        tool_class_map=None,
        **kwagrs
    ) -> ConstraintSet:
        # get the constraints from the base model of cost
        constr = []

        self.tool = conwip_tool
        self.L = self.tool.L

        self.variableCost = 0
        self.recurringCost = 0

        # assign the total cost of tool to be for each class and tool

        # separate costs according to class

        #TODO:  similar to cell ABC

        # flatten all the tool costs
        # create the splits

        return ConstraintSet(constr)