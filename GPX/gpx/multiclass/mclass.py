'single class model for use in a system in a multi-class context'

import gpkit
from gpkit import ConstraintSet, Model, Variable
import numpy as np

import gpx
from gpx import feeder


class MClass(Model):
    '''a single class for a multi-class modeling context

    Attributes
    ----------
    line : gpx.manufacturing.Fabline
        the fabrication line for the class

    cells : list of gpx.manufacturing.QNACell
        all of the cells which the class uses
        accessed to determine the shared resources with other classes

    processes : list of gpx.manufacturing.Process
        all processes required for the class


    GPX Variables
    -------------
    tot_cost    [USD]         Total cost
    lam         [count/hr]    Class production rate
    '''

    def setup(
        self,
        line: gpx.manufacturing.FabLine,
        feeder_lines: list[feeder.FeederLine] = [],
        by_split: bool = False,
        return_processes: bool = True,
        **kwargs,
    ):
        '''GPX setup

        Arguments
        ---------
        line : gpx.manufacturing.Fabline
            the single-product line to reference

        feeder_lines : list of feeder.FeederLine
            the feeder lines to reference

        by_split : boolean (default=False)
            determines if the system rates are determined by product splits
            must provide `system_rate` and `class_rate`

        Keyword Arguments
        ------------------
        class_rate : number           
            the production rate of the class
            used outside of by split specification

        class_split : number < 1.0
            the split of the individual class of the system production rate
            used with the split specification

        return_processes : boolean (default=True)
            return all of the processes from within the cells along with the class

        '''
        # check that the class is properly specified
        self.by_split = by_split
        if by_split:
            # if defined by splits, should have all the split information
            try:
                if kwargs['class_split'] > 1:
                    raise Exception('class split must be less than 1')
            except KeyError:
                raise Exception('must specify a split for the class in the split context')

        else:
            # if not defined by splits, should have rate information as input
            if 'class_rate' not in kwargs:
                raise Exception('must specify a class rate outside of the split context')

        # include the line from the product
        self.line = line
        self.cells = line.cells
        self.processes = [c.process for c in line.cells]
        # include any feeder lines
        self.feeder_lines = feeder_lines

        # create properties for the class
        self.W = Variable('W_{class}', 'min', 'class total flow time')

        constr = []  # create extra constraints
        self.cell_lams = []

        # map the cells back to this MC-Class
        for c in self.cells:
            c.parent_mcclass = self

        # maps feeder cells to parent class
        for fl in self.feeder_lines:
            for c in fl.cells:
                c.parent_mcclass = self

        if not by_split:
            _ = kwargs['class_rate']  # retain for compatibility; actual substitution handled upstream
            # define the class production rate as a free variable
            self.lam = Variable('\\lambda_{class}', 'count/hr', 'class production rate')

            # track the cell lambda variables for later linkage but avoid immediate substitution
            for c in self.line.cells:
                self.cell_lams.append(c.lam)

        else:
            # define the class by split
            self.X = Variable(
                'X_{class}',
                kwargs['class_split'],
                '-',
                'class rate split from system',
            )

            self.lam = Variable('\\lambda_{class}', 'count/hr', 'class production rate')

            for c in self.line.cells:
                # for each cell add a constraint for the rate of the cell
                constr.append(c.lam >= self.lam)
                # add the split to the cell
                c.X = self.X

        # return all of the rates
        self.all_lams = [*self.cell_lams, self.line.lam]

        # constr.extend(self.line)       # return just the line (does not include cell constraints)

        # find the additional flow times through the fedeer lines
        addl_W = [c.W for f in self.feeder_lines for c in f.cells]
        # create the constraint on the total flow time for the class
        constr.append(self.W >= self.line.W + np.sum(addl_W))

        if return_processes:
            # return the processes from the cells
            constr.extend(self.processes)

        self.W_cap = Variable('\\bar W_{class}', 'min', 'deadline on flow time')
        constr.append(self.W <= self.W_cap)

        # return the model
        return constr, line, ConstraintSet(self.feeder_lines)


class MClassFeeder(Model):
    """A light-weight class wrapper for a single feeder line."""

    def setup(self, feeder_line, *, parent_class, **kwargs):
        # remember who owns this feeder
        self.parent_class = parent_class

        # line + cells + processes only from the feeder
        self.line = feeder_line
        self.cells = list(feeder_line.cells)
        self.processes = [c.process for c in self.cells]

        # make sure the feeder line has a flow-time aggregate
        try:
            _ = self.line.W
        except AttributeError:
            self.line.W = sum(c.W for c in self.cells)

        # map cells back to their owning mcclass for later lookups/aggregation
        for c in self.cells:
            c.parent_mcclass = parent_class

        # class-level vars (same names as MClass so MCSystem can use them uniformly)
        self.W = Variable('W_{class}', 'min', 'feeder class total flow time')
        self.W_cap = Variable('\\bar W_{class}', 'min', 'deadline on feeder flow time')
        self.lam = Variable('\\lambda_{class}', 'count/hr', 'feeder class production rate')

        constr = []

        # feeder class flow time is at least the feeder line flow time
        constr.append(self.W >= self.line.W)
        constr.append(self.W <= self.W_cap)
        
        # for c in self.cells:
        #     constr.append(c.lam >= self.lam)

        # return processes if you want them surfaced; typically not required
        # constr.extend(self.processes)

        return constr, self.line