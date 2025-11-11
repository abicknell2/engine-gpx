'the objects and functions to create a single class (product) in a multi-class context'

import itertools

import gpkit
from gpkit.nomials.variables import Variable
import numpy as np

from gpx import Model
from gpx.multiclass import MCell, MClass


class MCSystem(Model):
    '''a system of one or more classes is in a multi-class context
    
    Attributes
    ----------
    shared_cells : list of lists of tuples
        the cells which are shared across classes are established here
        [[(MClass, QNAcell), (MClass, QNACell)]]

    classes : list of MClass
        determines the classes comprising the Multi-class system

    mcells : list of MCCell
        all of the multi-class cells

    single_cells : list of gpx.manufacturing.QNACell
        cells that are not shared
        (append the )

    by_split : boolean (default=False)
        the products in the system are defined by splits


    Variables
    ---------
    '''

    def setup(self, classes=[], feeders: list = None, cellmap=[], cellvars={}, by_split=False):
        '''
        Arguments
        ---------
        classes : list of MClass
            determines 
        cellmap : dict of lists of gpx.manufacturing.QNACells and properties
            {'<shared cell name>' : [<list of the QNACells sharing the resource]}
            cells that are in the same list are assumed to be shared across classes
        cellvars : dict of dicts (default={})
            {'<shared cell name>' : {'<var type>' : gpx.Variable}}
            attaches special variables to the cells
        by_split : boolean (default=False)
            the system is defined by a system rate and product splits

        Free Variables
        --------------
        lambda_{system}     count/hr        system production rate
            
        '''
        # define production rate of the system
        self.lam = Variable('\\lambda_{system}', 'count/hr', 'system production rate')

        # store the changeover and split states
        self.by_split = by_split

        # if there is only one class, this is not multi-class
        self.classes = classes
        self.feeders = feeders or []

        # loop over cell map and create mcells
        self.mcells = {}

        for name, c in cellmap.items():
            if not c:
                continue
            # create an MCell object for all the shared cells
            #TODO:  check in cellvars to see if there is changeover
            if 'Changeover Time' in cellvars[name]:
                # there is a changeover time associated with this cell
                has_changeover = True
                t_changeover = cellvars[name]['Changeover Time']
            else:
                # do not apply changeover to the cell
                has_changeover = False
                t_changeover = None

            mcell = MCell(cells=c, by_split=self.by_split, has_changeover=has_changeover, t_changeover=t_changeover)
            # expose the shared-cell display name for downstream tooling such as discrete solves
            setattr(mcell, "display_name", name)
            self.mcells[name] = mcell

        # create a flattened set of all the cells
        flat_cells = [cc for c in cellmap.values() for cc in c]
        self.flat_cells = flat_cells

        # find all the single cells
        self.single_cells = [cc for c in classes for cc in c.cells if cc not in flat_cells]

        ## Create constraints to return
        constr = []

        # if the system is in a split context, add the constraints for the rate of each class
        if self.by_split:
            for c in self.classes:
                constr.append(c.lam >= self.lam * c.X)

        for fc in self.feeders:
            constr.extend(fc)  # feeders QNACells & aux constraints
            constr.append(fc.lam == fc.parent_class.lam)
            
            # # NEW: in split mode, for NON-SHARED feeder lines, force exact qty scaling
            # if self.by_split and self.feeders:
            #     # count how many wrappers reference each physical feeder line
            #     _line_use = {}
            #     for _fc in self.feeders:
            #         _ln = getattr(_fc, "line", None)
            #         if _ln:
            #             _line_use[id(_ln)] = _line_use.get(id(_ln), 0) + 1

            #     for _fc in self.feeders:
            #         _ln = getattr(_fc, "line", None)
            #         _pc = getattr(_fc, "parent_class", None)
            #         if not (_ln and _pc):
            #             continue

            #         # only handle NON-SHARED feeder lines (used by exactly one product)
            #         if _line_use.get(id(_ln), 0) != 1:
            #             continue

            #         # quantities:
            #         q = getattr(_ln, "batch_qty", None)  # e.g. "FP010 Quantity // New Module"
            #         if q is None:
            #             continue

            #         try:
            #             # make qty dimensionless if it was created with unit "count"
            #             q_dimless = q / gpkit.ureg("count")
            #         except Exception:
            #             q_dimless = q  # already unitless

            #         # 1) Pin the feeder-line throughput to class rate × qty
            #         #    (units: [count²/h] if q carried "count", which matches FeederLine.lam)
            #         constr.append(_ln.lam >= _pc.lam * q)
            #         constr.append(_ln.lam <= _pc.lam * q)

            #         # 2) Pin the terminal feeder cell to class rate × qty (dimensionless)
            #         #    (units: [count/h], matches the QNACell lambda)
            #         if getattr(_ln, "cells", None):
            #             _last = _ln.cells[-1]
            #             constr.append(_last.lam >= _pc.lam * q_dimless)
            #             constr.append(_last.lam <= _pc.lam * q_dimless)

        # return full constraints for single cells
        constr.extend(self.single_cells)

        # extend constraints by the classes and their cells
        constr.extend(self.classes)
        constr.extend(self.mcells.values())

        return constr


class PhasedMCSystem(MCSystem):
    '''creates a multi-class system with phased product changeovers

    Attributes
    ----------
    changeover_method : str
        ('flow-out', 'progressive', 'best-case')
        the type of changeover 
    
    needs_localsolve : boolean (default=False)
        does the model need to be solved locally

    '''

    def setup(self, changeover='progressive'):
        ''' gpx setup

        Arguments
        ---------
        changeover : str (default='progressive')
            the type of change over
                'progressive'   : 
                'flow-out'      :
                'bestcase'      : choose the lowest cost between the options
        
        '''
        constr = super().setup()

        self.changeover_method = str(changeover).strip()

        #TODO:  find the constraint for progressive changeover

        self.tchangeover = Variable('t_{changeover}', 'min', 'total changeover time per period')
        self.nperiods = Variable('n_{periods}', 'count', 'number of production periods')

        #TODO:  create the cells from phased multiproduct cells
        #TODO:  pull in the lines for each product
        #TODO:  non-recurring cell cost by max m