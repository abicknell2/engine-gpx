'cell definition for a multi-class context'

import gpkit
from gpkit import Model, Variable
import numpy as np


class MCell(Model):

    def setup(self, cells=[], return_processes=True, by_split=False, has_changeover=False, t_changeover=None, **kwargs):
        '''
        Keyword Arguments
        -----------------
        classcells : OrderedDict of {class : cell}
            can ask the
            QUESTION: does this need to be an ordered dict?

        cells : list of gpx.manufacturing.QNACell

        return_processes : boolean (default=True)
            return the constraints from the processes as part of the cell constraint set

        by_split : boolean (default=False)
            the split parameter is defined directly (rather than inferred from production rates)
            this means the splits at the cell can be directly computed from the rate splits
            but we have to know them in advance

        has_changeover : boolean (default=False)
            flag if the cell has a changeover time between dissimilar products

        t_changeover : None, gpkit.Variable, gpkit.Monomial, gpkit.Posynomial
            if the shared resource has changeover times
            if None, then the a free variable will be declared

        
        Variables
        ---------
        m    [count]    workstation count
        c2d  [-]        Departure coefficient of variation

        Optional Variables
        ------------------
        t_change    [min]    Changeover time from one product to another
        
        
        Aggregate Performance Variables
        -------------------------------
        Wq_bar    [min]          queueing time
        Lq_bar    [count]        customers in queue
        rho_bar   [-]            Utilization
        alpha_bar [-]            (1-rho)
        ts_bar    [min]          Service time
        c2a_bar   [-]            Arrival coefficient of variation
        c2s_bar   [-]            Service coefficient of variation
        
        Calculated Variables
        --------------------
        lam_bar   [count/min]    production rate
        
        Variables Assigned to Single-Class Cells
        ----------------------------------------
        x        [-]      Fraction of parts in the multi-class cell
        compx    [-]      The (1-x) compliment to x
        W        [min]    Total flow time
        
        Vector (by class) Variables
        ---------------------------        
        W      [min]          Total flow time
        c2a    [-]            Arrival squared coefficient of variation
        c2s    [-]            Service squared coefficient of variation
        L      [count]        Average customers in the queue and in service
        '''

        # Variables
        m = self.m = Variable('m', 'count', 'workstation count')
        # m_min = self.m_min = Variable("m_min", 1e-12, "-", "Lower bound for effective workstation count")

        # Aggregate Variables
        Wq_bar = self.Wq_bar = Variable('\\bar{W_q}', 'min', 'queueing time')
        ts_bar = self.ts_bar = Variable('\\bar{t_s}', 'min', 'service time')
        lam_bar = self.lam_bar = Variable('\\bar{\\lambda}', 'count/hr', 'total production rate')
        rho_bar = self.rho_bar = Variable('\\bar{\\rho}', '-', 'utilization')
        alpha_bar = self.alpha_bar = Variable('\\bar{\\alpha}', '-', '1-rho')
        c2a_bar = self.c2a_bar = Variable('\\bar{c^2_a}', '-', 'Arrival coefficient of variation squared')
        c2s_bar = self.c2s_bar = Variable('\\bar{c^2_s}', '-', 'Service coefficient of variation')
        c2d_bar = self.c2d_bar = Variable('\\bar{c^2_d}', '-', 'Departure coefficient of variation')
        eta_t = self.eta_t = Variable('\\eta_{t}', 1, '-', 'Margin to total process time')

        # self.alpha_min = Variable('\\alpha_{min}', 1e-3, '-')
        self.x_min = Variable('x_{min}', 1e-6, '-', 'min arrival mix to avoid ts→0')
        self.ts_min = Variable('t_{s,min}', 1e-6, 'min', 'min service time to avoid division by zero')        
        
        # Calculated Variables
        self.cells = cells

        # create the changeover time variable
        if has_changeover:
            if t_changeover is None:
                # if none, then create a new variable
                self.t_change = t_change = Variable('t_{changeover', 'min', 'shared cell changeover time')
            else:
                # substitute the argument directly if not None
                self.t_change = t_change = t_changeover

        else:
            # if there is no changeover time, use 0
            t_change = 0

        # Find the number of incoming streams
        self.stream_count = N = len(cells)

        if N == 0:
            raise ValueError("MCell needs at least one QNACell")

        ## Create constraints to return as the model
        constr = []
        
        self.x = []
        self.compx = []
        self.xinv = []  # inventory split variable
        self.xitem = []  # item split variable
        # expose title-cased aliases for external helpers that expect either name
        self.xInv = self.xinv
        self.xItem = self.xitem

        constr += [self.m >= 1e-12]

        for i, c in enumerate(cells):
            xi = Variable(f"x_{i}", "-", "arrival mix parameter")
            self.x.append(xi)
            c.x = xi

            if not by_split:
                constr.append(xi >= self.x_min)
                constr.append(ts_bar >= self.ts_min)

            # create split variables that mirror the class-arrival split
            # build inventory split variables for each product stream
            c.xinv = xinv = Variable(f'xInv_{i}', '-', 'inventory split variable')
            self.xinv.append(xinv)
            constr.append(xinv == xi)

            # track the batch corrected split alongside the arrival mix
            c.xitem = xitem = Variable(f'xItem_{i}', '-', 'item-corrected split')
            self.xitem.append(xitem)
            constr.append(xitem == xi)

        # Calculated Aggregate Variables
        # constr.append(lam_bar >= np.sum([c.lam for c in cells]))   # cell rate exceeds required rate for class cells

        #TODO:  consider the capacity of the cell
        constr.append(  # TODO: similar constraint above could be redundant
            lam_bar >= np.sum([c.lam / c.k for c in cells])
        )  # cell rate exceeds required rate for class cells

        # constraints for aggregate variables
        constr.extend([
            ts_bar >= eta_t * np.sum([c.x * c.tnu for c in cells]),  # aggregate the service time
            c2a_bar >= np.sum([c.x * c.c2a for c in cells]),  # aggregate arrival behavior
            c2s_bar * ts_bar**2
            >= np.sum([c.x * c.c2s * c.tnu**2
                       for c in cells]),  # formulation using the weighted sum (seems to work with lam_bar as fixed var)
            # c2s_bar*ts_bar**2 == np.multiply(x,np.multiply(ts**2, c2s))**(1/n),  #TODO: check this formulation as using the geometric mean
        ])

        if len(cells) == 1:
            # this prevents adding in-cell buffer
            #TODO:  maybe allow this flexibility in other cells, too?
            constr.append(c2s_bar == cells[0].c2s)

        # add QNA constraints for the Aggregate Cell
        constr.extend([
            Wq_bar >= (rho_bar/alpha_bar) * (c2a_bar+c2s_bar) / 2 * ts_bar / m,
            1 >= alpha_bar + rho_bar,
            c2d_bar >= alpha_bar*c2a_bar + rho_bar*c2s_bar,
            lam_bar == rho_bar * m / ts_bar,
        ])

        # class cell constraints
        for c in cells:
            constr.extend([
                c.Wq >= Wq_bar,  # queueing time
                c.c2d >= c2d_bar,  # departure variation
                c.c2s >= c.process.cv**2,  # process SCV constraint
                # c.tnu >= c.t*c.nu,          # add the default calculation for in-cell time    # gets constrained later to include aux time
                self.m == c.m,  # link the shared cell count to the other workstation counts
            ])
            
            # constr.append(alpha_bar >= self.alpha_min)

            ## check to see if cell has queueing and adjust constraints
            aux_time = np.sum(c.aux_time)  # sum the aux times (if empty array, sums to 0)
            #TODO:  aux time should include any changeover times as well

            if has_changeover:
                # if there is changeover find the probability of the changeover
                xt_change = c.compx * t_change
            else:
                # if there is no changeover, just use any change directly
                xt_change = t_change

            aux_time = aux_time + xt_change

            if c.rebatch_from_previous:
                constr.append(c._get_rebatching())

            # add any auxiliary time to the process time to create new tnu
            constr.append(c.tnu >= c.t * c.nu + aux_time)

            if not c.secondary_cell:
                # secondary cells don't need these constraints
                if c.allows_queueing:
                    # add ququing time to the cell flow time
                    constr.append(c.W >= c.Wq + c.tnu)
                else:
                    # if queueing is not allowed at the cell, do not add to flow time
                    print(np.sum(c.aux_time))
                    constr.append(c.W >= c.tnu)

        return constr

    def get_constraint(self, var, bound='lower'):
        '''get a particular constraint that is not internally bounded by the QNA cell
        Should be used when formulating constraints elsewhere that will provide opposing pressure
        '''
        if var == self.W:
            # flow time
            return (self.W >= self.Wq_bar + self.ts)

        if var == self.L:
            # inventory
            pass
        if var == self.Lq:
            # inventory in queue
            pass


class MPhasedCell(Model):
    '''a multi-class cell with phased changeover

    inherits from MCell

    Variables
    ---------
    m    [count]    workstation count
    c2d  [-]        Departure coefficient of variation
    

    Optional Variables
    ------------------
    t_change    [min]    Chnageover time
    
    
    Aggregate Performance Variables
    -------------------------------
    Wq_bar    [min]          queueing time
    Lq_bar    [count]        customers in queue
    rho_bar   [-]            Utilization
    alpha_bar [-]            (1-rho)
    ts_bar    [min]          Service time
    c2a_bar   [-]            Arrival coefficient of variation
    c2s_bar   [-]            Service coefficient of variation
    p_change  [-]            Probability of changeover
    
    Calculated Variables
    --------------------
    lam_bar   [count/min]    production rate
    
    Variables Assigned to Single-Class Cells
    ----------------------------------------
    x    [-]      Fraction of parts in the multi-class cell
    W    [min]    Total flow time
    
    Vector (by class) Variables
    ---------------------------        
    W      [min]          Total flow time
    c2a    [-]            Arrival squared coefficient of variation
    c2s    [-]            Service squared coefficient of variation
    L      [count]        Average customers in the queue and in service
    '''

    def setup(self):
        constr = super().setup()

        #TODO:  cell workstation count (for non-recurring cost) is the max of the constituent cells

        # reset constraints
        constr = []

        # calculate total production rate
        constr.append(self.lam_bar >= np.sum([c.lam for c in self.cells]))

        pass