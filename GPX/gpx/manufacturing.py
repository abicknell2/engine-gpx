'gpx manufacturing models'

import numbers

# from helpers import optional_variable
import gpkit
from gpkit import Variable, parse_variables
import numpy as np

from gpx import Model
from gpx.helpers import optional_variable
from gpx.primitives import Process
from gpx.unit_helpers import is_continuous_physical_quantity


class QNACell(Model):
    '''cell model for the Queueing Network Analyzer

    Calculated Variables
    --------------------
    m      [count]    Number of parallel workstations
    c2a    [-]        Arrival coefficient of variation squared
    c2s    [-]        Process coefficient of variation squared
    c2d    [-]        Departure coefficient of variation squared
    rho    [-]        Cell utilization
    mu     [count/hr] Cell isolated production rate
    W      [min]      Total flow time through cell
    Wq     [min]      Expected queueing time
    alpha  [-]        (1-rho)
    CV     [-]        Process coefficient of variation
    lam    [count/hr] Production rate
    tnu    [min]      Average cycle time including efficiency

    Margin Analysis Variables
    -------------------------
    chicv    1.0    [-]    Margin sensitivity to cv
    nu       1.0    [-]    Margin sensitivity to cycle time

    Availability
    ------------
    e    [-]    Availability of the cell due to maintenance, etc.

    Note: Process time is the predicted champion time from the process models.
          The cell cycle time takes into account the efficiency factor to get
          the average process time
          The process time takes into account some non-value-add time

    Note: The efficiency also gives the sensitivity to the calculated process times
          (which wouldn't normally have a sensitivity)

    '''

    def setup(
        self,
        process=Process(),
        nu=1,
        e=1,
        capacity=1,  # inputs
        m=None,  # optimization overrides
        queueing_at_cell=True,
        add_aux_time=True,
        return_process=False,  # modeling controls
        rebatch_from_previous=None,
        secondary_cell=None,  # modeling controls
    ):
        '''setup the equations

        Args
        ----
        process : primitives.Process
            the process which is executed in the cell
        nu : variable, constant, monomial
            Default=1
            the "efficiency" of the process from champion time to expected cycle time
        capacity : variable, constant, monomial
            default=1
            the parallel capacity of the cell.
            modeled as if the process
        queueing_at_cell : boolean (default=True)
            allows for queuing before the cell
        add_aux_time : boolean
            add a variable for auxiliary time
        aux_time : list
            a list of auxiliary time variables to add
        '''

        # Calculated Variables
        if m is None:
            m = self.m = Variable('m', 'count', 'Number of parallel workstations')
        else:
            self.m = m  # use the input variable

        c2a = self.c2a = Variable('c2a', '-', 'Arrival coefficient of variation squared')
        c2s = self.c2s = Variable('c2s', '-', 'Process coefficient of variation squared')
        c2d = self.c2d = Variable('c2d', '-', 'Departure coefficient of variation squared')
        rho = self.rho = Variable('rho', '-', 'Cell utilization')
        # mu = self.mu = Variable('mu', 'count/hr', 'Cell isolated production rate')
        W = self.W = Variable('W', 'min', 'Total flow time through cell')
        Wq = self.Wq = Variable('Wq', 'min', 'Expected queueing time')
        alpha = self.alpha = Variable('\\alpha', '-', '(1-rho)')
        # CV = self.CV = Variable('CV', '-', 'Process coefficient of variation')

        lam_units = self._lam_units(capacity=capacity)
        lam = self.lam = Variable('\\lambda', lam_units, 'Production rate')
        tnu = self.tnu = Variable('t_{\\eta}', 'min', 'Average cycle time including efficiency')

        # don't want to display these variables
        self.alpha.private = True
        self.tnu.private = True

        # queueing information
        self.queue_at_cell = queueing_at_cell
        self.return_process = return_process

        self.secondary_cell = secondary_cell
        self.rebatch_from_previous = rebatch_from_previous

        # Margin Analysis variables
        self.chicv = chicv = Variable('\\chi_{cv}', 1, '-', 'margin sensitivity to process variation')
        # self.nu = nu = optional_variable(nu, variablename='\\eta_t')

        # check to see if parameter nu is a number
        self.nu = nu = Variable('\\eta_t', nu, '-', 'process time efficiency')
        self.e = e

        self.chicv.private = True
        self.nu.private = True

        # Process references
        self.process = process
        t = self.t = process.t
        cvp = self.cvp = process.cv
        k = self.k = capacity

        ## make the subsets of the constriants

        self._constr_cv = [
            c2d >= alpha*c2a + rho*c2s,
            c2s >= cvp**2 * chicv**2,  # margin analysis on variation
        ]
        self._constr_rho = [1 >= alpha + rho]  # calculate 1-rho
        self._constr_wq = [Wq >= self._get_wq_basis()]  # get the expected ququeing time
        # self._constr_lam = [lam <= rho*m/tnu*k]  # throughput and utilization
        self._constr_lam = [lam <= self._get_lam_basis()]

        self.constraints = [
            *self._constr_cv,
            *self._constr_rho,
            *self._constr_wq,
            *self._constr_lam,
        ]

        # self.constraints = [
        #     # W >= Wq + tnu,                          # Total flow time
        #     Wq >= (rho/alpha)*(c2a+c2s)/2*tnu/m/k,  # Kingmann approximation
        #     # tnu >= t*nu,                            # Process efficiency
        #     c2d >= alpha*c2a + rho*c2s,             # calculate departure cv
        #     1 >= alpha + rho,                       # calculate 1-rho
        #     lam <= rho*m/tnu*k,                       # throughput and utilization
        #     c2s >= cvp**2*chicv**2,                 # margin analysis on variation
        # ]

        # Queueing Policy
        self.allows_queueing = queueing_at_cell

        if queueing_at_cell:
            self.constraints.append(
                W >= Wq + tnu,  # Total flow time includes queueing
            )
        else:
            self.constraints.append(
                W >= tnu,  # Total flow time comprises only process time
            )

        # Construct auxiliary time
        self.aux_time = []
        self.constraints.append(tnu >= t * nu * e)
        if add_aux_time:
            # construct the variable for aux time
            self.taux = Variable('t_{auxiliary}', 'min', 'Additional in cell time')
            # self.constraints.append(tnu >= t*nu + self.taux)    # add the auxiliary variable to the process time
        else:
            # no auxiliary time is defined
            # self.constraints.append(tnu >= t*nu)    # just get the process time
            pass

        if return_process:
            self.constraints.append(self.process)

        return self.constraints

    def _lam_units(self, capacity) -> str:
        """Return the units string for production rate (lambda)"""
        return "count/hr"

    def _get_lam_basis(self):
        'the basis rate calculation'
        return self.rho * self.m / self.tnu * self.k

    def _get_wq_basis(self):
        'get the kingmann approximation for the cell queueing time'
        return self.rho / self.alpha * (self.c2a + self.c2s) / 2 * self.tnu / self.m / self.k

    def _get_w_basis(self):
        'get the flow time based on the queueing policy'
        if self.allows_queueing:
            return self.Wq + self.tnu
        else:
            return self.tnu

    def _get_rebatching(self, lam=None):
        'gets the additional constraint on queueing time driven by rebatching'
        # allows for a different lambda to be used in the constraint
        # defaults to the cell rate
        lam = lam or self.lam
        # return self.Wq >= self.k/lam/2
        return self.Wq >= self.k / (2 * lam)

    def disable_rate_constraint(self) -> bool:
        """Temporarily remove the rate constraint used for continuous solves."""

        if not getattr(self, "_constr_lam", None):
            return False

        if getattr(self, "_rate_constraint_backup", None):
            return False

        backup = list(self._constr_lam)
        if not backup:
            return False

        active = False
        for constraint in backup:
            try:
                # remove the constraint from the active set for the discrete attempt
                self.constraints.remove(constraint)
                active = True
            except ValueError:
                continue

        if not active:
            return False

        # remember the removed constraints so they can be restored later
        self._rate_constraint_backup = list(backup)
        self._constr_lam = []
        return True

    def restore_rate_constraint(self) -> None:
        """Reinstall the rate constraint if it was temporarily removed."""

        backup = getattr(self, "_rate_constraint_backup", None)
        if not backup:
            return

        for constraint in backup:
            if constraint not in self.constraints:
                # add the saved constraint back into the model
                self.constraints.append(constraint)
        self._constr_lam = list(backup)
        self._rate_constraint_backup = None


class FeederQNACell(QNACell):
    """
    QNACell variant that supports physical capacity units (e.g. kg/hr),
    bypassing normalization math designed for 'count'.
    """

    def _get_wq_basis(self):
        # Override to skip division by capacity with units like kg, L, etc.
        return self.rho / self.alpha * (self.c2a + self.c2s) / 2 * self.tnu / self.m

    def _lam_units(self, capacity):
        "Overload default behavior to capture continuous production units"
        try:
            cap_units = capacity.units
            if is_continuous_physical_quantity(cap_units):
                # get the physical units per unit time
                lam_units = capacity.units / gpkit.units.hour.units
                return str(lam_units.units)

            # if cap_units is None:   # TODO: catch if units 'count'
            # unitless or count
            return super()._lam_units(capacity)

        except AttributeError:
            return super()._lam_units(capacity)


class FabLine(Model):
    '''
    A CONWIP line approximated using the Queueing Network Analyzer

    '''

    def setup(
        self,
        cells: list[QNACell],
        return_cells: bool = True,
        total_queue_wip: Variable | None = None,
        total_queue_time: Variable | None = None,
    ):
        """
        Build GPkit model of a CONWIP line from an ordered sequence of cells.

        Parameters
        ----------
        cells : list of QNACell
            The primary flow cells in the line.
        return_cells : bool, optional
            If True, include the individual cells as constraints in the model output.
        total_queue_wip : Variable or None, optional
            If provided, a constraint is added to bind total queue WIP to this variable.
        total_queue_time : Variable or None, optional
            If provided, a constraint is added to bind total queueing time to this variable.

        Returns
        -------
        list
            A list of constraints for the GPkit model.
        """

        self.cells = cells

        lam = self.lam = Variable('\\lambda', 'count/hr', 'Line Continuous Production Rate')
        W = self.W = Variable('W', 'hr', 'Total flow time')
        L = self.L = Variable('L', 'count', 'Total WIP count')

        # Littles Law on Entire Line
        # previously, this was L >= lam*W
        constraints = [
            L == lam * W,
        ]

        # Connect Variations
        for i, j in enumerate(cells):
            if i > 0:
                constraints.append(cells[i].c2a >= cells[i - 1].c2d)
        # Loop first and last cells
        constraints.append(cells[-1].c2d <= cells[0].c2a)

        # Production rate has to be same everywhere
        constraints.extend([lam == cell.lam for cell in cells])

        # map cell blocking along the line
        # cell blocking wouldn't happen from the last cell into the first cell
        # this is just the normal CONWIP signal
        for i, c in enumerate(cells):
            if i != 0:
                # ignore the first cell
                if not c.allows_queueing:
                    # put queueing time in previous cell
                    cells[i - 1].aux_time.append(c.Wq)

        # sum auxiliary variables for applicable cells
        for c in cells:
            # for each cell sum the aux variables
            if not c.aux_time:
                # the array is empty so replace aux time with 0
                c.taux = 0
                # do not add additional constraint
            else:
                # extend the constraints
                constraints.extend([
                    c.taux >= np.sum(c.aux_time),  # sum the aux times
                    c.tnu >= c.t * c.nu + c.taux,  # additional constraint
                ])

        # Find total flow time
        constraints.append(W >= np.array([cell.W for cell in cells]).sum(), )

        # check for total queue variables
        if total_queue_time is not None:
            self.total_queue_time = optional_variable(
                total_queue_time,
                "W_{queue,total}",
                "hr",
                "Total queueing time for the line",
            )
            constraints.append(self.total_queue_time >= self._get_constr_queue_time())
            constraints.append(total_queue_time >= self.total_queue_time)
        else:
            self.total_queue_time = None

        if total_queue_wip is not None:
            self.total_queue_wip = optional_variable(
                total_queue_wip,
                "L_{queue,total}",
                "count",
                "Total queue inventory for the line",
            )
            constraints.append(self.total_queue_wip >= self._get_constr_queue_wip())
            constraints.append(total_queue_wip >= self.total_queue_wip)
        else:
            self.total_queue_wip = None

        # return the cell constraints if optioned
        if return_cells:
            constraints.extend(self.cells)

        return constraints

    def _get_constr_queue_time(self) -> gpkit.Posynomial:
        """creates the constraint for the total time spent in queues for a part in the system"""
        # sum the queueing time for all the cells
        return np.sum([c.Wq for c in self.cells])

    def _get_constr_queue_wip(self) -> gpkit.Posynomial:
        """creates the constraint for the total inventory that is in queues for the system"""
        # the sum of the queueing time divided by the rate of the product will give the queue quantity
        # this is according to little's law
        return self._get_constr_queue_time() * self.lam


class ConwipTooling(Model):
    'closes the loop on tooling'

    def setup(self, cells, capacity=1, starting_cell=1, ending_cell=0, indicies=None, loop_tool=True, idx=1):
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

        if indicies is not None:
            cellrange = [i - idx for i in indicies]  # assign the cell ranges to the indicies themselves with offset
            # set start and end values
            start0 = indicies[0] - idx
            end0 = indicies[-1] - idx

        else:  # there are no indicies defined
            if ending_cell == 0:
                ending_cell = len(cells)
            start0 = starting_cell - idx
            end0 = ending_cell - idx
            cellrange = range(start0, end0 + 1)

        L = self.L = Variable('L_{tool}', 'count', 'required tool count')

        if capacity is None:
            k = self.k = Variable('k_{capacity}', 'count', 'tool carrier capacity')
        else:
            k = self.k = capacity

        if loop_tool:  # loop the tool around from the last cell to the first
            constraints.append(cells[start0].c2a >= cells[end0].c2d)  # constrain variation based on the tool loop

        # constrain the tool count by adding all the wip in the cells
        constraints.append(L * k >= np.sum([cells[i].lam * cells[i].W for i in cellrange]))

        # add start and end cells to class attributes
        self.cell_start = cells[start0]
        self.cell_end = cells[end0]
        self.cell_range = [cells[i] for i in cellrange]

        return constraints


class CellSecondaryFlow(Model):
    '''cell secondary flow models

    Adds additional flows
    '''

    def setup(self, cells, flows):
        '''gpx setup

        Args
        ----
        cells : OrderedDict
            the cells in the primary flow
        flows : list of tuples
            describes the secondary flows as (arrival cell, departure cell)

        '''