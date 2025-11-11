'cells that can operate on alternative shift schedules'

from gpkit import Model, Variable

from gpx.manufacturing import QNACell
from gpx.primitives import Process


class OffShiftCell(QNACell):
    '''cell with off-shift operations


    
    '''

    def setup(self, source_cell, ton, toff, exit_cell=None, return_process=None):
        '''[summary]

        Parameters
        ----------
        source_cell : gpx.manufacturing.QNACell
            cell to make into the offshift cell.
            replaces a QNA cell direcly in a CONWIP system
        exit_cell : gpx.manufacturing.QNACell or None
            if `None` will assume the source_cell is the last
        ton : [type]
            [description]
        toff : [type]
            [description]
        return_process : [type], optional (default=None)
            [description], by default None

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        ValueError
            [description]
        '''
        constr = []

        self._test_attr = 'hello world'

        self.source_cell = source_cell
        self.exit_cell = exit_cell

        # update the attributes of the off-shift cell to the source cell
        self.__dict__.update(self.source_cell.__dict__)

        if not source_cell.allows_queueing or (exit_cell and not exit_cell.allows_queueing):
            raise ValueError('Off-shift cell and exit cell must allow queueing to operate off-shift')

        self.ton = ton
        self.toff = toff

        self.timeratio = timeratio = Variable(
            '\\frac{T_{on} + T_{off}}{T_{on}}', lambda m: (m[ton] + m[toff]) / m[ton], '-',
            'production ratio for offshift'
        )
        self.ptoff = ptoff = Variable(
            # '\\left( \\frac{T_{off}}{T_{on} + T_{off}} \\right)',
            '\\text{Prob}[T_{off}]',
            lambda m: m[toff] / (m[ton] + m[toff]),
            '-',
            'probability of ending up off shift',
        )

        # Margin Variables
        self.chicv = chicv = Variable('\\chi_{cv}', 1, '-', 'margin sensitivity to process variation')
        self.nu = nu = Variable('\\eta_t', 1, '-', 'process time efficiency')

        constr.extend([1 >= self.alpha + self.rho])  # reuse standrad constraints
        constr.extend([  # cv calculations
            self.c2d >= self.alpha * self.c2a + self.rho * self.c2s,
        ])

        constr.extend([
            self.c2s == self.process.cv**2 / timeratio
            * self.chicv**2,  # adjust for the aggregating effects of the off-shift work
            self.lam
            <= self._get_lam_basis() * timeratio,  # adjust the average rate to account for off-shift production
            self.W >= self._get_w_basis(),  # get the flow time
            self.tnu >= self.process.t * self.nu,  # get the processing time
        ])

        if exit_cell is None:
            # source_cell is the last in the line
            # put all the extra queueing before the source cell
            constr.extend([
                self.Wq >= source_cell._get_wq_basis() + toff*ptoff,  # queueing time with off-shift buildup
            ])
        else:
            constr.extend([
                self.Wq >= source_cell._get_wq_basis() + toff*ptoff/2.0,  # queueing time with off-shift buildup
                self.exit_cell.Wq >= self.exit_cell._get_wq_basis()
                + toff*ptoff/2.0,  # add the constraints on the queueing time for the exit cell
            ])

        if return_process is not None:
            self.return_process = return_process

        if self.return_process:
            constr.append(self.process)

        return constr

    def get_line_constraints(self, line):
        'get the constraints on the line'
        return [
            line.L >= self._get_L_off_basis(),  # has to be enough inventory to support off-shift production
            line.L >= self._get_L_on_basis(),  # sustain produciton on shift
        ]

    def _get_L_off_basis(self):
        return self.m / self.tnu * self.toff

    def _get_L_on_basis(self):
        return self.m / self.tnu * self.ton
