'serial processes'

import numpy as np

from gpx.primitives import Process


class Series(Process):
    '''Model for Processes in Series

    It itself is also a process

    Models
    ------
    t = \sum{t_{process}}


    '''

    def setup(self, *processes):
        process = super().setup()

        self.processes = processes
        return [
            process,
            self.t >= np.sum([p.t for p in processes]),
            self.stdev**2 >= np.sum([p.stdev**2 for p in processes]),
        ]
