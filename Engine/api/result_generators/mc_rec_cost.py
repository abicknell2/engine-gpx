import logging

import numpy as np

from api.constants import COST_ROUND_DEC
from api.result_generators.result_gens import ResultGenerator
from utils.settings import Settings


class MCRecCost(ResultGenerator):

    def __init__(
        self, gpxsol, settings: Settings, module, extant_costs=[], cell_costs=None, floorspace_costs=None, **kwargs
    ):
        'recurring cost results for a multiproduct system'
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        self.resultsname = 'recurringCosts'

        new_costs = []

        # get the current recurring cost resutls
        new_costs.extend(extant_costs)

        units = 'USD/hr'

        # cell costs
        for name, c in module.gpxObject['sharedCellCosts'].items():
            if c.recurringCost != 0:
                new_costs.append({
                    'name': f'{name} Recurring Cost',
                    'value': np.round(gpxsol(c.recurringCost).to(units).magnitude, decimals=COST_ROUND_DEC),
                    'unit': units,
                })

        # check for floor space
        #FUTURE:    use walrus operator
        # if thing := module.gpxObject.get('floorspaceCosts'):
        fs = module.gpxObject.get('floorspaceCost')
        if fs and fs.recurringCost != 0:
            new_costs.append({
                'name': 'Total Floor Space Recurring Cost',
                'value': np.round(gpxsol(fs.recurringCost).to(units).magnitude, decimals=COST_ROUND_DEC),
                'unit': units
            })

        # add to the solution
        self.results[self.resultsname] = new_costs


class MCNonRecCost(ResultGenerator):

    def __init__(self, gpxsol, settings: str, module, **kwargs):
        'non-recurring cost results for a multiproduct system'
        super().__init__(gpxsol=gpxsol, settings=settings, **kwargs)

        self.resultsname = 'costComponents'
        self.results[self.resultsname] = []
        entries = []

        # check for cell costs
        for name, c in module.gpxObject['sharedCellCosts'].items():
            try:
                entries.append([name, gpxsol['variables'][c.nonrecurringCost]])
            except KeyError:
                logging.warn('RESULT-GENS Multiclass NonRecurring Cost| Error {}'.format(name))

        # check for tool cost
        mod = module.gpxObject['toolCosts']
        if len(mod) > 0:
            for name, t in mod.items():
                entries.append([name, gpxsol['variables'][t.nonrecurringCost]])

        # floorspace
        fs = module.gpxObject.get('floorspaceCost')
        if fs and fs.nonrecurringCost != 0:
            entries.append(['Total Floor Space Non-Recurring Cost', gpxsol['variables'][fs.nonrecurringCost]])

        # format the results
        for e in entries:
            self.results[self.resultsname].append({
                'name': e[0],
                'value': np.round(e[1], decimals=COST_ROUND_DEC),
            })

        # create the results index enry
        self.results_index.append({
            'name': 'Capital Cost Components',
            'value': self.resultsname,
        })
