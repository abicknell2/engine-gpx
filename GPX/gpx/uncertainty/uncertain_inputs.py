'''Evaluate a model using uncertain input parameters

Example Use
-----------
- Import inputs and create uncertain variable objects
- Substitute "likely" value into the model
- (Solve best and worst cases)
- Find the risks based on likely sensitivity
    - Find risks based on best-likely-worst average sensitivity
- Generate samples from uncertain inputs (10 sample sets)
- Iterate through GP solves
- Bootstrap results to get confidence bounds


(Worst Case) -----|   (Likely-Case CI)   |----- (Best Case)
'''

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from gpkit import Variable
import numpy as np

import gpx.uncertainty.distributions


class UncertainInput(object):
    '''An uncertain input tied to a variable
    
    Attributes
    ----------
    var : gpx.Variable
        the model variable corresponding to the uncertainty
    uncertainty : distributions.UncertainDistribution
        probability function representing the uncertain distribution
    name : string
        a descriptive name
    sens : float
        the sensitivity of the gpx parent model to the variable
    min : float
        the minimum value of the uncertainty
    max : float
        the maximum value of the uncertainty
    likely : float
        the likely value of the uncertainty
    
    '''

    def __init__(self, var, uncertainty, name):
        '''Create an uncertain input based on a variable in an optimization

        Args
        ---
        var : GPkit variable
           the variable to link the uncertainty to

        uncertainty : distributions.UncertainDistribution
            representation of the uncertain input

        name : string
            a descriptive name
        '''
        self.var = var
        self.uncertainty = uncertainty
        self.name = name

        self.sens = None

    def create_from_params(self, var, params, name, distname='', **kwargs):
        '''create the uncertain variable directly without passing an uncertain dist

        Args
        ---
        var : gpx.Variable or Varkey
            the gpx variable which corresponds to the uncertain input
        params : list
            list of the parameters for making the distribution
            (see distributions.avail_distributions)
        distname : string
            the type of distribution 
            must be selected from distributions.avail_dist
            {
                'Beta Three Point' : BetaThreePoint,
                'Uniform' : UniformDistribtuion,
                'Three Point' : ThreePointVariable,
            }
        '''

        if distname not in gpx.uncertainty.distributions.avail_distributions:
            raise KeyError('Distribution not found %s' % distname)

        self.__init__(
            var,
            gpx.uncertainty.distributions.avail_distributions[distname](params, **kwargs),
            name,
        )

    def quantify_total_risk(self, solution, measure='gross'):
        '''quantify the risk of uncertainty to the solution

        Args
        ----
        solution : GPkit model solution
            a solution to a GP model

        measure : 'gross', 'net'
            how to calculate the total risk exposure
        '''
        sens = solution['sensitivities']['constants'][self.var]
        cost = solution['cost']

        risk_profile = self.uncertainty.risk_profile()
        if measure == 'gross':
            return np.abs(sens * risk_profile) * cost
        if measure == 'net':
            return sens * risk_profile * cost

    def sample(self, points=10):
        return self.uncertainty.sample(points)

    def set_base_sens(self, base_senss):
        '''set the base sensitivity of the variable
           (used mostly for best-worst from min-max calculation)
        
        Arguments
        ---------
        base_senss : keydict
            the entire sensitivities from solving the problem
        '''
        self.sens = base_senss[self.var]

    def get_best_case(self):
        'get the best case based off of base_sens'
        if self.sens > 0:
            return self.uncertainty.left
        else:
            return self.uncertainty.right

    def get_worst_case(self):
        'get the worst case from min-max based off of base_sens'
        if self.sens > 0:
            return self.uncertainty.right
        else:
            return self.uncertainty.left

    def subs_as_tuple(self, sens=None):
        '''creates the substitutions as a tuple for solving models

        Arguments
        ---------
        sens : keyDict
            sensitivities based on initial solve
            used to convert from min-max to best-worst

        Returns
        -------
        (best, likely, worst)
        '''
        #TODO: if the variable is only defined with min, max, use the sens to define best and worst
        if sens is not None:
            self.set_base_sens(sens)

        return (self.get_best_case(), self.uncertainty.likely, self.get_worst_case())

        # return (self.uncertainty.best, self.uncertainty.likely, self.uncertainty.worst)

    @property
    def units(self, replace_usd=False):
        if replace_usd:
            return '$' if 'USD' in str(self.var.units) else ''.join(str(self.var.units).split()[1:])
        else:
            return ''.join(str(self.var.units).split()[1:])


class UncertainModel(object):
    '''a gpx model with uncertain inputs

    Items
    -----
    gpmodel : gpkit.Model
    uncertainvars : list of UncertainInput
        the uncertain variables
    scenarios : list
        the list of solves with the substituted variables
    base_senss : keydict
        base-case sensitivities used for auto best-worst
    '''

    def __init__(self, gpmodel, *uvars, base_senss=None, bootstrap=True, samples=10, **kwargs):
        self.gpmodel = gpmodel
        self.uncertainvars = uvars
        self.is_boostrapped = bootstrap
        self.bootstrap_samples = samples
        self.base_senss = base_senss

    def gen_scenarios(self):
        ''' generate the sampled scenarios for the uncertain inputs'''
        uvars = {uv.var.key: np.array(uv.sample()) for uv in self.uncertainvars}
        self.scenarios = sweep_samples(self.gpmodel, uvars)

    def boostrap_variable(self, *varkeys, gethilo=False, stat=bs_stats.mean):
        '''bootstrap the expect for the list of vars

        Arguments
        ---------
        *varkeys : UncertainInput
            variables to return bootstrapped data for
        gethilo : boolean
            also return the best and worst case results

        Returns
        -------
        '''
        if not hasattr(self, 'scenarios') and self.is_boostrapped:
            self.gen_scenarios()

        results = {}
        for vk in varkeys:
            # bootstrap the expectation
            if self.is_boostrapped:
                sample = [sc['variables'][vk.key] for sc in self.scenarios]
                boot = bs.bootstrap(np.array(sample), stat_func=stat)
            #     bootresults = [
            #         boot.lower_bound,
            #         boot.value,
            #         boot.upper_bound,
            #     ]
            # else:
            #     bootresults = [0,0,0]

            if gethilo:
                cases = [
                    self.get_worst_case()['variables'][vk.key],
                    self.get_best_case()['variables'][vk.key],
                ]
                if self.is_boostrapped:
                    res = [
                        np.min(cases),
                        boot.lower_bound,
                        boot.value,
                        boot.upper_bound,
                        np.max(cases),
                    ]
                else:
                    res = [0, 0, 0, 0, 0]
            else:
                if self.is_boostrapped:
                    res = [
                        boot.lower_bound,
                        boot.value,
                        boot.upper_bound,
                    ]
                else:
                    res = [0, 0, 0, 0, 0]
            results[vk] = [np.float64(val) for val in res]
        return results

    def risk_eval(self, *uvars, measure='gross', abs_sum=True, sum_risk=True, include_points=False, points_as_obj=True):
        'evalute all risks'
        # if no uvars, return all
        if len(uvars) == 0:
            uvars = self.uncertainvars

        risks = {}
        for uv in uvars:
            row = {}
            risk = uv.quantify_total_risk(self.get_likely_case(), measure=measure)
            if sum_risk:
                if abs_sum:
                    row['sumRisk'] = np.float64(np.sum(np.abs(risk)).magnitude
                                                ) if hasattr(risk, 'magnitude') else np.float64(np.sum(np.abs(risk)))
                else:
                    row['sumRisk'] = np.float64(np.sum(risk).magnitude
                                                ) if hasattr(risk, 'magnitude') else np.float64(np.sum(risk))

            if include_points:
                row['riskPoints'] = np.float64(risk).tolist()
                row['x'] = np.float64(uv.uncertainty.x)
                if points_as_obj:
                    row['riskPoints'] = [{'value': x, 'risk': y} for x, y in zip(row['x'], row['riskPoints'])]
                    del row['x']
            row['riskUnit'] = ''.join(str(np.sum(risk)).split()[1:])
            row['varUnit'] = uv.units
            risks[uv.name] = row

        return risks

        # return {uv.name : {'sumRisk' : np.sum(uv.quantify_total_risk(self.get_likely_case(), measure=measure) if sum_risk else None,
        #                    'riskPoints' : uv.quantify_total_risk(self.get_likely_case(), measure=measure) if include_points else []}
        #        for uv in uvars}

        # old implementation
        # if sum_risk == True:
        #     return {uv.name : np.sum(uv.quantify_total_risk(self.get_likely_case(), measure=measure))
        #             for uv in uvars}
        #
        # return {uv.name : uv.quantify_total_risk(self.get_likely_case(), measure=measure)
        #         for uv in uvars}

    def solve_case(self, case='likely', **kwargs):
        '''
        Arguments
        ---------
        case : string
            'best'
            'worst'
            'likely'
        '''
        # selected the case
        case_idx = None

        if case == 'best':
            case_idx = 0
        elif case == 'likely':
            case_idx = 1
        elif case == 'worst':
            case_idx = 2
        else:
            raise Exception('Solution case not found: ' + str(case))

        # update the substitutions based on the case
        subs = {v.var: v.subs_as_tuple(sens=self.base_senss)[case_idx] for v in self.uncertainvars}
        # pass subsitutions to the model
        # self.gpmodel.substitutions.update(subs)
        self.gpmodel.update_substitutions(subs)
        # solve the model for the selected case
        return self.gpmodel.solve(**kwargs)

    def get_best_case(self, **kwargs):
        'get the best case solution'
        try:
            return self._best_case
        except AttributeError:
            self._best_case = self.solve_case(case='best', **kwargs)
            return self._best_case

    def get_worst_case(self, **kwargs):
        try:
            return self._worst_case
        except AttributeError:
            self._worst_case = self.solve_case(case='worst', **kwargs)
            return self._worst_case

    def get_likely_case(self, **kwargs):
        try:
            return self._likely_case
        except AttributeError:
            self._likely_case = self.solve_case(case='likely', **kwargs)
            return self._likely_case


def sweep_samples(model, sweepvars, samples=10):
    '''sweep samples instead of a multi-dimensional sweep

    Arguments
    ---------
    model : gpkit.Model
        the base model
    sweepvars : dict
        varkeys : list of samples
    samples : int
        the number of samples

    Returns
    -------
    list
        a list of the sampled solutions
    '''
    sampled_solutions = []
    for i in range(samples):
        subs = {varkey: samp[i] for varkey, samp in sweepvars.items()}
        model.substitutions.update(subs)
        sampled_solutions.append(model.solve())

    return sampled_solutions
