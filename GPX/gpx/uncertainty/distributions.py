'''distributions to be used with uncertainty analysis'''

import numpy as np
from scipy import stats


class UncertainDistribution(object):
    '''
    Items
    -----
    left : float
        the minimum (left) point on the distribution
    right : float
        the maximum (right) end of the distribution
    params : list of float
        defines the inputs to the distributions

    dist : scipy.stats.rv_continuous
        the distribution describing the random variable
    '''

    def __init__(self, params, points=100, **kwargs):
        self.params = params

        # check to see that parameters are monotomically increasing
        if not all(params[i] <= params[i + 1] for i in range(len(params) - 1)):
            # if this is not true for all params, raise error
            raise ValueError('Parameters are not strictly increasing:', params)

        self.left = np.float64(params[0])
        self.right = np.float64(params[-1])

        # self.best = np.float64(best)
        # self.worst = np.float64(worst)

        # define standard samples
        self.x = np.linspace(self.left, self.right, points)

        self.dist = None

    def get_pdf(self, x, **kwargs):
        '''returns the pdf of the distribution'''
        x = np.float64(x)
        return self.dist.pdf(x)

    def sample(self, size=10):
        '''get random samples from the distribution'''
        return self.dist.rvs(size)

    def risk_profile(self, x=None):
        '''returns the likelihood and percent change from modal value'''

        if x is None:
            x = self.x

        y = self.get_pdf(x)
        liklihood = y / y.sum()  # scale the results from the PDF to 1
        percent_diff = self.percent_diff = (x - self.expectation()) / self.expectation()
        risk_profile = percent_diff * liklihood
        return risk_profile

    def expectation(self, ismean=True):
        'the expected value of the distribution'
        return self.dist.mean()

    @property
    def likely(self):
        'the expectation'
        return self.expectation()


class UniformDistribtuion(UncertainDistribution):
    '''An uncertain variable with two inputs to a uniform distribution
    '''

    def __init__(self, params, points=100, **kwargs):
        '''Creates a uniform distribution based on best- and worst-case scenarios

        Args
        ----
        params : list
            [min, max]
        '''
        super().__init__(params, points, **kwargs)

        self.scale = self.right - self.left
        self.loc = self.left

        self.dist = stats.uniform(loc=self.loc, scale=self.scale)


class ThreePointVariable(UncertainDistribution):
    '''An uncertain variable with up to three points of uncertainty
    '''

    #TODO: somehow specify that we are using a triangle distribution from the stats class

    def __init__(self, params, points=100, expectationismean=True):
        '''

        Arguments
        ---------
        params : list
            [min, mode, max]
            defines the distribution
        '''
        super().__init__(params, points)
        self.mode = params[1]

        self.loc = self.left
        self.scale = self.right - self.left
        self.c = (self.mode - self.loc) / self.scale

        self.dist = stats.triang(self.c)

    def get_pdf(self, x, **kwagrs):
        '''get the probability distribution fuction

        Arguments
        ---------
        x : single value or np.array
            the points to return in the pdf
        '''
        x = np.float64(x)
        y = stats.triang.pdf(x, self.c, loc=self.loc, scale=self.scale)
        return y

    def sample(self, size=10):
        '''return random samples from the uncertain distribution'''
        return stats.triang.rvs(self.c, loc=self.loc, scale=self.scale, size=size)

    def expectation(self, ismean=True):
        'expected value'
        return stats.triang.mean(self.c, loc=self.loc, scale=self.scale)

    def std(self):
        'standard deviation'
        pass


class BetaThreePoint(UncertainDistribution):
    '''An uncertain variable with up to three points of uncertainty

    Source: https://github.com/dlmueller/PERT-Beta-Python/blob/master/pertbeta/betadist.py
    '''

    #TODO: somehow specify that we are using a triangle distribution from the stats class

    def __init__(self, left, mode, right, points=100):
        self.left = float(left)
        self.right = float(right)
        self.mode = float(mode)

        self.x = np.linspace(self.left, self.right, points)

    def get_pdf(self, x):
        x = np.float64(x)
        y = stats.triang.pdf(x, self.c, loc=self.loc, scale=self.scale)
        return y

    def risk_profile(self, x=None):
        '''returns the likelihood and percent change from modal value'''

        if x is None:
            x = self.x

        y = self.get_pdf(x)
        liklihood = y / y.sum()  # scale the results from the PDF to 1
        percent_diff = self.percent_diff = (x - self.mode) / self.mode
        risk_profile = percent_diff * liklihood
        return risk_profile

    def sample(self, size=10):
        '''return random samples from the uncertain distribution'''
        return stats.triang.rvs(self.c, loc=self.loc, scale=self.scale, size=size)

    def std(self):
        pass


avail_distributions = {
    'Beta Three Point': BetaThreePoint,
    'Uniform': UniformDistribtuion,
    'Three Point': ThreePointVariable,
}
