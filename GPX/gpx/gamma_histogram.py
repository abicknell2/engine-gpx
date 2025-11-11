'used to produce sample data from input gamma distribution'

#TODO@snill: implement from


class Gamma(object):
    '''simple gamma distribution

    Items
    -----
    alpha : float
        shape parameter
    beta : float
        rate parameter
    k : float
        shape parameter
    theta : float
        scale parameter
    '''

    def __init__(self):
        pass

    def from_mean_std(self, mean, std):
        '''creates a gamma distribution

        '''


def make_pdf(dist, upper_bound, lower_bound=0, pts=100):
    '''generates points on a pdf based on an input distribution

    Arguments
    ---------
    dist :
        reference distribution
    upper_bound : float
        the upper bound to sample
    lower_bound : float
        the lower bound to sample
    pts : int
        number of points to sample

    Returns
    -------
    '''
    pass
