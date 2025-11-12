'''expose the basic gpx modules'''

# from gpkit import Model
from gpkit import Variable, ureg

from .model import Model
from . import patches 

TIGHT_CONSTR_TOL = 1e-4
