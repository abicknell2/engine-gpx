'a VarKey like object for parameter keys'
from gpkit.repr_conventions import ReprMixin


class ParamKey:

    def __init__(self, name=None, **descr):
        self.varkey = name
        self.key = self
        self.keys = set((name))

    def shape(self):
        'returns the shape of the key. should be 1 since not in vector'
        return None
