from typing import cast

import gpkit

from api.constraint_interpreter import build_constraint_set
from api.module_types.module_type import ModuleType
from api.objects import CopernicusObject
from utils.constraints_helpers import sort_constraints
from utils.types.data import Parameter
from utils.types.shared import AcceptedTypes


class Design(ModuleType):
    """
    Items
    -----
    name : string
        the name of the module

    type : string
        the type of the module

    design : list of dicts
        the design constraints

    """

    def __init__(self, **kwargs):
        self.design = []
        super().__init__(**kwargs)
        # for param in self.variables:
        #     param.infoLabels['model'] = 'Design'

    def _dicttoobject(self, inputdict):
        design_dict = inputdict["design"]
        self._original_constraints = design_dict.get("variables", [])
        # enter the sorted constraints
        self.substitutions, self.constraints, self.lhs = sort_constraints(self._original_constraints)
        CopernicusObject._dicttoobject(self, design_dict)

        super()._dicttoobject(inputdict)

    # @property
    def gpx_dynamic_constraints(self, **kwargs):
        return build_constraint_set(self.gpx_variables, self.constraints, **kwargs)

    def gpx_constraints(self, **kwargs):
        return self.gpx_dynamic_constraints(**kwargs)
