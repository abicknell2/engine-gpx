"modeling modules"

import logging
from typing import Union

import api.constants as constants
from api.module_types.design import Design
from api.module_types.manufacturing import Manufacturing
from api.objects import CopernicusObject
from utils.types.shared import AcceptedTypes

logging.basicConfig(level=constants.LOG_LEVEL)

DESIGN_NOT_REQUIRED = ["service", "operations", "manufacturing"]

DESIGN_REQUIRED = ["custom"]


class ModelModules(CopernicusObject):
    """base class for input modules

    Items
    -----
    modules : list of ModuleType
        the different modules in the class

    """

    def __init__(self, **kwargs: AcceptedTypes) -> None:
        self.modules: list[Union[Manufacturing, Design]] = []
        super().__init__(**kwargs)

    def _dicttoobject(self, inputdict: AcceptedTypes, **kwargs: AcceptedTypes) -> None:
        """Creates the modules from an input dictionary

        Optional Keyword Arguments
        --------------------------
        designrequired : bool (default True)
            does the input require design input
            mostly gets passed on to manufacturing

        """
        self.modules = []
        for mod in inputdict:
            if isinstance(mod, dict) and mod.get("type") == "design":
                self.modules.append(Design(construct_from_dict=mod))
            elif isinstance(mod, dict) and mod.get("type") == "manufacturing":
                self.modules.append(Manufacturing(construct_from_dict=mod, **kwargs))

            else:
                ex = f"type not recognized for module: \n{mod}"
                # raise Exception(ex)
                logging.warn(ex)

    def modules_as_dict(self) -> dict[str, dict[str, Union[Manufacturing, Design]]]:
        """get the different modules as a dict of dicts

        Returns
        -------
        dict of dicts of ModuleType
            module type : modules by name
        """
        mod_dict: dict[str, dict[str, Union[Manufacturing, Design]]] = {"design": {}, "manufacturing": {}}
        for mod in self.modules:
            print(mod.name)
            mod_dict[mod.type][mod.name] = mod

        return mod_dict

    def modules_by_id(self) -> dict[Union[int, float, str], Union[Manufacturing, Design]]:
        "returns a dict of modules by their id"
        return {m.id: m for m in self.modules if m.id is not None}
