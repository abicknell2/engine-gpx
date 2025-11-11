"modeling modules"

from collections import OrderedDict as OD
import logging

import api.constants as constants
from utils.types.shared import AcceptedTypes

logging.basicConfig(level=constants.LOG_LEVEL)

DESIGN_NOT_REQUIRED: list[str] = ["service", "operations", "manufacturing"]

DESIGN_REQUIRED: list[str] = ["custom"]


def list_to_dict(
    inputlist: list[dict[str, AcceptedTypes]],
    newkey: str,
    keepkey: bool = True,
    asordereddict: bool = False
) -> dict[str, dict[str, AcceptedTypes]] | OD[str, dict[str, AcceptedTypes]]:
    "converts an input list of dicts to a dict by name"

    if asordereddict:
        newdict: OD[str, dict[str, AcceptedTypes]] = OD()
        for i in inputlist:
            key = str(i[newkey])
            newdict[key] = i
        return newdict

    return {str(item[newkey]): item for item in inputlist}


def list_to_od(inputlist: list[dict[str, AcceptedTypes]],
               newkey: str,
               keepkey: bool = True) -> OD[str, dict[str, AcceptedTypes]]:
    """converts a list to an OrderedDict
    Preserves the order of the input list

    Returns
    ------
    collections.OrderedDict
    """
    result: OD[str, dict[str, AcceptedTypes]] = OD()
    for i in inputlist:
        result[str(i[newkey])] = i
    return result


def has_valid_name(x: dict[str, AcceptedTypes], namekey: str = "name") -> bool:
    return bool(x[namekey].strip())


def strip_empty(modulelist: list[dict[str, AcceptedTypes]], namekey: str = "name") -> list[dict[str, AcceptedTypes]]:
    """Removes items without a valid name."""
    return [x for x in modulelist if has_valid_name(x, namekey)]
