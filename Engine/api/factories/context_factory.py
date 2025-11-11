from api.context.min_unit_cost import MinUnitCost
from api.context.ramp import rampContext
from api.context.uncertain import Uncertain
from utils.types.shared import AcceptedTypes


def make_context(kind: str, interactive: AcceptedTypes):
    if kind == "min_cost":
        return MinUnitCost(interactive)
    if kind == "uncertain":
        return Uncertain(interactive)
    if kind == "rate_ramp":
        return rampContext(interactive)
    return MinUnitCost(interactive)
