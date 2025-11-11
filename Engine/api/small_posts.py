"small post-processing steps"

import numpy as np

import gpx


def decompose_process_time(
    name: str,
    process: gpx.primitives.Process,
    sol: gpx.Model,
    cats: list[str] = ["Secondary", "Setup", "Quality"],
    splitchar: str = " ",
    splitidx: int = 0,
    splitstacks: bool = False,
    otherauto: bool = True,
) -> dict[str, float] | list[dict[str, str | float]]:
    """decompose a process into different categories based on variable names

    Arguments
    ---------
    name : the name of the process
    process :
    """
    pname = name.split(splitchar)[splitidx]

    total_time = sol["variables"][process.t.key]

    breakdown = {}

    for c in cats:
        varname = f"{pname} {c.upper()}"
        try:
            v = sol["variables"][varname]
            # check for units
            if v > 100:
                breakdown[c.lower()] = v
            else:
                breakdown[c.lower()] = float(v) * 60.0
        except KeyError:
            # breakdown[c.lower()] = 0.0
            pass

    if otherauto:
        other = total_time - np.sum(list(breakdown.values()))
        breakdown["primary"] = other

    if splitstacks:
        stacks = []
        for bname, bvalue in breakdown.items():
            stacks.append({
                "Stack Name Label": name,
                "Stack Segment Label": bname,
                "Segment Value (x)": bvalue,
            })
        return stacks

    return breakdown
