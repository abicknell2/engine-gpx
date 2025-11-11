from dataclasses import dataclass

import gpx


@dataclass
class costResult:
    "create a data object for costResults"

    displayname: str
    value: float
    var: gpx.Variable
    resource: str = "-"
