"update an uncertainty model with inputs from excel"

import gpx
from utils.types.shared import AcceptedTypes


def update_uncertainty_from_excel(
    umodel: gpx.Model,
    updates: str,
    besth: str = "Best-Case",
    likelyh: str = "Likely-Case",
    worsth: str = "Worst-Case",
    keyh: str = "AA KEY",
    unitsh: str = "AA UNITS",
    **kwargs: AcceptedTypes,
) -> None:
    """Updates an uncertainty model with inputs from Excel

    Arguments
    ---------
    umodel : JSON as .aam file
        the input model to update with the excel file
    updates : an excel spreadsheet
    besth : string
        the name of the column header for the best case
    likelyh : string
        the name of the column header for the likely case
    worsth : string
        the name of the column header for the worst case
    keyh : string
        column header for variable name
    unitsh : string
        column header for units

    Optional Keyword Arguments
    --------------------------
    sheet : string
        the name of the worksheet to translate

    Returns
    -------
    JSON-formatted string
    """

    if "sheet" in kwargs:
        pass
