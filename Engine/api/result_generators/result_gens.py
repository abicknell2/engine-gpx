import gpx
from utils.settings import Settings
from utils.types.shared import (AcceptedTypes, AllResults, CollectVars, ResultsDict, ResultsIndex)


class ResultGenerator:
    """result generator base cellVariationSensitivity

    Attributes
    ----------
    gpxsol : gpkit.solution
        solution to the gp
    collect_vars : dict
        variables of interest to return to the results collection
    results : dict of list of dicts
        each type of result is an entry in the dict
        each type of result is a list of dicts
    results_index : list of dicts
        provides a description to the type of results
        [{'name'  : <descriptive name of the result>,
          'value' : <corresponding key in the }]
    """

    def __init__(self, gpxsol: gpx.Model, settings: Settings, **kwargs: AcceptedTypes) -> None:
        """
        Arguments
        ---------
        gpxsol : gpkit.solution
            Solution to a solved GP
        """

        self.gpxsol = gpxsol
        self.results: ResultsDict = {}
        self.results_index: ResultsIndex = []
        self.collect_vars: CollectVars = {}
        self.aux_vars: list[gpx.Variable] = []  # a list of objects to return with the all vars
        self.summary_res: dict[str, str | float] = {}  # a dict of values to add to a product summary
        self.settings: Settings = settings

    def get_all_results(self, asdict: bool = True) -> AllResults:
        """generate all of the results and return

        Returns
        -------
        self.results, self.results_index, self.collect_vars
        """

        if asdict:
            return {
                "results": self.results,
                "results_index": self.results_index,
                "collect_vars": self.collect_vars,
            }

        return self.results, self.results_index, self.collect_vars
