"the basis objects for the copernicus interactions"

from collections import OrderedDict
import json
import logging
from typing import TYPE_CHECKING, Union

from gpkit import ConstraintSet
from gpkit.nomials.math import PosynomialInequality
import json2html

from api import translation
import gpx
import gpx.primitives
import utils.logger as logger

if TYPE_CHECKING:
    from utils.types.shared import AcceptedTypes, Substitutions

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.WARNING)


# from gpx import Model
class CopernicusObject:
    """base object for the Copernicus Engine

    Items
    -----
    infoLabels : dict
        a dictionary of additional keys and values to provide augmented labeling
        for example for implementing trl thermometer.

    gpxObject : iterable
        the related gpx object representation
        this is assigned during the InteractiveModel translation step

    displayParams : list
        the list of attributes which should be revealed externally

    variables : dict
        the variables which are defined in the model with the following entries
        name : data_types.Parameter
            the name referred to in the models and the object

    substitutions : "Substitutions"
        variables which have substitutions which must be performed
        name : substitutions
            the name should be the same as in the variables dict
            substitutions are represented as a tuple
                (value, units)

    # FUTURE ITEMS
    # ------------
    # translator
    #     a reference to the function to call to make the gpx objects
    """

    def __init__(self, **kwargs: "AcceptedTypes") -> None:
        # TODO: implement multiple constructor methods from dict
        """construct the object

        Optional Arguments
        ------------------
        construct_from_dict : dict
            construct the object from a dictionary
        """
        self.infoLabels: dict[str, str] = {}
        self.variables: dict[str, gpx.Variable] = {}
        self.substitutions: "Substitutions" = {}
        self.gpx_translator = translation.param_to_var

        if "construct_from_dict" in kwargs:
            self._dicttoobject(kwargs["construct_from_dict"])
        if "construct_from_init" in kwargs and "construct_from_dict":
            # look for kwargs to ceate the attributes
            for kw in kwargs:
                if hasattr(self, kw):
                    # set the attribute
                    setattr(self, kw, kwargs[kw])

    def __repr__(self, prettify: int = 4) -> str:
        """pretty printing of the object

        Arguments
        ---------
        indent : int
            0 for no pretty printing
        """
        try:
            return json.dumps(self.tojson(), indent=prettify)
        except BaseException:
            return str(vars(self))

    def _repr_html_(self) -> str:
        """HTML representation for ipython integration
        uses json2html library: https://pypi.org/project/json2html/
        """

        return json2html.json2html.convert(self.tojson())

    def getvars(self) -> dict[str, gpx.Variable]:
        """get the variables from the object"""
        return self.variables

    def getsubs(self) -> "Substitutions":
        """get the substitutions from the object"""
        return self.substitutions

    def _post_solve(self) -> None:
        """calculate some items following the solve

        Example
        -------
        wip in the queue and station for each cell
        """
        pass

    def extract_solution(self, solution: object) -> None:
        """create an object with the relevant returns

        Arguments
        ---------
        solution : results.Solution
        """
        # TODO@snill: implement. What should this be?
        pass

    def _dicttoobject(self, inputdict: "AcceptedTypes") -> None:
        """Creates an object from an input dict with
        Takes the objects out the JSON and puts them into object attributes
        see: https://stackoverflow.com/questions/6578986/how-to-convert-json-data-into-a-python-object

        Arguments
        --------
        inputdict : dict
            a dictionary with the attributes and the values to create the object
        """
        var_keys = vars(self).keys()
        for key, val in inputdict.items():
            if key in var_keys:
                # only if the key is there, add
                vars(self)[key] = val

        # object_decoder(self, inputjson)
        # vars(self).update(json.loads(inputjson))

    def tojson(self, nested_objects: list[str] | None = None) -> str:
        """Returns the JSON formatted version of the object

        Arguments
        ---------
        nested_objects : list
            a list of the attributes which may contain nested objects
        Returns
        -------
        string
            json formatted as a string
        """

        # TODO: return the json from the nested objects
        if nested_objects is None:
            return json.dumps(self.__dict__)

        # check if the object may be inherited from this object
        # isinstance(self, CopernicusObject)
        return json.dumps(self.__dict__)

    @property
    def gpx_constraints(self, **kwargs: "AcceptedTypes") -> Union[ConstraintSet, list[PosynomialInequality]]:
        """get the gpx constraints from the gpx_objects

        Returns
        -------
        list
            list of constraints
        """
        return []

    def create_gpx(self: gpx.primitives) -> None:
        "create a gpx object from itself"
        try:
            self.gpx_translator(self)
        except AttributeError:
            logger.debug(f"no translator found for: {self}")
            pass


# TODO: implement this sort of object
class ModelDict(OrderedDict[str, gpx.Model]):
    """a dictionary-like object to hold models
    wehn called, returns a list of constraints

    Should be able to work when nested in other ModelDicts
    """

    pass
