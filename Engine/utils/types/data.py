import json
from typing import TYPE_CHECKING, Optional

from api.objects import CopernicusObject
import api.translation as translation
import gpx
from gpx import Variable
import gpx.primitives

if TYPE_CHECKING:
    from utils.types.shared import AcceptedTypes


class Cell(CopernicusObject):
    """Copernicus cell object

    Items
    -----
    name : string
        a descriptive name of the cell
    id : string
    numWorkstations : float
        the number of workstations in the Cell
    override : boolean
    workstationCapacity : int
        the WIP capacity of a single workstation

    """

    def __init__(self) -> None:
        self.override: bool = False
        self.name: str = ""
        self.id: str = ""
        self.numWorkstations: int = 1
        self.workstationCapacity: int = 1


# FUTURE: create a factory to create objects based on the


class Process(CopernicusObject):
    """Process CopernicusObject

    Items
    -----
    type : string
        the name of the process
    cell : int
        the cell where the process is performed
    main : boolean
        is this process part of the main process chain
    id : string
        the unique process id
    """

    def __init__(self) -> None:
        self.type: str = ""
        self.cell: int = 0
        self.main: bool = True
        self.id: str = ""


class Manufacturing(CopernicusObject):
    """manufacturing

    Items
    -----


    """

    pass


class FeederLine(CopernicusObject):
    """Feeder Line

    A feeder line is a list of cells

    Items
    -----
    processes : list of type Process

    id : string
        the name of the feeder line
    qty : int
        the quantity which is produced on the feeder line
    joinid : string
        the name of the primary process where the feederline joins



    feederProcesses
    ----------------
    id : string
    type : string
        different thing for type so we can reuse processes
    cell : string
    feederLine : string
        the feederLine to which the process belongs

    feederLines
    -----------
    id : string
    to : string
        key for where the process feeds


    """

    def __init__(self) -> None:
        self.processes: list[dict[str, gpx.primitives.Process]] = []
        self.joinid: str = ""

    def _dicttoobject(self, inputdict: "AcceptedTypes", **kwargs: "AcceptedTypes") -> None:
        """overloads the default to take in keyed feeder lines

        Arguments
        ---------
        inputdict : dict
            the object as a dictionary

        Keyword Arguments
        -----------------
        key : string
            the name of the feederline to parse
        to : string
            the name of the primary process where the feederline joins
        """

        if "key" in kwargs:  # sort from a larger list of processes
            # this assumes that there are multiple feeder lines in the inputdict
            # use the key to only select the ones we want
            self.joinid = str(kwargs["to"])
            # include only the processes which are applicable to the line specified by
            # key
            # Invalid index type "str" for "str"; expected type "SupportsIndex | slice[Any, Any, Any]"
            # Argument missing for parameter "inputdict"
            self.processes = [
                Process._dicttoobject(p)  # type: ignore
                for p in inputdict
                if isinstance(p, dict) and p.get("feederLine") == kwargs["key"]
            ]  # TODO: Find correct type - _dicttoobject only ever returns none?
        else:
            super()._dicttoobject(inputdict)


class Design(CopernicusObject):
    """Design

    Items
    -----
    parameters : list of __main__.Parameter
    plyBook : plyBook
    """

    def __init__(self, **kwargs: "AcceptedTypes") -> None:
        # self.parameters = []
        super().__init__(**kwargs)

    def _dicttoobject(self, inputdict: "AcceptedTypes") -> None:
        """create the Design object from an input dictionary

        overloaded to handle the nesting of the parameter objects

        Arguments
        ---------
        inputdict : dict
            description of the object
        """
        # TODO: remove, as the design parameters were an old implementation
        # create a list of the parameters
        # if 'parameters' in inputdict:
        #     self.parameters = [Parameter(construct_from_dict=p)
        #                        for p in inputdict['parameters']]
        super()._dicttoobject(inputdict)


class Parameter(CopernicusObject):
    """Parameter

    Items
    -----
    name : string
    descr : string
        description
    source : string
        where the parameter comes from. typically one of three:
        "From Assumption"
        "Calculated Value"
        "User Input"
    key : string
        key used for better display and location in sub-modles and modules
    unit : string
        The units of the parameter
    value : float

    infoLabels : dict
        a dictionary of labels and values providing additional information
    """

    def __init__(self, **kwargs: "AcceptedTypes") -> None:
        self.name: str = ""
        self.source: str = ""
        self.unit: str = ""
        self.value: Optional[float] = None
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self.key: str = ""
        self.descr: str = ""
        self.tags: list[str] = []
        self.category: str = ""
        self.type: str = ""
        self.property: str = ""
        self.gpx_translator = translation.param_to_var
        super().__init__(**kwargs)

        # copy user-supplied kwargs onto the instance
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.descr == "":
            self.descr = self.name

        # correct for % units ignoring list inputs
        # TODO:  what is the effect of setting the substitution here AND in
        # `sort_constraints`
        if self.unit == "%" and not isinstance(self.value, list):
            self.unit = ""
            # check for percentage as min max
            # some values may be none when in a best-worst context
            if self.value:
                self.value = self.value / 100.0
            if self.min:
                self.min = self.min / 100.0
            if self.max:
                self.max = self.max / 100.0

    def __repr__(self, prettify: int = 4) -> str:
        """
        Pretty-print this Parameter as JSON. Any non-serialisable value
        is coerced to a string so we never drop into the unsafe fallback.
        """

        def _fallback(o):  # noqa: D401
            return str(o)

        return json.dumps(vars(self), indent=prettify, default=_fallback)

    @property
    def gpxObject(self, name_attr: str = "key") -> Variable:
        """gets the gpxobject

        Arguments
        ---------
        name_attr : str (default='key')
            the attribute to set as the name of the gpx variable

        """
        units = self.unit
        varkey = str(getattr(self, name_attr))
        return Variable(  # TODO: Find correct type
            varkey,
            # self.value,   # don't set the substitution here
            units,
            self.descr,
        )


class CostComponent(CopernicusObject):
    """Cost Component

    Used for displaying costs in the waterfall chart

    Items
    -----


    """


class Model(CopernicusObject):
    """model object

    Items
    -----
    design : __main__.Design
    manufacturing : __main__.Manufacturing

    partName : string
        the name of the part

    """

    def __init__(self) -> None:
        self.design: Design = Design()
