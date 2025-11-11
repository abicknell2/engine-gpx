import numbers
from typing import Optional, Union, cast

from gpkit import Variable
import numpy as np

from api.constraint_interpreter import build_constraint_set
import api.module_translation as module_translation
from api.module_types.module_type import ModuleType
from api.objects import CopernicusObject
from api.result_generators.cell_results import CellResults
from api.result_generators.celltimes import CellTimes
from api.result_generators.feeder_flow_times import FeederFlowtimes
from api.result_generators.floorspace import FloorSpace
from api.result_generators.labor_costs import LaborCost
from api.result_generators.line_results import LineResults
from api.result_generators.m_cell_results import MCellResults
from api.result_generators.mc_product_summary import MCProductSummary
from api.result_generators.non_recurring_cost_results import NRCostResults
from api.result_generators.off_shift_results import OffShiftResults
from api.result_generators.process_results import ProcessResults
from api.result_generators.product_summary import ProductSummary
from api.result_generators.recurring_cost_results import RecurrCostResults
from api.result_generators.result_gens import ResultGenerator
from api.result_generators.tool_results import ToolResults
from api.result_generators.total_cost_results import TotalCostResults
from api.result_generators.var_cost_results import VarCostResults
import gpx
import gpx.dag
import gpx.dag.parametric
from gpx.dag.parametric import ParametricConstraint, ParametricVariable
from gpx.feeder import FeederLine
import gpx.manufacturing
from gpx.manufacturing import ConwipTooling, FabLine, QNACell
from gpx.mfgcosts import FloorspaceCost
from gpx.non_recurring_costs import ConwipToolingCost
import gpx.primitives
from gpx.primitives import Cost, Process, UnitCost
from gpx.recurring_cost import LaborCost as GPXLaborCost
from utils.constraints_helpers import sort_constraints
from utils.list_helpers import list_to_dict
from utils.settings import Settings
from utils.types.data import Parameter
from utils.types.shared import AcceptedTypes


class Manufacturing(ModuleType):
    """

    Items
    -----
    name : string
        the name of the module
    type : string
        the type of the module
    design : Design
        a reference to the design module
    cells : OrderedDict
        the manufacturing cells
    processChain : OrderedDict
        the process chain ordered by production order
            name : utils.types.data.Process
    feederLines : dict
        the names of the different feeder lines
            id : string
    feederProcesses : OrderedDict
        the processes comprising the feeder lines
            id : Process
    tools : list of dicts
        tools as dicts
            name : tool name
            path : list of process id
    costs : list
        additional costs
    module_variables : dict
        key is the name of the variables
    cell_num_map : dict
        a mapping from cell number to cell name
    exposedVariables : list
        the exposed variable list
    designrequired : bool (default=True)
        is a design module a required input
    """

    def __init__(self, **kwargs: AcceptedTypes) -> None:
        self.cells: Union[dict[str, dict[str, str]], list[dict[str, str]]] = []
        self.processChain: list[dict[str, AcceptedTypes]] = []
        self.feederProcesses: list[dict[str, AcceptedTypes]] = []
        self.feederLines: list[dict[str, AcceptedTypes]] = []
        self.tools: list[dict[str, AcceptedTypes]] = []
        self.costs: Union[dict[str, dict[str, AcceptedTypes]], list[dict[str, AcceptedTypes]]] = []
        self.routes: list[dict[str, AcceptedTypes]] = []
        self.secProcesses: list[dict[str, AcceptedTypes]] = []
        self.exposedVariables: list[dict[str, str]] = []
        self.designrequired: bool = True
        self.design: "ModuleType | None" = None
        self.rates: list[dict[str, AcceptedTypes]] | None = None
        # self.inputFormat = []  # probably don't need this info from front-end
        self.variablesSimple: Union[list[dict[str, int | str]], dict[int | str, Parameter]] = {}
        self.has_offshift: bool = False
        self.partsPer: int = 1
        self.kitName: str = "kit"
        self.cell_num_map: dict[int, str] = {}
        self.ton: Variable | None = None
        self._offshift_hrsPerShift: Variable | None = None
        self._offshift_shiftsPerDay: Variable | None = None
        self._gpx_all_cells: dict[str, gpx.manufacturing.QNACell] = {}
        self.gpx_offshift: dict[str, Variable] = {}
        self.gpx_variables: dict[str, Variable] = {}

        super().__init__(**kwargs)

        if "designrequired" in kwargs:
            self.designrequired = bool(kwargs["designrequired"])

        self.cells = cast(
            dict[str, dict[str, str]],
            list_to_dict(inputlist=cast(list[dict[str, AcceptedTypes]], self.cells), newkey="name", asordereddict=True)
        )
        self.costs = list_to_dict(inputlist=self.costs, newkey="name")

    def _dicttoobject(self, inputdict: AcceptedTypes) -> None:
        """overloads the parent default

        Arguments
        ---------
        inputdict : dict
            Manufacturing object as an input dict
        """

        if not isinstance(inputdict, dict) or "manufacturing" not in inputdict:
            raise ValueError("inputdict must be a dictionary containing the key 'manufacturing'")

        mfg_dict = inputdict["manufacturing"]
        self._original_constraints = mfg_dict["variables"]
        self.substitutions, self.constraints, self.lhs = sort_constraints(mfg_dict["variables"])

        CopernicusObject._dicttoobject(self, mfg_dict)

        # FILTER INCOMING VARIABLES
        # filter out variables that do not have a value
        filtered_variables_simple: list[dict[str, int | str]] = []
        for vs in self.variablesSimple:
            if isinstance(vs, dict) and "value" in vs:
                filtered_variables_simple.append(vs)

        self.variablesSimple = filtered_variables_simple
        # filter out non-numeric values in the variablesSimple
        self.variablesSimple = list(filter(lambda vs: isinstance(vs["value"], numbers.Number), self.variablesSimple))
        # filter variables that do not have a type
        self.variablesSimple = list(filter(lambda vs: str(vs["type"]).strip() != "", self.variablesSimple))

        # filter_list = [
        #     (self.cells, 'name'),
        #     (self.processChain, 'type'),
        #     (self.tools, 'name'),
        # ]

        # filter cells, processChain
        filtered_cells = [c for c in self.cells if (c.get("name") or "").strip() != ""]
        self.cells = cast(Union[dict[str, dict[str, str]], list[dict[str, str]]], filtered_cells)

        self.processChain = [p for p in self.processChain if (p.get("type") or "").strip() != ""]
        self.tools = [t for t in self.tools if (t.get("name") or "").strip() != ""]

        # TODO:  check for off-shift cells
        offshift: list[dict[str, int | str]] = []
        for v in [*self.variables, *self.variablesSimple]:
            if isinstance(v, dict) and "property" in v and v["property"].lower().strip() == "extra shifts":
                offshift.append(v)

        self.has_offshift = len(offshift) > 0  # set the falg on the module if there is off-shift work

        if self.has_offshift:
            self._offshift_vars = offshift

        # replace 0 with empty
        inspect_list = ["secProcesses", "feederProcesses", "tools", "feederLines"]

        for a in inspect_list:
            # catch any 0 that were sent in place of empty lists
            if getattr(self, a) == 0:
                setattr(self, a, [])

        # add substituions from the simple variables
        simplvarssubs: dict[str, tuple[float, str]] = {}
        for vs in self.variablesSimple:
            if "unit" not in vs:
                # if there is no unit, assume unitless
                if isinstance(vs, dict):
                    vs["unit"] = ""

            if isinstance(vs["value"], numbers.Number):
                # add the substituion only if the value is a number
                simplvarssubs[str(vs["key"])] = (float(vs["value"]), str(vs["unit"]))

        self.substitutions.update(simplvarssubs)

        super()._dicttoobject(inputdict)

        # get the simple variables as a dict
        variables_simple_dict: dict[int | str, Parameter] = {}
        for sv in self.variablesSimple:
            if isinstance(sv, dict):
                variables_simple_dict[str(sv["key"])] = Parameter(construct_from_dict=sv)

        self.variablesSimple = variables_simple_dict

        # combine both sets of variables
        self.variables.update(cast(dict[str, Parameter], self.variablesSimple))

    def gpx_constraints(self, **kwargs: AcceptedTypes) -> list[AcceptedTypes]:
        """
        Required Keyword Arguments
        --------------------------
        designvars : dict
            the design variables which may be referenced by constraints in the manufacturing module
            name : gpx.Variable

        Returns
        -------
        list
            list of the constraints
        """
        # # check for the inputs
        design_vars = self._get_design_vars(kwargs)

        # create additional access to variables that are in the module but may not
        # have the full key
        addl_vars = {vname.split("//")[0].strip(): var for vname, var in self.gpx_variables.items() if "//" in vname}

        acyclic_input: bool = cast(bool, kwargs.pop('acyclic_input', False))
        constr = build_constraint_set(
            {
                **design_vars,
                **self.gpx_variables,
                **addl_vars,
            },
            self.constraints,
            acyclic_input,
            **kwargs,
        )

        try:
            if kwargs["dynamic_only"]:
                constr.extend(self.gpxObject["auxiliaryConstraints"])
                return constr
        except KeyError:
            pass

        # TRANSLATE

        # pull in the constraints from the models
        # constraints.extend(self.gpx_models.values())
        # duck-type the entries in the gpxObject as
        for obj in self.gpxObject.values():
            try:
                if isinstance(obj, dict):
                    constr.extend(obj.values())
                else:
                    constr.append(obj)
            except AttributeError:
                if isinstance(obj, dict):
                    constr.extend(obj.values())
                else:
                    constr.append(obj)

        return constr

    def _get_design_vars(self, kwargs: dict[str, AcceptedTypes]) -> dict[str, gpx.Variable]:
        if "designvars" not in kwargs:
            # use the design vars from the object
            if self.design is not None:
                design_vars: dict[str, gpx.Variable] = self.design.gpx_variables
            else:
                design_vars = {}
        elif not isinstance(kwargs["designvars"], dict):
            raise Exception("designvars must be a dict")
        else:
            design_vars = kwargs["designvars"]
        return design_vars

    def gpx_translate(self, settings: Settings, **kwargs: dict[str, AcceptedTypes]) -> None:
        # super().gpx_translate(self,**kwargs)
        # update list of variables to replace
        self._gpx_vars_for_exposed()
        self._replace_equivalent_constr()

        # if there are any substitutions, make that property as well
        if self.var_sub_names:
            self.gpx_var_subs = {self.gpx_variables[k]: self.gpx_variables[v] for k, v in self.var_sub_names.items()}

        # make substitutions directly in the gpx variables
        newvars = {k: self.gpx_var_subs[v] if v in self.gpx_var_subs else v for k, v in self.gpx_variables.items()}
        self.gpx_variables = newvars

        # run the translation
        module_translation.manufacturing_module(default_currency=settings.default_currency_iso, module=self)

    def get_acyclic_constraints(
        self,
        parametric_variables: Optional[dict[str, ParametricVariable]] = None,
        substitutions: Optional[dict[str, tuple[AcceptedTypes, AcceptedTypes]]] = None,
        additional_variables: dict[str, Variable] = {},
        **kwargs: AcceptedTypes,
    ) -> list[ParametricConstraint]:
        "overload default behavior to include variables from the design module"

        design_vars = self._get_design_vars(kwargs)

        if parametric_variables is None:
            parametric_variables = cast(dict[str, ParametricVariable], kwargs.pop('parametric_variables', None))

        if substitutions is None:
            substitutions = cast(dict[str, tuple[AcceptedTypes, AcceptedTypes]], kwargs.pop('substitutions', None))

        addl_vars = {vname.split("//")[0].strip(): var for vname, var in self.gpx_variables.items() if "//" in vname}
        merged_vars = {**design_vars, **self.gpx_variables, **addl_vars, **additional_variables}

        # call the default behavior
        return super().get_acyclic_constraints(
            additional_variables=merged_vars,
            parametric_variables=parametric_variables,
            substitutions=substitutions,
            **kwargs,
        )

    def _gpx_vars_for_exposed(self) -> None:
        "create gpx variables for any missing exposed variables"
        # default units
        aux_units_map = {
            "workstation count": "count",
            "queueing time": "min",
            "queue inventory": "count",
            "flow time": "min",
            "count": "count",
            "utilization": "-",
            "wip inventory": "count",
            "production rate": "count/hr",
        }

        # loop over the variables
        for v in self.exposedVariables:
            if v["name"] not in self.gpx_variables:
                self.gpx_variables[v["name"]] = Variable(
                    str(v["name"]),
                    aux_units_map[str(v["property"]).lower()],
                    "Expose Variable",
                )

    def _replace_equivalent_constr(self) -> None:
        "replaces constraints that are just equals"
        # TODO:  in the future use grpah techniques to generate
        new_constr: list[Union[str, float]] = []
        # make a list of the exposed variables
        expvars = {v["name"]: v for v in self.exposedVariables}

        # create gpx variables for the exposed variables if they are missing

        for c in self.constraints:
            # loop over the constraint set
            # yapf: disable
            if (
                c["sign"] == "="
                and isinstance(c["value"], (list, str)) and len(c["value"]) == 1
                and c["name"] in expvars
                and c["math"]
                and c["monomial"]
                and type(c["value"][0]) is str
            ):
                # yapf: enable
                # this should be an equivalent constraint and the value saved
                # update the list of substituted values
                self.var_sub_names[str(c["key"])] = c["value"][0]
                # additionally key from name
                self.var_sub_names[str(c["name"])] = c["value"][0]
            else:
                # otherwise, preseve the constraint as is
                new_constr.append(c)

        # update the module constraints with new constraint set with the
        # equivalence constraints removed
        self.constraints = new_constr

    def get_results(
        self,
        settings: Settings,
        sol: gpx.Model,
        suppressres_override: bool = False,
        **kwargs: dict[str, AcceptedTypes],
    ) -> list["ResultGenerator"]:
        "get the results from the module"

        if self.suppress_results and not suppressres_override:
            # suppress results if the module requests it and not overridden
            return []

        # get all of the cells in the manufacturing
        cells = self.get_all_cells()

        # send a big list of the generators
        # results.extend([
        #     result_gens.CellResults(gpxsol=sol,
        #                             cells=cells,
        #                             cells_in_order=self.get_cells_in_order()),
        #     result_gens.ProcessResults(gpxsol=sol,
        #                                processchain=self.processChain,
        #                                processes=self.gpxObject['processes']),
        #     result_gens.LineResults(gpxsol=sol,
        #                             fabline=self.gpxObject['fabLine']),
        #     result_gens.TotalCostResults(gpxsol=sol,
        #                                  unitcost=self.gpxObject['unitCost'])
        #     result_gens.VarCostResults(gpxsol=sol,
        #                                varcosts=self.gpxObject['varCosts'],
        #                                labcosts=self.gpxObject['laborCosts']),
        #     result_gens.NRCostResults(gpxsol=sol,
        #                               module=self),
        #     result_gens.ToolResults(gpxsol=sol, toolcosts=self.gpxObject['toolCosts']),
        #     result_gens.CellTimes(gpxsol=sol,
        #                           cells=cells,
        #                           cells_in_order=self.get_cells_in_order()),
        #     result_gens.FloorSpace(gpxsol=sol, fscosts=self.gpxObject['floorspaceCost']),
        #     result_gens.LaborCost(gpxsol=sol, labcosts=self.gpxObject['laborCosts']),
        # ])
        results: list[Union[ProcessResults, CellResults, MCellResults, LineResults, VarCostResults, RecurrCostResults,
                            NRCostResults, TotalCostResults, FeederFlowtimes, ToolResults, CellTimes, FloorSpace,
                            LaborCost, ProductSummary, OffShiftResults, MCProductSummary]] = []

        # get the cell results generator
        cell_results = CellResults(gpxsol=sol, cells=cells, cells_in_order=self.get_cells_in_order(), settings=settings)

        # add the result gen to the generators
        results.append(cell_results)

        results.append(
            ProcessResults(
                gpxsol=sol,
                processchain=cast(list[dict[str, gpx.primitives.Process]], self.processChain),
                processes=cast(dict[str, gpx.primitives.Process], self.gpxObject["processes"]),
                settings=settings
            ),
        )

        results.append(LineResults(gpxsol=sol, fabline=cast(FabLine, self.gpxObject["fabLine"]), settings=settings))

        var_cost = VarCostResults(
            gpxsol=sol,
            varcosts=cast(dict[str, Cost], self.gpxObject["varCosts"]),
            labcosts=cast(list[GPXLaborCost], self.gpxObject["laborCosts"]),
            invholdcost=cast(Cost, self.gpxObject["invHolding"]),
            settings=settings
        )
        results.append(var_cost)

        rec_cost = RecurrCostResults(
            gpxsol=sol,
            recur_costs=cast(dict[str, Cost], self.gpxObject["recurringCosts"]),
            cell_costs=cast(dict[str, Cost], self.gpxObject["cellCosts"]),
            fs_cost=cast(FloorspaceCost, self.gpxObject["floorspaceCost"]),
            settings=settings,
            tool_costs=cast(dict[str, ConwipToolingCost], self.gpxObject["toolCosts"]),
        )
        results.append(rec_cost)

        nonrec_cost = NRCostResults(gpxsol=sol, module=self, cellresults=cell_results, settings=settings)
        results.append(nonrec_cost)

        results.append(
            TotalCostResults(
                gpxsol=sol,
                unitcost=cast(UnitCost, self.gpxObject["unitCost"]),
                varCost=var_cost,
                recCost=rec_cost,
                nonrecCost=nonrec_cost,
                parts_per=self.partsPer,
                kit_name=self.kitName,
                settings=settings
            ),
        )

        results.append(
            FeederFlowtimes(
                gpxsol=sol,
                gpx_feeder_lines=cast(dict[str, FeederLine], self.gpxObject["feederLines"]),
                feeder_lines=cast(list[dict[str, FeederLine]], self.feederLines),
                feeder_processes=self.feederProcesses,
                secondary_processes=self.secProcesses,
                process_flow=self.processChain,
                gpx_processes=cast(dict[str, Process], self.gpxObject["processes"]),
                gpx_cells=cast(dict[str, QNACell], self.gpxObject["cells"]),
                gpx_feeder_cells=cast(dict[str, QNACell], self.gpxObject["feederCells"]),
                settings=settings,
            ),
        )

        results.append(
            ToolResults(
                gpxsol=sol,
                toolcosts=cast(dict[str, ConwipToolingCost], self.gpxObject["toolCosts"]),
                tools=cast(dict[str, ConwipTooling], self.gpxObject["tools"]),
                settings=settings
            ),
        )

        results.append(
            CellTimes(
                gpxsol=sol,
                cells=cells,
                cells_in_order=self.get_cells_in_order(),
                cell_results=cell_results,
                settings=settings
            ),
        )

        results.append(
            FloorSpace(
                gpxsol=sol,
                fscosts=cast(FloorspaceCost, self.gpxObject["floorspaceCost"]),
                cellresults=cell_results,
                settings=settings
            ),
        )

        results.append(
            LaborCost(
                gpxsol=sol,
                labcosts=cast(list[GPXLaborCost], self.gpxObject["laborCosts"]),
                cellresults=cell_results,
                settings=settings
            ),
        )

        results.append(ProductSummary(gpxsol=sol, settings=settings, resultslist=cast(list[ResultGenerator], results)))

        if self.has_offshift:
            offshift_dict: dict[str, gpx.Variable] = {
                var: self.gpx_variables[str(var)] for var in self._offshift_vars if var in self.gpx_variables
            }  # TODO: Find correct type - var comes through as a dict but used as a str key
            results.append(OffShiftResults(gpxsol=sol, offshiftdict=offshift_dict))

        # perform the round-up
        # self.cell_roundups(cellresults=cell_results)

        # return the combined results
        return cast(list[ResultGenerator], results)

    # TODO:  move this to a result generator
    def cell_roundups(self, cellresults: "CellResults") -> None:
        cell_margins = []
        for c in cellresults.cell_results:
            cname = c["name"]
            m = c["numWorkstations"]
            ccost = c["capitalCost"]
            m_ceil = np.ceil(m)
            ceil_ccost = m_ceil / m * ccost
            c["totalCapitalCeil"] = ceil_ccost
            c["totalCapitalCeilDiff"] = ceil_ccost - ccost
            cell_margins.extend([
                {
                    "name": cname,
                    "type": "Partial Capital Cost",
                    "value": ccost,
                },
                {
                    "name": cname,
                    "type": "Rounded Capital Cost",
                    "value": ceil_ccost - ccost,
                },
            ])

        # append to the results
        cellresults.results["costroundups"] = cell_margins
        cellresults.results_index.append({"name": "Capital Cost Roundups", "value": "costroundups"})

    def get_cells_in_order(self) -> list[tuple[int, str]]:
        """Return an ordered list of tuples of all of the gpx cells"""
        return sorted(self.cell_num_map.items(), key=lambda tup: tup[0])

    def get_all_cells(self) -> dict[str, gpx.manufacturing.QNACell]:
        "returns a dict of all cells in the module"
        cells = {
            **(self.gpxObject["cells"] if isinstance(self.gpxObject["cells"], dict) else {}),
            **(self.gpxObject["secondaryCells"] if isinstance(self.gpxObject["secondaryCells"], dict) else {}),
            **(self.gpxObject["feederCells"] if isinstance(self.gpxObject["feederCells"], dict) else {})
        }

        return cast(dict[str, QNACell], cells)

    def get_production_resources(
        self,
        filter_variables: bool = True,
        return_set: bool = False,
    ) -> list[gpx.Variable] | set[gpx.Variable]:
        """get a list of the production resource count varibales

        This is particularly useful for the discrete resource solves

        Returns
        -------
        list[gpx.Variable]
            _description_
        """
        if not self.gpxObject:
            raise ValueError("manufacturing module must have been gpx translated to get production resource variables")

        rescs = []
        # get the objects and their resource count handles
        objs = {"cells": "m", "feederCells": "m", "secondaryCells": "m", "tools": "L"}

        # rescs.extend([getattr(r, att) for obj, att in objs.items() for r in self.gpxObject[obj].values()])
        # return set(rescs) if return_set else rescs

        rescs.extend([(getattr(r, att), r) for obj, att in objs.items() for r in self.gpxObject[obj].values()])

        # if filter_variables:
        #     # make sure all entries are variables
        #     rescs = [r for r in rescs if isinstance(r, Variable)]

        return set(rescs) if return_set else list(rescs)

    @property
    def processes_by_id(self) -> dict[str, dict[str, AcceptedTypes]]:
        """Return a dict of processes indexed by the process id."""
        process_num_dict: dict[str, dict[str, AcceptedTypes]] = {
            str(p["id"]): p for p in self.processChain if "id" in p
        }  # primary processes
        process_num_dict.update({str(p["id"]): p for p in self.secProcesses})  # update for secondary process
        process_num_dict.update({str(p["id"]): p for p in self.feederProcesses})  # update for feeder processes

        return process_num_dict

    @property
    def processes_in_cell(self) -> None:
        "find the processes in the cell"
        # TODO: implement this using a filter function on the list of processes
        pass

    def get_cell_id_of_process(self, processid: str) -> str:
        "given a process number (primary or secondary), return the cell index where the process is"
        processes = self.processes_by_id
        processid = str(processid)

        return str(processes[processid]["cell"])

    def get_cell_idx_of_process(self, process: dict[str, AcceptedTypes] | str, idx: int = 1) -> int:
        "returns the cell index for a given process id"

        cell_name = (
            str(process["cell"])
            if isinstance(process, dict) and "cell" in process else self.get_cell_id_of_process(str(process))
        )  # get the name of the cell
        return self.cellnames.index(str(cell_name)) + idx  # find the ocurrance of the cell name and offset

    def get_cell_by_qna_name(self, qna_name: str) -> str:
        "gets the copernicus name based off of the qna cell name"
        for cname, c in self._gpx_all_cells.items():
            # if the name matches the qna name, return the cell name
            if c.lineagestr() == qna_name.strip():
                return cname

            if str(c).split("\n")[0] == qna_name.strip():
                return cname

        # if hasen't found a name, return None
        return ""

    @property
    def cellnames(self) -> list[str]:
        "get the names of all the cells in order"
        return list(self.cells.keys())

    @property
    def cellnamebynum(self, idx: int = 1) -> dict[int, str]:
        "the names of the cells in a dictionary"
        return {i + idx: self.cellnames[i] for i, _ in enumerate(self.cellnames)}

    def ramp_get_resource_int_vars(self) -> list[gpx.Variable]:
        "gets the variables that should be rampped"
        rescs = []  # the varkeys of the resources
        if len(getattr(self, "gpxObject", [])) == 0:
            # there are no objects
            raise ValueError("No resources to ramp were found")

        # Find all the resources
        # TODO:  consider relating back to input objects?
        # Cells
        celldictkeys = [
            "cells",
            "secondaryCells",
            "feederCells",
        ]  # list all of the keys in the gpxobject where the cells are
        # loop over the cell object dicts
        for cdk in celldictkeys:
            # loop over the cells in the dict and add to rescources
            rescs.extend([c.m for c in self.gpxObject.get(cdk, {}).values()])

        # Tools
        # append the tool counts
        rescs.extend([t.L for t in self.gpxObject.get("tools", {}).values()])

        # return all the resources
        return rescs

    def ramp_vars_by_uikey(self, keys: list[str] = []) -> dict[str, Variable]:
        "gets a dictionary of the of the gpx varkeys for the list of resources"

        resdict = {k: self.gpx_variables[k] for k in keys if k in self.gpx_variables}

        return resdict
