"""module-specific translators for copernicus modules to gpx models

Use
---
    Translation should put the appropriate gpx models into the gpx_models attribute in the module

"""
from collections import OrderedDict as OD
from collections import namedtuple
import itertools
import logging
import typing
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, cast

import gpkit
from gpkit import Variable
from gpkit.units import ureg
import numpy as np
import pint

from api.module_types.module_type import ModuleType
from api.small_scripts import remove_repeated_entries
import gpx.feeder
import gpx.manufacturing
from gpx.non_recurring_costs import CellCost, ConwipToolingCost
import gpx.offshift
import gpx.primitives
from gpx.serial import Series
from gpx.tooling import ConwipToolingByStop, ToolStop
import utils.logger as logger
from utils.types.data import Parameter
from utils.types.shared import AcceptedTypes
from utils.unit_helpers import is_continuous_unit

if TYPE_CHECKING:
    from api.module_types.manufacturing import Manufacturing


def manufacturing_module(
    default_currency: str, module: "Manufacturing", line_type: str = "cell", **kwargs: AcceptedTypes
) -> None:
    """translates the manufacturing module and adds the models to the module

    Arguments
    ---------
    module : Manufacturing
        the module to transform
    line_type : string
        the type of the manufacturing system
        supported types:

    Optional Keyword Arguments
    --------------------------
    min_rate : float
        required production rate in units per month

    Returns
    -------
    None
    """
    gpxvars: dict[str, Variable] = module.gpx_variables
    processes: dict[str, gpx.primitives.Process] = OD()
    cells: dict[str, gpx.manufacturing.QNACell] = OD()
    cellcosts: dict[str, CellCost] = OD()
    serialprocesses: list[gpx.primitives.Process] = []
    secondarycells: dict[str, gpx.manufacturing.QNACell] = {}
    # secondaryprocesses: dict[str, gpx.primitives.Process] = {}

    # check to make sure all variables are also in gpx_vars
    missingvars: list[Variable] = [vk for vk in module.variables.values() if vk not in gpxvars]

    # if missingvars:
    if missingvars:
        logging.warning(f"potential missing variables: {*missingvars, }")

        # check all variables simple
        logger.debug(
            f"all variablesSimple in variables: {all(vk in module.variables for vk in module.variablesSimple.keys())}",
        )

        # print the keys in gpxvars
        logging.warn(f"vars in gpxvars: {*list(gpxvars.keys()), }")
        # capture if any keys mismatch the parameter in the variable
        param_missmatch = [(vk, v.key) for vk, v in module.variables.items() if vk != v.key]
        if param_missmatch:
            logger.debug("mismatched dictkey and variable key")
            logger.debug("\n".join([str(p) for p in param_missmatch]))

        len_gpx: int = len(gpxvars)
        len_inputs: int = len(module.variables)
        if len_gpx != len_inputs:
            len_mismatch = (
                f"LENGTH MATCHING | INPUTS: {len_inputs} GENERATED: {len_gpx} IDENTIFIED_MISSING: {len(missingvars)}"
            )
            logger.debug(len_mismatch)
            # raise ValueError(len_mismatch)

        # raise ValueError(f'{len_mismatch} -- Failed to build model for module: manufacturing. Missing variables: {*missingvars,}')

    # cellnames = list(module.cells.keys())
    cellnames: list[str] = module.cellnames

    # cellaliasbynum = {i+1 : cellnames[i] for i in range(len(cellnames))}
    cellaliasbynum: dict[int, str] = module.cellnamebynum

    # dict of cell indicies by cell name
    # cellidxbyalias: dict[str, int] = {name: idx for idx, name in cellaliasbynum.items()}

    module.cell_num_map = cellaliasbynum

    if line_type == "cell":
        logging.info("translating the manufacturing module to gpx models for cell based system")

        # list of auxiliary constraints unassigned to a specific model
        aux_constraints: list[gpkit.ConstraintSet] = []

        # identify a set of unique primary processes
        # this allows for resuing process names to refer to the same process
        unique_process_names: set[str] = {str(p["type"]) for p in module.processChain}

        eps_t: Variable = Variable("\\epsilon_t", 1e-3, "min", "epsilon minutes")
        eps_cv: Variable = Variable("\\epsilon_{cv}", 1e-4, "-", "epsilon CV")
        # check to make sure all cell names are unique
        if len(cellnames) != len({*cellnames}):
            raise ValueError("Cell names must be unique!")

        # create only unique primary processes
        processes = {}

        # look for an all-processes cv
        cvvar: Variable = module.find_gpx_variable(
            filter_category="all processes",
            filter_property="CV",
            emptyisnone=True,
        )
        # create and add the process
        for name in unique_process_names:
            processes[name] = create_process(module=module, name=name, cvvar=cvvar, eps_t=eps_t, eps_cv=eps_cv)

        # find which process go into the cells using an OrderedDict to make sure
        # process order is maintained
        processes_in_cells: dict[str, list[str]] = OD()

        # find which processes go into which cells
        for p in module.processChain:
            if not p.get("cell") or str(p["cell"]).strip() == "":
                raise ValueError(f"Must assign a cell to process: {p.get('type', '')}")

            # if there is not a collector, add one
            if p["cell"] not in processes_in_cells.keys():
                processes_in_cells[str(p["cell"])] = []

            processes_in_cells[str(p["cell"])].append(str(p["type"]))
            # TODO: throw an error if the cell name is not in the cells list
            prev_cell_name = None  # the name of the previous cell

        for cellname, pic in processes_in_cells.items():
            if len(pic) == 1:  # just a single process in the cell
                proc = processes[pic[0]]  # select the gpx process based on the name
            else:
                print("\nSeries process for:", pic)

                # create a GPX serial process
                proc = Series(*[processes[name] for name in pic])

                # add to list of serial processes to collect later
                serialprocesses.append(proc)

            all_feeder_cell_ids = {str(fp["cell"]) for fp in module.feederProcesses}

            # find the name of the previous cell
            # prevcell_name = cellnames[i-1] if i>0 else None

            # add parallel cacpity to the cell
            try:
                cells[cellname] = add_parallel_capacity(
                    cellname,
                    module,
                    proc,
                    prevcell_name=prev_cell_name,
                    observe_queueing=True,
                    cellobj=module.cells[cellname],
                    aux_constr=aux_constraints,
                    all_feeder_cell_ids=all_feeder_cell_ids
                )
            except KeyError:
                # could not match process and cell
                raise ValueError(f'Cannot find cell "{cellname}" for process "{pic[0]}"')

            # update the name of the previous cell
            prev_cell_name = cellname

        # Primary off-shift processes
        # create the ton variable
        offshiftcells_main: dict[str, gpx.offshift.OffShiftCell] = {}
        if module.has_offshift:
            module.ton = Variable(
                "T_{on}",
                module._offshift_hrsPerShift * module._offshift_shiftsPerDay,
                "hrs",
                "On-Shift available production time",
            )
            # get the list of off shift vars
            offshiftcells_main = create_offshifts(module, cells, module._offshift_vars, module.ton)

        # Create the production Line
        if len(cells) == 0:
            # if there are no cells raise an issue
            raise ValueError("The main line must have at least one cell.")

        queue_time_var = module.find_gpx_variable(
            filter_category="line", filter_property="Queueing Time", emptyisnone=True
        )
        queue_wip_var = module.find_gpx_variable(
            filter_category="line", filter_property="Queue Inventory", emptyisnone=True
        )
        # create the manufacturing line with the list of cells in order of primary
        # production
        line: gpx.manufacturing.FabLine = gpx.manufacturing.FabLine(
            list(cells.values()),
            return_cells=False,
            total_queue_time=queue_time_var,
            total_queue_wip=queue_wip_var,
        )

        # find the aux constraints from any off-shift cells
        if module.has_offshift:
            addl_constr: Optional[list[gpkit.ConstraintSet]
                                  ] = update_offshift_line_constr(module, list(offshiftcells_main.values()), line)
            if addl_constr:
                aux_constraints.extend(addl_constr)

        # SECONDARY PROCESSES
        # add secondary processes which are unique
        # Argument 1 to <set> has incompatible type "*list[str | int | float | ProductionFinance | tuple[AcceptedTypes] | list[AcceptedTypes] | dict[str, AcceptedTypes] | None]"; expected "str"
        unique_secondary_processes_names: set[str] = {
            *[sp["type"] for sp in module.secProcesses if sp["type"] not in unique_process_names
              ],  # TODO: Find correct type
        }

        for name in unique_secondary_processes_names:
            processes[name] = create_process(name=name, module=module, cvvar=cvvar, eps_t=eps_t, eps_cv=eps_cv)

        # TODO:  create series processes if needed for secondary

        # create secondary cells
        for sp in module.secProcesses:
            name = str(sp["type"])
            # secondarycells[sp['cell']] = gpx.manufacturing.QNACell(processes[name])
            secondarycells[sp['cell']] = add_parallel_capacity(
                sp['cell'], module, processes[name], secondary_cell=True
            )  # adds workstation capacity
            # secondarycells[cellaliasbynum[sp['cell']]] = gpx.manufacturing.QNACell(processes[name])

        # map the routes for secondary flows
        secondary_constraints: list[gpkit.ConstraintSet] = []
        # add line constriant
        secondary_constraints.extend([line.lam <= cell.lam for cell in secondarycells.values()])

        # FEEDER PROCESSES
        # add feeder processes which are unique
        unique_feeder_processes_names: set[str] = {
            *[fp["type"] for fp in module.feederProcesses if fp["type"] not in processes],
        }

        for name in unique_feeder_processes_names:
            processes[name] = create_process(name=name, module=module, cvvar=cvvar, eps_t=eps_t, eps_cv=eps_cv)

        # create cells for feeder processes_by_id
        processes_in_cells: dict[str, list[str]] = group_processes_by_cell(module.feederProcesses)
        # create serial processes if necessary
        for key, pic in processes_in_cells.items():
            if len(pic) > 1:
                # serial process
                logger.debug(f"Module Translation|Serial Process for {*pic, }")
                proc = Series(
                    *[processes[name["type"]] for name in pic]
                )  # TODO: Find correct type, name is a str atm, not dict, need to look at processes_in_cells
                serialprocesses.append(proc)
                # replace the dictionary entry with the process
                processes_in_cells[key] = proc
            else:
                # just a single process
                if isinstance(pic[0], dict):
                    processes_in_cells[key] = processes[pic[0]["type"]]
                else:
                    raise TypeError(f"Expected a dictionary, but got {type(pic[0])}")

        feeder_gpx_cells: dict[str, gpx.manufacturing.QNACell] = {
            # cellname : gpx.manufacturing.QNACell(processes[process[0]['type']])
            cellname: add_parallel_capacity(
                cellname,
                module,
                process,
                aux_constr=aux_constraints,
            )  # considers capacity input
            for cellname, process in processes_in_cells.items()
        }

        # create the feeder lines from the lists of cells
        feeder_gpx_lines: dict[str, gpx.feeder.FeederLine] = {}
        feeder_aux_vars = filter(lambda x: x["category"] == "feeder line", module.exposedVariables)

        # list to put the feederline offshift cells into
        offshiftcells_feeder: dict[str, gpx.offshift.OffShiftCell] = {}

        # FEEDER LINE VARIABLE COSTS
        fl_var_costs = {}
        # object to track each variable cost that needs rolled up on the feeder line
        VarCostCalc = NamedTuple(
            "VarCostCalc", [
                ("fl_id", str),
                ("fl_obj", gpx.feeder.FeederLine),
                ("sum_cost", gpkit.Posynomial),
            ]
        )
        # collect the variable costs to calculate to get the quantity rollups
        fl_var_to_calc: typing.List[VarCostCalc] = []

        for fl in module.feederLines:
            line_offshiftcells: dict[str, gpx.offshift.OffShiftCell] = {}
            # get the cell names in each feeder
            cellsnames_in_feeder: list[str] = list(
                map(lambda y: str(y["cell"]), filter(lambda x: x["feederLine"] == fl["id"], module.feederProcesses)),
            )

            # get the id name of the feederline
            fl_id_str: str = fl["id"]

            # look for the feeder line quantity as part of the feeder line object
            flqty: Union[int, Variable] = fl.get("quantity")
            if not flqty:
                # look for the feeder line quantity in the variables simple
                flqtyvar_list: list[gpx.Variable] = cast(
                    list[gpx.Variable],
                    module.find_variables(
                        filter_category="feederLines",
                        filter_property="Quantity",
                        filter_type=fl_id_str,
                        emptyisnone=True,
                        return_data="list",
                    )
                )
                if flqtyvar_list is None:
                    # no variable was found
                    raise ValueError(f"Specify or constrain Feeder Line Quantity for {fl['id']}")

                # Ensure flqtyvar is always a list
                flqtyvar = flqtyvar_list[0]

                flqty = (
                    1 if isinstance(flqtyvar, Parameter) and flqtyvar.value == 1 else module.find_gpx_variable(
                        filter_category="feederLines",
                        filter_property="Quantity",
                        filter_type=fl_id_str,
                        emptyisnone=True,
                    )
                )

            # get all cells starting with cell ODict
            all_gpx_cells_dict: dict[str, gpx.manufacturing.QNACell] = {name: cell for name, cell in cells.items()}
            # add feeder_gpx_cells
            all_gpx_cells_dict.update(feeder_gpx_cells)
            # add secondarycells
            all_gpx_cells_dict.update(secondarycells)

            # get the feederline cells
            feedercellod: dict[str, gpx.manufacturing.QNACell] = OD()
            for cname in cellsnames_in_feeder:
                feedercellod[cname] = feeder_gpx_cells[cname]

            # check if the feederline is just empty
            feederline_is_empty: bool = not feedercellod

            # find any costs to the feeder line
            # find related costs
            costvars = module.find_gpx_variable(
                filter_category="feederLines",
                filter_property="Variable Cost",
                filter_type=fl_id_str,
            )

            cost_check = isinstance(costvars, gpkit.nomials.variables.Variable
                                    ) or (isinstance(costvars, list) and len(costvars) > 0)
            is_part_only_feeder_line = cost_check and feederline_is_empty
            # is_part_and_process_feeder_line = cost_check is not None and not feederline_is_empty

            # only proceed with building the feederline if there are cells
            if not feederline_is_empty:

                # create any offshift cells
                if module.has_offshift:
                    line_offshiftcells = create_offshifts(module, feedercellod, module._offshift_vars, module.ton)
                    # map any offshift cells to the feeder_gpx_cells
                    feeder_gpx_cells.update(line_offshiftcells)

                # # if feederline cell list is empty
                # if len(feedercellod) == 0:
                #     raise ValueError('Feeder Line "%s" is empty. There must be at least one cell.' % str(fl_id_str))

                # put the cells into a feederLine
                flobject = gpx.feeder.FeederLine(
                    cells=list(feedercellod.values()),
                    target_cell=all_gpx_cells_dict[module.get_cell_id_of_process(fl["to"])
                                                   ],  # the primary target cell of the feeder line
                    target_line=line,  # set the target line to the primary line
                    return_cells=False,  # get the cell constraints back
                    batch_qty=flqty,  # set the batching quantity
                    makeqtyvar=False,  # make the batch quantity a variable
                    dispname=fl_id_str,  # set the display name to the feeder line name
                )

                # update the feederline with any constraints from the offshift production
                if module.has_offshift:
                    addl_constr = update_offshift_line_constr(module, list(line_offshiftcells.values()), flobject)
                    if addl_constr:
                        aux_constraints.extend(addl_constr)

                # put the feederline together
                feeder_gpx_lines[fl_id_str] = flobject

                offshiftcells_feeder.update(line_offshiftcells)

                # update the feederline with any constraints from the offshift production
                # TODO: Duplication needs refactoring out
                if module.has_offshift:
                    addl_constr = update_offshift_line_constr(module, list(line_offshiftcells.values()), flobject)
                    if addl_constr:
                        aux_constraints.extend(addl_constr)

                feeder_gpx_lines[fl["id"]] = flobject

                offshiftcells_feeder.update(line_offshiftcells)

            elif is_part_only_feeder_line:
                # create the GPX object
                flobject = gpx.feeder.PartsOnlyFeederLine(
                    line_id=fl["id"],
                    target_line=line,
                    batch_qty=flqty,
                    dispname=fl_id_str,
                    default_currency=default_currency
                )
                feeder_gpx_lines[fl_id_str] = flobject

            # add the cost if exists
            if costvars:
                if not isinstance(costvars, list):
                    # make it a list
                    costvars = [costvars]
                # capture to roll up after all the feederlines are created
                fl_var_to_calc.append(VarCostCalc(fl_id=fl_id_str, fl_obj=flobject, sum_cost=np.sum(costvars)))

        # update the target lines for the feeders
        # collect all the feeder processes
        lines_by_process_id: dict[str, str] = {str(p["id"]): str(p["feederLine"]) for p in module.feederProcesses}
        lines_by_process_id.update({str(p["id"]): "" for p in module.secProcesses})
        # add the main line cells
        lines_by_process_id.update({str(p["id"]): "" for p in module.processChain})

        # go through each of the feeder lines
        for fl in module.feederLines:
            # get the line for the feed process
            feedline: str = lines_by_process_id[str(fl["to"])]
            if feedline:
                # call the update method
                feeder_gpx_lines[fl["id"]].update_target_line(feeder_gpx_lines[feedline])

        # update and generate the variable costs from the feeder line
        for flvc in fl_var_to_calc:
            # generate the actual variable cost now that counts should be updated
            fl_var_costs[flvc.fl_id] = gpx.recurring_cost.VariableCosts(
                flvc.fl_obj.cuml_batch_qty * flvc.sum_cost, default_currency=default_currency
            )

        # add the batch quantitiy variables to the module
        # for name, ln in feeder_gpx_lines.items():
        #     module.gpx_variables[name + ' Feeder Quantity'] = ln.batch_qty

        # create a dictionary of all the GPX cells keyed by name (alias)
        all_gpx_cells: dict[str, gpx.manufacturing.QNACell] = {**cells, **secondarycells, **feeder_gpx_cells}
        module._gpx_all_cells = all_gpx_cells

        # create the secondary routes referenceing cells by name
        try:
            for r in module.routes:
                if r["from"] != "" and r["to"] != "":
                    # find the target cell id from the process id
                    fromcell_id: str = module.get_cell_id_of_process(str(r["from"]))
                    tocell_id: str = module.get_cell_id_of_process(str(r["to"]))

                    # find the gpx cell objects
                    fromcell: gpx.manufacturing.QNACell = all_gpx_cells[fromcell_id]
                    tocell: gpx.manufacturing.QNACell = all_gpx_cells[tocell_id]

                    # connect the flow of variances
                    secondary_constraints.append(tocell.c2a >= fromcell.c2d)
        except TypeError:
            # we are sometimes getting `0` instead of `[]` to signify an empty route
            pass

        # CELL COSTS
        for name, cell in all_gpx_cells.items():
            # cellcosts[name] = CellCost(cell, workstation_cost=old_associate_cost(name, module))
            cellcosts[name] = CellCost(
                cell,
                workstation_cost=associate_cost(
                    module,
                    name,
                    "cells",
                    default_currency=default_currency,
                ),
                default_currency=default_currency
            )

        total_W: Variable = Variable("W_{total}", "min", "Total flow time for Invholding")
        # yapf: disable
        secondary_constraints.append(
            total_W
            >= line.W
            + np.sum([c.W for c in secondarycells.values()])
            + np.sum([fc.W for fc in feeder_gpx_cells.values()]),  # TODO: add better pressure for feeder lines
        )
        # yapf: enable

        # inv_total_L = Variable('L_{inventory-total}', 'count', 'The total inventory for counting inventory costs')
        # secondary_constraints.append(
        #     inv_total_L >= line.L + np.sum([fl.L for fl in feeder_gpx_lines.values()])    # also include the inventory in the feeder lines
        # )

        # Inventory holding costs
        # TODO: separate inventory holding costs for feeder lines
        invholding: gpx.recurring_cost.InventoryHolding = gpx.recurring_cost.InventoryHolding(
            inv_count=line.L, flow_time=total_W, default_currency=default_currency
        )

        # ADDITIONAL COSTS
        varcosts = {}
        recurcosts = {}
        nonreccosts = {}

        for name, costvar in module.costs.items():
            # prop = ' '.join([costvar['type'], 'Cost'])
            prop = costvar["type"]
            var = module.find_gpx_variable(
                filter_category="costs",
                filter_property=prop,
                filter_type=name,
                emptyisnone=True,
            )

            if var is not None:
                # only if the variable is constrained
                try:
                    if costvar["type"] == "Non-Recurring":
                        nonreccosts[name] = gpx.non_recurring_costs.NonRecurringCost(
                            var, default_currency=default_currency
                        )
                    elif costvar["type"] == "Recurring":
                        recurcosts[name] = gpx.recurring_cost.RecurringCosts(var, default_currency=default_currency)
                    elif costvar["type"] == "Variable":
                        varcosts[name] = gpx.recurring_cost.VariableCosts(var, default_currency=default_currency)
                except Exception as e:
                    # import pdb; pdb.set_trace()
                    logging.info(f"MODULTE-TRANSLATION | ERROR Raised: {e} Converting to ValueError {name}")
                    raise ValueError(f"Could not create cost for {name}. Check definition.")

        # add feederline costs to list of other variable costs
        if fl_var_costs:
            varcosts.update({f"{fn} Variable Cost": fcost for fn, fcost in fl_var_costs.items()})

        # ADD TOOLING COSTS
        toolcosts: dict[str, ConwipToolingCost] = {}
        tools: dict[str, gpx.tooling.ConwipToolingByStop] = {}
        cells_in_order_list: list[gpx.manufacturing.QNACell
                                  ] = [all_gpx_cells[c] if c in all_gpx_cells else None for c in module.cellnames]

        # for each tool, create the gpx cost object
        for tool in module.tools:
            # catch if path is an empty string
            if "route" in tool:
                if isinstance(tool["route"], list) and len(tool["route"]) == 0:  # empty tool route
                    process_route: Optional[list[str]] = None
                else:
                    process_route = cast(list[str], tool["route"])
            else:
                process_route = None

            if process_route is None:
                # if there is no tooling, break out of the loop
                break

            # TODO   look to see if there are any process ranges in the list
            # create a list of all the processes in order
            prnew: list[str] = []
            # list of processes only from primy flow
            pplist: list[str] = [str(p["id"]) for p in module.processChain]

            # loop over steps in the process route
            i: int = 0

            if not isinstance(process_route, list):
                raise TypeError(f"Expected process_route to be a list, but got {type(process_route)}")

            while i < len(process_route):
                # check to see if it's not the last element and it's part of a range
                if (i + 1) < len(process_route) and process_route[i + 1].strip() == "-":
                    pstart: str = str(process_route[i])
                    pend: str = str(process_route[i + 2])

                    if pstart not in pplist or pend not in pplist:
                        # only allow ranges in primary flow
                        raise ValueError(
                            f"Process {pstart} or {pend} not in primary path. Must be primary process to specify with range",
                        )

                    pstartidx: int = pplist.index(pstart)
                    pendidx: int = pplist.index(pend) + 1

                    # form the range
                    prange: list[str] = pplist[pstartidx:pendidx]
                    # add to the process list
                    prnew.extend(prange)

                    # skip i past the range that was just added
                    i += 3

                else:
                    # if not part of a range, add the process directly
                    prnew.append(str(process_route[i]))
                    i += 1

            # replace the process route with the new one
            process_route = prnew

            # convert to entry and processess format
            # get the cell name for every process
            ToolPath = namedtuple("ToolPath", "entry processes")
            paths: list[ToolPath] = []

            # find the tool path groupings
            def grouper_func(x: str) -> int:
                return module.get_cell_idx_of_process(x)

            for cell_index, ps in itertools.groupby(process_route if isinstance(process_route, list) else [],
                                                    key=grouper_func):
                # get the actual cell object
                # the results from the `get_cell_idx` are 1-indexed
                cell = cells_in_order_list[cell_index - 1]
                # get the process objects that correspond to the process id
                p_names: list[str] = [
                    str(module.processes_by_id[str(p)]["type"]) for p in ps if isinstance(p, (str, int, float))
                ]
                paths.append(ToolPath(entry=cell, processes=[processes[pn] for pn in p_names]))

            # convert the paths to a list of tool stop objects
            tool_stops: list[ToolStop] = [ToolStop(entrance_cell=p.entry, processes=p.processes) for p in paths]

            # convert from process route to cell route
            tool_route: list[dict[str, AcceptedTypes]] = [
                cast(dict[str, AcceptedTypes], module.get_cell_idx_of_process(p)) for p in process_route
            ]  # Unused variable 'tool_route'  [unused-variable]

            # remove repeated cell indicies
            tool_route_condensed: list = remove_repeated_entries(tool_route)

            toolname: str = str(tool["name"])

            # toolmodel = gpx.manufacturing.ConwipTooling(
            #     cells_in_order_list,
            #     indicies=tool_route_condensed,
            #     capacity=module.find_gpx_variable(filter_category='tools',
            #                                       filter_property=['Capacity', 'Lot Size'],
            #                                       filter_type=toolname)
            # )

            toolmodel: gpx.tooling.ConwipToolingByStop = gpx.tooling.ConwipToolingByStop(
                tool_stops=tool_stops,
                capacity=module.find_gpx_variable(
                    filter_category="tools",
                    filter_property=["Capacity", "Lot Size"],
                    filter_type=toolname,
                ),
            )

            tools[toolname] = toolmodel

            # Create the tooling cost model ONCE
            toolcosts[toolname] = ConwipToolingCost(
                toolmodel,
                tool_cost=associate_cost(
                    module,
                    toolname,
                    "tools",
                    default_currency=default_currency,
                ),
                default_currency=default_currency,
            )

        # FLOORSPACE COSTS
        floorspacecost: dict[str, gpx.mfgcosts.FloorspaceCost] = {}
        # TODO:  if there is a footprint variable, automatically create the floor space cost
        # check that there is a GPX variable defined for floorpsace price
        floorspacevars: Optional[Union[list[str], list[Parameter], dict[str, Parameter]]] = module.find_variables(
            filter_category="cells",
            filter_property="Floor Space",
            return_data="list",
            emptyisnone=True,
        )

        if floorspacevars is not None:
            # find the total footprint for the cells

            # collect variables
            # find the cell space vars
            cell_space: dict[str, Variable] = {}
            for fsv in floorspacevars:
                if fsv.type in all_gpx_cells:
                    # make sure the cell exists
                    if fsv.key in module.gpx_variables:
                        cell_space[fsv.type] = module.gpx_variables[fsv.key]
                    elif fsv.name in module.gpx_variables:
                        cell_space[fsv.key] = module.gpx_variables[fsv.name]
                    else:
                        # there was not the floor space variable found
                        errstr: str = f'Cannot find variable "{fsv.name}" for cell "{fsv.type}" while creating floor space'
                        logging.error(f"Translation | {errstr}")
                        raise ValueError(errstr)

            # create the recurring and non-recurring costs
            # check to make sure there is a cost defined:
            # if there is not a cost defined, the corresponding cost is 0
            # TODO:  correct the GPX cost generator to handle 0 (using epsilon for now)

            # floor space recurring cost
            fsrc: Optional[Variable] = module.gpx_variables.get("Floor Space Recurring Cost")

            # floor space non-recurring cost
            fsnrc: Optional[Variable] = module.gpx_variables.get("Floor Space Non-Recurring Cost")

            if fsnrc is None and fsrc is None:
                # check to make sure there is at least one floor space cost
                raise ValueError("A footprint is defined. Specify at least one floor space cost.")

            # create the cost object
            floorspacecost = gpx.mfgcosts.FloorspaceCost(
                cells=all_gpx_cells,
                cell_space=cell_space,
                cost_r=fsrc,
                cost_nr=fsnrc,
                default_currency=default_currency
            )

        # LABOR COSTS

        laborcosts: list[gpx.recurring_cost.ProcessLaborCost] = []

        # find if there are any labor cost variables
        labcostvars: list[Variable] | None = cast(
            list[Variable] | None,
            module.find_variables(
                filter_category="cells",
                filter_property="Head Count",
                return_data="list",
                emptyisnone=True,
            )
        )

        # get the process headcount labor costs
        proclabcostvars: list[Variable] | None = cast(
            list[Variable] | None,
            module.find_variables(
                filter_category="processes",
                filter_property="Head Count",
                return_data="list",
                emptyisnone=True,
            )
        )

        # create the list of prpcesses in each cell
        cellprocesses: dict[str, list[str]] = {}
        for c in all_gpx_cells.keys():
            # loop over all cell names
            procs: list[gpx.primitives.Process] = [
                p["type"] for p in filter(
                    lambda pp: pp["cell"] == c,
                    [*module.processChain, *module.feederProcesses, *module.secProcesses],
                )
            ]
            # TODO:  also add the processes from the feeder lines
            cellprocesses[c] = procs

        # creat the labor costs for the whole cells
        laborcosts = []
        if labcostvars or proclabcostvars:
            # find the total labor for the cells
            # make the dict tuples for the laborcost model
            if labcostvars:
                try:
                    # try getting headcount by key
                    headcts: dict[str, tuple[gpx.manufacturing.QNACell, Variable]] = {
                        hcv.type: (all_gpx_cells[hcv.type], module.gpx_variables[hcv.key]) for hcv in labcostvars
                    }
                except KeyError:
                    # if that fails, get the variables by their full name
                    headcts = {
                        hcv.type: (all_gpx_cells[hcv.type], module.gpx_variables[hcv.name]) for hcv in labcostvars
                    }
            else:
                headcts = {}

            # find the cost inputs
            labrc: Optional[Variable] = module.gpx_variables.get("Labor Variable Cost")
            if labrc is None:
                raise ValueError('Headcounts found. Define "Labor Variable Cost"')

            # find the cells in the feeder lines and the quantities

            procheadct: dict[str, tuple[gpx.manufacturing.Process, Variable]] = {}

            # find the variables for the process headcounts
            # TODO:  need to find the cell where the process is
            if proclabcostvars:
                try:
                    # try getting headcount by key
                    procheadct = {
                        hcv.type: (processes[hcv.type], module.gpx_variables[hcv.key]) for hcv in proclabcostvars
                    }
                except KeyError:
                    # if that fails, get the variables by their full name
                    procheadct = {
                        hcv.type: (processes[hcv.type], module.gpx_variables[hcv.key]) for hcv in proclabcostvars
                    }
            else:
                procheadct = {}

            # get the feeder line cell quantities
            feeder_cell_qty: dict[str, gpx.Variable] = feeder_qty_by_cell(
                feeder_gpx_lines, cast(list[dict[str, str]], module.feederProcesses)
            )

            # create labor costs with process headcounts
            # only create the object if there are labor costs
            laborcosts = [
                gpx.recurring_cost.ProcessLaborCost(
                    processes_in_cells=cellprocesses,
                    all_processes=processes,
                    all_gpx_cells=all_gpx_cells,
                    cells_headcount=headcts,
                    process_headcount=procheadct,
                    feeder_cell_qty=feeder_cell_qty,
                    laborRate=labrc,
                    default_currency=default_currency
                ),
            ]

        # ADD ADVANCED VARIABLES

        # define a map of the aux variable name to the GPX object attribute
        cell_attr_map: dict[str, Optional[str]] = {
            # 'workstation count' : 'm',
            "queueing time": "Wq",
            "flow time": "W",
            "utilization": "rho",
            "wip inventory": None,
            "queue inventory": None,
            # 'parallel capacity' : 'k',  # added when object is created
        }

        # aux_units_map: dict[str, str] = {
        #     "workstation count": "count",
        #     "queueing time": "min",
        #     "flow time": "min",
        #     "count": "count",
        #     "utilization": "-",
        #     "wip inventory": "count",
        #     "production rate": "count/hr",
        # }

        subs_to_delete: dict[str, list[str]] = {"cell": ["parallel capacity"]}

        # find the cell exposed variables
        cell_exposed_variables: list[dict[str, str]
                                     ] = [item for item in module.exposedVariables if item["category"] == "cells"]
        for cell_exposed_variable in cell_exposed_variables:
            # map the variable in module.gpxvars to the GPX object attribute
            # add equivalence constraint between the two
            if cell_exposed_variable["property"].lower() in cell_attr_map:
                # serach using lowercase!
                if str(cell_exposed_variable["type"]) not in all_gpx_cells:
                    logging.warning("could not find cell {} in all cells. probably not used by any process. skipping")
                else:
                    try:
                        # look for WIP calculations
                        if cell_exposed_variable["property"].lower() == "wip inventory":
                            # handle wip inventory manually since not automatically
                            # calculated at cell
                            cell = all_gpx_cells.get(str(cell_exposed_variable["type"]))
                            if cell:
                                aux_constraints.append(
                                    module.gpx_variables[cell_exposed_variable["name"]] == cell.lam * cell.W
                                )  # TODO: Find correct type
                            else:
                                raise ValueError("Could not find cell %s for calculating WIP inventory")
                        elif cell_exposed_variable["property"].lower() == "queue inventory":
                            # handle the queueing inventory separately
                            cell = all_gpx_cells.get(cell_exposed_variable["type"])
                            if cell:
                                aux_constraints.append(
                                    module.gpx_variables[cell_exposed_variable["name"]] == cell.lam * cell.Wq
                                )
                            else:
                                raise ValueError("Could not find cell %s for calculating Queue inventory")
                        else:
                            # use the attribute chart
                            aux_constraints.append(
                                module.gpx_variables[cell_exposed_variable["name"]]
                                # yapf: disable
                                == getattr(
                                    all_gpx_cells[cell_exposed_variable["type"]],
                                    cell_attr_map[cell_exposed_variable["property"].lower()] or ""
                                )
                                # yapf: enable
                            )

                        if cell_exposed_variable["property"] in subs_to_delete["cell"]:
                            # append the variable to the module-level substituion delete
                            module._del_subs_.append(
                                getattr(
                                    all_gpx_cells[cell_exposed_variable["type"]],
                                    cell_attr_map[cell_exposed_variable["property"].lower()] or ""
                                )
                            )
                    except KeyError as err:
                        raise ValueError(f"{cell_exposed_variable['name']} not found in constraints") from err

        # find the tool constraints
        tool_attr_map: dict[str, str] = {"Count": "L"}
        tools_exposed_variables = filter(lambda x: x["category"] == "tools", module.exposedVariables)
        for tools_exposed_variable in tools_exposed_variables:
            # map the variable in module.gpxvars to the GPX object attribute
            # add equivalence constraint between the two
            try:
                aux_constraints.append(
                    module.gpx_variables[tools_exposed_variable["name"]]
                    # yapf: disable
                    == getattr(
                        toolcosts[tools_exposed_variable["type"]], tool_attr_map[tools_exposed_variable["property"]]
                    ),
                    # yapf: enable
                )
            except KeyError as err:
                raise ValueError(f"{tools_exposed_variable['name']} not found in constraints") from err

        # line advanced variables
        line_attr_map: dict[str, str] = {
            "WIP Inventory": "L",
            "Flow Time": "W",
            "Production Rate": "lam",
            "Queue Inventory": "total_queue_wip",
            "Queueing Time": "total_queue_time",
        }
        line_exposed_variables = filter(lambda x: x["category"] == "line", module.exposedVariables)
        for v in line_exposed_variables:
            try:
                attr_name = line_attr_map[v["property"]]
                attr_val = getattr(line, attr_name, None)

                if attr_val is None:
                    continue

                aux_constraints.append(module.gpx_variables[v["name"]] == attr_val)

            except KeyError as err:
                raise ValueError(f"{v['name']} not found in constraints") from err

        # Floor Space variables
        floorspace_attr_map: dict[str, str] = {"Total Floor Space": "totalFloorSpace"}
        floorspace_exposed_variables = filter(lambda x: x["category"] == "floor space", module.exposedVariables)
        for floorspace_exposed_variable in floorspace_exposed_variables:
            try:
                aux_constraints.append(
                    module.gpx_variables[floorspace_exposed_variable["name"]]
                    # yapf: disable
                    == getattr(floorspacecost, floorspace_attr_map[floorspace_exposed_variable["property"]]),
                    # yapf: enable
                )
            except KeyError as err:
                raise ValueError(f"{floorspace_exposed_variable['name']} not found in constraints") from err

        # add the feederLine advanced variables
        # feeder line quantity is added when the gpx object is created

        # list of the costs from each step in the rate ramp
        rampcosts: list[list[Variable]] = []

        # TODO:  if there is a rate ramp, create and add the
        if hasattr(module, "rateramp"):
            # if there is a rmp object, generate the rate ramp

            # TODO:  eventually move this plugin to the financial module
            #       the mincost context should be getting the unit cost from the finance module, not mfg
            # there is a rate ramp
            # create a rate ramp object
            # have to get the ramp up time from the

            module.rateramp.add_costs(
                *list(toolcosts.values()),
                *list(cellcosts.values()),
                # *list(floorspacecost.values()),
                floorspacecost,
                # varcosts,
                *list(varcosts.values()),
                *list(recurcosts.values()),
                invholding,
                *list(nonreccosts.values()),
            )

            module.rateramp.add_labor_costs(*laborcosts)

            rampcosts = [module.rateramp.get_aux_cost()]

        # pass the different cost objects to the totaler
        # ASSEMBLED COMPONENTS
        # find the connections to the assembled components
        assm_costs: dict[str, gpx.recurring_cost.VariableCosts] = {}  # assembly variable costs
        if hasattr(module, "assembly_module"):
            # TODO:  get the list of the variable costs to add to the unit cost
            # module generation ensures that the assembly is made first
            # get the rate links
            assm_aux_constraints: list[gpkit.ConstraintSet] = []
            assm_rate_mons = module.assembly_module.get_rate_constraintset()
            assm_aux_constraints = [line.lam == mon for mon in assm_rate_mons]

            # get the unit costs as a variables
            assm_varcosts: dict[str, gpx.recurring_cost.VariableCosts] = module.assembly_module.get_variable_cost(
                asdict=True
            )

            # TODO:  this is adding the unit cost twice
            #       because we are also adding the assembly costs directly to the unit cost
            #       see if this can be removed
            # varcosts.update(assm_varcosts)

            # append costs to the assembly variable costs
            assm_costs.update(assm_varcosts)

            # add varcosts to constraints
            assm_aux_constraints.extend(list(assm_varcosts.values()))

            # add all the assembly auxiliary constraints to the module auxiliary
            # constraints
            aux_constraints.extend(assm_aux_constraints)

        # pass the different cost objects to the totaler for the unit cost
        # some costs that are dicts are fist converted to lists
        # collections are splatted for the unit cost function
        unitcost: gpx.primitives.UnitCost = gpx.primitives.UnitCost(
            *list(toolcosts.values()),
            *list(cellcosts.values()),
            # *list(floorspacecost.values()),
            floorspacecost,
            # varcosts,
            *list(varcosts.values()),
            *list(recurcosts.values()),
            invholding,
            *list(nonreccosts.values()),
            # laborcosts,
            # *list(laborcosts.values()),
            *laborcosts,
            *list(assm_costs.values()),
            # variable costs from assembled component unit costs
            # *assm_varcosts,
            # rate ramp costs
            *rampcosts,
            return_costs=False,
            num_units=None,
            rate=line.lam,
            nonrecurringCost=module.gpx_variables.get("Total Non-Recurring Cost"),
            recurringCost=module.gpx_variables.get("Total Recurring Cost"),
            variableCost=module.gpx_variables.get("Total Variable Cost"),
            default_currency=default_currency
        )

        # connect the assembly financial constraints
        if hasattr(module, "assembly_module"):
            # set holding rate and horizon equal to the parent
            aux_constraints.extend(
                module.assembly_module.get_finance_constraints(unitcost.horizon, invholding.holdingRate),
            )

        gpx_cell_names: list[str] = [f"{name} : {str(cell).split('\n')[0]}" for name, cell in all_gpx_cells.items()]

        # save the cell names to the object

        logging.info("===CELL VALUES===\n" + "\n".join(gpx_cell_names))

        # save all the off shift cells to the module
        module.gpx_offshift = {**offshiftcells_main, **offshiftcells_feeder}

        module.gpxObject = {
            "processes": processes,
            "cells": cells,
            "cellCosts": cellcosts,
            "fabLine": line,
            "unitCost": unitcost,
            "tools": tools,
            "toolCosts": toolcosts,
            "varCosts": varcosts,
            "recurringCosts": recurcosts,
            "invHolding": invholding,
            "serialProcesses": serialprocesses,
            "secondaryConstraints": secondary_constraints,
            "secondaryCells": secondarycells,
            "nonRecurringCost": nonreccosts,
            "feederCells": feeder_gpx_cells,
            "auxiliaryConstraints": aux_constraints,
            # 'feederLines'           : list(feeder_gpx_lines.values()),
            "feederLines": feeder_gpx_lines,
            "floorspaceCost": floorspacecost,
            "laborCosts": laborcosts,
            "rampCosts": rampcosts,
            "assmVarcosts": assm_costs,
            # 'feederLines' : list(itertools.chain.from_iterable(feeder_gpx_lines.values())),
            # 'nonRecurringCostVars' : nonrecvars,
            # 'laborCosts' : [laborcosts],
            "accumValue": [invholding.inventoryValue >= unitcost.unitCost],
            # 'lineConstraints' : line_constraints
        }

    else:
        raise Exception(f"system models not implemented for line type: {line_type}")


def group_processes_by_cell(items: list[dict[str, str]], cellkey: str = "cell") -> dict[str, list[dict[str, str]]]:
    """given an iterable of items return a dictionary

    Returns
    -------
    dict
        cell_name : list of process names

    """
    # create a set of cell names
    try:
        cell_names: set[str] = {*[p[cellkey] for p in items]}
    except KeyError as e:
        logging.error(str(e))
        raise ValueError("Feeder processes require cell inputs")

    # create a list of processes for every cell
    processes_in_cell: dict[str, list[dict[str, str]]] = {
        name: list(filter(lambda x: x[cellkey] == name, items)) for name in cell_names
    }

    return processes_in_cell


def add_parallel_capacity(
    cellname: str,
    module: "Manufacturing",
    process: gpx.primitives.Process,
    prevcell_name=None,
    observe_queueing=False,
    secondary_cell=False,
    cellobj: dict[str, str] = {},
    aux_constr: Optional[set[gpkit.ConstraintSet] | list[gpkit.ConstraintSet]] = None,
    all_feeder_cell_ids: set[str] = set(),
) -> gpx.manufacturing.QNACell:
    """new way of creating a cell with parallel capacity

    Arguments
    ---------
    observe_queueing : boolean (default=False)
        look to see if there should be queueing allowed in the cell
    cellobj : dict
        incldues the definition of the cell
    """
    try:
        capvar = module.find_variables(
            filter_category="cells",
            filter_type=cellname,
            filter_property=["Capacity", "Batch Size"],
            return_data="list",
        )
        if isinstance(capvar, list):
            capvar = capvar[0]
    except IndexError:
        raise ValueError(f'Missing cell "{cellname}"')

    # find the capacity of the previous cell
    rebatch_from_previous = False
    if prevcell_name:
        prevcell_cap = module.find_variables(
            filter_category='cells',
            filter_type=prevcell_name,
            filter_property=['Capacity', 'Batch Size'],
            return_data='list'
        )[0]
        try:
            if capvar.value > prevcell_cap.value:
                # batching needs to be enforced when there is a change in batch size
                logging.debug(f'setting batching to true from {prevcell_name} to {cellname}')
                rebatch_from_previous = True
        except TypeError:
            logging.debug(f'Could not directly compare cell capacity for {prevcell_name} into {cellname}')

    if capvar.value == 1:
        cap: int | gpx.Variable | list[gpx.Variable] | None = 1
    else:
        cap = module.find_gpx_variable(
            filter_category="cells",
            filter_type=cellname,
            filter_property=["Capacity", "Batch Size"],
        )

    # find if the cell has queueing before
    queuatcell = True
    if observe_queueing:
        try:
            if cellobj["noQueueing"]:
                queuatcell = False
        except KeyError:
            pass

    # look for mttr, mttf
    mttr = module.find_gpx_variable(
        filter_category="cells",
        filter_type=cellname,
        filter_property="MTTR",
        emptyisnone=True,
    )
    mttf = module.find_gpx_variable(
        filter_category="cells",
        filter_type=cellname,
        filter_property="MTTF",
        emptyisnone=True,
    )
    motf = module.find_gpx_variable(
        filter_category="cells",
        filter_type=cellname,
        filter_property="Mean Operations to Failure",
        emptyisnone=True,
    )

    # alternatively, look directly for machine availability
    avail = module.find_gpx_variable(
        filter_category="cells",
        filter_type=cellname,
        filter_property="Uptime",
        emptyisnone=True,
    )

    if mttr is not None and mttf is None and motf is None:
        raise ValueError(f'MTTR defined for "{cellname}" but not MTTF or "Mean Operations to Failure"')
    if mttf is not None and motf is not None:
        raise ValueError(f'Specify only one of "MTTF" or "Mean Operations to Failure" for {cellname}')
    if mttr is None and (mttf is not None or motf is not None):
        raise ValueError(f'MTTF or "Mean Operations to Fail" defined for "{cellname}" but not MTTR')

    e: gpkit.nomials.math.Posynomial = 1
    if mttr is not None:
        mttr = cast(Variable, mttr)
        # define nu based on availability
        if motf is not None:
            motf = cast(Variable, motf)
            # if using the motf, create a variable for mttf
            mttf = Variable("mttf", "hr", "mean time to fail")

        e = (mttr + mttf) / mttf

    if avail:
        avail = cast(Variable, avail)
        # throw error if availability and mttr both defined
        if mttr:
            raise ValueError(f"{cellname} Uptime and MTTR cannot both be defined")
        # override mttr mttf with availability if defined
        e = avail**-1

    kwargs = {"capacity": cap, "queueing_at_cell": queuatcell, "e": e}

    # if there is the variable for the workstation quantity and substitute directly
    # look for the count in exposed variables
    # flake8: noqa
    mvar_filtered: filter[dict[str, str]] = filter(
        # yapf: disable
        # pylint: disable=all
        lambda x: x["category"].lower() == "cells" and x["type"] == cellname and x["property"].lower() ==
        "workstation count",
        # pylint: enable=all
        # yapf: enable
        module.exposedVariables,
    )
    mvar: list[dict[str, str]] = list(mvar_filtered)
    # get the variable
    if mvar:
        # get the first element if there is a list
        mvar_list_item = mvar[0]
        mvar_list_item_name = module.gpx_variables.get(mvar_list_item["name"], None)
        # add the substitution directly to the cell constructor
        kwargs["m"] = mvar_list_item_name

    is_feeder = cellname in all_feeder_cell_ids

    if is_feeder:
        cell = gpx.manufacturing.FeederQNACell(process, **kwargs)
    elif hasattr(cap, "units") and is_continuous_unit(cap.units):
        cell = gpx.manufacturing.FeederQNACell(process, **kwargs)
    else:
        cell = gpx.manufacturing.QNACell(
            process, rebatch_from_previous=rebatch_from_previous, secondary_cell=secondary_cell, **kwargs
        )

    # if calculating mttf from motf, append the constraint
    if motf is not None:
        aux_constr.append(mttf == motf * cell.t / cell.rho)

    return cell


def create_process(
    name: str,
    module: "ModuleType",
    cvvar: float | None = None,
    eps_t: float = 1e-5,
    eps_cv: float = 1e-4,
    filter_category: str = "processes",
) -> gpx.manufacturing.Process:
    if module.find_variables(filter_category=filter_category, filter_property="Time", filter_type=name):
        # there is a time variable
        ptime = module.find_gpx_variable(
            filter_category=filter_category,
            filter_property="Time",
            filter_type=name,
            emptyisnone=True,
        )

    else:
        # there is no entry for time, assume epsilon
        logging.info(f"time variable not found for process: {name}")
        ptime = eps_t

    # check to make sure not cheating FPY
    try:
        fpy_vars = module.find_variables(
            filter_category=filter_category,
            filter_property="FPY",
            filter_type=name,
            return_data="list",
        )

        if fpy_vars and isinstance(fpy_vars, list):
            percentage_quantity = fpy_vars[0].value * ureg.pct
            if percentage_quantity.to_base_units() > 1:
                raise ValueError(f'"{name}" first pass yield exceed 100%')
    except (IndexError, AttributeError):
        # no variable found
        pass

    fpy = module.find_gpx_variable(
        filter_category=filter_category,
        filter_property="FPY",
        filter_type=name,
        emptyisnone=True,
    )
    fpy = 1 if fpy is None else fpy

    if cvvar is None:
        # TODO:  in the future, should we still look for a process CV even if the all process cv is defined
        # defined the standard deviation by process
        stdev = module.find_gpx_variable(
            filter_category=filter_category,
            filter_property="Std. Dev.",
            filter_type=name,
        )  # find the

        if stdev:
            # create the process and add it to the OD
            return gpx.manufacturing.Process(time=ptime, stdev=stdev, fpy=fpy)

        # if standard deviation is not defined, substitute epsilon
        cvvar = eps_cv

    # create the process using the standard cv
    return gpx.manufacturing.Process(time=ptime, cv=cvvar, fpy=fpy)


def aux_var_from_map(
    attr_map: dict[str, str],
    filter_category: str,
    module: "Manufacturing",
    objects: dict[str, Union[gpx.manufacturing.QNACell, ConwipToolingByStop, gpx.manufacturing.FabLine]],
) -> list[gpkit.ConstraintSet]:
    """add auxiliary variables based on attribute maps and objects

    Arguments
    ---------
    attr_map : dict
        {<Attribute name> : <attribute key in model>}
        maps from attribute descriptions to the model attributes
    filter_category : string
        the category name over which to filter the aux input variables
    module :
        module object on which to operate
    objects :
    """
    aux_constraints: list[gpkit.ConstraintSet] = []

    for c in filter(lambda x: x["category"] == filter_category, module.exposedVariables):
        try:
            aux_constraints.append(
                module.gpx_variables[c["name"]] == getattr(objects[c["type"]], attr_map[c["property"]]),
            )
        except KeyError as err:
            raise ValueError(f"{c['name']} not found in constrains") from err

    return aux_constraints


def find_process_cell(
    processname: str,
    celldict: dict[str, gpx.manufacturing.QNACell],
    module: "Manufacturing",
    return_name_only: bool = True,
) -> Union[str, gpx.manufacturing.QNACell]:
    "return the gpx cell where the process is"

    # turn the process chain into a dict
    procdict: dict[str, str] = {str(p["id"]): str(p["cell"]) for p in module.processChain}

    # find the cell name
    cellname: str = procdict[processname]

    if return_name_only:
        return cellname

    return celldict[cellname]


def associate_cost(
    module: "Manufacturing",
    filter_type: str,
    filter_category: str,
    default_currency: str,
) -> gpx.primitives.Cost:
    """associates a full cost object to an object name

    Arguments
    ---------
    module
        input module
    filter_type : str
        the name on which to filter
    filter_category : str
        the category to filter

    Returns
    -------
    gpx.primitives.Cost
        the multi-component cost object

    """
    cost_mapping: dict[str, str] = {
        "Non-Recurring Cost": "nonrecurringCost",
        "Recurring Cost": "recurringCost",
        "Variable Cost": "variableCost",
    }

    cost_obj: gpx.primitives.Cost = gpx.primitives.Cost(default_currency=default_currency)

    for key, attribute in cost_mapping.items():
        # loop through all the different costs

        # find the variable
        var: Optional[gpx.Variable] = module.find_gpx_variable(
            filter_category=filter_category,
            filter_type=filter_type,
            filter_property=key,
        )

        if not var:
            # If there is no variable found, set it to 0
            setattr(cost_obj, attribute, 0)
        else:
            setattr(cost_obj, attribute, var)

    return cost_obj


def old_associate_cost(
    objectname: str,
    module: "ModuleType",
) -> gpx.primitives.Cost:
    """associates a full cost object to an object name

    Arguments
    ---------
    objectname : string
        the name of the object to associate costs to

    Keyword Arguments
    -----------------
    module : ModuleType
        the module in which to find the costs

    Raises
    ------

    Returns
    -------
    gpx.primitives.Cost
        the multi-component cost object

    """
    cost_mapping: dict[str, str] = {
        "Non-Recurring Cost": "nonrecurringCost",
        "Recurring Cost": "recurringCost",
        "Variable Cost": "variableCost",
    }

    cost_obj: gpx.primitives.Cost = gpx.primitives.Cost(default_currency=default_currency)

    for key, attribute in cost_mapping.items():
        varname: str = f"{objectname} {key}"

        if varname in module.gpx_variables:
            setattr(cost_obj, attribute, module.gpx_variables[varname])
        else:
            # If there is no variable found, set it to 0
            setattr(cost_obj, attribute, 0)

    return cost_obj


def create_offshifts(
    module: "Manufacturing",
    cellod: dict[str, gpx.manufacturing.QNACell],
    offshiftvars: list[dict[str, str]],
    ton: gpx.Variable,
) -> dict[str, gpx.offshift.OffShiftCell]:
    oscellsdict: dict[str, gpx.offshift.OffShiftCell] = {}
    for v in offshiftvars:
        cellname: str = v["type"]
        if cellname in cellod:
            sourcecell: gpx.manufacturing.QNACell = cellod[cellname]

            # find the exit cell
            cellidx: int = list(cellod.keys()).index(cellname)
            if cellidx + 1 < len(cellod):  # add 1 since 0-indexed
                # this is not the last cell
                exitcell: Optional[gpx.manufacturing.QNACell] = list(cellod.values())[cellidx + 1]
            else:
                exitcell = None

            toffvar: Optional[gpx.Variable] = module.find_gpx_variable(
                filter_category="cells",
                filter_property="Extra Shifts",
                filter_type=v["type"],
                emptyisnone=True,
            )

            if not toffvar:
                raise ValueError(f"Could not find on shift time for {cellname}")

            # create an offshift object
            oscell: gpx.offshift.OffShiftCell = gpx.offshift.OffShiftCell(
                sourcecell,
                ton,
                toffvar,
                exit_cell=exitcell,  # source_cell  # on-shift time  # off-shift time
            )

            # replace the cell in the ordered dict
            cellod[cellname] = oscell

            oscellsdict[cellname] = oscell

    return oscellsdict


def update_offshift_line_constr(
    module: "Manufacturing",
    oslist: list[gpx.offshift.OffShiftCell],
    line: gpx.manufacturing.FabLine,
) -> list[gpkit.ConstraintSet]:
    """create the additional constraints on the line from having offshift cells

    Parameters
    ----------
    module : manufacturing module
        [description]
    oslist : list of gpx.offshift.OffShiftCell
        list of the offshift cells in the line
    line : gpx.manufacturing.FabLine
        the production line to reference

    Returns
    -------
    list of constraints
        the additional constraints to add to the model
    """
    if module.has_offshift and oslist and len(oslist) > 0:
        constr: list[gpkit.ConstraintSet] = []
        if hasattr(line, "L"):
            # use the inventory directly
            constr.append(line.L >= np.sum([osc._get_L_off_basis() for osc in oslist]))
            constr.append(line.L >= np.sum([osc._get_L_on_basis() for osc in oslist]))
        else:
            # use another constraint on ton
            constr.extend([osc.ton >= osc.toff for osc in oslist])

            return constr

    return []


def feeder_qty_by_cell(
    feeder_lines: dict[str, gpx.feeder.FeederLine],
    module_feederprocesses: list[dict[str, str]],
) -> dict[str, gpx.Variable]:
    "gets a dict of cells and the monomial"

    feeder_cell_qty: dict[str, gpx.Variable] = {}

    # flip the cell dict

    # loop over the different feeder lines
    for flname, fl in feeder_lines.items():
        # find the qty multipliers
        mults: list[gpx.Variable] = []

        # find the product of the quantities
        curfl: gpx.feeder.FeederLine = fl
        while True:
            mults.append(curfl.batch_qty)
            if isinstance(curfl.target_line, gpx.manufacturing.FabLine):
                # if this is a main line, break out of the loop
                break
            curfl = curfl.target_line

        # multiply to get the total qty
        qtymon: gpx.Variable = np.prod(mults)

        # find all the cells in the feeder
        cells_in_feeder: set[str] = {fp["cell"] for fp in module_feederprocesses if fp["feederLine"] == flname}

        # add all the cells for the feeder
        feeder_cell_qty.update({c: qtymon for c in cells_in_feeder})

    # return the results
    return feeder_cell_qty
