'translates the "data_types" objects to their corresponding gpx models'

# contains all of the connections to GPX
# loop through the structure of the model and

import logging
from typing import TYPE_CHECKING

from gpkit import Variable

import gpx
import gpx.design
import gpx.manufacturing
import gpx.primitives
import utils.logger as logger

if TYPE_CHECKING:
    from api.module_types.manufacturing import Manufacturing
    from api.module_types.module_type import ModuleType
    from utils.types.data import Parameter, Process

# from constraint_interpreter import build_constraint_set

logging.basicConfig(level=logging.DEBUG)

# def translate_object_to_gpx(object_to_translate):
#     '''Construct
#
#     Returns
#     ------
#     gpx.design.Design
#     '''
#
#     if object_to_translate is type(data_types.Design):
#         #TODO: which constructor to call
#         logger.debug('converting from Design')


def param_to_var(param: "Parameter") -> None:
    """translates a parameter to a gpx.Variable

    Arguments
    ---------
    param : data_types.Parameter
    """
    units = param.unit
    param.gpxObject = Variable(param.name, param.value, units, param.descr)  # TODO: Should this be a read-only var?


def make_gpx_process(process: "Process") -> None:
    """convert a copernicus process to gpx process
    Arguments
    ---------
    process : data_types.Process

    Returns
    -------
    gpx.manufacturing.Process
    """
    pass


def mfg_module_to_gpx_line(module: "Manufacturing") -> None:
    """converts a manufacturing module to gpx objects

    Arguments
    ---------
    module : Manufacturing

    Returns
    -------
    processes, cells, line
    """
    # module_processes = module.

    # generate Processes
    # generate cells


# def module_gen_processes(module):
#     'add gpx processes to the module processes'
#     processes = OD()
#     for p in module.processChain:
#         # put each process into a
#         pass


def module_gen_vars(module: "ModuleType") -> None:
    """generate the gpx variables for a module

    Arguments
    ---------
    module : ModuleType
    """
    for name, param in module.variables.items():
        units = str(param.unit)
        descr = str(param.name)
        module.gpx_variables[name] = Variable(descr, units, descr)


def make_cells(cells: list[gpx.manufacturing.QNACell], processes: list[gpx.primitives.Process]) -> None:
    pass


def make_gpx_processchain(mfgmodule: "Manufacturing") -> None:
    """convert manufacturing module to a processchain
    pass
    return constraints
    """


# def generate_gpx(interaction):
