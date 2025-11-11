'feeder line models'
import typing

import gpkit
from gpkit import Model, Variable, units
from pint import DimensionalityError

from gpx.helpers import optional_variable
from gpx.mixins.feeder_line_mixin import FeederLineMixin
from gpx.primitives import Process


class Batch(Process):
    '''Batching Process

    Variables
    ---------
    b    [count]    The number of items which need to be in the batch

    '''

    def setup(self, qbatch=None, lam=None, ca=None):
        '''model setup
        Arguments
        ---------
        qbatch
            batching quantity
        lam
            production rate
        ca
            Arrival coefficient of variation
        '''
        super().setup()
        self.lam = optional_variable(lam, variablename='\\lambda', units='count/hour', descr='Production Rate')
        self.b = optional_variable(qbatch, variablename='n_{batch}', units='count', descr='Batch Quantity')
        self.ca = optional_variable(ca, variablename='c_a', units='-', descr='Arrival coefficient of variation')
        self.b1 = Variable('(b-1)', lambda c: c[self.b] - 1, 'count')
        self.cs = Variable('c_s', '-', 'Process coefficient of variation')

        return [
            self.t >= 1.0 / (2.0 * self.lam) * self.b1,
            self.cs**2 >= (self.b + (6.0 * self.ca**2 + 1.0) * units('count')) / (3 * self.b1),
        ]


class BatchConvex(Process):
    '''Convex Batching Process

    A convex version of the batching process

    Variables
    ---------
    b      [count]       The number of items which need to be in the batch
    lamIn  [count/hr]    Input production rate
    lamOut [count/hr]    Output production rate
    ca     [-]           Arrival process coefficient of variation
    cs     [-]           Process coefficient of variation

    '''

    def setup(self, **kwargs):
        super().setup()

        # Batch size
        self.b = optional_variable(
            kwargs.get("b"),
            variablename="b",
            units="count",
            descr="The number of items which need to be in the batch",
        )

        # Input production rate
        self.lamIn = optional_variable(
            kwargs.get("lamIn"),
            variablename="\\lambda_{input}",
            units="count/hr",
            descr="Input production rate",
        )

        # Output production rate
        self.lamOut = optional_variable(
            kwargs.get("lamOut"),
            variablename="\\lambda_{output}",
            units="count/hr",
            descr="Output production rate",
        )

        # Arrival coefficient of variation
        self.ca = optional_variable(
            kwargs.get("ca"),
            variablename="cv_{arrival}",
            units="-",
            descr="Arrival coefficient of variation",
        )

        # Process coefficient of variation
        self.cs = optional_variable(
            kwargs.get("cs"),
            variablename="cv_{process}",
            units="-",
            descr="Process coefficient of variation",
        )

        return [
            self.lamOut == self.lamIn / self.b,  # the output rate is divided by the batch size
            self.t >= 1.0 / (2.0 * self.lamIn) * self.b,
            self.cs**2 >= (self.b + (6.0 * self.ca**2 + 1.0) * units('count')) / (3.0 * self.b),
        ]


class FeederLine(Model, FeederLineMixin):
    '''A Short Feeder Line

    The short feeder line comprises a CONWIP line which terminates in a batching process.
    Feeder lines help accomplish rate matches between different lines

    Usually there is very weak cost on inventory in feeder lines (unless there is tooling)
    May need to manually add an upper limit
    '''

    def setup(
        self,
        cells,
        target_cell,
        target_line=None,
        return_cells=False,
        batch_qty=1,
        makeqtyvar=False,
        dispname='',
        **kwargs
    ):
        '''
        Arguments
        ---------
        cells : list of gpx.manufacturing.QNACell
            the cells comprising the feeder line, in order
        target_cell : gpx.manufacturing.QNACell
            the primary cell where the feeder line joins the main production
        target_line : gpx.manufacturing.FabLine
            the primary line which the feeder line joins
        return_cells : boolean
            returns the cell constraints as well
        batch_qty : int, gpkit.Variable, gpkit.Monomial (Default=1)
            the quantity which has to be batched
        makeqtyvar : boolean (Default=False)
            create a GPX variable for the feederline quantity
        dispname : string
            optional display name

        Keyword Arguments
        -----------------
        is_batched : boolean
            does the feeder line need a batching process at the end

        '''
        constraints = []

        # set attributes
        self.cells = cells
        self.target_line = target_line

        if makeqtyvar:
            name = (dispname + ' Feeder Qty').strip()  # remove extra leading spaces if dispname is empty
            self.batch_qty = 1 if batch_qty == 1 else Variable(name, batch_qty, 'count', name)
            if batch_qty != 1:
                if isinstance(batch_qty, (gpkit.Monomial, gpkit.Variable)):
                    self.batch_qty = batch_qty
                else:
                    inferred_units = getattr(batch_qty, "units", None)
                    self.batch_qty = Variable(name, batch_qty, inferred_units, name)
        else:
            self.batch_qty = batch_qty

        if hasattr(self.batch_qty, "units") and self.batch_qty.units is not None:
            lam_units = target_cell.lam.units * self.batch_qty.units
        else:
            lam_units = target_cell.lam.units

        lam = self.lam = Variable(
            '\\lambda_{feeder}', units=str(lam_units), label='Feeder production rate (after any batching)'
        )

        if hasattr(self.batch_qty, "units"):
            try:
                units.of_division(self.batch_qty, units("kg"))  # succeeds only for mass
                batch_is_mass = True
            except DimensionalityError:
                batch_is_mass = False
        else:
            batch_is_mass = False

        if batch_is_mass:
            constraints.extend([
                lam >= target_cell.lam * self.batch_qty,
            ])
        else:
            constraints.extend([
                # target_line.lam*self.batch_qty <= lam,    # match line rate
                target_cell.lam * self.batch_qty <= lam,  # match the rate of the target cell
                cells[-1].c2d <= target_cell.c2a,  # influence variability of join cell
            ])

        # connect variation of all cells
        constraints.extend([cells[i + 1].c2a >= cells[i].c2d for i, _ in enumerate(cells[:-1])])

        # loop the variation around back to the beginning (kanban)
        constraints.append(cells[0].c2a >= cells[-1].c2d)

        # set the production rate for the cells to be that of the main line
        try:
            constraints.extend([lam <= cell.lam for cell in cells])
        except DimensionalityError as e:
            print(cell for cell in cells)
            unit_str = str(getattr(self.batch_qty, "units", "unknown")).strip()
            raise ValueError(
                f"Quantity for Feeder Line ID '{dispname}' is in `{unit_str}`, but one or more cell rates use incompatible units. "
                f"Set the batch size for the associated cells to compatible units (e.g. `{unit_str}`). "
            ) from e

        # Track Cumulative batch quantity
        self.cuml_batch_qty: typing.Union[int, Variable, gpkit.Monomial] = (
            # attribute is the math, not a full constraint
            self.batch_qty * getattr(target_line, "cuml_batch_qty", 1)
        )

        if return_cells:
            # return the constraints of the cells if requested
            constraints.extend(cells)

        return constraints


class PartsOnlyFeederLine(Model, FeederLineMixin):
    """
    A specialized feeder line that carries only parts (no internal processes).
    This can handle cost, batch quantities, and basic flow constraints, but does not
    require the definition of feeder processes/cells.
    """

    def setup(
        self,
        default_currency: str,
        line_id="PartsOnlyFeeder",
        target_line: FeederLine = None,
        batch_qty=1,
        variable_cost=None,
        dispname="Parts-Only Feeder Line",
        **kwargs
    ):
        """
        Arguments
        ---------
        line_id : str
            The unique identifier for this parts-only feeder line.
        target_line : gpx.feeder.FeederLine or None
            The line (or cell) this feeder line eventually feeds into.
        batch_qty : int or gpkit.Variable
            The number of parts batched at once.
        variable_cost : float or gpkit.Variable
            Cost per unit or batch to run parts through this line (if any).
        dispname : str
            Display name for logging/debugging.

        Returns
        -------
        constraints : list
            The list of GPkit constraints that define the model behavior.
        """
        constraints = []
        self.target_line: FeederLine = target_line

        if isinstance(batch_qty, (int, float)):
            self.batch_qty = 1 if batch_qty == 1 else Variable(f"{line_id}_batch_qty", batch_qty, "count", dispname)
        else:
            self.batch_qty = batch_qty

        lam = Variable(f"lam_{line_id}", "count/hr", f"Production rate for {dispname}")

        if variable_cost is not None:
            if isinstance(variable_cost, (int, float)):
                self.variable_cost = Variable(
                    f"{line_id}_var_cost", variable_cost, f"{default_currency}/count", f"Variable cost for {dispname}"
                )
            else:
                self.variable_cost = variable_cost
        else:
            self.variable_cost = None

        # Track Cumulative batch quantity
        self._calculate_batch_accumulation()

        return constraints
