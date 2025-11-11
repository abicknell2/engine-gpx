'components for parametric input equations'

# from gpx import ureg
from collections import namedtuple
from dataclasses import dataclass
from itertools import groupby, product
import numbers

from adce import adnumber
from gpkit import SolutionArray, VarKey, keydict, ureg
import networkx as nx
import numpy as np
import pint

from gpx.dag.paramkey import ParamKey

TOKEN_PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

LogGrad = namedtuple('LogGrad', ['c_var', 'v_var', 'dlogv_dlogc', 'c_ss'])


class ParametricVariable:

    def __init__(
        self,
        name: str,
        varkey: VarKey,
        source_var=None,
        qty: pint.Quantity = None,
        magnitude: float = None,
        is_input: bool = False,
        is_leaf: bool = False,
        unit: str = '',
    ):
        'defines the nodes'
        self.name = name  # name of the variable
        self.varkey = varkey  # the gp varkey
        self.qty = qty

        self.is_input = is_input  # whether the variable is an input variable
        self.is_leaf = is_leaf  # is the variable a leaf

        self.magnitude = magnitude

        self.source_var = source_var

        # check for unitless
        unit = unit.replace('-', '')
        self.unit = unit

        self.base_unit_val: float  # the value in base units
        self.adnum: adnumber  # the differentiable number

        self.defining_constraint = None  # if the variable is not an input, this should be defined

    def update_value(self, quantity=None, get_units=True):
        'update the value properties'
        if quantity:
            self.qty = quantity
            self.magnitude = getattr(quantity, 'magnitude', quantity)
            if get_units:
                self.unit = getattr(quantity, 'Units', '')
        else:
            try:
                self.qty = self.magnitude * ureg(self.unit)
            except Exception as e:
                print(e)

        self.base_unit_val = self.qty.to_base_units().magnitude
        self.adnum = adnumber(self.base_unit_val)

    def update_magnitude_only(self, new_magnitude):
        'update only the magnitude of a variable'
        self.magnitude = new_magnitude
        self.qty = self.magnitude * ureg(self.unit)
        self.base_unit_val = self.qty.to_base_units().magnitude
        self.adnum = adnumber(self.base_unit_val)

    def update_adnumber(self):
        if not self.base_unit_val:
            self.update_value()
        self.adnum = adnumber(self.base_unit_val)


class ParametricConstraint:
    '''a parametric constraint

    - this class also defines the edge of the graph
    '''

    def __init__(self, constraint_as_list: list, inputvars: dict[VarKey, ParametricVariable]) -> None:
        self.inputvars = inputvars  # input variables
        self.constraint: list = constraint_as_list  # the constraint represented as a list of
        self.outputvar: ParametricVariable  # the output variable

        # the function representation of the constraint
        self.func = None
        # run the factory
        self.func_factory()

        # self.sym_convert()
        # self.func_convert()

    def func_factory(self, constraint_as_list: list = None):
        'function factory to create the evaluate function for the class'

        precedence = TOKEN_PRECEDENCE
        expression = constraint_as_list if constraint_as_list else self.constraint

        #TODO:  change the power operator

        #TODO:  is this operating on varkeys or strings?

        def apply_operation(operators, operands):
            operator = operators.pop()
            operand2 = operands.pop()
            operand1 = operands.pop()
            if operator == '+':
                operands.append(operand1 + operand2)
            elif operator == '-':
                operands.append(operand1 - operand2)
            elif operator == '*':
                operands.append(operand1 * operand2)
            elif operator == '/':
                operands.append(operand1 / operand2)
            elif operator == '^':
                operands.append(operand1**operand2)

        def math_function(**variables):
            operators = []
            operands = []

            for token in expression:
                if token in precedence:
                    while (operators and operators[-1] in precedence
                           and precedence[operators[-1]] >= precedence[token]):
                        apply_operation(operators, operands)
                    operators.append(token)
                # elif token is number:
                elif isinstance(token, numbers.Number):
                    operands.append(token)
                else:
                    operands.append(variables[token])

            while operators:
                apply_operation(operators, operands)

            return operands[0]

        # Set the math function
        self.func = math_function

    def get_subgraph(self) -> nx.DiGraph:
        'get the directed subgraph for the constraint'
        subgraph = nx.DiGraph()
        # add the edges from all the inputs to the output var and color by the constraint
        subgraph.add_edges_from([(iv, self.outputvar) for iv in self.inputvars.values()], constraint=self)

        return subgraph

    def evaluate(self, inputs: dict[VarKey, ParametricVariable] = None) -> dict[VarKey, ParametricVariable]:
        'evaluate the constraint for the inputs (quantites and adnumbers) and save to the output variable'
        inputs = inputs if inputs else self.inputvars

        # evaluate the function
        qty_inputs = {str(k): v.qty for k, v in inputs.items()}
        adnum_inputs = {str(k): v.adnum for k, v in inputs.items()}

        # get the units of the target variable
        target_unit = self.outputvar.unit

        # evaluate with pint quantites
        constraint_name = self.outputvar.name
        try:
            qty_result = self.func(**qty_inputs)
        except Exception as exc:
            raise ValueError(f"{exc} in constraint {constraint_name}") from exc

        if hasattr(qty_result, "to"):
            try:
                if target_unit:
                    self.outputvar.qty = qty_result.to(target_unit)
                else:
                    self.outputvar.qty = qty_result
            except pint.DimensionalityError as exc:
                raise ValueError(f"{exc} in constraint {constraint_name}") from exc
        else:
            error = self._unitless_summary(self.constraint, qty_inputs, target_unit)
            typename = type(qty_result).__name__
            raise ValueError(
                f"Constraint {constraint_name} evaluated to a plain {typename}. {error}"
            )

        # evaluate with adnumbers
        self.outputvar.adnum = self.func(**adnum_inputs)

    def adiff(self) -> keydict:
        'perform the automatic differentiation aand return keydict of gradients'
        pass

    def update_output_var(self, outputvar: ParametricVariable):
        'update the outputvar variable'
        self.outputvar = outputvar
        outputvar.defining_constraint = self

    def _unitless_summary(self, expr: list, known_vars: dict[str, object], target_unit) -> str:
        """
        Build a short hint when the RHS ended up dimension-less.

        • lists any numeric literals that have no units
        • lists any tokens that look like numbers or unknown variables
        """
        naked_literals = [str(t) for t in expr if isinstance(t, numbers.Number)]
        unknown_tokens = [
            str(t) for t in expr
            if isinstance(t, str)
            and t not in known_vars           # not a recognised variable
            and t not in TOKEN_PRECEDENCE     # not an operator
        ]

        bits = []
        if naked_literals:
            bits.append("Literals without units: " + ", ".join(naked_literals))
        if unknown_tokens:
            bits.append("Unrecognised tokens: " + ", ".join(unknown_tokens))

        expected_unit = target_unit or "dimensionless"
        detail = "; ".join(bits) or "Result has no units"
        return (
            f"{detail} (expected '{expected_unit}'). "
            "Add explicit units to numeric literals or refer to other variables with units."
        )


class ParametricInputs:
    'holds all the parametric inputs and the graph'

    def __init__(self, constraints: list[ParametricConstraint]):
        self.constraints = constraints  # list of the constraints
        self.inputs = []  # parametric inputs
        self.terminal_nodes = []  # terminal (leaf) nodes. this will be substituted into the model
        self.intermed_nodes = []  # variables that are neither inputs nor outputs
        self.graph = nx.DiGraph()  # the graph of the parametic models

        self.nodes: list[ParametricVariable]

        self.create_graph()

    def add_constraint(self, constraint: ParametricConstraint):
        # update the list of inputs
        pass

    def create_graph(self):
        'update the graph from the constraints'
        # self.graph = nx.compose()
        self.graph = nx.compose_all([pc.get_subgraph() for pc in self.constraints])
        return self.graph

    def add_terminal_node(self, node):
        pass

    def update_inputs_from_subs(self, new_subs: dict[VarKey, pint.Quantity]):
        'update the acyclic inputs from a substitution dict'
        # loop over the nodes and look for subs
        for n in self.graph.nodes:
            if n.is_input:
                newval = new_subs.get(n.varkey, None)
                if newval:
                    #FUTURE:    use the walrus operator
                    n.update_magnitude_only(newval)

    def get_substitutions(
        self,
        include_intermed_subs: bool = True,
    ) -> dict[VarKey, pint.Quantity]:
        'get the substitution dict to send to the gpkit model'
        # get the graph in a topological sort order
        n: ParametricVariable
        try:
            ordered_nodes = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible as e:
            cycle = nx.find_cycle(self.graph, orientation='original')
            named_cycle = [
                (getattr(a, "name", str(a)), getattr(b, "name", str(b)), direction)
                for a, b, direction in cycle
            ]
            raise ValueError(f"Parametric cycle detected: {named_cycle}") from e

        for n in ordered_nodes:
            if n.is_input:
                # this does not need to be calculated
                # add tö inputs
                self.inputs.append(n)
                continue
            # otherwise evaluate the constraint that defines the variable
            n.defining_constraint.evaluate()
            # n.defining_constraint.evaluate(inputs=self.nodes)

            # if the node has 0 connectivity, color it a terminal leaf node (or 0 decendants)
            if not nx.descendants(self.graph, n):
                n.is_terminal = True
                n.is_leaf = True

                self.terminal_nodes.append(n)
            else:
                # add to intermedate nodes
                self.intermed_nodes.append(n)

        # generate a substitution dict from the nodes (which should all be calculated now)
        input_subs = {v.varkey: v.qty for v in self.inputs}
        outputs_subs = {v.varkey: v.qty for v in self.terminal_nodes}

        allsubs = {**input_subs, **outputs_subs}
        if include_intermed_subs:
            intermed_subs = {v.varkey: v.qty for v in self.intermed_nodes}
            allsubs.update(intermed_subs)
        return allsubs

    def _get_sorted_graph(self):
        'get the graph in a topological sort order'
        # the networkx gives the sorted list of nodes
        # if the node is an input, it can be skipped
        # if the node is not an input, need to evaluate the constraint
        pass

    def _get_gradients(self, sol):
        'gets all of the gradients of the terminal nodes to all of the input nodes'
        grads = [
            self._get_output_log_gradient(c_var=c, v_var=v, sol=sol, return_obj=True)
            for c, v in product(self.inputs, [*self.terminal_nodes, *self.intermed_nodes])
            # for c, v in product(self.inputs, self.terminal_nodes)
        ]
        return grads

    def _get_output_log_gradient(
        self,
        c_var: ParametricVariable,  # the input constant
        v_var: ParametricVariable,
        sol: SolutionArray = None,
        return_obj: bool = False,
        filter_intermediates: bool = True,
        **kwargs,
    ):
        'find the total log gradient'
        # calullate the gradient of the output variable v to input constant c
        dv_dc = v_var.adnum.gradient(c_var.adnum)[0]
        dlogv_dlogc = dv_dc * (c_var.adnum / v_var.adnum).real

        if return_obj:
            # calulate
            if not sol:
                # has to have a result object
                raise ValueError('log gradient calulation requires a solution input to return an object')

            if filter_intermediates:
                dlogcost_dlogv = sol['sensitivities']['constants'].get(v_var.varkey, 0.0)
            else:
                dlogcost_dlogv = sol['sensitivities']['constants'][v_var.varkey]
            c_ss = dlogcost_dlogv * dlogv_dlogc

            return LogGrad(
                c_var=c_var,
                v_var=v_var,
                dlogv_dlogc=dlogv_dlogc,
                c_ss=c_ss,
            )

        return dlogv_dlogc

    def update_results(self, sol: SolutionArray, inplace=False) -> SolutionArray:
        'update the results from the gpx solve'
        # add the values to the solutions
        # get all the gradients
        grads = self._get_gradients(sol)
        # sort and group by input variable
        c_grads = {
            c: np.sum([gg.c_ss
                       for gg in g])
            for c, g in groupby(sorted(grads, key=lambda x: str(x.c_var.varkey)), key=lambda x: x.c_var.varkey)
        }

        # potential sources of sensitivites for intermediate variables
        pot_sources = [
            sol['sensitivities']['constants'],
            sol['sensitivities']['variables'],
            sol['constants'],
        ]

        for node in self.graph.nodes:
            if node.is_input:
                # add to the variables and the constants
                sol['constants'][node.varkey] = sol['variables'][node.varkey] = node.qty.magnitude
                # update the sensitivity
                if node.varkey in sol['sensitivities']['constants']:
                    sol['sensitivities']['constants'][node.varkey] += c_grads[node.varkey]
                    sol['sensitivities']['variables'][node.varkey] += c_grads[node.varkey]
                else:
                    sol['sensitivities']['constants'][node.varkey] = c_grads[node.varkey]
                    sol['sensitivities']['variables'][node.varkey] = c_grads[node.varkey]

            elif node.is_leaf:
                # delete results that make it look like an input to the model
                for ps in pot_sources:
                    if node.varkey in ps:
                        del ps[node.varkey]

            else:
                # this is just an intermediate variable
                try:
                    if node.source_var:
                        # try and add using the full variable as the key
                        sol['variables'][node.source_var] = node.qty.magnitude
                    else:
                        sol['variables'][ParamKey(name=node.varkey)] = node.qty.magnitude
                except Exception as e:
                    # skip adding the variable
                    print(e)
                # delete any other sources for sensitivities
                for ps in pot_sources:
                    if node.varkey in ps:
                        del ps[node.varkey]

        return sol

    def check_graph(self):
        'checks to make sure the graph is indeed acyclic'
        # all leaf nodes göback to nodes that are in the list of input nodes


class XFromSplits(ParametricConstraint):
    """
    Computes arrival‑mix parameters for a shared resource from the
    (possibly user‑supplied) class‑split variables.

    split_var: ParametricVariable for the target products Xᵢ
    all_split_vars: iterable[ParametricVariable] for every class Xⱼ
    output_pv: ParametricVariable whose varkey is the QNACell.xᵢ
    """

    def __init__(
        self, split_var: ParametricVariable, all_split_vars: list[ParametricVariable], output_pv: ParametricVariable
    ):
        self.split_var = split_var
        self.all_split_vars = all_split_vars
        # map every input varkey to ParametricVariable
        invars = {pv.varkey.key: pv for pv in [split_var, *all_split_vars]}

        super().__init__(constraint_as_list=[], inputvars=invars)
        self.update_output_var(output_pv)  # sets self.outputvar
        self.func_factory()  # build callable

    def func_factory(self):
        num_key = str(self.split_var.varkey)
        den_keys = [str(pv.varkey) for pv in self.all_split_vars]

        def _ratio(**vals):
            num = vals[num_key]
            den = sum(vals[k] for k in den_keys)
            return 0 if den == 0 else num / den

        self.func = _ratio


class XSplitBySingleCell(ParametricConstraint):

    def __init__(self, output_pv, value: float = 1.0):
        super().__init__(constraint_as_list=[], inputvars={})
        self.update_output_var(output_pv)
        self._value = value
        self.func = self._constant_value

    def _constant_value(self):
        return self._value

    def get_subgraph(self):
        g = nx.DiGraph()
        g.add_node(self.outputvar)
        return g