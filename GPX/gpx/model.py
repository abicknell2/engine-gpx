"update a Model object for GPX"

from __future__ import annotations

import importlib.util
import logging
import math
from operator import itemgetter
from typing import TYPE_CHECKING, Optional, Union

import gpkit
from gpkit.exceptions import UnknownInfeasible
from gpkit.keydict import KeyDict
import numpy as np

import gpx.dag.parametric

# typing-only imports
if TYPE_CHECKING:
    import gpx.manufacturing as mn

    # define a type that is for production resources
    ProductionResource = Union[mn.QNACell, mn.ConwipTooling]


class Model(gpkit.Model):
    'create a model object'

    def __init__(
        self,
        cost=None,
        constraints=None,
        acyclic_constraints: list[gpx.dag.parametric.ParametricConstraint] = [],
        *args,
        **kwargs
    ):
        self.acyclic_constraints = acyclic_constraints
        super().__init__(cost, constraints, *args, **kwargs)

    def solve(self, **kwargs):
        if self.acyclic_constraints:
            # create the inputs
            self.acyclic_inputs = gpx.dag.parametric.ParametricInputs(constraints=self.acyclic_constraints)

            self.substitutions.update(self.acyclic_inputs.get_substitutions())

        # kwargs["solver"] = 'mosek_conif'
        solve_kwargs = dict(kwargs)
        solve_kwargs.setdefault("verbosity", 0)
        super().solve(**solve_kwargs)
        
        if self.acyclic_constraints:
            self.acyclic_inputs.update_results(self.solution)

        return self.solution

    def update_substitutions(self, new_subs=None):
        'updates the substituions but can also catch the changes to the acyclic inputs'
        if new_subs:
            self.substitutions.update(new_subs)

        # update the acyclic substitutions
        if self.acyclic_constraints:
            # update acyclic inputs from new substitutions
            self.acyclic_inputs.update_inputs_from_subs(self.substitutions)
            # self.acyclic_inputs.update_inputs_from_subs(new_subs)

            # get the substitutions back from the acyclic inputs
            self.substitutions.update(self.acyclic_inputs.get_substitutions())

    def discretesolve(
        self,
        discrete_resources: list[gpkit.Variable],
        target_variable: Optional[gpkit.Variable],
        target_value: Union[int, float, None],
        new_cost: gpkit.Posynomial = None,
        target_strategy: str = 'above',  # 'above' | 'below'
        rounding_strategy: str = 'round',  # 'floor' | 'round'
        sensitivity_strategy: str = 'min',  # 'min' | 'max'
        solve_orig=True,  # only matters when new cost is defined
        preserve_input_qty=True,  # if there is input on the quantity, respect
        **kwargs,
    ) -> object:
        '''Solve a discrete version of the model for production resources.'''
        if target_variable is None:
            raise ValueError('target_variable must be provided for discretesolve')

        target_map: dict[gpkit.Variable, object] = {target_variable: target_value}

        return self._discrete_solve_core(
            discrete_resources=discrete_resources,
            target_map=target_map,
            new_cost=new_cost,
            target_strategy=target_strategy,
            rounding_strategy=rounding_strategy,
            sensitivity_strategy=sensitivity_strategy,
            solve_orig=solve_orig,
            preserve_input_qty=preserve_input_qty,
            **kwargs,
        )

    def by_rate_discretesolve(
        self,
        discrete_resources: list[gpkit.Variable],
        target_dict: dict[gpkit.Variable, object],
        new_cost: gpkit.Posynomial = None,
        target_strategy: str = 'above',  # 'above' | 'below'
        rounding_strategy: str = 'round',  # 'floor' | 'round'
        sensitivity_strategy: str = 'min',  # 'min' | 'max'
        solve_orig=True,  # only matters when new cost is defined
        preserve_input_qty=True,  # if there is input on the quantity, respect
        relax_targets_on_infeasible: bool = False,
        **kwargs,
    ) -> object:
        '''Solve the discrete model when individual product rates are targets.'''
        if not target_dict:
            raise ValueError('target_dict must contain at least one target variable')

        return self._discrete_solve_core(
            discrete_resources=discrete_resources,
            target_map=target_dict,
            new_cost=new_cost,
            target_strategy=target_strategy,
            rounding_strategy=rounding_strategy,
            sensitivity_strategy=sensitivity_strategy,
            solve_orig=solve_orig,
            preserve_input_qty=preserve_input_qty,
            relax_targets_on_infeasible=relax_targets_on_infeasible,
            **kwargs,
        )

    def _discrete_solve_core(
        self,
        discrete_resources: list[gpkit.Variable],
        target_map: dict[gpkit.Variable, object],
        new_cost: gpkit.Posynomial = None,
        target_strategy: str = 'above',
        rounding_strategy: str = 'round',
        sensitivity_strategy: str = 'min',
        solve_orig=True,
        preserve_input_qty=True,
        relax_targets_on_infeasible: bool = False,
        **kwargs,
    ) -> object:
        '''Common implementation for discrete solves.'''
        if not target_map:
            raise ValueError('target_map must contain at least one target variable')

        def _as_variable(entry: gpkit.Variable | tuple[gpkit.Variable, object]) -> gpkit.Variable:
            return entry[0] if isinstance(entry, tuple) else entry

        # copy the resources so we can adjust the list without side effects
        original_discrete_resources = list(discrete_resources)
        resource_lookup: dict[gpkit.Variable, ProductionResource | object | None] = {}
        # store what changed during the run for debugging later on
        self.last_discrete_adjustments: list[dict[str, object]] = []
        self.last_discrete_feasibility_bump: dict[str, object] | None = None

        def _resource_label(resource: object | None) -> str | None:
            if not resource:
                return None
            for attr in ("display_name", "name", "key"):
                value = getattr(resource, attr, None)
                if value:
                    return str(value)
            return None

        if solve_orig:
            # run the smooth model first so we start from a valid plan
            _ = self.solve(**kwargs)

        # check to make sure there is already a solution
        if not self.solution:
            raise ValueError('the model needs to have been solved as continuous')

        # save the solution that was from the full continuous
        # remember the smooth solution for reporting
        continuous_solution = self.solution

        filtered_resources: list[gpkit.Variable] = []
        for entry in original_discrete_resources:
            var = _as_variable(entry)
            resource = entry[1] if isinstance(entry, tuple) and len(entry) > 1 else None

            # if inputs are to be repsected, find varbales that do not have substitutions
            if preserve_input_qty and var in continuous_solution['constants']:
                # remove resource variables that have inputs
                continue

            filtered_resources.append(var)
            resource_lookup[var] = resource

        if resource_lookup:
            logging.debug(
                "Discrete resource mapping: %s",
                {
                    str(var.key): (_resource_label(res) or "")
                    for var, res in resource_lookup.items()
                },
            )

        self.discrete_resources = filtered_resources

        # set up the proper rounder
        # find the numpy helper that matches the rounding plan
        rounder = getattr(np, rounding_strategy, np.round)
        
        # get the sensitivity strategy
        # pick the sensitivity combiner that the caller asked for
        sensFinder = max if sensitivity_strategy == 'max' else min

        # convert to the new cost
        if new_cost:
            # temporarily swap in the discrete cost shape
            orig_cost = self.cost
            self.cost = new_cost
        else:
            orig_cost = None

        missing = object()
        targets_relaxed = False

        def _pop_target_sub(var: gpkit.Variable) -> tuple[object | None, object]:
            if var in self.substitutions:
                value = self.substitutions[var]
                del self.substitutions[var]
                return var, value
            vkey = getattr(var, 'key', None)
            if vkey is not None and vkey in self.substitutions:
                value = self.substitutions[vkey]
                del self.substitutions[vkey]
                return vkey, value
            return None, missing

        target_substitutions: dict[gpkit.Variable, tuple[object | None, object]] = {}
        for target_var in target_map:
            target_substitutions[target_var] = _pop_target_sub(target_var)

        # allow callers to loosen rounding if needed
        tolerance = kwargs.pop('rounding_tolerance', 1e-6)
        adjustable_resources: list[gpkit.Variable] = []
        start_dict: dict[gpkit.Variable, float] = {}
        original_discrete_subs: dict[gpkit.Variable, object] = {}

        def _rounded_start(value: float) -> float:
            if rounding_strategy == 'floor':
                rounded = float(np.floor(value))
            elif rounding_strategy == 'ceil':
                rounded = float(np.ceil(value))
            elif rounding_strategy == 'round':
                rounded = math.floor(value + 0.5)
            else:
                rounded = float(rounder(value))
            # keep at least one workstation so the cell stays available
            return max(float(rounded), 1.0)

        baseline_values: dict[gpkit.Variable, float] = {}
        for dr in self.discrete_resources:
            original_discrete_subs[dr] = self.substitutions.get(dr, missing)
            baseline_val = float(self.solution(dr))
            baseline_values[dr] = baseline_val
            rounded_val = _rounded_start(baseline_val)

            adjustable_resources.append(dr)
            start_dict[dr] = rounded_val

        disabled_rate_controllers: list[ProductionResource] = []

        try:
            for dr in self.discrete_resources:
                resource = resource_lookup.get(dr)
                disable = getattr(resource, "disable_rate_constraint", None)
                if callable(disable) and disable():
                    # record which cells have their caps lifted
                    disabled_rate_controllers.append(resource)

            self.substitutions.update(start_dict)
            initial_start_dict = dict(start_dict)

            try:
                # try the discrete solve with just the rounded starts
                cur_solution = self.solve(**kwargs)
            except UnknownInfeasible:
                logging.debug(
                    "Discrete solve infeasible with initial rounded counts: %s",
                    {str(dr.key): start_dict[dr] for dr in adjustable_resources},
                )
                # Nearest-integer rounding may cut capacity; try ceilling the baseline counts.
                for dr in adjustable_resources:
                    ceil_val = max(math.ceil(baseline_values.get(dr, 0.0)), 1)
                    if ceil_val > start_dict.get(dr, 0):
                        start_dict[dr] = ceil_val
                        self.substitutions[dr] = ceil_val

                initial_start_dict = dict(start_dict)

                try:
                    # If the ceilled counts are sufficient we can stop here.
                    cur_solution = self.solve(**kwargs)
                except UnknownInfeasible:
                    logging.debug(
                        "Discrete solve still infeasible after ceiling counts: %s",
                        {str(dr.key): start_dict[dr] for dr in adjustable_resources},
                    )

                    if relax_targets_on_infeasible and not targets_relaxed and target_map:
                        logging.debug(
                            "Relaxing discrete rate targets to allow throughput to float with fixed counts",
                        )
                        for var in list(target_map.keys()):
                            target_map[var] = None
                            sub_key, _ = target_substitutions.get(var, (None, missing))
                            target_substitutions[var] = (sub_key, missing)
                        targets_relaxed = True

                        try:
                            cur_solution = self.solve(**kwargs)
                        except UnknownInfeasible:
                            logging.debug(
                                "Relaxed rate targets remained infeasible; falling back to safeguarded bumps",
                            )
                            targets_relaxed = False

                    if not targets_relaxed:

                        # only reach this path when other constraints still bite after lifting rates
                        # limit the number of safeguarded bumps so we stop after a fair sweep
                        max_adjustments = max(len(adjustable_resources), 1) * 10
                        # track how many bumps have been attempted
                        adjustments = 0
                        last_feasibility_bump: dict[str, object] | None = None
                        # run a guarded round robin only when genuine infeasibility remains
                        while True:
                            try:
                                # re-test after any safeguarded bump
                                cur_solution = self.solve(**kwargs)
                                if last_feasibility_bump:
                                    self.last_discrete_feasibility_bump = last_feasibility_bump
                                    resource_label = last_feasibility_bump.get("resource")
                                    log_suffix = (
                                        f" (resource {resource_label})" if resource_label else ""
                                    )
                                    logging.debug(
                                        "Discrete feasibility restored after bumping %s to %s%s",
                                        last_feasibility_bump["variable"],
                                        last_feasibility_bump["final"],
                                        log_suffix,
                                    )
                                break
                            except UnknownInfeasible:
                                logging.debug(
                                    "Discrete solve still infeasible after safeguarded bump %s: %s",
                                    adjustments + 1,
                                    {
                                        str(dr.key): start_dict[dr]
                                        for dr in adjustable_resources
                                    },
                                )
                                if adjustments >= max_adjustments:
                                    self.last_discrete_feasibility_bump = None
                                    raise
                                dr = adjustable_resources[
                                    adjustments % len(adjustable_resources)
                                ]
                                # increase the current cell in turn as a last resort
                                start_dict[dr] = start_dict.get(dr, 0) + 1
                                self.substitutions[dr] = start_dict[dr]
                                resource = resource_lookup.get(dr)
                                resource_label = _resource_label(resource)
                                last_feasibility_bump = {
                                    "variable": str(dr.key),
                                    "resource": resource_label,
                                    "initial": start_dict[dr] - 1,
                                    "final": start_dict[dr],
                                }
                                adjustments += 1

            target_strategy_flag = 1 if target_strategy == 'above' else -1

            def _to_float(value: object) -> float | None:
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    magnitude = getattr(value, 'magnitude', None)
                    if magnitude is not None:
                        try:
                            return float(magnitude)
                        except (TypeError, ValueError):
                            return None
                return None

            def _solution_value(solution: object, var: gpkit.Variable) -> object:
                try:
                    variables = solution['variables']
                except Exception:
                    variables = {}
                if var in variables:
                    return variables[var]
                vkey = getattr(var, 'key', None)
                if vkey is not None and vkey in variables:
                    return variables[vkey]
                try:
                    return solution(var)  # type: ignore[operator]
                except Exception:
                    return None

            def _targets_met(solution: object) -> bool:
                comparisons: list[bool] = []
                for var, desired in target_map.items():
                    desired_float = _to_float(desired)
                    if desired_float is None:
                        continue
                    current = _solution_value(solution, var)
                    current_float = _to_float(current)
                    if current_float is None:
                        continue
                    comparisons.append(
                        current_float * target_strategy_flag
                        >= desired_float * target_strategy_flag - tolerance
                    )
                return all(comparisons) if comparisons else True

            cur_solution_targets_met = _targets_met(cur_solution)

            # loop to find the end point
            while not cur_solution_targets_met:
                # gather sensitivities to choose the best knob to turn
                candidates = [
                    (self.solution['sensitivities']['variables'][dr], dr)
                    for dr in adjustable_resources
                    if dr in self.solution['sensitivities']['variables']
                ]
                if not candidates:
                    break

                # find the most sensitive variable
                most_sens = sensFinder(candidates, key=itemgetter(0))[1]
                self.substitutions[most_sens] += 1

                # re-solve
                cur_solution = self.solve(**kwargs)
                cur_solution_targets_met = _targets_met(cur_solution)

            # save the solution for the peak discrete (optimized target_variable)
            optimized_discrete_solution = self.solution

            adjustments_summary: list[dict[str, object]] = []
            for dr in adjustable_resources:
                initial_val = initial_start_dict.get(dr)
                final_val = start_dict.get(dr)
                if initial_val is None or final_val is None:
                    continue
                if math.isclose(initial_val, final_val):
                    continue
                resource = resource_lookup.get(dr)
                adjustments_summary.append(
                    {
                        "variable": str(dr.key),
                        "resource": _resource_label(resource),
                        "initial": initial_val,
                        "final": final_val,
                    }
                )

            self.last_discrete_adjustments = adjustments_summary
            if adjustments_summary:
                logging.debug("Discrete feasibility adjustments applied: %s", adjustments_summary)

        finally:
            for resource in reversed(disabled_rate_controllers):
                restore = getattr(resource, "restore_rate_constraint", None)
                if callable(restore):
                    # put the rate caps back before returning
                    restore()

        for dr, prior in original_discrete_subs.items():
            if prior is missing:
                try:
                    del self.substitutions[dr]
                except KeyError:
                    pass
            else:
                self.substitutions[dr] = prior

        if new_cost and orig_cost is not None:
            self.cost = orig_cost

        # substitute the target(s)
        for var, (sub_key, prior) in target_substitutions.items():
            key = sub_key or var
            if prior is missing:
                fallback = target_map.get(var)
                if fallback is None:
                    try:
                        del self.substitutions[key]
                    except KeyError:
                        pass
                else:
                    self.substitutions[key] = fallback
            else:
                self.substitutions[key] = prior

        # update the solution
        ## add the continuous solution
        self.solution['continuoussol'] = continuous_solution
        
        ## add the optimized discrete solution
        self.solution['discretesol'] = optimized_discrete_solution
        discrete_map = {
            dr: optimized_discrete_solution['variables'][dr]
            for dr in self.discrete_resources
        }
        
        ## add a list of the discrete resources to the solution
        self.solution['discreteVariables'] = KeyDict(discrete_map)

        if 'discreteVariables' not in continuous_solution:
            continuous_solution['discreteVariables'] = KeyDict(discrete_map)

        # return the solution
        return self.solution