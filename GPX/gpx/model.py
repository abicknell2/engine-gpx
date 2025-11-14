"update a Model object for GPX"

from __future__ import annotations

import importlib.util
import logging
import math
from operator import itemgetter
from typing import TYPE_CHECKING, Iterable, Optional, Union

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

        # wrap a single target into the generic target_map form
        target_map: dict[gpkit.Variable, object] = {target_variable: target_value}

        # delegate to the shared discrete solver core
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
        extra_relaxation_vars: Iterable[gpkit.Variable] | None = None,
        restore_relaxations: bool = True,
        **kwargs,
    ) -> object:
        '''Solve the discrete model when individual product rates are targets.'''
        if not target_dict:
            raise ValueError('target_dict must contain at least one target variable')

        # delegate to the shared discrete solver core with a full target map
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
            extra_relaxation_vars=extra_relaxation_vars,
            restore_relaxations=restore_relaxations,
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
        extra_relaxation_vars: Iterable[gpkit.Variable] | None = None,
        restore_relaxations: bool = True,
        **kwargs,
    ) -> object:
        '''Common implementation for discrete solves.'''
        if not target_map:
            raise ValueError('target_map must contain at least one target variable')

        # helper for unpacking (variable, resource) pairs
        def _as_variable(entry: gpkit.Variable | tuple[gpkit.Variable, object]) -> gpkit.Variable:
            return entry[0] if isinstance(entry, tuple) else entry

        # copy the resources so we can adjust the list without side effects
        original_discrete_resources = list(discrete_resources)
        # map variables back to their associated resource objects (if any)
        resource_lookup: dict[gpkit.Variable, ProductionResource | object | None] = {}
        # store what changed during the run for debugging later on
        self.last_discrete_adjustments: list[dict[str, object]] = []
        self.last_discrete_feasibility_bump: dict[str, object] | None = None

        # simple labelling helper so logs show something meaningful
        def _resource_label(resource: object | None) -> str | None:
            if not resource:
                return None
            for attr in ("display_name", "name", "key"):
                value = getattr(resource, attr, None)
                if value:
                    return str(value)
            return None

        def _trace(message: str, *args: object) -> None:
            """Emit discrete-solve debug information to both logging and stdout."""

            logging.debug(message, *args)
            try:
                formatted = message % args if args else message
            except Exception:
                formatted = message
            print(f"[discrete-solve] {formatted}")

        if solve_orig:
            # run the smooth model first so we start from a valid plan
            _ = self.solve(**kwargs)

        # check to make sure there is already a solution
        if not self.solution:
            raise ValueError('the model needs to have been solved as continuous')

        # save the solution that was from the full continuous
        # remember the smooth solution for reporting
        continuous_solution = self.solution

        # pick out the discrete resources we are actually allowed to change
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

        # log a readable mapping of discrete variable keys to resources
        if resource_lookup:
            _trace(
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

        # sentinel for "no prior substitution"
        missing = object()
        targets_relaxed = False

        # remove any existing substitution for a target-like variable and remember it
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
            if vkey is not None:
                key_str = str(vkey)
                if key_str in self.substitutions:
                    value = self.substitutions[key_str]
                    del self.substitutions[key_str]
                    return key_str, value
            return None, missing

        # record and strip any existing substitutions for all target variables
        target_substitutions: dict[gpkit.Variable, tuple[object | None, object]] = {}
        relaxed_target_priors: dict[gpkit.Variable, tuple[object | None, object]] = {}
        for target_var in target_map:
            target_substitutions[target_var] = _pop_target_sub(target_var)

        # ignore extra relaxation vars that are already in the target map
        extra_relaxation_vars = [var for var in (extra_relaxation_vars or []) if var not in target_map]
        extra_relax_substitutions: dict[gpkit.Variable, tuple[object | None, object]] = {}

        # record and strip existing substitutions for any extra relaxation variables
        for relax_var in extra_relaxation_vars:
            extra_relax_substitutions[relax_var] = _pop_target_sub(relax_var)

        # allow callers to loosen rounding if needed
        tolerance = kwargs.pop('rounding_tolerance', 1e-6)
        # variables we are free to adjust
        adjustable_resources: list[gpkit.Variable] = []
        # current discrete counts for each adjustable resource
        start_dict: dict[gpkit.Variable, float] = {}
        # remember original discrete substitutions so we can restore later
        original_discrete_subs: dict[gpkit.Variable, object] = {}

        # helper to apply the requested rounding strategy and keep at least one unit
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

        # baseline continuous solution values for each discrete resource
        baseline_values: dict[gpkit.Variable, float] = {}
        for dr in self.discrete_resources:
            original_discrete_subs[dr] = self.substitutions.get(dr, missing)
            baseline_val = float(self.solution(dr))
            baseline_values[dr] = baseline_val
            rounded_val = _rounded_start(baseline_val)

            adjustable_resources.append(dr)
            start_dict[dr] = rounded_val

        # track any resources whose rate constraints get temporarily disabled
        disabled_rate_controllers: list[ProductionResource] = []

        # use a relaxed continuous solve to propose new discrete counts
        def _derive_relaxed_counts() -> dict[gpkit.Variable, float]:
            """Solve with discrete counts free to estimate new integer targets."""
            if not adjustable_resources:
                return {}

            # temporarily remove any substitutions on the discrete resource variables
            resource_priors: dict[gpkit.Variable, tuple[object | None, object]] = {}
            relaxed_counts: dict[gpkit.Variable, float] = {}

            for dr in adjustable_resources:
                resource_priors[dr] = _pop_target_sub(dr)

            try:
                self.solve(**kwargs)
            except UnknownInfeasible:
                _trace(
                    "Relaxed continuous solve infeasible while deriving discrete counts"
                )
                relaxed_counts.clear()
            except Exception:
                logging.debug(
                    "Unexpected error while deriving relaxed discrete counts", exc_info=True
                )
                print(
                    "[discrete-solve] Unexpected error while deriving relaxed discrete counts (see logs)"
                )
                relaxed_counts.clear()
            else:
                # pull out the relaxed solution and turn it into integer ceilings
                suggestion: dict[str, float] = {}
                for dr in adjustable_resources:
                    try:
                        raw_value = float(self.solution(dr))
                    except Exception:
                        continue
                    if not math.isfinite(raw_value):
                        continue
                    count = max(math.ceil(raw_value - tolerance), 1.0)
                    relaxed_counts[dr] = count
                    suggestion[str(dr.key)] = count
                if suggestion:
                    _trace(
                        "Relaxed continuous solve suggested discrete counts: %s",
                        suggestion,
                    )
            finally:
                # restore substitutions so the rest of the solve sees the original state
                for dr, (sub_key, prior) in resource_priors.items():
                    key = sub_key or dr
                    if prior is missing:
                        self.substitutions.pop(key, None)
                    else:
                        self.substitutions[key] = prior

            return relaxed_counts

        # coarse search that grows counts until a feasible combination is found
        def _search_bulk_counts() -> dict[gpkit.Variable, float]:
            """Coarsely expand discrete counts to find a feasible combination."""
            if not adjustable_resources:
                return {}

            max_attempts = max(len(adjustable_resources) * 3, 6)
            trial_counts: dict[gpkit.Variable, float] = {
                var: max(float(start_dict.get(var, 1.0)), 1.0) for var in adjustable_resources
            }
            # snapshot current substitutions so we can restore after searching
            saved_subs: dict[gpkit.Variable, object] = {
                var: self.substitutions.get(var, missing) for var in adjustable_resources
            }

            # helper to apply a whole set of counts to the model
            def _set_counts(values: dict[gpkit.Variable, float]) -> None:
                for var, val in values.items():
                    self.substitutions[var] = max(float(val), 1.0)

            # helper to restore the saved state
            def _restore() -> None:
                for var, prior in saved_subs.items():
                    if prior is missing:
                        self.substitutions.pop(var, None)
                    else:
                        self.substitutions[var] = prior

            feasible: dict[gpkit.Variable, float] = {}
            try:
                # first, expand counts until some combination solves
                attempts = 0
                while attempts < max_attempts:
                    _set_counts(trial_counts)
                    try:
                        self.solve(**kwargs)
                    except UnknownInfeasible:
                        # grow all counts when infeasible
                        for var in adjustable_resources:
                            current = trial_counts.get(var, 1.0)
                            step = max(1.0, math.ceil(current * 0.5))
                            trial_counts[var] = max(current + step, current + 1.0)
                        attempts += 1
                        continue
                    else:
                        # record a feasible combination based on the solution
                        feasible = {
                            var: max(math.ceil(float(self.solution(var)) - tolerance), 1.0)
                            for var in adjustable_resources
                        }
                        break

                if not feasible:
                    return {}

                # then try to shrink individual counts with a simple binary search
                changed = True
                while changed:
                    changed = False
                    for var in adjustable_resources:
                        lower = max(start_dict.get(var, 1.0), 1.0)
                        upper = max(feasible.get(var, lower), lower)
                        while lower < upper:
                            mid = math.floor((lower + upper) / 2)
                            if mid < lower:
                                break
                            probe = dict(feasible)
                            probe[var] = max(float(mid), 1.0)
                            _set_counts(probe)
                            try:
                                self.solve(**kwargs)
                            except UnknownInfeasible:
                                lower = mid + 1
                            else:
                                new_val = max(
                                    math.ceil(float(self.solution(var)) - tolerance), 1.0
                                )
                                if new_val < feasible[var] - 1e-9:
                                    feasible[var] = new_val
                                    upper = new_val
                                    changed = True
                                else:
                                    break

                _trace(
                    "Bulk count search candidate counts: %s",
                    {str(var.key): feasible.get(var, 0.0) for var in adjustable_resources},
                )
                return feasible
            finally:
                _restore()

        # global scaling search that moves all counts together until feasible
        def _scale_counts_until_feasible(
            max_scale_factor: float = 64.0,
            refinement_iters: int = 12,
        ) -> tuple[dict[gpkit.Variable, float], object | None]:
            """Scale all discrete counts together until the relaxed model becomes feasible."""

            if not adjustable_resources:
                return {}, None

            # keep a copy of the starting counts in case we need to revert
            initial_counts = {dr: start_dict.get(dr, 1.0) for dr in adjustable_resources}

            # build counts from a scale factor and baseline continuous values
            def _counts_from_scale(scale: float) -> dict[gpkit.Variable, float]:
                scaled: dict[gpkit.Variable, float] = {}
                for dr in adjustable_resources:
                    base = baseline_values.get(dr, 1.0)
                    candidate = max(base * scale, 1.0)
                    scaled[dr] = max(math.ceil(candidate - tolerance), 1.0)
                return scaled

            # apply a set of counts into substitutions
            def _apply(counts: dict[gpkit.Variable, float]) -> None:
                for var, value in counts.items():
                    self.substitutions[var] = value

            lower_scale = 1.0
            upper_scale = 1.0
            feasible_counts: dict[gpkit.Variable, float] | None = None
            feasible_solution: object | None = None

            # grow the scale until we hit a feasible relaxed solution or hit the limit
            while upper_scale <= max_scale_factor:
                probe_counts = _counts_from_scale(upper_scale)
                _apply(probe_counts)
                try:
                    feasible_solution = self.solve(**kwargs)
                except UnknownInfeasible:
                    lower_scale = upper_scale
                    upper_scale *= 2.0
                else:
                    feasible_counts = probe_counts
                    break

            if feasible_counts is None:
                _trace(
                    "Scaled count search failed to restore feasibility (max factor %.2f)",
                    upper_scale,
                )
                _apply(initial_counts)
                return {}, None

            _trace(
                "Scaled count search candidate counts: %s",
                {str(dr.key): feasible_counts.get(dr, 0.0) for dr in adjustable_resources},
            )

            # refine the scale factor to get closer to a minimal feasible set of counts
            for _ in range(refinement_iters):
                if upper_scale - lower_scale <= 1e-3:
                    break
                mid = 0.5 * (lower_scale + upper_scale)
                probe_counts = _counts_from_scale(mid)
                _apply(probe_counts)
                try:
                    feasible_solution = self.solve(**kwargs)
                except UnknownInfeasible:
                    lower_scale = mid
                else:
                    feasible_counts = probe_counts
                    upper_scale = mid

            _apply(feasible_counts)
            return feasible_counts, feasible_solution

        # last resort path that bumps one discrete resource at a time in round-robin
        def _run_safeguarded_bumps() -> object:
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
                        _trace(
                            "Discrete feasibility restored after bumping %s to %s%s",
                            last_feasibility_bump["variable"],
                            last_feasibility_bump["final"],
                            log_suffix,
                        )
                    return cur_solution
                except UnknownInfeasible:
                    _trace(
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
                    # pick the next adjustable resource in a cyclic fashion
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

        try:
            # temporarily lift rate constraints on any resources that support it
            for dr in self.discrete_resources:
                resource = resource_lookup.get(dr)
                disable = getattr(resource, "disable_rate_constraint", None)
                if callable(disable) and disable():
                    # record which cells have their caps lifted
                    disabled_rate_controllers.append(resource)

            # seed substitutions with the initial rounded counts
            self.substitutions.update(start_dict)
            initial_start_dict = dict(start_dict)

            try:
                # try the discrete solve with just the rounded starts
                cur_solution = self.solve(**kwargs)
            except UnknownInfeasible:
                _trace(
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
                    _trace(
                        "Discrete solve still infeasible after ceiling counts: %s",
                        {str(dr.key): start_dict[dr] for dr in adjustable_resources},
                    )

                    # Before relaxing the target rates, try a global scaled search to
                    # see if simply giving every discrete resource a proportional bump
                    # restores feasibility.
                    scaled_counts, scaled_solution = _scale_counts_until_feasible()
                    if scaled_counts:
                        for dr, count in scaled_counts.items():
                            start_dict[dr] = count

                        cur_solution = scaled_solution or self.solution
                        targets_relaxed = True
                        _trace(
                            "Discrete feasibility restored after early scaled count search: %s",
                            {str(dr.key): start_dict[dr] for dr in adjustable_resources},
                        )

                    elif relax_targets_on_infeasible and not targets_relaxed and target_map:
                        _trace(
                            "Relaxing discrete rate targets to allow throughput to float with fixed counts",
                        )
                        # mark all target variables as "relaxed" for this pass
                        for var in list(target_map.keys()):
                            relaxed_target_priors[var] = target_substitutions.get(var, (None, missing))
                            target_map[var] = None
                            sub_key, _ = target_substitutions.get(var, (None, missing))
                            target_substitutions[var] = (sub_key, missing)
                        targets_relaxed = True

                        try:
                            # see if simply removing the target substitutions is enough
                            cur_solution = self.solve(**kwargs)
                        except UnknownInfeasible:
                            relaxed_success = False
                            relaxed_counts: dict[gpkit.Variable, float] = {}
                            if relax_targets_on_infeasible:
                                _trace(
                                    "Relaxed rate targets remained infeasible; deriving new discrete counts",
                                )
                                # try to derive better counts from a relaxed continuous solve
                                relaxed_counts = _derive_relaxed_counts()

                            if relaxed_counts:
                                # apply any relaxed counts we discovered
                                for dr, count in relaxed_counts.items():
                                    start_dict[dr] = count
                                    self.substitutions[dr] = count

                                try:
                                    cur_solution = self.solve(**kwargs)
                                except UnknownInfeasible:
                                    _trace(
                                        "Discrete solve still infeasible after applying relaxed counts",
                                    )
                                    targets_relaxed = False
                                else:
                                    relaxed_success = True
                                    targets_relaxed = True
                                    _trace(
                                        "Discrete feasibility restored after applying relaxed counts: %s",
                                        {str(dr.key): start_dict[dr] for dr in adjustable_resources},
                                    )
                            if not relaxed_success:
                                _trace(
                                    "Relaxed rate targets remained infeasible; falling back to safeguarded bumps",
                                )
                                targets_relaxed = False

                                # try scaling all counts together before going to bulk/bump paths
                                if relax_targets_on_infeasible and not targets_relaxed:
                                    scaled_counts, scaled_solution = _scale_counts_until_feasible()
                                    if scaled_counts:
                                        for dr, count in scaled_counts.items():
                                            start_dict[dr] = count

                                        cur_solution = scaled_solution or self.solution
                                        relaxed_success = True
                                        targets_relaxed = True
                                        _trace(
                                            "Discrete feasibility restored after scaled count search: %s",
                                            {str(dr.key): start_dict[dr] for dr in adjustable_resources},
                                        )

                                # if scaling also fails, fall back to the coarse bulk search
                                if relax_targets_on_infeasible and not targets_relaxed:
                                    bulk_counts = _search_bulk_counts()
                                    if bulk_counts:
                                        for dr, count in bulk_counts.items():
                                            start_dict[dr] = count
                                            self.substitutions[dr] = count

                                        try:
                                            cur_solution = self.solve(**kwargs)
                                        except UnknownInfeasible:
                                            _trace(
                                                "Discrete solve still infeasible after bulk count search"
                                            )
                                            targets_relaxed = False
                                        else:
                                            relaxed_success = True
                                            targets_relaxed = True
                                            _trace(
                                                "Discrete feasibility restored after bulk count search: %s",
                                                {str(dr.key): start_dict[dr] for dr in adjustable_resources},
                                            )

                    # if we never managed to keep targets relaxed, go to the safeguarded bump loop
                    if not targets_relaxed:
                        cur_solution = _run_safeguarded_bumps()

            # convert target strategy into a sign for comparisons
            target_strategy_flag = 1 if target_strategy == 'above' else -1

            # tolerant float conversion that also handles pint-like magnitudes
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

            # robustly look up a variable’s value in a solution object
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

            # check whether all numeric targets in target_map are currently satisfied
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

                # find the most sensitive variable according to the chosen strategy
                most_sens = sensFinder(candidates, key=itemgetter(0))[1]
                # bump that discrete resource by one unit
                self.substitutions[most_sens] += 1

                # re-solve and re-check the targets
                cur_solution = self.solve(**kwargs)
                cur_solution_targets_met = _targets_met(cur_solution)

            # save the solution for the peak discrete (optimised target_variable)
            optimised_discrete_solution = self.solution

            # build a summary of any discrete adjustments that actually changed values
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
                _trace("Discrete feasibility adjustments applied: %s", adjustments_summary)

        finally:
            # restore any rate constraints that were temporarily disabled
            for resource in reversed(disabled_rate_controllers):
                restore = getattr(resource, "restore_rate_constraint", None)
                if callable(restore):
                    # put the rate caps back before returning
                    restore()

        # restore the original discrete substitutions for all discrete resources
        for dr, prior in original_discrete_subs.items():
            if prior is missing:
                try:
                    del self.substitutions[dr]
                except KeyError:
                    pass
            else:
                self.substitutions[dr] = prior

        # put the original cost back if we temporarily switched to a discrete cost
        if new_cost and orig_cost is not None:
            self.cost = orig_cost

        # substitute the target(s)
        for var, (sub_key, prior) in target_substitutions.items():
            key = sub_key or var
            if prior is missing:
                # if there was no prior substitution, fall back to the current target value
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

        # restore substitutions for any extra relaxation variables if requested
        if restore_relaxations:
            for var, (sub_key, prior) in extra_relax_substitutions.items():
                key = sub_key or var
                if prior is missing:
                    try:
                        del self.substitutions[key]
                    except KeyError:
                        pass
                else:
                    self.substitutions[key] = prior

        # update the solution
        ## add the continuous solution
        self.solution['continuoussol'] = continuous_solution
        
        ## add the optimised discrete solution
        self.solution['discretesol'] = optimised_discrete_solution
        discrete_map = {
            dr: optimised_discrete_solution['variables'][dr]
            for dr in self.discrete_resources
        }
        
        ## add a list of the discrete resources to the solution
        self.solution['discreteVariables'] = KeyDict(discrete_map)

        if 'discreteVariables' not in continuous_solution:
            continuous_solution['discreteVariables'] = KeyDict(discrete_map)

        # return the solution
        return self.solution
