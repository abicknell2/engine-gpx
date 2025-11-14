from itertools import groupby
from typing import Optional, Union

import gpkit
from gpkit import Variable, VarKey, ureg
import numpy as np

from api.acyclic_interpreter import make_parametric_constraint, make_parametric_variable
from api.module_types.production_finance import ProductionFinance
import gpx
from gpx.dag.parametric import (
    ParametricVariable,
    XFromSplits,
    XSplitBySingleCell,
)
from gpx.manufacturing import QNACell
from gpx.multiclass import MClass
from gpkit import units as gpkit_units



def _val(
    subs: dict[Variable, Union[float, tuple[float, str]]],
    var: Variable,
) -> float:
    """
    Return a numeric value for *var*.

    • First look in *subs*  
    • Else fall back to var.value  
    • Raises KeyError if still symbolic
    """
    if var in subs:
        raw = subs[var]
        return float(raw[0]) if isinstance(raw, tuple) else float(raw)

    try:
        return float(var.value)  # constant on the varkey itself
    except (AttributeError, TypeError):
        raise KeyError


def nominal_flow_time(
    mc: MClass,
    subs: dict[Variable, Union[float, tuple[float, str]]],
) -> Optional[float]:
    """
    Sum nominal service times for all main-line and feeder cells.
    Returns None if any term is still symbolic.
    """
    total = 0.0
    all_cells = (mc.line.cells + [cell for feeder in mc.feeder_lines for cell in feeder.cells])

    for cell in all_cells:
        # try t_nu directly
        try:
            total += _val(subs, cell.tnu)
            continue
        except KeyError:
            pass

        # else fall back to t × ν
        try:
            total += _val(subs, cell.t) * _val(subs, cell.nu)
        except KeyError:
            return None

    return total


def update_w_cap(system_module, subs, duration=None, slack=0.25, hard_fallback=2e7):
    """
    Set W_cap for product classes and feeder wrappers.

    Cap recipe (choose the largest -> never over-tighten):
      - duration-based cap, *only* if duration is clearly in time units
      - nominal-minimum flow time estimate (raw cycle time, no queues)
      - hard_fallback
    """

    def _to_minutes_or_none(x):
        # Accept gpkit quantities or dicts like {"value": 5, "unit": "h"}; reject bare numbers
        try:
            if hasattr(x, "to"):
                return x.to("minute").magnitude
            if isinstance(x, dict):
                val = float(x.get("value", 0))
                unit = str(x.get("unit", "")).strip()
                if val > 0 and unit:
                    return (val * gpkit.ureg(unit)).to("minute").magnitude
        except Exception:
            pass
        return None  # do not guess if unitless

    def _nominal_flow_minutes(entity):
        # Sum service components along the path(s) that define this class/feeder.
        # Avoid queueing terms on purpose. This is a physics lower bound.
        total = 0.0
        for c in getattr(entity, "cells", []):
            # prefer bound t_eta if available, else t*eta_t or t
            t_eta = None
            try:
                t_eta = subs.get(c.t_eta, None)
            except Exception:
                pass
            if t_eta is not None:
                try:
                    total += float(t_eta.to("minute"))
                    continue
                except Exception:
                    pass

            # fallback
            t = subs.get(getattr(c, "t", None), None)
            eta_t = subs.get(getattr(c, "eta_t", None), None)
            try:
                if t is not None and eta_t is not None:
                    total += float((t * eta_t).to("minute"))
                elif t is not None:
                    total += float(t.to("minute"))
                else:
                    return None
            except Exception:
                return None

        return total

    dur_min = _to_minutes_or_none(duration)

    entities = list(system_module.mcclasses.values())
    entities += [fc for feeders in getattr(system_module, "feeder_classes", {}).values() for fc in feeders]

    for e in entities:
        if e.W_cap in subs:
            continue

        # candidate caps
        cap_candidates = []

        if dur_min is not None:
            cap_candidates.append((1.0 + slack) * dur_min)

        est = _nominal_flow_minutes(e)
        if est:
            cap_candidates.append((1.0 + slack) * est)

        # largest available candidate so cap is never tighter than physics
        cap = max(cap_candidates) if cap_candidates else hard_fallback
        subs[e.W_cap] = cap

    return subs

def set_x_rate_values(multiproduct, sysm, mcsys, newsubs):
    """
    Normalize *to the variable's own units* (mc.lam.units, mcsys.lam.units),
    and build the system rate by summing the already-normalized class rates.
    """
    prod_rates = sysm.prod_rates

    # Start an explicitly unitful zero in the system-rate variable's units
    try:
        unit_zero = 0 * mcsys.lam.units
    except Exception:
        # very defensive: fall back to count/hr if units missing, but prefer declared units
        unit_zero = 0 * gpkit.units("count/hr")

    system_rate_accum = unit_zero

    # Bind each *product class* rate and its main-line q-cell rates
    for pname, rate in prod_rates.items():
        if rate is None:
            raise ValueError(f"Product '{pname}' is missing a rate")

        mc = sysm.mcclasses[pname]
        target_units = getattr(mc.lam, "units", gpkit.units("count/hr"))

        # Convert/attach units to match the variable we are substituting
        if hasattr(rate, "to"):
            class_rate = rate.to(target_units)
        else:
            class_rate = rate * target_units  # attach the target variable's units

        # Bind class lam and any exposed q-lams
        multiproduct._substitutions_[mc.lam] = class_rate
        newsubs[mc.lam] = class_rate

        for qlam in getattr(mc, "all_lams", []):
            multiproduct._substitutions_[qlam] = class_rate
            newsubs[qlam] = class_rate

        # Accumulate the system rate in consistent units
        system_rate_accum = system_rate_accum + class_rate

    # Bind *feeder wrappers* to the parent class rate (copy-through)
    # NOTE: we no longer substitute feeder *line/cell* rates here; those are added via acyclic constraints
    for fc in (getattr(sysm, "feeders", None) or []):
        parent = getattr(fc, "parent_class", None)
        if not parent:
            continue

        # Prefer already-normalized parent rate we just assigned
        parent_rate = newsubs.get(parent.lam, getattr(parent.lam, "value", None))

        if parent_rate is None:
            # Last-chance: pull original and normalize to the parent's lam units
            pr = prod_rates.get(getattr(parent, "name", None))
            if pr is not None:
                pu = getattr(parent.lam, "units", gpkit.units("count/hr"))
                parent_rate = pr.to(pu) if hasattr(pr, "to") else pr * pu

        if parent_rate is None:
            continue  # nothing reliable to bind

        # wrapper's own lam
        multiproduct._substitutions_[fc.lam] = parent_rate
        newsubs[fc.lam] = parent_rate

        # IMPORTANT: do NOT set feeder line/cell lambdas here anymore.
        # Those are now handled in acyclic constraints to preserve sensitivities and avoid unit issues.

    # Bind the overall system rate (sum of normalized product rates) in mcsys.lam units
    sys_units = getattr(mcsys.lam, "units", gpkit.units("count/hr"))
    if hasattr(system_rate_accum, "to"):
        hourly_system_rate = system_rate_accum.to(sys_units)
    else:
        hourly_system_rate = system_rate_accum * sys_units  # defensive

    multiproduct._substitutions_[mcsys.lam] = hourly_system_rate
    newsubs[mcsys.lam] = hourly_system_rate

    return newsubs

def compute_rate_shares(owner, prod_rates: dict | None = None, classes: dict | None = None) -> dict[str, float]:
    """Return normalized shares derived from the configured product rates."""

    class_map = classes or getattr(owner, "mcclasses", {}) or {}
    rate_inputs = prod_rates or getattr(owner, "prod_rates", {}) or {}
    candidate_names = list(class_map.keys()) or list(rate_inputs.keys())

    rate_magnitudes: dict[str, float] = {}

    for pname in candidate_names:
        rate_val = rate_inputs.get(pname)
        if rate_val is None:
            continue

        try:
            quantity = rate_val if hasattr(rate_val, "to") else rate_val * gpkit.units("count/hr")
            magnitude = float(quantity.to("count/hr").magnitude)
        except Exception:
            try:
                magnitude = float(rate_val)
            except Exception:
                continue

        rate_magnitudes[pname] = magnitude

    if not rate_magnitudes:
        return {}

    total_rate = sum(rate_magnitudes.values())
    if total_rate <= 0:
        share = 1.0 / max(len(rate_magnitudes), 1)
        return {p: share for p in rate_magnitudes}

    return {p: mag / total_rate for p, mag in rate_magnitudes.items()}


def add_x_splits_to_acyclic_constraints(multiproduct, aconstr):
    split_pvs: dict[str, ParametricVariable] = {}

    if multiproduct.by_split:
        # Build one ParametricVariable for *each* class‑split x
        # using the numeric fraction already stored in multiproduct.prod_rates
        for pname, mc in multiproduct.mcclasses.items():
            split_vk = mc.X  # VarKey defined in MClass
            split_val = multiproduct.prod_rates[pname]  # e.g. 0.25  (never None)
            split_pvs[pname] = make_parametric_variable(
                inputvar=split_vk,
                name=f"{pname} :: Class Split",
                substitution=(split_val, "-"),  # numeric, unit‑less
            )
    else:
        normalized_shares = compute_rate_shares(multiproduct)

        for pname, share in normalized_shares.items():
            split_key = VarKey(f"{pname} :: Rate Split Input", units="-")
            split_pvs[pname] = make_parametric_variable(
                key=split_key,
                name=f"{pname} :: Class Split",
                substitution=(share, "-"),
            )

    all_split_pvs = list(split_pvs.values())

    attr_name_map = {
        "x": ("x",),
        "xInv": ("xInv", "xinv"),
        "xItem": ("xItem", "xitem"),
    }

    for name, mcell in multiproduct.gpxObject["system"].mcells.items():
        if all(isinstance(q, QNACell) and getattr(q, "is_feeder", False) for q in mcell.cells):
            continue

        if len(mcell.cells) == 1:
            _append_single_cell_splits(aconstr, mcell, name)
            continue

        idx_by_prod = {
            q.product_name: i for i, q in enumerate(mcell.cells) if hasattr(q, "product_name")
        }

        attr_lists: dict[str, list] = {}
        for label, candidates in attr_name_map.items():
            for candidate in candidates:
                raw_list = getattr(mcell, candidate, None)
                if raw_list is not None:
                    attr_lists[label] = raw_list
                    break

        for pname, split_pv in split_pvs.items():
            idx = idx_by_prod.get(pname)
            if idx is None:
                continue

            for label, raw_list in attr_lists.items():
                try:
                    raw_var = raw_list[idx]
                except (IndexError, TypeError):
                    continue
                if raw_var is None:
                    continue

                label_suffix = f"{label}_{idx}"
                out_pv = make_parametric_variable(
                    inputvar=raw_var,
                    name=f"{pname} :: {name} {label_suffix}",
                )
                aconstr.append(
                    XFromSplits(
                        split_var=split_pv,
                        all_split_vars=all_split_pvs,
                        output_pv=out_pv,
                    )
                )

    for cell_name, mcell in multiproduct.gpxObject["system"].mcells.items():
        # find the indices of the secondary QNA‐cells in this MCell
        sec_idxs = [i for i, q in enumerate(mcell.cells) if getattr(q, "secondary_cell", False)]
        if not sec_idxs:
            continue

        # for each of the three split‐types (x, xinv, xitem)
        for label, candidates in attr_name_map.items():
            raw_list = None
            for candidate in candidates:
                raw_list = getattr(mcell, candidate, None)
                if raw_list is not None:
                    break
            if raw_list is None:
                continue  # e.g. maybe this MCell has no xitem variable at all

            for i in sec_idxs:
                raw_var = raw_list[i]  # this is the GPkit Variable
                prod = mcell.cells[i].product_name
                split_pv = split_pvs.get(prod)
                if split_pv is None:
                    continue

                # wrap it in a ParametricVariable for the acyclic interpreter
                out_pv = make_parametric_variable(
                    inputvar=raw_var,
                    name=f"{cell_name} :: sec-{label}_{i}",
                )

                aconstr.append(
                    XFromSplits(
                        split_var=split_pv,
                        all_split_vars=all_split_pvs,
                        output_pv=out_pv,
                    )
                )

    return aconstr



def add_rate_links_to_acyclic_constraints(sysm, aconstr):
    """Extend acyclic graph with:
       line.lam = parent_class.lam * batch_qty
       (optionally) feeder cell lam = line.lam  (only if units match)
    Only builds constraints when both inputs are numeric to avoid orphan PVs.
    """
    feeders = getattr(sysm, "feeders", None) or []
    if not feeders:
        return aconstr

    # pull best-known substitutions from likely owners
    subs: dict = {}
    for owner in (sysm,
                  getattr(sysm, "interaction", None),
                  getattr(sysm, "interactive", None)):
        if owner is None:
            continue
        try:
            subs.update(getattr(owner, "_substitutions_", {}) or {})
        except Exception:
            pass
        try:
            subs.update(getattr(owner, "substitutions", {}) or {})
        except Exception:
            pass

    # shared PV registry so the same var maps to the same PV across constraints
    pv_registry: dict[str, gpx.dag.parametric.ParametricVariable] = {}

    def _units_str(vk, default=""):
        try:
            # gpkit Var has unitsstr() on varkey; gpx Variable may also carry units
            if hasattr(vk, "unitsstr") and callable(vk.unitsstr):
                s = vk.unitsstr() or ""
                return s if s != "-" else default
        except Exception:
            pass
        return default

    def _norm_sub(val, vk):
        # normalize to (magnitude, unit_str)
        try:
            if isinstance(val, tuple) and len(val) == 2:
                return (float(val[0]), str(val[1]))
        except Exception:
            pass
        try:
            # pint Quantity
            if hasattr(val, "to"):
                unit = _units_str(vk, "")
                # if vk has no declared unit, leave unit empty so interpreter won’t convert
                if unit:
                    vq = val.to(unit)
                    return (float(vq.magnitude), unit)
                else:
                    return (float(val.magnitude), str(val.units))
        except Exception:
            pass
        try:
            # plain number
            return (float(val), _units_str(vk, ""))
        except Exception:
            return None

    def _get_sub(vk):
        # look up by object, then by key string, else fall back to constant value on varkey
        if vk in subs:
            return _norm_sub(subs[vk], vk)
        k = str(getattr(vk, "key", vk))
        if k in subs:
            return _norm_sub(subs[k], vk)
        try:
            vval = getattr(vk, "value", None)
            if vval is not None:
                return _norm_sub(vval, vk)
        except Exception:
            pass
        return None

    def _find_batch_vk(line):
        # prefer explicit attribute
        if hasattr(line, "batch_qty"):
            return getattr(line, "batch_qty")
        # try cumulative monomial exponents (FeederLine.cuml_batch_qty, etc.)
        for attr in ("cuml_batch_qty",):
            mon = getattr(line, attr, None)
            if mon is None:
                continue
            # gpkit Monomial: mon.exp or mon.exps[0] is a dict {VarKey: exp}
            try:
                if hasattr(mon, "exp") and isinstance(mon.exp, dict) and mon.exp:
                    return next(iter(mon.exp.keys()))
                if hasattr(mon, "exps") and isinstance(mon.exps, tuple) and mon.exps:
                    return next(iter(mon.exps[0].keys()))
            except Exception:
                pass
        # last-resort heuristic
        try:
            for vk in getattr(line, "varkeys", set()):
                if "Quantity" in str(vk):
                    return vk
        except Exception:
            pass
        return None

    def _compatible_units(u_src: str, u_dst: str) -> bool:
        # only emit identity when both sides declare the same unit string
        us = (u_src or "").strip()
        ud = (u_dst or "").strip()
        return bool(us and ud and us == ud)

    new_nodes = []

    for feeder in feeders:
        parent = getattr(feeder, "parent_class", None) or getattr(feeder, "parent", None)
        line = getattr(feeder, "line", None)
        if not (parent and line):
            continue

        wrapper_vk = getattr(parent, "lam", None)
        line_vk = getattr(line, "lam", None) or getattr(line, "lam_feeder", None)
        batch_vk = _find_batch_vk(line)
        if not (wrapper_vk and line_vk and batch_vk):
            continue

        wrapper_sub = _get_sub(wrapper_vk)
        batch_sub = _get_sub(batch_vk)

        # only build when inputs are numeric; otherwise the GP constraints will bind during solve
        if wrapper_sub is None or batch_sub is None:
            continue

        # 1) line rate = wrapper rate * batch qty
        variables_1 = {
            str(wrapper_vk.key): wrapper_vk,
            str(batch_vk.key): batch_vk,
            str(line_vk.key): line_vk,
        }
        c1 = {
            "name": f"{getattr(parent, 'name', 'Class')} :: Feeder Wrapper Rate",
            "key": str(line_vk.key),
            "unit": _units_str(line_vk, "count**2/h"),
            "value": [str(wrapper_vk.key), "*", str(batch_vk.key)],
            "constrno": f"feeder_wrapper_rate::{str(line_vk.key)}",
        }
        pc1 = make_parametric_constraint(
            inputconstraint=c1,
            variables=variables_1,
            parametric_vars=pv_registry,
            substitutions={
                str(wrapper_vk.key): wrapper_sub,
                str(batch_vk.key): batch_sub,
            },
        )
        new_nodes.append(pc1)

        # 2) optionally propagate to a downstream feeder cell when units match
        try:
            cells = getattr(line, "cells", None) or []
            target_cell = cells[-1] if cells else None
        except Exception:
            target_cell = None

        if target_cell is not None:
            cell_lam_vk = getattr(target_cell, "lam", None)
            if cell_lam_vk is not None:
                u_line = _units_str(line_vk, "")
                u_cell = _units_str(cell_lam_vk, "")
                if _compatible_units(u_line, u_cell):
                    variables_2 = {
                        str(line_vk.key): line_vk,
                        str(cell_lam_vk.key): cell_lam_vk,
                    }
                    c2 = {
                        "name": f"{getattr(parent, 'name', 'Class')} :: Feeder Line To Cell",
                        "key": str(cell_lam_vk.key),
                        "unit": u_cell,
                        "value": [str(line_vk.key)],
                        "constrno": f"feeder_line_to_cell::{str(cell_lam_vk.key)}",
                    }
                    pc2 = make_parametric_constraint(
                        inputconstraint=c2,
                        variables=variables_2,
                        parametric_vars=pv_registry,
                        substitutions={},  # line value comes from c1 via shared PV
                    )
                    new_nodes.append(pc2)
                # else: let the solver constraints tie line → cell

    if new_nodes:
        aconstr.extend(new_nodes)
    return aconstr



def _append_single_cell_splits(aconstr, mcell, name):
    xi_vk = mcell.x[0]
    pname = getattr(mcell.cells[0], "product_name", "Cell0")
    xi_pv = make_parametric_variable(
        inputvar=xi_vk, name=f"{pname} :: {name} x_0", substitution=(1, "-")
    )
    aconstr.append(XSplitBySingleCell(xi_pv))
    if getattr(mcell, "xitem", None):
        xi_vk_item = mcell.xitem[0]
        xi_pv_item = make_parametric_variable(
            inputvar=xi_vk_item,
            name=f"{pname} :: {name} xItem_0",
            substitution=(1.0, "-"),
        )
        aconstr.append(XSplitBySingleCell(xi_pv_item, value=1.0))

    if getattr(mcell, "xinv", None):
        xi_vk_inv = mcell.xinv[0]
        xi_pv_inv = make_parametric_variable(
            inputvar=xi_vk_inv, name=f"{pname} :: {name} xInv_0", substitution=(1, "-")
        )
        aconstr.append(XSplitBySingleCell(xi_pv_inv, value=1.0))
    return aconstr


def set_floorspace_costs(multiproduct):
    floorspacecost = None
    floorspace_vars = multiproduct.find_variables(
        filter_category='cells',
        filter_property='Floor Space',
        return_data='dict',
        emptyisnone=True,
    )

    if floorspace_vars:
        # if it is coming from variants, should be no problem using the first cell information if variants
        if True:
            # sort by cell name and groupby
            # groupby cells
            floorspace_vars_bycell = {var.type: var.gpxObject for var in floorspace_vars.values()}

            # Get the floor space recurring cost
            fsrc = getattr(
                next(
                    filter(
                        lambda x: x.name == 'Floor Space Recurring Cost',
                        multiproduct.find_variables(
                            filter_category='floor space', filter_property='Cost', return_data='list'
                        ),
                    ), None
                ),  # Iterator should default to None
                'gpxObject',
                None
            )  # safe to get the attribute

            # get the floor space non-recurring cost
            fsnrc = getattr(
                next(
                    filter(
                        lambda x: x.name == 'Floor Space Non-Recurring Cost',
                        multiproduct.find_variables(
                            filter_category='floor space', filter_property='Cost', return_data='list'
                        ),
                    ), None
                ),  # Iterator should default to None
                'gpxObject',
                None
            )  # safe to get the attribute

            # make the floorspace costs
            floorspacecost = gpx.mfgcosts.FloorspaceCost(
                cells=multiproduct.shared_cells_gpx_dict,
                cell_space=floorspace_vars_bycell,
                default_currency=multiproduct.settings.default_currency_iso,
                cost_nr=fsnrc,
                cost_r=fsrc,
                separate_costs=True
            )

            # add to the gpx object
            multiproduct.gpxObject['floorspaceCost'] = floorspacecost
    return floorspacecost