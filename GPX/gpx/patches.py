"""Runtime patches for third-party libraries used by GPX."""

# Quick recap for the team: this module is where I patched the zero-division
# crash we saw coming out of GPkit. Their
# ``PosynomialInequality.sens_from_dual`` helper divides by the constraint's
# constant coefficient when it folds dual values back into primal sensitivities.
# When that coefficient is effectively zero (which happens in our discrete
# solve), the division trips ``ZeroDivisionError`` and the whole solve aborts.
#
# To keep the behaviour identical in the normal case, I copied the GPkit
# routine verbatim and only changed the tiny bit that handles the constant-term
# contribution. Now, whenever the constant coefficient vanishes, we simply skip
# that portion of the calculation and continue using the dual values provided by
# the solver. That preserves GPkit's math when the coefficient is non-zero while
# letting us side-step the pathological case that caused the crash.
#
# The ``apply_patches`` function at the bottom performs the actual override:
# importing ``gpx.patches`` rebinds
# ``PosynomialInequality.sens_from_dual`` to this guarded copy. Because our
# package's ``__init__`` imports ``patches`` on load, the swap happens before any
# models are solved, so every subsequent call into GPkit automatically picks up
# the patched logic.

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
from gpkit.nomials.map import HashVector
from gpkit.nomials.math import PosynomialInequality

_LOGGER = logging.getLogger(__name__)


def _safe_sens_from_dual(
    self: PosynomialInequality,
    la: float,
    nu: np.ndarray,
    _result: Any,
) -> Tuple[HashVector, float]:
    """Replica of gpkit's PosynomialInequality.sens_from_dual with zero guards."""
    presub, = self.unsubbed
    if hasattr(self, "pmap"):
        nu_ = np.zeros(len(presub.hmap))
        for i, mmap in enumerate(self.pmap):
            for idx, percentage in mmap.items():
                nu_[idx] += percentage * nu[i]
        del self.pmap  # not needed after dual has been derived
        if hasattr(self, "const_mmap"):
            const_coeff = getattr(self, "const_coeff", None)
            if const_coeff is None or abs(const_coeff) < 1e-12:
                _LOGGER.debug(
                    "Skipping constant dual contribution for %s due to near-zero coefficient",
                    self,
                )
            else:
                scale = (1 - const_coeff) / const_coeff
                for idx, percentage in self.const_mmap.items():
                    nu_[idx] += percentage * la * scale
            del self.const_mmap  # not needed after dual has been derived
            if hasattr(self, "const_coeff"):
                del self.const_coeff
        nu = nu_
    self.v_ss = HashVector()
    if self.parent:
        self.parent.v_ss = self.v_ss
    if self.generated_by:
        self.generated_by.v_ss = self.v_ss
    for nu_i, exp in zip(nu, presub.hmap):
        for vk, x in exp.items():
            self.v_ss[vk] = nu_i * x + self.v_ss.get(vk, 0)
    return self.v_ss, la


def apply_patches() -> None:
    """Apply GPX specific runtime patches."""
    PosynomialInequality.sens_from_dual = _safe_sens_from_dual  # type: ignore[assignment]


apply_patches()
