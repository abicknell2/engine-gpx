"""Runtime patches for third-party libraries used by GPX."""

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