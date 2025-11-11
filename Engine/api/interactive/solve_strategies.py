from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

from utils.types.shared import AcceptedTypes


@runtime_checkable
class _ContextLike(Protocol):

    def solve(
        self,
        *,
        solver: str,
        verbosity: int,
        max_rate: Optional[float] = None,
        **kwargs: AcceptedTypes,
    ):
        pass


class SolveStrategy(ABC):
    """Abstract base for pluggable solve strategies."""

    @abstractmethod
    def solve(
        self,
        *,
        context: _ContextLike,
        max_rate: Optional[float],
        **kwargs: AcceptedTypes,
    ):
        pass


class BasicSolveStrategy(SolveStrategy):
    """Just forward to `context.solve()` with cvxopt."""

    def solve(
        self,
        *,
        context: _ContextLike,
        max_rate: Optional[float],
        **kwargs: AcceptedTypes,
    ):
        return context.solve(solver="cvxopt", verbosity=0, max_rate=max_rate, **kwargs)


class TradeSweepStrategy(SolveStrategy):
    """
    The real sweep logic still lives in `InteractiveModel._trade_sweep()`;
    this wrapper exists so callers can switch strategies cleanly.
    """

    def solve(
        self,
        *,
        context: _ContextLike,
        max_rate: Optional[float],
        **kwargs: AcceptedTypes,
    ):
        return context.solve(solver="cvxopt", verbosity=0, max_rate=max_rate, **kwargs)
