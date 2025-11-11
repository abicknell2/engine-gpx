from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


@dataclass
class Settings:
    # data
    default_currency: str = "$"
    default_currency_iso: str = "USD"
    original_currency_data: Optional[list] = None
    hrs_per_shift: float = 7.0
    shifts_per_day: float = 2.0
    days_per_week: float = 5.0
    weeks_per_year: float = 50.0
    rate_unit: str = "hr"
    duration_unit: str = "years"

    # helpers
    def reset_defaults(self) -> None:
        """
        Return the object to its pristine state.  
        Only test-code should call this.
        """
        self.default_currency = "$"
        self.default_currency_iso = "USD"
        self.original_currency_data = None
        self.hrs_per_shift: float = 7.0
        self.shifts_per_day: float = 2.0
        self.days_per_week: float = 5.0
        self.weeks_per_year: float = 50.0
        self.rate_unit: str = "hr"
        self.duration_unit: str = "years"

    @contextmanager
    def temporary(self, **overrides: Any) -> Iterator["Settings"]:
        """
        Context-manager to override values temporarily.

        Example
        -------
        >>> with settings.temporary(default_currency="£", default_currency_iso="GBP"):
        ...     # do something in GBP…
        ...     use_model()
        >>> # <- values automatically back to previous state
        """
        # snapshot current state
        backup: Dict[str, Any] = {
            "default_currency": self.default_currency,
            "default_currency_iso": self.default_currency_iso,
            "original_currency_data": self.original_currency_data,
        }
        try:
            # apply overrides
            for k, v in overrides.items():
                setattr(self, k, v)
            yield self
        finally:
            # restore snapshot
            for k, v in backup.items():
                setattr(self, k, v)

    def clone(self) -> "Settings":
        return Settings(
            default_currency=self.default_currency,
            default_currency_iso=self.default_currency_iso,
            original_currency_data=list(self.original_currency_data) if self.original_currency_data else None,
            hrs_per_shift=self.hrs_per_shift,
            shifts_per_day=self.shifts_per_day,
            days_per_week=self.days_per_week,
            weeks_per_year=self.weeks_per_year,
            rate_unit=self.rate_unit,
            duration_unit=self.duration_unit,
        )
