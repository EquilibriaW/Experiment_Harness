"""Budget guard: time tracking with efficiency metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BudgetGuard:
    """Tracks time budget and efficiency metrics."""

    max_hours: float = 24.0
    low_budget_threshold: float = 0.2  # fraction of budget remaining

    # ── Internal ────────────────────────────────────────────────
    start_time: float = field(default_factory=time.monotonic, init=False)
    rounds_completed: int = field(default=0, init=False)

    def record_round(self) -> None:
        self.rounds_completed += 1

    @property
    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.start_time) / 3600

    @property
    def remaining_hours(self) -> float:
        return max(0, self.max_hours - self.elapsed_hours)

    @property
    def budget_low(self) -> bool:
        return self.remaining_hours / self.max_hours < self.low_budget_threshold

    @property
    def budget_exhausted(self) -> bool:
        return self.elapsed_hours >= self.max_hours

    def check(self) -> str | None:
        """Return stop reason if budget exhausted, else None."""
        if self.budget_exhausted:
            return f"Time limit reached ({self.max_hours:.1f}h)"
        return None

    def status(self) -> str:
        return (
            f"Round {self.rounds_completed} | "
            f"{self.elapsed_hours:.1f}/{self.max_hours:.1f}h | "
            f"{self.remaining_hours:.1f}h remaining"
        )
