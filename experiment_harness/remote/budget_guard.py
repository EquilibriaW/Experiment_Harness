"""Budget enforcement: time-based only."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BudgetGuard:
    """Enforces a time limit on the experiment loop."""

    max_hours: float = 12.0

    # ── Internal state ──────────────────────────────────────────
    rounds_completed: int = field(default=0, init=False)
    start_time: float = field(default_factory=time.monotonic, init=False)

    def record_round(self) -> None:
        self.rounds_completed += 1

    @property
    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.start_time) / 3600

    def check(self) -> str | None:
        """Return a stop reason string if time limit is hit, else None."""
        if self.elapsed_hours >= self.max_hours:
            return f"Time limit reached ({self.max_hours:.1f}h)"
        return None

    def status(self) -> str:
        return (
            f"Round {self.rounds_completed} | "
            f"{self.elapsed_hours:.1f}/{self.max_hours:.1f}h"
        )
