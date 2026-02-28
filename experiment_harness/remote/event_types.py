"""Event types for the event-driven experiment loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class EventKind(Enum):
    INIT = auto()
    GPU_FREED = auto()
    RUN_COMPLETED = auto()
    RUN_FAILED = auto()
    BUDGET_TICK = auto()
    IDLE_ALERT = auto()
    QUEUE_EMPTY = auto()
    SHUTDOWN = auto()


@dataclass
class Event:
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0  # set by event loop

    def __str__(self) -> str:
        extras = ", ".join(f"{k}={v}" for k, v in self.data.items())
        return f"Event({self.kind.name}{', ' + extras if extras else ''})"
