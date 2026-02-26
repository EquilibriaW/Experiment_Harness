"""Abstract base class for agent adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentResult:
    """Result of an agent invocation."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float = 0.0
    estimated_cost_usd: Optional[float] = None

    @property
    def output(self) -> str:
        """Combined output for logging."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        return "\n".join(parts)


class AgentAdapter(ABC):
    """ABC for wrapping a coding agent CLI."""

    @abstractmethod
    def run(self, prompt: str, working_dir: str) -> AgentResult:
        """Send a prompt to the agent and return the result.

        Args:
            prompt: The full prompt text.
            working_dir: Directory the agent should operate in.

        Returns:
            AgentResult with captured output.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this agent."""
