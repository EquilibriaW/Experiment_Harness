"""Claude Code CLI agent adapter."""

from __future__ import annotations

import json
import subprocess
import time

try:
    from .base import AgentAdapter, AgentResult
except ImportError:
    from base import AgentAdapter, AgentResult


class ClaudeAdapter(AgentAdapter):
    """Wraps Claude Code CLI with full system access.

    Uses --dangerously-skip-permissions to bypass all permission prompts,
    giving the agent unrestricted file read/write, shell exec, network,
    and GPU access. Intended for isolated pods only.
    """

    def __init__(self, max_turns: int = 30) -> None:
        self.max_turns = max_turns

    @property
    def name(self) -> str:
        return "claude"

    def run(self, prompt: str, working_dir: str) -> AgentResult:
        start = time.monotonic()
        cmd = [
            "claude",
            "-p", prompt,
            "--dangerously-skip-permissions",
            "--output-format", "json",
            "--max-turns", str(self.max_turns),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=working_dir,
            )
            duration = time.monotonic() - start

            cost = self._parse_cost(proc.stdout)

            return AgentResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_seconds=duration,
                estimated_cost_usd=cost,
            )
        except FileNotFoundError:
            duration = time.monotonic() - start
            return AgentResult(
                success=False,
                stdout="",
                stderr="claude CLI not found. Is it installed and on PATH?",
                exit_code=-1,
                duration_seconds=duration,
            )

    def _parse_cost(self, stdout: str) -> float | None:
        """Try to extract cost from claude JSON output."""
        try:
            data = json.loads(stdout)
            # Claude Code JSON output has cost_usd or similar fields
            for key in ("cost_usd", "cost", "total_cost"):
                if key in data:
                    return float(data[key])
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            pass
        return None
