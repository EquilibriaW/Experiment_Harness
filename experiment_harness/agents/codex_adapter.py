"""Codex CLI agent adapter."""

from __future__ import annotations

import json
import subprocess
import time

try:
    from .base import AgentAdapter, AgentResult
except ImportError:
    from base import AgentAdapter, AgentResult


class CodexAdapter(AgentAdapter):
    """Wraps Codex CLI with full system access.

    Uses --yolo (alias for --dangerously-bypass-approvals-and-sandbox)
    to give the agent unrestricted read/write/exec. This is the correct
    flag for headless automation on an isolated pod — --full-auto only
    grants workspace-write sandbox which blocks package installs, GPU
    access in some paths, etc.
    """

    @property
    def name(self) -> str:
        return "codex"

    def run(self, prompt: str, working_dir: str) -> AgentResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["codex", "exec", "--yolo", "--json", "-"],
                input=prompt,
                capture_output=True,
                text=True,
                cwd=working_dir,
            )
            duration = time.monotonic() - start

            # Try to extract cost from JSON output
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
                stderr="codex CLI not found. Is it installed and on PATH?",
                exit_code=-1,
                duration_seconds=duration,
            )

    def _parse_cost(self, stdout: str) -> float | None:
        """Try to extract estimated cost from codex JSON output."""
        try:
            for line in stdout.strip().splitlines():
                data = json.loads(line)
                if "cost" in data:
                    return float(data["cost"])
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return None
