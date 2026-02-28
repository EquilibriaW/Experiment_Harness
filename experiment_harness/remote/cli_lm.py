"""CLI-based LM adapter for dspy.

Routes LLM calls through the Codex CLI binary (subscription-based,
not per-token) instead of API calls via dspy.LM/litellm.

Usage:
    --sub-model cli:codex    # uses codex CLI
    --sub-model cli:claude   # uses claude CLI

The adapter runs the CLI in an isolated temp directory so sub-LM calls
cannot accidentally modify experiment files.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class CliLM:
    """dspy.LM-compatible wrapper that routes calls through a CLI binary.

    Satisfies the interface dspy.RLM expects for sub_lm:
    callable(prompt=..., messages=...) -> list[str]
    """

    def __init__(self, backend: str = "codex", timeout: int = 120) -> None:
        self.backend = backend
        self.model = f"cli:{backend}"
        self.history: list[dict] = []
        self.timeout = timeout
        # Isolated workdir prevents accidental file edits from agentic CLI
        self._workdir = tempfile.mkdtemp(prefix=f"{backend}_sub_")

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs,
    ) -> list[str]:
        if messages and not prompt:
            prompt = _messages_to_prompt(messages)
        if not prompt:
            return [""]

        if self.backend == "codex":
            completion = self._run_codex(prompt)
        elif self.backend == "claude":
            completion = self._run_claude(prompt)
        else:
            completion = f"(Unknown CLI backend: {self.backend})"

        self.history.append({
            "prompt": prompt[:300],
            "response": completion[:300],
        })
        return [completion]

    def _run_codex(self, prompt: str) -> str:
        """Run codex exec, passing prompt via stdin, capturing last message."""
        # Use -o to write the final agent message to a temp file
        out_file = Path(self._workdir) / "_last_message.txt"
        cmd = [
            "codex", "exec", "--yolo",
            "-o", str(out_file),
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True, text=True,
                timeout=self.timeout,
                cwd=self._workdir,
            )
            # Read the output file (contains just the agent's last message)
            if out_file.exists():
                completion = out_file.read_text(encoding="utf-8").strip()
                out_file.unlink(missing_ok=True)
                if completion:
                    return completion
            # Fallback: parse stdout
            completion = result.stdout.strip()
            if result.returncode != 0 and not completion:
                return (
                    f"(codex error, exit {result.returncode}: "
                    f"{result.stderr.strip()[:200]})"
                )
            return completion
        except subprocess.TimeoutExpired:
            return f"(codex CLI timed out after {self.timeout}s)"
        except FileNotFoundError:
            return "(codex CLI not found — is it installed?)"

    def _run_claude(self, prompt: str) -> str:
        """Run claude CLI, capturing printed output."""
        cmd = [
            "claude", "--print", "--dangerously-skip-permissions",
            "-p", prompt,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=self.timeout,
                cwd=self._workdir,
            )
            completion = result.stdout.strip()
            if result.returncode != 0 and not completion:
                return (
                    f"(claude error, exit {result.returncode}: "
                    f"{result.stderr.strip()[:200]})"
                )
            return completion
        except subprocess.TimeoutExpired:
            return f"(claude CLI timed out after {self.timeout}s)"
        except FileNotFoundError:
            return "(claude CLI not found — is it installed?)"

    def inspect_history(self, n: int = 1) -> list[dict]:
        """Recent call history (for dspy debugging)."""
        return self.history[-n:]


def make_cli_lm(spec: str, timeout: int = 120) -> CliLM:
    """Parse a 'cli:<backend>' spec into a CliLM instance."""
    if not spec.startswith("cli:"):
        raise ValueError(f"Expected 'cli:<backend>', got '{spec}'")
    backend = spec.split(":", 1)[1]
    return CliLM(backend=backend, timeout=timeout)


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten chat messages into a single prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)
