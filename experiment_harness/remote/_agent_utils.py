"""Shared agent utilities for planner, executor, analyst."""

from __future__ import annotations

from pathlib import Path


def get_agent(agent_name: str, max_turns: int = 30):
    """Load and return the appropriate agent adapter."""
    if agent_name == "codex":
        from codex_adapter import CodexAdapter
        return CodexAdapter()
    elif agent_name == "claude":
        from claude_adapter import ClaudeAdapter
        return ClaudeAdapter(max_turns=max_turns)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def read_file(path: Path, max_chars: int = 50_000) -> str:
    """Read a file, truncating if necessary."""
    path = Path(path)
    if not path.exists():
        return "(file not found)"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated at {max_chars} chars)"
    return text
