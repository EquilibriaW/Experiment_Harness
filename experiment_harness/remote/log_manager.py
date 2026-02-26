"""Manage reflection.md with hard word limit."""

from __future__ import annotations

import re
from pathlib import Path


class ReflectionManager:
    """Reads and writes reflection.md with a hard word limit.

    reflection.md is the ONLY persistent document across rounds.
    Every agent invocation starts fresh — this file is their sole memory.
    """

    def __init__(self, path: str | Path, word_limit: int = 8000) -> None:
        self.path = Path(path)
        self.word_limit = word_limit

    def read(self) -> str:
        if not self.path.exists():
            return ""
        text = self.path.read_text(encoding="utf-8")
        return self._truncate(text)

    def write(self, content: str) -> None:
        content = self._truncate(content)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content, encoding="utf-8")

    def word_count(self, text: str | None = None) -> int:
        if text is None:
            text = self.read()
        return len(text.split())

    def _truncate(self, text: str) -> str:
        """If text exceeds word_limit, keep header + latest sections, drop oldest."""
        words = text.split()
        if len(words) <= self.word_limit:
            return text

        sections = self._split_sections(text)

        if len(sections) <= 1:
            return " ".join(words[: self.word_limit])

        header = sections[0]
        header_words = len(header.split())

        remaining_budget = self.word_limit - header_words
        kept_sections: list[str] = []

        for section in reversed(sections[1:]):
            section_words = len(section.split())
            if section_words <= remaining_budget:
                kept_sections.insert(0, section)
                remaining_budget -= section_words
            else:
                break

        if not kept_sections:
            latest = sections[-1]
            latest_words = latest.split()
            kept_sections = [" ".join(latest_words[:remaining_budget])]

        truncation_notice = (
            "\n\n> [Earlier sections truncated to stay within "
            f"{self.word_limit}-word limit]\n\n"
        )

        dropped = len(sections) - 1 - len(kept_sections)
        if dropped > 0:
            return header + truncation_notice + "\n\n".join(kept_sections)
        else:
            return header + "\n\n".join(kept_sections)

    def _split_sections(self, text: str) -> list[str]:
        """Split text by ## headers, keeping each header with its content."""
        parts = re.split(r"(?=^## )", text, flags=re.MULTILINE)
        return [p for p in parts if p.strip()]
