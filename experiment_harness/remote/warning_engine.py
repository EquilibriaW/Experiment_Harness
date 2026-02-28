"""Warning engine: scaffold-level pattern detection.

Checks experiment state for conditions that warrant research agent attention.
Warnings are injected into the research agent's prompt, not handled by
the scheduling loop.

The distinction matters:
- Mechanical failures (OOM, missing file) → scheduling fixer (fast, no strategy)
- Strategic warnings (hypothesis falsified, no progress) → research agent (deliberate)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from experiment_state import ExperimentState
from budget_guard import BudgetGuard
from metrics_poller import MetricsPoller


@dataclass
class Warning:
    severity: str  # "critical" | "high" | "medium" | "low"
    category: str  # "hypothesis" | "progress" | "budget" | "anomaly" | "utilization"
    message: str
    evidence: list[str] | None = None


class WarningEngine:
    """Checks state and produces warnings for the research agent."""

    def __init__(
        self,
        state: ExperimentState,
        budget: BudgetGuard,
        metrics_poller: Optional[MetricsPoller] = None,
        stagnation_rounds: int = 3,
        hypothesis_failure_threshold: int = 3,
    ) -> None:
        self.state = state
        self.budget = budget
        self.metrics_poller = metrics_poller
        self.stagnation_rounds = stagnation_rounds
        self.hypothesis_failure_threshold = hypothesis_failure_threshold

        self._fired_once: set[str] = set()  # don't repeat one-shot warnings

    def check(self) -> list[Warning]:
        """Run all checks and return active warnings."""
        warnings: list[Warning] = []
        warnings.extend(self._check_budget())
        warnings.extend(self._check_stagnation())
        warnings.extend(self._check_hypotheses())
        warnings.extend(self._check_consecutive_failures())
        warnings.extend(self._check_anomalies())
        return warnings

    def format_for_prompt(self, warnings: list[Warning]) -> str:
        """Format warnings for injection into agent prompt."""
        if not warnings:
            return "(no warnings)"

        parts = []
        for w in sorted(warnings, key=lambda x: {"critical": 0, "high": 1,
                                                   "medium": 2, "low": 3}[x.severity]):
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}[w.severity]
            parts.append(f"{icon} [{w.severity.upper()}] {w.message}")
            if w.evidence:
                for e in w.evidence[:5]:
                    parts.append(f"   └ {e}")

        return "\n".join(parts)

    # ── Individual checks ───────────────────────────────────────

    def _check_budget(self) -> list[Warning]:
        warnings = []
        remaining = self.budget.remaining_hours
        fraction = remaining / max(self.budget.max_hours, 0.01)

        if fraction < 0.1 and "budget_critical" not in self._fired_once:
            self._fired_once.add("budget_critical")
            warnings.append(Warning(
                severity="critical",
                category="budget",
                message=(f"Only {remaining:.1f}h ({fraction*100:.0f}%) budget remaining. "
                         f"Wrap up NOW: reproduce best result, final ablations only."),
            ))
        elif fraction < 0.25 and "budget_low" not in self._fired_once:
            self._fired_once.add("budget_low")
            warnings.append(Warning(
                severity="high",
                category="budget",
                message=(f"{remaining:.1f}h ({fraction*100:.0f}%) remaining. "
                         f"Prioritize high-confidence experiments. "
                         f"No more exploratory runs."),
            ))
        elif fraction < 0.5 and "budget_half" not in self._fired_once:
            self._fired_once.add("budget_half")
            warnings.append(Warning(
                severity="medium",
                category="budget",
                message=f"Half budget used. {remaining:.1f}h remaining. Review strategy.",
            ))

        return warnings

    def _check_stagnation(self) -> list[Warning]:
        """No improvement in N consecutive completed runs."""
        warnings = []
        state = self.state.read()
        best = state.get("best_result")
        if not best:
            return warnings

        best_value = best.get("value")
        best_run_id = best.get("run_id")
        if best_value is None:
            return warnings

        # Count runs completed after the best run
        runs = state.get("runs", [])
        completed = [r for r in runs if r["status"] == "completed"]
        if len(completed) < self.stagnation_rounds + 1:
            return warnings

        # Find index of best run
        best_idx = None
        for i, r in enumerate(completed):
            if r["run_id"] == best_run_id:
                best_idx = i
                break

        if best_idx is None:
            return warnings

        runs_since_best = completed[best_idx + 1:]
        if len(runs_since_best) >= self.stagnation_rounds:
            evidence = []
            for r in runs_since_best[-self.stagnation_rounds:]:
                metric_key = best.get("metric", "loss")
                val = r.get("metrics", {}).get(metric_key)
                if val is not None:
                    evidence.append(
                        f"{r['run_id']}: {metric_key}={val:.4f} "
                        f"(best: {best_value:.4f})"
                    )

            if evidence:
                warnings.append(Warning(
                    severity="high",
                    category="progress",
                    message=(f"No improvement in {len(runs_since_best)} runs since "
                             f"best result ({best_run_id}). "
                             f"Consider a fundamentally different approach."),
                    evidence=evidence,
                ))

        return warnings

    def _check_hypotheses(self) -> list[Warning]:
        """Hypotheses that have been falsified repeatedly."""
        warnings = []
        state = self.state.read()

        for h in state.get("hypotheses", []):
            if h.get("status") != "active":
                continue

            # Count negative evidence
            negative = [e for e in h.get("evidence", [])
                        if any(word in e.lower() for word in
                               ["falsified", "rejected", "worse", "no improvement",
                                "failed", "disproved", "negative"])]

            if len(negative) >= self.hypothesis_failure_threshold:
                warnings.append(Warning(
                    severity="high",
                    category="hypothesis",
                    message=(f"Hypothesis '{h.get('description', h['id'])}' has "
                             f"{len(negative)} pieces of negative evidence. "
                             f"Consider abandoning this direction."),
                    evidence=negative[-3:],
                ))

        return warnings

    def _check_consecutive_failures(self) -> list[Warning]:
        """Many runs failing in a row."""
        warnings = []
        state = self.state.read()
        runs = state.get("runs", [])

        if len(runs) < 3:
            return warnings

        # Count consecutive failures from the end
        consecutive = 0
        for r in reversed(runs):
            if r["status"] == "failed":
                consecutive += 1
            elif r["status"] in ("completed", "running"):
                break

        if consecutive >= 3:
            recent_errors = []
            for r in reversed(runs):
                if r["status"] == "failed" and len(recent_errors) < 3:
                    error = r.get("error", "unknown")[:100]
                    recent_errors.append(f"{r['run_id']}: {error}")

            warnings.append(Warning(
                severity="high",
                category="progress",
                message=(f"{consecutive} consecutive failures. "
                         f"The approach may be fundamentally broken."),
                evidence=recent_errors,
            ))

        return warnings

    def _check_anomalies(self) -> list[Warning]:
        """Convert metrics poller anomalies into warnings."""
        warnings = []
        if not self.metrics_poller:
            return warnings

        for anomaly in self.metrics_poller.pop_anomalies():
            warnings.append(Warning(
                severity=anomaly.severity,
                category="anomaly",
                message=anomaly.message,
            ))

        return warnings
