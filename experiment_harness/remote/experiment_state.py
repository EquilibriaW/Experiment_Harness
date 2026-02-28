"""Structured experiment state — the single source of truth.

This replaces reflection.md as the system's memory. All state is stored
as JSON and is programmatically readable by both the scaffold and agents.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from threading import Lock


# ── Data models ─────────────────────────────────────────────────────────────

@dataclass
class GpuInfo:
    id: int
    name: str = ""
    util_pct: float = 0.0
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    temp_c: int = 0


@dataclass
class ResourceSnapshot:
    gpus: list[GpuInfo] = field(default_factory=list)
    gpu_count: int = 0
    gpus_free: int = 0
    gpus_busy: list[int] = field(default_factory=list)  # GPU IDs in use by runs
    disk_free_gb: float = 0.0


@dataclass
class RunRecord:
    """A single training run."""
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | running | completed | failed | killed
    gpu_ids: list[int] = field(default_factory=list)
    hypothesis: str = ""
    predicted_outcome: str = ""
    kill_criteria: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    live_metrics: dict[str, Any] = field(default_factory=dict)
    metrics_history: list[dict] = field(default_factory=list)
    error: str = ""
    log_path: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    pid: int = 0
    train_command: str = ""

    @property
    def duration_minutes(self) -> float:
        end = self.end_time or time.time()
        if self.start_time <= 0:
            return 0.0
        return (end - self.start_time) / 60.0


@dataclass
class ExperimentSpec:
    """A proposed experiment in the queue."""
    spec_id: str
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    predicted_outcome: str = ""
    uncertainty: str = "high"  # high | medium | low
    gpu_requirement: int = 1
    kill_criteria: str = ""
    priority: int = 0  # lower = higher priority
    requires_code_change: bool = False
    code_change_description: str = ""
    train_command: str = ""  # override default train command


@dataclass
class Hypothesis:
    id: str
    description: str
    status: str = "active"  # active | confirmed | rejected | paused
    evidence: list[str] = field(default_factory=list)


# ── State manager ───────────────────────────────────────────────────────────

class ExperimentState:
    """Thread-safe structured state backed by a JSON file.

    This is the source of truth for the entire experiment. Agents receive
    summaries of this state; the scaffold reads/writes it programmatically.
    """

    def __init__(self, state_path: Path, spec_text: str = "") -> None:
        self.path = state_path
        self._lock = Lock()

        if self.path.exists():
            self._load()
        else:
            self._state: dict[str, Any] = {
                "goal": spec_text[:2000] if spec_text else "",
                "resources": asdict(ResourceSnapshot()),
                "runs": [],
                "experiment_list": [],
                "hypotheses": [],
                "best_result": None,
                "budget": {
                    "elapsed_hours": 0.0,
                    "max_hours": 24.0,
                    "total_gpu_hours": 0.0,
                    "total_agent_calls": 0,
                    "total_agent_cost_usd": 0.0,
                },
                "summary": "",  # short human-readable summary (replaces reflection.md prose)
                "round": 0,
            }
            self._save()

    # ── Thread-safe read/write ──────────────────────────────────

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        self._state = json.loads(text)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._state, indent=2, default=str),
            encoding="utf-8",
        )

    def read(self) -> dict[str, Any]:
        with self._lock:
            self._load()
            return json.loads(json.dumps(self._state, default=str))

    def update(self, fn) -> dict[str, Any]:
        """Apply a mutation function to the state, save, and return it."""
        with self._lock:
            self._load()
            fn(self._state)
            self._save()
            return json.loads(json.dumps(self._state, default=str))

    # ── Convenience accessors ───────────────────────────────────

    def get_runs(self, status: Optional[str] = None) -> list[dict]:
        state = self.read()
        runs = state.get("runs", [])
        if status:
            runs = [r for r in runs if r.get("status") == status]
        return runs

    def get_resources(self) -> dict:
        return self.read().get("resources", {})

    # ── Run management ──────────────────────────────────────────

    def add_run(self, run: RunRecord) -> None:
        def _add(s):
            s["runs"].append(asdict(run))
        self.update(_add)

    def update_run(self, run_id: str, **kwargs) -> None:
        def _update(s):
            for r in s["runs"]:
                if r["run_id"] == run_id:
                    r.update(kwargs)
                    break
        self.update(_update)

    def complete_run(self, run_id: str, metrics: dict, status: str = "completed") -> None:
        def _complete(s):
            for r in s["runs"]:
                if r["run_id"] == run_id:
                    r["status"] = status
                    r["end_time"] = time.time()
                    r["metrics"] = metrics
                    break
            # Update best result
            best = s.get("best_result")
            if metrics and status == "completed":
                # Use "val_loss" or first metric as comparison (lower is better)
                key = "val_loss" if "val_loss" in metrics else next(iter(metrics), None)
                if key:
                    val = metrics[key]
                    if best is None or val < best.get("value", float("inf")):
                        s["best_result"] = {
                            "run_id": run_id,
                            "metric": key,
                            "value": val,
                        }
        self.update(_complete)

    # ── Experiment list management ─────────────────────────────
    #
    # The experiment list is an ordered list. Position = priority.
    # Front of list = next to run. The planner/analyst own this list
    # and can replace it wholesale, insert, remove, or reorder.
    # The scheduler only ever pops from the front.

    def set_experiment_list(self, specs: list[ExperimentSpec]) -> None:
        """Replace the entire experiment list (planner's main action)."""
        def _set(s):
            s["experiment_list"] = [asdict(sp) for sp in specs]
        self.update(_set)

    def set_experiment_list_raw(self, specs: list[dict]) -> None:
        """Replace experiment list from raw dicts (for mixed sources)."""
        def _set(s):
            s["experiment_list"] = list(specs)
        self.update(_set)

    def get_experiment_list(self) -> list[dict]:
        """Return the current experiment list (ordered, front = next)."""
        return self.read().get("experiment_list", [])

    def pop_front(self, gpu_available: int = 1) -> Optional[dict]:
        """Pop the first experiment that fits in available GPUs."""
        result = None
        def _pop(s):
            nonlocal result
            lst = s.get("experiment_list", [])
            for i, spec in enumerate(lst):
                if spec.get("gpu_requirement", 1) <= gpu_available:
                    result = lst.pop(i)
                    s["experiment_list"] = lst
                    return
        self.update(_pop)
        return result

    def list_size(self) -> int:
        return len(self.get_experiment_list())

    def insert_experiment(self, spec: ExperimentSpec, position: int = 0) -> None:
        """Insert an experiment at a given position (0 = front = next to run)."""
        def _insert(s):
            lst = s.get("experiment_list", [])
            lst.insert(position, asdict(spec))
            s["experiment_list"] = lst
        self.update(_insert)

    def remove_experiment(self, spec_id: str) -> None:
        """Remove an experiment from the list by spec_id."""
        def _remove(s):
            s["experiment_list"] = [
                x for x in s.get("experiment_list", [])
                if x.get("spec_id") != spec_id
            ]
        self.update(_remove)

    # ── Resource updates ────────────────────────────────────────

    def update_resources(self, snapshot: ResourceSnapshot) -> None:
        def _update(s):
            s["resources"] = asdict(snapshot)
        self.update(_update)

    # ── Budget ──────────────────────────────────────────────────

    def update_budget(self, **kwargs) -> None:
        def _update(s):
            s["budget"].update(kwargs)
        self.update(_update)

    def record_agent_call(self, cost_usd: float = 0.0) -> None:
        def _record(s):
            s["budget"]["total_agent_calls"] += 1
            s["budget"]["total_agent_cost_usd"] += cost_usd
        self.update(_record)

    def increment_round(self) -> int:
        result = 0
        def _inc(s):
            nonlocal result
            s["round"] = s.get("round", 0) + 1
            result = s["round"]
        self.update(_inc)
        return result

    # ── Hypotheses ──────────────────────────────────────────────

    def update_hypotheses(self, hypotheses: list[Hypothesis]) -> None:
        def _update(s):
            s["hypotheses"] = [asdict(h) for h in hypotheses]
        self.update(_update)

    # ── Summary ─────────────────────────────────────────────────

    def update_summary(self, summary: str) -> None:
        def _update(s):
            s["summary"] = summary[:5000]
        self.update(_update)

    # ── Agent-facing state summary ──────────────────────────────

    def agent_summary(self, include_list: bool = True) -> str:
        """Compact structured summary for agent prompts.

        This is the handle-based approach: agents get a summary and can
        request details via tools, rather than getting everything pasted in.
        """
        state = self.read()
        resources = state.get("resources", {})
        budget = state.get("budget", {})
        runs = state.get("runs", [])
        experiment_list = state.get("experiment_list", [])
        hypotheses = state.get("hypotheses", [])
        best = state.get("best_result")

        completed_runs = [r for r in runs if r["status"] == "completed"]
        running_runs = [r for r in runs if r["status"] == "running"]
        failed_runs = [r for r in runs if r["status"] == "failed"]

        goal = state.get("goal", "")

        parts = []

        if goal:
            parts.append("## Research Goal")
            parts.append(goal)

        parts.append("\n## Resource Snapshot")
        parts.append(json.dumps(resources, indent=2))

        parts.append("\n## Budget")
        parts.append(json.dumps(budget, indent=2))

        if best:
            parts.append(f"\n## Best Result So Far")
            parts.append(json.dumps(best, indent=2))

        if running_runs:
            parts.append(f"\n## Currently Running ({len(running_runs)} runs)")
            for r in running_runs:
                parts.append(f"- {r['run_id']}: GPUs {r.get('gpu_ids', [])}, "
                           f"running {r.get('duration_minutes', 0):.0f}min"
                           f" — {r.get('hypothesis', 'no hypothesis')}")

        if completed_runs:
            parts.append(f"\n## Completed Runs ({len(completed_runs)})")
            for r in completed_runs[-10:]:  # last 10
                metrics = r.get("metrics", {})
                parts.append(f"- {r['run_id']}: {json.dumps(metrics) if metrics else 'no metrics'}"
                           f" — {r.get('hypothesis', '')}")

        if failed_runs:
            parts.append(f"\n## Failed Runs ({len(failed_runs)})")
            for r in failed_runs[-5:]:  # last 5
                parts.append(f"- {r['run_id']}: {r.get('error', 'unknown error')[:200]}")

        if hypotheses:
            parts.append(f"\n## Hypotheses")
            for h in hypotheses:
                parts.append(f"- [{h['status']}] {h['id']}: {h['description']}")

        if include_list and experiment_list:
            parts.append(f"\n## Experiment List ({len(experiment_list)} items, front = next to run)")
            for i, spec in enumerate(experiment_list):
                marker = "→" if i == 0 else " "
                parts.append(f"  {marker} {i+1}. {spec.get('description', spec.get('spec_id', '?'))}")

        summary = state.get("summary", "")
        if summary:
            parts.append(f"\n## Current Summary\n{summary}")

        return "\n".join(parts)
