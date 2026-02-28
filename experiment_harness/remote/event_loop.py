"""Event-driven experiment loop.

Two loops coupled by the experiment list:

SCHEDULING LOOP (this file):
  Fast, deterministic, no LLM (except bounded fixers).
  Pops from experiment list, launches on free GPUs, monitors processes,
  polls live metrics, checks warning conditions, decides when to wake
  the research agent.

RESEARCH LOOP (research_loop.py):
  Slow, LLM-driven, background thread.
  Invoked on specific triggers. Analyzes results, updates hypotheses,
  curates the experiment list, implements code changes.

The scheduling loop decides WHEN to invoke the research agent.
The research agent decides WHAT to do.
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from budget_guard import BudgetGuard
from event_types import Event, EventKind
from experiment_state import ExperimentState, ExperimentSpec
from resource_monitor import ResourceMonitor
from run_manager import RunManager
from metrics_poller import MetricsPoller
from warning_engine import WarningEngine
from research_agent import ResearchLoop
from executor import Executor


# ── Fixer prompt (scheduling-level, mechanical) ────────────────────────────

TRAIN_FIX_PROMPT = """\
You are a training debugger working in {experiment_dir}.

A training run failed. Diagnose and fix the issue.

## Run Info
- Run ID: {run_id}
- Command: {train_command}
- Exit code: {exit_code}

## Error (last 100 lines)
```
{log_tail}
```

## Instructions
1. Identify the root cause.
2. Fix the code or config. Common: OOM, missing module, shape mismatch, CUDA error.
3. Your fix must not break `pytest -x --tb=short -q`.
4. Do NOT restructure or "improve" unrelated code.
"""


class EventLoop:
    """Event-driven experiment controller.

    The scheduling loop is this class. The research loop is ResearchLoop,
    invoked from here as a background thread on specific triggers.
    """

    def __init__(
        self,
        spec_path: str,
        experiment_dir: str = "/workspace/experiment",
        agent_name: str = "codex",
        max_hours: float = 24.0,
        agent_max_turns: int = 30,
        train_command: Optional[str] = None,
        poll_interval: float = 15.0,
        metrics_poll_interval: float = 30.0,
        idle_threshold: float = 120.0,
        max_consecutive_failures: int = 3,
        research_heartbeat_minutes: float = 15.0,
        anomaly_thresholds: Optional[dict] = None,
        # RLM config for research agent
        model: Optional[str] = None,
        sub_model: Optional[str] = None,
        research_max_iterations: int = 20,
        research_max_llm_calls: int = 50,
        research_max_output_chars: int = 10_000,
    ) -> None:
        self.spec_path = Path(spec_path)
        self.experiment_dir = Path(experiment_dir)
        self.agent_name = agent_name
        self.train_command = train_command
        self.max_consecutive_failures = max_consecutive_failures
        self.research_heartbeat_seconds = research_heartbeat_minutes * 60

        self.events: queue.Queue[Event] = queue.Queue()
        self.budget = BudgetGuard(max_hours=max_hours)

        # State
        spec_text = self.spec_path.read_text(encoding="utf-8") if self.spec_path.exists() else ""
        state_path = self.experiment_dir / "experiment_state.json"
        self.state = ExperimentState(state_path, spec_text=spec_text)
        self.state.update_budget(max_hours=max_hours)

        # Subprocess management
        self.run_manager = RunManager(
            state=self.state, event_queue=self.events,
            experiment_dir=str(self.experiment_dir),
            default_train_command=train_command,
        )

        # GPU monitoring
        self.resource_monitor = ResourceMonitor(
            state=self.state, event_queue=self.events,
            poll_interval=poll_interval,
            idle_threshold_seconds=idle_threshold,
            experiment_dir=str(self.experiment_dir),
        )

        # Live metrics extraction (anomaly thresholds are human-set)
        self.metrics_poller = MetricsPoller(
            state=self.state,
            experiment_dir=self.experiment_dir,
            poll_interval=metrics_poll_interval,
            anomaly_thresholds=anomaly_thresholds,
        )

        # Warning detection
        self.warning_engine = WarningEngine(
            state=self.state,
            budget=self.budget,
            metrics_poller=self.metrics_poller,
        )

        # Research agent (OpenAI Agents SDK)
        self.research = ResearchLoop(
            state=self.state,
            experiment_dir=self.experiment_dir,
            spec_path=self.spec_path,
            warning_engine=self.warning_engine,
            model=model or "gpt-5.3-codex",
            sub_model=sub_model or "cli:codex",
            max_iterations=research_max_iterations,
            agent_name=agent_name,
        )

        # Executor for pre-launch code changes
        self.executor = Executor(
            state=self.state, experiment_dir=self.experiment_dir,
            spec_path=self.spec_path, agent_name=agent_name,
            agent_max_turns=agent_max_turns,
        )

        # Tracking
        self._research_running = False
        self._research_lock = threading.Lock()
        self._consecutive_failures = 0
        self._pending_triggers: list[str] = []
        self._logs_dir = self.experiment_dir / "logs"
        self._log_lock = threading.Lock()
        self._log_seq = 0

    # ── Main loop ───────────────────────────────────────────────

    def run(self) -> None:
        print(f"{'='*60}")
        print(f"  Experiment Harness v2")
        print(f"{'='*60}")
        print(f"  Agent:          {self.agent_name}")
        print(f"  Spec:           {self.spec_path}")
        print(f"  Dir:            {self.experiment_dir}")
        print(f"  Train command:  {self.train_command or '(auto-detect)'}")
        print(f"  Budget:         {self.budget.max_hours}h")
        print(f"{'='*60}\n")

        self._setup()
        self.resource_monitor.start()
        self.metrics_poller.start()
        self._start_budget_ticker()
        self._start_research_heartbeat()
        self.events.put(Event(kind=EventKind.INIT))

        try:
            while True:
                stop = self.budget.check()
                if stop:
                    print(f"\n=== STOPPING: {stop} ===")
                    break
                try:
                    event = self.events.get(timeout=30)
                except queue.Empty:
                    continue
                event.timestamp = time.time()
                self._handle_event(event)
                if event.kind == EventKind.SHUTDOWN:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.resource_monitor.stop()
            self.metrics_poller.stop()
            self._finalize()

    def _handle_event(self, event: Event) -> None:
        self.state.update_budget(elapsed_hours=self.budget.elapsed_hours)
        print(f"\n[Event] {event}")

        if event.kind == EventKind.INIT:
            self._on_init()
        elif event.kind == EventKind.GPU_FREED:
            self._schedule()
        elif event.kind == EventKind.RUN_COMPLETED:
            self._consecutive_failures = 0
            self._on_run_completed(event)
        elif event.kind == EventKind.RUN_FAILED:
            self._consecutive_failures += 1
            self._on_run_failed(event)
        elif event.kind == EventKind.BUDGET_TICK:
            self._on_budget_tick()
        elif event.kind == EventKind.IDLE_ALERT:
            self._on_idle_alert(event)
        elif event.kind == EventKind.QUEUE_EMPTY:
            self._schedule()

    # ── Event handlers ──────────────────────────────────────────

    def _on_init(self) -> None:
        snap = self.resource_monitor.snapshot()
        self.state.update_resources(snap)
        print(f"[Init] GPUs: {snap.gpu_count} total, {snap.gpus_free} free")
        print(f"[Init] Disk: {snap.disk_free_gb:.1f} GB free")

        if self.state.list_size() > 0:
            print(f"[Init] Existing experiment list: {self.state.list_size()} items")
            self._schedule()
        else:
            # Cold start: research agent must provide initial list (blocking)
            print("[Init] Cold start — invoking research agent (blocking)...")
            self._invoke_research_sync("Cold start: no experiments yet. "
                                       "Analyze the spec and create an initial "
                                       "experiment plan.")
            self._schedule()

    def _on_run_completed(self, event: Event) -> None:
        """Run completed. Schedule first (fast), then wake research agent."""
        run_id = event.data.get("run_id", "")
        metrics = event.data.get("metrics", {})
        print(f"[Complete] {run_id}: {metrics}")
        self._log_event("run_completed", {
            "run_id": run_id, "metrics": metrics,
            "exit_code": event.data.get("exit_code"),
        })

        # Fill GPUs immediately
        self._schedule()

        # Wake research agent to analyze results
        self._trigger_research(
            f"Run {run_id} completed with metrics: "
            f"{json.dumps(metrics, default=str)}"
        )

    def _on_run_failed(self, event: Event) -> None:
        """Run failed. Mechanical fix first, then schedule, then wake research."""
        run_id = event.data.get("run_id", "")
        error = event.data.get("error", "")[:200]
        self._log_event("run_failed", {
            "run_id": run_id, "error": error,
            "exit_code": event.data.get("exit_code"),
        })

        if self._consecutive_failures <= self.max_consecutive_failures:
            print(f"[Fail] {run_id} failed, invoking fixer...")
            self._invoke_train_fixer(run_id)
        else:
            print(f"[Fail] {self._consecutive_failures} consecutive failures, "
                  f"skipping mechanical fix")

        self._schedule()

        # Wake research agent — failures may require strategic rethinking
        self._trigger_research(
            f"Run {run_id} failed: {error}. "
            f"({self._consecutive_failures} consecutive failures)"
        )

    def _on_budget_tick(self) -> None:
        status = self.budget.status()
        running = self.run_manager.running_count()
        list_size = self.state.list_size()
        print(f"[Budget] {status} | {running} running | {list_size} in list")

        # Check warnings (may trigger research agent)
        warnings = self.warning_engine.check()
        if warnings:
            warning_msgs = [w.message for w in warnings]
            print(f"[Budget] {len(warnings)} active warnings")
            self._trigger_research(
                f"Budget tick with warnings: {'; '.join(warning_msgs[:3])}"
            )

    def _on_idle_alert(self, event: Event) -> None:
        gpu_id = event.data.get("gpu_id", "?")
        idle_s = event.data.get("idle_seconds", 0)
        print(f"[Idle] GPU {gpu_id} idle for {idle_s:.0f}s")

        self._schedule()

        # If still idle after scheduling, research agent may need to provide more work
        snap = self.resource_monitor.snapshot()
        if snap.gpus_free > 0 and self.state.list_size() == 0:
            self._trigger_research(
                f"GPU {gpu_id} idle for {idle_s:.0f}s and experiment list is empty. "
                f"Provide more experiments."
            )

    # ── The scheduler (trivial, never calls LLM) ───────────────

    def _schedule(self) -> None:
        """Pop front of experiment list, launch on free GPUs. No LLM."""
        snap = self.resource_monitor.snapshot()
        busy_ids = set(snap.gpus_busy)
        all_gpu_ids = [g.id for g in snap.gpus]
        free_ids = [g for g in all_gpu_ids if g not in busy_ids]

        if not free_ids:
            return

        launched = 0
        while free_ids and self.state.list_size() > 0:
            spec = self.state.pop_front(gpu_available=len(free_ids))
            if not spec:
                break

            gpu_req = spec.get("gpu_requirement", 1)
            assigned = free_ids[:gpu_req]

            # Pre-launch code changes if needed
            if spec.get("requires_code_change"):
                print(f"[Schedule] {spec.get('spec_id')} needs code change")
                success = self.executor.execute(spec)
                if not success:
                    print(f"[Schedule] Executor failed, skipping")
                    continue

            run_id = self.run_manager.launch(spec=spec, gpu_ids=assigned)
            self._log_event("launch", {
                "run_id": run_id, "spec_id": spec.get("spec_id"),
                "gpu_ids": assigned,
            })
            for g in assigned:
                free_ids.remove(g)
            launched += 1

        if launched > 0:
            snap2 = self.resource_monitor.snapshot()
            print(f"[Schedule] Launched {launched}, {snap2.gpus_free} GPUs now free")

        # Queue stranded: GPUs free but no spec fits
        if launched == 0 and free_ids and self.state.list_size() > 0:
            n_free = len(free_ids)
            queue = self.state.read().get("experiment_list", [])
            min_gpu_req = min((s.get("gpu_requirement", 1) for s in queue), default=1)
            needs_code = sum(1 for s in queue if s.get("requires_code_change"))
            self._trigger_research(
                f"QUEUE STRANDED: {n_free} GPU(s) free but no spec fits. "
                f"Queue has {len(queue)} items (min GPU req: {min_gpu_req}, "
                f"{needs_code} need code changes). "
                f"Provide experiments that can run on {n_free} GPU(s)."
            )

        # Deadlock prevention: nothing running, nothing queued → stall
        if (self.state.list_size() == 0
                and self.run_manager.running_count() == 0):
            self._trigger_research(
                "DEADLOCK: no experiments running or queued. "
                "You must provide experiments or the system stalls."
            )

    # ── Research agent invocation ──────────────────────────────

    def _trigger_research(self, reason: str) -> None:
        """Queue a research agent invocation (background thread).

        Multiple triggers may accumulate — they're batched into a single
        invocation when the agent is free.
        """
        with self._research_lock:
            self._pending_triggers.append(reason)
        self._maybe_invoke_research_async()

    def _maybe_invoke_research_async(self) -> None:
        """Start research agent in background if not already running."""
        with self._research_lock:
            if self._research_running:
                return  # already running, triggers will be picked up next time
            if not self._pending_triggers:
                return
            self._research_running = True
            # Consume all pending triggers into one invocation (inside lock)
            triggers = list(self._pending_triggers)
            self._pending_triggers.clear()
        trigger_reason = "; ".join(triggers)

        def _run():
            try:
                self._run_research(trigger_reason)
            except Exception as e:
                print(f"[Research/bg] Error: {e}")
            finally:
                with self._research_lock:
                    self._research_running = False
                # If triggers accumulated while we were running, go again
                if self._pending_triggers:
                    self._maybe_invoke_research_async()

        t = threading.Thread(target=_run, daemon=True, name="research-agent")
        t.start()

    def _invoke_research_sync(self, reason: str) -> None:
        """Blocking research invocation. Used only on cold start."""
        self._run_research(reason)

    def _run_research(self, trigger_reason: str) -> None:
        """Execute one research agent invocation and apply output."""
        snap = self.resource_monitor.snapshot()

        # Refresh live metrics before invocation
        self.metrics_poller.poll_once()

        self._log_event("research_start", {"trigger": trigger_reason})

        output = self.research.invoke(
            trigger_reason=trigger_reason,
            budget_status=self.budget.status(),
            gpus_free=snap.gpus_free,
            gpu_count=max(snap.gpu_count, 1),
            budget_low=self.budget.budget_low,
        )

        # Apply output atomically
        self._apply_research_output(output)

        self._log_event("research_complete", {
            "n_specs": len(output.specs),
            "n_kills": len(output.kill_runs),
            "reflection": output.reflection[:300] if output.reflection else "",
        })

        # If GPUs are free and we have new experiments, trigger scheduling
        snap2 = self.resource_monitor.snapshot()
        if snap2.gpus_free > 0 and self.state.list_size() > 0:
            self.events.put(Event(kind=EventKind.GPU_FREED,
                                  data={"gpus_free": snap2.gpus_free}))

    def _apply_research_output(self, output) -> None:
        """Apply research agent's decisions.

        The agent writes decisions via SUBMIT() in the dspy.RLM REPL.
        We just need to handle kill requests and log what changed.
        """
        # Kill requested runs
        for run_id in output.kill_runs:
            running = self.state.get_runs(status="running")
            if any(r["run_id"] == run_id for r in running):
                self.run_manager.kill(run_id, reason="research agent requested kill")
                print(f"[Research] Killed {run_id}")

        # Log what changed (state already written by agent)
        if output.specs:
            print(f"[Research] Experiment list updated ({len(output.specs)} items):")
            for i, s in enumerate(output.specs[:5]):
                print(f"  {i+1}. {s.description[:70]}")
            if len(output.specs) > 5:
                print(f"  ... +{len(output.specs) - 5} more")
        elif output.list_unchanged:
            print(f"[Research] Experiment list unchanged")
        else:
            print(f"[Research] WARNING: Experiment list may be empty")

        if output.reflection:
            print(f"[Research] Reflection updated ({len(output.reflection)} chars)")

        if output.hypotheses:
            active = sum(1 for h in output.hypotheses if h.status == "active")
            print(f"[Research] {len(output.hypotheses)} hypotheses "
                  f"({active} active)")

        self.budget.record_round()

    # ── Mechanical fixer (scheduling-level, no strategy) ───────

    def _invoke_train_fixer(self, run_id: str) -> None:
        from _agent_utils import get_agent

        run = next((r for r in self.state.get_runs() if r["run_id"] == run_id), None)
        if not run:
            return

        log_tail = self.run_manager.get_log_tail(run_id, n_lines=100)
        train_cmd = run.get("train_command", self.train_command or "python train.py")

        prompt = TRAIN_FIX_PROMPT.format(
            experiment_dir=self.experiment_dir,
            run_id=run_id,
            train_command=train_cmd,
            exit_code=run.get("exit_code", -1),
            log_tail=log_tail,
        )

        agent = get_agent(self.agent_name, max_turns=15)
        print(f"  [Fixer] Invoking {self.agent_name}...")
        start = time.time()
        result = agent.run(prompt=prompt, working_dir=str(self.experiment_dir))
        duration = time.time() - start
        print(f"  [Fixer] Done in {duration:.0f}s (exit={result.exit_code})")
        self.state.record_agent_call(cost_usd=result.estimated_cost_usd or 0.0)
        self._log_event("train_fix", {"run_id": run_id, "duration_s": duration})

    # ── Background tickers ─────────────────────────────────────

    def _start_budget_ticker(self, interval: float = 300.0) -> None:
        def tick():
            while not self.budget.budget_exhausted:
                time.sleep(interval)
                self.events.put(Event(kind=EventKind.BUDGET_TICK))
        threading.Thread(target=tick, daemon=True, name="budget-ticker").start()

    def _start_research_heartbeat(self) -> None:
        """Periodically wake the research agent even if no events fired.

        This handles the case where long-running experiments produce
        interesting live metrics but haven't finished yet.
        """
        interval = self.research_heartbeat_seconds
        if interval <= 0:
            return

        def heartbeat():
            while not self.budget.budget_exhausted:
                time.sleep(interval)
                # Only trigger if the research agent hasn't been invoked recently
                since_last = time.time() - self.research.last_invocation_time
                if since_last > interval * 0.8:
                    running = self.run_manager.running_count()
                    if running > 0:
                        self._trigger_research(
                            f"Heartbeat: {running} experiments running, "
                            f"checking live metrics and progress"
                        )

        threading.Thread(target=heartbeat, daemon=True, name="research-heartbeat").start()

    # ── Setup / teardown ────────────────────────────────────────

    def _setup(self) -> None:
        self._archive_previous_run()
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "run_logs").mkdir(parents=True, exist_ok=True)

    def _archive_previous_run(self) -> None:
        if not self._logs_dir.exists():
            return
        log_files = list(self._logs_dir.glob("*.json"))
        if not log_files:
            return
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        archive = self.experiment_dir / "traces" / f"run_{ts}"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self._logs_dir), str(archive / "logs"))
        state_path = self.experiment_dir / "experiment_state.json"
        if state_path.exists():
            shutil.copy2(str(state_path), str(archive / "experiment_state.json"))

    def _finalize(self) -> None:
        for r in self.state.get_runs(status="running"):
            self.run_manager.kill(r["run_id"], reason="harness shutdown")
        self.state.update_budget(elapsed_hours=self.budget.elapsed_hours)

        state = self.state.read()
        completed = [r for r in state.get("runs", []) if r["status"] == "completed"]
        failed = [r for r in state.get("runs", []) if r["status"] == "failed"]
        best = state.get("best_result")

        print(f"\n{'='*60}")
        print(f"  Experiment Complete")
        print(f"  {self.budget.status()}")
        print(f"  Runs: {len(completed)} completed, {len(failed)} failed")
        if best:
            print(f"  Best: {best['metric']}={best['value']:.4f} ({best['run_id']})")
        print(f"  State: {self.experiment_dir / 'experiment_state.json'}")
        print(f"{'='*60}")

    def _log_event(self, event_type: str, data: dict) -> None:
        with self._log_lock:
            self._logs_dir.mkdir(parents=True, exist_ok=True)
            self._log_seq += 1
            seq = self._log_seq
        entry = {
            "seq": seq, "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "budget": self.budget.status(), **data,
        }
        path = self._logs_dir / f"{seq:04d}_{event_type}.json"
        path.write_text(json.dumps(entry, indent=2, default=str), encoding="utf-8")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Experiment harness v2")
    p.add_argument("--spec", required=True)
    p.add_argument("--experiment-dir", default="/workspace/experiment")
    p.add_argument("--agent", default="codex", choices=["codex", "claude"])
    p.add_argument("--max-hours", type=float, default=24.0)
    p.add_argument("--agent-max-turns", type=int, default=30)
    p.add_argument("--train-command", default=None)
    p.add_argument("--poll-interval", type=float, default=15.0)
    p.add_argument("--metrics-poll-interval", type=float, default=30.0)
    p.add_argument("--idle-threshold", type=float, default=120.0)
    p.add_argument("--max-consecutive-failures", type=int, default=3)
    p.add_argument("--research-heartbeat-minutes", type=float, default=15.0)
    p.add_argument("--diverge-ratio", type=float, default=None,
                   help="Loss divergence threshold (e.g., 2.0 = loss > 2x min)")
    p.add_argument("--plateau-steps", type=int, default=None,
                   help="Steps with <threshold improvement = plateau")
    p.add_argument("--plateau-min-improvement", type=float, default=0.01,
                   help="Minimum improvement fraction (default 0.01 = 1%%)")
    # RLM config
    p.add_argument("--model", default=None,
                   help="Root LM for research agent (e.g., openai/gpt-4o)")
    p.add_argument("--sub-model", default=None,
                   help="Sub-LM for recursive calls (e.g., openai/gpt-4o-mini)")
    p.add_argument("--research-max-iterations", type=int, default=20)
    p.add_argument("--research-max-llm-calls", type=int, default=50)
    p.add_argument("--research-max-output-chars", type=int, default=10000,
                   help="REPL output truncation (forces symbolic recursion)")
    args = p.parse_args()

    anomaly_thresholds = {}
    if args.diverge_ratio is not None:
        anomaly_thresholds["diverge_ratio"] = args.diverge_ratio
    if args.plateau_steps is not None:
        anomaly_thresholds["plateau_steps"] = args.plateau_steps
        anomaly_thresholds["plateau_min_improvement"] = args.plateau_min_improvement

    EventLoop(
        spec_path=args.spec,
        experiment_dir=args.experiment_dir,
        agent_name=args.agent,
        max_hours=args.max_hours,
        agent_max_turns=args.agent_max_turns,
        train_command=args.train_command,
        poll_interval=args.poll_interval,
        metrics_poll_interval=args.metrics_poll_interval,
        idle_threshold=args.idle_threshold,
        max_consecutive_failures=args.max_consecutive_failures,
        research_heartbeat_minutes=args.research_heartbeat_minutes,
        anomaly_thresholds=anomaly_thresholds or None,
        model=args.model,
        sub_model=args.sub_model,
        research_max_iterations=args.research_max_iterations,
        research_max_llm_calls=args.research_max_llm_calls,
        research_max_output_chars=args.research_max_output_chars,
    ).run()


if __name__ == "__main__":
    main()
