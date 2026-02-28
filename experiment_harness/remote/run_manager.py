"""Run manager: launches and monitors training as background subprocesses.

Training runs are OS processes, not agent invocations. This keeps training
running independently while agents plan/analyze. If a run crashes, the
failure is recorded and surfaced to the analyst — no LLM babysitter needed.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from event_types import Event, EventKind
from experiment_state import ExperimentState, RunRecord


class RunManager:
    """Manages training subprocesses and monitors their completion."""

    def __init__(
        self,
        state: ExperimentState,
        event_queue,  # queue.Queue[Event]
        experiment_dir: str = "/workspace/experiment",
        default_train_command: Optional[str] = None,
    ) -> None:
        self.state = state
        self.event_queue = event_queue
        self.experiment_dir = Path(experiment_dir)
        self.default_train_command = default_train_command

        self._processes: dict[str, subprocess.Popen] = {}
        self._watchers: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._run_counter = 0

    def next_run_id(self) -> str:
        self._run_counter += 1
        return f"run_{self._run_counter:03d}"

    def launch(
        self,
        spec: dict,
        gpu_ids: list[int],
        run_id: Optional[str] = None,
    ) -> str:
        """Launch a training run as a background subprocess.

        Args:
            spec: Experiment spec dict (from queue).
            gpu_ids: GPU IDs to assign via CUDA_VISIBLE_DEVICES.
            run_id: Optional explicit run ID.

        Returns:
            The run_id.
        """
        run_id = run_id or self.next_run_id()

        # Determine train command
        train_cmd = spec.get("train_command", "") or self.default_train_command
        if not train_cmd:
            train_cmd = self._detect_train_command(spec)

        if not train_cmd:
            # Record failure immediately
            record = RunRecord(
                run_id=run_id,
                config=spec.get("config", {}),
                status="failed",
                gpu_ids=gpu_ids,
                hypothesis=spec.get("hypothesis", ""),
                error="No train command found. Set --train-command or ensure train.py exists.",
                start_time=time.time(),
                end_time=time.time(),
            )
            self.state.add_run(record)
            self.event_queue.put(Event(
                kind=EventKind.RUN_FAILED,
                data={"run_id": run_id, "error": record.error},
            ))
            return run_id

        # Write config to file for the training script to read
        config = spec.get("config", {})
        config_path = self.experiment_dir / f".run_configs/{run_id}_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        env["RUN_ID"] = run_id
        env["RUN_CONFIG"] = str(config_path)

        # Log file
        log_dir = self.experiment_dir / "run_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.log"

        # Record run
        record = RunRecord(
            run_id=run_id,
            config=config,
            status="running",
            gpu_ids=gpu_ids,
            hypothesis=spec.get("hypothesis", ""),
            predicted_outcome=spec.get("predicted_outcome", ""),
            kill_criteria=spec.get("kill_criteria", ""),
            log_path=str(log_path),
            start_time=time.time(),
            train_command=train_cmd,
        )
        self.state.add_run(record)

        # Launch subprocess
        try:
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                train_cmd,
                shell=True,
                cwd=str(self.experiment_dir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # new process group for clean kill
            )

            with self._lock:
                self._processes[run_id] = proc

            self.state.update_run(run_id, pid=proc.pid)
            print(f"  [RunManager] Launched {run_id}: PID={proc.pid}, "
                  f"GPUs={gpu_ids}, cmd={train_cmd[:80]}")

            # Start watcher thread
            watcher = threading.Thread(
                target=self._watch_process,
                args=(run_id, proc, log_file, log_path),
                daemon=True,
                name=f"watcher-{run_id}",
            )
            self._watchers[run_id] = watcher
            watcher.start()

        except Exception as e:
            error_msg = f"Failed to launch: {e}"
            self.state.update_run(run_id, status="failed", error=error_msg, end_time=time.time())
            self.event_queue.put(Event(
                kind=EventKind.RUN_FAILED,
                data={"run_id": run_id, "error": error_msg},
            ))

        return run_id

    def kill(self, run_id: str, reason: str = "killed by harness") -> None:
        """Kill a running training process."""
        with self._lock:
            proc = self._processes.get(run_id)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            print(f"  [RunManager] Killed {run_id}: {reason}")

        self.state.update_run(run_id, status="killed", error=reason, end_time=time.time())

    def running_count(self) -> int:
        return len(self.state.get_runs(status="running"))

    def _watch_process(
        self,
        run_id: str,
        proc: subprocess.Popen,
        log_file,
        log_path: Path,
    ) -> None:
        """Wait for a subprocess to finish and post the appropriate event."""
        try:
            proc.wait()
        finally:
            log_file.close()

        with self._lock:
            self._processes.pop(run_id, None)

        exit_code = proc.returncode
        metrics = self._extract_metrics(log_path)

        if exit_code == 0:
            self.state.complete_run(run_id, metrics=metrics, status="completed")
            print(f"  [RunManager] {run_id} completed: metrics={metrics}")
            self.event_queue.put(Event(
                kind=EventKind.RUN_COMPLETED,
                data={"run_id": run_id, "metrics": metrics, "exit_code": exit_code},
            ))
        else:
            error = self._extract_error(log_path)
            self.state.complete_run(run_id, metrics=metrics, status="failed")
            self.state.update_run(run_id, error=error[:1000])
            print(f"  [RunManager] {run_id} failed: exit={exit_code}")
            self.event_queue.put(Event(
                kind=EventKind.RUN_FAILED,
                data={"run_id": run_id, "error": error[:500], "exit_code": exit_code},
            ))

    def _detect_train_command(self, spec: dict) -> Optional[str]:
        """Try to determine a train command from the experiment directory."""
        config = spec.get("config", {})
        config_path = None

        # If spec has config, write it and pass as argument
        if config:
            run_id = spec.get("spec_id", "auto")
            config_path = self.experiment_dir / f".run_configs/{run_id}_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        for script in ["train.py", "main.py", "scripts/train.py", "run.py"]:
            if (self.experiment_dir / script).exists():
                cmd = f"python {script}"
                if config_path:
                    cmd += f" --config {config_path}"
                return cmd

        return None

    def _extract_metrics(self, log_path: Path) -> dict[str, float]:
        """Parse training log for common metric patterns.

        Two strategies:
        1. If stdout contains "Metrics: <path>" pointing to a JSONL file,
           read the last line of that file for structured metrics.
        2. Fall back to regex extraction from stdout text.
        """
        metrics: dict[str, float] = {}
        if not log_path.exists():
            return metrics

        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return metrics

        # Strategy 1: JSONL metrics file reference in stdout
        #   e.g. "Metrics: runs/dummy/metrics.jsonl"
        jsonl_match = re.search(r"Metrics:\s*(\S+\.jsonl)", text)
        if jsonl_match:
            jsonl_rel = jsonl_match.group(1)
            jsonl_path = self.experiment_dir / jsonl_rel
            metrics = self._read_jsonl_metrics(jsonl_path)
            if metrics:
                return metrics

        # Strategy 2: Regex on stdout (last 5000 chars)
        tail = text[-5000:]
        patterns = [
            (r"(?:final[_ ])?loss[:\s=]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", "loss"),
            (r"val[_ ]loss[:\s=]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", "val_loss"),
            (r"(?:val[_ ])?accuracy[:\s=]+([0-9]+\.?[0-9]*)", "accuracy"),
            (r"(?:val[_ ])?perplexity[:\s=]+([0-9]+\.?[0-9]*)", "perplexity"),
            (r"(?:val[_ ])?bleu[:\s=]+([0-9]+\.?[0-9]*)", "bleu"),
            (r"eval[_ ]loss[:\s=]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", "eval_loss"),
        ]
        for pattern, name in patterns:
            matches = re.findall(pattern, tail, re.IGNORECASE)
            if matches:
                try:
                    metrics[name] = float(matches[-1])
                except ValueError:
                    pass

        return metrics

    def _read_jsonl_metrics(self, jsonl_path: Path) -> dict[str, float]:
        """Read final metrics from a JSONL file (last non-empty line)."""
        if not jsonl_path.exists():
            return {}
        try:
            lines = jsonl_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            if not lines:
                return {}
            # Last line has final metrics
            last = json.loads(lines[-1])
            if not isinstance(last, dict):
                return {}
            # Extract numeric values, skip metadata prefixed with "meta/"
            metrics: dict[str, float] = {}
            for k, v in last.items():
                if isinstance(v, (int, float)) and not k.startswith("meta/"):
                    # Flatten nested key names: "k/loss_total" -> "loss_total"
                    short_key = k.rsplit("/", 1)[-1] if "/" in k else k
                    metrics[short_key] = float(v)
            return metrics
        except (json.JSONDecodeError, OSError):
            return {}

    def _extract_error(self, log_path: Path) -> str:
        """Extract error message from training log."""
        if not log_path.exists():
            return "No log file found"
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = text.strip().splitlines()
            # Return last 30 lines, which usually contain the traceback
            return "\n".join(lines[-30:])
        except Exception as e:
            return f"Could not read log: {e}"

    def get_log_tail(self, run_id: str, n_lines: int = 50) -> str:
        """Get the last N lines of a run's log. Handle-based log access."""
        runs = self.state.get_runs()
        for r in runs:
            if r["run_id"] == run_id and r.get("log_path"):
                log_path = Path(r["log_path"])
                if log_path.exists():
                    try:
                        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        return "\n".join(lines[-n_lines:])
                    except Exception:
                        pass
        return f"(no log found for {run_id})"
