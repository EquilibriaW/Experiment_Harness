"""Metrics poller: extracts live metrics from training log files.

Runs as a background thread. Every poll_interval seconds, reads the tail of
each running experiment's log file, extracts metrics via regex, and writes
them to experiment_state.json under the run record.

This is the bridge between "training runs as a subprocess" and "research
agent sees structured progress data."

Also detects anomalies: NaN, loss divergence, loss plateau.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from experiment_state import ExperimentState


@dataclass
class MetricSnapshot:
    step: int = 0
    loss: Optional[float] = None
    eval_loss: Optional[float] = None
    accuracy: Optional[float] = None
    lr: Optional[float] = None
    samples_per_sec: Optional[float] = None
    epoch: Optional[float] = None
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if v is not None:
                d[k] = v
        return d


@dataclass
class Anomaly:
    kind: str  # "nan" | "diverged" | "plateaued"
    run_id: str
    message: str
    step: int = 0
    severity: str = "high"  # "high" | "medium" | "low"


class MetricsPoller:
    """Polls training logs and extracts live metrics.

    Anomaly thresholds are set by the user in the experiment spec or
    harness config. Anomalies are INFORMATIONAL — they feed into warnings
    for the research agent, they do NOT auto-kill runs. The research agent
    decides what to do with them.
    """

    def __init__(
        self,
        state: ExperimentState,
        experiment_dir: Path,
        poll_interval: float = 30.0,
        anomaly_thresholds: Optional[dict] = None,
    ) -> None:
        self.state = state
        self.experiment_dir = Path(experiment_dir)
        self.poll_interval = poll_interval

        # Human-set thresholds (from experiment spec or harness config)
        # All are optional — if not set, that check is skipped
        defaults = {
            "diverge_ratio": None,       # e.g., 2.0 = loss > 2x min → diverged
            "plateau_steps": None,        # e.g., 500 = <1% improvement over 500 steps
            "plateau_min_improvement": None,  # e.g., 0.01 = 1%
        }
        thresholds = anomaly_thresholds or {}
        self.thresholds = {**defaults, **thresholds}

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Track metric history per run for anomaly detection
        self._histories: dict[str, list[MetricSnapshot]] = {}
        self._anomalies: list[Anomaly] = []
        self._reported_anomalies: set[str] = set()  # (run_id, kind) already reported

    @property
    def anomalies(self) -> list[Anomaly]:
        return list(self._anomalies)

    def pop_anomalies(self) -> list[Anomaly]:
        """Return and clear pending anomalies."""
        result = list(self._anomalies)
        self._anomalies.clear()
        return result

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="metrics-poller")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def poll_once(self) -> None:
        """Manual single poll (for testing or immediate refresh)."""
        self._poll_all_runs()

    def get_history(self, run_id: str) -> list[dict]:
        """Get metric history for a run."""
        return [s.to_dict() for s in self._histories.get(run_id, [])]

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_all_runs()
            except Exception as e:
                print(f"[MetricsPoller] Error: {e}")
            self._stop.wait(self.poll_interval)

    def _poll_all_runs(self) -> None:
        running = self.state.get_runs(status="running")
        for run in running:
            run_id = run["run_id"]
            log_path = run.get("log_path", "")
            if not log_path:
                continue

            # Try JSONL metrics file first, fall back to log regex
            snapshot = self._extract_from_jsonl(Path(log_path))
            if not snapshot or snapshot.step == 0:
                snapshot = self._extract_from_log(Path(log_path))
            if snapshot and snapshot.step > 0:
                # Update history
                history = self._histories.setdefault(run_id, [])
                if not history or snapshot.step > history[-1].step:
                    history.append(snapshot)
                    # Keep history bounded (last 200 snapshots)
                    if len(history) > 200:
                        self._histories[run_id] = history[-200:]

                # Write to state
                self._update_run_metrics(run_id, snapshot)

                # Check anomalies
                self._check_anomalies(run_id, snapshot, history)

    def _extract_from_jsonl(self, log_path: Path) -> Optional[MetricSnapshot]:
        """Extract latest metrics from a JSONL file referenced in the log."""
        import json as _json
        import re as _re

        if not log_path.exists():
            return None

        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        # Look for "Metrics: <path>.jsonl" in stdout
        m = _re.search(r"Metrics:\s*(\S+\.jsonl)", text)
        if not m:
            return None

        jsonl_path = self.experiment_dir / m.group(1)
        if not jsonl_path.exists():
            return None

        try:
            lines = jsonl_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            if not lines:
                return None
            last = _json.loads(lines[-1])
            if not isinstance(last, dict):
                return None

            snapshot = MetricSnapshot(timestamp=time.time())
            snapshot.step = last.get("step", last.get("train/step", 0))

            # Map JSONL keys to snapshot fields
            for key in ("k/loss_total", "loss", "train/loss", "train_loss"):
                if key in last and isinstance(last[key], (int, float)):
                    snapshot.loss = float(last[key])
                    break
            for key in ("eval_loss", "val_loss", "eval/loss"):
                if key in last and isinstance(last[key], (int, float)):
                    snapshot.eval_loss = float(last[key])
                    break
            for key in ("lr", "train/lr", "learning_rate"):
                if key in last and isinstance(last[key], (int, float)):
                    snapshot.lr = float(last[key])
                    break
            for key in ("sys/tok_data_per_s", "samples_per_sec", "throughput"):
                if key in last and isinstance(last[key], (int, float)):
                    snapshot.samples_per_sec = float(last[key])
                    break

            return snapshot
        except (_json.JSONDecodeError, OSError):
            return None

    def _extract_from_log(self, log_path: Path) -> Optional[MetricSnapshot]:
        """Extract latest metrics from a training log file."""
        if not log_path.exists():
            return None

        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        # Read last 5000 chars for efficiency
        text = text[-5000:]

        snapshot = MetricSnapshot(timestamp=time.time())

        # Step/iteration patterns
        step_patterns = [
            r"(?:step|iter(?:ation)?|global_step)[:\s=]+(\d+)",
            r"(\d+)/\d+\s+\[",  # progress bar style: 1500/5000 [
            r"(?:steps?|iters?)\s+(\d+)",
        ]
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    snapshot.step = int(matches[-1])
                    break
                except ValueError:
                    pass

        # Loss patterns
        loss_patterns = [
            (r"(?:train[_ ])?loss[:\s=]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", "loss"),
            (r"(?:val[_ ]|eval[_ ])loss[:\s=]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)", "eval_loss"),
        ]
        for pattern, attr in loss_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    setattr(snapshot, attr, float(matches[-1]))
                except ValueError:
                    pass

        # Learning rate
        lr_matches = re.findall(
            r"(?:lr|learning[_ ]rate)[:\s=]+([0-9]+\.?[0-9]*(?:e[+-]?\d+)?)",
            text, re.IGNORECASE,
        )
        if lr_matches:
            try:
                snapshot.lr = float(lr_matches[-1])
            except ValueError:
                pass

        # Throughput
        throughput_matches = re.findall(
            r"([0-9]+\.?[0-9]*)\s*(?:samples?|examples?|it)/s(?:ec)?",
            text, re.IGNORECASE,
        )
        if throughput_matches:
            try:
                snapshot.samples_per_sec = float(throughput_matches[-1])
            except ValueError:
                pass

        # Epoch
        epoch_matches = re.findall(
            r"epoch[:\s=]+([0-9]+\.?[0-9]*)", text, re.IGNORECASE,
        )
        if epoch_matches:
            try:
                snapshot.epoch = float(epoch_matches[-1])
            except ValueError:
                pass

        # Accuracy
        acc_matches = re.findall(
            r"(?:val[_ ])?(?:accuracy|acc)[:\s=]+([0-9]+\.?[0-9]*)",
            text, re.IGNORECASE,
        )
        if acc_matches:
            try:
                snapshot.accuracy = float(acc_matches[-1])
            except ValueError:
                pass

        return snapshot

    def _update_run_metrics(self, run_id: str, snapshot: MetricSnapshot) -> None:
        """Write live metrics and condensed history to experiment state."""
        history = self._histories.get(run_id, [])

        # Condensed history: keep ~20 evenly spaced points
        if len(history) > 20:
            step_size = len(history) // 20
            condensed = history[::step_size] + [history[-1]]
        else:
            condensed = list(history)

        live = snapshot.to_dict()
        history_dicts = [s.to_dict() for s in condensed]

        # Compute trend (loss change per step over last 5 snapshots)
        trend = None
        if len(history) >= 2:
            recent = history[-5:]
            if recent[-1].loss is not None and recent[0].loss is not None:
                step_delta = recent[-1].step - recent[0].step
                if step_delta > 0:
                    loss_delta = recent[-1].loss - recent[0].loss
                    trend = loss_delta / step_delta

        if trend is not None:
            live["loss_trend_per_step"] = round(trend, 6)

        def _update(s):
            for r in s.get("runs", []):
                if r["run_id"] == run_id:
                    r["live_metrics"] = live
                    r["metrics_history"] = history_dicts
                    break
        self.state.update(_update)

    def _check_anomalies(
        self, run_id: str, snapshot: MetricSnapshot, history: list[MetricSnapshot],
    ) -> None:
        """Detect anomalies in training metrics. Informational only.

        All thresholds are human-set. If a threshold is None, that check
        is skipped entirely. Detected anomalies are queued for the warning
        engine — they do NOT trigger auto-kills.
        """
        if snapshot.loss is None:
            return

        # NaN / Inf — always check, this is not a threshold judgment
        if math.isnan(snapshot.loss) or math.isinf(snapshot.loss):
            key = (run_id, "nan")
            if key not in self._reported_anomalies:
                self._reported_anomalies.add(key)
                self._anomalies.append(Anomaly(
                    kind="nan", run_id=run_id,
                    message=f"Loss is NaN/Inf at step {snapshot.step}",
                    step=snapshot.step, severity="high",
                ))
            return

        losses = [s.loss for s in history if s.loss is not None
                  and not math.isnan(s.loss) and not math.isinf(s.loss)]
        if len(losses) < 3:
            return

        min_loss = min(losses)

        # Divergence (only if threshold set)
        diverge_ratio = self.thresholds.get("diverge_ratio")
        if diverge_ratio is not None and min_loss > 0:
            if snapshot.loss > diverge_ratio * min_loss:
                key = (run_id, "diverged")
                if key not in self._reported_anomalies:
                    self._reported_anomalies.add(key)
                    self._anomalies.append(Anomaly(
                        kind="diverged", run_id=run_id,
                        message=(f"Loss diverged: {snapshot.loss:.4f} at step "
                                 f"{snapshot.step}, min was {min_loss:.4f} "
                                 f"({snapshot.loss/min_loss:.1f}x, "
                                 f"threshold: {diverge_ratio}x)"),
                        step=snapshot.step, severity="medium",
                    ))

        # Plateau (only if threshold set)
        plateau_steps = self.thresholds.get("plateau_steps")
        plateau_min_improvement = self.thresholds.get("plateau_min_improvement", 0.01)
        if plateau_steps is not None and len(history) >= 5:
            old_idx = max(0, len(history) - 10)
            old_loss = history[old_idx].loss
            old_step = history[old_idx].step
            if (old_loss is not None and old_step > 0
                    and snapshot.step - old_step >= plateau_steps):
                improvement = (old_loss - snapshot.loss) / max(abs(old_loss), 1e-8)
                if improvement < plateau_min_improvement:
                    key = (run_id, "plateaued")
                    if key not in self._reported_anomalies:
                        self._reported_anomalies.add(key)
                        self._anomalies.append(Anomaly(
                            kind="plateaued", run_id=run_id,
                            message=(f"Loss plateaued at ~{snapshot.loss:.4f} "
                                     f"for {snapshot.step - old_step} steps "
                                     f"(improvement: {improvement*100:.2f}%, "
                                     f"threshold: {plateau_min_improvement*100:.0f}%)"),
                            step=snapshot.step, severity="low",
                        ))
