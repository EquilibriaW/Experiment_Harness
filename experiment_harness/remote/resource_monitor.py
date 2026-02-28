"""Resource monitor: background thread polling GPU state.

Posts events to the event queue when GPUs become free or idle.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from typing import Optional

from event_types import Event, EventKind
from experiment_state import ExperimentState, GpuInfo, ResourceSnapshot


class ResourceMonitor:
    """Polls nvidia-smi and disk, updates state, posts events."""

    def __init__(
        self,
        state: ExperimentState,
        event_queue,  # queue.Queue[Event]
        poll_interval: float = 15.0,
        idle_threshold_seconds: float = 120.0,
        experiment_dir: str = "/workspace/experiment",
    ) -> None:
        self.state = state
        self.event_queue = event_queue
        self.poll_interval = poll_interval
        self.idle_threshold = idle_threshold_seconds
        self.experiment_dir = experiment_dir

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Track GPU idle time
        self._gpu_idle_since: dict[int, float] = {}
        self._last_busy_gpus: set[int] = set()
        self._reported_mismatches: set[str] = set()
        self._underutil_since: dict[int, float] = {}
        self._gpu_to_pids: dict[int, set[int]] = {}  # gpu_idx → pids

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="resource-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def snapshot(self) -> ResourceSnapshot:
        """Take an immediate resource snapshot.

        GPU busy/free is derived from OBSERVED nvidia-smi process listings,
        not from intended assignments. This catches:
        - DDP failures falling back to single GPU
        - CUDA_VISIBLE_DEVICES misinterpretation
        - Training scripts that don't use all assigned GPUs
        """
        gpus = self._query_gpus()
        gpu_pids = self._query_gpu_processes()  # pid → set of gpu indices
        running_runs = self.state.get_runs(status="running")

        # Map our run PIDs to actually observed GPUs
        observed_busy: set[int] = set()
        assigned_busy: set[int] = set()
        run_gpu_mismatches: list[dict] = []

        for r in running_runs:
            pid = r.get("pid")
            assigned = set(r.get("gpu_ids", []))
            assigned_busy.update(assigned)

            if pid and pid in gpu_pids:
                observed = gpu_pids[pid]
                observed_busy.update(observed)
                if observed != assigned:
                    run_gpu_mismatches.append({
                        "run_id": r["run_id"],
                        "assigned": sorted(assigned),
                        "observed": sorted(observed),
                    })
            elif pid:
                # PID exists in state but not on any GPU — may have crashed
                # or not started GPU work yet. Don't count as busy.
                pass

        # Also count any unknown GPU processes (not ours)
        all_observed_pids = set()
        for pid_set in gpu_pids.values():
            all_observed_pids.update(pid_set)
        # observed_busy already has our runs' GPUs; add any GPU with
        # processes we don't recognize (someone else using the box)
        for gpu_idx, pids_on_gpu in self._gpu_to_pids.items():
            our_pids = {r.get("pid") for r in running_runs}
            foreign = pids_on_gpu - our_pids
            if foreign:
                observed_busy.add(gpu_idx)

        disk = shutil.disk_usage(self.experiment_dir)

        snap = ResourceSnapshot(
            gpus=gpus,
            gpu_count=len(gpus),
            gpus_free=len(gpus) - len(observed_busy),
            gpus_busy=sorted(observed_busy),
            disk_free_gb=disk.free / (1024 ** 3),
        )
        snap.assigned_busy = sorted(assigned_busy)
        snap.observed_busy = sorted(observed_busy)
        snap.run_gpu_mismatches = run_gpu_mismatches
        return snap

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = self.snapshot()
                self.state.update_resources(snap)

                self._check_idle(snap)
                self._check_freed(snap)
            except Exception as e:
                print(f"[ResourceMonitor] Error: {e}")

            self._stop.wait(self.poll_interval)

    def _check_freed(self, snap: ResourceSnapshot) -> None:
        """Detect newly freed GPUs and post events."""
        current_busy = set(snap.gpus_busy)
        newly_freed = self._last_busy_gpus - current_busy

        if newly_freed:
            self.event_queue.put(Event(
                kind=EventKind.GPU_FREED,
                data={"gpu_ids": list(newly_freed), "gpus_free": snap.gpus_free},
            ))

        self._last_busy_gpus = current_busy

    def _check_idle(self, snap: ResourceSnapshot) -> None:
        """Detect GPUs that have been idle beyond threshold."""
        now = time.time()
        busy = set(snap.gpus_busy)

        for gpu in snap.gpus:
            if gpu.id in busy:
                self._gpu_idle_since.pop(gpu.id, None)
            elif gpu.util_pct < 5:
                if gpu.id not in self._gpu_idle_since:
                    self._gpu_idle_since[gpu.id] = now
                elif now - self._gpu_idle_since[gpu.id] > self.idle_threshold:
                    self.event_queue.put(Event(
                        kind=EventKind.IDLE_ALERT,
                        data={
                            "gpu_id": gpu.id,
                            "idle_seconds": now - self._gpu_idle_since[gpu.id],
                            "gpus_free": snap.gpus_free,
                        },
                    ))
                    # Reset to avoid spamming — next alert after another threshold period
                    self._gpu_idle_since[gpu.id] = now
            else:
                self._gpu_idle_since.pop(gpu.id, None)

        # Warn about assigned-vs-observed mismatches
        mismatches = getattr(snap, "run_gpu_mismatches", [])
        for m in mismatches:
            if m["run_id"] not in self._reported_mismatches:
                self._reported_mismatches.add(m["run_id"])
                print(f"[ResourceMonitor] WARNING: {m['run_id']} assigned "
                      f"GPUs {m['assigned']} but observed on {m['observed']}")

        # Warn about assigned-busy GPUs that are underutilized
        assigned = set(getattr(snap, "assigned_busy", []))
        observed = set(getattr(snap, "observed_busy", []))
        ghost_busy = assigned - observed
        for gpu_id in ghost_busy:
            gpu = next((g for g in snap.gpus if g.id == gpu_id), None)
            if gpu and gpu.util_pct < 5:
                if gpu_id not in self._underutil_since:
                    self._underutil_since[gpu_id] = now
                elif now - self._underutil_since[gpu_id] > 180:  # 3 min
                    print(f"[ResourceMonitor] WARNING: GPU {gpu_id} assigned "
                          f"but 0% util for {now - self._underutil_since[gpu_id]:.0f}s")
                    self._underutil_since[gpu_id] = now  # reset
            else:
                self._underutil_since.pop(gpu_id, None)

    def _query_gpus(self) -> list[GpuInfo]:
        """Query nvidia-smi for GPU state."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []

            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append(GpuInfo(
                        id=int(parts[0]),
                        name=parts[1],
                        util_pct=float(parts[2]),
                        mem_used_gb=float(parts[3]) / 1024,
                        mem_total_gb=float(parts[4]) / 1024,
                        temp_c=int(float(parts[5])),
                    ))
            return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            return []

    def _query_gpu_processes(self) -> dict[int, set[int]]:
        """Query nvidia-smi for which PIDs are on which GPUs.

        Returns: {pid: set of gpu indices}
        Also populates self._gpu_to_pids: {gpu_idx: set of pids}
        """
        pid_to_gpus: dict[int, set[int]] = {}
        self._gpu_to_pids = {}
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,gpu_uuid",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return pid_to_gpus

            # Also need uuid → index mapping
            uuid_result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,gpu_uuid",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            uuid_to_idx: dict[str, int] = {}
            if uuid_result.returncode == 0:
                for line in uuid_result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",", 1)]
                    if len(parts) == 2:
                        uuid_to_idx[parts[1]] = int(parts[0])

            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",", 1)]
                if len(parts) == 2:
                    try:
                        pid = int(parts[0])
                        gpu_uuid = parts[1]
                        gpu_idx = uuid_to_idx.get(gpu_uuid)
                        if gpu_idx is not None:
                            pid_to_gpus.setdefault(pid, set()).add(gpu_idx)
                            self._gpu_to_pids.setdefault(gpu_idx, set()).add(pid)
                    except (ValueError, KeyError):
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return pid_to_gpus
