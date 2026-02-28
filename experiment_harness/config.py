"""Configuration dataclass for the experiment harness v2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class HarnessConfig:
    """All configuration for a harness run."""

    # ── Experiment ──────────────────────────────────────────────
    spec_path: Path = field(default_factory=lambda: Path("spec.md"))
    project_dir: Optional[Path] = None
    project_repo: Optional[str] = None
    project_branch: Optional[str] = None
    train_command: Optional[str] = None
    experiment_dir: Path = field(default_factory=lambda: Path("/workspace/experiment"))

    # ── Agent ───────────────────────────────────────────────────
    agent: str = "codex"  # "codex" or "claude"
    agent_max_turns: int = 30

    # ── Model / RLM ──────────────────────────────────────────────
    model: Optional[str] = None
    sub_model: Optional[str] = None
    research_max_iterations: int = 20
    research_max_output_chars: int = 10_000
    diverge_ratio: Optional[float] = None
    plateau_steps: Optional[int] = None

    # ── Budget ──────────────────────────────────────────────────
    max_hours: float = 24.0

    # ── Resource monitoring ─────────────────────────────────────
    poll_interval: float = 15.0        # seconds between GPU polls
    idle_threshold: float = 120.0      # seconds before idle alert

    # ── SSH / RunPod ────────────────────────────────────────────
    ssh_host: Optional[str] = None
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    ssh_user: str = "root"

    # ── RunPod provisioning ─────────────────────────────────────
    runpod_api_key: Optional[str] = None
    gpu_type: str = "NVIDIA RTX A5000"
    cloud_type: str = "COMMUNITY"
    docker_image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    volume_size_gb: int = 50
    container_disk_gb: int = 20

    # ── Auth forwarding ─────────────────────────────────────────
    forward_auth: bool = True

    # ── Monitoring ──────────────────────────────────────────────
    monitor: bool = False
    monitor_interval: int = 30

    def __post_init__(self) -> None:
        self.spec_path = Path(self.spec_path)
        self.experiment_dir = Path(self.experiment_dir)
        if self.project_dir is not None:
            self.project_dir = Path(self.project_dir)
        if self.runpod_api_key is None:
            self.runpod_api_key = os.environ.get("RUNPOD_API_KEY")

    @property
    def remote_spec_path(self) -> Path:
        return self.experiment_dir / "spec.md"

    @property
    def remote_harness_dir(self) -> Path:
        return Path("/workspace/harness")
