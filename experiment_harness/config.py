"""Configuration dataclass for the experiment harness."""

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
    project_dir: Optional[Path] = None  # local project folder to upload
    project_repo: Optional[str] = None  # git repo URL to clone instead of SCP
    project_branch: Optional[str] = None  # branch to clone (default: repo default)
    train_command: Optional[str] = None  # command to run training (e.g. "python train.py")
    experiment_dir: Path = field(default_factory=lambda: Path("/workspace/experiment"))

    # ── Agent ───────────────────────────────────────────────────
    agent: str = "codex"  # "codex" or "claude"
    agent_max_turns: int = 30  # for claude adapter

    # ── Review cycle ────────────────────────────────────────────
    review_turns: int = 2  # max reviewer↔implementer turns per round
    reviewer_agent: Optional[str] = None  # defaults to same as agent

    # ── Budget ──────────────────────────────────────────────────
    max_hours: float = 24.0

    # ── Reflection ──────────────────────────────────────────────
    reflection_word_limit: int = 8000
    reflection_path: Path = field(default_factory=lambda: Path("/workspace/experiment/reflection.md"))
    backup_dir: Path = field(default_factory=lambda: Path("/workspace/experiment/backups"))

    # ── SSH / RunPod ────────────────────────────────────────────
    ssh_host: Optional[str] = None
    ssh_port: int = 22
    ssh_key: Optional[str] = None
    ssh_user: str = "root"

    # ── RunPod provisioning ─────────────────────────────────────
    runpod_api_key: Optional[str] = None
    gpu_type: str = "NVIDIA RTX A5000"
    cloud_type: str = "COMMUNITY"  # COMMUNITY or SECURE
    docker_image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    volume_size_gb: int = 50
    container_disk_gb: int = 20

    # ── Auth forwarding ─────────────────────────────────────────
    forward_auth: bool = True  # copy local CLI auth to pod

    # ── Monitoring ──────────────────────────────────────────────
    monitor: bool = False
    monitor_interval: int = 30  # seconds between status polls

    # ── GPU safety ──────────────────────────────────────────────
    gpu_temp_limit: int = 85  # celsius
    min_disk_gb: float = 5.0

    def __post_init__(self) -> None:
        self.spec_path = Path(self.spec_path)
        self.experiment_dir = Path(self.experiment_dir)
        self.reflection_path = Path(self.reflection_path)
        self.backup_dir = Path(self.backup_dir)

        if self.project_dir is not None:
            self.project_dir = Path(self.project_dir)

        if self.reviewer_agent is None:
            self.reviewer_agent = self.agent

        if self.runpod_api_key is None:
            self.runpod_api_key = os.environ.get("RUNPOD_API_KEY")

    @property
    def remote_spec_path(self) -> Path:
        return self.experiment_dir / "spec.md"

    @property
    def remote_harness_dir(self) -> Path:
        return Path("/workspace/harness")
