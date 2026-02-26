#!/usr/bin/env python3
"""Orchestrator: CLI entry point for the experiment harness.

Provisions a RunPod GPU (or connects to an existing host), uploads project
files + harness + CLI auth, starts the autonomous loop in tmux, and
optionally monitors progress.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from config import HarnessConfig
from runpod_manager import RunPodManager
from ssh_runner import SSHRunner


HARNESS_DIR = Path(__file__).parent
REMOTE_HARNESS_DIR = "/workspace/harness"
REMOTE_EXPERIMENT_DIR = "/workspace/experiment"
TMUX_SESSION = "experiment"

# Local auth config paths to try forwarding
AUTH_PATHS = [
    ("~/.claude", "/root/.claude"),            # Claude Code
    ("~/.codex", "/root/.codex"),              # Codex
    ("~/.config/openai", "/root/.config/openai"),  # OpenAI alt location
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autonomous ML Experiment Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-provision RunPod + run with a project folder
  python orchestrator.py \\
    --spec my_project/spec.md \\
    --project-dir ./my_project \\
    --agent codex \\
    --gpu-type "NVIDIA RTX A5000" \\
    --max-hours 8 --monitor

  # Use a git repo (much faster than SCP for large projects)
  python orchestrator.py \\
    --spec my_project/spec.md \\
    --project-repo https://github.com/user/experiment.git \\
    --agent codex \\
    --ssh-host 1.2.3.4 --ssh-port 22222

  # Use existing pod with local folder
  python orchestrator.py \\
    --spec my_project/spec.md \\
    --project-dir ./my_project \\
    --agent claude \\
    --ssh-host 1.2.3.4 --ssh-port 22222 --ssh-key ~/.ssh/id_rsa

  # Skip auth forwarding (you already logged in on the pod)
  python orchestrator.py \\
    --spec spec.md --project-dir . --agent codex \\
    --ssh-host 1.2.3.4 --no-forward-auth
        """,
    )

    p.add_argument("--spec", required=True, help="Path to experiment spec (.md)")
    p.add_argument("--project-dir", help="Local project folder to upload via SCP")
    p.add_argument("--project-repo", help="Git repo URL to clone on the pod (faster than SCP)")
    p.add_argument("--project-branch", help="Branch to clone (default: repo default)")
    p.add_argument("--train-command", help="Command to run training (e.g. 'python train.py')")
    p.add_argument("--agent", default="codex", choices=["codex", "claude"])
    p.add_argument("--reviewer-agent", default=None, choices=["codex", "claude"],
                    help="Agent for code review (defaults to same as --agent)")

    # SSH (existing pod)
    p.add_argument("--ssh-host", help="SSH host (skip RunPod provisioning)")
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--ssh-key", help="Path to SSH private key")
    p.add_argument("--ssh-user", default="root")

    # RunPod provisioning
    p.add_argument("--gpu-type", default="NVIDIA RTX A5000")
    p.add_argument("--cloud-type", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    p.add_argument("--docker-image",
                    default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    p.add_argument("--volume-size-gb", type=int, default=50)
    p.add_argument("--container-disk-gb", type=int, default=20)

    # Budget
    p.add_argument("--max-hours", type=float, default=24.0)

    # Agent
    p.add_argument("--agent-max-turns", type=int, default=30)
    p.add_argument("--review-turns", type=int, default=2,
                    help="Max reviewer↔implementer turns per round")

    # Reflection
    p.add_argument("--reflection-word-limit", type=int, default=8000)

    # Auth
    p.add_argument("--no-forward-auth", action="store_true",
                    help="Skip copying local CLI auth to the pod")

    # Restart (skip upload/bootstrap, just restart the loop)
    p.add_argument("--restart", action="store_true",
                    help="Kill existing loop and restart (skip upload/bootstrap)")

    # Monitoring
    p.add_argument("--monitor", action="store_true", help="Monitor loop after starting")
    p.add_argument("--monitor-interval", type=int, default=30)

    # RunPod API key (only needed for provisioning, not for agent auth)
    p.add_argument("--runpod-api-key", help="RunPod API key (or set RUNPOD_API_KEY)")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> HarnessConfig:
    return HarnessConfig(
        spec_path=Path(args.spec),
        project_dir=Path(args.project_dir) if args.project_dir else None,
        project_repo=args.project_repo,
        project_branch=args.project_branch,
        train_command=args.train_command,
        agent=args.agent,
        reviewer_agent=args.reviewer_agent,
        review_turns=args.review_turns,
        ssh_host=args.ssh_host,
        ssh_port=args.ssh_port,
        ssh_key=args.ssh_key,
        ssh_user=args.ssh_user,
        gpu_type=args.gpu_type,
        cloud_type=args.cloud_type,
        docker_image=args.docker_image,
        volume_size_gb=args.volume_size_gb,
        container_disk_gb=args.container_disk_gb,
        max_hours=args.max_hours,
        agent_max_turns=args.agent_max_turns,
        reflection_word_limit=args.reflection_word_limit,
        forward_auth=not args.no_forward_auth,
        monitor=args.monitor,
        monitor_interval=args.monitor_interval,
        runpod_api_key=args.runpod_api_key,
    )


def provision_pod(config: HarnessConfig) -> tuple[str, int, str | None]:
    """Create a RunPod instance and return (host, port, pod_id)."""
    if not config.runpod_api_key:
        print("ERROR: RunPod API key required. Set RUNPOD_API_KEY or use --runpod-api-key")
        sys.exit(1)

    mgr = RunPodManager(config.runpod_api_key)
    pod = mgr.create_pod(
        gpu_type=config.gpu_type,
        cloud_type=config.cloud_type,
        docker_image=config.docker_image,
        volume_size_gb=config.volume_size_gb,
        container_disk_gb=config.container_disk_gb,
    )

    if not pod.ssh_host or not pod.ssh_port:
        raise RuntimeError(f"Pod {pod.pod_id} has no SSH endpoint")

    return pod.ssh_host, pod.ssh_port, pod.pod_id


def forward_auth(ssh: SSHRunner) -> None:
    """Copy local CLI auth configs (Claude, Codex) to the pod."""
    print("\n[Auth] Forwarding local CLI auth to pod...")
    forwarded = 0

    for local_pattern, remote_dest in AUTH_PATHS:
        local_path = Path(os.path.expanduser(local_pattern))
        if not local_path.exists():
            continue

        if local_path.is_dir() and any(local_path.iterdir()):
            print(f"[Auth]   {local_path} -> {remote_dest}")
            ssh.upload_dir(local_path, remote_dest)
            forwarded += 1
        elif local_path.is_file():
            remote_parent = os.path.dirname(remote_dest)
            ssh.run(f"mkdir -p {remote_parent}")
            ssh.upload_file(local_path, remote_dest)
            forwarded += 1

    if forwarded == 0:
        print("[Auth] WARNING: No local auth configs found.")
        print("[Auth]   Expected: ~/.claude/ (Claude Code) or ~/.codex/ (Codex)")
        print("[Auth]   You may need to SSH in and run 'claude login' or 'codex login'")
    else:
        print(f"[Auth] Forwarded {forwarded} auth config(s)")


def clone_project_repo(ssh: SSHRunner, config: HarnessConfig) -> None:
    """Clone a git repo on the pod instead of uploading via SCP."""
    repo = config.project_repo
    print(f"\n[Git] Cloning {repo} -> {REMOTE_EXPERIMENT_DIR}")

    # Remove existing experiment dir if present
    ssh.run(f"rm -rf {REMOTE_EXPERIMENT_DIR}")

    branch_flag = f"-b {config.project_branch}" if config.project_branch else ""
    cmd = f"git clone --depth 1 {branch_flag} {repo} {REMOTE_EXPERIMENT_DIR}"
    exit_code = ssh.run_print(cmd, timeout=300)
    if exit_code != 0:
        raise RuntimeError(f"git clone failed (exit {exit_code})")
    print("[Git] Clone complete.")


def upload_harness_files(ssh: SSHRunner, config: HarnessConfig) -> None:
    """Upload all harness files to the remote pod."""
    print("\n[Upload] Uploading harness files...")

    # Upload remote modules (loop_runner, budget_guard, log_manager)
    remote_dir = HARNESS_DIR / "remote"
    for f in remote_dir.iterdir():
        if f.suffix == ".py" or f.suffix == ".sh":
            ssh.upload_file(f, f"{REMOTE_HARNESS_DIR}/{f.name}")

    # Upload agent adapters
    agents_dir = HARNESS_DIR / "agents"
    for f in agents_dir.iterdir():
        if f.suffix == ".py":
            ssh.upload_file(f, f"{REMOTE_HARNESS_DIR}/{f.name}")

    # Upload project: either clone from git or SCP local dir
    if config.project_repo:
        clone_project_repo(ssh, config)
    elif config.project_dir and config.project_dir.is_dir():
        print(f"[Upload] Uploading project: {config.project_dir} -> {REMOTE_EXPERIMENT_DIR}")
        ssh.upload_dir(config.project_dir, REMOTE_EXPERIMENT_DIR)

    # Upload spec
    ssh.upload_file(config.spec_path, f"{REMOTE_EXPERIMENT_DIR}/spec.md")

    # Make bootstrap.sh executable
    ssh.run(f"chmod +x {REMOTE_HARNESS_DIR}/bootstrap.sh")

    print("[Upload] Done.")


def run_bootstrap(ssh: SSHRunner) -> None:
    """Run the bootstrap script on the remote pod."""
    print("\n[Bootstrap] Installing dependencies...")
    exit_code = ssh.run_print(
        f"bash {REMOTE_HARNESS_DIR}/bootstrap.sh",
        timeout=600,
    )
    if exit_code != 0:
        print("[Bootstrap] WARNING: bootstrap.sh had errors (may be non-fatal)")


def start_loop(ssh: SSHRunner, config: HarnessConfig) -> None:
    """Start the experiment loop in a tmux session."""
    cmd_parts = [
        f"cd {REMOTE_HARNESS_DIR} &&",
        "python -u loop_runner.py",
        f"--spec {REMOTE_EXPERIMENT_DIR}/spec.md",
        f"--experiment-dir {REMOTE_EXPERIMENT_DIR}",
        f"--agent {config.agent}",
    ]

    if config.reviewer_agent and config.reviewer_agent != config.agent:
        cmd_parts.append(f"--reviewer-agent {config.reviewer_agent}")

    if config.train_command:
        cmd_parts.append(f"--train-command '{config.train_command}'")

    cmd_parts += [
        f"--max-hours {config.max_hours}",
        f"--reflection-word-limit {config.reflection_word_limit}",
        f"--review-turns {config.review_turns}",
        f"--agent-max-turns {config.agent_max_turns}",
        f"2>&1 | stdbuf -oL tee {REMOTE_EXPERIMENT_DIR}/loop.log",
    ]
    command = " ".join(cmd_parts)

    print(f"\n[Loop] Starting experiment loop in tmux session '{TMUX_SESSION}'...")
    ssh.start_tmux_session(TMUX_SESSION, command)
    print("[Loop] Loop is running. You can disconnect safely.")
    print(f"[Loop] To attach: ssh {config.ssh_user}@{config.ssh_host} "
          f"-p {config.ssh_port} -t 'tmux attach -t {TMUX_SESSION}'")


def monitor_loop(ssh: SSHRunner, config: HarnessConfig) -> None:
    """Poll the tmux session for status updates."""
    print(f"\n[Monitor] Watching experiment (Ctrl+C to stop monitoring)...")
    print(f"[Monitor] Polling every {config.monitor_interval}s")

    try:
        while True:
            if not ssh.tmux_is_running(TMUX_SESSION):
                print("\n[Monitor] Experiment loop has finished.")
                local_log = Path(f"reflection_{int(time.time())}.md")
                try:
                    ssh.download_file(
                        f"{REMOTE_EXPERIMENT_DIR}/reflection.md",
                        local_log,
                    )
                    print(f"[Monitor] Final reflection saved to: {local_log}")
                except Exception as e:
                    print(f"[Monitor] Could not download reflection: {e}")
                break

            output = ssh.tmux_capture(TMUX_SESSION, lines=10)
            if output.strip():
                print(f"\n--- Latest output ---")
                print(output.strip())
                print(f"---------------------")

            time.sleep(config.monitor_interval)
    except KeyboardInterrupt:
        print("\n[Monitor] Stopped monitoring (loop continues on pod).")


def main() -> None:
    args = parse_args()
    config = build_config(args)

    # Validate inputs
    if not config.spec_path.exists():
        print(f"ERROR: Spec file not found: {config.spec_path}")
        sys.exit(1)
    if not args.restart:
        if config.project_dir and config.project_repo:
            print("ERROR: Use --project-dir OR --project-repo, not both")
            sys.exit(1)
        if config.project_dir and not config.project_dir.is_dir():
            print(f"ERROR: Project dir not found: {config.project_dir}")
            sys.exit(1)

    try:
        # Get SSH connection
        if config.ssh_host:
            host, port = config.ssh_host, config.ssh_port
        else:
            if args.restart:
                print("ERROR: --restart requires --ssh-host (existing pod)")
                sys.exit(1)
            host, port, _ = provision_pod(config)

        with SSHRunner(host=host, port=port, user=config.ssh_user,
                       key_path=config.ssh_key) as ssh:

            if args.restart:
                # Just kill old loop and restart — no upload, no bootstrap
                print("[Restart] Killing existing loop and restarting...")
                start_loop(ssh, config)
            else:
                # Full setup: auth, upload, bootstrap, start
                if config.forward_auth:
                    forward_auth(ssh)

                upload_harness_files(ssh, config)
                run_bootstrap(ssh)
                start_loop(ssh, config)

            if config.monitor:
                monitor_loop(ssh, config)
            else:
                print(f"\nExperiment is running autonomously on {host}:{port}")
                print(f"To restart: python orchestrator.py --restart --spec {config.spec_path} "
                      f"--ssh-host {host} --ssh-port {port} --agent {config.agent}")
                print(f"To attach:  ssh {config.ssh_user}@{host} -p {port} "
                      f"-t 'tmux attach -t {TMUX_SESSION}'")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
