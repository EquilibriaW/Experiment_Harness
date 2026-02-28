#!/usr/bin/env python3
"""Orchestrator v2: provisions pod, uploads files, starts event-driven loop.

Same local-side responsibilities as v1, but starts event_loop.py instead
of loop_runner.py on the remote pod.
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

AUTH_PATHS = [
    ("~/.claude", "/root/.claude"),
    ("~/.codex", "/root/.codex"),
    ("~/.config/openai", "/root/.config/openai"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment Harness v2 — Event-Driven",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Auto-provision RunPod + run
  python orchestrator.py \\
    --spec my_project/spec.md \\
    --project-dir ./my_project \\
    --agent codex --max-hours 8 --monitor

  # Existing pod
  python orchestrator.py \\
    --spec spec.md --project-dir . --agent claude \\
    --ssh-host 1.2.3.4 --ssh-port 22222

  # With explicit train command
  python orchestrator.py \\
    --spec spec.md --project-dir . \\
    --train-command "python train.py --epochs 100" \\
    --ssh-host 1.2.3.4 --ssh-port 22222
        """,
    )

    p.add_argument("--spec", required=True, help="Path to experiment spec (.md)")
    p.add_argument("--project-dir", help="Local project folder to upload")
    p.add_argument("--project-repo", help="Git repo URL to clone on pod")
    p.add_argument("--project-branch", help="Branch to clone")
    p.add_argument("--train-command", help="Training command (e.g. 'python train.py')")
    p.add_argument("--agent", default="codex", choices=["codex", "claude"])

    # SSH
    p.add_argument("--ssh-host", help="SSH host (skip RunPod provisioning)")
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--ssh-key", help="Path to SSH private key")
    p.add_argument("--ssh-user", default="root")

    # RunPod
    p.add_argument("--gpu-type", default="NVIDIA RTX A5000")
    p.add_argument("--cloud-type", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    p.add_argument("--docker-image",
                    default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    p.add_argument("--volume-size-gb", type=int, default=50)
    p.add_argument("--container-disk-gb", type=int, default=20)

    # Budget / behavior
    p.add_argument("--max-hours", type=float, default=24.0)
    p.add_argument("--agent-max-turns", type=int, default=30)
    p.add_argument("--poll-interval", type=float, default=15.0,
                   help="GPU monitoring interval (seconds)")
    p.add_argument("--idle-threshold", type=float, default=120.0,
                   help="GPU idle alert threshold (seconds)")

    # Model / RLM
    p.add_argument("--model", default="gpt-5.3-codex",
                   help="Root LM for research agent (e.g., gpt-5.3-codex)")
    p.add_argument("--sub-model", default=None,
                   help="Sub-LM for recursive calls (e.g., anthropic/claude-haiku-4-5)")
    p.add_argument("--research-max-iterations", type=int, default=20)
    p.add_argument("--research-max-output-chars", type=int, default=10000)
    p.add_argument("--diverge-ratio", type=float, default=None,
                   help="Loss divergence threshold (e.g., 2.0)")
    p.add_argument("--plateau-steps", type=int, default=None,
                   help="Steps with <threshold improvement = plateau")

    # Auth
    p.add_argument("--no-forward-auth", action="store_true")

    # Restart
    p.add_argument("--restart", action="store_true",
                   help="Kill existing loop and restart")

    # Monitoring
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--monitor-interval", type=int, default=30)

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
        poll_interval=args.poll_interval,
        idle_threshold=args.idle_threshold,
        forward_auth=not args.no_forward_auth,
        monitor=args.monitor,
        monitor_interval=args.monitor_interval,
        runpod_api_key=args.runpod_api_key,
        model=args.model,
        sub_model=args.sub_model,
        research_max_iterations=args.research_max_iterations,
        research_max_output_chars=args.research_max_output_chars,
        diverge_ratio=args.diverge_ratio,
        plateau_steps=args.plateau_steps,
    )


def provision_pod(config: HarnessConfig) -> tuple[str, int, str | None]:
    if not config.runpod_api_key:
        print("ERROR: RunPod API key required. Set RUNPOD_API_KEY or use --runpod-api-key")
        sys.exit(1)
    mgr = RunPodManager(config.runpod_api_key)
    pod = mgr.create_pod(
        gpu_type=config.gpu_type, cloud_type=config.cloud_type,
        docker_image=config.docker_image, volume_size_gb=config.volume_size_gb,
        container_disk_gb=config.container_disk_gb,
    )
    if not pod.ssh_host or not pod.ssh_port:
        raise RuntimeError(f"Pod {pod.pod_id} has no SSH endpoint")
    return pod.ssh_host, pod.ssh_port, pod.pod_id


def forward_auth(ssh: SSHRunner) -> None:
    print("\n[Auth] Forwarding local CLI auth to pod...")
    forwarded = 0
    for local_pattern, remote_dest in AUTH_PATHS:
        local_path = Path(os.path.expanduser(local_pattern))
        if not local_path.exists():
            continue
        if local_path.is_dir() and any(local_path.iterdir()):
            ssh.upload_dir(local_path, remote_dest)
            forwarded += 1
        elif local_path.is_file():
            ssh.run(f"mkdir -p {os.path.dirname(remote_dest)}")
            ssh.upload_file(local_path, remote_dest)
            forwarded += 1
    if forwarded == 0:
        print("[Auth] WARNING: No local auth configs found.")
    else:
        print(f"[Auth] Forwarded {forwarded} auth config(s)")


def upload_files(ssh: SSHRunner, config: HarnessConfig) -> None:
    """Upload harness + project files to the pod."""
    print("\n[Upload] Uploading harness files...")

    # Upload remote modules
    remote_dir = HARNESS_DIR / "remote"
    for f in remote_dir.iterdir():
        if f.suffix in (".py", ".sh"):
            ssh.upload_file(f, f"{REMOTE_HARNESS_DIR}/{f.name}")

    # Upload agent adapters
    agents_dir = HARNESS_DIR / "agents"
    for f in agents_dir.iterdir():
        if f.suffix == ".py":
            ssh.upload_file(f, f"{REMOTE_HARNESS_DIR}/{f.name}")

    # Upload project
    if config.project_repo:
        print(f"[Upload] Cloning {config.project_repo}...")
        ssh.run(f"rm -rf {REMOTE_EXPERIMENT_DIR}")
        branch = f"-b {config.project_branch}" if config.project_branch else ""
        exit_code = ssh.run_print(
            f"git clone --depth 1 {branch} {config.project_repo} {REMOTE_EXPERIMENT_DIR}",
            timeout=300,
        )
        if exit_code != 0:
            raise RuntimeError("git clone failed")
    elif config.project_dir and config.project_dir.is_dir():
        print(f"[Upload] Uploading project: {config.project_dir}")
        ssh.upload_dir(config.project_dir, REMOTE_EXPERIMENT_DIR)

    ssh.upload_file(config.spec_path, f"{REMOTE_EXPERIMENT_DIR}/spec.md")
    ssh.run(f"chmod +x {REMOTE_HARNESS_DIR}/bootstrap.sh")
    print("[Upload] Done.")


def run_bootstrap(ssh: SSHRunner) -> None:
    print("\n[Bootstrap] Installing dependencies...")
    exit_code = ssh.run_print(f"bash {REMOTE_HARNESS_DIR}/bootstrap.sh", timeout=600)
    if exit_code != 0:
        print("[Bootstrap] WARNING: had errors (may be non-fatal)")


def start_loop(ssh: SSHRunner, config: HarnessConfig) -> None:
    """Start the event-driven loop in tmux."""
    cmd_parts = [
        # Ensure Deno and other tools are on PATH (bashrc not sourced in tmux)
        "export PATH=/root/.deno/bin:/usr/local/bin:/usr/bin:/bin:$PATH &&",
        f"export PYTHONPATH={REMOTE_EXPERIMENT_DIR}:${{PYTHONPATH:-}} &&",
        f"cd {REMOTE_HARNESS_DIR} &&",
        "python -u event_loop.py",
        f"--spec {REMOTE_EXPERIMENT_DIR}/spec.md",
        f"--experiment-dir {REMOTE_EXPERIMENT_DIR}",
        f"--agent {config.agent}",
        f"--max-hours {config.max_hours}",
        f"--agent-max-turns {config.agent_max_turns}",
        f"--poll-interval {config.poll_interval}",
        f"--idle-threshold {config.idle_threshold}",
    ]
    if config.train_command:
        cmd_parts.append(f"--train-command '{config.train_command}'")
    if config.model:
        cmd_parts.append(f"--model {config.model}")
    if config.sub_model:
        cmd_parts.append(f"--sub-model {config.sub_model}")
    if config.research_max_iterations != 20:
        cmd_parts.append(f"--research-max-iterations {config.research_max_iterations}")
    if config.research_max_output_chars != 10_000:
        cmd_parts.append(f"--research-max-output-chars {config.research_max_output_chars}")
    if config.diverge_ratio is not None:
        cmd_parts.append(f"--diverge-ratio {config.diverge_ratio}")
    if config.plateau_steps is not None:
        cmd_parts.append(f"--plateau-steps {config.plateau_steps}")
    cmd_parts.append(f"2>&1 | stdbuf -oL tee {REMOTE_EXPERIMENT_DIR}/loop.log")

    command = " ".join(cmd_parts)
    print(f"\n[Loop] Starting event loop in tmux session '{TMUX_SESSION}'...")
    ssh.start_tmux_session(TMUX_SESSION, command)
    print("[Loop] Event loop is running.")
    print(f"[Loop] Attach: ssh {config.ssh_user}@{config.ssh_host} "
          f"-p {config.ssh_port} -t 'tmux attach -t {TMUX_SESSION}'")


def monitor_loop(ssh: SSHRunner, config: HarnessConfig) -> None:
    print(f"\n[Monitor] Watching experiment (Ctrl+C to stop)...")
    try:
        while True:
            if not ssh.tmux_is_running(TMUX_SESSION):
                print("\n[Monitor] Loop has finished.")
                break
            output = ssh.tmux_capture(TMUX_SESSION, lines=10)
            if output.strip():
                print(f"\n--- Latest ---\n{output.strip()}\n--------------")
            time.sleep(config.monitor_interval)
    except KeyboardInterrupt:
        print("\n[Monitor] Stopped (loop continues on pod).")


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if not config.spec_path.exists():
        print(f"ERROR: Spec not found: {config.spec_path}")
        sys.exit(1)
    if not args.restart:
        if config.project_dir and config.project_repo:
            print("ERROR: Use --project-dir OR --project-repo, not both")
            sys.exit(1)

    try:
        if config.ssh_host:
            host, port = config.ssh_host, config.ssh_port
        else:
            if args.restart:
                print("ERROR: --restart requires --ssh-host")
                sys.exit(1)
            host, port, _ = provision_pod(config)

        with SSHRunner(host=host, port=port, user=config.ssh_user,
                       key_path=config.ssh_key) as ssh:
            if args.restart:
                print("[Restart] Killing existing loop and restarting...")
                start_loop(ssh, config)
            else:
                if config.forward_auth:
                    forward_auth(ssh)
                upload_files(ssh, config)
                run_bootstrap(ssh)
                start_loop(ssh, config)

            if config.monitor:
                monitor_loop(ssh, config)
            else:
                print(f"\nRunning autonomously on {host}:{port}")
                print(f"Restart: python orchestrator.py --restart --spec {config.spec_path} "
                      f"--ssh-host {host} --ssh-port {port} --agent {config.agent}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()
