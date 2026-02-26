"""SSH/SCP wrapper using paramiko for remote pod communication."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import paramiko


class SSHRunner:
    """Manages SSH connection and file transfers to a remote pod."""

    def __init__(
        self,
        host: str,
        port: int = 22,
        user: str = "root",
        key_path: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.key_path = key_path
        self._client: Optional[paramiko.SSHClient] = None

    def connect(self) -> None:
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        kwargs: dict = {
            "hostname": self.host,
            "port": self.port,
            "username": self.user,
        }
        if self.key_path:
            kwargs["key_filename"] = os.path.expanduser(self.key_path)
        else:
            # Try default SSH key locations
            for default_key in ["~/.ssh/id_ed25519", "~/.ssh/id_rsa"]:
                expanded = os.path.expanduser(default_key)
                if os.path.exists(expanded):
                    kwargs["key_filename"] = expanded
                    break

        self._client.connect(**kwargs)
        print(f"[SSH] Connected to {self.user}@{self.host}:{self.port}")

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> paramiko.SSHClient:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._client

    def run(self, command: str, timeout: int = 60) -> tuple[int, str, str]:
        """Run a command and return (exit_code, stdout, stderr)."""
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return exit_code, out, err

    def run_print(self, command: str, timeout: int = 60) -> int:
        """Run a command and print output in real time. Returns exit code."""
        print(f"[SSH] $ {command}")
        exit_code, out, err = self.run(command, timeout=timeout)
        if out.strip():
            print(out)
        if err.strip():
            print(f"STDERR: {err}")
        return exit_code

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a single file via SFTP."""
        local_path = Path(local_path)
        sftp = self.client.open_sftp()
        try:
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote_path)
            self._mkdir_p(sftp, remote_dir)
            sftp.put(str(local_path), remote_path)
            print(f"[SCP] {local_path} -> {remote_path}")
        finally:
            sftp.close()

    def upload_dir(self, local_dir: str | Path, remote_dir: str) -> None:
        """Upload a directory recursively via SFTP."""
        local_dir = Path(local_dir)
        sftp = self.client.open_sftp()
        try:
            for local_file in local_dir.rglob("*"):
                if local_file.is_file():
                    rel = local_file.relative_to(local_dir)
                    remote_path = f"{remote_dir}/{rel}"
                    remote_parent = os.path.dirname(remote_path)
                    self._mkdir_p(sftp, remote_parent)
                    sftp.put(str(local_file), remote_path)
                    print(f"[SCP] {local_file} -> {remote_path}")
        finally:
            sftp.close()

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        """Download a single file via SFTP."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp = self.client.open_sftp()
        try:
            sftp.get(remote_path, str(local_path))
            print(f"[SCP] {remote_path} -> {local_path}")
        finally:
            sftp.close()

    def file_exists(self, remote_path: str) -> bool:
        """Check if a remote file exists."""
        try:
            sftp = self.client.open_sftp()
            try:
                sftp.stat(remote_path)
                return True
            except FileNotFoundError:
                return False
            finally:
                sftp.close()
        except Exception:
            return False

    def _mkdir_p(self, sftp: paramiko.SFTPClient, remote_dir: str) -> None:
        """Recursively create remote directories."""
        if not remote_dir or remote_dir == "/":
            return
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            parent = os.path.dirname(remote_dir)
            self._mkdir_p(sftp, parent)
            try:
                sftp.mkdir(remote_dir)
            except OSError:
                pass  # May already exist (race condition)

    def start_tmux_session(
        self,
        session_name: str,
        command: str,
    ) -> None:
        """Start a command in a tmux session (survives SSH disconnect)."""
        # Kill existing session if present
        self.run(f"tmux kill-session -t {session_name} 2>/dev/null || true")
        time.sleep(0.5)

        # Start new detached session
        tmux_cmd = f'tmux new-session -d -s {session_name} "{command}"'
        exit_code, out, err = self.run(tmux_cmd)
        if exit_code != 0:
            raise RuntimeError(f"Failed to start tmux session: {err}")
        print(f"[SSH] Started tmux session '{session_name}'")

    def tmux_is_running(self, session_name: str) -> bool:
        """Check if a tmux session is still running."""
        exit_code, _, _ = self.run(f"tmux has-session -t {session_name} 2>/dev/null")
        return exit_code == 0

    def tmux_capture(self, session_name: str, lines: int = 50) -> str:
        """Capture recent output from a tmux session."""
        exit_code, out, _ = self.run(
            f"tmux capture-pane -t {session_name} -p -S -{lines}"
        )
        return out if exit_code == 0 else ""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
