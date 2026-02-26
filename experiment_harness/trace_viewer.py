#!/usr/bin/env python3
"""Trace viewer: local web UI for experiment harness logs.

Serves a collapsible tree view of per-round, per-phase JSON traces.

Usage:
  # View local logs
  python trace_viewer.py --logs-dir /path/to/experiment/logs

  # Fetch from pod first, then view
  python trace_viewer.py --ssh-host 1.2.3.4 --ssh-port 22222 --ssh-key ~/.ssh/id_ed25519

  # Auto-refresh from pod every 30s
  python trace_viewer.py --ssh-host 1.2.3.4 --ssh-port 22222 --watch
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs


DEFAULT_PORT = 8384
REMOTE_LOGS_DIR = "/workspace/experiment/logs"
REMOTE_REFLECTION = "/workspace/experiment/reflection.md"
REMOTE_TRACES_DIR = "/workspace/experiment/traces"


def fetch_logs_scp(
    host: str,
    port: int,
    user: str,
    key: str | None,
    local_dir: Path,
) -> None:
    """SCP the logs directory from the pod to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)

    key_args = ["-i", os.path.expanduser(key)] if key else []

    # Clean stale round_* dirs so archived runs don't linger as "current"
    for item in local_dir.iterdir():
        if item.is_dir() and item.name.startswith("round_"):
            shutil.rmtree(item)

    # Fetch logs
    cmd = [
        "scp", "-r", "-o", "StrictHostKeyChecking=no",
        "-P", str(port),
        *key_args,
        f"{user}@{host}:{REMOTE_LOGS_DIR}/",
        str(local_dir),
    ]
    subprocess.run(cmd, capture_output=True)

    # SCP -r creates a nested logs/ subdir — flatten it
    nested = local_dir / "logs"
    if nested.is_dir():
        for item in nested.iterdir():
            dest = local_dir / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
        nested.rmdir()

    # Fetch reflection.md
    cmd_ref = [
        "scp", "-o", "StrictHostKeyChecking=no",
        "-P", str(port),
        *key_args,
        f"{user}@{host}:{REMOTE_REFLECTION}",
        str(local_dir / "reflection.md"),
    ]
    subprocess.run(cmd_ref, capture_output=True)

    # Fetch traces/ directory (archived runs)
    traces_local = local_dir / "traces"
    traces_local.mkdir(parents=True, exist_ok=True)
    cmd_traces = [
        "scp", "-r", "-o", "StrictHostKeyChecking=no",
        "-P", str(port),
        *key_args,
        f"{user}@{host}:{REMOTE_TRACES_DIR}/",
        str(traces_local),
    ]
    subprocess.run(cmd_traces, capture_output=True)

    # SCP -r creates a nested traces/ subdir — flatten it
    nested_traces = traces_local / "traces"
    if nested_traces.is_dir():
        for item in nested_traces.iterdir():
            dest = traces_local / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
        nested_traces.rmdir()


def load_logs(logs_dir: Path) -> dict:
    """Load all round logs into a structured dict."""
    rounds = {}

    if not logs_dir.exists():
        return {"rounds": [], "reflection": ""}

    for round_dir in sorted(logs_dir.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue

        round_num = round_dir.name.split("_", 1)[1]
        phases = []

        for phase_file in sorted(round_dir.glob("*.json")):
            try:
                data = json.loads(phase_file.read_text(encoding="utf-8"))
                phases.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        if phases:
            total_duration = sum(p.get("duration_seconds", 0) for p in phases)
            any_failed = any(
                p.get("exit_code", 0) != 0 and p.get("exit_code") is not None
                for p in phases
                if p.get("phase") not in ("smoke_test",)
            )
            smoke_failed = any(
                p.get("phase") == "smoke_test" and not p.get("extra", {}).get("passed", True)
                for p in phases
            )
            rounds[round_num] = {
                "round_num": round_num,
                "phases": phases,
                "total_duration": total_duration,
                "status": "failed" if (any_failed or smoke_failed) else "ok",
            }

    # Load reflection
    reflection = ""
    ref_path = logs_dir / "reflection.md"
    if ref_path.exists():
        reflection = ref_path.read_text(encoding="utf-8")
    # Also try parent dir
    elif (logs_dir.parent / "reflection.md").exists():
        reflection = (logs_dir.parent / "reflection.md").read_text(encoding="utf-8")

    return {
        "rounds": [rounds[k] for k in sorted(rounds.keys(), key=lambda x: int(x))],
        "reflection": reflection,
    }


def list_runs(logs_dir: Path) -> list[dict]:
    """Scan for archived runs + current run. Returns list sorted by timestamp."""
    runs = []

    # Archived runs in traces/
    traces_dir = logs_dir.parent / "traces"
    if not traces_dir.exists():
        # logs_dir might be a temp dir with traces/ as a sibling or child
        traces_dir = logs_dir / "traces"
    for run_dir in sorted(traces_dir.glob("run_*")) if traces_dir.exists() else []:
        run_logs = run_dir / "logs"
        if not run_logs.is_dir():
            continue
        has_reflection = (run_dir / "reflection.md").exists()
        runs.append({
            "name": run_dir.name,
            "logs_path": str(run_logs),
            "reflection_path": str(run_dir / "reflection.md") if has_reflection else None,
            "has_reflection": has_reflection,
        })

    # Current run (always last)
    has_current_rounds = any(
        d.is_dir() and d.name.startswith("round_")
        for d in logs_dir.iterdir()
    ) if logs_dir.exists() else False
    runs.append({
        "name": "current",
        "logs_path": str(logs_dir),
        "reflection_path": None,  # loaded from default location
        "has_reflection": True,
        "is_current": True,
        "has_rounds": has_current_rounds,
    })

    return runs


class TraceHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for trace viewer API + static HTML."""

    logs_dir: Path  # set on class before serving

    def log_message(self, format, *args):
        pass  # silence request logs

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/traces":
            run_name = params.get("run", [None])[0]
            if run_name and run_name != "current":
                # Load from an archived run
                runs = list_runs(self.logs_dir)
                run_entry = next((r for r in runs if r["name"] == run_name), None)
                if run_entry:
                    logs_path = Path(run_entry["logs_path"])
                    data = load_logs(logs_path)
                    # Override reflection with the archived one if available
                    if run_entry.get("reflection_path"):
                        ref_path = Path(run_entry["reflection_path"])
                        if ref_path.exists():
                            data["reflection"] = ref_path.read_text(encoding="utf-8")
                    self._serve_json(data)
                else:
                    self._serve_json({"rounds": [], "reflection": ""})
            else:
                self._serve_json(load_logs(self.logs_dir))
        elif path == "/api/runs":
            self._serve_json(list_runs(self.logs_dir))
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _serve_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode("utf-8"))


def watch_loop(host, port, user, key, local_dir, interval):
    """Periodically re-fetch logs from the pod."""
    while True:
        time.sleep(interval)
        try:
            fetch_logs_scp(host, port, user, key, local_dir)
        except Exception as e:
            print(f"[watch] fetch error: {e}")


# ── HTML ─────────────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Experiment Trace Viewer</title>
<style>
:root {
  --bg: #0d1117;
  --bg-card: #161b22;
  --bg-hover: #1c2333;
  --bg-panel: #0d1117;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --text-bright: #f0f6fc;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --yellow: #d29922;
  --orange: #db6d28;
  --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  overflow: hidden;
}

.topbar {
  height: 48px;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 20px;
  font-size: 13px;
}
.topbar .title { font-weight: 600; font-size: 14px; color: var(--text-bright); }
.topbar .stat { color: var(--text-dim); }
.topbar .stat b { color: var(--text); font-weight: 500; }
.topbar .run-select {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  font-family: var(--sans);
  cursor: pointer;
  max-width: 240px;
}
.topbar .run-select:hover { border-color: var(--accent); }
.topbar .refresh-btn {
  margin-left: auto;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 4px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-family: var(--sans);
}
.topbar .refresh-btn:hover { background: var(--bg-hover); color: var(--text); }

.container { display: flex; height: calc(100vh - 48px); }

/* ── Sidebar ─────────────────────────────── */
.sidebar {
  width: 320px;
  min-width: 280px;
  border-right: 1px solid var(--border);
  overflow-y: auto;
  background: var(--bg-card);
  display: flex;
  flex-direction: column;
}
.sidebar::-webkit-scrollbar { width: 6px; }
.sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.sidebar-tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.sidebar-tab {
  flex: 1; padding: 8px; text-align: center; font-size: 12px; font-weight: 500;
  color: var(--text-dim); cursor: pointer; border: none; border-bottom: 2px solid transparent;
  background: transparent; font-family: var(--sans);
}
.sidebar-tab:hover { color: var(--text); }
.sidebar-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

.sidebar-panel { overflow-y: auto; flex: 1; }

.round-group { border-bottom: 1px solid var(--border); }
.round-header {
  display: flex; align-items: center; padding: 10px 16px;
  cursor: pointer; gap: 10px; font-size: 13px; transition: background 0.1s;
}
.round-header:hover { background: var(--bg-hover); }
.round-header .arrow {
  color: var(--text-dim); font-size: 10px; transition: transform 0.15s;
  width: 12px; text-align: center; flex-shrink: 0;
}
.round-header.expanded .arrow { transform: rotate(90deg); }
.round-header .round-name { font-weight: 600; color: var(--text-bright); }
.round-header .round-time {
  color: var(--text-dim); font-size: 12px; margin-left: auto; font-family: var(--mono);
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.status-dot.ok { background: var(--green); }
.status-dot.failed { background: var(--red); }

.phase-list { display: none; padding: 0 0 8px 0; }
.round-group.expanded .phase-list { display: block; }

.phase-item {
  display: flex; align-items: center; padding: 6px 16px 6px 28px;
  gap: 8px; font-size: 12px; cursor: pointer; transition: background 0.1s;
  border-left: 2px solid transparent; position: relative;
}
.phase-item::before {
  content: '';
  position: absolute;
  left: 36px;
  top: 0;
  bottom: 0;
  width: 1px;
  background: var(--border);
}
.phase-item:last-child::before { bottom: 50%; }
.phase-item::after {
  content: '';
  position: absolute;
  left: 36px;
  top: 50%;
  width: 8px;
  height: 1px;
  background: var(--border);
}
.phase-item:hover { background: var(--bg-hover); }
.phase-item.active { background: var(--bg-hover); border-left-color: var(--accent); }

.phase-item .phase-icon {
  font-size: 13px; width: 18px; text-align: center; flex-shrink: 0;
  margin-left: 20px; z-index: 1; position: relative;
}
.phase-item .phase-name { color: var(--text); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.phase-item .phase-agent { color: var(--text-dim); font-size: 11px; }
.phase-item .phase-time { color: var(--text-dim); font-family: var(--mono); font-size: 11px; flex-shrink: 0; }
.phase-item .phase-status { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.phase-status.ok { background: var(--green); }
.phase-status.failed { background: var(--red); }
.phase-status.timeout { background: var(--yellow); }

/* ── Detail panel ────────────────────────── */
.detail {
  flex: 1; overflow: hidden; background: var(--bg-panel);
  display: flex; flex-direction: column;
}
.detail::-webkit-scrollbar { width: 6px; }
.detail::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.detail-empty {
  display: flex; align-items: center; justify-content: center;
  flex: 1; color: var(--text-dim); font-size: 14px;
}
.detail-header {
  padding: 16px 24px; border-bottom: 1px solid var(--border);
  background: var(--bg-card); flex-shrink: 0;
}
.detail-header h2 { font-size: 16px; font-weight: 600; color: var(--text-bright); margin-bottom: 6px; }
.detail-meta { display: flex; gap: 16px; font-size: 12px; color: var(--text-dim); flex-wrap: wrap; }
.detail-meta .meta-item { display: flex; gap: 4px; }
.detail-meta .meta-label { color: var(--text-dim); }
.detail-meta .meta-value { color: var(--text); font-family: var(--mono); }

.detail-tabs {
  display: flex; border-bottom: 1px solid var(--border);
  background: var(--bg-card); flex-shrink: 0; padding: 0 24px;
}
.detail-tab {
  padding: 8px 16px; font-size: 13px; color: var(--text-dim); cursor: pointer;
  border: none; border-bottom: 2px solid transparent;
  background: transparent; font-family: var(--sans);
}
.detail-tab:hover { color: var(--text); }
.detail-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.detail-tab .tab-badge {
  font-size: 10px; background: var(--border); color: var(--text-dim);
  padding: 1px 6px; border-radius: 10px; margin-left: 6px;
}

.detail-content { padding: 20px 24px; flex: 1; overflow-y: auto; }
.detail-content::-webkit-scrollbar { width: 6px; }
.detail-content::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
.content-block { display: none; }
.content-block.active { display: block; }

pre.trace-output {
  background: #0a0e14; border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; font-family: var(--mono); font-size: 12px; line-height: 1.6;
  color: var(--text); overflow-x: auto; white-space: pre-wrap; word-break: break-word;
}
pre.trace-output::-webkit-scrollbar { width: 6px; height: 6px; }
pre.trace-output::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.rendered-output {
  font-size: 13px; line-height: 1.7;
}
.rendered-output h1 { font-size: 18px; margin: 16px 0 8px; color: var(--text-bright); }
.rendered-output h2 { font-size: 15px; margin: 14px 0 6px; color: var(--accent); }
.rendered-output h3 { font-size: 13px; margin: 12px 0 4px; color: var(--text-bright); }
.rendered-output p { margin-bottom: 8px; }
.rendered-output strong { color: var(--text-bright); }
.rendered-output em { color: var(--text-dim); font-style: italic; }
.rendered-output ul, .rendered-output ol { padding-left: 20px; margin-bottom: 8px; }
.rendered-output li { margin-bottom: 3px; }
.rendered-output code {
  background: #1c2333; padding: 1px 5px; border-radius: 3px;
  font-family: var(--mono); font-size: 11px;
}
.rendered-output pre {
  background: #0a0e14; border: 1px solid var(--border); border-radius: 6px;
  padding: 10px; font-family: var(--mono); font-size: 11px; overflow-x: auto; margin: 6px 0;
}
.rendered-output pre code { background: none; padding: 0; }

.empty-content { color: var(--text-dim); font-style: italic; font-size: 13px; padding: 20px 0; }

/* ── Agent event stream ────────────────── */
.agent-event { margin-bottom: 12px; border-left: 3px solid var(--border); padding-left: 12px; }
.agent-event.evt-reasoning { border-color: #6b7280; }
.agent-event.evt-agent_message { border-color: var(--accent); }
.agent-event.evt-command { border-color: #f59e0b; }
.agent-event.evt-file_change { border-color: #10b981; }
.agent-event.evt-todo { border-color: #8b5cf6; }
.agent-event.evt-error { border-color: var(--red); }
.agent-event.evt-turn_meta { border-color: var(--text-dim); }

.evt-label {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  font-weight: 600; margin-bottom: 4px;
}
.evt-reasoning .evt-label { color: #6b7280; }
.evt-agent_message .evt-label { color: var(--accent); }
.evt-command .evt-label { color: #f59e0b; }
.evt-file_change .evt-label { color: #10b981; }
.evt-todo .evt-label { color: #8b5cf6; }
.evt-error .evt-label { color: var(--red); }
.evt-turn_meta .evt-label { color: var(--text-dim); }

.evt-text { font-size: 13px; line-height: 1.6; color: var(--text); }
.evt-text p { margin-bottom: 6px; }
.evt-text strong { color: var(--text-bright); }
.evt-text code { background: #1c2333; padding: 1px 5px; border-radius: 3px; font-family: var(--mono); font-size: 11px; }
.evt-text pre { background: #0a0e14; border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-family: var(--mono); font-size: 11px; overflow-x: auto; margin: 6px 0; white-space: pre-wrap; word-break: break-word; }

.evt-cmd-line { font-family: var(--mono); font-size: 11px; color: #f59e0b; background: #1c2333; padding: 4px 8px; border-radius: 4px; margin-bottom: 4px; word-break: break-all; }
.evt-cmd-output { font-family: var(--mono); font-size: 11px; color: var(--text-dim); background: #0a0e14; padding: 8px; border-radius: 4px; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
.evt-cmd-exit { font-size: 10px; margin-top: 2px; }
.evt-cmd-exit.ok { color: var(--green); }
.evt-cmd-exit.fail { color: var(--red); }

.evt-file-list { list-style: none; padding: 0; margin: 0; }
.evt-file-list li { font-family: var(--mono); font-size: 12px; padding: 2px 0; }
.evt-file-list .kind { font-size: 10px; text-transform: uppercase; margin-right: 6px; padding: 1px 5px; border-radius: 3px; }
.evt-file-list .kind-create { background: #065f46; color: #6ee7b7; }
.evt-file-list .kind-update { background: #1e3a5f; color: #93c5fd; }
.evt-file-list .kind-delete { background: #7f1d1d; color: #fca5a5; }

.evt-todo-list { list-style: none; padding: 0; margin: 0; }
.evt-todo-list li { font-size: 12px; padding: 2px 0; }
.evt-todo-check { font-family: var(--mono); margin-right: 6px; }
.evt-todo-done { color: var(--green); }
.evt-todo-pending { color: var(--text-dim); }

.evt-toggle { font-size: 11px; color: var(--accent); cursor: pointer; border: none; background: none; padding: 0; margin-top: 4px; font-family: var(--sans); }
.evt-toggle:hover { text-decoration: underline; }

.evt-tokens { font-family: var(--mono); font-size: 11px; color: var(--text-dim); }

/* ── Stats cards (for phase extra data) ── */
.stats-cards {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px; margin-bottom: 16px;
}
.stat-card {
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px;
}
.stat-card .stat-label { font-size: 11px; color: var(--text-dim); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-card .stat-value { font-size: 16px; font-weight: 600; font-family: var(--mono); color: var(--text-bright); }
.stat-card .stat-value.ok { color: var(--green); }
.stat-card .stat-value.failed { color: var(--red); }

/* ── Meta table ──────────────────────────── */
.meta-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.meta-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
.meta-table td:first-child { color: var(--text-dim); font-weight: 500; width: 160px; white-space: nowrap; }
.meta-table td:last-child { font-family: var(--mono); font-size: 12px; }
.meta-table pre { background: #0a0e14; border-radius: 4px; padding: 8px; margin: 0; font-size: 11px; overflow-x: auto; }

/* ── Reflection panel ────────────────────── */
.reflection-panel { padding: 24px; font-size: 14px; line-height: 1.7; }
.reflection-panel h1 { font-size: 20px; margin-bottom: 12px; color: var(--text-bright); }
.reflection-panel h2 { font-size: 16px; margin: 20px 0 8px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 4px; }
.reflection-panel h3 { font-size: 14px; margin: 16px 0 6px; color: var(--text-bright); }
.reflection-panel p { margin-bottom: 8px; }
.reflection-panel ul, .reflection-panel ol { padding-left: 24px; margin-bottom: 8px; }
.reflection-panel li { margin-bottom: 4px; }
.reflection-panel code { background: #1c2333; padding: 2px 6px; border-radius: 4px; font-family: var(--mono); font-size: 12px; }
.reflection-panel pre { background: #0a0e14; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: var(--mono); font-size: 12px; overflow-x: auto; margin: 8px 0; }
.reflection-panel strong { color: var(--text-bright); }

/* ── Toggle raw/rendered ─────────────────── */
.view-toggle {
  display: flex; gap: 0; margin-bottom: 12px; border: 1px solid var(--border);
  border-radius: 6px; overflow: hidden; width: fit-content;
}
.view-toggle button {
  padding: 4px 12px; font-size: 11px; background: transparent; border: none;
  color: var(--text-dim); cursor: pointer; font-family: var(--sans);
}
.view-toggle button.active { background: var(--border); color: var(--text-bright); }
</style>
</head>
<body>

<div class="topbar">
  <span class="title">Experiment Traces</span>
  <select class="run-select" id="run-select" onchange="onRunChange(this.value)">
    <option value="current">Current Run</option>
  </select>
  <span class="stat" id="stat-rounds"></span>
  <span class="stat" id="stat-time"></span>
  <span class="stat" id="stat-phases"></span>
  <button class="refresh-btn" onclick="fetchData()">Refresh</button>
</div>

<div class="container">
  <div class="sidebar">
    <div class="sidebar-tabs">
      <button class="sidebar-tab active" data-panel="rounds-panel" onclick="switchSidebarTab(this)">Rounds</button>
      <button class="sidebar-tab" data-panel="reflection-panel" onclick="switchSidebarTab(this)">Reflection</button>
    </div>
    <div id="rounds-panel" class="sidebar-panel"></div>
    <div id="reflection-panel" class="sidebar-panel reflection-panel" style="display:none;"></div>
  </div>
  <div class="detail" id="detail">
    <div class="detail-empty">Select a phase from the sidebar</div>
  </div>
</div>

<script>
let traceData = null;
let traceFingerprint = null; // quick-check to avoid needless re-renders
let viewModes = {}; // track raw vs rendered per tab
let viewTexts = {}; // store raw text for view toggles (avoids huge DOM attributes)
// Track user selection so auto-refresh doesn't yank them away
let userSelectedKey = null;  // "round_N:seq_M" or null = auto-select latest
let currentRun = 'current'; // which run to display

function computeFingerprint(data) {
  const rounds = data.rounds || [];
  const totalPhases = rounds.reduce((s, r) => s + r.phases.length, 0);
  const last = rounds.length > 0 ? rounds[rounds.length - 1] : null;
  const lastTs = last && last.phases.length > 0
    ? last.phases[last.phases.length - 1].timestamp || ''
    : '';
  return `${rounds.length}:${totalPhases}:${lastTs}`;
}

async function fetchData() {
  // Save state BEFORE fetch — find the actual scrolling element
  const detail = document.getElementById('detail');
  const detailContent = detail ? detail.querySelector('.detail-content') : null;
  const roundsPanel = document.getElementById('rounds-panel');
  const savedDetailScroll = detailContent ? detailContent.scrollTop : 0;
  const savedRoundsScroll = roundsPanel ? roundsPanel.scrollTop : 0;
  const activeTab = detail ? detail.querySelector('.detail-tab.active') : null;
  const savedTab = activeTab ? activeTab.dataset.tab : null;

  let newData;
  try {
    const url = currentRun && currentRun !== 'current'
      ? `/api/traces?run=${encodeURIComponent(currentRun)}`
      : '/api/traces';
    const resp = await fetch(url);
    if (!resp.ok) return;
    newData = await resp.json();
  } catch (e) {
    return;
  }

  // Skip re-render if data hasn't changed
  const newFp = computeFingerprint(newData);
  if (traceFingerprint && newFp === traceFingerprint) {
    return;
  }
  traceData = newData;
  traceFingerprint = newFp;

  try {
    render(savedTab);
  } catch (e) {
    console.error('render error:', e);
    return;
  }

  // Restore scroll after browser fully lays out the new DOM (double-rAF)
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      try {
        const d = document.getElementById('detail');
        const dc = d ? d.querySelector('.detail-content') : null;
        const rp = document.getElementById('rounds-panel');
        if (dc) dc.scrollTop = savedDetailScroll;
        if (rp) rp.scrollTop = savedRoundsScroll;
      } catch (e) {}
    });
  });
}

function render(restoreTab) {
  if (!traceData) return;
  const rounds = traceData.rounds || [];

  const totalPhases = rounds.reduce((s, r) => s + r.phases.length, 0);
  const totalTime = rounds.reduce((s, r) => s + r.total_duration, 0);
  document.getElementById('stat-rounds').innerHTML = `<b>${rounds.length}</b> rounds`;
  document.getElementById('stat-time').innerHTML = `<b>${fmtDuration(totalTime)}</b> total`;
  document.getElementById('stat-phases').innerHTML = `<b>${totalPhases}</b> phases`;

  const panel = document.getElementById('rounds-panel');
  panel.innerHTML = '';

  rounds.forEach((round, ri) => {
    const group = document.createElement('div');
    group.className = 'round-group' + (ri === rounds.length - 1 ? ' expanded' : '');

    const header = document.createElement('div');
    header.className = 'round-header' + (ri === rounds.length - 1 ? ' expanded' : '');
    header.innerHTML = `
      <span class="arrow">\u25B6</span>
      <span class="status-dot ${esc(round.status)}"></span>
      <span class="round-name">Round ${esc(String(round.round_num))}</span>
      <span class="round-time">${fmtDuration(round.total_duration)}</span>
    `;
    header.onclick = () => {
      group.classList.toggle('expanded');
      header.classList.toggle('expanded');
    };

    const phaseList = document.createElement('div');
    phaseList.className = 'phase-list';

    round.phases.forEach((phase) => {
      const item = document.createElement('div');
      item.className = 'phase-item';
      const icon = phaseIcon(phase.phase);
      const status = phaseStatus(phase);
      item.innerHTML = `
        <span class="phase-icon">${icon}</span>
        <span class="phase-name">${esc(phaseName(phase.phase))}</span>
        ${phase.agent ? `<span class="phase-agent">${esc(phase.agent)}</span>` : ''}
        <span class="phase-time">${fmtDuration(phase.duration_seconds || 0)}</span>
        <span class="phase-status ${esc(status)}"></span>
      `;
      item.dataset.phaseKey = `r${round.round_num}:s${phase.seq}`;
      item.onclick = (e) => {
        e.stopPropagation();
        document.querySelectorAll('.phase-item').forEach(el => el.classList.remove('active'));
        item.classList.add('active');
        userSelectedKey = item.dataset.phaseKey;
        showPhaseDetail(phase);
      };
      phaseList.appendChild(item);
    });

    group.appendChild(header);
    group.appendChild(phaseList);
    panel.appendChild(group);
  });

  const refPanel = document.getElementById('reflection-panel');
  refPanel.innerHTML = traceData.reflection
    ? renderMarkdown(traceData.reflection)
    : '<p class="empty-content">No reflection.md found</p>';

  // Restore user selection, or auto-select latest if user hasn't clicked anything
  let restored = false;
  if (userSelectedKey) {
    const target = document.querySelector(`.phase-item[data-phase-key="${userSelectedKey}"]`);
    if (target) {
      target.classList.add('active');
      for (const r of rounds) {
        for (const p of r.phases) {
          if (`r${r.round_num}:s${p.seq}` === userSelectedKey) {
            showPhaseDetail(p, restoreTab);
            restored = true;
            break;
          }
        }
        if (restored) break;
      }
    }
  }
  if (!restored && rounds.length > 0) {
    const lastRound = rounds[rounds.length - 1];
    if (lastRound.phases.length > 0) {
      showPhaseDetail(lastRound.phases[lastRound.phases.length - 1]);
      const items = document.querySelectorAll('.phase-item');
      if (items.length > 0) items[items.length - 1].classList.add('active');
    }
  }
}

function showPhaseDetail(phase, forceTab) {
  const detail = document.getElementById('detail');
  const tabs = [];
  if (phase.prompt) tabs.push({ id: 'prompt', label: 'Prompt', size: phase.prompt.length });
  if (phase.stdout) tabs.push({ id: 'stdout', label: 'Output', size: phase.stdout.length });
  if (phase.stderr) tabs.push({ id: 'stderr', label: 'Stderr', size: phase.stderr.length });
  tabs.push({ id: 'meta', label: 'Meta' });

  const defaultTab = (forceTab && tabs.find(t => t.id === forceTab)) || tabs.find(t => t.id === 'stdout') || tabs[0];

  detail.innerHTML = `
    <div class="detail-header">
      <h2>${esc(phaseName(phase.phase))}</h2>
      <div class="detail-meta">
        ${phase.agent ? `<span class="meta-item"><span class="meta-label">Agent:</span> <span class="meta-value">${esc(phase.agent)}</span></span>` : ''}
        <span class="meta-item"><span class="meta-label">Duration:</span> <span class="meta-value">${fmtDuration(phase.duration_seconds || 0)}</span></span>
        <span class="meta-item"><span class="meta-label">Exit:</span> <span class="meta-value">${esc(String(phase.exit_code ?? 'n/a'))}</span></span>
        ${phase.timed_out ? '<span class="meta-item" style="color:var(--yellow)">TIMED OUT</span>' : ''}
        ${phase.timestamp ? `<span class="meta-item"><span class="meta-label">Time:</span> <span class="meta-value">${esc(new Date(phase.timestamp).toLocaleTimeString())}</span></span>` : ''}
      </div>
    </div>
    <div class="detail-tabs">
      ${tabs.map(t => `
        <button class="detail-tab ${t.id === defaultTab.id ? 'active' : ''}"
                data-tab="${t.id}" onclick="switchTab(this)">
          ${t.label}${t.size ? `<span class="tab-badge">${fmtSize(t.size)}</span>` : ''}
        </button>
      `).join('')}
    </div>
    <div class="detail-content">
      ${tabs.map(t => `
        <div class="content-block ${t.id === defaultTab.id ? 'active' : ''}" data-tab="${t.id}">
          ${renderTabContent(t.id, phase)}
        </div>
      `).join('')}
    </div>
  `;
}

function renderTabContent(tabId, phase) {
  if (tabId === 'prompt') {
    if (!phase.prompt) return '<p class="empty-content">No prompt</p>';
    return viewToggle('prompt', phase.prompt, 'markdown');
  }
  if (tabId === 'stdout') {
    if (!phase.stdout) return '<p class="empty-content">No output</p>';
    return viewToggle('stdout', phase.stdout, 'auto');
  }
  if (tabId === 'stderr') {
    if (!phase.stderr) return '<p class="empty-content">No stderr</p>';
    return `<pre class="trace-output">${esc(phase.stderr)}</pre>`;
  }
  if (tabId === 'meta') {
    return renderMetaTab(phase);
  }
  return '';
}

function viewToggle(key, text, renderType) {
  const mode = viewModes[key] || 'rendered';
  const rawActive = mode === 'raw' ? 'active' : '';
  const rendActive = mode === 'rendered' ? 'active' : '';
  let rendered;
  if (renderType === 'auto') {
    rendered = isNDJSON(text) ? renderAgentOutput(text) : renderMarkdown(text);
  } else {
    rendered = renderMarkdown(text);
  }
  // Store raw text in JS map (avoids huge DOM attributes)
  viewTexts[key] = { text, renderType };
  return `
    <div class="view-toggle">
      <button class="${rendActive}" onclick="setViewMode('${key}', 'rendered', this)">Rendered</button>
      <button class="${rawActive}" onclick="setViewMode('${key}', 'raw', this)">Raw</button>
    </div>
    <div class="view-content" data-view-key="${key}">
      ${mode === 'raw'
        ? `<pre class="trace-output">${esc(text)}</pre>`
        : `<div class="rendered-output">${rendered}</div>`}
    </div>
  `;
}

function setViewMode(key, mode, btn) {
  viewModes[key] = mode;
  // Toggle buttons
  const toggle = btn.parentElement;
  toggle.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  // Swap content without rebuilding the whole detail panel
  const container = toggle.nextElementSibling;
  const stored = viewTexts[key];
  if (!container || !stored) return;
  const text = stored.text;
  if (mode === 'raw') {
    container.innerHTML = `<pre class="trace-output">${esc(text)}</pre>`;
  } else {
    const renderType = stored.renderType || 'markdown';
    const rendered = renderType === 'auto'
      ? (isNDJSON(text) ? renderAgentOutput(text) : renderMarkdown(text))
      : renderMarkdown(text);
    container.innerHTML = `<div class="rendered-output">${rendered}</div>`;
  }
}

/* ── NDJSON agent output parser ─────────── */

function isNDJSON(text) {
  if (!text) return false;
  const first = text.trimStart();
  return first.startsWith('{"type":');
}

function renderAgentOutput(text) {
  const lines = text.trim().split('\n');
  const events = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    try { events.push(JSON.parse(line)); } catch(e) {}
  }

  // Deduplicate: keep only completed versions (skip started/updated for same id)
  const completed = new Map();
  const others = [];
  for (const evt of events) {
    const id = evt.item?.id;
    if (!id) { others.push(evt); continue; }
    const prev = completed.get(id);
    if (!prev || evt.type === 'item.completed') {
      completed.set(id, evt);
    }
  }
  const deduped = [];
  const seen = new Set();
  for (const evt of events) {
    const id = evt.item?.id;
    if (!id) {
      deduped.push(evt);
      continue;
    }
    if (seen.has(id)) continue;
    // Only include the completed version
    const best = completed.get(id);
    if (best) { deduped.push(best); seen.add(id); }
  }

  let html = '';
  let evtIdx = 0;
  for (const evt of deduped) {
    const item = evt.item || {};
    const type = item.type || evt.type;

    if (type === 'reasoning') {
      html += renderEvtReasoning(item, evtIdx++);
    } else if (type === 'agent_message') {
      html += renderEvtMessage(item);
    } else if (type === 'command_execution') {
      html += renderEvtCommand(item, evtIdx++);
    } else if (type === 'file_change') {
      html += renderEvtFileChange(item);
    } else if (type === 'todo_list') {
      html += renderEvtTodo(item);
    } else if (type === 'error') {
      html += renderEvtError(item);
    } else if (evt.type === 'turn.completed' && evt.usage) {
      html += renderEvtTurnMeta(evt);
    }
  }
  return html || '<p class="empty-content">No agent events parsed</p>';
}

function renderEvtReasoning(item, idx) {
  const text = item.text || '';
  return `<div class="agent-event evt-reasoning">
    <div class="evt-label">Thinking</div>
    <div class="evt-text">${renderMarkdown(text)}</div>
  </div>`;
}

function renderEvtMessage(item) {
  return `<div class="agent-event evt-agent_message">
    <div class="evt-label">Agent</div>
    <div class="evt-text">${renderMarkdown(item.text || '')}</div>
  </div>`;
}

function renderEvtCommand(item, idx) {
  const cmd = item.command || '';
  const output = item.aggregated_output || '';
  const exit = item.exit_code;
  const collapsed = output.length > 500;
  const toggleId = 'cmd-' + idx;
  const exitClass = exit === 0 ? 'ok' : (exit != null ? 'fail' : '');
  const exitLabel = exit != null ? (exit === 0 ? 'exit 0' : 'exit ' + exit) : '';
  return `<div class="agent-event evt-command">
    <div class="evt-label">Command</div>
    <div class="evt-cmd-line">${esc(cmd)}</div>
    ${output ? `<div class="evt-cmd-output" id="${toggleId}" ${collapsed ? 'style="max-height:120px"' : ''}>${esc(output)}</div>
    ${collapsed ? `<button class="evt-toggle" onclick="toggleCmdOutput('${toggleId}', this)">Show more</button>` : ''}` : ''}
    ${exitLabel ? `<div class="evt-cmd-exit ${exitClass}">${exitLabel}</div>` : ''}
  </div>`;
}

function toggleCmdOutput(id, btn) {
  const el = document.getElementById(id);
  if (!el) return;
  if (el.style.maxHeight === 'none') {
    el.style.maxHeight = '120px';
    btn.textContent = 'Show more';
  } else {
    el.style.maxHeight = 'none';
    btn.textContent = 'Show less';
  }
}

function renderEvtFileChange(item) {
  const changes = item.changes || [];
  if (!changes.length) return '';
  let list = '';
  for (const c of changes) {
    const kind = c.kind || 'update';
    const kclass = 'kind-' + kind;
    const path = c.path || '?';
    const short = path.split('/').slice(-2).join('/');
    list += `<li><span class="kind ${kclass}">${kind}</span>${esc(short)}</li>`;
  }
  return `<div class="agent-event evt-file_change">
    <div class="evt-label">File Changes</div>
    <ul class="evt-file-list">${list}</ul>
  </div>`;
}

function renderEvtTodo(item) {
  const items = item.items || [];
  if (!items.length) return '';
  let list = '';
  for (const t of items) {
    const cls = t.completed ? 'evt-todo-done' : 'evt-todo-pending';
    const check = t.completed ? '\u2713' : '\u25cb';
    list += `<li class="${cls}"><span class="evt-todo-check">${check}</span>${esc(t.text || '')}</li>`;
  }
  return `<div class="agent-event evt-todo">
    <div class="evt-label">Todo</div>
    <ul class="evt-todo-list">${list}</ul>
  </div>`;
}

function renderEvtError(item) {
  return `<div class="agent-event evt-error">
    <div class="evt-label">Error</div>
    <div class="evt-text">${esc(item.message || '')}</div>
  </div>`;
}

function renderEvtTurnMeta(evt) {
  const u = evt.usage;
  return `<div class="agent-event evt-turn_meta">
    <div class="evt-label">Turn Complete</div>
    <div class="evt-tokens">Input: ${(u.input_tokens||0).toLocaleString()} tokens (${(u.cached_input_tokens||0).toLocaleString()} cached) | Output: ${(u.output_tokens||0).toLocaleString()} tokens</div>
  </div>`;
}

function renderMetaTab(phase) {
  let html = '';

  // Show stats cards for smoke_test or any phase with extra data
  const extra = phase.extra || {};
  if (Object.keys(extra).length > 0) {
    html += '<div class="stats-cards">';
    if (extra.passed !== undefined) {
      html += `<div class="stat-card"><div class="stat-label">Status</div><div class="stat-value ${extra.passed ? 'ok' : 'failed'}">${extra.passed ? 'PASSED' : 'FAILED'}</div></div>`;
    }
    // Render any extra key-value pairs as cards (future-proof)
    for (const [k, v] of Object.entries(extra)) {
      if (k === 'passed' || k === 'stats' || v === null || v === undefined) continue;
      if (typeof v === 'object') continue;
      html += `<div class="stat-card"><div class="stat-label">${esc(k)}</div><div class="stat-value" style="font-size:12px">${typeof v === 'number' ? v.toFixed(4) : esc(String(v))}</div></div>`;
    }
    html += '</div>';
  }

  // Standard meta table
  const rows = [];
  rows.push(['Phase', phase.phase]);
  rows.push(['Round', phase.round]);
  rows.push(['Sequence', phase.seq]);
  if (phase.agent) rows.push(['Agent', phase.agent]);
  rows.push(['Timestamp', phase.timestamp || 'n/a']);
  rows.push(['Duration', fmtDuration(phase.duration_seconds || 0)]);
  rows.push(['Exit Code', phase.exit_code ?? 'n/a']);
  rows.push(['Timed Out', phase.timed_out ? 'Yes' : 'No']);

  const skip = new Set(['phase','round','seq','agent','timestamp','duration_seconds',
                        'exit_code','timed_out','prompt','stdout','stderr','extra']);
  for (const [k, v] of Object.entries(phase)) {
    if (!skip.has(k) && v !== null && v !== undefined) {
      rows.push([k, typeof v === 'object' ? JSON.stringify(v, null, 2) : String(v)]);
    }
  }

  html += `<table class="meta-table">
    ${rows.map(([k, v]) => {
      const val = String(v);
      const cell = val.includes('\n')
        ? `<pre>${esc(val)}</pre>`
        : esc(val);
      return `<tr><td>${esc(k)}</td><td>${cell}</td></tr>`;
    }).join('')}
  </table>`;

  return html;
}

function switchTab(btn) {
  const tabId = btn.dataset.tab;
  const tabBar = btn.parentElement;
  tabBar.querySelectorAll('.detail-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  const content = tabBar.nextElementSibling; // .detail-content
  content.querySelectorAll('.content-block').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tabId);
  });
}

function switchSidebarTab(btn) {
  const panelId = btn.dataset.panel;
  document.querySelectorAll('.sidebar-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.sidebar-panel').forEach(p => {
    p.style.display = p.id === panelId ? '' : 'none';
  });
}

// ── Helpers ──────────────────────────────────

function phaseIcon(phase) {
  if (phase === 'implement') return '\u{1F528}';
  if (phase === 'smoke_test') return '\u{2714}';
  if (phase === 'training' || phase === 'train') return '\u{1F4CA}';
  if (phase === 'reflect') return '\u{1F4DD}';
  if (phase.includes('reviewer')) return '\u{1F50D}';
  if (phase.includes('implementer')) return '\u{1F528}';
  if (phase === 'auto_fix_smoke') return '\u{1F527}';
  return '\u25CF';
}

function phaseName(phase) {
  const names = {
    'implement': 'Implement',
    'smoke_test': 'Smoke Test',
    'training': 'Training',
    'train': 'Training',
    'reflect': 'Reflect',
    'auto_fix_smoke': 'Auto-fix Smoke',
  };
  if (names[phase]) return names[phase];
  const m = phase.match(/review_t(\d+)_(reviewer|implementer)/);
  if (m) return `Review T${m[1]} (${m[2][0].toUpperCase() + m[2].slice(1)})`;
  return phase;
}

function phaseStatus(phase) {
  if (phase.phase === 'smoke_test' || phase.phase === 'training' || phase.phase === 'train') {
    const passed = phase.extra?.passed ?? (phase.exit_code === 0);
    return passed ? 'ok' : 'failed';
  }
  if (phase.exit_code !== null && phase.exit_code !== undefined && phase.exit_code !== 0) return 'failed';
  return 'ok';
}

function fmtDuration(secs) {
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m${Math.round(secs % 60)}s`;
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return `${h}h${m}m`;
}

function fmtSize(chars) {
  if (chars < 1024) return `${chars}c`;
  if (chars < 1024 * 1024) return `${(chars / 1024).toFixed(1)}k`;
  return `${(chars / (1024 * 1024)).toFixed(1)}M`;
}

function esc(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function renderMarkdown(md) {
  let html = esc(md);
  // Extract fenced code blocks into placeholders to protect from transforms
  const codeBlocks = [];
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(`<pre><code>${code}</code></pre>`);
    return `\x00CB${idx}\x00`;
  });
  // Extract inline code spans
  html = html.replace(/`([^`]+)`/g, (_, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(`<code>${code}</code>`);
    return `\x00CB${idx}\x00`;
  });
  // Apply transforms (safe now — code is placeholder-protected)
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li class="ul">$1</li>');
  // Ordered lists
  html = html.replace(/^(\d+)\. (.+)$/gm, '<li class="ol">$2</li>');
  // Wrap consecutive list items in proper list tags
  html = html.replace(/((?:<li class="ol">.*<\/li>\n?)+)/g, '<ol>$1</ol>');
  html = html.replace(/((?:<li class="ul">.*<\/li>\n?)+)/g, '<ul>$1</ul>');
  // Clean up class markers from li tags
  html = html.replace(/<li class="(?:ul|ol)">/g, '<li>');
  // Paragraphs for remaining loose text
  html = html.replace(/^(?!<[huplo\x00])((?!<).+)$/gm, '<p>$1</p>');
  // Restore code blocks
  html = html.replace(/\x00CB(\d+)\x00/g, (_, idx) => codeBlocks[parseInt(idx)]);
  return html;
}

async function fetchRuns() {
  try {
    const resp = await fetch('/api/runs');
    if (!resp.ok) return;
    const runs = await resp.json();
    const sel = document.getElementById('run-select');
    const prevValue = sel.value;
    sel.innerHTML = '';
    for (const run of runs) {
      const opt = document.createElement('option');
      opt.value = run.name;
      if (run.name === 'current') {
        opt.textContent = 'Current Run' + (run.has_rounds ? '' : ' (empty)');
      } else {
        // Format: "run_2026-02-25T17-22-56" → "2026-02-25 17:22:56"
        const ts = run.name.replace('run_', '').replace('T', ' ').replace(/-/g, function(m, i) {
          // First two dashes are date separators, rest are time separators
          return i > 10 ? ':' : '-';
        });
        opt.textContent = ts;
      }
      sel.appendChild(opt);
    }
    // Restore previous selection if it still exists
    if ([...sel.options].some(o => o.value === prevValue)) {
      sel.value = prevValue;
    } else {
      sel.value = 'current';
      currentRun = 'current';
    }
  } catch (e) {}
}

function onRunChange(value) {
  currentRun = value;
  traceFingerprint = null; // force re-render
  userSelectedKey = null;  // reset selection for new run
  fetchData();
}

fetchRuns();
fetchData();
setInterval(fetchData, 15000);
setInterval(fetchRuns, 30000);
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description="Experiment trace viewer")
    p.add_argument("--logs-dir", help="Local path to logs directory")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)

    # SSH options for fetching from pod
    p.add_argument("--ssh-host", help="Fetch logs from remote pod")
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--ssh-user", default="root")
    p.add_argument("--ssh-key", help="SSH private key path")

    # Watch mode
    p.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="Re-fetch from pod every N seconds (0=disabled)")

    p.add_argument("--no-open", action="store_true", help="Don't open browser")
    args = p.parse_args()

    # Determine logs directory
    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
    elif args.ssh_host:
        logs_dir = Path(tempfile.mkdtemp(prefix="trace_viewer_"))
        print(f"[Viewer] Fetching logs from {args.ssh_host}:{args.ssh_port}...")
        fetch_logs_scp(args.ssh_host, args.ssh_port, args.ssh_user, args.ssh_key, logs_dir)
        print(f"[Viewer] Logs cached at {logs_dir}")
    else:
        print("ERROR: Provide --logs-dir or --ssh-host")
        sys.exit(1)

    TraceHandler.logs_dir = logs_dir

    # Watch mode: re-fetch in background thread
    if args.watch > 0 and args.ssh_host:
        t = threading.Thread(
            target=watch_loop,
            args=(args.ssh_host, args.ssh_port, args.ssh_user, args.ssh_key,
                  logs_dir, args.watch),
            daemon=True,
        )
        t.start()
        print(f"[Viewer] Watching pod every {args.watch}s")

    import socket as _socket
    class ReusableHTTPServer(http.server.HTTPServer):
        allow_reuse_address = True
        def server_bind(self):
            self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            if hasattr(_socket, 'SO_REUSEPORT'):
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
            super().server_bind()
    server = ReusableHTTPServer(("127.0.0.1", args.port), TraceHandler)
    url = f"http://localhost:{args.port}"
    print(f"[Viewer] Serving at {url}")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Viewer] Stopped.")


if __name__ == "__main__":
    main()
