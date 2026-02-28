#!/usr/bin/env python3
"""Trace viewer v2: local web UI for experiment harness event logs.

Serves a dashboard + event timeline for the v2 event-driven architecture.

Usage:
  # View local logs
  python trace_viewer.py --logs-dir /path/to/experiment/logs

  # Fetch from pod first, then view
  python trace_viewer.py --ssh-host 1.2.3.4 --ssh-port 22222 --ssh-key ~/.ssh/id_ed25519

  # Auto-refresh from pod every 30s
  python trace_viewer.py --ssh-host 1.2.3.4 --ssh-port 22222 --watch 30
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
REMOTE_STATE_FILE = "/workspace/experiment/experiment_state.json"
REMOTE_TRACES_DIR = "/workspace/experiment/traces"
REMOTE_RUN_LOGS_DIR = "/workspace/experiment/run_logs"


def fetch_logs_scp(
    host: str,
    port: int,
    user: str,
    key: str | None,
    local_dir: Path,
) -> None:
    """SCP event logs, state, and traces from the pod."""
    local_dir.mkdir(parents=True, exist_ok=True)

    key_args = ["-i", os.path.expanduser(key)] if key else []
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]

    # Clean stale event files so deleted events don't linger
    for item in local_dir.iterdir():
        if item.is_file() and item.suffix == ".json" and item.name[0].isdigit():
            item.unlink()

    # 1. Fetch event logs
    cmd = [
        "scp", "-r", *ssh_opts,
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
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        nested.rmdir()

    # 2. Fetch experiment_state.json
    cmd_state = [
        "scp", *ssh_opts,
        "-P", str(port),
        *key_args,
        f"{user}@{host}:{REMOTE_STATE_FILE}",
        str(local_dir / "experiment_state.json"),
    ]
    subprocess.run(cmd_state, capture_output=True)

    # 3. Fetch run_logs/ (training stdout)
    run_logs_local = local_dir / "run_logs"
    run_logs_local.mkdir(parents=True, exist_ok=True)
    cmd_rlogs = [
        "scp", "-r", *ssh_opts,
        "-P", str(port),
        *key_args,
        f"{user}@{host}:{REMOTE_RUN_LOGS_DIR}/",
        str(run_logs_local),
    ]
    subprocess.run(cmd_rlogs, capture_output=True)
    # Flatten nested
    nested_rlogs = run_logs_local / "run_logs"
    if nested_rlogs.is_dir():
        for item in nested_rlogs.iterdir():
            dest = run_logs_local / item.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(item), str(dest))
        nested_rlogs.rmdir()

    # 4. Fetch traces/ (archived runs)
    traces_local = local_dir / "traces"
    traces_local.mkdir(parents=True, exist_ok=True)
    cmd_traces = [
        "scp", "-r", *ssh_opts,
        "-P", str(port),
        *key_args,
        f"{user}@{host}:{REMOTE_TRACES_DIR}/",
        str(traces_local),
    ]
    subprocess.run(cmd_traces, capture_output=True)
    nested_traces = traces_local / "traces"
    if nested_traces.is_dir():
        for item in nested_traces.iterdir():
            dest = traces_local / item.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
        nested_traces.rmdir()


def load_events(logs_dir: Path) -> list[dict]:
    """Load all NNNN_event.json files from a logs directory."""
    events = []
    if not logs_dir.exists():
        return events

    for f in sorted(logs_dir.glob("[0-9]*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_file"] = f.name
            events.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return events


def load_state(logs_dir: Path) -> dict | None:
    """Load experiment_state.json if it exists."""
    state_path = logs_dir / "experiment_state.json"
    if not state_path.exists():
        # Try parent
        state_path = logs_dir.parent / "experiment_state.json"
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def load_run_log(logs_dir: Path, run_id: str, tail: int = 200) -> str:
    """Load last N lines of a run log."""
    rl_dir = logs_dir / "run_logs"
    if not rl_dir.exists():
        rl_dir = logs_dir.parent / "run_logs"
    log_path = rl_dir / f"{run_id}.log"
    if not log_path.exists():
        return f"No log found for {run_id}"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])
    except OSError as e:
        return f"Error reading log: {e}"


def list_runs(logs_dir: Path) -> list[dict]:
    """Scan for archived runs + current run."""
    runs = []

    # Archived runs in traces/
    traces_dir = logs_dir / "traces"
    if not traces_dir.exists():
        traces_dir = logs_dir.parent / "traces"
    for run_dir in sorted(traces_dir.glob("run_*")) if traces_dir and traces_dir.exists() else []:
        run_logs = run_dir / "logs"
        if not run_logs.is_dir():
            continue
        has_state = (run_dir / "experiment_state.json").exists()
        event_count = len(list(run_logs.glob("[0-9]*.json")))
        runs.append({
            "name": run_dir.name,
            "logs_path": str(run_logs),
            "has_state": has_state,
            "event_count": event_count,
        })

    # Current run
    event_count = len(list(logs_dir.glob("[0-9]*.json")))
    runs.append({
        "name": "current",
        "logs_path": str(logs_dir),
        "has_state": load_state(logs_dir) is not None,
        "is_current": True,
        "event_count": event_count,
    })

    return runs


class TraceHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for v2 trace viewer API + static HTML."""

    logs_dir: Path  # set on class before serving

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/events":
            run_name = params.get("run", [None])[0]
            logs_path = self._resolve_logs_path(run_name)
            self._serve_json(load_events(logs_path))
        elif path == "/api/state":
            run_name = params.get("run", [None])[0]
            logs_path = self._resolve_logs_path(run_name)
            state = load_state(logs_path)
            self._serve_json(state or {})
        elif path == "/api/runs":
            self._serve_json(list_runs(self.logs_dir))
        elif path == "/api/run_log":
            run_id = params.get("run_id", [None])[0]
            tail = int(params.get("tail", ["200"])[0])
            if run_id:
                self._serve_text(load_run_log(self.logs_dir, run_id, tail))
            else:
                self.send_error(400, "Missing run_id param")
        else:
            self.send_error(404)

    def _resolve_logs_path(self, run_name: str | None) -> Path:
        if run_name and run_name != "current":
            runs = list_runs(self.logs_dir)
            entry = next((r for r in runs if r["name"] == run_name), None)
            if entry:
                return Path(entry["logs_path"])
        return self.logs_dir

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

    def _serve_text(self, text: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))


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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment Trace Viewer v2</title>
<style>
:root {
  --bg: #0d1117;
  --bg-card: #161b22;
  --bg-hover: #1c2128;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --orange: #d29922;
  --purple: #bc8cff;
  --cyan: #39d2c0;
  --font-mono: 'SF Mono', 'Cascadia Code', 'Fira Code', Menlo, monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.5;
}

.app {
  display: grid;
  grid-template-columns: 1fr 400px;
  grid-template-rows: auto 1fr;
  height: 100vh;
  gap: 1px;
  background: var(--border);
}

header {
  grid-column: 1 / -1;
  background: var(--bg-card);
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  border-bottom: 1px solid var(--border);
}

header h1 {
  font-size: 16px;
  font-weight: 600;
  white-space: nowrap;
}

.run-select {
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 13px;
  font-family: var(--font-mono);
}

.header-stats {
  display: flex;
  gap: 16px;
  margin-left: auto;
  font-size: 13px;
  color: var(--text-dim);
  font-family: var(--font-mono);
}

.header-stats .stat {
  display: flex;
  align-items: center;
  gap: 4px;
}

.header-stats .stat-value {
  color: var(--text);
  font-weight: 600;
}

.auto-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: rgba(88, 166, 255, 0.15);
  color: var(--accent);
  font-family: var(--font-mono);
}

/* ── Main panel (left) ── */
.main-panel {
  background: var(--bg);
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* ── Sidebar (right) ── */
.sidebar {
  background: var(--bg);
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* ── Cards ── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

.card-header {
  padding: 10px 14px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.card-body { padding: 12px 14px; }

/* ── Dashboard cards ── */
.dashboard {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.dash-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
}

.dash-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  margin-bottom: 4px;
}

.dash-value {
  font-size: 22px;
  font-weight: 700;
  font-family: var(--font-mono);
}

.dash-sub {
  font-size: 12px;
  color: var(--text-dim);
  margin-top: 2px;
  font-family: var(--font-mono);
}

/* ── Runs table ── */
.runs-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.runs-table th {
  text-align: left;
  padding: 8px 10px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  font-weight: 600;
}

.runs-table td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}

.runs-table tr:last-child td { border-bottom: none; }

.runs-table tr:hover { background: var(--bg-hover); }

.status-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--font-mono);
}

.status-running  { background: rgba(88, 166, 255, 0.15); color: var(--accent); }
.status-completed { background: rgba(63, 185, 80, 0.15); color: var(--green); }
.status-failed    { background: rgba(248, 81, 73, 0.15); color: var(--red); }
.status-queued    { background: rgba(139, 148, 158, 0.15); color: var(--text-dim); }
.status-killed    { background: rgba(210, 153, 34, 0.15); color: var(--orange); }

/* ── Event timeline ── */
.timeline { list-style: none; }

.timeline-item {
  display: flex;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.1s;
}

.timeline-item:hover { background: var(--bg-hover); margin: 0 -14px; padding: 10px 14px; }
.timeline-item:last-child { border-bottom: none; }

.tl-seq {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-dim);
  min-width: 36px;
  text-align: right;
  padding-top: 2px;
}

.tl-icon {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  flex-shrink: 0;
}

.tl-body { flex: 1; min-width: 0; }

.tl-title {
  font-weight: 600;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.tl-time {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-dim);
  font-weight: 400;
}

.tl-detail {
  font-size: 12px;
  color: var(--text-dim);
  margin-top: 3px;
  white-space: pre-wrap;
  word-break: break-word;
}

/* Event type colors */
.ev-launch .tl-icon       { background: rgba(88, 166, 255, 0.15); color: var(--accent); }
.ev-research_start .tl-icon { background: rgba(188, 140, 255, 0.15); color: var(--purple); }
.ev-research_complete .tl-icon { background: rgba(57, 210, 192, 0.15); color: var(--cyan); }
.ev-train_fix .tl-icon     { background: rgba(210, 153, 34, 0.15); color: var(--orange); }
.ev-run_completed .tl-icon { background: rgba(63, 185, 80, 0.15); color: var(--green); }
.ev-run_failed .tl-icon    { background: rgba(248, 81, 73, 0.15); color: var(--red); }
.ev-gpu_freed .tl-icon     { background: rgba(139, 148, 158, 0.15); color: var(--text-dim); }
.ev-budget_tick .tl-icon   { background: rgba(139, 148, 158, 0.1); color: var(--text-dim); }
.ev-idle_alert .tl-icon    { background: rgba(210, 153, 34, 0.15); color: var(--orange); }
.ev-queue_empty .tl-icon   { background: rgba(139, 148, 158, 0.15); color: var(--text-dim); }
.ev-shutdown .tl-icon      { background: rgba(248, 81, 73, 0.15); color: var(--red); }
.ev-init .tl-icon          { background: rgba(63, 185, 80, 0.15); color: var(--green); }

/* ── Detail modal ── */
.modal-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  z-index: 100;
  align-items: center;
  justify-content: center;
}

.modal-overlay.open { display: flex; }

.modal {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  width: 90%;
  max-width: 700px;
  max-height: 80vh;
  overflow-y: auto;
  padding: 20px;
}

.modal h3 {
  font-size: 16px;
  margin-bottom: 12px;
}

.modal pre {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  font-family: var(--font-mono);
  font-size: 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 60vh;
}

.modal-close {
  float: right;
  background: none;
  border: none;
  color: var(--text-dim);
  font-size: 20px;
  cursor: pointer;
}
.modal-close:hover { color: var(--text); }

/* ── Hypotheses ── */
.hyp-item {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
.hyp-item:last-child { border-bottom: none; }

.hyp-status {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--font-mono);
  margin-right: 6px;
}

.hyp-active   { background: rgba(88, 166, 255, 0.15); color: var(--accent); }
.hyp-confirmed { background: rgba(63, 185, 80, 0.15); color: var(--green); }
.hyp-rejected  { background: rgba(248, 81, 73, 0.15); color: var(--red); }
.hyp-paused    { background: rgba(139, 148, 158, 0.15); color: var(--text-dim); }

.hyp-desc {
  font-size: 13px;
  margin-top: 4px;
}

.hyp-evidence {
  font-size: 12px;
  color: var(--text-dim);
  margin-top: 2px;
}

/* ── Summary ── */
.summary-text {
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 300px;
  overflow-y: auto;
  line-height: 1.6;
}

/* ── Queue ── */
.queue-item {
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.queue-item:last-child { border-bottom: none; }

.queue-spec-id {
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--accent);
}

.queue-desc {
  color: var(--text-dim);
  font-size: 12px;
  margin-top: 2px;
}

.queue-meta {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 2px;
}

/* ── Metrics sparkline ── */
.spark-container {
  width: 100%;
  height: 80px;
  position: relative;
}
.spark-container canvas { width: 100%; height: 100%; }

/* ── Tabs ── */
.tab-bar {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
}
.tab-btn {
  padding: 8px 16px;
  background: none;
  border: none;
  color: var(--text-dim);
  font-size: 13px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  font-family: var(--font-sans);
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}
.tab-content { display: none; }
.tab-content.active { display: block; }

/* ── Empty state ── */
.empty {
  text-align: center;
  padding: 40px 20px;
  color: var(--text-dim);
}
.empty-icon { font-size: 32px; margin-bottom: 8px; }

/* ── GPU bar ── */
.gpu-bar {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.gpu-chip {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 10px;
  font-family: var(--font-mono);
  font-size: 11px;
  min-width: 120px;
}
.gpu-chip-busy { border-color: var(--accent); }
.gpu-chip .gpu-name {
  font-weight: 600;
  color: var(--text);
  margin-bottom: 2px;
}
.gpu-chip .gpu-detail { color: var(--text-dim); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ── Log viewer button ── */
.btn-sm {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  cursor: pointer;
  font-family: var(--font-mono);
}
.btn-sm:hover { color: var(--text); border-color: var(--text-dim); }
</style>
</head>
<body>

<div class="app">
  <header>
    <h1>Experiment Trace Viewer</h1>
    <select class="run-select" id="runSelect" onchange="switchRun(this.value)">
      <option value="current">current</option>
    </select>
    <div class="header-stats" id="headerStats"></div>
    <span class="auto-badge" id="autoRefresh">auto-refresh 15s</span>
  </header>

  <!-- Left: main panel -->
  <div class="main-panel">
    <!-- Dashboard row -->
    <div class="dashboard" id="dashboard"></div>

    <!-- Runs table -->
    <div class="card">
      <div class="card-header">
        Runs
        <span id="runsCount" style="font-family:var(--font-mono)"></span>
      </div>
      <div class="card-body" style="padding:0; overflow-x:auto;">
        <table class="runs-table" id="runsTable">
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>GPU</th>
              <th>Hypothesis</th>
              <th>Metrics</th>
              <th>Duration</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="runsBody"></tbody>
        </table>
      </div>
    </div>

    <!-- Tabs: Timeline / Queue / Hypotheses -->
    <div class="card">
      <div class="tab-bar">
        <button class="tab-btn active" onclick="showTab('timeline',this)">Event Timeline</button>
        <button class="tab-btn" onclick="showTab('queue',this)">Queue</button>
        <button class="tab-btn" onclick="showTab('hypotheses',this)">Hypotheses</button>
      </div>
      <div id="tab-timeline" class="tab-content active">
        <div class="card-body" style="padding:0;">
          <ul class="timeline" id="timeline"></ul>
        </div>
      </div>
      <div id="tab-queue" class="tab-content">
        <div class="card-body" id="queueBody"></div>
      </div>
      <div id="tab-hypotheses" class="tab-content">
        <div class="card-body" id="hypothesesBody"></div>
      </div>
    </div>
  </div>

  <!-- Right: sidebar -->
  <div class="sidebar">
    <!-- Summary / Reflection -->
    <div class="card">
      <div class="card-header">Agent Summary</div>
      <div class="card-body">
        <div class="summary-text" id="summaryText">No summary yet.</div>
      </div>
    </div>

    <!-- GPUs -->
    <div class="card">
      <div class="card-header">GPUs</div>
      <div class="card-body">
        <div class="gpu-bar" id="gpuBar"></div>
      </div>
    </div>

    <!-- Best Result -->
    <div class="card">
      <div class="card-header">Best Result</div>
      <div class="card-body" id="bestResult">
        <div class="empty"><div class="empty-icon">--</div>No results yet</div>
      </div>
    </div>

    <!-- Goal -->
    <div class="card">
      <div class="card-header">Goal</div>
      <div class="card-body">
        <div class="summary-text" id="goalText" style="max-height:200px; font-size:12px; color:var(--text-dim);">--</div>
      </div>
    </div>
  </div>
</div>

<!-- Event detail modal -->
<div class="modal-overlay" id="modalOverlay" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <h3 id="modalTitle"></h3>
    <pre id="modalBody"></pre>
  </div>
</div>

<script>
const EVENT_ICONS = {
  launch: '\u25B6',
  research_start: '\uD83D\uDD2C',
  research_complete: '\u2714',
  train_fix: '\uD83D\uDD27',
  run_completed: '\u2705',
  run_failed: '\u274C',
  gpu_freed: '\uD83D\uDDA5',
  budget_tick: '\u23F1',
  idle_alert: '\u26A0',
  queue_empty: '\uD83D\uDCED',
  shutdown: '\u23F9',
  init: '\uD83D\uDE80',
};

let currentRun = 'current';
let lastEvents = [];
let lastState = {};

function switchRun(name) {
  currentRun = name;
  fetchData();
}

function showTab(id, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('open');
}

function showEventDetail(ev) {
  document.getElementById('modalTitle').textContent =
    `#${ev.seq} ${ev.event}  —  ${fmtTime(ev.timestamp)}`;
  const copy = {...ev};
  delete copy._file;
  document.getElementById('modalBody').textContent = JSON.stringify(copy, null, 2);
  document.getElementById('modalOverlay').classList.add('open');
}

function showRunLog(runId) {
  document.getElementById('modalTitle').textContent = `Run Log: ${runId}`;
  document.getElementById('modalBody').textContent = 'Loading...';
  document.getElementById('modalOverlay').classList.add('open');
  fetch(`/api/run_log?run_id=${encodeURIComponent(runId)}&tail=300`)
    .then(r => r.text())
    .then(text => { document.getElementById('modalBody').textContent = text; })
    .catch(err => { document.getElementById('modalBody').textContent = 'Error loading log: ' + err.message; });
}

function fmtTime(ts) {
  if (!ts) return '--';
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return ts; }
}

function fmtDuration(seconds) {
  if (!seconds || seconds <= 0) return '--';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds/60)}m ${Math.round(seconds%60)}s`;
  return `${Math.floor(seconds/3600)}h ${Math.floor((seconds%3600)/60)}m`;
}

function fmtMetric(val) {
  if (val === undefined || val === null) return '--';
  if (typeof val === 'number') {
    if (Math.abs(val) < 0.001) return val.toExponential(2);
    return val.toFixed(4);
  }
  return String(val);
}

function renderDashboard(state) {
  const el = document.getElementById('dashboard');
  if (!state || !state.budget) {
    el.innerHTML = '<div class="empty">Waiting for experiment state...</div>';
    return;
  }
  const b = state.budget || {};
  const nRuns = (state.runs || []).length;
  const nRunning = (state.runs || []).filter(r => r.status === 'running').length;
  const nCompleted = (state.runs || []).filter(r => r.status === 'completed').length;
  const nQueued = (state.experiment_list || []).length;
  const elapsed = (b.elapsed_hours || 0).toFixed(1);
  const maxH = (b.max_hours || 24).toFixed(0);
  const remaining = Math.max(0, (b.max_hours || 24) - (b.elapsed_hours || 0)).toFixed(1);
  const pct = Math.min(100, ((b.elapsed_hours || 0) / (b.max_hours || 24)) * 100).toFixed(0);

  el.innerHTML = `
    <div class="dash-card">
      <div class="dash-label">Budget</div>
      <div class="dash-value">${elapsed}h <span style="color:var(--text-dim);font-size:14px">/ ${maxH}h</span></div>
      <div class="dash-sub">${remaining}h remaining (${pct}%)</div>
    </div>
    <div class="dash-card">
      <div class="dash-label">Round</div>
      <div class="dash-value">${state.round || 0}</div>
      <div class="dash-sub">research iterations</div>
    </div>
    <div class="dash-card">
      <div class="dash-label">Runs</div>
      <div class="dash-value">${nRuns}</div>
      <div class="dash-sub">${nRunning} running, ${nCompleted} done</div>
    </div>
    <div class="dash-card">
      <div class="dash-label">Queue</div>
      <div class="dash-value">${nQueued}</div>
      <div class="dash-sub">experiments pending</div>
    </div>
    <div class="dash-card">
      <div class="dash-label">Agent Calls</div>
      <div class="dash-value">${b.total_agent_calls || 0}</div>
      <div class="dash-sub">$${(b.total_agent_cost_usd || 0).toFixed(2)} USD</div>
    </div>
    <div class="dash-card">
      <div class="dash-label">GPU Hours</div>
      <div class="dash-value">${(b.total_gpu_hours || 0).toFixed(1)}</div>
      <div class="dash-sub">${(state.resources || {}).gpu_count || '?'} GPUs total</div>
    </div>
  `;
}

function renderHeaderStats(state) {
  const el = document.getElementById('headerStats');
  if (!state || !state.budget) { el.innerHTML = ''; return; }
  const b = state.budget;
  el.innerHTML = `
    <span class="stat">Round <span class="stat-value">${state.round || 0}</span></span>
    <span class="stat"><span class="stat-value">${((b.elapsed_hours||0)).toFixed(1)}</span>/${(b.max_hours||24).toFixed(0)}h</span>
    <span class="stat">Runs <span class="stat-value">${(state.runs||[]).length}</span></span>
  `;
}

function renderRuns(state) {
  const tbody = document.getElementById('runsBody');
  const countEl = document.getElementById('runsCount');
  if (!state || !state.runs || state.runs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">No runs yet</td></tr>';
    countEl.textContent = '';
    return;
  }

  const runs = [...state.runs].sort((a, b) => {
    const order = { running: 0, queued: 1, completed: 2, failed: 3, killed: 4 };
    return (order[a.status] || 5) - (order[b.status] || 5);
  });

  countEl.textContent = `${runs.length}`;
  tbody.innerHTML = runs.map(r => {
    const dur = r.end_time && r.start_time
      ? fmtDuration(r.end_time - r.start_time)
      : r.start_time ? fmtDuration(Date.now()/1000 - r.start_time) + '...' : '--';

    const metrics = r.metrics || {};
    const live = r.live_metrics || {};
    const metricStr = Object.keys(metrics).length > 0
      ? Object.entries(metrics).map(([k,v]) => `${k}: ${fmtMetric(v)}`).join(', ')
      : live.loss !== undefined ? `loss: ${fmtMetric(live.loss)} (step ${live.step || '?'})` : '--';

    const gpuStr = (r.gpu_ids || []).map(g => `GPU ${g}`).join(', ') || '--';
    const hyp = r.hypothesis ? (r.hypothesis.length > 60 ? r.hypothesis.slice(0, 60) + '...' : r.hypothesis) : '--';

    return `<tr>
      <td><span style="font-family:var(--font-mono);font-weight:600">${r.run_id}</span></td>
      <td><span class="status-badge status-${r.status}">${r.status}</span></td>
      <td style="font-family:var(--font-mono);font-size:12px">${gpuStr}</td>
      <td style="font-size:12px;max-width:200px;overflow:hidden;text-overflow:ellipsis" title="${(r.hypothesis||'').replace(/"/g,'&quot;')}">${hyp}</td>
      <td style="font-family:var(--font-mono);font-size:12px">${metricStr}</td>
      <td style="font-family:var(--font-mono);font-size:12px">${dur}</td>
      <td><button class="btn-sm" onclick="showRunLog('${r.run_id}')">log</button></td>
    </tr>`;
  }).join('');
}

function renderTimeline(events) {
  const el = document.getElementById('timeline');
  if (!events || events.length === 0) {
    el.innerHTML = '<li class="empty"><div class="empty-icon">--</div>No events yet</li>';
    return;
  }

  // Show most recent first
  const sorted = [...events].reverse();
  el.innerHTML = sorted.map(ev => {
    const icon = EVENT_ICONS[ev.event] || '\u2022';
    const evClass = `ev-${ev.event}`;
    let detail = '';

    switch (ev.event) {
      case 'launch':
        detail = `Run ${ev.run_id || '?'} on GPU ${(ev.gpu_ids || []).join(', ')}`;
        break;
      case 'research_start':
        detail = ev.trigger || '';
        break;
      case 'research_complete':
        detail = `${ev.n_specs || 0} experiments queued, ${ev.n_kills || 0} killed`;
        if (ev.reflection) detail += `\n${ev.reflection}`;
        break;
      case 'train_fix':
        detail = `Run ${ev.run_id || '?'} fixed in ${fmtDuration(ev.duration_s || 0)}`;
        break;
      case 'run_completed':
        detail = `Run ${ev.run_id || '?'}`;
        if (ev.metrics) detail += ` — ${JSON.stringify(ev.metrics)}`;
        break;
      case 'run_failed':
        detail = `Run ${ev.run_id || '?'}`;
        if (ev.error) detail += `: ${ev.error}`;
        break;
      default:
        // Show any extra fields
        const extra = Object.entries(ev)
          .filter(([k]) => !['seq','event','timestamp','budget','_file'].includes(k))
          .map(([k,v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
          .join(', ');
        detail = extra;
    }

    // Truncate long details
    if (detail.length > 300) detail = detail.slice(0, 300) + '...';

    return `<li class="timeline-item ${evClass}" onclick='showEventDetail(${JSON.stringify(ev).replace(/'/g,"\\'")})'">
      <span class="tl-seq">#${ev.seq || '?'}</span>
      <span class="tl-icon">${icon}</span>
      <div class="tl-body">
        <div class="tl-title">
          ${ev.event}
          <span class="tl-time">${fmtTime(ev.timestamp)}</span>
        </div>
        ${detail ? `<div class="tl-detail">${escHtml(detail)}</div>` : ''}
      </div>
    </li>`;
  }).join('');
}

function renderQueue(state) {
  const el = document.getElementById('queueBody');
  if (!state || !state.experiment_list || state.experiment_list.length === 0) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">--</div>Queue empty</div>';
    return;
  }
  el.innerHTML = state.experiment_list.map(exp => `
    <div class="queue-item">
      <div><span class="queue-spec-id">${exp.spec_id || '?'}</span>
        <span style="margin-left:8px;font-size:11px;color:var(--text-dim)">priority: ${exp.priority ?? '?'} | gpus: ${exp.gpu_requirement || 1} | uncertainty: ${exp.uncertainty || '?'}</span>
      </div>
      <div class="queue-desc">${escHtml(exp.description || '')}</div>
      ${exp.hypothesis ? `<div class="queue-meta">H: ${escHtml(exp.hypothesis)}</div>` : ''}
      ${exp.requires_code_change ? `<div class="queue-meta" style="color:var(--orange)">Requires code change: ${escHtml(exp.code_change_description || 'yes')}</div>` : ''}
    </div>
  `).join('');
}

function renderHypotheses(state) {
  const el = document.getElementById('hypothesesBody');
  if (!state || !state.hypotheses || state.hypotheses.length === 0) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">--</div>No hypotheses yet</div>';
    return;
  }
  el.innerHTML = state.hypotheses.map(h => `
    <div class="hyp-item">
      <div>
        <span class="hyp-status hyp-${h.status || 'active'}">${h.status || 'active'}</span>
        <span style="font-family:var(--font-mono);font-size:12px;color:var(--text-dim)">${h.id || ''}</span>
      </div>
      <div class="hyp-desc">${escHtml(h.description || '')}</div>
      ${(h.evidence || []).length > 0 ? `<div class="hyp-evidence">Evidence: ${h.evidence.map(e => escHtml(e)).join('; ')}</div>` : ''}
    </div>
  `).join('');
}

function renderGPUs(state) {
  const el = document.getElementById('gpuBar');
  if (!state || !state.resources || !state.resources.gpus) {
    el.innerHTML = '<span style="color:var(--text-dim)">No GPU info</span>';
    return;
  }
  const busy = new Set(state.resources.gpus_busy || []);
  el.innerHTML = state.resources.gpus.map(g => `
    <div class="gpu-chip ${busy.has(g.id) ? 'gpu-chip-busy' : ''}">
      <div class="gpu-name">GPU ${g.id}: ${g.name || '?'}</div>
      <div class="gpu-detail">${(g.mem_used_gb||0).toFixed(1)}/${(g.mem_total_gb||0).toFixed(0)} GB | ${(g.util_pct||0).toFixed(0)}% | ${g.temp_c||'?'}C</div>
      <div class="gpu-detail">${busy.has(g.id) ? '<span style="color:var(--accent)">BUSY</span>' : '<span style="color:var(--green)">FREE</span>'}</div>
    </div>
  `).join('');
}

function renderBestResult(state) {
  const el = document.getElementById('bestResult');
  if (!state || !state.best_result || !state.best_result.run_id) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">--</div>No results yet</div>';
    return;
  }
  const br = state.best_result;
  el.innerHTML = `
    <div style="text-align:center">
      <div style="font-size:11px;text-transform:uppercase;color:var(--text-dim);letter-spacing:0.05em">${br.metric || 'metric'}</div>
      <div style="font-size:28px;font-weight:700;font-family:var(--font-mono);color:var(--green);margin:4px 0">${fmtMetric(br.value)}</div>
      <div style="font-size:12px;font-family:var(--font-mono);color:var(--text-dim)">${br.run_id}</div>
    </div>
  `;
}

function renderSummary(state) {
  const el = document.getElementById('summaryText');
  el.textContent = (state && state.summary) ? state.summary : 'No summary yet.';
}

function renderGoal(state) {
  const el = document.getElementById('goalText');
  el.textContent = (state && state.goal) ? state.goal : '--';
}

function escHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

async function fetchData() {
  try {
    const [evRes, stRes] = await Promise.all([
      fetch(`/api/events?run=${currentRun}`),
      fetch(`/api/state?run=${currentRun}`),
    ]);
    lastEvents = await evRes.json();
    lastState = await stRes.json();

    renderDashboard(lastState);
    renderHeaderStats(lastState);
    renderRuns(lastState);
    renderTimeline(lastEvents);
    renderQueue(lastState);
    renderHypotheses(lastState);
    renderGPUs(lastState);
    renderBestResult(lastState);
    renderSummary(lastState);
    renderGoal(lastState);
  } catch (err) {
    console.error('Fetch error:', err);
  }
}

async function fetchRuns() {
  try {
    const res = await fetch('/api/runs');
    const runs = await res.json();
    const sel = document.getElementById('runSelect');
    const cur = sel.value;
    sel.innerHTML = runs.map(r =>
      `<option value="${r.name}" ${r.name === cur ? 'selected' : ''}>${r.name}${r.is_current ? ' (live)' : ''} [${r.event_count} events]</option>`
    ).join('');
  } catch (err) {
    console.error('Runs fetch error:', err);
  }
}

// Keyboard: Escape closes modal
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

// Init
fetchData();
fetchRuns();
setInterval(fetchData, 15000);
setInterval(fetchRuns, 30000);
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description="Experiment trace viewer v2")
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
