#!/usr/bin/env bash
# Bootstrap script for RunPod: installs agent CLIs, Python deps, and tools.
# Auth: CLI tools use subscription auth forwarded from your laptop.
# No API keys needed — the orchestrator copies your local auth configs.
set -euo pipefail

echo "=== Experiment Harness Bootstrap ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# ── System deps ─────────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq tmux jq curl wget git > /dev/null

# ── Python deps ─────────────────────────────────────────────
pip install --quiet --upgrade pip
pip install --quiet \
    pytest \
    pynvml \
    torch torchvision torchaudio \
    numpy \
    transformers \
    datasets \
    accelerate \
    wandb \
    tensorboard

# ── Node.js (needed for Codex + Claude CLIs) ────────────────
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt-get install -y -qq nodejs > /dev/null
fi

# ── Codex CLI ───────────────────────────────────────────────
if ! command -v codex &> /dev/null; then
    echo "Installing Codex CLI..."
    npm install -g @openai/codex 2>/dev/null || echo "WARN: Codex CLI install failed"
fi

# ── Claude Code CLI ─────────────────────────────────────────
if ! command -v claude &> /dev/null; then
    echo "Installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code 2>/dev/null || echo "WARN: Claude CLI install failed"
fi

# ── Auth check ──────────────────────────────────────────────
# Auth configs should be forwarded by the orchestrator before this runs.
# If they weren't, the loop will fail on the first agent call.
echo ""
echo "Auth status:"
if [ -d "$HOME/.claude" ] && [ "$(ls -A $HOME/.claude 2>/dev/null)" ]; then
    echo "  Claude: auth config found"
else
    echo "  Claude: NO auth config — run 'claude login' or use --forward-auth"
fi
if [ -d "$HOME/.codex" ] && [ "$(ls -A $HOME/.codex 2>/dev/null)" ]; then
    echo "  Codex:  auth config found"
elif [ -d "$HOME/.config/openai" ] && [ "$(ls -A $HOME/.config/openai 2>/dev/null)" ]; then
    echo "  Codex:  auth config found (in .config/openai)"
else
    echo "  Codex:  NO auth config — run 'codex login' or use --forward-auth"
fi

# ── Setup workspace ────────────────────────────────────────
mkdir -p /workspace/experiment/metrics
mkdir -p /workspace/harness

echo ""
echo "=== Bootstrap Complete ==="
echo "Node:   $(node --version 2>/dev/null || echo 'not installed')"
echo "Python: $(python --version)"
echo "pytest: $(python -m pytest --version 2>/dev/null || echo 'not installed')"
echo "codex:  $(codex --version 2>/dev/null || echo 'not installed')"
echo "claude: $(claude --version 2>/dev/null || echo 'not installed')"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
