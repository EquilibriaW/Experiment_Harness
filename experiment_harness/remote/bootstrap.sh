#!/usr/bin/env bash
# Bootstrap for RunPod: installs agent CLIs, Python deps, tools.
set -euo pipefail

echo "=== Experiment Harness v2 Bootstrap ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

# ── System deps ─────────────────────────────────────────────
apt-get update -qq && apt-get install -y -qq tmux jq curl wget git > /dev/null

# ── Python deps ─────────────────────────────────────────────
# --break-system-packages for PEP 668 (Ubuntu 24.04+)
pip install --quiet --upgrade pip --break-system-packages
pip install --quiet --break-system-packages \
    pytest \
    pynvml \
    torch torchvision torchaudio \
    numpy \
    transformers \
    datasets \
    accelerate \
    wandb \
    tensorboard \
    openai-agents

# ── Deno (required by dspy.RLM for sandboxed REPL) ─────────
if ! command -v deno &> /dev/null; then
    echo "Installing Deno..."
    curl -fsSL https://deno.land/install.sh | sh > /dev/null 2>&1
    export DENO_INSTALL="$HOME/.deno"
    export PATH="$DENO_INSTALL/bin:$PATH"
    echo 'export DENO_INSTALL="$HOME/.deno"' >> ~/.bashrc
    echo 'export PATH="$DENO_INSTALL/bin:$PATH"' >> ~/.bashrc
fi

# ── Node.js ─────────────────────────────────────────────────
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
    apt-get install -y -qq nodejs > /dev/null
fi

# ── Codex CLI ───────────────────────────────────────────────
if ! command -v codex &> /dev/null; then
    echo "Installing Codex CLI..."
    npm install -g @openai/codex 2>/dev/null || echo "WARN: Codex install failed"
fi

# ── Claude Code CLI ─────────────────────────────────────────
if ! command -v claude &> /dev/null; then
    echo "Installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code 2>/dev/null || echo "WARN: Claude install failed"
fi

# ── Auth check ──────────────────────────────────────────────
echo ""
echo "Auth status:"
if [ -d "$HOME/.claude" ] && [ "$(ls -A $HOME/.claude 2>/dev/null)" ]; then
    echo "  Claude: auth config found"
else
    echo "  Claude: NO auth config"
fi
if [ -d "$HOME/.codex" ] && [ "$(ls -A $HOME/.codex 2>/dev/null)" ]; then
    echo "  Codex:  auth config found"
elif [ -d "$HOME/.config/openai" ] && [ "$(ls -A $HOME/.config/openai 2>/dev/null)" ]; then
    echo "  Codex:  auth config found (in .config/openai)"
else
    echo "  Codex:  NO auth config"
fi

# ── Workspace ───────────────────────────────────────────────
mkdir -p /workspace/experiment/run_logs
mkdir -p /workspace/experiment/logs
mkdir -p /workspace/harness

echo ""
echo "=== Bootstrap Complete ==="
echo "Node:   $(node --version 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "pytest: $(python -m pytest --version 2>/dev/null || echo 'N/A')"
echo "codex:  $(codex --version 2>/dev/null || echo 'N/A')"
echo "claude: $(claude --version 2>/dev/null || echo 'N/A')"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"

# ── Experiment-specific setup (if provided) ────────────────
if [ -f /workspace/experiment/setup.sh ]; then
    echo "=== Running experiment setup ==="
    bash /workspace/experiment/setup.sh
fi
