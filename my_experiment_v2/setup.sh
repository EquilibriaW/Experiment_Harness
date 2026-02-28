#!/usr/bin/env bash
# Setup script for k-offset causal LM experiment (nanochat overlay).
# Runs on the pod after bootstrap.sh, inside /workspace/experiment.
set -euo pipefail

EXPERIMENT_DIR="/workspace/experiment"
NANOCHAT_SRC="/tmp/nanochat_src"

echo "=== K-Offset Experiment Setup ==="

# ── 1. Clone nanochat ────────────────────────────────────────
if [ ! -d "$NANOCHAT_SRC" ]; then
    echo "[setup] Cloning nanochat..."
    git clone --depth 1 https://github.com/karpathy/nanochat.git "$NANOCHAT_SRC"
else
    echo "[setup] nanochat already cloned"
fi

# ── 2. Overlay: copy nanochat into experiment dir (no clobber) ─
echo "[setup] Overlaying nanochat into $EXPERIMENT_DIR (experiment files win)..."
cp -rn "$NANOCHAT_SRC"/. "$EXPERIMENT_DIR"/ 2>/dev/null || true

# ── 3. Install nanochat dependencies ──────────────────────────
# nanochat has a flat layout that confuses setuptools, so we install
# deps directly rather than `pip install -e .`. The train script runs
# from $EXPERIMENT_DIR so imports resolve via cwd.
cd "$EXPERIMENT_DIR"
echo "[setup] Installing nanochat dependencies..."
pip install --quiet --break-system-packages \
    "datasets>=4.0.0" "fastapi>=0.117.1" "psutil>=7.1.0" \
    "python-dotenv>=1.2.1" "regex>=2025.9.1" "rustbpe>=0.1.0" \
    "scipy>=1.15.3" "tabulate>=0.9.0" "tiktoken>=0.11.0" \
    "tokenizers>=0.22.0" "transformers>=4.57.3" "wandb>=0.21.3" \
    "zstandard>=0.25.0" "matplotlib>=3.10.8"
# Ensure experiment dir is on PYTHONPATH for imports
export PYTHONPATH="$EXPERIMENT_DIR:${PYTHONPATH:-}"
echo "export PYTHONPATH=\"$EXPERIMENT_DIR:\${PYTHONPATH:-}\"" >> ~/.bashrc

# ── 4. Apply patches ─────────────────────────────────────────
if [ -d "$EXPERIMENT_DIR/patches" ]; then
    echo "[setup] Applying patches..."
    for patchfile in "$EXPERIMENT_DIR"/patches/*.diff "$EXPERIMENT_DIR"/patches/*.patch; do
        [ -f "$patchfile" ] || continue
        echo "  Applying $(basename "$patchfile")..."
        patch -p0 < "$patchfile"
    done
    echo "[setup] Patches applied."
else
    echo "[setup] No patches/ directory found, skipping."
fi

# ── 5. Git init for apply_code_change rollback ────────────────
echo "[setup] Initializing git repo for rollback support..."
cd "$EXPERIMENT_DIR"
git init
git config user.email "experiment@harness"
git config user.name "Experiment Harness"
git add -A
git commit -m "Initial state: nanochat + overlay + patches"

# ── 6. Bootstrap data ────────────────────────────────────────
echo "[setup] Bootstrapping nanochat data..."
python -m scripts.bootstrap_nanochat_data --num-shards 4 --workers 4 --train-tokenizer

# ── 7. Smoke test ─────────────────────────────────────────────
echo "[setup] Running smoke test..."
python train.py --run smoke --num-iterations 3 --k-max 4

echo "=== K-Offset Experiment Setup Complete ==="
