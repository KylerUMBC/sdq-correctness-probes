#!/usr/bin/env bash
# SDQ RunPod Setup Script
# Run once after attaching a new pod to your network volume.
# Prerequisites:
#   - RunPod pod with a network volume mounted at /workspace
#   - GITHUB_TOKEN: personal access token with repo read scope (private repo)
#   - HF_TOKEN: HuggingFace token; must have accepted Gemma 2 license at huggingface.co/google/gemma-2-2b
#   - Recommended pod: 24GB VRAM (RTX 3090/4090 or A5000), 50GB+ network volume
#
# Usage:
#   export HF_TOKEN=hf_...
#   export GITHUB_TOKEN=ghp_...
#   bash setup_pod.sh

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/SDQ-v1"
MODEL_CACHE="$WORKSPACE/models/gemma-2-2b"

# ── 1. Verify tokens ──────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Export it before running: export HF_TOKEN=hf_..."
    exit 1
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "ERROR: GITHUB_TOKEN is not set."
    echo "Export it before running: export GITHUB_TOKEN=ghp_..."
    exit 1
fi

REPO_URL="https://${GITHUB_TOKEN}@github.com/KylerUMBC/SDQ-v1.git"

# ── 2. Clone or update repo ───────────────────────────────────────────────────
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "Repo already exists — pulling latest..."
    git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
    git -C "$REPO_DIR" pull
else
    echo "Cloning SDQ repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu124

# ── 4. Create persistent directories on network volume ────────────────────────
mkdir -p "$MODEL_CACHE"
mkdir -p "$WORKSPACE/data/runs"

# ── 5. Symlink network-volume paths into repo so config paths resolve ─────────
#    configs/model.yaml uses ./models/gemma 2 2B and ./data/runs
#    We symlink these to the network volume so data survives pod restarts.
ln -sfn "$MODEL_CACHE" "$REPO_DIR/models/gemma 2 2B"
ln -sfn "$WORKSPACE/data/runs" "$REPO_DIR/data/runs"

# ── 6. Write .env ─────────────────────────────────────────────────────────────
cat > "$REPO_DIR/.env" <<ENV
HF_TOKEN=${HF_TOKEN}
ENV

# ── 7. Download Gemma 2 2B ────────────────────────────────────────────────────
# Skip if weights already present on the network volume.
if [[ -f "$MODEL_CACHE/config.json" ]]; then
    echo "Model already cached at $MODEL_CACHE — skipping download."
else
    echo "Downloading Gemma 2 2B (this takes a few minutes)..."
    python - <<PYEOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="google/gemma-2-2b",
    local_dir="$MODEL_CACHE",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
print("Download complete.")
PYEOF
fi

# ── 8. Smoke test ─────────────────────────────────────────────────────────────
echo "Running smoke test (non-integration tests only)..."
python -m pytest tests/ -x -q --ignore=tests/test_integration.py 2>&1 | tail -5

echo ""
echo "Setup complete."
echo "  Repo:   $REPO_DIR"
echo "  Model:  $MODEL_CACHE"
echo "  Runs:   $WORKSPACE/data/runs"
echo ""
echo "Next step — re-capture hidden states on GPU:"
echo "  cd $REPO_DIR && python capture_runs.py"
