#!/bin/bash
# Lambda Cloud training script for IVF-Bench Qwen 3.5-9B VLM ORPO sweep.
#
# Usage on a fresh Lambda instance (2x H100):
#   scp -i ~/.ssh/<your-key>.pem \
#       configs/training_orpo_qwen_vlm_sweep.yaml \
#       scripts/lambda_train.sh \
#       ubuntu@<IP>:~/
#   ssh -i ~/.ssh/<your-key>.pem ubuntu@<IP>
#   bash lambda_train.sh <WANDB_API_KEY> <HF_TOKEN> [config_filename]
#
# Default config: training_orpo_qwen_vlm_sweep.yaml

set -euo pipefail

WANDB_KEY="${1:?Usage: bash lambda_train.sh <WANDB_API_KEY> <HF_TOKEN> [config]}"
HF_TOKEN="${2:?Usage: bash lambda_train.sh <WANDB_API_KEY> <HF_TOKEN> [config]}"
CONFIG="${3:-training_orpo_qwen_vlm_sweep.yaml}"

echo "=== System info ==="
uname -a
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 --version

echo
echo "=== Installing aitraining (VLM-patched, 0.0.53+) + deps ==="
pip install --user -q --upgrade pip
# numpy<2 pinned to avoid Ubuntu-system torch NumPy-1.x ABI mismatch.
pip install --user -q 'aitraining>=0.0.53' wandb optuna pillow 'numpy<2'
export PATH="$HOME/.local/bin:$PATH"

echo
echo "=== Patching aitraining FIELD_SCOPES (image_column, hub_private) ==="
# Upstream bug as of 0.0.53: VLM params not registered, CLI rejects them with
# enforce_scope=True. Apply ephemeral fix until next release.
python3 - <<'PYPATCH'
import autotrain, os, sys
p = os.path.join(os.path.dirname(autotrain.__file__), "cli", "run_llm.py")
src = open(p).read()
anchor = '    "repo_id": ["all"],\n    "wandb_token": ["all"],\n    "unsloth": ["all"],'
patch = '    "repo_id": ["all"],\n    "hub_private": ["all"],\n    "wandb_token": ["all"],\n    "unsloth": ["all"],\n    # VLM preference training (ORPO/DPO with images)\n    "image_column": ["dpo", "orpo"],'
if "hub_private" in src and '"image_column":' in src:
    print("  already patched")
elif anchor in src:
    open(p, "w").write(src.replace(anchor, patch))
    print("  patched OK")
else:
    print("  WARNING: anchor not found, manual patch needed", file=sys.stderr); sys.exit(1)
PYPATCH

echo
echo "=== Writable AUTOTRAIN_PROJECTS_DIR ==="
# Avoids PermissionError when aitraining tries to mkdir /home/trainings.
export AUTOTRAIN_PROJECTS_DIR="$HOME/trainings"
mkdir -p "$AUTOTRAIN_PROJECTS_DIR"

echo
echo "=== Login: wandb + HF ==="
export WANDB_API_KEY="$WANDB_KEY"
export HF_TOKEN="$HF_TOKEN"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
wandb login --relogin "$WANDB_KEY" 2>/dev/null
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

echo
echo "=== Smoke-test dataset access ==="
python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("monostate/ivf-bench-orpo-qwen9b", split="train[:1]")
row = ds[0]
img = row["images"][0]
print(f"  dataset OK: img={img.size} prompt_chars={len(row['prompt'][0]['content'])} chosen_chars={len(row['chosen'][0]['content'])}")
PY

echo
echo "=== Starting ORPO sweep ==="
echo "Config: $CONFIG"
echo "Base model: Qwen/Qwen3.5-9B (unified multimodal)"
echo "Dataset: monostate/ivf-bench-orpo-qwen9b (500 train / 50 eval, 550 images)"
echo
nvidia-smi --query-gpu=memory.used --format=csv

aitraining --config "$CONFIG"

echo
echo "=== Training complete ==="
ls -la ivf-bench-qwen9b-orpo-sweep/ 2>/dev/null || echo "Check output directory"
