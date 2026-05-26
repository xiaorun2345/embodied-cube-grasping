#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default to the 10 successful episodes recorded in this tutorial.
DATASET_REPO_ID="${DATASET_REPO_ID:-local/panda_6dof_7ctrl_10_success}"
DATASET_ROOT="${DATASET_ROOT:-outputs/lerobot_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_panda_10_success_test}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
JOB_NAME="${JOB_NAME:-smolvla_panda_10_success_test}"
STEPS="${STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
export DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR POLICY_PATH JOB_NAME STEPS BATCH_SIZE

# Keep Hugging Face cache inside the project so training can run from restricted shells too.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/$DATASET_ROOT}"

# Optional mirror. Override with HF_ENDPOINT=https://huggingface.co if you do not want the mirror.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$OUTPUT_DIR" "$HF_HOME"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train not found. Activate the conda environment first:"
  echo "  conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla"
  exit 1
fi

if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  echo "Dataset metadata not found:"
  echo "  $PROJECT_ROOT/$DATASET_ROOT/meta/info.json"
  echo
  echo "Record the 10-success dataset first:"
  echo "  MUJOCO_GL=egl cube-grasp-record \\"
  echo "    --episodes 10 \\"
  echo "    --steps 280 \\"
  echo "    --width 640 \\"
  echo "    --height 480 \\"
  echo "    --raw-dir outputs/cube_grasp_success_10_raw \\"
  echo "    --repo-id local/panda_6dof_7ctrl_10_success \\"
  echo "    --success-only"
  exit 1
fi

python - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os

repo_id = os.environ["DATASET_REPO_ID"]
root = Path(os.environ["DATASET_ROOT"])
dataset = LeRobotDataset(repo_id, root=root)

print("Dataset check:")
print(f"  repo_id: {repo_id}")
print(f"  root: {root}")
print(f"  episodes: {dataset.meta.total_episodes}")
print(f"  frames: {dataset.meta.total_frames}")
print(f"  features: {list(dataset.features.keys())}")
PY

echo
echo "Starting SmolVLA training:"
echo "  DATASET_REPO_ID=$DATASET_REPO_ID"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  POLICY_PATH=$POLICY_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  STEPS=$STEPS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  HF_HOME=$HF_HOME"
echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1, stop before lerobot-train."
  exit 0
fi

lerobot-train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$POLICY_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --wandb.enable=false
