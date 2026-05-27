#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default to the 10 successful episodes recorded in this tutorial.
DATASET_REPO_ID="${DATASET_REPO_ID:-local/panda_6dof_7ctrl_dualcam_state7_200_success}"
DATASET_ROOT="${DATASET_ROOT:-outputs/lerobot_datasets_dualcam_state7}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_panda_dualcam_state7_200_success_test}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
LOCAL_POLICY_DIR="${LOCAL_POLICY_DIR:-outputs/pretrained/smolvla_panda_dualcam_state7_base}"
JOB_NAME="${JOB_NAME:-smolvla_panda_dualcam_state7_10_success_test}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
export DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR POLICY_PATH LOCAL_POLICY_DIR JOB_NAME STEPS BATCH_SIZE

# Keep Hugging Face cache inside the project so training can run from restricted shells too.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/$DATASET_ROOT}"

# Use the official endpoint by default. If your network can access a mirror, override it:
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/train_smolvla_10_success.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$HF_HUB_OFFLINE}"

if [ -d "$OUTPUT_DIR" ] && [ "${RESUME:-0}" != "1" ]; then
  NEW_OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
  echo "Output directory already exists:"
  echo "  $OUTPUT_DIR"
  echo "Use a fresh output directory instead:"
  echo "  $NEW_OUTPUT_DIR"
  OUTPUT_DIR="$NEW_OUTPUT_DIR"
  export OUTPUT_DIR
fi

mkdir -p "$(dirname "$OUTPUT_DIR")" "$HF_HOME"

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
  echo "    --raw-dir outputs/cube_grasp_dualcam_state7_10_raw \\"
  echo "    --lerobot-root outputs/lerobot_datasets_dualcam_state7 \\"
  echo "    --repo-id local/panda_6dof_7ctrl_dualcam_state7_10_success \\"
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
echo "  LOCAL_POLICY_DIR=$LOCAL_POLICY_DIR"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  STEPS=$STEPS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  HF_HOME=$HF_HOME"
echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo "  HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1, stop before lerobot-train."
  exit 0
fi

TRAIN_POLICY_PATH="$POLICY_PATH"
if [ "$POLICY_PATH" = "lerobot/smolvla_base" ]; then
  TRAIN_POLICY_PATH="$(python scripts/prepare_smolvla_panda_policy.py --source "$POLICY_PATH" --output "$LOCAL_POLICY_DIR")"
fi

lerobot-train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$TRAIN_POLICY_PATH" \
  --policy.push_to_hub=false \
  --output_dir="$OUTPUT_DIR" \
  --resume="${RESUME:-false}" \
  --job_name="$JOB_NAME" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --wandb.enable=false
