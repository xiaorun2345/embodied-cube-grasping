#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default to the successful dual-camera dataset recorded in this tutorial.
DATASET_REPO_ID="${DATASET_REPO_ID:-local/panda_6dof_7ctrl_dualcam_state7_200_success}"
DATASET_ROOT="${DATASET_ROOT:-outputs/lerobot_datasets_dualcam_state7}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_panda_dualcam_state7_200_success_test}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
LOCAL_POLICY_DIR="${LOCAL_POLICY_DIR:-outputs/pretrained/smolvla_panda_dualcam_state7_base}"
JOB_NAME="${JOB_NAME:-smolvla_panda_dualcam_state7_10_success_test}"
EPOCHS="${EPOCHS:-5}"
STEPS="${STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LOG_FREQ="${LOG_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
RESUME="${RESUME:-false}"
RESUME_CONFIG="${RESUME_CONFIG:-$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json}"
export DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR POLICY_PATH LOCAL_POLICY_DIR JOB_NAME EPOCHS STEPS BATCH_SIZE
export LOG_FREQ SAVE_FREQ EVAL_FREQ WANDB_ENABLE WANDB_PROJECT
export RESUME RESUME_CONFIG

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

case "${RESUME,,}" in
  1|true|yes|y) RESUME_ENABLED=1 ;;
  *) RESUME_ENABLED=0 ;;
esac

if [ -d "$OUTPUT_DIR" ] && [ "$RESUME_ENABLED" != "1" ]; then
  NEW_OUTPUT_DIR="${OUTPUT_DIR}_$(date +%Y%m%d_%H%M%S)"
  echo "Output directory already exists:"
  echo "  $OUTPUT_DIR"
  echo "Use a fresh output directory instead:"
  echo "  $NEW_OUTPUT_DIR"
  OUTPUT_DIR="$NEW_OUTPUT_DIR"
  export OUTPUT_DIR
fi

mkdir -p "$(dirname "$OUTPUT_DIR")" "$HF_HOME"

if [ "$RESUME_ENABLED" = "1" ] && [ ! -f "$RESUME_CONFIG" ]; then
  AUTO_RESUME_CONFIG="$(find "$OUTPUT_DIR/checkpoints" -path "*/pretrained_model/train_config.json" 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "$AUTO_RESUME_CONFIG" ]; then
    RESUME_CONFIG="$AUTO_RESUME_CONFIG"
    export RESUME_CONFIG
  else
    echo "Resume config not found:"
    echo "  $RESUME_CONFIG"
    echo
    echo "Use a checkpoint train_config.json, for example:"
    echo "  RESUME_CONFIG=outputs/smolvla_panda_dualcam_state7_200_success_test/checkpoints/010000/pretrained_model/train_config.json"
    exit 1
  fi
fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train not found. Activate the conda environment first:"
  echo "  conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla"
  exit 1
fi

if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  echo "Dataset metadata not found:"
  echo "  $PROJECT_ROOT/$DATASET_ROOT/meta/info.json"
  echo
  echo "Record a success-only dataset first:"
  echo "  MUJOCO_GL=egl cube-grasp-record \\"
  echo "    --episodes 200 \\"
  echo "    --steps 280 \\"
  echo "    --width 640 \\"
  echo "    --height 480 \\"
  echo "    --raw-dir outputs/cube_grasp_dualcam_state7_200_raw \\"
  echo "    --lerobot-root outputs/lerobot_datasets_dualcam_state7 \\"
  echo "    --repo-id local/panda_6dof_7ctrl_dualcam_state7_200_success \\"
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

DATASET_FRAMES="$(python - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os

dataset = LeRobotDataset(os.environ["DATASET_REPO_ID"], root=Path(os.environ["DATASET_ROOT"]))
print(dataset.meta.total_frames)
PY
)"
export DATASET_FRAMES

if [ -z "$STEPS" ]; then
  STEPS="$(python - <<'PY'
import math
import os

frames = int(os.environ["DATASET_FRAMES"])
epochs = float(os.environ["EPOCHS"])
batch_size = int(os.environ["BATCH_SIZE"])
print(math.ceil(frames * epochs / batch_size))
PY
)"
  STEPS_NOTE="computed from EPOCHS=$EPOCHS, frames=$DATASET_FRAMES, batch_size=$BATCH_SIZE"
else
  STEPS_NOTE="explicitly set by STEPS"
fi
export STEPS

echo
echo "Starting SmolVLA training:"
echo "  DATASET_REPO_ID=$DATASET_REPO_ID"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  POLICY_PATH=$POLICY_PATH"
echo "  LOCAL_POLICY_DIR=$LOCAL_POLICY_DIR"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  RESUME=$RESUME"
echo "  RESUME_CONFIG=$RESUME_CONFIG"
echo "  EPOCHS=$EPOCHS"
echo "  DATASET_FRAMES=$DATASET_FRAMES"
echo "  STEPS=$STEPS"
echo "  STEPS_NOTE=$STEPS_NOTE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  LOG_FREQ=$LOG_FREQ"
echo "  SAVE_FREQ=$SAVE_FREQ"
echo "  EVAL_FREQ=$EVAL_FREQ"
echo "  WANDB_ENABLE=$WANDB_ENABLE"
echo "  WANDB_PROJECT=$WANDB_PROJECT"
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

COMMON_ARGS=(
  --job_name="$JOB_NAME"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --log_freq="$LOG_FREQ"
  --save_freq="$SAVE_FREQ"
  --eval_freq="$EVAL_FREQ"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
)

if [ "$RESUME_ENABLED" = "1" ]; then
  lerobot-train \
    --config_path="$RESUME_CONFIG" \
    --resume=true \
    "${COMMON_ARGS[@]}"
else
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
    --resume=false \
    "${COMMON_ARGS[@]}"
fi
