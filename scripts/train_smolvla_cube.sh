#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-local/panda_6dof_7ctrl_dualcam_state7}"
DATASET_ROOT="${DATASET_ROOT:-outputs/lerobot_datasets_dualcam_state7}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_panda_dualcam_state7}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
LOCAL_POLICY_DIR="${LOCAL_POLICY_DIR:-outputs/pretrained/smolvla_panda_dualcam_state7_base}"
JOB_NAME="${JOB_NAME:-smolvla_panda_6dof_7ctrl}"
EPOCHS="${EPOCHS:-5}"
STEPS="${STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"
RESUME="${RESUME:-false}"
RESUME_CONFIG="${RESUME_CONFIG:-$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json}"
export DATASET_REPO_ID DATASET_ROOT OUTPUT_DIR POLICY_PATH LOCAL_POLICY_DIR JOB_NAME EPOCHS STEPS BATCH_SIZE
export LOG_FREQ SAVE_FREQ EVAL_FREQ WANDB_ENABLE WANDB_PROJECT RESUME RESUME_CONFIG
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
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

mkdir -p "$(dirname "$OUTPUT_DIR")"

if [ "$RESUME_ENABLED" = "1" ] && [ ! -f "$RESUME_CONFIG" ]; then
  AUTO_RESUME_CONFIG="$(find "$OUTPUT_DIR/checkpoints" -path "*/pretrained_model/train_config.json" 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "$AUTO_RESUME_CONFIG" ]; then
    RESUME_CONFIG="$AUTO_RESUME_CONFIG"
    export RESUME_CONFIG
  else
    echo "Resume config not found:"
    echo "  $RESUME_CONFIG"
    exit 1
  fi
fi

if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  echo "Dataset metadata not found:"
  echo "  $PWD/$DATASET_ROOT/meta/info.json"
  exit 1
fi

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

echo "Starting SmolVLA training:"
echo "  DATASET_REPO_ID=$DATASET_REPO_ID"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  RESUME=$RESUME"
echo "  RESUME_CONFIG=$RESUME_CONFIG"
echo "  EPOCHS=$EPOCHS"
echo "  DATASET_FRAMES=$DATASET_FRAMES"
echo "  STEPS=$STEPS"
echo "  STEPS_NOTE=$STEPS_NOTE"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo

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
