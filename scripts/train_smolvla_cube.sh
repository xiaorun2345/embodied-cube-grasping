#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-local/panda_6dof_7ctrl_dualcam_state7}"
DATASET_ROOT="${DATASET_ROOT:-outputs/lerobot_datasets_dualcam_state7}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_panda_dualcam_state7}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
LOCAL_POLICY_DIR="${LOCAL_POLICY_DIR:-outputs/pretrained/smolvla_panda_dualcam_state7_base}"
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
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

mkdir -p "$(dirname "$OUTPUT_DIR")"

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
  --job_name="smolvla_panda_6dof_7ctrl" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --wandb.enable=false
