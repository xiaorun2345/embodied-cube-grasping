#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-local/panda_6dof_7ctrl}"
DATASET_ROOT="${DATASET_ROOT:-outputs/lerobot_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_panda_6dof_7ctrl}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "$OUTPUT_DIR"

lerobot-train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --policy.path="$POLICY_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="smolvla_panda_6dof_7ctrl" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --wandb.enable=false
