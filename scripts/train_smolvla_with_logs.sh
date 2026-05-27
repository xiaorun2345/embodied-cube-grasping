#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p outputs/train_logs
LOG_FILE="${LOG_FILE:-outputs/train_logs/smolvla_$(date +%Y%m%d_%H%M%S).log}"

echo "Training log will be written to:"
echo "  $LOG_FILE"
echo

LOG_FREQ="${LOG_FREQ:-10}" \
WANDB_ENABLE="${WANDB_ENABLE:-false}" \
bash scripts/train_smolvla_10_success.sh 2>&1 | tee "$LOG_FILE"
