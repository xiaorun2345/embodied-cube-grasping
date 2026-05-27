#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "$HF_HOME"

echo "Download SmolVLA pretrained models"
echo "  HF_HOME=$HF_HOME"
echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo "  HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo

python - <<'PY'
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]

downloads = [
    (
        "lerobot/smolvla_base",
        [
            "config.json",
            "model.safetensors",
            "policy_*.json",
            "policy_*.safetensors",
        ],
    ),
    (
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        [
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.json",
            "merges.txt",
            "vocab.json",
        ],
    ),
]

for repo_id, allow_patterns in downloads:
    print(f"\n==> downloading {repo_id}")
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    print(f"cached at: {path}")

print("\nDone. You can now run:")
print("  bash scripts/train_smolvla_10_success.sh")
PY
