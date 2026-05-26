#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-$PWD/../.conda-lerobot-smolvla}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LEROBOT_DIR="${LEROBOT_DIR:-$PWD/third_party/lerobot}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required for this setup script. Install Miniconda/Anaconda or install pyproject dependencies manually."
  exit 1
fi

conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip setuptools wheel

eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

mkdir -p "$(dirname "$LEROBOT_DIR")"
if [ ! -d "$LEROBOT_DIR/.git" ]; then
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi

python -m pip install -e "$LEROBOT_DIR[smolvla,training]"

echo
echo "Environment ready:"
echo "  conda activate $ENV_PREFIX"
echo "  cube-grasp-demo --out outputs/cube_grasp_demo.mp4"
echo "  cube-grasp-record --episodes 50 --repo-id local/panda_6dof_7ctrl"
