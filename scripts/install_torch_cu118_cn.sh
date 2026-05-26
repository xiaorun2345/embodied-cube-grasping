#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/mkls/xiao_run/.conda-lerobot-smolvla/bin/python}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/home/mkls/xiao_run/.cache/pip}"

"$PYTHON_BIN" -m pip install --force-reinstall --timeout 120 --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu118/ \
  "torch==2.7.1+cu118" \
  "torchvision==0.22.1+cu118"

"$PYTHON_BIN" -m pip install --force-reinstall --timeout 120 --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "numpy==2.2.6" \
  "fsspec==2026.2.0" \
  "setuptools==80.10.2"

"$PYTHON_BIN" -m pip install --force-reinstall --timeout 120 --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "nvidia-cublas-cu11==11.11.3.6" \
  "nvidia-cuda-cupti-cu11==11.8.87" \
  "nvidia-cuda-nvrtc-cu11==11.8.89" \
  "nvidia-cuda-runtime-cu11==11.8.89" \
  "nvidia-cudnn-cu11==9.1.0.70" \
  "nvidia-cufft-cu11==10.9.0.58" \
  "nvidia-curand-cu11==10.3.0.86" \
  "nvidia-cusolver-cu11==11.4.1.48" \
  "nvidia-cusparse-cu11==11.7.5.86" \
  "nvidia-nccl-cu11==2.21.5" \
  "nvidia-nvtx-cu11==11.8.86"

"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" - <<'PY'
import torch

print("torch", torch.__version__)
print("torch_cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
