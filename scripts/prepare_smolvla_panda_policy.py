#!/usr/bin/env python
"""Prepare a local SmolVLA policy config for this MuJoCo Panda dataset.

The public lerobot/smolvla_base checkpoint is configured for 3 cameras,
6 state values and 6 action values. This demo dataset has:
  - observation.images.front
  - observation.images.top_oblique
  - observation.state with 7 values
  - action with 7 values

This script creates a small local policy directory with a patched config.json
and symlinks to the downloaded checkpoint files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="lerobot/smolvla_base")
    parser.add_argument("--output", type=Path, default=Path("outputs/pretrained/smolvla_panda_dualcam_state7_base"))
    args = parser.parse_args()

    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME")
    snapshot = Path(
        snapshot_download(
            repo_id=args.source,
            cache_dir=cache_dir,
            local_files_only=True,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "policy_*.json",
                "policy_*.safetensors",
            ],
        )
    )

    args.output.mkdir(parents=True, exist_ok=True)

    for src in snapshot.iterdir():
        if src.name == "config.json" or not src.is_file():
            continue
        dst = args.output / src.name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            raise FileExistsError(f"Refuse to overwrite existing file: {dst}")
        dst.symlink_to(src.resolve())

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config["input_features"] = {
        "observation.images.front": {
            "type": "VISUAL",
            "shape": [3, 480, 640],
        },
        "observation.images.top_oblique": {
            "type": "VISUAL",
            "shape": [3, 480, 640],
        },
        "observation.state": {
            "type": "STATE",
            "shape": [7],
        },
    }
    config["output_features"] = {
        "action": {
            "type": "ACTION",
            "shape": [7],
        }
    }
    config["push_to_hub"] = False
    config["repo_id"] = None

    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
