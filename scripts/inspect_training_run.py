#!/usr/bin/env python
"""Print useful metrics from a LeRobot training output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a LeRobot/SmolVLA training run.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=Path("outputs/smolvla_panda_dualcam_state7_200_success_test"),
    )
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint_dir(args.run_dir)
    pretrained_dir = checkpoint_dir / "pretrained_model"
    training_state_dir = checkpoint_dir / "training_state"

    train_cfg = load_json(pretrained_dir / "train_config.json")
    policy_cfg = load_json(pretrained_dir / "config.json")
    scheduler = load_json(training_state_dir / "scheduler_state.json")
    training_step = load_json(training_state_dir / "training_step.json")

    print(f"run_dir: {args.run_dir}")
    print(f"checkpoint: {checkpoint_dir}")
    print()

    dataset = train_cfg["dataset"]
    print("Dataset")
    print(f"  repo_id: {dataset['repo_id']}")
    print(f"  root: {dataset['root']}")
    print(f"  frames from normalizer: {stat_count(pretrained_dir, 'observation.state'):.0f}")
    print()

    print("Training")
    print(f"  step: {training_step.get('step')}")
    print(f"  configured steps: {train_cfg.get('steps')}")
    print(f"  batch_size: {train_cfg.get('batch_size')}")
    print(f"  approximate samples seen: {int(train_cfg.get('steps', 0)) * int(train_cfg.get('batch_size', 0))}")
    frames = stat_count(pretrained_dir, "observation.state")
    if frames:
        seen = int(train_cfg.get("steps", 0)) * int(train_cfg.get("batch_size", 0))
        print(f"  approximate epochs: {seen / frames:.2f}")
    print(f"  optimizer: {train_cfg.get('optimizer', {}).get('type')}")
    print(f"  optimizer lr: {train_cfg.get('optimizer', {}).get('lr')}")
    print(f"  scheduler last_lr: {scheduler.get('_last_lr')}")
    print(f"  scheduler last_epoch: {scheduler.get('last_epoch')}")
    print()

    print("Policy")
    print(f"  type: {policy_cfg.get('type')}")
    print(f"  pretrained_path: {policy_cfg.get('pretrained_path')}")
    print(f"  input_features: {list(policy_cfg.get('input_features', {}).keys())}")
    print(f"  output_features: {policy_cfg.get('output_features')}")
    print(f"  chunk_size: {policy_cfg.get('chunk_size')}")
    print(f"  n_action_steps: {policy_cfg.get('n_action_steps')}")
    print(f"  freeze_vision_encoder: {policy_cfg.get('freeze_vision_encoder')}")
    print(f"  train_expert_only: {policy_cfg.get('train_expert_only')}")
    print()

    print_stats(pretrained_dir, "observation.state")
    print_stats(pretrained_dir, "action")


def resolve_checkpoint_dir(run_dir: Path) -> Path:
    if (run_dir / "pretrained_model").is_dir():
        return run_dir
    checkpoints = run_dir / "checkpoints"
    if (checkpoints / "last" / "pretrained_model").is_dir():
        return checkpoints / "last"
    candidates = sorted(p for p in checkpoints.iterdir() if (p / "pretrained_model").is_dir())
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {checkpoints}")
    return candidates[-1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_normalizer_stats(pretrained_dir: Path) -> dict:
    path = pretrained_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    return load_file(path)


def stat_count(pretrained_dir: Path, key: str) -> float:
    stats = load_normalizer_stats(pretrained_dir)
    tensor = stats.get(f"{key}.count")
    if tensor is None:
        return 0.0
    return float(tensor.flatten()[0].item())


def print_stats(pretrained_dir: Path, key: str) -> None:
    stats = load_normalizer_stats(pretrained_dir)
    print(f"{key} stats")
    for suffix in ("count", "mean", "std", "min", "max", "q10", "q50", "q90"):
        tensor = stats.get(f"{key}.{suffix}")
        if tensor is None:
            continue
        values = [round(float(v), 5) for v in tensor.flatten().tolist()]
        print(f"  {suffix}: {values}")
    print()


if __name__ == "__main__":
    main()
