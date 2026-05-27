from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

from .env import CubeGraspEnv, TASK_DESCRIPTION


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LeRobot policy in the MuJoCo cube-grasp env.")
    parser.add_argument("--policy-path", required=True, help="Path or Hub id produced by lerobot-train.")
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("outputs/cube_grasp_policy_eval.mp4"))
    args = parser.parse_args()

    policy = _load_policy(args.policy_path)
    policy.eval()
    env = CubeGraspEnv(width=640, height=480, seed=args.seed, max_steps=args.steps)
    obs, info = env.reset(seed=args.seed)
    frames = []

    try:
        for _ in range(args.steps):
            action = _policy_action(policy, obs)
            obs, _, terminated, truncated, info = env.step(action)
            frames.append(obs["observation.image"])
            if terminated or truncated:
                break
    finally:
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, frames, fps=30)
    print(f"wrote {args.out}")
    print(f"cube_on_goal={info['cube_on_goal']} cube_lifted={info['cube_lifted']}")


def _load_policy(policy_path: str) -> Any:
    try:
        from lerobot.common.policies.factory import make_policy
        from lerobot.configs import parser as lerobot_parser
        from lerobot.configs.train import TrainPipelineConfig

        cfg = lerobot_parser.get_path_arg("policy.path", policy_path)
        train_cfg = TrainPipelineConfig(policy=cfg)
        return make_policy(train_cfg= train_cfg, ds_meta=None)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the policy through LeRobot's factory API. "
            "Use this script with the same LeRobot checkout used for training, or run "
            "`lerobot-eval` if your installed version provides it."
        ) from exc


def _policy_action(policy: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
    batch = {
        "observation.images.front": torch.from_numpy(obs["observation.images.front"]).permute(2, 0, 1).float()[None]
        / 255.0,
        "observation.images.top_oblique": torch.from_numpy(obs["observation.images.top_oblique"])
        .permute(2, 0, 1)
        .float()[None]
        / 255.0,
        "observation.state": torch.from_numpy(obs["observation.state"]).float()[None],
        "task": [TASK_DESCRIPTION],
    }
    with torch.no_grad():
        if hasattr(policy, "select_action"):
            out = policy.select_action(batch)
        else:
            out = policy(batch)
    if isinstance(out, dict):
        out = out.get("action", out.get("actions"))
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float32).reshape(-1)[:7].clip(-1.0, 1.0)


if __name__ == "__main__":
    main()
