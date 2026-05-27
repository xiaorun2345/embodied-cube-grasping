from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

from .env import CubeGraspEnv, TASK_DESCRIPTION


DEFAULT_OUTPUT_ROOT = Path("outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained LeRobot policy in the MuJoCo cube-grasp env.")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Path to a checkpoint pretrained_model directory. Default: latest outputs/*/checkpoints/*/pretrained_model.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", choices=["front", "top_oblique"], default="front")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--out", type=Path, default=Path("outputs/cube_grasp_policy_eval.mp4"))
    parser.add_argument("--viewer", action="store_true", help="Show the MuJoCo viewer while running.")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    configure_offline_hf_cache()
    policy_path = resolve_policy_path(args.policy_path)
    device = resolve_device(args.device)

    print(f"policy_path: {policy_path}")
    print(f"device: {device}")

    policy, preprocessor, postprocessor = load_lerobot_policy(policy_path, device=device)
    env = CubeGraspEnv(width=args.width, height=args.height, camera=args.camera, seed=args.seed, max_steps=args.steps)

    try:
        if args.viewer:
            run_with_viewer(env, policy, preprocessor, postprocessor, args)
        else:
            run_to_video(env, policy, preprocessor, postprocessor, args)
    finally:
        env.close()


def configure_offline_hf_cache() -> None:
    project_root = Path(__file__).resolve().parents[2]
    hf_home = project_root / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", os.environ["HF_HUB_OFFLINE"])


def resolve_policy_path(policy_path: Path | None) -> Path:
    if policy_path is not None:
        path = policy_path
        if path.name != "pretrained_model" and (path / "pretrained_model").is_dir():
            path = path / "pretrained_model"
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Policy config not found: {path / 'config.json'}")
        return path

    candidates = sorted(DEFAULT_OUTPUT_ROOT.glob("*/checkpoints/*/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(
            "No trained policy found under outputs/*/checkpoints/*/pretrained_model. "
            "Pass --policy-path explicitly."
        )
    return candidates[-1]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lerobot_policy(policy_path: Path, *, device: str):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    cfg.device = device

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(
        policy_path,
        config=cfg,
        local_files_only=True,
        strict=False,
    )
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "device_processor": {"device": "cpu"},
        },
    )
    return policy, preprocessor, postprocessor


def run_to_video(env: CubeGraspEnv, policy: Any, preprocessor: Any, postprocessor: Any, args: argparse.Namespace) -> None:
    frames: list[np.ndarray] = []
    results = []

    for episode in range(args.episodes):
        if hasattr(policy, "reset"):
            policy.reset()
        obs, info = env.reset(seed=args.seed + episode)
        lifted_once = False

        for _step in range(args.steps):
            action = select_action(policy, preprocessor, postprocessor, obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            lifted_once = lifted_once or bool(info["cube_lifted"])
            frames.append(obs["observation.image"])
            if terminated or truncated:
                break

        success = bool(lifted_once and info["cube_on_goal"])
        results.append(success)
        print(
            f"episode {episode}: success={success} "
            f"lifted_once={lifted_once} cube_on_goal={info['cube_on_goal']} step={info['step']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, frames, fps=int(args.fps))
    print(f"wrote {args.out}")
    print(f"success: {sum(results)}/{len(results)}")


def run_with_viewer(env: CubeGraspEnv, policy: Any, preprocessor: Any, postprocessor: Any, args: argparse.Namespace) -> None:
    import mujoco
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        configure_viewer_camera(viewer, env, args.camera)
        for episode in range(args.episodes):
            if not viewer.is_running():
                break
            if hasattr(policy, "reset"):
                policy.reset()
            obs, info = env.reset(seed=args.seed + episode)
            lifted_once = False

            start = time.perf_counter()
            for step in range(args.steps):
                if not viewer.is_running():
                    break
                action = select_action(policy, preprocessor, postprocessor, obs)
                obs, _reward, terminated, truncated, info = env.step(action)
                lifted_once = lifted_once or bool(info["cube_lifted"])
                viewer.sync()

                target_time = start + (step + 1) / args.fps
                sleep_time = target_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if terminated or truncated:
                    break

            success = bool(lifted_once and info["cube_on_goal"])
            print(
                f"episode {episode}: success={success} "
                f"lifted_once={lifted_once} cube_on_goal={info['cube_on_goal']} step={info['step']}"
            )


def select_action(policy: Any, preprocessor: Any, postprocessor: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
    batch = {
        "observation.images.front": torch.from_numpy(obs["observation.images.front"])
        .permute(2, 0, 1)
        .float()
        / 255.0,
        "observation.images.top_oblique": torch.from_numpy(obs["observation.images.top_oblique"])
        .permute(2, 0, 1)
        .float()
        / 255.0,
        "observation.state": torch.from_numpy(obs["observation.state"]).float(),
        "task": TASK_DESCRIPTION,
    }
    batch = preprocessor(batch)
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = postprocessor(action)
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)[:7].clip(-1.0, 1.0)


def configure_viewer_camera(viewer: Any, env: CubeGraspEnv, camera: str) -> None:
    camera_id = env.mujoco.mj_name2id(env.model, env.mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        raise ValueError(f"Camera not found: {camera}")
    viewer.cam.type = env.mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


if __name__ == "__main__":
    main()
