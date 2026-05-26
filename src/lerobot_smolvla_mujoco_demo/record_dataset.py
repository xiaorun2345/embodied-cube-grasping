from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange

from .env import ACTION_NAMES, STATE_NAMES, TASK_DESCRIPTION, CubeGraspEnv
from .scripted_grasp import scripted_grasp_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Record scripted cube-grasp demonstrations.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", default="front", choices=["front", "top_oblique"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/cube_grasp_raw"))
    parser.add_argument("--lerobot-root", type=Path, default=Path("outputs/lerobot_datasets"))
    parser.add_argument("--repo-id", default="local/panda_6dof_7ctrl")
    parser.add_argument("--no-lerobot", action="store_true", help="Only write raw compressed npz episodes.")
    parser.add_argument("--success-only", action="store_true", help="Only save episodes that lift or place the target cube.")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    env = CubeGraspEnv(width=args.width, height=args.height, camera=args.camera, seed=args.seed, max_steps=args.steps)
    dataset = None if args.no_lerobot else _maybe_create_lerobot_dataset(args)

    successes = 0
    try:
        for episode in trange(args.episodes, desc="episodes"):
            obs, info = env.reset(seed=args.seed + episode)
            images: list[np.ndarray] = []
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            dones: list[bool] = []

            for step in range(args.steps):
                action = scripted_grasp_action(env, step)
                obs, reward, terminated, truncated, info = env.step(action)

                images.append(obs["observation.image"])
                states.append(obs["observation.state"])
                actions.append(action.astype(np.float32))
                rewards.append(float(reward))
                dones.append(bool(terminated or truncated or step == args.steps - 1))

                if terminated or truncated:
                    break

            success = bool(info["cube_on_goal"] or info["cube_lifted"])
            successes += int(success)
            if args.success_only and not success:
                continue

            _save_raw_episode(args.raw_dir, episode, images, states, actions, rewards, dones)

            if dataset is not None:
                for image, state, action in zip(images, states, actions):
                    _add_lerobot_frame(
                        dataset,
                        image=image,
                        state=state,
                        action=action,
                    )
                _save_lerobot_episode(dataset)

    finally:
        env.close()

    if dataset is not None:
        _finalize_lerobot_dataset(dataset)

    print(f"raw dataset: {args.raw_dir}")
    print(f"LeRobot repo_id: {args.repo_id} root={args.lerobot_root}" if dataset is not None else "LeRobot export skipped")
    print(f"scripted grasp/lift success episodes: {successes}/{args.episodes}")
    if args.success_only:
        print("success-only mode: failed episodes were skipped")


def _save_raw_episode(
    raw_dir: Path,
    episode: int,
    images: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    dones: list[bool],
) -> None:
    np.savez_compressed(
        raw_dir / f"episode_{episode:05d}.npz",
        images=np.asarray(images, dtype=np.uint8),
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        state_names=np.asarray(STATE_NAMES),
        action_names=np.asarray(ACTION_NAMES),
        task=np.asarray(TASK_DESCRIPTION),
    )


def _maybe_create_lerobot_dataset(args: argparse.Namespace) -> Any | None:
    try:
        LeRobotDataset = _import_lerobot_dataset()
    except Exception as exc:
        print(f"LeRobot is not importable, writing raw npz only: {exc}")
        return None

    features = {
        "observation.images.front": {
            "dtype": "image",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }

    create_kwargs = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        "root": args.lerobot_root,
        "robot_type": "mujoco_panda_6dof_gripper",
        "features": features,
        "use_videos": True,
    }
    try:
        return LeRobotDataset.create(**create_kwargs)
    except TypeError:
        create_kwargs.pop("use_videos")
        return LeRobotDataset.create(**create_kwargs)


def _import_lerobot_dataset():
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset


def _add_lerobot_frame(dataset: Any, *, image: np.ndarray, state: np.ndarray, action: np.ndarray) -> None:
    frame = {
        "observation.images.front": image,
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
        "task": TASK_DESCRIPTION,
    }
    dataset.add_frame(frame)


def _save_lerobot_episode(dataset: Any) -> None:
    try:
        dataset.save_episode(task=TASK_DESCRIPTION)
    except TypeError:
        dataset.save_episode()


def _finalize_lerobot_dataset(dataset: Any) -> None:
    for method_name in ("finalize", "consolidate", "compute_stats"):
        method = getattr(dataset, method_name, None)
        if method is None:
            continue
        try:
            method()
        except TypeError:
            continue


if __name__ == "__main__":
    main()
