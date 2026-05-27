"""Replay recorded raw NPZ episode actions in the MuJoCo viewer.

Run:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    unset MUJOCO_GL
    python examples/08_replay_recorded_actions.py

Replay one episode:
    python examples/08_replay_recorded_actions.py --episode 3 --camera top_oblique
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from lerobot_smolvla_mujoco_demo import CubeGraspEnv


CAMERAS = ("free", "front", "top_oblique")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded raw NPZ action trajectories.")
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/cube_grasp_dualcam_state7_200_raw"))
    parser.add_argument("--episode", type=int, default=None, help="Replay one episode index. Default replays all.")
    parser.add_argument("--record-seed", type=int, default=42, help="Seed used during data collection.")
    parser.add_argument("--camera", choices=CAMERAS, default="front")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pause-between", type=float, default=0.8)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    episode_files = find_episode_files(args.raw_dir, args.episode)
    if not episode_files:
        raise SystemExit(f"No episode_*.npz files found in {args.raw_dir}")

    env = CubeGraspEnv(width=640, height=480, max_steps=100000)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_camera(viewer, env, args.camera)

            while viewer.is_running():
                for episode_file in episode_files:
                    if not viewer.is_running():
                        break
                    replay_episode(env, viewer, episode_file, args)
                    time.sleep(args.pause_between)

                if not args.loop:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(1.0 / args.fps)
                    break
    finally:
        env.close()


def find_episode_files(raw_dir: Path, episode: int | None) -> list[Path]:
    if episode is not None:
        path = raw_dir / f"episode_{episode:05d}.npz"
        return [path] if path.exists() else []
    return sorted(raw_dir.glob("episode_*.npz"))


def replay_episode(env: CubeGraspEnv, viewer: mujoco.viewer.Handle, episode_file: Path, args: argparse.Namespace) -> None:
    episode_index = episode_index_from_path(episode_file)
    seed = args.record_seed + episode_index
    data = np.load(episode_file)
    actions = data["actions"].astype(np.float32)

    obs, info = env.reset(seed=seed)
    print(f"replay {episode_file.name}: frames={len(actions)}, reset_seed={seed}, task={info['task']}")

    start = time.perf_counter()
    for frame_idx, action in enumerate(actions):
        if not viewer.is_running():
            break

        _, _, terminated, truncated, info = env.step(action)
        viewer.sync()

        target_time = start + (frame_idx + 1) / args.fps
        sleep_time = target_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        if terminated or truncated:
            break

    print(
        f"done {episode_file.name}: "
        f"cube_lifted={info['cube_lifted']}, cube_on_goal={info['cube_on_goal']}, step={info['step']}"
    )


def episode_index_from_path(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.npz$", path.name)
    if not match:
        return 0
    return int(match.group(1))


def configure_camera(viewer: mujoco.viewer.Handle, env: CubeGraspEnv, camera: str) -> None:
    if camera == "free":
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = -42
        viewer.cam.elevation = -28
        viewer.cam.lookat[:] = [0.42, -0.02, 0.20]
        return

    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        raise ValueError(f"Camera not found: {camera}")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


if __name__ == "__main__":
    main()
