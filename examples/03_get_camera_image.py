"""Render fixed MuJoCo camera images and save them as PNG files.

Run on a server/headless terminal:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    MUJOCO_GL=egl python examples/03_get_camera_image.py

Run one camera only:
    MUJOCO_GL=egl python examples/03_get_camera_image.py --camera front
    MUJOCO_GL=egl python examples/03_get_camera_image.py --camera top_oblique
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio

from lerobot_smolvla_mujoco_demo import CubeGraspEnv


CAMERAS = ("front", "top_oblique")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save MuJoCo fixed camera images.")
    parser.add_argument("--camera", choices=[*CAMERAS, "all"], default="all")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/camera_images"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cameras = CAMERAS if args.camera == "all" else (args.camera,)

    for camera in cameras:
        env = CubeGraspEnv(width=args.width, height=args.height, camera=camera)
        try:
            obs, info = env.reset(seed=args.seed)

            image_from_obs = obs["observation.image"]
            image_from_render = env.render()

            obs_path = args.out_dir / f"{camera}_observation.png"
            render_path = args.out_dir / f"{camera}_render.png"
            iio.imwrite(obs_path, image_from_obs)
            iio.imwrite(render_path, image_from_render)

            print(f"camera: {camera}")
            print(f"task: {info['task']}")
            print(f"observation image shape: {image_from_obs.shape}")
            print(f"saved: {obs_path}")
            print(f"saved: {render_path}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
