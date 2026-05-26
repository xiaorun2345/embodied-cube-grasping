"""Open the MuJoCo window and show the reset scene.

Run:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    unset MUJOCO_GL
    python examples/01_show_scene.py
"""

from __future__ import annotations

import time

import mujoco.viewer

from lerobot_smolvla_mujoco_demo import CubeGraspEnv


def main() -> None:
    env = CubeGraspEnv(width=640, height=480)
    obs, info = env.reset(seed=7)

    print("task:", info["task"])
    print("image shape:", obs["observation.image"].shape)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = -42
            viewer.cam.elevation = -28
            viewer.cam.lookat[:] = [0.42, -0.02, 0.20]

            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 30)
    finally:
        env.close()


if __name__ == "__main__":
    main()
