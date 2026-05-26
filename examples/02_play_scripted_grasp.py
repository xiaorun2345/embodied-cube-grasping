"""Open the MuJoCo window and play the scripted cube-grasp demo.

Run:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    unset MUJOCO_GL
    python examples/02_play_scripted_grasp.py
"""

from __future__ import annotations

import time

import mujoco.viewer

from lerobot_smolvla_mujoco_demo import CubeGraspEnv, scripted_grasp_action


def main() -> None:
    env = CubeGraspEnv(width=640, height=480, max_steps=280)
    env.reset(seed=7)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = -42
            viewer.cam.elevation = -28
            viewer.cam.lookat[:] = [0.42, -0.02, 0.20]

            step = 0
            while viewer.is_running():
                action = scripted_grasp_action(env, step)
                _, _, terminated, truncated, _ = env.step(action)

                viewer.sync()
                time.sleep(1 / 30)

                step += 1
                if terminated or truncated or step >= 280:
                    step = 0
                    env.reset(seed=7)
    finally:
        env.close()


if __name__ == "__main__":
    main()
