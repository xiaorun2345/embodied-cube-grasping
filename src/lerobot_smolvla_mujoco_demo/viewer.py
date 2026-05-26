from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a realtime MuJoCo viewer for the cube-grasp environment.")
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true", help="Replay the scripted grasp continuously.")
    parser.add_argument("--manual", action="store_true", help="Only open the scene; do not run the scripted expert.")
    args = parser.parse_args()

    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ.pop("MUJOCO_GL")

    import mujoco.viewer

    from .env import CubeGraspEnv
    from .scripted_grasp import scripted_grasp_action

    env = CubeGraspEnv(width=320, height=240, seed=args.seed, max_steps=args.steps)
    env.reset(seed=args.seed)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = -42
            viewer.cam.elevation = -28
            viewer.cam.lookat[:] = [0.42, -0.02, 0.20]

            while viewer.is_running():
                env.reset(seed=args.seed)
                step = 0
                episode_start = time.perf_counter()

                while viewer.is_running() and step < args.steps:
                    if not args.manual:
                        env.step(scripted_grasp_action(env, step))
                    else:
                        env.step(env.action_space.sample() * 0.0)

                    viewer.sync()
                    step += 1

                    target_time = episode_start + step / args.fps
                    sleep_time = target_time - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if not args.loop:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(1.0 / args.fps)
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
