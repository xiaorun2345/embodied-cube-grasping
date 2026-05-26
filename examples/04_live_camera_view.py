"""Show a realtime fixed-camera video stream from the MuJoCo environment.

Run with a desktop/VNC/X11 display:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    unset MUJOCO_GL
    python examples/04_live_camera_view.py --camera front

Use OpenCV window instead of MuJoCo viewer:
    MUJOCO_GL=egl python examples/04_live_camera_view.py --backend opencv --camera front

Keys:
    q or Esc: quit
    r: reset episode
"""

from __future__ import annotations

import argparse
import os
import time

from lerobot_smolvla_mujoco_demo import CubeGraspEnv, scripted_grasp_action


CAMERAS = ("front", "top_oblique")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a realtime camera video window.")
    parser.add_argument("--backend", choices=["mujoco", "opencv"], default="mujoco")
    parser.add_argument("--camera", choices=CAMERAS, default="front")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manual", action="store_true", help="Show the camera stream without scripted grasp motion.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run forever.")
    parser.add_argument("--no-window", action="store_true", help="Render frames without opening an OpenCV window.")
    args = parser.parse_args()

    if args.backend == "mujoco" and args.no_window:
        raise SystemExit("--no-window is only useful with --backend opencv.")
    if args.backend == "opencv":
        run_opencv_camera(args)
    else:
        run_mujoco_camera(args)


def run_mujoco_camera(args: argparse.Namespace) -> None:
    if os.environ.get("MUJOCO_GL") == "egl":
        print("MUJOCO_GL=egl is for offscreen rendering. For a realtime viewer window, use: unset MUJOCO_GL")

    env = CubeGraspEnv(width=args.width, height=args.height, camera=args.camera, seed=args.seed, max_steps=args.steps)
    env.reset(seed=args.seed)

    import mujoco
    import mujoco.viewer

    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if camera_id < 0:
        raise ValueError(f"Camera not found: {args.camera}")

    step = 0
    frame_count = 0
    episode_start = time.perf_counter()

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id

            while viewer.is_running():
                if args.manual:
                    action = env.action_space.sample() * 0.0
                else:
                    action = scripted_grasp_action(env, step)

                _, _, terminated, truncated, _ = env.step(action)
                viewer.sync()

                step += 1
                frame_count += 1
                if terminated or truncated or step >= args.steps:
                    step = 0
                    env.reset(seed=args.seed)
                    episode_start = time.perf_counter()

                if args.max_frames and frame_count >= args.max_frames:
                    break

                target_time = episode_start + max(step, 1) / args.fps
                sleep_time = target_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        env.close()

    print(f"rendered frames: {frame_count}")


def run_opencv_camera(args: argparse.Namespace) -> None:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit("OpenCV is not installed. Use --backend mujoco, or install opencv-python.") from exc

    if os.environ.get("MUJOCO_GL") != "egl":
        print("Tip: for OpenCV backend on a server, use MUJOCO_GL=egl.")

    env = CubeGraspEnv(width=args.width, height=args.height, camera=args.camera, seed=args.seed, max_steps=args.steps)
    env.reset(seed=args.seed)

    window_name = f"MuJoCo camera: {args.camera}"
    if not args.no_window:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            env.close()
            raise SystemExit(
                "OpenCV cannot open GUI windows in this environment. "
                "Use the default MuJoCo backend instead:\n"
                "  unset MUJOCO_GL\n"
                f"  python examples/04_live_camera_view.py --camera {args.camera}\n"
                "Or run with a desktop/VNC/NoMachine/X11 display."
            ) from exc

    step = 0
    frame_count = 0
    episode_start = time.perf_counter()

    try:
        while True:
            action = env.action_space.sample() * 0.0 if args.manual else scripted_grasp_action(env, step)
            _, _, terminated, truncated, _ = env.step(action)
            rgb = env.render()

            if not args.no_window:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    step = 0
                    env.reset(seed=args.seed)
                    episode_start = time.perf_counter()
                    continue

            step += 1
            frame_count += 1
            if terminated or truncated or step >= args.steps:
                step = 0
                env.reset(seed=args.seed)
                episode_start = time.perf_counter()

            if args.max_frames and frame_count >= args.max_frames:
                break

            target_time = episode_start + max(step, 1) / args.fps
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        env.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    print(f"rendered frames: {frame_count}")


if __name__ == "__main__":
    main()
