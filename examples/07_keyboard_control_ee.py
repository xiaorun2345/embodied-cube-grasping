"""Control the gripper end-effector target with the keyboard.

This is easier than 06_keyboard_control_arm.py because you control a Cartesian
target [x, y, z_down, gripper_closed], not six individual arm joints.

Run:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    unset MUJOCO_GL
    python examples/07_keyboard_control_ee.py --camera front

Key map:
    W / S : x forward/back
    A / D : y left/right
    R / F : raise/lower gripper
    Z / X : open/close gripper fully
    B     : back to home target
    N     : reset cube and arm
    P     : print target

Guided grasp shortcuts:
    1 : move above cube
    2 : lower to cube
    3 : close gripper
    4 : lift cube
    5 : move above tray
    6 : lower to tray
    7 : open gripper
"""

from __future__ import annotations

import argparse
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from lerobot_smolvla_mujoco_demo import CubeGraspEnv
from lerobot_smolvla_mujoco_demo.scripted_grasp import (
    cartesian_target_to_joint_control,
    normalize_joint_targets,
)


CAMERAS = ("free", "front", "top_oblique")
HOME_EE_TARGET = np.array([0.40, 0.00, 0.060, 0.0], dtype=np.float32)
ABOVE_Z_DOWN = 0.055
GRASP_Z_DOWN = 0.190
LIFT_Z_DOWN = 0.052
TRAY_Z_DOWN = 0.150


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard-control the gripper end-effector target.")
    parser.add_argument("--camera", choices=CAMERAS, default="front")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--xy-step", type=float, default=0.015, help="Meters changed per x/y key press.")
    parser.add_argument("--z-step", type=float, default=0.015, help="Meters changed per vertical key press.")
    parser.add_argument("--gripper-step", type=float, default=1.0, help="Gripper closed amount changed per key press.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run forever.")
    args = parser.parse_args()

    env = CubeGraspEnv(width=640, height=480, seed=args.seed, max_steps=100000)
    env.reset(seed=args.seed)
    target = HOME_EE_TARGET.copy()
    pending_reset = False

    print_help()
    print_target(target)

    def key_callback(key: int) -> None:
        nonlocal pending_reset

        if key == glfw.KEY_N:
            pending_reset = True
            return
        if key == glfw.KEY_P:
            print_target(target)
            print_grasp_status(env)
            return

        before = target.copy()
        apply_key_to_target(env, target, key, xy_step=args.xy_step, z_step=args.z_step, gripper_step=args.gripper_step)
        if not np.allclose(before, target):
            print_target(target)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            configure_camera(viewer, env, args.camera)
            last_frame = time.perf_counter()

            while viewer.is_running():
                if pending_reset:
                    env.reset(seed=args.seed)
                    target[:] = HOME_EE_TARGET
                    pending_reset = False
                    print_target(target)
                    viewer.sync()
                    continue

                joint_targets = cartesian_target_to_joint_control(*target)
                action = normalize_joint_targets(joint_targets)
                env.step(action)
                viewer.sync()

                if args.max_frames and env.step_count >= args.max_frames:
                    break

                now = time.perf_counter()
                sleep_time = max(0.0, (1.0 / args.fps) - (now - last_frame))
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame = time.perf_counter()
    finally:
        env.close()


def apply_key_to_target(
    env: CubeGraspEnv,
    target: np.ndarray,
    key: int,
    *,
    xy_step: float,
    z_step: float,
    gripper_step: float,
) -> None:
    if key == glfw.KEY_W:
        target[0] += xy_step
    elif key == glfw.KEY_S:
        target[0] -= xy_step
    elif key == glfw.KEY_A:
        target[1] += xy_step
    elif key == glfw.KEY_D:
        target[1] -= xy_step
    elif key == glfw.KEY_R:
        target[2] -= z_step
    elif key == glfw.KEY_F:
        target[2] += z_step
    elif key == glfw.KEY_Z:
        target[3] = max(0.0, target[3] - gripper_step)
    elif key == glfw.KEY_X:
        target[3] = min(1.0, target[3] + gripper_step)
    elif key == glfw.KEY_B:
        target[:] = HOME_EE_TARGET
    elif key == glfw.KEY_1:
        target[:] = above_cube_target(env, closed=0.0)
    elif key == glfw.KEY_2:
        target[:] = at_cube_target(env, closed=0.0)
    elif key == glfw.KEY_3:
        target[3] = 1.0
    elif key == glfw.KEY_4:
        target[:] = above_cube_target(env, closed=1.0)
    elif key == glfw.KEY_5:
        target[:] = above_tray_target(env, closed=1.0)
    elif key == glfw.KEY_6:
        target[:] = at_tray_target(env, closed=1.0)
    elif key == glfw.KEY_7:
        target[3] = 0.0

    clamp_target(env, target)


def above_cube_target(env: CubeGraspEnv, *, closed: float) -> np.ndarray:
    cube = env.cube_pose()
    return np.array([cube[0], cube[1], ABOVE_Z_DOWN, closed], dtype=np.float32)


def at_cube_target(env: CubeGraspEnv, *, closed: float) -> np.ndarray:
    cube = env.cube_pose()
    return np.array([cube[0], cube[1], GRASP_Z_DOWN, closed], dtype=np.float32)


def above_tray_target(env: CubeGraspEnv, *, closed: float) -> np.ndarray:
    return np.array([env.goal_pos[0], env.goal_pos[1], LIFT_Z_DOWN, closed], dtype=np.float32)


def at_tray_target(env: CubeGraspEnv, *, closed: float) -> np.ndarray:
    return np.array([env.goal_pos[0], env.goal_pos[1], TRAY_Z_DOWN, closed], dtype=np.float32)


def clamp_target(env: CubeGraspEnv, target: np.ndarray) -> None:
    target[0] = np.clip(target[0], *env.workspace.x)
    target[1] = np.clip(target[1], *env.workspace.y)
    target[2] = np.clip(target[2], *env.workspace.z_down)
    target[3] = np.clip(target[3], 0.0, 1.0)


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


def print_help() -> None:
    print(
        """
End-effector keyboard control
-----------------------------
W / S : x forward/back
A / D : y left/right
R / F : raise/lower gripper
Z / X : open/close gripper fully
B     : back to home target
N     : reset cube and arm
P     : print target

Guided grasp shortcuts:
1 : move above cube
2 : lower to cube
3 : close gripper
4 : lift cube
5 : move above tray
6 : lower to tray
7 : open gripper

Close the viewer window to quit.
""".strip()
    )


def print_target(target: np.ndarray) -> None:
    print(
        "ee_target: "
        f"x={target[0]:.3f}, y={target[1]:.3f}, z_down={target[2]:.3f}, gripper_closed={target[3]:.2f}"
    )


def print_grasp_status(env: CubeGraspEnv) -> None:
    status = env.grasp_status()
    print(
        "grasp_status: "
        f"closed={status['closed']}, grasped={status['grasped']}, can_grasp={status['can_grasp']}, "
        f"xy_dist={status['xy_dist']:.3f}, z_dist={status['z_dist']:.3f}, "
        f"finger_target={status['finger_target']:.3f}"
    )


if __name__ == "__main__":
    main()
