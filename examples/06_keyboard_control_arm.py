"""Control the MuJoCo arm with keyboard joint commands.

Run:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    unset MUJOCO_GL
    python examples/06_keyboard_control_arm.py

Key map:
    A / D : panda_joint1 -
    W / S : panda_joint2 +/-
    Q / E : panda_joint3 -/+
    R / F : panda_joint4 +/-
    T / G : panda_joint5 +/-
    Y / H : panda_joint6 +/-
    Z / X : gripper open/close
    B     : back to home pose
    N     : reset cube and arm
    P     : print current target values
"""

from __future__ import annotations

import argparse
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from lerobot_smolvla_mujoco_demo.env import ACTION_NAMES, ARM_JOINT_RANGES, HOME_TARGETS, CubeGraspEnv


CAMERAS = ("free", "front", "top_oblique")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard-control the 6-DOF arm and gripper.")
    parser.add_argument("--camera", choices=CAMERAS, default="front")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--joint-step", type=float, default=0.08, help="Radians changed per key press.")
    parser.add_argument("--gripper-step", type=float, default=0.15, help="Gripper closed amount changed per key press.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run forever.")
    args = parser.parse_args()

    env = CubeGraspEnv(width=640, height=480, seed=args.seed, max_steps=100000)
    env.reset(seed=args.seed)
    targets = HOME_TARGETS.copy()
    pending_home = False
    pending_reset = False

    print_help()
    print_targets(targets)

    def key_callback(key: int) -> None:
        nonlocal pending_home, pending_reset

        if key == glfw.KEY_B:
            pending_home = True
            return
        if key == glfw.KEY_N:
            pending_reset = True
            return
        if key == glfw.KEY_P:
            print_targets(targets)
            return

        before = targets.copy()
        apply_key_to_targets(targets, key, joint_step=args.joint_step, gripper_step=args.gripper_step)
        if not np.allclose(before, targets):
            print_targets(targets)

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            configure_camera(viewer, env, args.camera)
            last_frame = time.perf_counter()

            while viewer.is_running():
                if pending_reset:
                    env.reset(seed=args.seed)
                    targets = HOME_TARGETS.copy()
                    pending_reset = False
                    pending_home = False
                    print_targets(targets)
                    viewer.sync()
                    continue
                if pending_home:
                    targets = HOME_TARGETS.copy()
                    pending_home = False
                    print_targets(targets)

                action = normalize_targets(targets)
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


def apply_key_to_targets(targets: np.ndarray, key: int, *, joint_step: float, gripper_step: float) -> None:
    if key == glfw.KEY_A:
        targets[0] -= joint_step
    elif key == glfw.KEY_D:
        targets[0] += joint_step
    elif key == glfw.KEY_W:
        targets[1] += joint_step
    elif key == glfw.KEY_S:
        targets[1] -= joint_step
    elif key == glfw.KEY_Q:
        targets[2] -= joint_step
    elif key == glfw.KEY_E:
        targets[2] += joint_step
    elif key == glfw.KEY_R:
        targets[3] += joint_step
    elif key == glfw.KEY_F:
        targets[3] -= joint_step
    elif key == glfw.KEY_T:
        targets[4] += joint_step
    elif key == glfw.KEY_G:
        targets[4] -= joint_step
    elif key == glfw.KEY_Y:
        targets[5] += joint_step
    elif key == glfw.KEY_H:
        targets[5] -= joint_step
    elif key == glfw.KEY_Z:
        targets[6] -= gripper_step
    elif key == glfw.KEY_X:
        targets[6] += gripper_step

    for i, (lo, hi) in enumerate([*ARM_JOINT_RANGES, (0.0, 1.0)]):
        targets[i] = np.clip(targets[i], lo, hi)


def normalize_targets(targets: np.ndarray) -> np.ndarray:
    values = []
    for value, (lo, hi) in zip(targets, [*ARM_JOINT_RANGES, (0.0, 1.0)]):
        values.append(2.0 * (float(value) - lo) / (hi - lo) - 1.0)
    return np.asarray(values, dtype=np.float32).clip(-1.0, 1.0)


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
Keyboard arm control
--------------------
A / D : panda_joint1 -/+
W / S : panda_joint2 +/-
Q / E : panda_joint3 -/+
R / F : panda_joint4 +/-
T / G : panda_joint5 +/-
Y / H : panda_joint6 +/-
Z / X : gripper open/close
B     : back to home pose
N     : reset cube and arm
P     : print current target values
Close the viewer window to quit.
""".strip()
    )


def print_targets(targets: np.ndarray) -> None:
    text = ", ".join(f"{name}={value:.3f}" for name, value in zip(ACTION_NAMES, targets))
    print(f"targets: {text}")


if __name__ == "__main__":
    main()
