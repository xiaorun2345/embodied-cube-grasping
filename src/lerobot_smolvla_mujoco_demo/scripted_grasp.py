from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import ARM_JOINT_RANGES


class GraspEnvLike(Protocol):
    goal_pos: np.ndarray
    step_count: int

    def cube_pose(self) -> np.ndarray: ...


ARM_LINK_LENGTHS = (0.34, 0.32)
ARM_SHOULDER_Z = 0.31
EE_HOME_Z = 0.36


def scripted_grasp_action(env: GraspEnvLike, step: int | None = None) -> np.ndarray:
    """Return a normalized 7D expert action for the cube-to-tray task."""

    if step is None:
        step = env.step_count

    cube = env.cube_pose()[:3]
    goal = env.goal_pos

    high = 0.055
    grasp = 0.190
    lift = 0.052

    keyframes = [
        (0, np.array([0.40, 0.00, high, 0.00], dtype=np.float32)),
        (45, np.array([cube[0], cube[1], high, 0.00], dtype=np.float32)),
        (90, np.array([cube[0], cube[1], grasp, 0.00], dtype=np.float32)),
        (126, np.array([cube[0], cube[1], grasp, 1.00], dtype=np.float32)),
        (176, np.array([cube[0], cube[1], lift, 1.00], dtype=np.float32)),
        (222, np.array([goal[0], goal[1], lift, 1.00], dtype=np.float32)),
        (245, np.array([goal[0], goal[1], 0.150, 1.00], dtype=np.float32)),
        (270, np.array([goal[0], goal[1], 0.150, 0.00], dtype=np.float32)),
    ]

    target = keyframes[-1][1]
    for (s0, a0), (s1, a1) in zip(keyframes, keyframes[1:]):
        if s0 <= step < s1:
            alpha = (step - s0) / max(s1 - s0, 1)
            alpha = smoothstep(alpha)
            target = (1.0 - alpha) * a0 + alpha * a1
            break

    joint_targets = cartesian_target_to_joint_control(*target)
    return normalize_joint_targets(joint_targets)


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def cartesian_target_to_joint_control(x: float, y: float, z_down: float, closed: float) -> np.ndarray:
    return np.array([*cartesian_to_arm_joints(x, y, z_down), closed], dtype=np.float32)


def cartesian_to_arm_joints(x: float, y: float, z_down: float) -> tuple[float, float, float, float, float, float]:
    base = float(np.arctan2(y, max(x, 1e-6)))
    radial = float(np.hypot(x, y))
    wrist_z = EE_HOME_Z - z_down
    dz = wrist_z - ARM_SHOULDER_Z
    upper, forearm = ARM_LINK_LENGTHS

    dist = float(np.hypot(radial, dz))
    dist = float(np.clip(dist, abs(upper - forearm) + 1e-4, upper + forearm - 1e-4))
    cos_elbow = (dist * dist - upper * upper - forearm * forearm) / (2.0 * upper * forearm)
    cos_elbow = float(np.clip(cos_elbow, -1.0, 1.0))
    elbow = -float(np.arccos(cos_elbow))
    shoulder = float(
        np.arctan2(dz, radial) - np.arctan2(forearm * np.sin(elbow), upper + forearm * np.cos(elbow))
    )
    wrist = -(shoulder + elbow)
    wrist_yaw = -0.35 * base
    wrist_roll = 0.0

    return (
        float(np.clip(base, -2.8, 2.8)),
        float(np.clip(shoulder, -1.8, 1.8)),
        float(np.clip(elbow, -2.5, 2.5)),
        float(np.clip(wrist, -2.7, 2.7)),
        float(np.clip(wrist_yaw, -2.8, 2.8)),
        float(np.clip(wrist_roll, -2.8, 2.8)),
    )


def normalize_joint_targets(targets: np.ndarray) -> np.ndarray:
    ranges = [*ARM_JOINT_RANGES, (0.0, 1.0)]
    values = []
    for value, (lo, hi) in zip(targets, ranges):
        values.append(2.0 * (float(value) - lo) / (hi - lo) - 1.0)
    return np.asarray(values, dtype=np.float32).clip(-1.0, 1.0)
