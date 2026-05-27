from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
]
ARM_JOINT_RANGES = [
    (-2.8, 2.8),
    (-1.8, 1.8),
    (-2.5, 2.5),
    (-2.7, 2.7),
    (-2.8, 2.8),
    (-2.8, 2.8),
]
ACTION_NAMES = [*ARM_JOINT_NAMES, "gripper_closed"]
HOME_TARGETS = np.array(
    [0.0, 0.8552878, -1.8401252, 0.9848374, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
STATE_NAMES = [
    *ARM_JOINT_NAMES,
    "gripper_closed",
]
TASK_DESCRIPTION = "Pick up the blue cube and place it inside the gold tray."


@dataclass(frozen=True)
class Workspace:
    x: tuple[float, float] = (0.18, 0.70)
    y: tuple[float, float] = (-0.26, 0.26)
    z_down: tuple[float, float] = (0.0, 0.27)
    gripper: tuple[float, float] = (0.0, 0.032)


class CubeGraspEnv(gym.Env):
    """A compact MuJoCo tabletop cube-grasp task with a Panda-style arm.

    The public action is normalized to [-1, 1] and contains six arm joint
    targets plus one gripper-closure value.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        camera: str = "front",
        frame_skip: int = 10,
        max_steps: int = 260,
        seed: int | None = None,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        xml_path = files("lerobot_smolvla_mujoco_demo").joinpath("assets/mujoco/cube_grasp_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width = width
        self.height = height
        self.camera = camera
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.workspace = Workspace()
        self.goal_pos = np.array([0.55, -0.16, 0.079], dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        self._arm_joint_targets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._finger_target = 0.0
        self._grasped = False
        self._cube_grasp_z_offset = 0.093

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(ACTION_NAMES),), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation.image": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8),
                "observation.images.front": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8),
                "observation.images.top_oblique": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8),
                "observation.state": spaces.Box(-np.inf, np.inf, shape=(len(STATE_NAMES),), dtype=np.float32),
            }
        )

        self._joint_qpos = {
            name: self.model.jnt_qposadr[
                self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in [
                *ARM_JOINT_NAMES,
                "left_gripper",
                "right_gripper",
                "cube_free",
                "distractor_orange_free",
                "distractor_purple_free",
            ]
        }
        self._wrist_body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "panda_wrist")
        cube_joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self._cube_qvel_adr = self.model.jnt_dofadr[cube_joint_id]
        self._joint_dof = {
            name: self.model.jnt_dofadr[
                self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in [
                *ARM_JOINT_NAMES,
                "left_gripper",
                "right_gripper",
            ]
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.mujoco.mj_resetData(self.model, self.data)
        cube_xy = self._sample_cube_xy(options)
        cube_adr = self._joint_qpos["cube_free"]
        self.data.qpos[cube_adr : cube_adr + 7] = np.array(
            [cube_xy[0], cube_xy[1], 0.077, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        distractor_xys = self._sample_distractor_xys(cube_xy, options)
        for name, xy in zip(["distractor_orange_free", "distractor_purple_free"], distractor_xys):
            adr = self._joint_qpos[name]
            self.data.qpos[adr : adr + 7] = np.array(
                [xy[0], xy[1], 0.077, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        self.data.qvel[:] = 0.0
        self._grasped = False

        self._set_targets(HOME_TARGETS)
        self._last_action = self._normalize_targets(HOME_TARGETS)
        self._step_count = 0

        for _ in range(80):
            self._apply_arm_joint_targets()
            self.mujoco.mj_step(self.model, self.data)
        self._apply_arm_joint_targets()
        self.mujoco.mj_forward(self.model, self.data)

        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self._last_action = action
        self._set_targets(self._denormalize_action(action))
        self.mujoco.mj_forward(self.model, self.data)
        self._sync_grasped_cube()

        for _ in range(self.frame_skip):
            self._apply_arm_joint_targets()
            self.mujoco.mj_step(self.model, self.data)
            self._sync_grasped_cube()
        self._apply_arm_joint_targets()
        self._sync_grasped_cube()
        self.mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        info = self._info()
        reward = float(info["cube_on_goal"]) + 0.25 * float(info["cube_lifted"])
        terminated = bool(info["cube_on_goal"])
        truncated = self._step_count >= self.max_steps
        return self._observation(), reward, terminated, truncated, info

    def render(self, camera: str | None = None):
        self.renderer.update_scene(self.data, camera=camera or self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()

    @property
    def step_count(self) -> int:
        return self._step_count

    def cube_pose(self) -> np.ndarray:
        cube_adr = self._joint_qpos["cube_free"]
        return self.data.qpos[cube_adr : cube_adr + 7].astype(np.float32).copy()

    def grasp_status(self) -> dict[str, float | bool]:
        wrist = self.data.xpos[self._wrist_body_id]
        cube = self.cube_pose()
        xy_dist = float(np.linalg.norm(cube[:2] - wrist[:2]))
        z_dist = abs(float(cube[2] - (wrist[2] - self._cube_grasp_z_offset)))
        closed = self._finger_target > 0.024
        return {
            "closed": closed,
            "grasped": self._grasped,
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "can_grasp": bool(closed and xy_dist < 0.085 and z_dist < 0.075),
            "finger_target": float(self._finger_target),
            "wrist_x": float(wrist[0]),
            "wrist_y": float(wrist[1]),
            "wrist_z": float(wrist[2]),
            "cube_x": float(cube[0]),
            "cube_y": float(cube[1]),
            "cube_z": float(cube[2]),
        }

    def _sample_cube_xy(self, options: dict[str, Any] | None) -> np.ndarray:
        if options and "cube_xy" in options:
            return np.asarray(options["cube_xy"], dtype=np.float32)
        return np.array(
            [
                self._rng.uniform(0.36, 0.48),
                self._rng.uniform(-0.05, 0.10),
            ],
            dtype=np.float32,
        )

    def _sample_distractor_xys(self, cube_xy: np.ndarray, options: dict[str, Any] | None) -> list[np.ndarray]:
        if options and "distractor_xys" in options:
            return [np.asarray(xy, dtype=np.float32) for xy in options["distractor_xys"]]

        positions: list[np.ndarray] = []
        blocked = [np.asarray(cube_xy, dtype=np.float32), self.goal_pos[:2]]
        zones = [
            ((0.30, 0.42), (-0.20, -0.08)),
            ((0.56, 0.68), (0.08, 0.20)),
        ]
        for x_range, y_range in zones:
            for _attempt in range(100):
                xy = np.array(
                    [
                        self._rng.uniform(*x_range),
                        self._rng.uniform(*y_range),
                    ],
                    dtype=np.float32,
                )
                if all(float(np.linalg.norm(xy - other)) > 0.10 for other in [*blocked, *positions]):
                    positions.append(xy)
                    break
            else:
                fallback = np.array(
                    [sum(x_range) * 0.5, sum(y_range) * 0.5],
                    dtype=np.float32,
                )
                positions.append(fallback)
        return positions

    def _set_targets(self, targets: np.ndarray) -> None:
        arm_targets = np.asarray(targets[: len(ARM_JOINT_NAMES)], dtype=np.float32)
        closed = float(np.clip(targets[-1], 0.0, 1.0))
        finger = closed * self.workspace.gripper[1]
        self._arm_joint_targets = tuple(float(v) for v in arm_targets)
        self._finger_target = finger
        self.data.ctrl[:] = [*self._arm_joint_targets, finger, finger]
        self._apply_arm_joint_targets()

    def _apply_arm_joint_targets(self) -> None:
        for name, value in zip(ARM_JOINT_NAMES, self._arm_joint_targets):
            self.data.qpos[self._joint_qpos[name]] = value
            self.data.qvel[self._joint_dof[name]] = 0.0
        for name in ["left_gripper", "right_gripper"]:
            self.data.qpos[self._joint_qpos[name]] = self._finger_target
            self.data.qvel[self._joint_dof[name]] = 0.0

    def _sync_grasped_cube(self) -> None:
        wrist = self.data.xpos[self._wrist_body_id]
        cube = self.cube_pose()
        closed = self._finger_target > 0.024
        if not closed:
            if self._grasped:
                cube_adr = self._joint_qpos["cube_free"]
                place_xy = self.goal_pos[:2] if np.linalg.norm(wrist[:2] - self.goal_pos[:2]) < 0.10 else wrist[:2]
                self.data.qpos[cube_adr : cube_adr + 7] = np.array(
                    [
                        place_xy[0],
                        place_xy[1],
                        0.087,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float64,
                )
                self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6] = 0.0
            self._grasped = False
            return

        if not self._grasped:
            xy_dist = float(np.linalg.norm(cube[:2] - wrist[:2]))
            z_dist = abs(float(cube[2] - (wrist[2] - self._cube_grasp_z_offset)))
            self._grasped = xy_dist < 0.085 and z_dist < 0.075

        if self._grasped:
            cube_adr = self._joint_qpos["cube_free"]
            self.data.qpos[cube_adr : cube_adr + 7] = np.array(
                [
                    wrist[0],
                    wrist[1],
                    max(wrist[2] - self._cube_grasp_z_offset, 0.077),
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            self.data.qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6] = 0.0

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        ranges = [*ARM_JOINT_RANGES, (0.0, 1.0)]
        values = []
        for value, (lo, hi) in zip(action, ranges):
            values.append(lo + 0.5 * (float(value) + 1.0) * (hi - lo))
        return np.array(values, dtype=np.float32)

    def _normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        ranges = [*ARM_JOINT_RANGES, (0.0, 1.0)]
        values = []
        for value, (lo, hi) in zip(targets, ranges):
            values.append(2.0 * (float(value) - lo) / (hi - lo) - 1.0)
        return np.asarray(values, dtype=np.float32).clip(-1.0, 1.0)

    def _joint_state(self) -> np.ndarray:
        joints = [float(self.data.qpos[self._joint_qpos[name]]) for name in ARM_JOINT_NAMES]
        gripper_closed = float(self._finger_target / self.workspace.gripper[1]) if self.workspace.gripper[1] else 0.0
        return np.asarray([*joints, gripper_closed], dtype=np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        front_image = self.render("front").copy()
        top_image = self.render("top_oblique").copy()
        image = top_image if self.camera == "top_oblique" else front_image
        return {
            "observation.image": image,
            "observation.images.front": front_image,
            "observation.images.top_oblique": top_image,
            "observation.state": self._joint_state(),
        }

    def _info(self, *, render: bool = True) -> dict[str, Any]:
        cube = self.cube_pose()
        xy_dist = float(np.linalg.norm(cube[:2] - self.goal_pos[:2]))
        cube_lifted = bool(cube[2] > 0.135)
        cube_on_goal = bool(xy_dist < 0.055 and 0.065 <= cube[2] <= 0.105)
        return {
            "task": TASK_DESCRIPTION,
            "step": self._step_count,
            "cube_pose": cube,
            "goal_pos": self.goal_pos.copy(),
            "cube_lifted": cube_lifted,
            "cube_on_goal": cube_on_goal,
            "image_shape": (self.height, self.width, 3) if render else None,
        }


def make_env(**kwargs: Any) -> CubeGraspEnv:
    return CubeGraspEnv(**kwargs)
