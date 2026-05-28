#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot_smolvla_mujoco_demo.env import CubeGraspEnv
from lerobot_smolvla_mujoco_demo.eval_policy import (
    configure_offline_hf_cache,
    load_lerobot_policy,
    maybe_latch_gripper,
    resolve_device,
    select_action,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one policy rollout with grasp diagnostics.")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=Path(
            "outputs/smolvla_panda_dualcam_state7_200_success_test_20260527_163016/"
            "checkpoints/063750/pretrained_model"
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", choices=["front", "top_oblique"], default="front")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--gripper-latch", action="store_true")
    parser.add_argument("--latch-close-threshold", type=float, default=0.0)
    parser.add_argument("--latch-release-threshold", type=float, default=0.25)
    args = parser.parse_args()

    configure_offline_hf_cache()
    device = resolve_device(args.device)
    print(f"policy_path: {args.policy_path}")
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print()

    policy, preprocessor, postprocessor = load_lerobot_policy(args.policy_path, device=device)
    env = CubeGraspEnv(
        width=args.width,
        height=args.height,
        camera=args.camera,
        seed=args.seed,
        max_steps=args.steps,
    )
    latch_args = SimpleNamespace(
        gripper_latch=args.gripper_latch,
        latch_close_threshold=args.latch_close_threshold,
        latch_release_threshold=args.latch_release_threshold,
    )

    try:
        if hasattr(policy, "reset"):
            policy.reset()
        obs, info = env.reset(seed=args.seed)
        gripper_latched = False
        lifted_once = False

        for step in range(args.steps):
            raw_action = select_action(policy, preprocessor, postprocessor, obs)
            action, gripper_latched = maybe_latch_gripper(raw_action, env, gripper_latched, latch_args)
            obs, _reward, terminated, truncated, info = env.step(action)
            lifted_once = lifted_once or bool(info["cube_lifted"])

            if step % args.every == 0 or step >= args.steps - 10 or terminated or truncated:
                status = env.grasp_status()
                cube = info["cube_pose"]
                goal = info["goal_pos"]
                d_goal = float(np.linalg.norm(cube[:2] - goal[:2]))
                print(
                    f"step={step:03d} "
                    f"raw_g={raw_action[-1]:+.3f} act_g={action[-1]:+.3f} latch={gripper_latched} "
                    f"lift={int(info['cube_lifted'])} goal={int(info['cube_on_goal'])} "
                    f"cube=({cube[0]:+.3f},{cube[1]:+.3f},{cube[2]:+.3f}) d_goal={d_goal:.3f} "
                    f"wrist=({status['wrist_x']:+.3f},{status['wrist_y']:+.3f},{status['wrist_z']:+.3f}) "
                    f"xy={status['xy_dist']:.3f} z={status['z_dist']:.3f} grasped={int(status['grasped'])}"
                )

            if terminated or truncated:
                break

        print()
        print(
            f"final: success={bool(lifted_once and info['cube_on_goal'])} "
            f"lifted_once={lifted_once} cube_on_goal={info['cube_on_goal']} step={info['step']}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
