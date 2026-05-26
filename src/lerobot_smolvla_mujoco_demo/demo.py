from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
from tqdm import trange

from .env import CubeGraspEnv
from .scripted_grasp import scripted_grasp_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scripted MuJoCo cube-grasp demo.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", default="front", choices=["front", "top_oblique"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("outputs/cube_grasp_demo.mp4"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = CubeGraspEnv(width=args.width, height=args.height, camera=args.camera, seed=args.seed, max_steps=args.steps)
    frames = []
    successes = 0

    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed + episode)
            for step in trange(args.steps, desc=f"episode {episode}", leave=False):
                obs, _, terminated, truncated, info = env.step(scripted_grasp_action(env, step))
                frames.append(obs["observation.image"])
                if terminated or truncated:
                    break
            successes += int(info["cube_on_goal"] or info["cube_lifted"])
    finally:
        env.close()

    iio.imwrite(args.out, frames, fps=30)
    print(f"wrote {args.out}")
    print(f"scripted grasp/lift success episodes: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
