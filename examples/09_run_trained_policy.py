"""Run the trained SmolVLA policy in the MuJoCo cube-grasp scene.

Video evaluation:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    MUJOCO_GL=egl python examples/09_run_trained_policy.py

Realtime MuJoCo viewer:
    unset MUJOCO_GL
    python examples/09_run_trained_policy.py --viewer --camera front

Explicit checkpoint:
    python examples/09_run_trained_policy.py \
      --policy-path outputs/smolvla_panda_dualcam_state7_200_success_test/checkpoints/010000/pretrained_model
"""

from __future__ import annotations

from lerobot_smolvla_mujoco_demo.eval_policy import main


if __name__ == "__main__":
    main()
