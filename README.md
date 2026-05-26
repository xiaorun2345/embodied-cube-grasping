# Embodied Cube Grasping

An embodied AI system for cube grasping and robotic manipulation. This project combines LeRobot, MuJoCo simulation, and SmolVLA training helpers for a cube-to-tray grasping task.

这个仓库包含一个独立 demo：MuJoCo 桌面抓取场景、脚本专家采集、LeRobotDataset 导出，以及 SmolVLA 微调入口。

## 目录

- `src/lerobot_smolvla_mujoco_demo/assets/mujoco/cube_grasp_scene.xml`：带灯光、材质、桌面、目标托盘和 Panda 风格机械臂的 MuJoCo 场景。
- `src/lerobot_smolvla_mujoco_demo/env.py`：Gymnasium 环境，只负责加载 XML、reset、step、render；动作是归一化的 7 维控制量：6 个机械臂关节目标 + 1 个夹爪闭合量。
- `src/lerobot_smolvla_mujoco_demo/scripted_grasp.py`：脚本专家抓取动作，负责生成抓方块并放进托盘的 7 维动作。
- `src/lerobot_smolvla_mujoco_demo/demo.py`：运行脚本抓取 demo，输出 mp4。
- `src/lerobot_smolvla_mujoco_demo/record_dataset.py`：采集专家轨迹，写 raw `npz`，安装 LeRobot 后也写 LeRobot 数据集。
- `scripts/train_smolvla_cube.sh`：用 `lerobot-train` 启动 SmolVLA 训练。

## 安装

推荐新环境：

```bash
cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
bash scripts/setup_lerobot_smolvla.sh
conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
```

如果你已经有合适的 Python 环境：

```bash
cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
python -m pip install -e .
git clone --depth 1 https://github.com/huggingface/lerobot.git third_party/lerobot
python -m pip install -e "third_party/lerobot[smolvla,training]"
```

当前 LeRobot 0.5.2 要求 Python 3.12 或更新版本。

无显示器服务器上建议：

```bash
export MUJOCO_GL=egl
```

如果 EGL 不可用，改成 `osmesa`。

## 运行抓取 Demo

```bash
cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
cube-grasp-demo --episodes 1 --out outputs/cube_grasp_demo.mp4
```

输出视频在 `outputs/cube_grasp_demo.mp4`。也可以切顶视角：

```bash
cube-grasp-demo --camera top_oblique --out outputs/cube_grasp_top.mp4
```

## 采集 LeRobot 数据集

先跑小规模确认：

```bash
cube-grasp-record --episodes 5 --repo-id local/panda_6dof_7ctrl_test
```

正式给 SmolVLA 微调建议先采 50 到 200 条：

```bash
cube-grasp-record --episodes 100 --repo-id local/panda_6dof_7ctrl --success-only
```

数据会写到：

- `outputs/cube_grasp_raw/episode_*.npz`
- `outputs/lerobot_datasets/local/panda_6dof_7ctrl`（如果 LeRobot 可导入）

## 训练 SmolVLA

```bash
DATASET_REPO_ID=local/panda_6dof_7ctrl \
DATASET_ROOT=outputs/lerobot_datasets \
OUTPUT_DIR=outputs/smolvla_panda_6dof_7ctrl \
STEPS=5000 \
BATCH_SIZE=16 \
bash scripts/train_smolvla_cube.sh
```

如果只是验证刚采集的 10 条成功轨迹能否启动训练：

```bash
bash scripts/train_smolvla_10_success.sh
```

默认策略基座是 `lerobot/smolvla_base`。显存不足时先把 `BATCH_SIZE` 降到 `4` 或 `8`，再减少图像分辨率采集数据。

本机已用国内源把 PyTorch 匹配到当前驱动可用的 CUDA 11.8 wheel：

```bash
conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

期望结果是 `torch 2.7.1+cu118`、CUDA build `11.8`、`cuda_available=True`、设备为 `NVIDIA GeForce RTX 4090`。如果需要重装，用：

```bash
bash scripts/install_torch_cu118_cn.sh
```

## 备注

这个 demo 现在使用 Panda/Franka 风格的简化 6 关节机械臂，加一个夹爪闭合控制量，总动作维度是 7。注意这不是官方 Franka Panda 的完整 7 轴动力学模型；如果以后换成官方 7 轴 Franka，通常会变成 7 个机械臂关节 + 1 个夹爪控制，总动作维度 8。
