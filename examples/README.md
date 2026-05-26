# MuJoCo Environment Examples

运行前先进入项目并激活环境：

```bash
cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
source /home/mkls/anaconda3/etc/profile.d/conda.sh
conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
unset MUJOCO_GL
```

## 01_show_scene.py

只加载环境并显示 reset 后的场景：

```bash
python examples/01_show_scene.py
```

适合先学习：

- 如何 import 环境
- 如何创建 `CubeGraspEnv`
- 如何调用 `env.reset()`
- 如何打开 `mujoco.viewer`

## 02_play_scripted_grasp.py

加载环境并循环播放脚本专家抓取：

```bash
python examples/02_play_scripted_grasp.py
```

适合继续学习：

- 如何从 `scripted_grasp.py` 调用 `scripted_grasp_action(env, step)`
- 如何调用 `env.step(action)`
- MuJoCo viewer 如何实时刷新
- 一次抓取 episode 如何 reset

如果窗口打不开，说明当前终端没有图形显示环境。需要本地桌面、VNC、NoMachine 或 X11 转发。

## 03_get_camera_image.py

从 XML 里定义的固定相机渲染图像，并保存成 PNG：

```bash
MUJOCO_GL=egl python examples/03_get_camera_image.py
```

当前相机名：

```text
front
top_oblique
```

只保存一个相机：

```bash
MUJOCO_GL=egl python examples/03_get_camera_image.py --camera front
```

输出目录：

```text
outputs/camera_images/
```

## 04_live_camera_view.py

实时查看固定相机的视频画面，不保存图片：

```bash
unset MUJOCO_GL
python examples/04_live_camera_view.py --camera front
```

也可以看顶视角相机：

```bash
unset MUJOCO_GL
python examples/04_live_camera_view.py --camera top_oblique
```

窗口快捷键：

```text
关闭窗口 退出
```

如果你明确想用 OpenCV 窗口，可以改用：

```bash
MUJOCO_GL=egl python examples/04_live_camera_view.py --backend opencv --camera front
```

## 05_camera_stream_server.py

如果没有图形桌面，使用浏览器实时查看相机视频流：

```bash
MUJOCO_GL=egl python examples/05_camera_stream_server.py --camera front --host 0.0.0.0 --port 8000
```

然后在浏览器打开：

```text
http://服务器IP:8000
```

本机运行时打开：

```text
http://127.0.0.1:8000
```

## 06_keyboard_control_arm.py

用键盘直接控制 6 个机械臂关节和夹爪：

```bash
unset MUJOCO_GL
python examples/06_keyboard_control_arm.py --camera front
```

按键：

```text
A / D : panda_joint1 -/+
W / S : panda_joint2 +/-
Q / E : panda_joint3 -/+
R / F : panda_joint4 +/-
T / G : panda_joint5 +/-
Y / H : panda_joint6 +/-
Z / X : gripper open/close
B     : 回到 home 姿态
N     : 重置方块和机械臂
P     : 在终端打印当前目标值
```

## 07_keyboard_control_ee.py

用键盘控制夹爪末端目标位置，比直接控制关节更适合手动抓取：

```bash
unset MUJOCO_GL
python examples/07_keyboard_control_ee.py --camera front
```

基础按键：

```text
W / S : x forward/back
A / D : y left/right
R / F : raise/lower gripper
Z / X : open/close gripper fully
B     : 回到 home 目标点
N     : 重置方块和机械臂
P     : 在终端打印当前末端目标和抓取状态
```

辅助抓取快捷键：

```text
1 : 移动到方块上方
2 : 下探到方块
3 : 闭合夹爪
4 : 抬起方块
5 : 移动到托盘上方
6 : 下降到托盘
7 : 张开夹爪
```

场景参照物：

```text
红色轴：桌面 X 方向
绿色轴：桌面 Y 方向
蓝色轴：桌面 Z 方向
绿色竖杆/横杠：cube 抬起高度参考线
青色十字准星：夹爪末端对准点
橙色/紫色 cube：随机摆放的干扰方块，当前任务仍然抓蓝色 cube
```

## 08_replay_recorded_actions.py

回放 raw npz 数据里的动作轨迹：

```bash
unset MUJOCO_GL
python examples/08_replay_recorded_actions.py
```

只回放第 3 条：

```bash
python examples/08_replay_recorded_actions.py --episode 3 --camera top_oblique
```

循环播放：

```bash
python examples/08_replay_recorded_actions.py --loop
```
