# Manager-Only MuJoCo 推理对齐记录

## 背景

精简版 `bfm.tracking_inference_mujoco` 用来在 manager-only 仓库里直接跑 checkpoint 的 MuJoCo viewer，不再依赖完整仓库旧的 `humanoidverse/envs` 和 `humanoidverse/simulator`。

这条路径需要和完整仓库命令对齐：

```bash
cd /home/thl/wt_wbc/BFM-Zero

MUJOCO_GL=egl PYTHONBREAKPOINT=0 .venv/bin/python -m humanoidverse.tracking_inference \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --data-path /home/thl/wt_wbc/BFM-Zero/humanoidverse/data/lafan_29dof.pkl \
  --motion-list 25 \
  --steps 20 \
  --device cpu \
  --simulator mujoco \
  --policy-runtime onnx \
  --disable-dr \
  --disable-obs-noise \
  --headless \
  --show-reference \
  --no-real-time \
  --progress-every 5
```

## 已修正的差异

- MuJoCo XML：精简版默认改为完整仓库同款 `bfm/data/robots/g1/scene_29dof_freebase_mujoco.xml`，不再用 motion asset `g1_29dof.xml`。
- body 顺序：scene XML 中过滤 `hand` body 后必须逐项等于 no-head `robot.body_names`。
- actuator 顺序：scene XML 有 6 个 freebase actuator，policy 的 29 维 torque 写入 `ctrl[6:]`。
- root angular velocity：motion lib 给的是 world angular velocity，写入 MuJoCo freejoint `qvel[3:6]` 前要转成 root-local。
- observation `base_ang_vel`：MuJoCo freejoint `qvel[3:6]` 已经按 local 语义使用，obs 里只乘 `0.25`，不要再 rotate inverse 一次。
- action 处理：policy action 先按旧 env 的 normalize 规则放大到 `[-5, 5]`，再按 `action_scale=0.25` 和 `effort/stiffness` 生成 PD target。
- torque limit：按完整仓库旧 env 使用 `dof_effort_limit_list`，不要额外乘 `0.8`。
- rollout 时序：默认先做一次 zero-action warmup step，对齐完整仓库 `tracking_inference.py` 的 reset 后 warmup。

## 当前验证结果

精简版 20 step headless：

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=egl PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.tracking_inference_mujoco \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --data-path /home/thl/wt_wbc/BFM-Zero-ManagerOnly/bfm/data/lafan_29dof.pkl \
  --motion-list 25 \
  --steps 20 \
  --device cpu \
  --policy-runtime onnx \
  --headless \
  --show-reference \
  --no-real-time \
  --progress-every 5
```

结果：

- ONNX/Torch action max diff: `1.18371e-06`
- `joint_abs_error_mean`: `0.07747638281434774`

精简版 200 step headless：

- ONNX/Torch action max diff: `1.18371e-06`
- `joint_abs_error_mean`: `0.08474992379546166`

viewer smoke：

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=glfw PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.tracking_inference_mujoco \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --data-path /home/thl/wt_wbc/BFM-Zero-ManagerOnly/bfm/data/lafan_29dof.pkl \
  --motion-list 25 \
  --steps 80 \
  --device cpu \
  --policy-runtime onnx \
  --no-headless \
  --show-reference \
  --real-time \
  --progress-every 40
```

结果：

- 正常打开 viewer 并退出。
- `joint_abs_error_mean`: `0.07917609005235135`

## Trace 对比

保存前几步 trace：

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=egl PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.tracking_inference_mujoco \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --data-path /home/thl/wt_wbc/BFM-Zero-ManagerOnly/bfm/data/lafan_29dof.pkl \
  --motion-list 25 \
  --steps 20 \
  --device cpu \
  --policy-runtime onnx \
  --headless \
  --show-reference \
  --no-real-time \
  --debug-trace-path /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0/tracking_inference_mujoco/slim_debug_trace_25.npz
```

trace 里包含：

- reset qpos/qvel
- reset world/local root angular velocity
- warmup raw/processed action、target、torque、qpos/qvel
- 前几步 raw/processed action、target、torque、qpos/qvel、state、last_action、history_actor

## 注意

这个 checkpoint 的 `obs_space["privileged_state"]` 是 `(448,)`，精简版 manager checkpoint 推理按 448 对齐。完整仓库 old MuJoCo env 启动日志里会打印 `max_local_self: 447`，但当前 actor 推理只吃 `state/last_action/history_actor`，可视化动作主要由这三个输入和 z 决定。
