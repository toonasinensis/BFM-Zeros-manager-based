# Manager-Only 推理命令

本文只记录精简版仓库 `/home/thl/wt_wbc/BFM-Zero-ManagerOnly` 的推理命令。

默认 checkpoint：

```bash
/home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0
```

默认 Python：

```bash
/home/thl/wt_wbc/BFM-Zero/.venv/bin/python
```

## 1. MuJoCo Tracking 可视化

这个命令用于看 motion tracking 效果，默认 motion `25`，不保存视频，窗口实时播放。

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

.venv/bin/python \
  -m bfm.tracking_inference_mujoco \
  --model-folder /home/thl/wt_wbc/BFM-Zero-ManagerOnly/results/obs0.25expert1-20260622-161544 \
  --data-path /home/thl/wt_wbc/BFM-Zero-ManagerOnly/bfm/data/named_lafan \
  --motion-list 25 \
  --steps 100000 \
  --device cpu \
  --policy-runtime onnx \
  --no-headless \
  --show-reference \
  --real-time \
  --progress-every 200
```

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
  --no-real-time
```

输出：

- `exported/FBcprAuxModel.onnx`
- `tracking_inference_mujoco/zs_<motion_id>.pkl`
- `tracking_inference_mujoco/summary.json`

## 2. Manager/IsaacLab Tracking 推理

这个命令用于跑 manager env 的 checkpoint 推理 smoke。需要 IsaacLab/IsaacSim 环境可用。

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

CUDA_VISIBLE_DEVICES=0 ACCEPT_EULA=Y /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.tracking_inference_manager \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --motion-list 25 \
  --steps 100 \
  --device cuda:0 \
  --headless
```

开窗口看 manager env：

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

CUDA_VISIBLE_DEVICES=0 ACCEPT_EULA=Y /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.tracking_inference_manager \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --motion-list 25 \
  --steps 500 \
  --device cuda:0 \
  --no-headless
```

如果想让程序只看见物理 GPU 3，把 `CUDA_VISIBLE_DEVICES=0` 改成：

```bash
CUDA_VISIBLE_DEVICES=3
```

命令里的 `--device cuda:0` 不用改。

## 3. Reward Inference

这个命令用于从 replay buffer 采样，按 reward task 重新打 reward，然后调用 `model.reward_wr_inference()` 得到 reward-z。

短 smoke：

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=egl PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.reward_inference \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --tasks move-ego-0-0 \
  --num-samples 128 \
  --n-inferences 1 \
  --device cuda \
  --buffer-device cuda \
  --max-workers 1 \
  --skip-rollouts \
  --no-export-onnx
```

全量 reward inference：

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=egl PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.reward_inference \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --num-samples 150000 \
  --n-inferences 1 \
  --device cpu \
  --buffer-device cpu \
  --max-workers 8 \
  --skip-rollouts
```

只算某几个 task：

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=egl PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.reward_inference \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --tasks move-ego-0-0 move-ego-0-0.7 rotate-z-5-0.5 \
  --num-samples 150000 \
  --n-inferences 1 \
  --device cpu \
  --buffer-device cpu \
  --max-workers 8 \
  --skip-rollouts
```

输出：

- `reward_inference/reward_locomotion.pkl`
- `reward_inference/summary.json`
- 可选 `exported/FBcprAuxModel.onnx`

## 4. Reward-Z MuJoCo 可视化

这个命令会先算 `move-ego-0-0` 的 reward-z，然后用轻量 MuJoCo viewer 看这个 z 控出来的动作。

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=glfw PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.reward_inference \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --tasks move-ego-0-0 \
  --num-samples 4096 \
  --n-inferences 1 \
  --device cpu \
  --buffer-device cpu \
  --max-workers 1 \
  --no-skip-rollouts \
  --rollout-task-limit 1 \
  --episode-length 100000 \
  --no-headless \
  --real-time
```

headless rollout smoke：

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=egl PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.reward_inference \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --tasks move-ego-0-0 \
  --num-samples 1024 \
  --n-inferences 1 \
  --device cpu \
  --buffer-device cpu \
  --max-workers 1 \
  --no-skip-rollouts \
  --rollout-task-limit 1 \
  --episode-length 200 \
  --headless \
  --no-real-time
```

## 5. 常见参数

- `--model-folder`：结果目录，里面需要有 `checkpoint/`。
- `--motion-list`：tracking 推理使用的 motion id，例如 `25`。
- `--steps`：tracking rollout 步数。
- `--device cpu`：MuJoCo/ONNX 可视化推荐 CPU，少占 GPU。
- `--policy-runtime onnx`：MuJoCo tracking 默认用 ONNX policy。
- `--buffer-device cpu`：reward inference 推荐 CPU，避免 replay buffer 挤爆 GPU。
- `--max-workers`：reward relabel 并行数；小 smoke 用 `1`，全量可以用 `8`。
- `--skip-rollouts`：只算 reward-z，不看动作。
- `--no-skip-rollouts --no-headless`：算 reward-z 后打开 MuJoCo 窗口看动作。
