# BFM-Zero Manager-Only 版本

这个仓库是从完整 `BFM-Zero` 裁出来的 manager-only 训练仓库。

## 保留内容

- standalone manager 训练入口：`bfm.train_manager_standalone`
- hook 化训练主循环：`bfm/manager_training/standalone_*`
- ManagerBased G1 no-head env：`bfm/manager_envs/g1`
- 当前训练用 agent/model/buffer/eval 底座：`agents/fb -> fb_cpr -> fb_cpr_aux`
- no-head G1 robot config、G1 asset、`lafan_29dof_10s-clipped.pkl`

## 删除内容

- 旧 `bfm/train.py`
- 旧 `bfm/envs`
- 旧 `bfm/simulator`
- 旧 Isaac env wrapper `agents/envs/bfm_isaac.py`
- 旧 non-manager eval 分支

## 单卡训练

让进程只看见物理 GPU 3，并在程序内部使用 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=3 ACCEPT_EULA=Y .venv/bin/python -m bfm.train_manager_standalone \
  --settings.device cuda:0 \
  --settings.agent-device cuda \
  --settings.buffer-device cuda \
  --settings.work-dir results/bfmzero-manager-slim-dr-physgpu3 \
  --settings.wandb-name bfmzero-manager-slim-dr-physgpu3 \
  --settings.wandb-group bfmzero-manager-slim-dr \
  --settings.use-wandb \
  --settings.enable-domain-randomization
```

注意：使用 `CUDA_VISIBLE_DEVICES=N` 后，代码里仍然写 `cuda:0`，因为此时可见的第 0 张卡就是物理 GPU `N`。

## Manager Checkpoint 推理

精简版保留 manager/IsaacLab 推理入口：

```bash
CUDA_VISIBLE_DEVICES=3 ACCEPT_EULA=Y .venv/bin/python -m bfm.tracking_inference_manager \
  --model-folder results/bfmzero-manager-slim-dr-physgpu3 \
  --motion-list 25 \
  --steps 100 \
  --device cuda:0 \
  --no-headless
```
 
## MuJoCo 可视化推理

精简版也保留一个轻量 MuJoCo viewer 入口，不依赖旧 `bfm/envs` 或旧 `simulator`：

- 默认使用完整仓库 MuJoCo 可视化同款 XML：`bfm/data/robots/g1/scene_29dof_freebase_mujoco.xml`。
- action 处理对齐旧路径：policy `[-1, 1]` action 先放大到 `[-5, 5]`，再按 `action_scale=0.25` 和 `effort/stiffness` 生成 PD target。
- reset 会把 motion lib 的 world root angular velocity 转成 MuJoCo freejoint 的 local angular velocity。
- 默认执行一次 zero-action warmup step，用来对齐完整仓库 `tracking_inference.py` 的 rollout 时序；如需关闭可加 `--no-full-repo-warmup-step`。

```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly


MUJOCO_GL=glfw PYTHONBREAKPOINT=0 .venv/bin/python \
  -m bfm.tracking_inference_mujoco \
  --model-folder results/bfmzero-manager-named50hz-smoothsigma5-ablation-cuda0 \
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

## Reward Inference

精简版保留 reward inference 入口：`bfm.reward_inference`。
```bash
cd /home/thl/wt_wbc/BFM-Zero-ManagerOnly

MUJOCO_GL=egl PYTHONBREAKPOINT=0 /home/thl/wt_wbc/BFM-Zero/.venv/bin/python \
  -m bfm.reward_inference \
  --model-folder /home/thl/wt_wbc/BFM-Zero/results/bfmzero-manager-nohead-minimal-cuda0 \
  --tasks move-ego-0-0 \
  --num-samples 128 \
  --n-inferences 1 \
  --device cpu \
  --buffer-device cpu \
  --max-workers 1 \
  --skip-rollouts \
  --no-export-onnx
```

 

开窗口观察 reward-z 动作：

```bash
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

输出：

- `reward_inference/reward_locomotion.pkl`
- `reward_inference/summary.json`
- 可选 `exported/FBcprAuxModel.onnx`

## 快速检查

```bash
.venv/bin/python -m py_compile $(find bfm -name '*.py' -print)
ACCEPT_EULA=Y .venv/bin/python -m bfm.train_manager_standalone --help
ACCEPT_EULA=Y .venv/bin/python -m bfm.tracking_inference_manager --help
.venv/bin/python -m bfm.tracking_inference_mujoco --help
/home/thl/wt_wbc/BFM-Zero/.venv/bin/python -m bfm.reward_inference --help
```
