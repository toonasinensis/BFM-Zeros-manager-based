# Manager Standalone Hook 版训练命令

日期: 2026-06-11

这份记录当前 hook 化后的 standalone manager-only 训练入口。入口不走 `bfm/train.py` 的 `TrainConfig/Workspace`，而是直接走:

```text
bfm.train_manager_standalone
  -> manager_training.standalone_runner
  -> manager_training.standalone_trainer
  -> TrainContext + hooks
```

定时任务现在在 hooks 里:

- `CheckpointHook`
- `EvaluationHook`
- `AgentUpdateHook`
- `TrainLogHook`

## 默认长训命令

默认配置已经是当前 manager no-head full train 参数。建议显式写全，方便 W&B 和 workdir 对齐:

```bash
cd /home/thl/wt_wbc/BFM-Zero

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
BFM_DISABLE_TORCH_COMPILE=1 \
.venv/bin/python -m bfm.train_manager_standalone \
  --settings.device cuda:0 \
  --settings.agent-device cuda \
  --settings.buffer-device cuda \
  --settings.work-dir results/bfmzero-manager-standalone-hook-cuda0 \
  --settings.no-compile-model \
  --settings.online-parallel-envs 1024 \
  --settings.num-env-steps 384000000 \
  --settings.log-every-updates 10240 \
  --settings.update-agent-every 1024 \
  --settings.num-seed-steps 10240 \
  --settings.num-agent-updates 16 \
  --settings.batch-size 1024 \
  --settings.buffer-size 3000000 \
  --settings.checkpoint-every-steps 1024000 \
  --settings.checkpoint-buffer \
  --settings.enable-eval \
  --settings.eval-every-steps 1024000 \
  --settings.prioritization \
  --settings.enable-domain-randomization \
  --settings.use-wandb \
  --settings.no-disable-tqdm \
  --settings.wandb-name bfmzero-manager-standalone-dr-cuda0-$(date +%Y%m%d-%H%M%S) \
  --settings.wandb-entity xiechunyang1-hajimi \
  --settings.wandb-project hajimi \
  --settings.wandb-group bfmzero-manager-standalone-hook
```

输出目录:

```text
results/bfmzero-manager-standalone-hook-cuda0/
```

## Smoke 命令

这个会启动 Isaac，小规模验证 hook 版 trainer 的 build/reset/rollout/update 链路。cuda0 如果有长训在跑，换 `cuda:1` 或先不要跑。

```bash
cd /home/thl/wt_wbc/BFM-Zero

BFM_DISABLE_TORCH_COMPILE=1 \
.venv/bin/python -m bfm.train_manager_standalone \
  --settings.device cuda:0 \
  --settings.agent-device cuda \
  --settings.buffer-device cuda \
  --settings.work-dir results/bfmzero-manager-standalone-hook-smoke \
  --settings.no-compile-model \
  --settings.online-parallel-envs 4 \
  --settings.num-env-steps 128 \
  --settings.num-seed-steps 32 \
  --settings.update-agent-every 32 \
  --settings.num-agent-updates 1 \
  --settings.batch-size 32 \
  --settings.buffer-size 4096 \
  --settings.checkpoint-every-steps 1000000000 \
  --settings.no-checkpoint-buffer \
  --settings.no-enable-eval \
  --settings.no-prioritization \
  --settings.no-enable-domain-randomization \
  --settings.no-use-wandb \
  --settings.no-disable-tqdm \
  --settings.training-max-num-seqs 4
```

## 只测 hooks

不启动 Isaac，只测 checkpoint/eval/log/update hooks 的触发和 `TrainContext` 传参:

```bash
cd /home/thl/wt_wbc/BFM-Zero

.venv/bin/python -m bfm.manager_training.standalone_hook_smoke
```

## 换 GPU

例如跑 `cuda:1`:

```bash
--settings.device cuda:1 \
--settings.work-dir results/bfmzero-manager-standalone-hook-cuda1
```

通常保持:

```bash
--settings.agent-device cuda
--settings.buffer-device cuda
```

入口会先 `torch.cuda.set_device(settings.device)`，所以 agent/buffer 用 `cuda` 会落到当前设置的卡。

## 只让进程看到一张卡

如果想让训练进程完全看不到别的显卡，用 `CUDA_VISIBLE_DEVICES`。注意 CUDA 会把可见显卡重新编号，所以只暴露物理 1 号卡时，程序内部仍然写 `cuda:0`:

```bash
cd /home/thl/wt_wbc/BFM-Zero

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
CUDA_VISIBLE_DEVICES=1 \
BFM_DISABLE_TORCH_COMPILE=1 \
.venv/bin/python -m bfm.train_manager_standalone \
  --settings.device cuda:0 \
  --settings.agent-device cuda \
  --settings.buffer-device cuda \
  --settings.work-dir results/bfmzero-manager-standalone-hook-physgpu1 \
  --settings.no-compile-model \
  --settings.online-parallel-envs 1024 \
  --settings.num-env-steps 384000000 \
  --settings.log-every-updates 10240 \
  --settings.update-agent-every 1024 \
  --settings.num-seed-steps 10240 \
  --settings.num-agent-updates 16 \
  --settings.batch-size 1024 \
  --settings.buffer-size 3000000 \
  --settings.checkpoint-every-steps 1024000 \
  --settings.checkpoint-buffer \
  --settings.enable-eval \
  --settings.eval-every-steps 1024000 \
  --settings.prioritization \
  --settings.enable-domain-randomization \
  --settings.use-wandb \
  --settings.no-disable-tqdm \
  --settings.wandb-name bfmzero-manager-standalone-dr-physgpu1-$(date +%Y%m%d-%H%M%S) \
  --settings.wandb-entity xiechunyang1-hajimi \
  --settings.wandb-project hajimi \
  --settings.wandb-group bfmzero-manager-standalone-hook
```

如果只想用物理 0 号卡，就改成:

```bash
CUDA_VISIBLE_DEVICES=0
```

## W&B Run Name

现在 standalone 入口支持单独设置 W&B run name，不会再全部叫 `BFM-Zero`。

命令行写法:

```bash
--settings.wandb-name bfmzero-manager-dr-cuda0-$(date +%Y%m%d-%H%M%S)
```

环境变量写法:

```bash
BFMZERO_WANDB_NAME=bfmzero-manager-dr-cuda0-$(date +%Y%m%d-%H%M%S) \
.venv/bin/python -m bfm.train_manager_standalone
```

如果不显式传 `wandb_name`，代码会用 `work_dir` 的最后一级目录名作为 run name。

## Domain Randomization

standalone manager 训练默认开启:

```text
enable_domain_randomization=True
```

当前 manager DR 已接入这些旧 BFM-Zero 训练项:

- `randomize_link_mass`: startup，body 列表来自 no-head robot config 的 `randomize_link_body_names`，范围 `[0.95, 1.05]`。
- `randomize_friction`: startup，static/dynamic friction 范围 `[0.5, 1.25]`。
- `randomize_base_com`: startup，`torso_link` COM 偏置 xyz 范围 `[-0.02, 0.02]`。
- `randomize_default_dof_pos`: reset，范围 `[-0.02, 0.02]`，同时更新 manager action offset 和 `dof_pos` obs offset。
- `push_robots`: interval，`[1, 3]` 秒触发，xy 和 roll/pitch/yaw 最大速度 `0.5`。

eval/parity reset 会设置 manager eval mode:

- interval push 在 eval 期间跳过。
- `reset_to_motion()` / `reset_to_motions()` 不会随机 default dof offset。

关闭 DR:

```bash
--settings.no-enable-domain-randomization
```

环境变量关闭:

```bash
BFMZERO_MANAGER_ENABLE_DOMAIN_RANDOMIZATION=0
```

## 关闭 W&B

本地调试:

```bash
--settings.no-use-wandb
```

或者保留 W&B 但离线:

```bash
WANDB_MODE=offline
```

如果网页看不到曲线，优先用命令里的 `env -u http_proxy ...` 去掉代理；之前机器上 W&B filestream 会被本地 proxy 搞超时。

## 默认参数速查

当前默认值来自 `BFMZeroManagerTrainSettings`:

```text
device=cuda:0
agent_device=cuda
buffer_device=cuda
online_parallel_envs=1024
num_env_steps=384000000
log_every_updates=10240
update_agent_every=1024
num_seed_steps=10240
num_agent_updates=16
batch_size=1024
buffer_size=3000000
checkpoint_every_steps=1024000
eval_every_steps=1024000
enable_eval=True
prioritization=True
use_wandb=True
wandb_name=
motion_file=bfm/data/lafan_29dof_10s-clipped.pkl
robot_config=g1/g1_29dof_hard_waist_no_head
enable_domain_randomization=True
```

## 注意事项

- `prioritization=True` 必须同时 `enable_eval=True`。
- checkpoint 仍写到 `work_dir/checkpoint/`，结构保持兼容。
- W&B key 保持 `train/...` 和 `eval/humanoidverse_tracking_eval/...`。
- replay buffer key 保持当前 agent 需要的格式，没有因为 hook 重构改数据格式。
