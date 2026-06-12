# Manager Standalone 训练调用链

本文档记录当前 standalone manager-only 训练入口的真实调用链。它用于防止后续上下文压缩后忘记：这条链路不走 `bfm/train.py` 的 `TrainConfig/Workspace`，入口和训练主循环都在新文件里。

## 入口命令

默认全量训练入口：

```bash
.venv/bin/python -m bfm.train_manager_standalone
```

推荐显式指定 workdir：

```bash
.venv/bin/python -m bfm.train_manager_standalone \
  --settings.device cuda:0 \
  --settings.work-dir results/bfmzero-manager-standalone-full-cuda0
```

当前入口文件是 `bfm/train_manager_standalone.py`：

```text
tyro.cli(main)
  -> main(settings=BFMZeroManagerTrainSettings.from_env())
  -> run_standalone_manager_training(settings)
```

## 顶层调用链

```text
bfm/train_manager_standalone.py
  -> bfm/manager_training/standalone_runner.py
    -> set_default_cuda_device(settings.device)
    -> build_standalone_manager_train_config(settings)
    -> StandaloneManagerTrainer(cfg)
    -> trainer.train()
      -> trainer.train_online()
```

这条链路没有 import `bfm.train`，也不构建旧的 `Workspace`。

## 配置构建

文件：`bfm/manager_training/standalone_config.py`

主要对象：

- `BFMZeroManagerTrainSettings`
- `StandaloneManagerTrainingConfig`
- `build_standalone_manager_train_config()`
- `build_bfmzero_aux_agent_config()`

`BFMZeroManagerTrainSettings` 是 CLI/env 层配置。默认值对齐当前 manager full 训练：

- `device=cuda:0`
- `agent_device=cuda`
- `buffer_device=cuda`
- `online_parallel_envs=1024`
- `num_env_steps=384000000`
- `log_every_updates=10240`
- `update_agent_every=1024`
- `num_seed_steps=10240`
- `num_agent_updates=16`
- `batch_size=1024`
- `buffer_size=3000000`
- `checkpoint_every_steps=1024000`
- `enable_eval=True`
- `prioritization=True`
- `use_wandb=True`
- `motion_file=bfm/data/lafan_29dof_10s-clipped.pkl`
- `robot_config=g1/g1_29dof_hard_waist_no_head`

`build_standalone_manager_train_config()` 固定构建：

- `FBcprAuxAgentConfig`
- `BFMZeroManagerIsaacConfig`
- `BFMZeroManagerTrackingEvaluationConfig`，当 `enable_eval=True`

注意：`training_max_num_seqs=None` 时 expert buffer 使用全量 motion。smoke 时可以显式设小，例如 `--settings.training-max-num-seqs 4`。

## Trainer 初始化

文件：`bfm/manager_training/standalone_trainer.py`

`StandaloneManagerTrainer.__init__(cfg)` 做这些事：

1. `cfg.env.build(num_envs=cfg.online_parallel_envs)` 构建 manager env。
2. 读取 `single_observation_space` 和 `single_action_space`。
3. 检查 obs 必须包含 `time`，然后从 agent obs space 中删除 `time`。
4. 创建 `work_dir`。
5. 保存：
   - `manager_env_info.json`
   - `config.json`
6. 创建 `train_log.txt` 的 `CSVLogger`。
7. `set_seed_everywhere(cfg.seed)`。
8. `load_agent_or_build()` 加载 checkpoint 或新建 agent。
9. 构建 eval 对象。
10. 如果 `use_wandb=True`，初始化 W&B。

agent obs keys 实际仍是：

```text
state
privileged_state
last_action
history_actor
```

`time` 只用于 rollout step count，不进入 agent observation。

## Expert Buffer

文件：`bfm/manager_training/standalone_expert.py`

调用位置：

```text
StandaloneManagerTrainer.train_online()
  -> self._load_expert_buffer()
    -> load_manager_expert_trajectories(...)
```

内部流程：

1. 新建 `BFMZeroMotionProvider`。
2. `provider.load_for_training(max_num_seqs=cfg.training_max_num_seqs)`。
3. 对每条已加载 motion 生成整段 expert episode。
4. 用 motion reference 计算：
   - `state`
   - `last_action`
   - `privileged_state`
   - `history_actor`
   - `terminated`
   - `truncated`
   - `motion_id`
5. 构建 `TrajectoryDictBuffer`，作为 `replay_buffer["expert_slicer"]`。

这部分仍复用 BFM-Zero 当前 pkl/MotionProvider 语义，不切 `.npz` loader。

## Replay Buffer

文件：`bfm/manager_training/standalone_checkpoint.py`

调用位置：

```text
StandaloneManagerTrainer.train_online()
  -> self._allocate_replay_buffer(expert_buffer)
    -> load_or_create_train_buffer(work_dir, cfg)
```

训练 replay buffer 固定使用：

```text
TrajectoryDictBufferMultiDim
```

key 保持 agent 当前需要的格式：

```text
observation
action
z
terminated
truncated
step_count
reward
aux_rewards
```

最后组合成：

```python
replay_buffer = {
    "train": train_buffer,
    "expert_slicer": expert_buffer,
}
```

## 主训练循环

文件：`bfm/manager_training/standalone_trainer.py`

主函数：

```text
StandaloneManagerTrainer.train()
  -> train_online()
```

`train_online()` 顺序：

1. 加载 expert buffer。
2. 创建/加载 train replay buffer。
3. `td, info = train_env.reset()`。
4. 创建 step checker：
   - checkpoint
   - eval
   - agent update
   - train log
5. 按 `online_parallel_envs` 为步长循环到 `num_env_steps`。

每个循环 step：

```text
1. 到 checkpoint 时间则 save_checkpoint()
2. 到 eval 时间则 eval()
3. 把 env obs 转成 torch，并 pop 掉 time 得到 step_count
4. maybe_update_rollout_context()
5. seed 阶段采随机 action，否则 agent.act()
6. train_env.step(action)
7. _transition_data() 打包 transition
8. replay_buffer["train"].extend(data)
9. 到 update 时间则 agent.update(replay_buffer, t)
10. 到 log 时间则写 train_log.txt 和 wandb train/...
```

`_transition_data()` 会把 manager env 的 `info["aux_rewards"]` 打成：

```text
data["aux_rewards"][reward_name] = value[None, ..., None]
```

所以 aux reward replay packing 仍保持当前 agent 需要的格式。

## Eval 和 Prioritization

eval 固定用：

```text
BFMZeroManagerTrackingEvaluation
```

W&B key 保持：

```text
eval/humanoidverse_tracking_eval/...
```

如果 `prioritization=True`：

1. eval 结果里读取每条 motion 的 `emd`。
2. clamp 到 `[0.5, 2.0]`。
3. 乘 `prioritization_scale=2.0`。
4. 默认 `prioritization_mode=exp`，即 `2 ** priority`。
5. 同时更新：
   - `train_env.update_motion_sampling_weights(...)`
   - `expert_slicer.update_priorities(...)`

如果 `prioritization=True` 但 `enable_eval=False`，config build 阶段直接报错。

## Checkpoint

文件：`bfm/manager_training/standalone_checkpoint.py`

目录结构保持兼容：

```text
work_dir/
  checkpoint/
    model/
    optimizers/
    buffers/
      train/
    train_status.json
```

加载逻辑：

```text
如果 checkpoint/train_status.json 存在：
  读取 time
  cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
否则：
  cfg.agent.build(obs_space=obs_space, action_dim=action_dim)
```

保存逻辑：

```text
agent.save(checkpoint_dir)
如果 checkpoint_buffer=True：
  replay_buffer["train"].save(checkpoint_dir / "buffers" / "train")
写 train_status.json
```

## 日志

本地日志：

```text
work_dir/train_log.txt
```

W&B train key：

```text
train/<metric_name>
```

W&B eval key：

```text
eval/humanoidverse_tracking_eval/<metric_name>
```

这些名字没有为了 standalone 改掉，方便和旧曲线对比。

## Smoke 命令

如果 cuda0 没有别的长训，可以跑 GPU smoke：

```bash
rm -rf results/bfmzero-manager-standalone-smoke

.venv/bin/python -m bfm.train_manager_standalone \
  --settings.device cuda:0 \
  --settings.work-dir results/bfmzero-manager-standalone-smoke \
  --settings.no-compile-model \
  --settings.online-parallel-envs 4 \
  --settings.num-env-steps 128 \
  --settings.log-every-updates 1 \
  --settings.update-agent-every 32 \
  --settings.num-seed-steps 32 \
  --settings.num-agent-updates 1 \
  --settings.batch-size 32 \
  --settings.buffer-size 2048 \
  --settings.checkpoint-every-steps 1000000 \
  --settings.no-checkpoint-buffer \
  --settings.no-enable-eval \
  --settings.no-prioritization \
  --settings.no-use-wandb \
  --settings.disable-tqdm \
  --settings.training-max-num-seqs 4
```

如果 cuda0 正在跑长训，GPU 显存不够，可以只为 smoke 把 agent 和 buffer 放 CPU：

```bash
rm -rf results/bfmzero-manager-standalone-smoke

.venv/bin/python -m bfm.train_manager_standalone \
  --settings.device cuda:0 \
  --settings.agent-device cpu \
  --settings.buffer-device cpu \
  --settings.work-dir results/bfmzero-manager-standalone-smoke \
  --settings.no-compile-model \
  --settings.online-parallel-envs 4 \
  --settings.num-env-steps 128 \
  --settings.log-every-updates 1 \
  --settings.update-agent-every 32 \
  --settings.num-seed-steps 32 \
  --settings.num-agent-updates 1 \
  --settings.batch-size 32 \
  --settings.buffer-size 2048 \
  --settings.checkpoint-every-steps 1000000 \
  --settings.no-checkpoint-buffer \
  --settings.no-enable-eval \
  --settings.no-prioritization \
  --settings.no-use-wandb \
  --settings.disable-tqdm \
  --settings.training-max-num-seqs 4
```

这个 CPU-agent smoke 只用于验证链路，不代表正式训练性能。

## 当前耦合点

standalone trainer 不再耦合旧 `TrainConfig/Workspace`，但仍复用这些底层模块：

- `FBcprAuxAgent`
- `BFMZeroManagerIsaacConfig`
- `BFMZeroManagerTrackingEvaluation`
- `TrajectoryDictBuffer`
- `TrajectoryDictBufferMultiDim`
- `CSVLogger`
- `EveryNStepsChecker`
- `BFMZeroMotionProvider`
- manager env 的 no-head G1 spec/motion/reward/obs 逻辑

这些复用是有意保留的：standalone 只重写训练 orchestration，不复制 agent/env/eval/buffer 的核心实现。
