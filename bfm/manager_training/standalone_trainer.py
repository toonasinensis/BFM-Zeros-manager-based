from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils._pytree import tree_map
from tqdm import tqdm

from bfm.agents.buffers.transition import dtype_numpytotorch_lower_precision
from bfm.agents.misc.loggers import CSVLogger
from bfm.agents.utils import set_seed_everywhere

from .standalone_checkpoint import load_agent_or_build, load_or_create_train_buffer
from .standalone_context import TrainContext, TrainRuntime, TrainState
from .standalone_hooks import AgentUpdateHook, CheckpointHook, EvaluationHook, HookList, TrainLogHook

TRAIN_LOG_FILENAME = "train_log.txt"


def _randomize_episode_length_buf_once(train_env) -> None:
    episode_length_buf = train_env.episode_length_buf
    max_episode_length = int(getattr(train_env.unwrapped, "max_episode_length", 0) or 0)
    if max_episode_length <= 1:
        return
    episode_length_buf[:] = torch.randint(
        0,
        max_episode_length,
        episode_length_buf.shape,
        device=episode_length_buf.device,
        dtype=episode_length_buf.dtype,
    )


class StandaloneManagerTrainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
        self.obs_space = self.train_env.single_observation_space
        self.action_space = self.train_env.single_action_space
        if len(self.action_space.shape) != 1:
            raise AssertionError("Only 1D manager action space is supported.")
        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        with (self.work_dir / "manager_env_info.json").open("w") as file:
            json.dump(self.train_env_info, file, indent=2)
        with (self.work_dir / "config.json").open("w") as f:
            json.dump(self.cfg.to_json_dict(), f, indent=4)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)
        set_seed_everywhere(self.cfg.seed)
        self.agent, self._checkpoint_time = load_agent_or_build(
            self.work_dir,
            self.cfg,
            obs_space=self.obs_space,
            action_dim=self.action_dim,
        )
        self.agent._model.train()
        self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        self.evaluate = len(self.evaluations) > 0
        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()}
        self.priorization_eval_name = self._find_prioritization_eval_name()
        if self.cfg.use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        wandb_dir = Path(os.environ.get("WANDB_DIR", "./_wandb"))
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_name = self.cfg.wandb_name or Path(self.cfg.work_dir).name
        wandb.init(
            entity=self.cfg.wandb_ename,
            project=self.cfg.wandb_pname,
            group=self.cfg.wandb_gname,
            name=wandb_name,
            config=self.cfg.to_json_dict(),
            dir=str(wandb_dir),
        )

    def _find_prioritization_eval_name(self) -> str | None:
        from bfm.agents.evaluations.bfmzero_manager import BFMZeroManagerTrackingEvaluation

        if not self.cfg.prioritization:
            return None
        for name, evaluation in self.evaluations.items():
            if isinstance(evaluation, BFMZeroManagerTrackingEvaluation):
                return name
        raise ValueError("Prioritization requires manager tracking evaluation to be enabled.")

    def train(self) -> None:
        self.train_online()

    def _load_expert_buffer(self):
        from .standalone_expert import load_manager_expert_trajectories

        return load_manager_expert_trajectories(
            self.train_env,
            self.cfg.agent,
            device=self.cfg.buffer_device,
            max_num_seqs=self.cfg.training_max_num_seqs,
            base_ang_vel_obs_scale=self.cfg.expert_base_ang_vel_obs_scale,
        )

    def _allocate_replay_buffer(self, expert_buffer):
        replay_buffer = {"train": load_or_create_train_buffer(self.work_dir, self.cfg)}
        replay_buffer["expert_slicer"] = expert_buffer
        return replay_buffer

    def _transition_data(self, obs, action, terminated, truncated, step_count, reward, info, new_info, context, history_context):
        data = {
            "observation": tree_map(lambda x: x[None, ...], obs),
            "action": action[None, ...],
            "terminated": terminated[None, ..., None],
            "truncated": truncated[None, ..., None],
            "step_count": step_count[None, ..., None],
            "reward": reward[None, ..., None],
        }
        data["observation"].pop("history", None)
        if context is not None:
            data["z"] = context[None, ...]
        if history_context is not None:
            data["history_context"] = history_context[None, ...]
        if "qpos" in info:
            data["qpos"] = info["qpos"][None, ...]
        if "qvel" in info:
            data["qvel"] = info["qvel"][None, ...]
        if "aux_rewards" in new_info:
            data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
        return data

    def _make_context(self, replay_buffer) -> TrainContext:
        td, info = self.train_env.reset()
        _randomize_episode_length_buf_once(self.train_env)
        runtime = TrainRuntime(
            cfg=self.cfg,
            work_dir=self.work_dir,
            train_env=self.train_env,
            agent=self.agent,
            replay_buffer=replay_buffer,
            evaluations=self.evaluations,
            eval_loggers=self.eval_loggers,
            train_logger=self.train_logger,
            prioritization_eval_name=self.priorization_eval_name,
            checkpoint_time=self._checkpoint_time,
        )
        state = TrainState(
            t=self._checkpoint_time,
            td=td,
            info=info,
            terminated=np.zeros(self.cfg.online_parallel_envs, dtype=bool),
            truncated=np.zeros(self.cfg.online_parallel_envs, dtype=bool),
        )
        return TrainContext(runtime=runtime, state=state)

    def _rollout_once(self, ctx: TrainContext) -> None:
        train_env = ctx.runtime.train_env
        with torch.no_grad():
            obs = tree_map(
                lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=ctx.runtime.agent.device),
                ctx.state.td,
            )
            step_count = train_env.episode_length_buf.unsqueeze(-1).to(device=ctx.runtime.agent.device)
            history_context = None
            ctx.state.z_context = ctx.runtime.agent.maybe_update_rollout_context(
                z=ctx.state.z_context,
                step_count=step_count,
                replay_buffer=ctx.runtime.replay_buffer,
            )
            if ctx.timestep < ctx.cfg.num_seed_steps:
                action = train_env.action_space.sample().astype(np.float32)
            else:
                action = ctx.runtime.agent.act(obs=obs, z=ctx.state.z_context, mean=False).cpu().detach().numpy()

        new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)
        transition_truncated = ctx.state.truncated
        if ctx.state.force_truncate_current_transition:
            transition_truncated = np.ones_like(transition_truncated, dtype=bool)
        data = self._transition_data(
            obs,
            action,
            ctx.state.terminated,
            transition_truncated,
            step_count,
            new_reward,
            ctx.state.info,
            new_info,
            ctx.state.z_context,
            history_context,
        )
        ctx.runtime.replay_buffer["train"].extend(data)
        ctx.state.td = new_td
        ctx.state.terminated = new_terminated
        ctx.state.truncated = new_truncated
        ctx.state.info = new_info

    def train_online(self) -> None:
        expert_buffer = self._load_expert_buffer()
        print("Creating the training environment")
        print("Allocating buffers")
        replay_buffer = self._allocate_replay_buffer(expert_buffer)

        print("Starting training")
        progb = tqdm(total=self.cfg.num_env_steps, disable=self.cfg.disable_tqdm)
        ctx = self._make_context(replay_buffer)
        hooks = HookList(
            [
                CheckpointHook(),
                EvaluationHook(use_shared_train_env=len(self.evaluations) > 0),
                AgentUpdateHook(),
                TrainLogHook(),
            ]
        )
        hooks.setup(ctx)

        for t in range(self._checkpoint_time, self.cfg.num_env_steps + self.cfg.online_parallel_envs, self.cfg.online_parallel_envs):
            ctx.state.t = t
            hooks.before_rollout(ctx)
            self._rollout_once(ctx)
            hooks.after_rollout(ctx)
            for metrics in ctx.state.pending_update_metrics:
                hooks.after_update(ctx, metrics)
            ctx.state.pending_update_metrics.clear()

            hooks.on_loop_end(ctx)
            progb.update(self.cfg.online_parallel_envs)
        hooks.close(ctx)
        self.train_env.close()
