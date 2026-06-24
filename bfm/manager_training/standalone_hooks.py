from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import wandb

from bfm.agents.utils import EveryNStepsChecker

from .standalone_checkpoint import save_checkpoint
from .standalone_context import TrainContext


class TrainHook:
    def setup(self, ctx: TrainContext) -> None:
        del ctx

    def before_rollout(self, ctx: TrainContext) -> None:
        del ctx

    def after_rollout(self, ctx: TrainContext) -> None:
        del ctx

    def after_update(self, ctx: TrainContext, metrics: dict[str, torch.Tensor]) -> None:
        del ctx, metrics

    def on_loop_end(self, ctx: TrainContext) -> None:
        del ctx

    def close(self, ctx: TrainContext) -> None:
        del ctx


class HookList:
    def __init__(self, hooks: list[TrainHook]):
        self.hooks = hooks

    def setup(self, ctx: TrainContext) -> None:
        for hook in self.hooks:
            hook.setup(ctx)

    def before_rollout(self, ctx: TrainContext) -> None:
        for hook in self.hooks:
            hook.before_rollout(ctx)

    def after_rollout(self, ctx: TrainContext) -> None:
        for hook in self.hooks:
            hook.after_rollout(ctx)

    def after_update(self, ctx: TrainContext, metrics: dict[str, torch.Tensor]) -> None:
        for hook in self.hooks:
            hook.after_update(ctx, metrics)

    def on_loop_end(self, ctx: TrainContext) -> None:
        for hook in self.hooks:
            hook.on_loop_end(ctx)

    def close(self, ctx: TrainContext) -> None:
        for hook in reversed(self.hooks):
            hook.close(ctx)


class CheckpointHook(TrainHook):
    def setup(self, ctx: TrainContext) -> None:
        self.checker = EveryNStepsChecker(ctx.runtime.checkpoint_time, ctx.cfg.checkpoint_every_steps)

    def before_rollout(self, ctx: TrainContext) -> None:
        t = ctx.timestep
        if t == ctx.runtime.checkpoint_time:
            return
        if not self.checker.check(t):
            return
        self.checker.update_last_step(t)
        save_checkpoint(ctx.runtime.work_dir, ctx.cfg, ctx.runtime.agent, ctx.runtime.replay_buffer, t)


class EvaluationHook(TrainHook):
    def __init__(self, *, use_shared_train_env: bool = True):
        self.use_shared_train_env = bool(use_shared_train_env)

    def setup(self, ctx: TrainContext) -> None:
        self.evaluate = len(ctx.runtime.evaluations) > 0
        self.checker = EveryNStepsChecker(ctx.runtime.checkpoint_time, ctx.cfg.eval_every_steps)
        if ctx.cfg.prioritization and ctx.runtime.prioritization_eval_name is None:
            raise ValueError("Prioritization requires manager tracking evaluation to be enabled.")

    def before_rollout(self, ctx: TrainContext) -> None:
        ctx.state.force_truncate_current_transition = False
        if not self._should_eval(ctx.timestep, ctx):
            self._mark_current_transition_for_next_eval(ctx)
            return
        episode_length_buf = None
        if self.use_shared_train_env:
            episode_length_buf = ctx.runtime.train_env.episode_length_buf.detach().clone()
        eval_metrics = self._run_eval(ctx)
        ctx.state.last_eval_metrics = eval_metrics
        self.checker.update_last_step(ctx.timestep)
        if self.use_shared_train_env:
            ctx.reset_rollout_state(episode_length_buf=episode_length_buf)
        if ctx.cfg.prioritization:
            self._apply_prioritization(ctx, eval_metrics)
        self._mark_current_transition_for_next_eval(ctx)

    def _mark_current_transition_for_next_eval(self, ctx: TrainContext) -> None:
        next_t = ctx.timestep + ctx.cfg.online_parallel_envs
        ctx.state.force_truncate_current_transition = self.use_shared_train_env and self._should_eval(next_t, ctx)

    def _should_eval(self, t: int, ctx: TrainContext) -> bool:
        return self.evaluate and (self.checker.check(t) or t == ctx.runtime.checkpoint_time)

    def _run_eval(self, ctx: TrainContext) -> dict[str, Any]:
        print(f"Starting evaluation at time {ctx.timestep}")
        evaluation_results = {}
        agent = ctx.runtime.agent
        for evaluation_name, evaluation in ctx.runtime.evaluations.items():
            logger = ctx.runtime.eval_loggers[evaluation_name]
            agent._model.train(False)
            evaluation_metrics, wandb_dict = evaluation.run(
                timestep=ctx.timestep,
                agent_or_model=agent,
                replay_buffer=ctx.runtime.replay_buffer,
                logger=logger,
                env=ctx.runtime.train_env,
            )
            if ctx.cfg.use_wandb and wandb_dict is not None:
                wandb.log({f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()}, step=ctx.timestep)
            evaluation_results[evaluation_name] = evaluation_metrics
        agent._model.train()
        return evaluation_results

    def _apply_prioritization(self, ctx: TrainContext, eval_metrics: dict[str, Any]) -> None:
        eval_name = ctx.runtime.prioritization_eval_name
        if eval_name is None:
            raise ValueError("Prioritization eval name is not configured.")
        metrics = eval_metrics[eval_name]
        expert_slicer = ctx.runtime.replay_buffer["expert_slicer"]
        num_eval_motions = len(metrics)
        num_expert_motions = len(expert_slicer.motion_ids)
        if num_eval_motions != num_expert_motions:
            raise AssertionError(
                f"Mismatch in number of motions returned by the eval: eval={num_eval_motions}, expert={num_expert_motions}"
            )

        index_in_buffer, name_in_buffer = {}, {}
        for i, motion_id in enumerate(expert_slicer.motion_ids):
            index_in_buffer[motion_id] = i
            if hasattr(expert_slicer, "file_names"):
                name_in_buffer[motion_id] = expert_slicer.file_names[i]

        priorities, idxs = [], []
        for _, metric in metrics.items():
            priorities.append(metric["emd"])
            idxs.append(index_in_buffer[metric["motion_id"]])

        priorities_t = (
            torch.clamp(
                torch.tensor(priorities, dtype=torch.float32, device=ctx.runtime.agent.device),
                min=ctx.cfg.prioritization_min_val,
                max=ctx.cfg.prioritization_max_val,
            )
            * ctx.cfg.prioritization_scale
        )
        if ctx.cfg.prioritization_mode == "exp":
            priorities_t = 2**priorities_t
        elif ctx.cfg.prioritization_mode == "bin":
            bins = torch.floor(priorities_t)
            for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                mask = bins == i
                n = mask.sum().item()
                if n > 0:
                    priorities_t[mask] = 1 / n
        elif ctx.cfg.prioritization_mode != "lin":
            raise ValueError(f"Unsupported prioritization mode {ctx.cfg.prioritization_mode}")

        ctx.runtime.train_env.update_motion_sampling_weights(
            priorities=list(priorities_t),
            motion_indexes=idxs,
            file_name=name_in_buffer,
        )
        expert_slicer.update_priorities(
            priorities=priorities_t.to(ctx.cfg.buffer_device),
            idxs=torch.tensor(np.array(idxs), device=ctx.cfg.buffer_device),
        )


class AgentUpdateHook(TrainHook):
    def setup(self, ctx: TrainContext) -> None:
        self.checker = EveryNStepsChecker(ctx.runtime.checkpoint_time, ctx.cfg.update_agent_every)

    def after_rollout(self, ctx: TrainContext) -> None:
        if len(ctx.runtime.replay_buffer["train"]) <= 0:
            return
        if ctx.timestep <= ctx.cfg.num_seed_steps or not self.checker.check(ctx.timestep):
            return
        self.checker.update_last_step(ctx.timestep)
        for _ in range(ctx.cfg.num_agent_updates):
            metrics = ctx.runtime.agent.update(ctx.runtime.replay_buffer, ctx.timestep)
            ctx.state.pending_update_metrics.append(metrics)


class TrainLogHook(TrainHook):
    def setup(self, ctx: TrainContext) -> None:
        self.checker = EveryNStepsChecker(ctx.runtime.checkpoint_time, ctx.cfg.log_every_updates)

    def after_update(self, ctx: TrainContext, metrics: dict[str, torch.Tensor]) -> None:
        if ctx.state.total_metrics is None:
            ctx.state.num_metric_updates = 1
            ctx.state.total_metrics = {key: value.float().clone() for key, value in metrics.items()}
            return
        ctx.state.num_metric_updates += 1
        ctx.state.total_metrics = {key: ctx.state.total_metrics[key] + value.float() for key, value in metrics.items()}

    def on_loop_end(self, ctx: TrainContext) -> None:
        if ctx.state.total_metrics is None or not self.checker.check(ctx.timestep):
            return
        self.checker.update_last_step(ctx.timestep)
        metrics = {}
        for key in sorted(ctx.state.total_metrics.keys()):
            value = ctx.state.total_metrics[key] / ctx.state.num_metric_updates
            metrics[key] = np.round(value.mean().item(), 6)
        metrics["duration [minutes]"] = (time.time() - ctx.state.start_time) / 60
        metrics["FPS"] = (1 if ctx.timestep == 0 else ctx.cfg.log_every_updates) / (time.time() - ctx.state.fps_start_time)
        if ctx.cfg.use_wandb:
            wandb.log({f"train/{key}": value for key, value in metrics.items()}, step=ctx.timestep)
        print(metrics)
        ctx.state.total_metrics = None
        ctx.state.num_metric_updates = 0
        ctx.state.fps_start_time = time.time()
        metrics["timestep"] = ctx.timestep
        ctx.runtime.train_logger.log(metrics)
