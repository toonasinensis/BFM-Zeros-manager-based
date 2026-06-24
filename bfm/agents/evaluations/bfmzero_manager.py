import collections
import dataclasses
import numbers
import typing as tp
from typing import Any, Dict

import numpy as np
import ot
import torch
from torch.utils._pytree import tree_map
from tqdm import tqdm

from bfm.agents.envs.bfmzero_manager_isaac import BFMZeroManagerIsaacConfig, BFMZeroManagerVectorEnv
from bfm.manager_envs.mdp.motion_provider import BFMZeroMotionProvider

from .base import BaseEvalConfig, extract_model


QVEL_IDX = 23


def _distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x.pow(2).sum(1).reshape(-1, 1)
    y_norm = y.pow(2).sum(1).reshape(1, -1)
    return torch.sqrt(torch.clamp(x_norm + y_norm - 2 * torch.matmul(x, y.T), min=0.0))


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor, prefix: str = "") -> dict[str, float]:
    agent_obs = next_obs.detach().float().cpu()
    tracked_obs = tracking_target.detach().float().cpu()
    cost_matrix = _distance_matrix(agent_obs, tracked_obs).numpy()
    agent_mass = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    target_mass = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(agent_mass, target_mass, cost_matrix, numItermax=100000)
    return {f"{prefix}emd": transport_cost}


def distance_proximity(
    next_obs: torch.Tensor,
    tracking_target: torch.Tensor,
    bound: float = 2.0,
    margin: float = 2.0,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    distance = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = distance <= bound
    out_bounds_mask = distance > bound + margin
    proximity = in_bounds_mask + ((bound + margin - distance) / margin) * (~in_bounds_mask) * (~out_bounds_mask)
    return {
        f"{prefix}proximity": proximity.mean(),
        f"{prefix}distance": distance.mean(),
    }


def compute_joint_pos_metrics(joint_pos: torch.Tensor, target_joint_pos: torch.Tensor) -> dict[str, torch.Tensor | float]:
    stats = {}
    stats["mpjpe_l"] = torch.norm(joint_pos - target_joint_pos, dim=-1).mean(-1) * 1000

    target_vel = target_joint_pos[:, 1:] - target_joint_pos[:, :-1]
    pred_vel = joint_pos[:, 1:] - joint_pos[:, :-1]
    stats["vel_dist"] = torch.norm(pred_vel - target_vel, dim=-1).mean(-1) * 1000

    target_accel = target_joint_pos[:, :-2] - 2 * target_joint_pos[:, 1:-1] + target_joint_pos[:, 2:]
    pred_accel = joint_pos[:, :-2] - 2 * joint_pos[:, 1:-1] + joint_pos[:, 2:]
    stats["accel_dist"] = torch.norm(pred_accel - target_accel, dim=-1).mean(-1) * 100

    stats.update(distance_proximity(next_obs=joint_pos, tracking_target=target_joint_pos, prefix=""))
    stats.update(emd_numpy(next_obs=joint_pos, tracking_target=target_joint_pos, prefix=""))
    return stats


def _tracking_z(model, obs: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    z = model.backward_map(tree_map(lambda x: x[1:].to(device), obs)).clone()
    for step in range(z.shape[0]):
        end_idx = min(step + 1, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return model.project_z(z)


@dataclasses.dataclass
class _MotionReference:
    motion_id: int
    motion_file: str
    z: torch.Tensor
    state: torch.Tensor
    dof_pos: torch.Tensor


def _legacy_tracking_metrics(
    *,
    motion_id: int,
    motion_file: str,
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    pred_joint_pos: torch.Tensor,
    target_joint_pos: torch.Tensor,
) -> dict[str, Any]:
    metrics = {}
    obs_len = min(pred_state.shape[0], target_state.shape[0])
    joint_len = min(pred_joint_pos.shape[0], target_joint_pos.shape[0])

    next_obs = pred_state[:obs_len, :QVEL_IDX].float()
    tracking_target = target_state[:obs_len, :QVEL_IDX].float()
    metrics.update(distance_proximity(next_obs=next_obs, tracking_target=tracking_target, prefix="obs_state_"))
    metrics.update(emd_numpy(next_obs=next_obs, tracking_target=tracking_target, prefix="obs_state_"))
    metrics.update(
        compute_joint_pos_metrics(
            joint_pos=pred_joint_pos[:joint_len].float(),
            target_joint_pos=target_joint_pos[:joint_len].float(),
        )
    )

    for key, value in list(metrics.items()):
        if isinstance(value, torch.Tensor):
            metrics[key] = value.detach().cpu().tolist()
    metrics["motion_id"] = motion_id
    metrics["motion_file"] = motion_file
    return metrics


class BFMZeroManagerTrackingEvaluationConfig(BaseEvalConfig):
    name: tp.Literal["BFMZeroManagerTrackingEvaluationConfig"] = "BFMZeroManagerTrackingEvaluationConfig"
    name_in_logs: str = "humanoidverse_tracking_eval"
    env: BFMZeroManagerIsaacConfig | None = None
    num_envs: int = 1024
    n_episodes_per_motion: int = 1
    include_results_from_all_envs: bool = False
    max_eval_motions: int | None = None
    disable_tqdm: bool = True

    def build(self):
        return BFMZeroManagerTrackingEvaluation(self)


class BFMZeroManagerTrackingEvaluation:
    def __init__(self, config: BFMZeroManagerTrackingEvaluationConfig):
        self.cfg = config

    def _build_reference_provider(self, env: BFMZeroManagerVectorEnv) -> BFMZeroMotionProvider:
        return BFMZeroMotionProvider(
            motion_file=env.motion_file,
            spec=env.adapter.spec,
            num_envs=env.num_envs,
            device=env.device,
        )

    def _motion_name(self, provider: BFMZeroMotionProvider, motion_id: int) -> str:
        provider.load_for_global_motion(motion_id)
        return str(provider.motion_lib.curr_motion_keys[provider.local_motion_id])

    def _reference(self, provider: BFMZeroMotionProvider, model, motion_id: int, device: torch.device) -> _MotionReference:
        backward_obs, ref_dict = provider.reference_backward_observation(
            motion_id,
            step_dt=self._step_dt,
            use_root_height_obs=True,
        )
        return _MotionReference(
            motion_id=motion_id,
            motion_file=self._motion_name(provider, motion_id),
            z=_tracking_z(model, backward_obs, device),
            state=backward_obs["state"].detach().cpu(),
            dof_pos=ref_dict["dof_pos"].detach().cpu(),
        )

    def _evaluate_chunk(
        self,
        env: BFMZeroManagerVectorEnv,
        agent_or_model,
        refs: dict[int, _MotionReference],
        assigned_motion_ids: torch.Tensor,
    ) -> dict[str, dict[str, Any]]:
        device = env.device
        observation, reset_info = env.reset_to_motions(assigned_motion_ids, frame_id=0, to_numpy=False)
        state_log = [observation["state"].detach().cpu()]
        joint_pos_log = [reset_info["joint_pos_abs"].detach().cpu()]
        max_steps = max(ref.z.shape[0] for ref in refs.values())

        with torch.no_grad():
            for step in tqdm(range(max_steps), desc="BFMZero Manager Tracking Eval", disable=self.cfg.disable_tqdm):
                z_batch = []
                for env_id in range(env.num_envs):
                    ref = refs[int(assigned_motion_ids[env_id].item())]
                    z_batch.append(ref.z[step % ref.z.shape[0]])
                z_batch_t = torch.stack(z_batch).to(device)
                action = agent_or_model.act(observation, z_batch_t, mean=True)
                observation, _reward, _terminated, _truncated, info = env.step(action, to_numpy=False)
                state_log.append(observation["state"].detach().cpu())
                joint_pos_log.append(info["joint_pos_abs"].detach().cpu())

        states = torch.stack(state_log)
        joint_pos = torch.stack(joint_pos_log)
        metrics = {}
        seen_motion_counts = collections.defaultdict(int)
        for env_id in range(env.num_envs):
            motion_id = int(assigned_motion_ids[env_id].item())
            motion_repetition = seen_motion_counts[motion_id]
            seen_motion_counts[motion_id] += 1
            if motion_repetition > 0 and not self.cfg.include_results_from_all_envs:
                continue
            ref = refs[motion_id]
            rollout_len = min(ref.z.shape[0] + 1, joint_pos.shape[0], states.shape[0], ref.dof_pos.shape[0], ref.state.shape[0])
            metric_key = ref.motion_file
            if motion_repetition > 0:
                metric_key = f"{metric_key}_repetition#{motion_repetition}"
            metrics[metric_key] = _legacy_tracking_metrics(
                motion_id=motion_id,
                motion_file=ref.motion_file,
                pred_state=states[:rollout_len, env_id, : ref.state.shape[-1]],
                target_state=ref.state[:rollout_len],
                pred_joint_pos=joint_pos[:rollout_len, env_id],
                target_joint_pos=ref.dof_pos[:rollout_len],
            )
        return metrics

    def run(self, *, timestep, agent_or_model, logger, env: BFMZeroManagerVectorEnv | None = None, **kwargs) -> Dict[str, Any]:
        del kwargs
        if env is None:
            if self.cfg.env is None:
                raise ValueError("Either env or cfg.env must be provided")
            env, _ = self.cfg.env.build(num_envs=self.cfg.num_envs)
        elif self.cfg.env is not None:
            raise ValueError("Both env and cfg.env are provided; please provide only one evaluation env.")

        model = extract_model(agent_or_model)
        self._step_dt = env.adapter.step_dt
        provider = self._build_reference_provider(env)
        motion_count = provider.motion_lib._num_unique_motions
        motion_ids = list(range(motion_count))
        if self.cfg.max_eval_motions is not None:
            motion_ids = motion_ids[: self.cfg.max_eval_motions]

        if hasattr(env, "set_is_evaluating"):
            env.set_is_evaluating()
        try:
            metrics = {}
            device = env.device
            for repetition_i in range(self.cfg.n_episodes_per_motion):
                run_metrics = {}
                for start in range(0, len(motion_ids), env.num_envs):
                    chunk = motion_ids[start : start + env.num_envs]
                    assigned = torch.tensor(
                        [chunk[i % len(chunk)] for i in range(env.num_envs)],
                        dtype=torch.long,
                        device=device,
                    )
                    refs = {motion_id: self._reference(provider, model, motion_id, device) for motion_id in chunk}
                    run_metrics.update(self._evaluate_chunk(env, agent_or_model, refs, assigned))
                if self.cfg.n_episodes_per_motion == 1:
                    metrics = run_metrics
                else:
                    for key, value in run_metrics.items():
                        metrics[f"{key}_repetition#{repetition_i}"] = value
        finally:
            if hasattr(env, "set_is_training"):
                env.set_is_training()

        aggregate = collections.defaultdict(list)
        wandb_dict = {}
        for _, metr in metrics.items():
            for key, value in metr.items():
                if isinstance(value, numbers.Number):
                    aggregate[key].append(value)
        for key, value in aggregate.items():
            wandb_dict[key] = float(np.mean(value))
            wandb_dict[f"{key}#std"] = float(np.std(value))

        if logger is not None:
            for key, value in metrics.items():
                value["motion_name"] = key
                value["timestep"] = timestep
                logger.log(value)

        return metrics, wandb_dict
