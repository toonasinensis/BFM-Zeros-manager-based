from __future__ import annotations

import numpy as np
import torch

from bfm.agents.buffers.trajectory import TrajectoryDictBuffer
from bfm.manager_envs.g1.motion_provider import BFMZeroMotionProvider
from bfm.manager_envs.g1.observations import compute_humanoid_observations_max
from bfm.manager_envs.g1.spec import BFMZERO_HISTORY_CONFIG, BFMZERO_HISTORY_ORDER
from bfm.utils.torch_utils import quat_rotate_inverse


def load_manager_expert_trajectories(env, agent_cfg, *, device: str = "cpu", max_num_seqs: int | None = None):
    provider = BFMZeroMotionProvider(
        motion_file=env.motion_file,
        spec=env.adapter.spec,
        num_envs=env.num_envs,
        device=env.device,
    )
    provider.load_for_training(max_num_seqs=max_num_seqs)

    episodes = []
    file_names = []
    for local_motion_id in range(provider.motion_lib.num_motions()):
        motion_len = provider.motion_lib._motion_lengths[local_motion_id]
        motion_times = torch.arange(int(np.ceil((motion_len / env.adapter.step_dt).detach().cpu())), device=env.device) * env.adapter.step_dt
        motion_id = torch.full((motion_times.shape[0],), local_motion_id, dtype=torch.long, device=env.device)
        motion_res = provider.motion_lib.get_motion_state(motion_id, motion_times)
        file_names.append(provider.motion_lib.curr_motion_keys[local_motion_id])

        ref_body_pos = motion_res["rg_pos_t"]
        ref_body_rots = motion_res["rg_rot_t"]
        ref_body_vels = motion_res["body_vel_t"]
        ref_body_angular_vels = motion_res["body_ang_vel_t"]

        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=True,
        )
        privileged_state = torch.cat([value for value in obs_dict.values()], dim=-1)

        ref_dof_pos = motion_res["dof_pos"] - provider.default_joint_pos
        ref_dof_vel = motion_res["dof_vel"]
        ref_ang_vel = ref_body_angular_vels[:, 0]
        projected_gravity = quat_rotate_inverse(
            ref_body_rots[:, 0],
            provider.gravity_vec.repeat(privileged_state.shape[0], 1),
            w_last=True,
        )
        state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], dim=-1)
        last_action = torch.zeros_like(ref_dof_pos)

        dims = {
            "actions": last_action.shape[-1],
            "base_ang_vel": ref_ang_vel.shape[-1],
            "dof_pos": ref_dof_pos.shape[-1],
            "dof_vel": ref_dof_vel.shape[-1],
            "projected_gravity": projected_gravity.shape[-1],
        }
        history_actor_dim = sum(BFMZERO_HISTORY_CONFIG[key] * dims[key] for key in BFMZERO_HISTORY_ORDER)
        history_actor = torch.zeros(state.shape[0], history_actor_dim, dtype=torch.float32, device=env.device)

        truncated = torch.zeros(state.shape[0], dtype=torch.bool, device=env.device)
        truncated[-1] = True
        episodes.append(
            {
                "observation": {
                    "state": state,
                    "last_action": last_action,
                    "privileged_state": privileged_state,
                    "history_actor": history_actor,
                },
                "terminated": torch.zeros(state.shape[0], dtype=torch.bool, device=env.device),
                "truncated": truncated,
                "motion_id": provider.motion_lib._curr_motion_ids[local_motion_id].repeat(state.shape[0]).long(),
            }
        )

    expert_buffer = TrajectoryDictBuffer(
        episodes=episodes,
        seq_length=agent_cfg.model.seq_length,
        device=device,
    )
    expert_buffer.file_names = file_names
    return expert_buffer
