from __future__ import annotations

from collections import OrderedDict

import torch

from bfm.utils.torch_utils import calc_heading_quat_inv, my_quat_rotate, quat_mul, quat_to_tan_norm


def compute_humanoid_observations_max(
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    body_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
    local_root_obs: bool,
    root_height_obs: bool,
) -> OrderedDict[str, torch.Tensor]:
    obs_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(root_rot, w_last=True)

    if root_height_obs:
        obs_dict["root_height"] = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2).repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
        heading_rot_inv_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last=True)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not local_root_obs:
        root_rot_obs = quat_to_tan_norm(root_rot, w_last=True)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_dict["local_body_pos"] = local_body_pos
    obs_dict["local_body_rot"] = local_body_rot_obs
    obs_dict["local_body_vel"] = local_body_vel
    obs_dict["local_body_ang_vel"] = local_body_ang_vel
    return obs_dict
