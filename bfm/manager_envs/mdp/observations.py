from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import torch

from bfm.utils.torch_utils import calc_heading_quat_inv, my_quat_rotate, quat_mul, quat_to_tan_norm, wxyz_to_xyzw

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg


def _as_wxyz(quat: torch.Tensor, quat_format: Literal["xyzw", "wxyz"]) -> torch.Tensor:
    if quat_format == "wxyz":
        return quat
    if quat_format == "xyzw":
        return quat[..., [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion format {quat_format!r}.")


def compute_humanoid_observations_max(
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    body_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
    local_root_obs: bool,
    root_height_obs: bool,
    quat_format: Literal["xyzw", "wxyz"] = "xyzw",
) -> OrderedDict[str, torch.Tensor]:
    obs_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    body_rot_wxyz = _as_wxyz(body_rot, quat_format)
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot_wxyz[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot_inv_xyzw = calc_heading_quat_inv(wxyz_to_xyzw(root_rot), w_last=True)

    if root_height_obs:
        obs_dict["root_height"] = root_h

    heading_rot_inv_expand = heading_rot_inv_xyzw.unsqueeze(-2).repeat((1, body_pos.shape[1], 1))
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

    flat_body_rot = body_rot_wxyz.reshape(body_rot_wxyz.shape[0] * body_rot_wxyz.shape[1], body_rot_wxyz.shape[2])
    flat_body_rot_xyzw = wxyz_to_xyzw(flat_body_rot)
    flat_local_body_rot_xyzw = quat_mul(flat_heading_rot_inv, flat_body_rot_xyzw, w_last=True)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot_xyzw, w_last=True)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot_wxyz.shape[0],
        body_rot_wxyz.shape[1] * flat_local_body_rot_obs.shape[1],
    )

    if not local_root_obs:
        root_rot_obs = quat_to_tan_norm(wxyz_to_xyzw(root_rot), w_last=True)
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

 
def bfmzero_default_joint_pos(env, robot: Articulation, joint_ids=None) -> torch.Tensor:
    default_joint_pos = robot.data.default_joint_pos
    offset = getattr(env, "bfmzero_default_dof_pos_offset", None)
    if isinstance(offset, torch.Tensor):
        default_joint_pos = default_joint_pos + offset
    if joint_ids is None:
        return default_joint_pos
    return default_joint_pos[:, joint_ids]


def bfmzero_state_terms(
    env,
    robot: Articulation,
    *,
    joint_ids=None,
    base_ang_vel_obs_scale: float = 0.25,
) -> dict[str, torch.Tensor]:
    joint_pos = robot.data.joint_pos if joint_ids is None else robot.data.joint_pos[:, joint_ids]
    joint_vel = robot.data.joint_vel if joint_ids is None else robot.data.joint_vel[:, joint_ids]
    default_joint_pos = bfmzero_default_joint_pos(env, robot, joint_ids)
    dof_pos = joint_pos - default_joint_pos
    projected_gravity = robot.data.projected_gravity_b
    base_ang_vel = robot.data.root_ang_vel_b * float(base_ang_vel_obs_scale)
    return {
        "state": torch.cat([dof_pos, joint_vel, projected_gravity, base_ang_vel], dim=-1),
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        "dof_pos": dof_pos,
        "dof_vel": joint_vel,
        "joint_pos_abs": joint_pos,
    }


def bfmzero_privileged_state_terms(
    robot: Articulation,
    *,
    body_ids=None,
    root_height_obs: bool = True,
) -> dict[str, torch.Tensor]:
    body_ids = body_ids if body_ids is not None else slice(None)
    body_quat_wxyz = robot.data.body_quat_w[:, body_ids]
    obs_dict = compute_humanoid_observations_max(
        robot.data.body_pos_w[:, body_ids],
        body_quat_wxyz,
        robot.data.body_lin_vel_w[:, body_ids],
        robot.data.body_ang_vel_w[:, body_ids],
        local_root_obs=True,
        root_height_obs=root_height_obs,
        quat_format="wxyz",
    )
    return {
        "privileged_state": torch.cat([value for value in obs_dict.values()], dim=-1),
    }


def bfmzero_state_obs(
    env,
    asset_cfg=None,
    base_ang_vel_obs_scale: float = 0.25,
) -> torch.Tensor:
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")
    robot = env.scene[asset_cfg.name]
    return bfmzero_state_terms(env, robot, base_ang_vel_obs_scale=base_ang_vel_obs_scale)["state"]


def bfmzero_privileged_state_obs(
    env,
    asset_cfg=None,
    root_height_obs: bool = True,
) -> torch.Tensor:
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")
    robot = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)
    return bfmzero_privileged_state_terms(robot, body_ids=body_ids, root_height_obs=root_height_obs)["privileged_state"]
