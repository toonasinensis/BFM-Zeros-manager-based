from __future__ import annotations

from typing import Literal

import torch
import isaaclab.envs.mdp as isaac_mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from bfm.utils.torch_utils import quat_rotate_inverse

from .motion_provider import wxyz_to_xyzw
from .observations import compute_humanoid_observations_max
from .spec import BFMZERO_BASE_ANG_VEL_OBS_SCALE


def _robot(env, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def _env_ids_tensor(env, env_ids, device: torch.device | str) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)


def _scene_entity_ids_tensor(asset: Articulation, ids, device: torch.device | str, *, count: int) -> torch.Tensor:
    if ids is None or isinstance(ids, slice):
        return torch.arange(count, dtype=torch.long, device=device)
    return torch.as_tensor(ids, dtype=torch.long, device=device).reshape(-1)


def _resolve_dist_fn(distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"):
    if distribution == "uniform":
        return math_utils.sample_uniform
    if distribution == "log_uniform":
        return math_utils.sample_log_uniform
    if distribution == "gaussian":
        return math_utils.sample_gaussian
    raise ValueError(f"Unrecognized distribution {distribution!r}.")


def bfmzero_default_joint_pos(env, robot: Articulation, joint_ids=None) -> torch.Tensor:
    default_joint_pos = robot.data.default_joint_pos
    offset = getattr(env, "bfmzero_default_dof_pos_offset", None)
    if offset is not None:
        default_joint_pos = default_joint_pos + offset.to(default_joint_pos.device)
    if joint_ids is None:
        return default_joint_pos
    return default_joint_pos[:, joint_ids]


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def bfmzero_state_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    root_quat_xyzw = wxyz_to_xyzw(robot.data.root_quat_w)
    gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=env.device).reshape(1, 3).repeat(env.num_envs, 1)
    projected_gravity = quat_rotate_inverse(root_quat_xyzw, gravity, w_last=True)
    base_ang_vel = (
        quat_rotate_inverse(root_quat_xyzw, robot.data.root_ang_vel_w, w_last=True)
        * BFMZERO_BASE_ANG_VEL_OBS_SCALE
    )
    return torch.cat(
        [
            robot.data.joint_pos - bfmzero_default_joint_pos(env, robot),
            robot.data.joint_vel,
            projected_gravity,
            base_ang_vel,
        ],
        dim=-1,
    )


def bfmzero_privileged_state_obs(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_height_obs: bool = True,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)
    obs_dict = compute_humanoid_observations_max(
        robot.data.body_pos_w[:, body_ids],
        wxyz_to_xyzw(robot.data.body_quat_w[:, body_ids]),
        robot.data.body_lin_vel_w[:, body_ids],
        robot.data.body_ang_vel_w[:, body_ids],
        local_root_obs=True,
        root_height_obs=root_height_obs,
    )
    return torch.cat([value for value in obs_dict.values()], dim=-1)


def randomize_body_com(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    asset = _robot(env, asset_cfg)
    env_ids_cpu = _env_ids_tensor(env, env_ids, "cpu")
    body_ids = _scene_entity_ids_tensor(asset, asset_cfg.body_ids, "cpu", count=asset.num_bodies)

    if not hasattr(env, "bfmzero_default_coms"):
        env.bfmzero_default_coms = asset.root_physx_view.get_coms().clone()
    if not hasattr(env, "bfmzero_base_com_bias"):
        env.bfmzero_base_com_bias = torch.zeros((env.num_envs, 3), dtype=torch.float32, device="cpu")

    coms = asset.root_physx_view.get_coms()
    coms[env_ids_cpu[:, None], body_ids] = env.bfmzero_default_coms[env_ids_cpu[:, None], body_ids].clone()

    low, high = distribution_params
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low, dtype=torch.float32, device=coms.device)
    else:
        low = low.to(device=coms.device, dtype=torch.float32)
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(high, dtype=torch.float32, device=coms.device)
    else:
        high = high.to(device=coms.device, dtype=torch.float32)
    bias = _resolve_dist_fn(distribution)(low, high, (env_ids_cpu.numel(), 3), device=coms.device)
    env.bfmzero_base_com_bias[env_ids_cpu] = bias.cpu()

    if operation == "add":
        coms[env_ids_cpu[:, None], body_ids, :3] += bias[:, None, :]
    elif operation == "abs":
        coms[env_ids_cpu[:, None], body_ids, :3] = bias[:, None, :]
    elif operation == "scale":
        coms[env_ids_cpu[:, None], body_ids, :3] *= bias[:, None, :]
    else:
        raise ValueError(f"Unknown COM randomization operation {operation!r}.")
    asset.root_physx_view.set_coms(coms, env_ids_cpu)


def randomize_default_joint_pos_offset(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float],
    action_term_name: str = "joint_pos",
    apply_during_eval: bool = False,
) -> None:
    asset = _robot(env, asset_cfg)
    env_ids_t = _env_ids_tensor(env, env_ids, asset.device)
    joint_ids = _scene_entity_ids_tensor(asset, asset_cfg.joint_ids, asset.device, count=asset.data.joint_pos.shape[1])

    if not hasattr(env, "bfmzero_default_dof_pos_offset"):
        env.bfmzero_default_dof_pos_offset = torch.zeros_like(asset.data.default_joint_pos)

    should_randomize = apply_during_eval or not bool(getattr(env, "bfmzero_is_evaluating", False))
    if should_randomize:
        low, high = distribution_params
        sampled = math_utils.sample_uniform(low, high, (env_ids_t.numel(), joint_ids.numel()), device=asset.device)
    else:
        sampled = torch.zeros((env_ids_t.numel(), joint_ids.numel()), dtype=torch.float32, device=asset.device)
    env.bfmzero_default_dof_pos_offset[env_ids_t[:, None], joint_ids] = sampled

    action_term = env.action_manager.get_term(action_term_name)
    action_offset = getattr(action_term, "_offset", None)
    action_joint_ids = getattr(action_term, "_joint_ids", None)
    if isinstance(action_offset, torch.Tensor) and action_joint_ids is not None:
        action_joint_ids_t = _scene_entity_ids_tensor(asset, action_joint_ids, asset.device, count=asset.data.joint_pos.shape[1])
        action_term._offset[env_ids_t] = (
            asset.data.default_joint_pos[env_ids_t[:, None], action_joint_ids_t]
            + env.bfmzero_default_dof_pos_offset[env_ids_t[:, None], action_joint_ids_t]
        )


def push_by_setting_velocity_if_training(
    env,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    if bool(getattr(env, "bfmzero_is_evaluating", False)):
        return
    isaac_mdp.push_by_setting_velocity(env, env_ids, velocity_range=velocity_range, asset_cfg=asset_cfg)
