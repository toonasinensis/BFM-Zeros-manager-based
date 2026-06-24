from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from bfm.utils.torch_utils import quat_apply, quat_rotate_inverse, wrap_to_pi


BFMZERO_AUX_REWARD_NAMES = (
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
)

BFMZERO_ENV_REWARD_SCALES = {
    "penalty_torques": -0.000001,
    "penalty_action_rate": -0.5,
    "limits_dof_pos": -10.0,
    "limits_dof_vel": -5.0,
    "limits_torque": -5.0,
    "penalty_undesired_contact": -1.0,
    "penalty_feet_ori": -0.1,
    "feet_heading_alignment": -0.1,
    "penalty_ankle_roll": -0.5,
    "penalty_slippage": -1.0,
}

BFMZERO_RAW_AUX_REWARD_NAMES = tuple(BFMZERO_ENV_REWARD_SCALES.keys())
BFMZERO_PENALTY_REWARD_NAMES = frozenset(BFMZERO_AUX_REWARD_NAMES)
BFMZERO_INITIAL_PENALTY_SCALE = 0.1


def _robot(env, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def _joint_ids(asset_cfg: SceneEntityCfg):
    return asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)


def _body_ids(asset_cfg: SceneEntityCfg):
    return asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)


def penalty_torques(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return torch.sum(torch.square(robot.data.applied_torque[:, _joint_ids(asset_cfg)]), dim=1)


def penalty_action_rate(env, action_scale: float, action_clip_value: float) -> torch.Tensor:
    action_term = env.action_manager.get_term("joint_pos")
    current = action_term.raw_actions * float(action_scale)
    current = torch.clamp(current, -float(action_clip_value), float(action_clip_value))
    previous = getattr(env, "bfmzero_last_action_obs", torch.zeros_like(current))
    return torch.sum(torch.square(previous - current), dim=1)


def limits_dof_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    joint_ids = _joint_ids(asset_cfg)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    out_of_limits = -(joint_pos - limits[..., 0]).clip(max=0.0)
    out_of_limits += (joint_pos - limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def limits_torque(
    env,
    soft_torque_limit: float,
    effort_limits: tuple[float, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    effort_limits_t = torch.tensor(effort_limits, dtype=torch.float32, device=env.device).reshape(1, -1)
    torque = robot.data.applied_torque[:, _joint_ids(asset_cfg)]
    return torch.sum((torch.abs(torque) - effort_limits_t * float(soft_torque_limit)).clip(min=0.0), dim=1)


def limits_dof_vel(
    env,
    soft_dof_vel_limit: float,
    velocity_limits: tuple[float, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    velocity_limits_t = torch.tensor(velocity_limits, dtype=torch.float32, device=env.device).reshape(1, -1)
    joint_vel = robot.data.joint_vel[:, _joint_ids(asset_cfg)]
    return torch.sum((torch.abs(joint_vel) - velocity_limits_t * float(soft_dof_vel_limit)).clip(min=0.0, max=1.0), dim=1)


def penalty_undesired_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, _body_ids(sensor_cfg)]
    undesired = torch.any(torch.abs(forces) > float(threshold), dim=(1, 2))
    return undesired.to(dtype=torch.float32)


def penalty_feet_ori(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = _robot(env, asset_cfg)
    contact = contact_sensor.data.net_forces_w[:, _body_ids(sensor_cfg), 2] > 1.0
    foot_quat = robot.data.body_quat_w[:, _body_ids(asset_cfg)]
    foot_quat_xyzw = foot_quat[..., [1, 2, 3, 0]]
    gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=env.device).reshape(1, 1, 3)
    foot_gravity = quat_rotate_inverse(foot_quat_xyzw.reshape(-1, 4), gravity.repeat(env.num_envs, foot_quat.shape[1], 1).reshape(-1, 3), w_last=True)
    foot_gravity = foot_gravity.reshape(env.num_envs, foot_quat.shape[1], 3)
    return torch.sum(torch.sum(torch.square(foot_gravity[..., :2]), dim=-1).sqrt() * contact, dim=1)


def penalty_ankle_roll(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    ankle_roll = robot.data.joint_pos[:, _joint_ids(asset_cfg)]
    return torch.sum(torch.square(ankle_roll), dim=1)


def feet_heading_alignment(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    forward = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=env.device).reshape(1, 3)
    foot_quat_xyzw = robot.data.body_quat_w[:, _body_ids(asset_cfg)][..., [1, 2, 3, 0]]
    root_quat_xyzw = robot.data.root_quat_w[:, [1, 2, 3, 0]]
    foot_forward = quat_apply(
        foot_quat_xyzw.reshape(-1, 4),
        forward.repeat(env.num_envs * foot_quat_xyzw.shape[1], 1),
        w_last=True,
    ).reshape(env.num_envs, foot_quat_xyzw.shape[1], 3)
    root_forward = quat_apply(root_quat_xyzw, forward.repeat(env.num_envs, 1), w_last=True)
    foot_heading = torch.atan2(foot_forward[..., 1], foot_forward[..., 0])
    root_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0]).unsqueeze(-1)
    return torch.sum(torch.abs(wrap_to_pi(foot_heading - root_heading)), dim=1)


def penalty_slippage(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = _robot(env, asset_cfg)
    foot_vel = robot.data.body_lin_vel_w[:, _body_ids(asset_cfg)]
    contact = torch.norm(contact_sensor.data.net_forces_w[:, _body_ids(sensor_cfg)], dim=-1) > 1.0
    return torch.sum(torch.norm(foot_vel, dim=-1) * contact, dim=1)



def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

