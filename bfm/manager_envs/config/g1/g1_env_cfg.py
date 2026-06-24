from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bfm.manager_envs.config.env_cfg import (
    BFMZeroActionsCfg,
    BFMZeroManagerEnvCfg as BFMZeroBaseManagerEnvCfg,
    BFMZeroRewardsCfg,
    BFMZeroSceneCfg,
    all_joint_cfg,
    apply_bfmzero_runtime_to_manager_cfg,
    apply_bfmzero_spec_to_manager_cfg as apply_base_spec_to_manager_cfg,
    body_cfg,
    build_bfmzero_manager_env_from_cfg,
    contact_cfg,
    enable_bfmzero_domain_randomization as enable_base_domain_randomization,
    feet_body_names,
    penalized_contact_body_names,
)
from bfm.manager_envs.mdp import rewards as bfm_rewards

from .g1_spec import (
    BFMZERO_BASE_ANG_VEL_OBS_SCALE,
    BFMZERO_DEFAULT_MOTION_FILE,
    BFMZERO_ROBOT_CONFIG,
    BFMZeroG1Spec,
    assert_bfmzero_spec_consistent,
    load_bfmzero_g1_spec,
)


def build_bfmzero_g1_articulation_cfg(spec: BFMZeroG1Spec | None = None) -> ArticulationCfg:
    spec = spec or load_bfmzero_g1_spec()
    assert_bfmzero_spec_consistent(spec)
    joint_pos = {name: float(spec.default_joint_angles[name]) for name in spec.dof_names}
    stiffness = {name: spec.p_gain_for_joint(name) for name in spec.dof_names}
    damping = {name: spec.d_gain_for_joint(name) for name in spec.dof_names}
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(spec.usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=list(spec.dof_names),
                effort_limit_sim={name: spec.effort_limits[i] for i, name in enumerate(spec.dof_names)},
                velocity_limit_sim={name: spec.velocity_limits[i] for i, name in enumerate(spec.dof_names)},
                stiffness=stiffness,
                damping=damping,
                armature={name: spec.armatures[i] for i, name in enumerate(spec.dof_names)},
                friction={name: spec.joint_frictions[i] for i, name in enumerate(spec.dof_names)},
            )
        },
    )


_SPEC = load_bfmzero_g1_spec()
_ROBOT_CFG = build_bfmzero_g1_articulation_cfg(_SPEC)


def _ankle_roll_joint_names(spec: BFMZeroG1Spec) -> list[str]:
    return list(spec.ankle_roll_dof_names)


@configclass
class BFMZeroG1SceneCfg(BFMZeroSceneCfg):
    robot: ArticulationCfg = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1ActionsCfg(BFMZeroActionsCfg):
    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_SPEC.dof_names),
        preserve_order=True,
        use_default_offset=True,
        scale=_SPEC.manager_action_scale_by_joint(),
    )


@configclass
class G1RewardsCfg(BFMZeroRewardsCfg):
    penalty_torques = RewTerm(
        func=bfm_rewards.penalty_torques,
        weight=1.0,
        params={"asset_cfg": all_joint_cfg(_SPEC)},
    )
    penalty_action_rate = RewTerm(
        func=bfm_rewards.penalty_action_rate,
        weight=1.0,
        params={"action_scale": _SPEC.action_obs_scale, "action_clip_value": _SPEC.action_clip_value},
    )
    limits_dof_pos = RewTerm(
        func=bfm_rewards.limits_dof_pos,
        weight=1.0,
        params={"asset_cfg": all_joint_cfg(_SPEC)},
    )
    limits_torque = RewTerm(
        func=bfm_rewards.limits_torque,
        weight=1.0,
        params={"asset_cfg": all_joint_cfg(_SPEC), "soft_torque_limit": 0.95, "effort_limits": _SPEC.effort_limits},
    )
    limits_dof_vel = RewTerm(
        func=bfm_rewards.limits_dof_vel,
        weight=1.0,
        params={"asset_cfg": all_joint_cfg(_SPEC), "soft_dof_vel_limit": 0.95, "velocity_limits": _SPEC.velocity_limits},
    )
    penalty_undesired_contact = RewTerm(
        func=bfm_rewards.penalty_undesired_contact,
        weight=1.0,
        params={"sensor_cfg": contact_cfg(penalized_contact_body_names(_SPEC)), "threshold": 1.0},
    )
    penalty_feet_ori = RewTerm(
        func=bfm_rewards.penalty_feet_ori,
        weight=1.0,
        params={"sensor_cfg": contact_cfg(feet_body_names(_SPEC)), "asset_cfg": body_cfg(feet_body_names(_SPEC))},
    )
    penalty_ankle_roll = RewTerm(
        func=bfm_rewards.penalty_ankle_roll,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ankle_roll_joint_names(_SPEC), preserve_order=True)},
    )
    feet_heading_alignment = RewTerm(
        func=bfm_rewards.feet_heading_alignment,
        weight=1.0,
        params={"asset_cfg": body_cfg(feet_body_names(_SPEC))},
    )
    penalty_slippage = RewTerm(
        func=bfm_rewards.penalty_slippage,
        weight=1.0,
        params={"sensor_cfg": contact_cfg(feet_body_names(_SPEC)), "asset_cfg": body_cfg(feet_body_names(_SPEC))},
    )


@configclass
class BFMZeroManagerEnvCfg(BFMZeroBaseManagerEnvCfg):
    scene: BFMZeroG1SceneCfg = BFMZeroG1SceneCfg(num_envs=1, env_spacing=5.0)
    actions: G1ActionsCfg = G1ActionsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()


def apply_bfmzero_spec_to_manager_cfg(cfg: BFMZeroManagerEnvCfg, spec: BFMZeroG1Spec) -> None:
    assert_bfmzero_spec_consistent(spec)
    apply_base_spec_to_manager_cfg(cfg, spec)
    cfg.scene.robot = build_bfmzero_g1_articulation_cfg(spec).replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.commands.motion.load_robot_spec = load_bfmzero_g1_spec

    all_joints = all_joint_cfg(spec)
    feet = feet_body_names(spec)
    penalized = penalized_contact_body_names(spec)
    ankle_roll = _ankle_roll_joint_names(spec)

    cfg.rewards.penalty_torques.params["asset_cfg"] = all_joints
    cfg.rewards.penalty_action_rate.params["action_scale"] = spec.action_obs_scale
    cfg.rewards.penalty_action_rate.params["action_clip_value"] = spec.action_clip_value
    cfg.rewards.limits_dof_pos.params["asset_cfg"] = all_joints
    cfg.rewards.limits_torque.params["asset_cfg"] = all_joints
    cfg.rewards.limits_torque.params["effort_limits"] = spec.effort_limits
    cfg.rewards.limits_dof_vel.params["asset_cfg"] = all_joints
    cfg.rewards.limits_dof_vel.params["velocity_limits"] = spec.velocity_limits
    cfg.rewards.penalty_undesired_contact.params["sensor_cfg"] = contact_cfg(penalized)
    cfg.rewards.penalty_feet_ori.params["sensor_cfg"] = contact_cfg(feet)
    cfg.rewards.penalty_feet_ori.params["asset_cfg"] = body_cfg(feet)
    cfg.rewards.penalty_ankle_roll.params["asset_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=ankle_roll,
        preserve_order=True,
    )
    cfg.rewards.feet_heading_alignment.params["asset_cfg"] = body_cfg(feet)
    cfg.rewards.penalty_slippage.params["sensor_cfg"] = contact_cfg(feet)
    cfg.rewards.penalty_slippage.params["asset_cfg"] = body_cfg(feet)
    cfg.observations.policy.state.params["base_ang_vel_obs_scale"] = float(
        getattr(spec, "base_ang_vel_obs_scale", BFMZERO_BASE_ANG_VEL_OBS_SCALE)
    )


def enable_bfmzero_domain_randomization(cfg: BFMZeroManagerEnvCfg, spec: BFMZeroG1Spec) -> None:
    enable_base_domain_randomization(cfg, spec)


def build_bfmzero_manager_env(
    *,
    num_envs: int = 1,
    device: str = "cuda:0",
    motion_file: str | Path = BFMZERO_DEFAULT_MOTION_FILE,
    robot_config: str = BFMZERO_ROBOT_CONFIG,
    default_motion_id: int = 0,
    episode_length_s: float = 10.0,
    training_randomize_motions: bool = False,
    training_max_num_seqs: int | None = None,
    base_ang_vel_obs_scale: float | None = None,
    enable_domain_randomization: bool = False,
    render_mode: str | None = None,
) -> ManagerBasedRLEnv:
    spec = load_bfmzero_g1_spec(robot_config)
    cfg = BFMZeroManagerEnvCfg()
    apply_bfmzero_spec_to_manager_cfg(cfg, spec)
    if enable_domain_randomization:
        enable_bfmzero_domain_randomization(cfg, spec)
    apply_bfmzero_runtime_to_manager_cfg(
        cfg,
        num_envs=num_envs,
        device=device,
        motion_file=motion_file,
        default_motion_id=default_motion_id,
        episode_length_s=episode_length_s,
        training_randomize_motions=training_randomize_motions,
        training_max_num_seqs=training_max_num_seqs,
        base_ang_vel_obs_scale=base_ang_vel_obs_scale,
    )
    return build_bfmzero_manager_env_from_cfg(
        cfg,
        render_mode=render_mode,
        domain_randomization_enabled=enable_domain_randomization,
    )
