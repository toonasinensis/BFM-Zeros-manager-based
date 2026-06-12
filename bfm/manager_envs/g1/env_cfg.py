from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from . import rewards as bfm_rewards
from .motion_command import BFMZeroMotionCommandCfg
from .spec import BFMZERO_DEFAULT_MOTION_FILE, BFMZeroG1Spec, assert_bfmzero_spec_consistent, load_bfmzero_g1_spec


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


def _feet_body_names(spec: BFMZeroG1Spec) -> list[str]:
    return [name for name in spec.body_names if spec.foot_name in name]


def _penalized_contact_body_names(spec: BFMZeroG1Spec) -> list[str]:
    names: list[str] = []
    for key in spec.penalize_contacts_on:
        names.extend([body_name for body_name in spec.body_names if key in body_name])
    return names


def _ankle_roll_joint_names(spec: BFMZeroG1Spec) -> list[str]:
    return [spec.left_ankle_dof_names[1], spec.right_ankle_dof_names[1]]


def _all_joint_cfg(spec: BFMZeroG1Spec) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(spec.dof_names), preserve_order=True)


def _body_cfg(names: list[str]) -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=names, preserve_order=True)


def _contact_cfg(names: list[str]) -> SceneEntityCfg:
    return SceneEntityCfg("contact_forces", body_names=names, preserve_order=True)


@configclass
class BFMZeroG1SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, update_period=0.005, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    motion = BFMZeroMotionCommandCfg(
        asset_name="robot",
        motion_file=BFMZERO_DEFAULT_MOTION_FILE,
        default_motion_id=0,
        reset_robot_state=True,
        lie_down_init_prob=0.3,
        lie_down_root_height=0.5,
    )


@configclass
class ActionsCfg:
    joint_pos = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(_SPEC.dof_names),
        preserve_order=True,
        use_default_offset=True,
        scale=_SPEC.manager_action_scale_by_joint(),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(
            func=mdp.bfmzero_state_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        privileged_state = ObsTerm(
            func=mdp.bfmzero_privileged_state_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=list(_SPEC.body_names), preserve_order=True),
                "root_height_obs": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    penalty_torques = RewTerm(
        func=bfm_rewards.penalty_torques,
        weight=1.0,
        params={"asset_cfg": _all_joint_cfg(_SPEC)},
    )
    penalty_action_rate = RewTerm(
        func=bfm_rewards.penalty_action_rate,
        weight=1.0,
        params={"action_scale": _SPEC.action_obs_scale, "action_clip_value": _SPEC.action_clip_value},
    )
    limits_dof_pos = RewTerm(
        func=bfm_rewards.limits_dof_pos,
        weight=1.0,
        params={"asset_cfg": _all_joint_cfg(_SPEC)},
    )
    limits_torque = RewTerm(
        func=bfm_rewards.limits_torque,
        weight=1.0,
        params={"asset_cfg": _all_joint_cfg(_SPEC), "soft_torque_limit": 0.95, "effort_limits": _SPEC.effort_limits},
    )
    limits_dof_vel = RewTerm(
        func=bfm_rewards.limits_dof_vel,
        weight=1.0,
        params={"asset_cfg": _all_joint_cfg(_SPEC), "soft_dof_vel_limit": 0.95, "velocity_limits": _SPEC.velocity_limits},
    )
    penalty_undesired_contact = RewTerm(
        func=bfm_rewards.penalty_undesired_contact,
        weight=1.0,
        params={"sensor_cfg": _contact_cfg(_penalized_contact_body_names(_SPEC)), "threshold": 1.0},
    )
    penalty_feet_ori = RewTerm(
        func=bfm_rewards.penalty_feet_ori,
        weight=1.0,
        params={"sensor_cfg": _contact_cfg(_feet_body_names(_SPEC)), "asset_cfg": _body_cfg(_feet_body_names(_SPEC))},
    )
    penalty_ankle_roll = RewTerm(
        func=bfm_rewards.penalty_ankle_roll,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ankle_roll_joint_names(_SPEC), preserve_order=True)},
    )
    feet_heading_alignment = RewTerm(
        func=bfm_rewards.feet_heading_alignment,
        weight=1.0,
        params={"asset_cfg": _body_cfg(_feet_body_names(_SPEC))},
    )
    penalty_slippage = RewTerm(
        func=bfm_rewards.penalty_slippage,
        weight=1.0,
        params={"sensor_cfg": _contact_cfg(_feet_body_names(_SPEC)), "asset_cfg": _body_cfg(_feet_body_names(_SPEC))},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class EventCfg:
    randomize_link_mass: EventTerm | None = None
    randomize_friction: EventTerm | None = None
    randomize_base_com: EventTerm | None = None
    randomize_default_dof_pos: EventTerm | None = None
    push_robot: EventTerm | None = None


@configclass
class CurriculumCfg:
    pass


@configclass
class BFMZeroManagerEnvCfg(ManagerBasedRLEnvCfg):
    scene: BFMZeroG1SceneCfg = BFMZeroG1SceneCfg(num_envs=1, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


def apply_bfmzero_spec_to_manager_cfg(cfg: BFMZeroManagerEnvCfg, spec: BFMZeroG1Spec) -> None:
    assert_bfmzero_spec_consistent(spec)
    cfg.scene.robot = build_bfmzero_g1_articulation_cfg(spec).replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.joint_names = list(spec.dof_names)
    cfg.actions.joint_pos.scale = spec.manager_action_scale_by_joint()
    cfg.commands.motion.robot_config = spec.config_name

    cfg.observations.policy.privileged_state.params["asset_cfg"] = SceneEntityCfg(
        "robot",
        body_names=list(spec.body_names),
        preserve_order=True,
    )

    all_joint_cfg = _all_joint_cfg(spec)
    feet = _feet_body_names(spec)
    penalized = _penalized_contact_body_names(spec)
    ankle_roll = _ankle_roll_joint_names(spec)

    cfg.rewards.penalty_torques.params["asset_cfg"] = all_joint_cfg
    cfg.rewards.penalty_action_rate.params["action_scale"] = spec.action_obs_scale
    cfg.rewards.penalty_action_rate.params["action_clip_value"] = spec.action_clip_value
    cfg.rewards.limits_dof_pos.params["asset_cfg"] = all_joint_cfg
    cfg.rewards.limits_torque.params["asset_cfg"] = all_joint_cfg
    cfg.rewards.limits_torque.params["effort_limits"] = spec.effort_limits
    cfg.rewards.limits_dof_vel.params["asset_cfg"] = all_joint_cfg
    cfg.rewards.limits_dof_vel.params["velocity_limits"] = spec.velocity_limits
    cfg.rewards.penalty_undesired_contact.params["sensor_cfg"] = _contact_cfg(penalized)
    cfg.rewards.penalty_feet_ori.params["sensor_cfg"] = _contact_cfg(feet)
    cfg.rewards.penalty_feet_ori.params["asset_cfg"] = _body_cfg(feet)
    cfg.rewards.penalty_ankle_roll.params["asset_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=ankle_roll,
        preserve_order=True,
    )
    cfg.rewards.feet_heading_alignment.params["asset_cfg"] = _body_cfg(feet)
    cfg.rewards.penalty_slippage.params["sensor_cfg"] = _contact_cfg(feet)
    cfg.rewards.penalty_slippage.params["asset_cfg"] = _body_cfg(feet)


def enable_bfmzero_domain_randomization(cfg: BFMZeroManagerEnvCfg, spec: BFMZeroG1Spec) -> None:
    cfg.events.randomize_link_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": _body_cfg(list(spec.randomize_link_body_names)),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    cfg.events.randomize_friction = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1024,
        },
    )
    cfg.events.randomize_base_com = EventTerm(
        func=mdp.randomize_body_com,
        mode="startup",
        params={
            "asset_cfg": _body_cfg([spec.torso_name]),
            "distribution_params": (
                (-0.02, -0.02, -0.02),
                (0.02, 0.02, 0.02),
            ),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    cfg.events.randomize_default_dof_pos = EventTerm(
        func=mdp.randomize_default_joint_pos_offset,
        mode="reset",
        params={
            "asset_cfg": _all_joint_cfg(spec),
            "distribution_params": (-0.02, 0.02),
            "action_term_name": "joint_pos",
            "apply_during_eval": False,
        },
    )
    cfg.events.push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_if_training,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


def build_bfmzero_manager_env(
    *,
    num_envs: int = 1,
    device: str = "cuda:0",
    motion_file: str | Path = BFMZERO_DEFAULT_MOTION_FILE,
    robot_config: str = _SPEC.config_name,
    default_motion_id: int = 0,
    episode_length_s: float = 10.0,
    training_randomize_motions: bool = False,
    training_max_num_seqs: int | None = None,
    enable_domain_randomization: bool = False,
    render_mode: str | None = None,
) -> ManagerBasedRLEnv:
    spec = load_bfmzero_g1_spec(robot_config)
    cfg = BFMZeroManagerEnvCfg()
    apply_bfmzero_spec_to_manager_cfg(cfg, spec)
    if enable_domain_randomization:
        enable_bfmzero_domain_randomization(cfg, spec)
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = device
    cfg.commands.motion.motion_file = str(motion_file)
    cfg.commands.motion.default_motion_id = int(default_motion_id)
    cfg.commands.motion.training_randomize_motions = bool(training_randomize_motions)
    cfg.commands.motion.training_max_num_seqs = training_max_num_seqs
    cfg.episode_length_s = float(episode_length_s)
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode)
    env.bfmzero_domain_randomization_enabled = bool(enable_domain_randomization)
    env.bfmzero_is_evaluating = False
    return env
