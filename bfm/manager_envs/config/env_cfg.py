from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from bfm.manager_envs.mdp import events as bfm_events
from bfm.manager_envs.mdp import observations as bfm_observations
from bfm.manager_envs.mdp.motion_command import BFMZeroMotionCommandCfg
from bfm.manager_envs.mdp.spec import (
    BFMZERO_BASE_ANG_VEL_OBS_SCALE,
    BFMZERO_DEFAULT_MOTION_FILE,
    BFMZeroRobotSpec,
)


def all_joint_cfg(spec: BFMZeroRobotSpec) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(spec.dof_names), preserve_order=True)


def body_cfg(names: list[str]) -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=names, preserve_order=True)


def contact_cfg(names: list[str]) -> SceneEntityCfg:
    return SceneEntityCfg("contact_forces", body_names=names, preserve_order=True)


def feet_body_names(spec: BFMZeroRobotSpec) -> list[str]:
    return [name for name in spec.body_names if spec.foot_name in name]


def penalized_contact_body_names(spec: BFMZeroRobotSpec) -> list[str]:
    names: list[str] = []
    for key in spec.penalize_contacts_on:
        names.extend([body_name for body_name in spec.body_names if key in body_name])
    return names


@configclass
class BFMZeroSceneCfg(InteractiveSceneCfg):
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
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class BFMZeroCommandsCfg:
    motion = BFMZeroMotionCommandCfg(
        asset_name="robot",
        motion_file=BFMZERO_DEFAULT_MOTION_FILE,
        default_motion_id=0,
        reset_robot_state=True,
        lie_down_init_prob=0.3,
        lie_down_root_height=0.5,
    )


@configclass
class BFMZeroActionsCfg:
    pass


@configclass
class BFMZeroObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state = ObsTerm(
            func=bfm_observations.bfmzero_state_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "base_ang_vel_obs_scale": BFMZERO_BASE_ANG_VEL_OBS_SCALE,
            },
        )
        privileged_state = ObsTerm(
            func=bfm_observations.bfmzero_privileged_state_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "root_height_obs": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class BFMZeroRewardsCfg:
    pass


@configclass
class BFMZeroTerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class BFMZeroEventCfg:
    randomize_link_mass: EventTerm | None = None
    randomize_friction: EventTerm | None = None
    randomize_base_com: EventTerm | None = None
    randomize_default_dof_pos: EventTerm | None = None
    push_robot: EventTerm | None = None


@configclass
class BFMZeroCurriculumCfg:
    pass


@configclass
class BFMZeroManagerEnvCfg(ManagerBasedRLEnvCfg):
    scene: BFMZeroSceneCfg = BFMZeroSceneCfg(num_envs=1, env_spacing=5.0)
    observations: BFMZeroObservationsCfg = BFMZeroObservationsCfg()
    actions: BFMZeroActionsCfg = BFMZeroActionsCfg()
    commands: BFMZeroCommandsCfg = BFMZeroCommandsCfg()
    rewards: BFMZeroRewardsCfg = BFMZeroRewardsCfg()
    terminations: BFMZeroTerminationsCfg = BFMZeroTerminationsCfg()
    events: BFMZeroEventCfg = BFMZeroEventCfg()
    curriculum: BFMZeroCurriculumCfg = BFMZeroCurriculumCfg()

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


def apply_bfmzero_spec_to_manager_cfg(cfg: BFMZeroManagerEnvCfg, spec: BFMZeroRobotSpec) -> None:
    if hasattr(cfg.actions, "joint_pos"):
        cfg.actions.joint_pos.joint_names = list(spec.dof_names)
        cfg.actions.joint_pos.scale = spec.manager_action_scale_by_joint()
    cfg.commands.motion.robot_config = spec.config_name
    cfg.observations.policy.state.params["base_ang_vel_obs_scale"] = float(
        getattr(spec, "base_ang_vel_obs_scale", BFMZERO_BASE_ANG_VEL_OBS_SCALE)
    )
    cfg.observations.policy.privileged_state.params["asset_cfg"] = SceneEntityCfg(
        "robot",
        body_names=list(spec.body_names),
        preserve_order=True,
    )


def enable_bfmzero_domain_randomization(cfg: BFMZeroManagerEnvCfg, spec: BFMZeroRobotSpec) -> None:
    cfg.events.randomize_link_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": body_cfg(list(spec.randomize_link_body_names)),
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
        func=bfm_events.randomize_body_com,
        mode="startup",
        params={
            "asset_cfg": body_cfg([spec.torso_name]),
            "distribution_params": (
                (-0.02, -0.02, -0.02),
                (0.02, 0.02, 0.02),
            ),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    cfg.events.randomize_default_dof_pos = EventTerm(
        func=bfm_events.randomize_default_joint_pos_offset,
        mode="reset",
        params={
            "asset_cfg": all_joint_cfg(spec),
            "distribution_params": (-0.02, 0.02),
            "action_term_name": "joint_pos",
            "apply_during_eval": False,
        },
    )
    cfg.events.push_robot = EventTerm(
        func=bfm_events.push_by_setting_velocity_if_training,
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


def apply_bfmzero_runtime_to_manager_cfg(
    cfg: BFMZeroManagerEnvCfg,
    *,
    num_envs: int = 1,
    device: str = "cuda:0",
    motion_file: str | Path = BFMZERO_DEFAULT_MOTION_FILE,
    default_motion_id: int = 0,
    episode_length_s: float = 10.0,
    training_randomize_motions: bool = False,
    training_max_num_seqs: int | None = None,
    base_ang_vel_obs_scale: float | None = None,
) -> None:
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = device
    cfg.commands.motion.motion_file = str(motion_file)
    cfg.commands.motion.default_motion_id = int(default_motion_id)
    cfg.commands.motion.training_randomize_motions = bool(training_randomize_motions)
    cfg.commands.motion.training_max_num_seqs = training_max_num_seqs
    if base_ang_vel_obs_scale is not None:
        cfg.commands.motion.base_ang_vel_obs_scale = float(base_ang_vel_obs_scale)
        cfg.observations.policy.state.params["base_ang_vel_obs_scale"] = float(base_ang_vel_obs_scale)
    cfg.episode_length_s = float(episode_length_s)


def build_bfmzero_manager_env_from_cfg(
    cfg: BFMZeroManagerEnvCfg,
    *,
    render_mode: str | None = None,
    domain_randomization_enabled: bool = False,
) -> ManagerBasedRLEnv:
    env = ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode)
    env.bfmzero_domain_randomization_enabled = bool(domain_randomization_enabled)
    env.bfmzero_is_evaluating = False
    return env
