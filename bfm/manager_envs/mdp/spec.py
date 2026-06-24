from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

BFM_PACKAGE_NAME = "bfm"
LEGACY_PACKAGE_NAME = "humanoidverse"

BFMZERO_DEFAULT_MOTION_FILE = "bfm/data/lafan_29dof_10s-clipped.pkl"

BFMZERO_OBS_KEYS = ("state", "privileged_state", "last_action", "history_actor")
BFMZERO_ACTOR_KEYS = ("state", "last_action", "history_actor")
BFMZERO_BACKWARD_KEYS = ("state", "privileged_state")
BFMZERO_HISTORY_CONFIG = {
    "base_ang_vel": 4,
    "projected_gravity": 4,
    "dof_pos": 4,
    "dof_vel": 4,
    "actions": 4,
}
BFMZERO_HISTORY_ORDER = tuple(sorted(BFMZERO_HISTORY_CONFIG))
BFMZERO_BASE_ANG_VEL_OBS_SCALE = 1.0


@dataclass(frozen=True)
class BFMZeroRobotSpec:
    """Common robot contract used by the manager MDP/motion code."""

    config_name: str
    bfm_dir: Path
    dof_names: tuple[str, ...]
    body_names: tuple[str, ...]
    motion_body_names: tuple[str, ...]
    default_joint_angles: dict[str, float]
    effort_limits: tuple[float, ...]
    velocity_limits: tuple[float, ...]
    armatures: tuple[float, ...]
    joint_frictions: tuple[float, ...]
    stiffness_by_key: dict[str, float]
    damping_by_key: dict[str, float]
    action_scale: float
    action_clip_value: float
    normalize_action_to: float
    normalize_action_from: float
    action_rescale: bool
    usd_path: Path
    mjcf_path: Path
    motion_asset_root: Path
    motion_asset_file_name: str
    motion_urdf_file_name: str
    torso_name: str
    foot_name: str
    penalize_contacts_on: tuple[str, ...]
    randomize_link_body_names: tuple[str, ...]
    num_bodies: int

    @property
    def num_actions(self) -> int:
        return len(self.dof_names)

    @property
    def observation_body_names(self) -> tuple[str, ...]:
        return self.motion_body_names

    @property
    def default_joint_pos(self) -> tuple[float, ...]:
        return tuple(float(self.default_joint_angles[name]) for name in self.dof_names)

    @property
    def action_obs_scale(self) -> float:
        if not self.normalize_action_to:
            return 1.0
        return float(self.normalize_action_to) / float(self.normalize_action_from)

    def p_gain_for_joint(self, joint_name: str) -> float:
        stem = joint_name.replace("_joint", "")
        for key, value in self.stiffness_by_key.items():
            if key in stem:
                return float(value)
        raise KeyError(f"No stiffness key in BFM-Zero config matches joint {joint_name!r}.")

    def d_gain_for_joint(self, joint_name: str) -> float:
        stem = joint_name.replace("_joint", "")
        for key, value in self.damping_by_key.items():
            if key in stem:
                return float(value)
        raise KeyError(f"No damping key in BFM-Zero config matches joint {joint_name!r}.")

    def manager_action_scale_by_joint(self) -> dict[str, float]:
        scales = {}
        for index, joint_name in enumerate(self.dof_names):
            scale = float(self.action_scale)
            if self.action_rescale:
                scale *= float(self.effort_limits[index]) / self.p_gain_for_joint(joint_name)
            scale *= self.action_obs_scale
            scales[joint_name] = scale
        return scales


def get_bfm_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path, *, bfm_dir: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    bfm_dir = bfm_dir or get_bfm_dir()
    repo_root = bfm_dir.parent
    raw = str(path)
    if raw in (BFM_PACKAGE_NAME, LEGACY_PACKAGE_NAME):
        return bfm_dir
    for package_name in (BFM_PACKAGE_NAME, LEGACY_PACKAGE_NAME):
        prefix = f"{package_name}/"
        if raw.startswith(prefix):
            return bfm_dir / raw[len(prefix) :]
    return repo_root / path


def as_tuple(values: Any, cast=float) -> tuple:
    return tuple(cast(value) for value in values)


_as_tuple = as_tuple


def load_robot_config(config_name: str):
    bfm_dir = get_bfm_dir()
    config_dir = bfm_dir / "config"
    robot_base = OmegaConf.load(config_dir / "robot" / "robot_base.yaml")
    robot_cfg = OmegaConf.load(config_dir / "robot" / f"{config_name}.yaml")
    return OmegaConf.merge(robot_base, robot_cfg)


_load_robot_config = load_robot_config


def robot_spec_kwargs(config_name: str, robot: Any, *, bfm_dir: Path | None = None) -> dict[str, Any]:
    bfm_dir = bfm_dir or get_bfm_dir()

    asset_root = resolve_repo_path(robot.asset.asset_root, bfm_dir=bfm_dir)
    motion_asset_root = resolve_repo_path(robot.motion.asset.assetRoot, bfm_dir=bfm_dir)
    usd_path = asset_root / robot.asset.usd_file
    mjcf_path = motion_asset_root / robot.motion.asset.assetFileName

    motion_body_names = robot.motion.get("body_names", robot.body_names)

    return {
        "config_name": config_name,
        "bfm_dir": bfm_dir,
        "dof_names": as_tuple(robot.dof_names, str),
        "body_names": as_tuple(robot.body_names, str),
        "motion_body_names": as_tuple(motion_body_names, str),
        "default_joint_angles": {str(k): float(v) for k, v in robot.init_state.default_joint_angles.items()},
        "effort_limits": as_tuple(robot.dof_effort_limit_list, float),
        "velocity_limits": as_tuple(robot.dof_vel_limit_list, float),
        "armatures": as_tuple(robot.dof_armature_list, float),
        "joint_frictions": as_tuple(robot.dof_joint_friction_list, float),
        "stiffness_by_key": {str(k): float(v) for k, v in robot.control.stiffness.items()},
        "damping_by_key": {str(k): float(v) for k, v in robot.control.damping.items()},
        "action_scale": float(robot.control.action_scale),
        "action_clip_value": float(robot.control.action_clip_value),
        "normalize_action_to": float(robot.control.normalize_action_to),
        "normalize_action_from": float(robot.control.normalize_action_from),
        "action_rescale": bool(robot.control.action_rescale),
        "usd_path": usd_path,
        "mjcf_path": mjcf_path,
        "motion_asset_root": motion_asset_root,
        "motion_asset_file_name": str(robot.motion.asset.assetFileName),
        "motion_urdf_file_name": str(robot.motion.asset.urdfFileName),
        "torso_name": str(robot.torso_name),
        "foot_name": str(robot.foot_name),
        "penalize_contacts_on": as_tuple(robot.penalize_contacts_on, str),
        "randomize_link_body_names": as_tuple(robot.get("randomize_link_body_names", robot.body_names), str),
        "num_bodies": int(robot.num_bodies),
    }


def assert_robot_spec_consistent(spec: BFMZeroRobotSpec, *, expected_num_actions: int | None = None) -> None:
    if expected_num_actions is not None and spec.num_actions != expected_num_actions:
        raise AssertionError(f"Expected {expected_num_actions} BFM-Zero actions, got {spec.num_actions}.")
    for name, values in {
        "effort_limits": spec.effort_limits,
        "velocity_limits": spec.velocity_limits,
        "armatures": spec.armatures,
        "joint_frictions": spec.joint_frictions,
        "default_joint_pos": spec.default_joint_pos,
    }.items():
        if len(values) != spec.num_actions:
            raise AssertionError(f"{name} has length {len(values)} but dof_names has length {spec.num_actions}.")
    if spec.body_names != spec.motion_body_names:
        raise AssertionError("robot.body_names and robot.motion.body_names must stay in the same order.")
    if len(spec.body_names) != spec.num_bodies:
        raise AssertionError(
            f"robot.num_bodies={spec.num_bodies} but body_names has length {len(spec.body_names)}."
        )
    missing_randomize_bodies = [name for name in spec.randomize_link_body_names if name not in spec.body_names]
    if missing_randomize_bodies:
        raise AssertionError(f"randomize_link_body_names contains bodies not in body_names: {missing_randomize_bodies}")
    if not spec.usd_path.is_file():
        raise FileNotFoundError(f"Robot USD not found: {spec.usd_path}")
    if not spec.mjcf_path.is_file():
        raise FileNotFoundError(f"Motion MJCF not found: {spec.mjcf_path}")


def assert_model_matches_bfmzero_contract(model: Any, obs: dict[str, Any] | None = None) -> None:
    action_dim = int(getattr(model, "action_dim", -1))
    if action_dim != 29:
        raise AssertionError(f"Checkpoint action_dim must be 29, got {action_dim}.")
    z_dim = int(model.cfg.archi.z_dim)
    if z_dim != 256:
        raise AssertionError(f"Checkpoint z_dim must be 256, got {z_dim}.")
    actor_keys = tuple(model.cfg.archi.actor.input_filter.key)
    backward_keys = tuple(model.cfg.archi.b.input_filter.key)
    if actor_keys != BFMZERO_ACTOR_KEYS:
        raise AssertionError(f"Actor keys mismatch: {actor_keys} != {BFMZERO_ACTOR_KEYS}")
    if backward_keys != BFMZERO_BACKWARD_KEYS:
        raise AssertionError(f"Backward keys mismatch: {backward_keys} != {BFMZERO_BACKWARD_KEYS}")
    if obs is None:
        return
    missing = [key for key in BFMZERO_OBS_KEYS if key not in obs]
    if missing:
        raise AssertionError(f"Manager observation is missing checkpoint keys: {missing}")
    if hasattr(model, "obs_space"):
        for key in BFMZERO_OBS_KEYS:
            expected = tuple(model.obs_space[key].shape)
            actual = tuple(obs[key].shape[1:])
            if actual != expected:
                raise AssertionError(f"Observation {key!r} shape mismatch: {actual} != {expected}")

__all__ = [
    "BFMZERO_ACTOR_KEYS",
    "BFMZERO_BACKWARD_KEYS",
    "BFMZERO_BASE_ANG_VEL_OBS_SCALE",
    "BFMZERO_DEFAULT_MOTION_FILE",
    "BFMZERO_HISTORY_CONFIG",
    "BFMZERO_HISTORY_ORDER",
    "BFMZERO_OBS_KEYS",
    "BFMZeroRobotSpec",
    "as_tuple",
    "assert_model_matches_bfmzero_contract",
    "assert_robot_spec_consistent",
    "get_bfm_dir",
    "load_robot_config",
    "resolve_repo_path",
    "robot_spec_kwargs",
]
