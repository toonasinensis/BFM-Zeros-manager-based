from __future__ import annotations

from dataclasses import dataclass

from bfm.manager_envs.mdp.spec import (
    BFMZERO_ACTOR_KEYS,
    BFMZERO_BACKWARD_KEYS,
    BFMZERO_BASE_ANG_VEL_OBS_SCALE,
    BFMZERO_DEFAULT_MOTION_FILE,
    BFMZERO_HISTORY_CONFIG,
    BFMZERO_HISTORY_ORDER,
    BFMZERO_OBS_KEYS,
    BFMZeroRobotSpec,
    as_tuple,
    assert_model_matches_bfmzero_contract,
    assert_robot_spec_consistent,
    get_bfm_dir,
    load_robot_config,
    resolve_repo_path,
    robot_spec_kwargs,
)

BFMZERO_NO_HEAD_ROBOT_CONFIG = "g1/g1_29dof_hard_waist_no_head"
BFMZERO_ROBOT_CONFIG = BFMZERO_NO_HEAD_ROBOT_CONFIG


@dataclass(frozen=True)
class BFMZeroG1Spec(BFMZeroRobotSpec):
    ankle_roll_dof_names: tuple[str, ...]


def load_bfmzero_g1_spec(config_name: str = BFMZERO_ROBOT_CONFIG) -> BFMZeroG1Spec:
    cfg = load_robot_config(config_name)
    robot = cfg.robot
    return BFMZeroG1Spec(
        **robot_spec_kwargs(config_name, robot, bfm_dir=get_bfm_dir()),
        ankle_roll_dof_names=as_tuple(robot.ankle_roll_dof_names, str),
    )


def assert_bfmzero_spec_consistent(spec: BFMZeroG1Spec) -> None:
    assert_robot_spec_consistent(spec, expected_num_actions=29)
    missing = [joint_name for joint_name in spec.ankle_roll_dof_names if joint_name not in spec.dof_names]
    if missing:
        raise AssertionError(f"ankle_roll_dof_names contains joints not in dof_names: {missing}")


__all__ = [
    "BFMZERO_ACTOR_KEYS",
    "BFMZERO_BACKWARD_KEYS",
    "BFMZERO_BASE_ANG_VEL_OBS_SCALE",
    "BFMZERO_DEFAULT_MOTION_FILE",
    "BFMZERO_HISTORY_CONFIG",
    "BFMZERO_HISTORY_ORDER",
    "BFMZERO_NO_HEAD_ROBOT_CONFIG",
    "BFMZERO_OBS_KEYS",
    "BFMZERO_ROBOT_CONFIG",
    "BFMZeroG1Spec",
    "assert_bfmzero_spec_consistent",
    "assert_model_matches_bfmzero_contract",
    "load_bfmzero_g1_spec",
    "resolve_repo_path",
]
