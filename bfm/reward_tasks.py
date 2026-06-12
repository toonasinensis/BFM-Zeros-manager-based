from __future__ import annotations

import abc
import dataclasses
import inspect
import re
import sys
from typing import Optional

import mujoco
import numpy as np
from dm_control.utils import rewards

COORD_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALIGNMENT_BOUNDS = {"x": (-0.1, 0.1), "z": (0.9, float("inf")), "y": (-0.1, 0.1)}

REWARD_LIMITS = {
    "l": [0.6, 0.8, 0.2],
    "m": [1.0, float("inf"), 0.1],
}


def rot2eul(rotation: np.ndarray) -> np.ndarray:
    beta = -np.arcsin(rotation[2, 0])
    alpha = np.arctan2(rotation[2, 1] / np.cos(beta), rotation[2, 2] / np.cos(beta))
    gamma = np.arctan2(rotation[1, 0] / np.cos(beta), rotation[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def get_xpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise KeyError(f"MuJoCo body {name!r} is required for reward inference.")
    return data.xpos[body_id].copy()


def get_xmat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise KeyError(f"MuJoCo body {name!r} is required for reward inference.")
    return data.xmat[body_id].reshape((3, 3)).copy()


def get_torso_upright(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if body_id < 0:
        raise KeyError("MuJoCo body 'torso_link' is required for reward inference.")
    return float(data.xmat[body_id][-2])


def get_center_of_mass_linvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_link_subtreelinvel")
    if sensor_id < 0:
        raise KeyError("MuJoCo sensor 'torso_link_subtreelinvel' is required for reward inference.")
    start = model.sensor_adr[sensor_id]
    end = start + model.sensor_dim[sensor_id]
    return data.sensordata[start:end].copy()


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"MuJoCo sensor {name!r} is required for reward inference.")
    start = model.sensor_adr[sensor_id]
    end = start + model.sensor_dim[sensor_id]
    return data.sensordata[start:end].copy()


class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]: ...

    def __call__(self, model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray) -> float:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        return float(self.compute(model, data))


@dataclasses.dataclass
class ZeroReward(RewardFunction):
    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        return 0.0

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() in ("none", "zero", "rewardfree"):
            return ZeroReward()
        return None


@dataclasses.dataclass
class LocomotionReward(RewardFunction):
    move_speed: float = 5.0
    stand_height: float = 0.5
    move_angle: float = 0.0
    egocentric_target: bool = True
    stay_low: bool = False

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]

        if self.stay_low:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height * 0.95, self.stand_height * 1.05),
                margin=self.stand_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation

        if 0 <= self.move_speed <= 0.01:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.2).mean()
            angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
            return float(stand_reward * dont_move * dont_rotate)

        vel = center_of_mass_velocity[[0, 1]]
        com_velocity = np.linalg.norm(vel)
        move = rewards.tolerance(
            com_velocity,
            bounds=(self.move_speed - 0.1 * self.move_speed, self.move_speed + 0.1 * self.move_speed),
            margin=self.move_speed / 2,
            value_at_margin=0.5,
            sigmoid="gaussian",
        )
        move = (5 * move + 1) / 6
        if np.isclose(com_velocity, 0.0):
            angle_reward = 1.0
        else:
            direction = vel / (com_velocity + 1e-6)
            target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
            angle_reward = (target_direction.dot(direction) + 1.0) / 2.0
        return float(stand_reward * move * angle_reward)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        match = re.search(r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$", name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed)
        match = re.search(r"^move-ego-low(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$", name)
        if match:
            stand_height, move_angle, move_speed = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed, stay_low=True, stand_height=stand_height)
        return None


@dataclasses.dataclass
class RotationReward(RewardFunction):
    axis: str = "x"
    target_ang_velocity: float = 5.0
    stand_pelvis_height: float = 0.8

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)
        target_abs_velocity = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(target_abs_velocity, target_abs_velocity + 5),
            margin=target_abs_velocity / 2,
            value_at_margin=0,
            sigmoid="linear",
        )
        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )
        return float(move * height_reward * aligned)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        match = re.search(r"^rotate-(x|y|z)-(-?\d+\.*\d*)-(\d+\.*\d*)$", name)
        if match:
            axis, target_ang_velocity, stand_pelvis_height = match.group(1), float(match.group(2)), float(match.group(3))
            return RotationReward(axis=axis, target_ang_velocity=target_ang_velocity, stand_pelvis_height=stand_pelvis_height)
        return None


@dataclasses.dataclass
class ArmsReward(RewardFunction):
    left_pose: str = "m"
    right_pose: str = "m"
    stand_height: float = 0.5

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        standing = rewards.tolerance(
            root_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.2).mean()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5
        return float(stand_reward * dont_move * left_arm * right_arm * dont_rotate)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        match = re.search(r"^raisearms-(l|m|h|x)-(l|m|h|x)", name)
        if match:
            return ArmsReward(left_pose=match.group(1), right_pose=match.group(2))
        return None


@dataclasses.dataclass
class SitOnGroundReward(RewardFunction):
    pelvis_height_th: float = 0.0
    constrained_knees: bool = False
    knees_not_on_ground: bool = False

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        left_knee_pos = get_xpos(model, data, name="left_knee_link")[-1]
        right_knee_pos = get_xpos(model, data, name="right_knee_link")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
        pelvis_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.pelvis_height_th, self.pelvis_height_th + 0.1),
            sigmoid="linear",
            margin=0.7,
            value_at_margin=0,
        )
        knee_reward = 1
        if self.constrained_knees:
            knee_reward *= rewards.tolerance(left_knee_pos, bounds=(0, 0.1), sigmoid="linear", margin=0.7, value_at_margin=0)
            knee_reward *= rewards.tolerance(right_knee_pos, bounds=(0, 0.1), sigmoid="linear", margin=0.7, value_at_margin=0)
        if self.knees_not_on_ground:
            knee_reward *= rewards.tolerance(left_knee_pos, bounds=(0.2, 1), sigmoid="linear", margin=0.1, value_at_margin=0)
            knee_reward *= rewards.tolerance(right_knee_pos, bounds=(0.2, 1), sigmoid="linear", margin=0.1, value_at_margin=0)
        return float(cost_orientation * dont_move * dont_rotate * pelvis_reward * (2 * knee_reward + 1) / 3)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name == "sitonground":
            return SitOnGroundReward(pelvis_height_th=0.0, constrained_knees=True)
        match = re.search(r"^crouch-(\d+\.*\d*)$", name)
        if match:
            return SitOnGroundReward(pelvis_height_th=float(match.group(1)), knees_not_on_ground=True)
        return None


@dataclasses.dataclass
class MoveArmsReward(RewardFunction):
    move_speed: float = 5.0
    stand_height: float = 0.5
    move_angle: float = 0.0
    egocentric_target: bool = True
    low_height: float = 0.5
    stay_low: bool = False
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]

        if self.stay_low:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.low_height / 2, self.low_height),
                margin=self.low_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation

        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        if self.move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.2).mean()
            angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
            return float(stand_reward * dont_move * dont_rotate * left_arm * right_arm)

        vel = center_of_mass_velocity[[0, 1]]
        com_velocity = np.linalg.norm(vel)
        move = rewards.tolerance(
            com_velocity,
            bounds=(self.move_speed - 0.1 * self.move_speed, self.move_speed + 0.1 * self.move_speed),
            margin=self.move_speed / 2,
            value_at_margin=0.5,
            sigmoid="gaussian",
        )
        move = (5 * move + 1) / 6
        if np.isclose(com_velocity, 0.0):
            angle_reward = 1.0
        else:
            direction = vel / (com_velocity + 1e-6)
            target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
            angle_reward = (target_direction.dot(direction) + 1.0) / 2.0
        return float(stand_reward * move * angle_reward * left_arm * right_arm)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        match = re.search(r"^move-arms-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$", name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return MoveArmsReward(move_angle=move_angle, move_speed=move_speed, left_pose=match.group(3), right_pose=match.group(4))
        match = re.search(r"^move-ego-low-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$", name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return MoveArmsReward(
                move_angle=move_angle,
                move_speed=move_speed,
                stay_low=True,
                left_pose=match.group(3),
                right_pose=match.group(4),
            )
        return None


@dataclasses.dataclass
class SpinArmsReward(RewardFunction):
    axis: str = "z"
    target_ang_velocity: float = 5.0
    stand_pelvis_height: float = 0.5
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)
        target_abs_velocity = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(target_abs_velocity, target_abs_velocity + 5),
            margin=target_abs_velocity / 2,
            value_at_margin=0,
            sigmoid="linear",
        )
        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5
        return float(move * height_reward * aligned * left_arm * right_arm)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        match = re.search(r"^spin-arms-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$", name)
        if match:
            return SpinArmsReward(target_ang_velocity=float(match.group(1)), left_pose=match.group(2), right_pose=match.group(3))
        return None


def make_from_name(name: str | None = None) -> RewardFunction:
    if name is None:
        raise ValueError("Reward task name cannot be None.")
    module = sys.modules[__name__]
    for _, reward_cls in inspect.getmembers(module, inspect.isclass):
        if not issubclass(reward_cls, RewardFunction) or inspect.isabstract(reward_cls):
            continue
        reward_obj = reward_cls.reward_from_name(name)
        if reward_obj is not None:
            return reward_obj
    raise ValueError(f"Unknown reward name: {name}")
