from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from bfm.utils.torch_utils import quat_mul, quat_rotate, quat_rotate_inverse

from . import mdp
from .env_cfg import build_bfmzero_manager_env
from .motion_command import BFMZeroMotionCommand
from .observations import compute_humanoid_observations_max
from .motion_provider import finite_dict_assert, wxyz_to_xyzw
from .rewards import (
    BFMZERO_ENV_REWARD_SCALES,
    BFMZERO_INITIAL_PENALTY_SCALE,
    BFMZERO_PENALTY_REWARD_NAMES,
    BFMZERO_RAW_AUX_REWARD_NAMES,
)
from .spec import (
    BFMZERO_BASE_ANG_VEL_OBS_SCALE,
    BFMZERO_DEFAULT_MOTION_FILE,
    BFMZERO_HISTORY_CONFIG,
    BFMZERO_HISTORY_ORDER,
    BFMZERO_ROBOT_CONFIG,
    BFMZeroG1Spec,
    assert_bfmzero_spec_consistent,
    assert_model_matches_bfmzero_contract,
    load_bfmzero_g1_spec,
    resolve_repo_path,
)


class BFMZeroHistory:
    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dims = {
            "actions": 29,
            "base_ang_vel": 3,
            "dof_pos": 29,
            "dof_vel": 29,
            "projected_gravity": 3,
        }
        self.history = {
            key: torch.zeros(self.num_envs, length, self.dims[key], dtype=torch.float32, device=self.device)
            for key, length in BFMZERO_HISTORY_CONFIG.items()
        }

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            for value in self.history.values():
                value.zero_()
            return
        if env_ids.numel() == 0:
            return
        for value in self.history.values():
            value[env_ids] = 0.0

    def add(self, terms: dict[str, torch.Tensor]) -> None:
        for key, value in terms.items():
            if key not in self.history:
                continue
            old = self.history[key].clone()
            self.history[key][:, 1:] = old[:, :-1]
            self.history[key][:, 0] = value

    def query_actor(self) -> torch.Tensor:
        values = []
        for key in BFMZERO_HISTORY_ORDER:
            length = BFMZERO_HISTORY_CONFIG[key]
            value = self.history[key][:, :length]
            values.append(value.reshape(self.num_envs, -1))
        return torch.cat(values, dim=-1)


@dataclass
class BFMZeroManagerBuildConfig:
    num_envs: int = 1
    device: str = "cuda:0"
    motion_file: str | Path = BFMZERO_DEFAULT_MOTION_FILE
    robot_config: str = BFMZERO_ROBOT_CONFIG
    default_motion_id: int = 0
    episode_length_s: float = 10.0
    training_randomize_motions: bool = False
    training_max_num_seqs: int | None = None
    enable_domain_randomization: bool = False
    render_mode: str | None = None


class BFMZeroManagerVectorEnvAdapter:
    """Expose a manager-based IsaacLab env through the dict observation contract used by BFM-Zero checkpoints."""

    def __init__(self, env, *, spec: BFMZeroG1Spec | None = None):
        self.env = env
        self.unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self.spec = spec or load_bfmzero_g1_spec()
        assert_bfmzero_spec_consistent(self.spec)
        self.robot = self.unwrapped.scene["robot"]
        self.motion_command: BFMZeroMotionCommand = self.unwrapped.command_manager.get_term("motion")
        self.num_envs = int(self.unwrapped.num_envs)
        self.device = torch.device(self.unwrapped.device)
        self.joint_indexes = torch.tensor(
            self.robot.find_joints(list(self.spec.dof_names), preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(list(self.spec.body_names), preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.history = BFMZeroHistory(self.num_envs, self.device)
        self.last_action = torch.zeros(self.num_envs, self.spec.num_actions, dtype=torch.float32, device=self.device)
        self.unwrapped.bfmzero_last_action_obs = self.last_action.clone()
        self.aux_reward_names = tuple(name for name in BFMZERO_RAW_AUX_REWARD_NAMES if name in self._reward_term_names())
        self.assert_order_contract()

    @classmethod
    def build(cls, cfg: BFMZeroManagerBuildConfig | None = None) -> "BFMZeroManagerVectorEnvAdapter":
        cfg = cfg or BFMZeroManagerBuildConfig()
        motion_file = resolve_repo_path(cfg.motion_file)
        spec = load_bfmzero_g1_spec(cfg.robot_config)
        env = build_bfmzero_manager_env(
            num_envs=cfg.num_envs,
            device=cfg.device,
            motion_file=motion_file,
            robot_config=cfg.robot_config,
            default_motion_id=cfg.default_motion_id,
            episode_length_s=cfg.episode_length_s,
            training_randomize_motions=cfg.training_randomize_motions,
            training_max_num_seqs=cfg.training_max_num_seqs,
            enable_domain_randomization=cfg.enable_domain_randomization,
            render_mode=cfg.render_mode,
        )
        return cls(env, spec=spec)

    @property
    def step_dt(self) -> float:
        return float(self.unwrapped.step_dt)

    @property
    def num_actions(self) -> int:
        return self.spec.num_actions

    def assert_order_contract(self) -> None:
        found_joints = tuple(self.robot.joint_names[index] for index in self.joint_indexes.tolist())
        found_bodies = tuple(self.robot.body_names[index] for index in self.body_indexes.tolist())
        if found_joints != self.spec.dof_names:
            raise AssertionError(f"Manager joint lookup order mismatch: {found_joints} != {self.spec.dof_names}")
        if found_bodies != self.spec.body_names:
            raise AssertionError(f"Manager body lookup order mismatch: {found_bodies} != {self.spec.body_names}")
        total_action_dim = int(getattr(self.unwrapped.action_manager, "total_action_dim", -1))
        if total_action_dim != self.spec.num_actions:
            raise AssertionError(f"Manager action dim mismatch: {total_action_dim} != {self.spec.num_actions}")
        action_term = self.unwrapped.action_manager.get_term("joint_pos")
        action_joint_names = tuple(getattr(action_term, "_joint_names", ()))
        if action_joint_names and action_joint_names != self.spec.dof_names:
            raise AssertionError(f"Manager action joint order mismatch: {action_joint_names} != {self.spec.dof_names}")

    def order_contract(self) -> dict[str, Any]:
        return {
            "robot_config": self.spec.config_name,
            "dof_names": list(self.spec.dof_names),
            "body_names": list(self.spec.body_names),
            "motion_body_names": list(self.spec.motion_body_names),
            "extend_body_names": list(self.spec.extend_body_names),
            "manager_joint_names": [self.robot.joint_names[index] for index in self.joint_indexes.tolist()],
            "manager_body_names": [self.robot.body_names[index] for index in self.body_indexes.tolist()],
            "action_dim": self.num_actions,
            "enable_domain_randomization": bool(getattr(self.unwrapped, "bfmzero_domain_randomization_enabled", False)),
        }

    def set_is_evaluating(self, *args, **kwargs) -> None:
        del args, kwargs
        self.unwrapped.bfmzero_is_evaluating = True

    def set_is_training(self) -> None:
        self.unwrapped.bfmzero_is_evaluating = False

    def _reward_term_names(self) -> tuple[str, ...]:
        return tuple(getattr(self.unwrapped.reward_manager, "_term_names", ()))

    def assert_checkpoint_contract(self, model: Any, obs: dict[str, torch.Tensor] | None = None) -> None:
        assert_model_matches_bfmzero_contract(model, obs)

    def _processed_action_obs(self, action: torch.Tensor | None = None) -> torch.Tensor:
        if action is None:
            return self.last_action
        action = action.to(device=self.device, dtype=torch.float32)
        if action.shape != (self.num_envs, self.spec.num_actions):
            raise AssertionError(f"Action shape mismatch: {tuple(action.shape)} != {(self.num_envs, self.spec.num_actions)}")
        processed = action * self.spec.action_obs_scale
        processed = torch.clamp(processed, -self.spec.action_clip_value, self.spec.action_clip_value)
        return processed

    def _extend_body_tensors(  #[deprecate] self.spec.extend_body_names之后训练不应该加这个
        self,
        body_pos: torch.Tensor,
        body_quat_xyzw: torch.Tensor,
        body_vel: torch.Tensor,
        body_ang_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.spec.extend_body_names:
            return body_pos, body_quat_xyzw, body_vel, body_ang_vel

        extend_pos_values = []
        extend_quat_values = []
        extend_vel_values = []
        extend_ang_vel_values = []
        body_name_to_index = {name: index for index, name in enumerate(self.spec.body_names)}
        for parent_name, local_pos, local_rot_wxyz in zip(
            self.spec.extend_parent_names,
            self.spec.extend_pos,
            self.spec.extend_rot_wxyz,
            strict=True,
        ):
            parent_index = body_name_to_index[parent_name]
            parent_quat = body_quat_xyzw[:, parent_index]
            local_pos_t = torch.tensor(local_pos, dtype=torch.float32, device=self.device).reshape(1, 3).repeat(self.num_envs, 1)
            local_rot_xyzw = torch.tensor(
                [local_rot_wxyz[1], local_rot_wxyz[2], local_rot_wxyz[3], local_rot_wxyz[0]],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 4).repeat(self.num_envs, 1)
            rotated_pos = quat_rotate(parent_quat, local_pos_t, w_last=True)
            extend_pos = quat_rotate(local_rot_xyzw, rotated_pos, w_last=True) + body_pos[:, parent_index]
            extend_quat = quat_mul(parent_quat, local_rot_xyzw, w_last=True)
            extend_ang_vel = body_ang_vel[:, parent_index]
            angular_velocity_contribution = torch.cross(body_ang_vel[:, parent_index], local_pos_t, dim=1)
            extend_vel = body_vel[:, parent_index] + angular_velocity_contribution
            extend_pos_values.append(extend_pos)
            extend_quat_values.append(extend_quat)
            extend_vel_values.append(extend_vel)
            extend_ang_vel_values.append(extend_ang_vel)

        return (
            torch.cat([body_pos, torch.stack(extend_pos_values, dim=1)], dim=1),
            torch.cat([body_quat_xyzw, torch.stack(extend_quat_values, dim=1)], dim=1),
            torch.cat([body_vel, torch.stack(extend_vel_values, dim=1)], dim=1),
            torch.cat([body_ang_vel, torch.stack(extend_ang_vel_values, dim=1)], dim=1),
        )

    def _compute_terms(self, processed_action: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        root_quat_xyzw = wxyz_to_xyzw(self.robot.data.root_quat_w)
        gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).reshape(1, 3).repeat(self.num_envs, 1)
        projected_gravity = quat_rotate_inverse(root_quat_xyzw, gravity, w_last=True) 
        base_ang_vel = (
            quat_rotate_inverse(root_quat_xyzw, self.robot.data.root_ang_vel_w, w_last=True)
            * BFMZERO_BASE_ANG_VEL_OBS_SCALE
        )
        joint_pos = self.robot.data.joint_pos[:, self.joint_indexes]
        joint_vel = self.robot.data.joint_vel[:, self.joint_indexes]
        default_joint_pos = mdp.bfmzero_default_joint_pos(self.unwrapped, self.robot, self.joint_indexes)
        dof_pos = joint_pos - default_joint_pos
        body_pos = self.robot.data.body_pos_w[:, self.body_indexes]
        body_quat_xyzw = wxyz_to_xyzw(self.robot.data.body_quat_w[:, self.body_indexes])
        body_vel = self.robot.data.body_lin_vel_w[:, self.body_indexes]
        body_ang_vel = self.robot.data.body_ang_vel_w[:, self.body_indexes]
        body_pos, body_quat_xyzw, body_vel, body_ang_vel = self._extend_body_tensors(body_pos, body_quat_xyzw, body_vel, body_ang_vel)
        max_local_self_dict = compute_humanoid_observations_max(
            body_pos,
            body_quat_xyzw,
            body_vel,
            body_ang_vel,
            local_root_obs=True,
            root_height_obs=True,
        )
        privileged_state = torch.cat([value for value in max_local_self_dict.values()], dim=-1)
        action_obs = self._processed_action_obs(processed_action)
        terms = {
            "state": torch.cat([dof_pos, joint_vel, projected_gravity, base_ang_vel], dim=-1),
            "privileged_state": privileged_state,
            "last_action": action_obs,
            "history_actor": self.history.query_actor(),
            "actions": action_obs,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos,
            "dof_vel": joint_vel,
            "joint_pos_abs": joint_pos,
        }
        finite_dict_assert({key: value for key, value in terms.items() if key != "joint_pos_abs"}, context="manager observation")
        return terms

    def _observation_from_terms(self, terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "state": terms["state"],
            "privileged_state": terms["privileged_state"],
            "last_action": terms["last_action"],
            "history_actor": terms["history_actor"],
        }

    def get_observation(self) -> dict[str, torch.Tensor]:
        return self._observation_from_terms(self._compute_terms())

    def mujoco_qpos(self) -> torch.Tensor:
        root_pos = self.robot.data.root_pos_w
        root_quat_wxyz = self.robot.data.root_quat_w
        joint_pos = self.robot.data.joint_pos[:, self.joint_indexes]
        return torch.cat([root_pos, root_quat_wxyz, joint_pos], dim=-1).detach().cpu()

    def mujoco_qvel(self) -> torch.Tensor:
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_b
        joint_vel = self.robot.data.joint_vel[:, self.joint_indexes]
        return torch.cat([root_lin_vel, root_ang_vel, joint_vel], dim=-1).detach().cpu()

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.unwrapped.reset()
        self.history.reset()
        self.last_action.zero_()
        self.unwrapped.bfmzero_last_action_obs = self.last_action.clone()
        terms = self._compute_terms(self.last_action)
        obs = self._observation_from_terms(terms)
        return obs, {
            "joint_pos_abs": terms["joint_pos_abs"].detach().clone(),
            "motion_id": self.motion_command.global_motion_ids.detach().clone(),
            "motion_time": self.motion_command.motion_times.detach().clone(),
        }

    def reset_to_motion(self, global_motion_id: int, frame_id: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        previous_eval_mode = bool(getattr(self.unwrapped, "bfmzero_is_evaluating", False))
        self.unwrapped.bfmzero_is_evaluating = True
        try:
            self.unwrapped.reset()
            self.motion_command.set_motion_frame(global_motion_id, frame_id, reset_robot=True)
        finally:
            self.unwrapped.bfmzero_is_evaluating = previous_eval_mode
        self.unwrapped.scene.write_data_to_sim()
        self.unwrapped.sim.forward()
        self.history.reset()
        self.last_action.zero_()
        self.unwrapped.bfmzero_last_action_obs = self.last_action.clone()
        terms = self._compute_terms(self.last_action)
        obs = self._observation_from_terms(terms)
        return obs, {
            "joint_pos_abs": terms["joint_pos_abs"].detach().clone(),
            "motion_time": self.motion_command.motion_times.detach().clone(),
        }

    def reset_to_motions(
        self,
        global_motion_ids: torch.Tensor,
        frame_id: int = 0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        global_motion_ids = global_motion_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if global_motion_ids.numel() != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} motion ids, got {global_motion_ids.numel()}.")
        previous_eval_mode = bool(getattr(self.unwrapped, "bfmzero_is_evaluating", False))
        self.unwrapped.bfmzero_is_evaluating = True
        try:
            self.unwrapped.reset()
            motion_times = torch.full(
                (self.num_envs,),
                float(frame_id) * self.step_dt,
                dtype=torch.float32,
                device=self.device,
            )
            self.motion_command.set_training_motion_times(global_motion_ids, motion_times, reset_robot=True)
        finally:
            self.unwrapped.bfmzero_is_evaluating = previous_eval_mode
        self.unwrapped.scene.write_data_to_sim()
        self.unwrapped.sim.forward()
        self.history.reset()
        self.last_action.zero_()
        self.unwrapped.bfmzero_last_action_obs = self.last_action.clone()
        terms = self._compute_terms(self.last_action)
        obs = self._observation_from_terms(terms)
        return obs, {
            "joint_pos_abs": terms["joint_pos_abs"].detach().clone(),
            "motion_id": self.motion_command.global_motion_ids.detach().clone(),
            "motion_time": self.motion_command.motion_times.detach().clone(),
        }

    def update_motion_sampling_weights(self, priorities: list, motion_indexes: list, file_name: dict[int, str] | None = None) -> None:
        provider = self.motion_command.provider
        max_num_seqs = min(provider.num_envs, provider.motion_lib._num_unique_motions)
        provider.load_for_training(max_num_seqs=max_num_seqs)
        provider.motion_lib.update_sampling_weight_by_id(priorities=priorities, motions_id=motion_indexes, file_name=file_name)

    def get_backward_observation(self, global_motion_id: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self.motion_command.provider.reference_backward_observation(
            global_motion_id,
            step_dt=self.step_dt,
            use_root_height_obs=True,
        )

    def _reward_terms_from_reward_manager(self) -> dict[str, torch.Tensor]:
        names = self._reward_term_names()
        step_reward = self.unwrapped.reward_manager._step_reward
        missing = [name for name in BFMZERO_RAW_AUX_REWARD_NAMES if name not in names]
        if missing:
            raise AssertionError(f"Manager RewardManager is missing reward terms: {missing}")
        return {name: step_reward[:, index].detach().clone() for index, name in enumerate(names)}

    def _aux_rewards_from_terms(self, reward_terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: reward_terms[name].detach().clone() for name in BFMZERO_RAW_AUX_REWARD_NAMES}

    def _scaled_env_reward(self, reward_terms: dict[str, torch.Tensor]) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for name, scale in BFMZERO_ENV_REWARD_SCALES.items():
            term_scale = float(scale)
            if name in BFMZERO_PENALTY_REWARD_NAMES:
                term_scale *= BFMZERO_INITIAL_PENALTY_SCALE
            reward += reward_terms[name] * term_scale * self.step_dt
        return reward

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        processed_action = self._processed_action_obs(action)
        raw_action = action.to(device=self.device, dtype=torch.float32)
        _, _manager_reward, terminated, truncated, extras = self.unwrapped.step(raw_action)
        reset_ids = torch.logical_or(terminated, truncated).nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self.history.reset(reset_ids)
            processed_action[reset_ids] = 0.0
            raw_action = raw_action.clone()
            raw_action[reset_ids] = 0.0
        terms = self._compute_terms(raw_action)
        obs = self._observation_from_terms(terms)
        self.history.add(
            {
                "actions": terms["actions"],
                "base_ang_vel": terms["base_ang_vel"],
                "projected_gravity": terms["projected_gravity"],
                "dof_pos": terms["dof_pos"],
                "dof_vel": terms["dof_vel"],
            }
        )
        self.last_action = processed_action.detach().clone()
        self.unwrapped.bfmzero_last_action_obs = self.last_action.clone()
        reward_terms = self._reward_terms_from_reward_manager()
        aux_rewards = self._aux_rewards_from_terms(reward_terms)
        reward = self._scaled_env_reward(reward_terms)
        extras = dict(extras)
        extras["joint_pos_abs"] = terms["joint_pos_abs"].detach().clone()
        extras["aux_rewards"] = aux_rewards
        return obs, reward, terminated, truncated, extras

    def close(self) -> None:
        self.unwrapped.close()
