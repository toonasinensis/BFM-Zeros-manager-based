import typing as tp
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch.utils._pytree import tree_map

from bfm.agents.base import BaseConfig
from bfm.agents.buffers.trajectory import TrajectoryDictBuffer
from bfm.manager_envs.config.g1.g1_spec import (
    BFMZERO_DEFAULT_MOTION_FILE,
    BFMZERO_BASE_ANG_VEL_OBS_SCALE,
    BFMZERO_HISTORY_CONFIG,
    BFMZERO_HISTORY_ORDER,
    BFMZERO_NO_HEAD_ROBOT_CONFIG,
    assert_bfmzero_spec_consistent,
    load_bfmzero_g1_spec,
    resolve_repo_path,
)
from bfm.manager_envs.mdp.motion_provider import BFMZeroMotionProvider
from bfm.manager_envs.mdp.observations import compute_humanoid_observations_max
from bfm.utils.torch_utils import quat_rotate_inverse


def _to_numpy_tree(value):
    return tree_map(lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x, value)


def load_expert_trajectories_from_bfmzero_manager_motion_lib(env: "BFMZeroManagerVectorEnv", agent_cfg, device: str = "cpu"):
    provider = BFMZeroMotionProvider(
        motion_file=env.motion_file,
        spec=env.adapter.spec,
        num_envs=env.num_envs,
        device=env.device,
        base_ang_vel_obs_scale=getattr(env.adapter, "base_ang_vel_obs_scale", None),
    )
    provider.load_for_training()

    episodes = []
    file_names = []

    for local_motion_id in range(provider.motion_lib.num_motions()):
        motion_len = provider.motion_lib._motion_lengths[local_motion_id]
        motion_times = torch.arange(int(np.ceil((motion_len / env.adapter.step_dt).detach().cpu())), device=env.device) * env.adapter.step_dt
        motion_id = torch.full((motion_times.shape[0],), local_motion_id, dtype=torch.long, device=env.device)
        motion_res = provider.motion_lib.get_motion_state(motion_id, motion_times)
        file_names.append(provider.motion_lib.curr_motion_keys[local_motion_id])

        ref_body_pos = motion_res["rg_pos_t"]
        ref_body_rots_xyzw = motion_res["rg_rot_t_xyzw"]
        ref_body_vels = motion_res["body_vel_t"]
        ref_body_angular_vels = motion_res["body_ang_vel_t"]

        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots_xyzw,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=True,
            quat_format="xyzw",
        )
        privileged_state = torch.cat([value for value in obs_dict.values()], dim=-1)

        ref_dof_pos = motion_res["dof_pos"] - provider.default_joint_pos
        ref_dof_vel = motion_res["dof_vel"]
        ref_base_ang_vel = ref_body_angular_vels[:, 0] * provider.base_ang_vel_obs_scale
        projected_gravity = quat_rotate_inverse(
            ref_body_rots_xyzw[:, 0],
            provider.gravity_vec.repeat(privileged_state.shape[0], 1),
            w_last=True,
        )
        state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_base_ang_vel], dim=-1)
        last_action = torch.zeros_like(ref_dof_pos)

        history_actor_dim = 0
        dims = {
            "actions": last_action.shape[-1],
            "base_ang_vel": ref_base_ang_vel.shape[-1],
            "dof_pos": ref_dof_pos.shape[-1],
            "dof_vel": ref_dof_vel.shape[-1],
            "projected_gravity": projected_gravity.shape[-1],
        }
        for key in BFMZERO_HISTORY_ORDER:
            history_actor_dim += BFMZERO_HISTORY_CONFIG[key] * dims[key]
        history_actor = torch.zeros(state.shape[0], history_actor_dim, dtype=torch.float32, device=env.device)

        truncated = torch.zeros(state.shape[0], dtype=torch.bool, device=env.device)
        truncated[-1] = True
        episodes.append(
            {
                "observation": {
                    "state": state,
                    "last_action": last_action,
                    "privileged_state": privileged_state,
                    "history_actor": history_actor,
                },
                "terminated": torch.zeros(state.shape[0], dtype=torch.bool, device=env.device),
                "truncated": truncated,
                "motion_id": provider.motion_lib._curr_motion_ids[local_motion_id].repeat(state.shape[0]).long(),
            }
        )

    expert_buffer = TrajectoryDictBuffer(
        episodes=episodes,
        seq_length=agent_cfg.model.seq_length,
        device=device,
    )
    expert_buffer.file_names = file_names
    return expert_buffer


class BFMZeroManagerVectorEnv(VectorEnv):
    def __init__(
        self,
        adapter,
        *,
        motion_file: str | Path,
    ):
        super().__init__()
        self.adapter = adapter
        self._env = adapter
        self.num_envs = adapter.num_envs
        self.motion_file = str(motion_file)
        self._last_obs_torch, _ = self.reset(to_numpy=False)

        observation_spaces = {}
        for key, value in self._last_obs_torch.items():
            observation_spaces[key] = gymnasium.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=tuple(value.shape),
                dtype=np.float32,
            )
        self.observation_space = gymnasium.spaces.Dict(observation_spaces)

        action_limit = float(adapter.spec.normalize_action_from)
        self.single_action_space = gymnasium.spaces.Box(
            low=-action_limit,
            high=action_limit,
            shape=(adapter.num_actions,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.num_envs, 1)),
            high=np.tile(self.single_action_space.high, (self.num_envs, 1)),
            shape=(self.num_envs, adapter.num_actions),
            dtype=np.float32,
        )

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    @property
    def base_env(self):
        return self.adapter.unwrapped

    @property
    def unwrapped(self):
        return self.base_env

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.base_env.episode_length_buf

    @property
    def single_observation_space(self):
        single_obs_spaces = {}
        for key, space in self.observation_space.spaces.items():
            single_obs_spaces[key] = gymnasium.spaces.Box(
                low=space.low[0],
                high=space.high[0],
                shape=space.shape[1:],
                dtype=space.dtype,
            )
        return gymnasium.spaces.Dict(single_obs_spaces)

    def _manager_state_snapshot(self, to_numpy: bool = True) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        robot = self.adapter.robot
        qpos = torch.cat(
            [
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                robot.data.joint_pos[:, self.adapter.joint_indexes],
            ],
            dim=-1,
        ).detach()
        qvel = torch.cat(
            [
                robot.data.root_lin_vel_w,
                robot.data.root_ang_vel_b,
                robot.data.joint_vel[:, self.adapter.joint_indexes],
            ],
            dim=-1,
        ).detach()
        if to_numpy:
            return qpos.cpu().numpy(), qvel.cpu().numpy()
        return qpos, qvel

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        to_numpy: bool = True,
    ):
        del seed, options
        observation, info = self.adapter.reset()
        qpos, qvel = self._manager_state_snapshot(to_numpy=to_numpy)
        info = dict(info)
        info["qpos"] = qpos
        info["qvel"] = qvel
        if to_numpy:
            observation = _to_numpy_tree(observation)
        return observation, info

    def reset_to_motions(self, global_motion_ids, frame_id: int = 0, to_numpy: bool = True):
        if isinstance(global_motion_ids, np.ndarray):
            global_motion_ids_t = torch.tensor(global_motion_ids, dtype=torch.long, device=self.device)
        else:
            global_motion_ids_t = torch.as_tensor(global_motion_ids, dtype=torch.long, device=self.device)
        observation, info = self.adapter.reset_to_motions(global_motion_ids_t, frame_id=frame_id)
        qpos, qvel = self._manager_state_snapshot(to_numpy=to_numpy)
        info = dict(info)
        info["qpos"] = qpos
        info["qvel"] = qvel
        if to_numpy:
            observation = _to_numpy_tree(observation)
        return observation, info

    def step(self, actions, to_numpy: bool = True):
        if isinstance(actions, np.ndarray):
            actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        else:
            actions_t = actions.to(device=self.device, dtype=torch.float32)
        observation, reward, terminated, truncated, info = self.adapter.step(actions_t)
        qpos, qvel = self._manager_state_snapshot(to_numpy=to_numpy)
        info = dict(info)
        info["qpos"] = qpos
        info["qvel"] = qvel
        if to_numpy:
            observation = _to_numpy_tree(observation)
            reward = reward.detach().cpu().numpy()
            terminated = terminated.detach().cpu().numpy()
            truncated = truncated.detach().cpu().numpy()
        return observation, reward, terminated, truncated, info


    def update_motion_sampling_weights(self, priorities: list, motion_indexes: list, file_name: dict[int, str] | None = None) -> None:
        self.adapter.update_motion_sampling_weights(priorities=priorities, motion_indexes=motion_indexes, file_name=file_name)

    def set_is_evaluating(self, *args, **kwargs) -> None:
        self.adapter.set_is_evaluating(*args, **kwargs)

    def set_is_training(self) -> None:
        self.adapter.set_is_training()

    def close(self):
        return self.adapter.close()


_bfmzero_manager_env_singleton: BFMZeroManagerVectorEnv | None = None


class BFMZeroManagerIsaacConfig(BaseConfig):
    name: tp.Literal["bfmzero_manager_isaac"] = "bfmzero_manager_isaac"

    device: str = "cuda:0"
    lafan_tail_path: str = BFMZERO_DEFAULT_MOTION_FILE
    robot_config: str = BFMZERO_NO_HEAD_ROBOT_CONFIG
    enable_cameras: bool = False
    headless: bool = True
    max_episode_length_s: float | None = 10.0
    include_history_actor: bool = True
    root_height_obs: bool = True
    default_motion_id: int = 0
    training_randomize_motions: bool = True
    training_max_num_seqs: int | None = None
    base_ang_vel_obs_scale: float = BFMZERO_BASE_ANG_VEL_OBS_SCALE
    enable_domain_randomization: bool = False
    render_mode: str | None = None

    def build(self, num_envs: int = 1) -> tuple[BFMZeroManagerVectorEnv, dict[str, Any]]:
        global _bfmzero_manager_env_singleton
        if not self.include_history_actor:
            raise ValueError("BFMZero manager training requires include_history_actor=True.")
        if not self.root_height_obs:
            raise ValueError("BFMZero manager training currently expects root_height_obs=True.")
        assert num_envs >= 1

        if _bfmzero_manager_env_singleton is not None:
            if num_envs != _bfmzero_manager_env_singleton.num_envs:
                raise ValueError(
                    f"BFMZero manager env was already created with num_envs={_bfmzero_manager_env_singleton.num_envs}, "
                    f"but requested num_envs={num_envs}."
                )
            return _bfmzero_manager_env_singleton, {}

        from bfm.manager_envs.mdp.isaac_app import instantiate_isaac_sim

        instantiate_isaac_sim(num_envs, enable_cameras=self.enable_cameras, headless=self.headless)

        from bfm.manager_envs.mdp.adapter import BFMZeroManagerBuildConfig, BFMZeroManagerVectorEnvAdapter

        spec = load_bfmzero_g1_spec(self.robot_config)
        assert_bfmzero_spec_consistent(spec)
        motion_file = resolve_repo_path(self.lafan_tail_path)
        adapter = BFMZeroManagerVectorEnvAdapter.build(
            BFMZeroManagerBuildConfig(
                num_envs=num_envs,
                device=self.device,
                motion_file=motion_file,
                robot_config=self.robot_config,
                default_motion_id=self.default_motion_id,
                episode_length_s=float(self.max_episode_length_s or 10.0),
                training_randomize_motions=self.training_randomize_motions,
                training_max_num_seqs=self.training_max_num_seqs,
                base_ang_vel_obs_scale=self.base_ang_vel_obs_scale,
                enable_domain_randomization=self.enable_domain_randomization,
                render_mode=self.render_mode,
            )
        )
        env = BFMZeroManagerVectorEnv(adapter, motion_file=motion_file)
        _bfmzero_manager_env_singleton = env
        return env, {
            "robot_config": self.robot_config,
            "base_ang_vel_obs_scale": self.base_ang_vel_obs_scale,
            "enable_domain_randomization": self.enable_domain_randomization,
        }
