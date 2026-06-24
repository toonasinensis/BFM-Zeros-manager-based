from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Callable

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .motion_provider import BFMZeroMotionProvider
from .spec import BFMZERO_BASE_ANG_VEL_OBS_SCALE, BFMZERO_DEFAULT_MOTION_FILE, BFMZeroRobotSpec
from bfm.utils.torch_utils import quat_from_angle_axis, quat_mul, xyzw_to_wxyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BFMZeroMotionCommand(CommandTerm):
    cfg: "BFMZeroMotionCommandCfg"

    def __init__(self, cfg: "BFMZeroMotionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.spec: BFMZeroRobotSpec = cfg.load_robot_spec(cfg.robot_config)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.provider = BFMZeroMotionProvider(
            motion_file=cfg.motion_file,
            spec=self.spec,
            num_envs=self.num_envs,
            device=self.device,
            base_ang_vel_obs_scale=cfg.base_ang_vel_obs_scale,
        )
        self.local_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.global_motion_ids = torch.full(
            (self.num_envs,),
            int(cfg.default_motion_id),
            dtype=torch.long,
            device=self.device,
        )
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(list(self.spec.body_names), preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.joint_indexes = torch.tensor(
            self.robot.find_joints(list(self.spec.dof_names), preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.metrics["motion_id"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_time"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.stack((self.global_motion_ids.float(), self.motion_times), dim=-1)

    @property
    def step_dt(self) -> float:
        return float(getattr(self._env, "step_dt", self._env.cfg.sim.dt * self._env.cfg.decimation))

    def set_motion_time(
        self,
        global_motion_id: int,
        motion_time: float | torch.Tensor = 0.0,
        *,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        reset_robot: bool = True,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_t.numel() == 0:
            return env_ids_t

        global_motion_id = int(global_motion_id)
        self.provider.load_for_global_motion(global_motion_id)
        motion_time_t = torch.as_tensor(motion_time, dtype=torch.float32, device=self.device)
        if motion_time_t.ndim == 0:
            motion_time_t = motion_time_t.expand(env_ids_t.numel())
        if motion_time_t.numel() != env_ids_t.numel():
            raise ValueError(f"motion_time has {motion_time_t.numel()} values for {env_ids_t.numel()} env ids.")
        self.local_motion_ids[env_ids_t] = self.provider.local_motion_id
        self.global_motion_ids[env_ids_t] = global_motion_id
        self.motion_times[env_ids_t] = motion_time_t
        if reset_robot:
            self._reset_robot_to_current_motion(env_ids_t)
        return env_ids_t

    def set_motion_frame(
        self,
        global_motion_id: int,
        frame_id: int = 0,
        *,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        reset_robot: bool = True,
    ) -> torch.Tensor:
        return self.set_motion_time(
            global_motion_id,
            float(frame_id) * self.step_dt,
            env_ids=env_ids,
            reset_robot=reset_robot,
        )

    def set_training_motion_times(
        self,
        global_motion_ids: torch.Tensor,
        motion_times: float | torch.Tensor = 0.0,
        *,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        reset_robot: bool = True,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids_t.numel() == 0:
            return env_ids_t

        global_motion_ids_t = torch.as_tensor(global_motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        if global_motion_ids_t.numel() != env_ids_t.numel():
            raise ValueError(f"global_motion_ids has {global_motion_ids_t.numel()} values for {env_ids_t.numel()} env ids.")
        self.provider.load_for_global_motion_batch(global_motion_ids_t)
        motion_times_t = torch.as_tensor(motion_times, dtype=torch.float32, device=self.device)
        if motion_times_t.ndim == 0:
            motion_times_t = motion_times_t.expand(env_ids_t.numel())
        if motion_times_t.numel() != env_ids_t.numel():
            raise ValueError(f"motion_times has {motion_times_t.numel()} values for {env_ids_t.numel()} env ids.")

        local_motion_ids = self.provider.local_ids_for_global(global_motion_ids_t)
        self.local_motion_ids[env_ids_t] = local_motion_ids
        self.global_motion_ids[env_ids_t] = global_motion_ids_t
        self.motion_times[env_ids_t] = motion_times_t
        if reset_robot:
            self._reset_robot_to_current_motion(env_ids_t)
        return env_ids_t

    def current_motion_state(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device) if env_ids is None else env_ids
        return self.provider.state_at_local_times(self.local_motion_ids[env_ids], self.motion_times[env_ids])

    def _apply_lie_down_init(self, root_pos: torch.Tensor, root_rot_xyzw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prob = float(self.cfg.lie_down_init_prob)
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"lie_down_init_prob must be in [0, 1], got {prob}.")
        if prob == 0.0:
            return root_pos, root_rot_xyzw

        mask = torch.rand(root_pos.shape[0], device=root_pos.device) < prob
        if not bool(mask.any()):
            return root_pos, root_rot_xyzw

        root_pos = root_pos.clone()
        root_rot_xyzw = root_rot_xyzw.clone()
        root_pos[mask, 2] = float(self.cfg.lie_down_root_height)

        num_lie_down = int(mask.sum().item())
        sign = torch.where(
            torch.rand((), device=root_pos.device) < 0.5,
            torch.tensor(1.0, dtype=torch.float32, device=root_pos.device),
            torch.tensor(-1.0, dtype=torch.float32, device=root_pos.device),
        )
        angles = torch.full((num_lie_down,), -torch.pi / 2.0, dtype=torch.float32, device=root_pos.device) * sign
        axes = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=root_pos.device).reshape(1, 3).expand(num_lie_down, -1)
        rot_quat = quat_from_angle_axis(angles, axes, w_last=True)
        root_rot_xyzw[mask] = quat_mul(rot_quat, root_rot_xyzw[mask], w_last=True)
        return root_pos, root_rot_xyzw

    def _reset_robot_to_current_motion(self, env_ids: torch.Tensor, *, apply_lie_down: bool = False) -> None:
        state = self.current_motion_state(env_ids)
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_pos = state["root_pos"]
        root_rot_xyzw = state["root_rot_xyzw"]
        if apply_lie_down:
            root_pos, root_rot_xyzw = self._apply_lie_down_init(root_pos, root_rot_xyzw)
        root_state[:, :3] = root_pos + self._env.scene.env_origins[env_ids]
        root_state[:, 3:7] = xyzw_to_wxyz(root_rot_xyzw)
        root_state[:, 7:10] = state["root_vel"]
        root_state[:, 10:13] = state["root_ang_vel"]

        joint_pos = state["dof_pos"][:, : len(self.spec.dof_names)].clone()
        joint_vel = state["dof_vel"][:, : len(self.spec.dof_names)].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.joint_indexes, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.training_randomize_motions:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            if not self.provider.training_motions_loaded:
                if self.cfg.training_max_num_seqs is None:
                    max_num_seqs = min(self.num_envs, self.provider.motion_lib._num_unique_motions)
                else:
                    max_num_seqs = self.cfg.training_max_num_seqs
                self.provider.load_for_training(max_num_seqs=max_num_seqs)
            local_ids, global_ids, motion_times = self.provider.sample_training_motions(
                env_ids_t.numel(),
                truncate_time=self.cfg.training_truncate_time,
            )
            self.local_motion_ids[env_ids_t] = local_ids
            self.global_motion_ids[env_ids_t] = global_ids
            self.motion_times[env_ids_t] = motion_times
            if self.cfg.reset_robot_state:
                self._reset_robot_to_current_motion(env_ids_t, apply_lie_down=True)
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.set_motion_time(self.cfg.default_motion_id, 0.0, env_ids=env_ids_t, reset_robot=False)
        if self.cfg.reset_robot_state:
            self._reset_robot_to_current_motion(env_ids_t, apply_lie_down=True)

    def _update_command(self):
        self.motion_times += self.step_dt
        if not self.cfg.loop_motion:
            length = self.provider.motion_lib.get_motion_length(self.local_motion_ids)
            self.motion_times = torch.minimum(self.motion_times, length)
        self.metrics["motion_id"] = self.global_motion_ids.float()
        self.metrics["motion_time"] = self.motion_times

    def _update_metrics(self):
        state = self.current_motion_state()
        target = state["rg_pos_t"][:, : len(self.spec.body_names)]
        current = self.robot.data.body_pos_w[:, self.body_indexes] - self._env.scene.env_origins[:, None, :]
        body_pos_error = torch.norm(target - current, dim=-1)
        self.metrics["error_body_pos"] = body_pos_error.mean(dim=-1)
        self.metrics["max_error_body_pos"] = body_pos_error.max(dim=-1).values

    def _set_debug_vis_impl(self, debug_vis: bool):
        del debug_vis

    def _debug_vis_callback(self, event):
        del event


@configclass
class BFMZeroMotionCommandCfg(CommandTermCfg):
    class_type: type = BFMZeroMotionCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    asset_name: str = MISSING
    motion_file: str = BFMZERO_DEFAULT_MOTION_FILE
    robot_config: str = MISSING
    load_robot_spec: Callable[[str], BFMZeroRobotSpec] = MISSING
    default_motion_id: int = 0
    reset_robot_state: bool = True
    loop_motion: bool = False
    training_randomize_motions: bool = False
    training_max_num_seqs: int | None = None
    training_truncate_time: float | None = None
    base_ang_vel_obs_scale: float | None = BFMZERO_BASE_ANG_VEL_OBS_SCALE
    lie_down_init_prob: float = 0.0
    lie_down_root_height: float = 0.5
