from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bfm.utils.torch_utils import quat_rotate_inverse

from .light_motion_lib import BFMZeroLightMotionLib
from .observations import compute_humanoid_observations_max
from .spec import BFMZERO_DEFAULT_MOTION_FILE, BFMZeroG1Spec, load_bfmzero_g1_spec


def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return quat[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return quat[..., [1, 2, 3, 0]]


def finite_dict_assert(values: dict[str, torch.Tensor], *, context: str) -> None:
    bad = []
    for key, value in values.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            bad.append(key)
    if bad:
        raise FloatingPointError(f"Non-finite tensors in {context}: {bad}")


class BFMZeroMotionProvider:
    """BFM-Zero pkl motion provider with explicit global-to-local motion ids."""

    def __init__(
        self,
        *,
        motion_file: str | Path | None = None,
        spec: BFMZeroG1Spec | None = None,
        num_envs: int = 1,
        device: str | torch.device = "cuda:0",
    ):
        self.spec = spec or load_bfmzero_g1_spec()
        self.motion_file = Path(motion_file or BFMZERO_DEFAULT_MOTION_FILE)
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.motion_lib = BFMZeroLightMotionLib(
            motion_file=self.motion_file,
            spec=self.spec,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.current_global_motion_id: int | None = None
        self.training_motions_loaded = False
        self.default_joint_pos = torch.tensor(self.spec.default_joint_pos, dtype=torch.float32, device=self.device).reshape(1, -1)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).reshape(1, 3)

    @property
    def local_motion_id(self) -> int:
        if self.current_global_motion_id is None or self.motion_lib._curr_motion_ids is None:
            return 0
        curr_motion_ids = self.motion_lib._curr_motion_ids.to(device=self.device, dtype=torch.long)
        matches = (curr_motion_ids == int(self.current_global_motion_id)).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            raise KeyError(f"Global motion id {self.current_global_motion_id} is not loaded in the manager motion lib.")
        return int(matches[0].item())

    def load_for_global_motion(self, global_motion_id: int) -> None:
        global_motion_id = int(global_motion_id)
        if self.current_global_motion_id == global_motion_id:
            return
        if global_motion_id < 0 or global_motion_id >= self.motion_lib._num_unique_motions:
            raise IndexError(f"Motion id {global_motion_id} is outside [0, {self.motion_lib._num_unique_motions}).")
        self.motion_lib.load_motions_for_evaluation(start_idx=global_motion_id)
        self.current_global_motion_id = global_motion_id
        self.training_motions_loaded = False

    def load_for_global_motion_batch(self, global_motion_ids: torch.Tensor) -> None:
        global_motion_ids = global_motion_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if global_motion_ids.numel() == 0:
            return
        start_idx = int(global_motion_ids.min().item())
        self.motion_lib.load_motions_for_evaluation(start_idx=start_idx)
        self.current_global_motion_id = None
        self.training_motions_loaded = False

    def load_for_training(self, max_num_seqs: int | None = None) -> None:
        self.motion_lib.load_motions_for_training(max_num_seqs=max_num_seqs)
        self.current_global_motion_id = None
        self.training_motions_loaded = True

    def ensure_training_motions_loaded(self, max_num_seqs: int | None = None) -> None:
        if not self.training_motions_loaded:
            self.load_for_training(max_num_seqs=max_num_seqs)

    def local_ids_for_global(self, global_motion_ids: torch.Tensor) -> torch.Tensor:
        if self.motion_lib._curr_motion_ids is None:
            self.load_for_global_motion_batch(global_motion_ids)
        global_motion_ids = global_motion_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        curr_motion_ids = self.motion_lib._curr_motion_ids.to(device=self.device, dtype=torch.long)
        local_ids = []
        for global_motion_id in global_motion_ids.tolist():
            matches = (curr_motion_ids == int(global_motion_id)).nonzero(as_tuple=False).flatten()
            if matches.numel() == 0:
                self.load_for_global_motion_batch(global_motion_ids)
                curr_motion_ids = self.motion_lib._curr_motion_ids.to(device=self.device, dtype=torch.long)
                matches = (curr_motion_ids == int(global_motion_id)).nonzero(as_tuple=False).flatten()
            if matches.numel() == 0:
                raise KeyError(f"Global motion id {global_motion_id} is not loaded in the manager motion lib.")
            local_ids.append(matches[0])
        return torch.stack(local_ids).to(device=self.device, dtype=torch.long)

    def sample_training_motions(self, num_samples: int, *, truncate_time: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training_motions_loaded:
            self.load_for_training(max_num_seqs=min(self.num_envs, self.motion_lib._num_unique_motions))
        local_motion_ids = self.motion_lib.sample_motions(int(num_samples))
        global_motion_ids = self.motion_lib._curr_motion_ids[local_motion_ids].long()
        motion_times = self.motion_lib.sample_time(local_motion_ids, truncate_time=truncate_time)
        return local_motion_ids, global_motion_ids, motion_times

    def motion_length(self, global_motion_id: int) -> torch.Tensor:
        self.load_for_global_motion(global_motion_id)
        return self.motion_lib.get_motion_length(torch.tensor([self.local_motion_id], device=self.device))[0]

    def control_times(self, global_motion_id: int, step_dt: float) -> torch.Tensor:
        length = self.motion_length(global_motion_id)
        steps = int(torch.ceil(length / float(step_dt)).item())
        return torch.arange(steps, dtype=torch.float32, device=self.device) * float(step_dt)

    def state_at_times(self, global_motion_id: int, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:
        self.load_for_global_motion(global_motion_id)
        motion_times = motion_times.to(device=self.device, dtype=torch.float32).reshape(-1)
        motion_ids = torch.full((motion_times.shape[0],), self.local_motion_id, dtype=torch.long, device=self.device)
        return self.motion_lib.get_motion_state(motion_ids, motion_times)

    def state_at_local_times(self, local_motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:
        local_motion_ids = local_motion_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        motion_times = motion_times.to(device=self.device, dtype=torch.float32).reshape(-1)
        if local_motion_ids.numel() != motion_times.numel():
            raise ValueError(
                f"local_motion_ids has {local_motion_ids.numel()} values but motion_times has {motion_times.numel()} values."
            )
        return self.motion_lib.get_motion_state(local_motion_ids, motion_times)

    def state_at_time(self, global_motion_id: int, motion_time: float | torch.Tensor) -> dict[str, torch.Tensor]:
        motion_time_t = torch.as_tensor([motion_time], dtype=torch.float32, device=self.device)
        return self.state_at_times(global_motion_id, motion_time_t)

    def reference_backward_observation(
        self,
        global_motion_id: int,
        *,
        step_dt: float,
        use_root_height_obs: bool = True,
        velocity_multiplier: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        motion_times = self.control_times(global_motion_id, step_dt)
        motion_state = self.state_at_times(global_motion_id, motion_times)

        ref_body_pos = motion_state["rg_pos_t"]
        ref_body_rots = motion_state["rg_rot_t"]
        ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
        ref_body_angular_vels = motion_state["body_ang_vel_t"] * velocity_multiplier
        ref_dof_pos = motion_state["dof_pos"] - self.default_joint_pos
        ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier

        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_angular_vels,
            local_root_obs=True,
            root_height_obs=use_root_height_obs,
        )
        max_local_self_obs = torch.cat([value for value in obs_dict.values()], dim=-1)
        base_quat = ref_body_rots[:, 0]
        ref_ang_vel = ref_body_angular_vels[:, 0]
        projected_gravity = quat_rotate_inverse(
            base_quat,
            self.gravity_vec.repeat(max_local_self_obs.shape[0], 1),
            w_last=True,
        )
        state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], dim=-1)
        last_action = ref_dof_pos

        obs = {
            "state": state,
            "last_action": last_action,
            "privileged_state": max_local_self_obs,
        }
        ref_dict: dict[str, Any] = {
            "motion_times": motion_times,
            "dof_pos": motion_state["dof_pos"],
            "ref_dof_pos": ref_dof_pos,
            "ref_dof_vel": ref_dof_vel,
            "projected_gravity": projected_gravity,
            "ref_body_pos": ref_body_pos,
            "ref_body_rots": ref_body_rots,
            "ref_body_vels": ref_body_vels,
            "ref_body_angular_vels": ref_body_angular_vels,
            "max_local_self_obs": max_local_self_obs,
        }
        finite_dict_assert(obs, context=f"reference observation motion={global_motion_id}")
        return obs, ref_dict
