from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from easydict import EasyDict
from loguru import logger

from bfm.utils.torch_utils import slerp

from .g1_kinematics import BFMZeroG1Kinematics
from .spec import BFMZeroG1Spec


def _to_torch(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.from_numpy(value)


class BFMZeroLightMotionLib:
    """Minimal BFM-Zero pkl motion lib for manager training and tracking."""

    def __init__(
        self,
        *,
        motion_file: str | Path,
        spec: BFMZeroG1Spec,
        num_envs: int,
        device: str | torch.device,
        step_dt: float = 1 / 50,
    ):
        self.motion_file = Path(motion_file)
        self.spec = spec
        self.num_envs = int(num_envs)
        self._device = torch.device(device)
        self._sim_fps = 1 / float(step_dt)
        self.all_motions_loaded = False
        self.has_action = False

        self.kinematics = BFMZeroG1Kinematics(spec, device=torch.device("cpu"))
        self.num_joints = self.kinematics.num_bodies_augment
        self.num_bodies = self.kinematics.num_bodies

        logger.info(f"Loading lightweight BFM-Zero motion data from {self.motion_file}...")
        self._load_data(self.motion_file)
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions, device=self._device)
        self._success_rate = torch.zeros(self._num_unique_motions, device=self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions, device=self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions, device=self._device) / self._num_unique_motions
        self._sampling_batch_prob = self._sampling_prob
        self._num_motions = 0

    def _load_data(self, motion_file: str | Path) -> None:
        motion_file = Path(motion_file)
        if not motion_file.is_file():
            raise FileNotFoundError(f"BFM-Zero light motion lib only supports a pkl file, got: {motion_file}")
        motion_data = joblib.load(motion_file)
        if not isinstance(motion_data, dict):
            raise TypeError(f"Expected BFM-Zero motion pkl to contain a dict, got {type(motion_data)!r}.")
        self._motion_data_load = motion_data
        self._motion_data_keys = np.array(list(motion_data.keys()))
        self._motion_data_list = np.array(list(motion_data.values()), dtype=object)
        self._num_unique_motions = len(self._motion_data_list)
        logger.info(f"Loaded {self._num_unique_motions} lightweight BFM-Zero motions")

    def update_sampling_weight_by_id(self, priorities: list, motions_id: list, file_name: dict[int, str] | None = None) -> None:
        if len(motions_id) != len(priorities):
            raise AssertionError("motions_id and priorities must have the same length")
        priorities_t = torch.as_tensor(priorities, dtype=torch.float32, device=self._device)
        if not torch.isfinite(priorities_t).all() or priorities_t.sum() <= 0:
            raise AssertionError("Priorities must be finite and have a positive sum")

        new_sampling_prob = torch.zeros(self._num_unique_motions, dtype=torch.float32, device=self._device)
        normalized = priorities_t / priorities_t.sum()
        for raw_motion_id, priority in zip(motions_id, normalized, strict=True):
            motion_id = int(raw_motion_id)
            if file_name is not None:
                expected_name = str(self._motion_data_keys[motion_id])
                actual_name = str(file_name[motion_id])
                if expected_name != actual_name:
                    raise AssertionError(f"Motion ID {motion_id} does not match file name {actual_name!r} != {expected_name!r}")
            new_sampling_prob[motion_id] = priority

        self._sampling_prob = new_sampling_prob / new_sampling_prob.sum()
        self._update_sampling_batch_prob()

    def _update_sampling_batch_prob(self) -> None:
        if self._curr_motion_ids is None:
            return
        batch_prob = self._sampling_prob[self._curr_motion_ids]
        if batch_prob.sum() <= 0:
            batch_prob = torch.ones_like(batch_prob) / batch_prob.numel()
        else:
            batch_prob = batch_prob / batch_prob.sum()
        self._sampling_batch_prob = batch_prob

    def load_motions_for_training(self, max_num_seqs: int | None = None) -> None:
        if max_num_seqs is not None and max_num_seqs > self.num_envs:
            raise AssertionError("max_num_seqs must be <= num_envs")
        if self.all_motions_loaded:
            return
        if max_num_seqs is None or max_num_seqs >= self._num_unique_motions:
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        else:
            self.all_motions_loaded = False
            self.load_motions(random_sample=True, num_motions_to_load=max_num_seqs)

    def load_motions_for_evaluation(self, start_idx: int = 0) -> None:
        if self.all_motions_loaded:
            return
        num_motions = min(self.num_envs, self._num_unique_motions)
        self.all_motions_loaded = num_motions == self._num_unique_motions
        self.load_motions(random_sample=False, start_idx=start_idx, num_motions_to_load=num_motions)

    def load_motions(
        self,
        *,
        random_sample: bool = True,
        start_idx: int = 0,
        max_len: int = -1,
        num_motions_to_load: int | None = None,
    ) -> None:
        self._clear_loaded_tensors()
        num_motion_to_load = int(num_motions_to_load or self.num_envs)
        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.clamp(
                torch.arange(num_motion_to_load, device=self._device) + int(start_idx),
                max=self._num_unique_motions - 1,
            )
        self._curr_motion_ids = sample_idxes.long()
        self.curr_motion_keys = [str(self._motion_data_keys[int(idx)]) for idx in self._curr_motion_ids.detach().cpu().tolist()]
        self._update_sampling_batch_prob()

        logger.info(f"Light loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:10]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:10]}, ....")

        motions = []
        motion_lengths = []
        motion_fps = []
        motion_dt = []
        motion_num_frames = []
        motion_bodies = []
        motion_aa = []

        for motion_data in self._motion_data_list[sample_idxes.detach().cpu().numpy()]:
            curr_motion = self._build_motion(motion_data, max_len=max_len)
            num_frames = int(curr_motion.global_rotation.shape[0])
            fps = int(curr_motion.fps)
            dt = 1.0 / fps
            motion_lengths.append(dt * (num_frames - 1))
            motion_fps.append(fps)
            motion_dt.append(dt)
            motion_num_frames.append(num_frames)
            motion_bodies.append(torch.zeros(17))
            motion_aa.append(np.zeros((num_frames, self.num_joints * 3), dtype=np.float32))
            motions.append(curr_motion)

        self._motion_lengths = torch.tensor(motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(motion_num_frames, device=self._device)
        self._motion_bodies = torch.stack(motion_bodies).to(self._device).float()
        self._motion_aa = torch.tensor(np.concatenate(motion_aa), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)

        if hasattr(motions[0], "global_translation_extend"):
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)

        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)

        logger.info(
            f"Light loaded {self.num_motions():d} motions with a total length of {float(self.get_total_length()):.3f}s "
            f"and {self.gts.shape[0]} frames."
        )
        del motions
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _build_motion(self, motion_data: dict[str, Any], *, max_len: int = -1):
        seq_len = int(motion_data["root_trans_offset"].shape[0])
        if max_len == -1 or seq_len < max_len:
            start, end = 0, seq_len
        else:
            start, end = 0, max_len
        trans = _to_torch(motion_data["root_trans_offset"]).clone()[start:end]
        pose_aa = _to_torch(motion_data["pose_aa"]).clone()[start:end]
        dt = 1 / int(motion_data["fps"])
        curr_motion = self.kinematics.fk_batch(pose_aa[None], trans[None], return_full=True, dt=dt)
        return EasyDict({key: value.squeeze() if torch.is_tensor(value) else value for key, value in curr_motion.items()})

    def _clear_loaded_tensors(self) -> None:
        for name in (
            "gts",
            "grs",
            "lrs",
            "grvs",
            "gravs",
            "gavs",
            "gvs",
            "dvs",
            "dof_pos",
            "gts_t",
            "grs_t",
            "gvs_t",
            "gavs_t",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def num_motions(self) -> int:
        return self._num_motions

    def get_total_length(self):
        return self._motion_lengths.sum()

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps[motion_ids]).ceil().int()

    def sample_time(self, motion_ids: torch.Tensor, truncate_time: float | None = None) -> torch.Tensor:
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            if truncate_time < 0.0:
                raise AssertionError("truncate_time must be non-negative")
            motion_len = motion_len - float(truncate_time)
        return phase * motion_len

    def sample_motions(self, n: int) -> torch.Tensor:
        return torch.multinomial(self._sampling_batch_prob, num_samples=int(n), replacement=True).to(self._device)

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        return self._motion_lengths[motion_ids]

    def _calc_frame_blend(self, time: torch.Tensor, motion_len: torch.Tensor, num_frames: torch.Tensor, dt: torch.Tensor):
        time = time.clone()
        phase = torch.clip(time / motion_len, 0.0, 1.0)
        time[time < 0] = 0
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)
        return frame_idx0, frame_idx1, blend

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor, offset=None) -> dict[str, torch.Tensor]:
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.dof_pos[f0l]
        local_rot1 = self.dof_pos[f1l]
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]
        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]
        rg_pos0 = self.gts[f0l]
        rg_pos1 = self.gts[f1l]
        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        for value in (local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1):
            if value.dtype == torch.float64:
                raise AssertionError("BFM-Zero light motion tensors must not be float64")

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)
        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]
        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)

        if hasattr(self, "gts_t"):
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel

        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1).clone(),
            "motion_aa": self._motion_aa[f0l].clone(),
            "motion_bodies": self._motion_bodies[motion_ids].clone(),
            "rg_pos": rg_pos.clone(),
            "rb_rot": rb_rot.clone(),
            "body_vel": body_vel.clone(),
            "body_ang_vel": body_ang_vel.clone(),
            "rg_pos_t": rg_pos_t.clone(),
            "rg_rot_t": rg_rot_t.clone(),
            "body_vel_t": body_vel_t.clone(),
            "body_ang_vel_t": body_ang_vel_t.clone(),
        }
