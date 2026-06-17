from __future__ import annotations

import gc
from pathlib import Path
import re
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


def _name_permutation(actual_names: tuple[str, ...], expected_names: tuple[str, ...], *, context: str, path: Path) -> np.ndarray:
    name_to_index: dict[str, int] = {}
    duplicates = []
    for index, name in enumerate(actual_names):
        if name in name_to_index:
            duplicates.append(name)
        name_to_index[name] = index
    if duplicates:
        raise ValueError(f"{path}: duplicate {context} names: {duplicates}")

    missing = [name for name in expected_names if name not in name_to_index]
    if missing:
        raise ValueError(f"{path}: missing {context} names required by spec: {missing}")
    return np.asarray([name_to_index[name] for name in expected_names], dtype=np.int64)


def _as_name_tuple(value: np.ndarray) -> tuple[str, ...]:
    return tuple(str(name) for name in value.reshape(-1).tolist())


def _slice_bounds(seq_len: int, max_len: int) -> tuple[int, int]:
    if max_len == -1 or seq_len < max_len:
        return 0, seq_len
    return 0, int(max_len)


_NAMED_NPZ_CHUNK_RE = re.compile(r"^(?P<family>.+)\.named_chunk(?P<chunk>\d+)_")


def _named_npz_legacy_key(path: Path) -> str | None:
    match = _NAMED_NPZ_CHUNK_RE.match(path.name)
    if match is None:
        return None
    return f"{match.group('family')}_clip{int(match.group('chunk'))}"


def _candidate_legacy_order_files(motion_path: Path) -> tuple[Path, ...]:
    roots = [motion_path.parent]
    if motion_path.is_file():
        roots.append(motion_path.parent.parent)
    return tuple(root / "lafan_29dof_10s-clipped.pkl" for root in roots)


def _legacy_order_for_named_npz(motion_path: Path) -> dict[str, int] | None:
    for legacy_path in _candidate_legacy_order_files(motion_path):
        if not legacy_path.is_file():
            continue
        motion_data = joblib.load(legacy_path)
        if not isinstance(motion_data, dict):
            raise TypeError(f"Expected legacy order pkl to contain a dict, got {type(motion_data)!r}: {legacy_path}")
        logger.info(f"Ordering named npz motions by legacy pkl order from {legacy_path}")
        return {str(key): index for index, key in enumerate(motion_data.keys())}
    return None


def _sort_named_npz_files(motion_path: Path, files: list[Path]) -> list[Path]:
    legacy_order = _legacy_order_for_named_npz(motion_path)
    if legacy_order is None:
        return sorted(files)

    missing = []

    def sort_key(path: Path) -> tuple[int, str]:
        legacy_key = _named_npz_legacy_key(path)
        if legacy_key is None or legacy_key not in legacy_order:
            missing.append(path.name)
            return len(legacy_order), path.name
        return legacy_order[legacy_key], path.name

    ordered = sorted(files, key=sort_key)
    if missing:
        logger.warning(f"{len(missing)} named npz motions are not in legacy order; keeping them after ordered motions: {missing[:5]}")
    return ordered


def _recompute_dof_velocity_like_legacy(dof_pos: torch.Tensor, dt: float) -> torch.Tensor:
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must be 2D, got shape {tuple(dof_pos.shape)}")
    if dof_pos.shape[0] <= 1:
        return torch.zeros_like(dof_pos)
    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / float(dt)
    tail = dof_vel[-2:-1] if dof_vel.shape[0] > 1 else dof_vel[-1:]
    return torch.cat([dof_vel, tail], dim=0)


class _LegacyPklMotionSource:
    format_name = "legacy_pkl"

    def __init__(self, motion_file: Path):
        if not motion_file.is_file():
            raise FileNotFoundError(f"BFM-Zero legacy motion source expects a pkl file, got: {motion_file}")
        motion_data = joblib.load(motion_file)
        if not isinstance(motion_data, dict):
            raise TypeError(f"Expected BFM-Zero motion pkl to contain a dict, got {type(motion_data)!r}.")
        self.motion_data_load = motion_data
        self.keys = np.array(list(motion_data.keys()))
        self.records = np.array(list(motion_data.values()), dtype=object)

    def build_motion(self, motion_lib: "BFMZeroLightMotionLib", motion_data: dict[str, Any], *, max_len: int = -1):
        return motion_lib._build_legacy_motion(motion_data, max_len=max_len)


class _NamedNpzMotionSource:
    format_name = "named_npz"
    required_keys = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    )

    def __init__(self, motion_path: Path, spec: BFMZeroG1Spec):
        self.spec = spec
        if motion_path.is_dir():
            files = _sort_named_npz_files(motion_path, list(motion_path.glob("*.npz")))
            if not files:
                raise FileNotFoundError(f"BFM-Zero named npz motion directory has no .npz files: {motion_path}")
        elif motion_path.is_file() and motion_path.suffix == ".npz":
            files = [motion_path]
        else:
            raise FileNotFoundError(f"BFM-Zero named npz motion source expects a .npz file or directory, got: {motion_path}")

        self.keys = np.array([path.name for path in files], dtype=object)
        self.records = np.array(files, dtype=object)
        self._joint_permutation_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._body_permutation_cache: dict[tuple[str, ...], np.ndarray] = {}

    def _validate_required_keys(self, data: np.lib.npyio.NpzFile, path: Path) -> None:
        missing = [key for key in self.required_keys if key not in data.files]
        if missing:
            raise KeyError(f"{path}: missing named npz motion keys: {missing}")

    def _joint_permutation(self, joint_names: tuple[str, ...], path: Path) -> np.ndarray:
        cached = self._joint_permutation_cache.get(joint_names)
        if cached is None:
            cached = _name_permutation(joint_names, self.spec.dof_names, context="joint", path=path)
            self._joint_permutation_cache[joint_names] = cached
        return cached

    def _body_permutation(self, body_names: tuple[str, ...], path: Path) -> np.ndarray:
        cached = self._body_permutation_cache.get(body_names)
        if cached is None:
            cached = _name_permutation(body_names, self.spec.observation_body_names, context="body", path=path)
            self._body_permutation_cache[body_names] = cached
        return cached

    def build_motion(self, motion_lib: "BFMZeroLightMotionLib", motion_path: Path, *, max_len: int = -1):
        motion_path = Path(motion_path)
        with np.load(motion_path, allow_pickle=False) as data:
            self._validate_required_keys(data, motion_path)
            fps_values = np.asarray(data["fps"]).reshape(-1)
            if fps_values.size == 0:
                raise ValueError(f"{motion_path}: fps is empty")
            fps = int(fps_values[0])
            if fps <= 0:
                raise ValueError(f"{motion_path}: fps must be positive, got {fps}")

            joint_names = _as_name_tuple(np.asarray(data["joint_names"]))
            body_names = _as_name_tuple(np.asarray(data["body_names"]))
            joint_perm = self._joint_permutation(joint_names, motion_path)
            body_perm = self._body_permutation(body_names, motion_path)

            joint_pos_raw = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel_raw = np.asarray(data["joint_vel"], dtype=np.float32)
            body_pos_raw = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_wxyz_raw = np.asarray(data["body_quat_w"], dtype=np.float32)
            body_lin_vel_raw = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
            body_ang_vel_raw = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

        if joint_pos_raw.ndim != 2 or joint_pos_raw.shape[1] != len(joint_names):
            raise ValueError(f"{motion_path}: joint_pos shape {joint_pos_raw.shape} does not match joint_names={len(joint_names)}")
        seq_len = int(joint_pos_raw.shape[0])
        expected_joint_shape = (seq_len, len(joint_names))
        if joint_vel_raw.shape != expected_joint_shape:
            raise ValueError(f"{motion_path}: joint_vel shape {joint_vel_raw.shape} != {expected_joint_shape}")

        expected_body_vec_shape = (seq_len, len(body_names), 3)
        expected_body_quat_shape = (seq_len, len(body_names), 4)
        for key, value, expected_shape in (
            ("body_pos_w", body_pos_raw, expected_body_vec_shape),
            ("body_quat_w", body_quat_wxyz_raw, expected_body_quat_shape),
            ("body_lin_vel_w", body_lin_vel_raw, expected_body_vec_shape),
            ("body_ang_vel_w", body_ang_vel_raw, expected_body_vec_shape),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{motion_path}: {key} shape {value.shape} != {expected_shape}")

        start, end = _slice_bounds(seq_len, max_len)
        joint_pos = np.ascontiguousarray(joint_pos_raw[start:end, joint_perm], dtype=np.float32)
        joint_vel_raw = np.ascontiguousarray(joint_vel_raw[start:end, joint_perm], dtype=np.float32)
        body_pos = np.ascontiguousarray(body_pos_raw[start:end, body_perm], dtype=np.float32)
        body_quat_xyzw = np.ascontiguousarray(body_quat_wxyz_raw[start:end, body_perm][..., [1, 2, 3, 0]], dtype=np.float32)
        body_lin_vel_raw = np.ascontiguousarray(body_lin_vel_raw[start:end, body_perm], dtype=np.float32)
        body_ang_vel_raw = np.ascontiguousarray(body_ang_vel_raw[start:end, body_perm], dtype=np.float32)

        quat_norm = np.linalg.norm(body_quat_xyzw, axis=-1, keepdims=True)
        if not np.isfinite(quat_norm).all() or np.any(quat_norm <= 1.0e-8):
            raise FloatingPointError(f"{motion_path}: body_quat_w contains non-finite or zero-length quaternions")
        body_quat_xyzw = np.ascontiguousarray(body_quat_xyzw / quat_norm, dtype=np.float32)

        for key, value in (
            ("joint_pos", joint_pos),
            ("joint_vel", joint_vel_raw),
            ("body_pos_w", body_pos),
            ("body_quat_w", body_quat_xyzw),
            ("body_lin_vel_w", body_lin_vel_raw),
            ("body_ang_vel_w", body_ang_vel_raw),
        ):
            if not np.isfinite(value).all():
                raise FloatingPointError(f"{motion_path}: {key} contains non-finite values")

        dt = 1.0 / fps
        body_pos_t = torch.from_numpy(body_pos)
        body_quat_t = torch.from_numpy(body_quat_xyzw)
        joint_pos_t = torch.from_numpy(joint_pos)
        body_lin_vel = motion_lib.kinematics._compute_velocity(body_pos_t.unsqueeze(0), dt).squeeze(0)
        body_ang_vel = motion_lib.kinematics._compute_angular_velocity(body_quat_t.unsqueeze(0), dt).squeeze(0)
        joint_vel = _recompute_dof_velocity_like_legacy(joint_pos_t, dt)

        num_motion_bodies = len(self.spec.motion_body_names)
        local_rotation = np.zeros_like(body_quat_xyzw)
        local_rotation[..., 3] = 1.0
        return EasyDict(
            {
                "global_translation": torch.from_numpy(body_pos[:, :num_motion_bodies]),
                "global_rotation": torch.from_numpy(body_quat_xyzw[:, :num_motion_bodies]),
                "local_rotation": torch.from_numpy(local_rotation),
                "global_root_velocity": body_lin_vel[:, 0],
                "global_root_angular_velocity": body_ang_vel[:, 0],
                "global_angular_velocity": body_ang_vel[:, :num_motion_bodies],
                "global_velocity": body_lin_vel[:, :num_motion_bodies],
                "dof_vels": joint_vel,
                "dof_pos": joint_pos_t,
                "global_translation_extend": torch.from_numpy(body_pos),
                "global_rotation_extend": torch.from_numpy(body_quat_xyzw),
                "global_velocity_extend": body_lin_vel,
                "global_angular_velocity_extend": body_ang_vel,
                "fps": fps,
            }
        )


def _make_motion_source(motion_file: Path, spec: BFMZeroG1Spec):
    if motion_file.is_dir() or motion_file.suffix == ".npz":
        return _NamedNpzMotionSource(motion_file, spec)
    return _LegacyPklMotionSource(motion_file)


class BFMZeroLightMotionLib:
    """Minimal BFM-Zero motion lib for manager training and tracking."""

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
        self._motion_source = _make_motion_source(motion_file, self.spec)
        self._motion_format = self._motion_source.format_name
        if hasattr(self._motion_source, "motion_data_load"):
            self._motion_data_load = self._motion_source.motion_data_load
        self._motion_data_keys = self._motion_source.keys
        self._motion_data_list = self._motion_source.records
        self._num_unique_motions = len(self._motion_data_list)
        logger.info(f"Loaded {self._num_unique_motions} lightweight BFM-Zero motions format={self._motion_format}")

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
            curr_motion = self._motion_source.build_motion(self, motion_data, max_len=max_len)
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

    def _build_legacy_motion(self, motion_data: dict[str, Any], *, max_len: int = -1):
        seq_len = int(motion_data["root_trans_offset"].shape[0])
        start, end = _slice_bounds(seq_len, max_len)
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
