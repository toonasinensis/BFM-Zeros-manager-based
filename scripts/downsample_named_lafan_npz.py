#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1.0e-8):
        raise FloatingPointError("Quaternion contains non-finite or zero-length values.")
    return (quat / norm).astype(np.float32, copy=False)


def _align_quat_signs(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    flat = quat.reshape(quat.shape[0], -1, 4)
    for t in range(1, flat.shape[0]):
        dot = np.sum(flat[t - 1] * flat[t], axis=-1, keepdims=True)
        flat[t] = np.where(dot < 0.0, -flat[t], flat[t])
    return quat


def _interp_linear(values: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    out = np.empty((dst_t.shape[0], flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst_t, src_t, flat[:, col]).astype(np.float32)
    return out.reshape((dst_t.shape[0],) + values.shape[1:])


def _interp_quat_wxyz(quat: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    quat = _align_quat_signs(_normalize_quat_wxyz(quat.astype(np.float32, copy=False)))
    flat = quat.reshape(quat.shape[0], -1, 4)
    dst = np.empty((dst_t.shape[0], flat.shape[1], 4), dtype=np.float32)

    idx1 = np.searchsorted(src_t, dst_t, side="right")
    idx1 = np.clip(idx1, 1, src_t.shape[0] - 1)
    idx0 = idx1 - 1
    denom = np.maximum(src_t[idx1] - src_t[idx0], 1.0e-8)
    alpha = ((dst_t - src_t[idx0]) / denom).astype(np.float32)

    for out_i, (i0, i1, a) in enumerate(zip(idx0, idx1, alpha, strict=True)):
        q0 = flat[i0]
        q1 = flat[i1]
        dot = np.sum(q0 * q1, axis=-1, keepdims=True)
        q1 = np.where(dot < 0.0, -q1, q1)
        dot = np.abs(dot)
        linear_mask = dot > 0.9995

        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * a
        sin_theta = np.sin(theta)
        s0 = np.cos(theta) - dot * sin_theta / np.maximum(sin_theta_0, 1.0e-8)
        s1 = sin_theta / np.maximum(sin_theta_0, 1.0e-8)
        slerped = s0 * q0 + s1 * q1
        lerped = (1.0 - a) * q0 + a * q1
        dst[out_i] = np.where(linear_mask, lerped, slerped)

    return _normalize_quat_wxyz(dst.reshape((dst_t.shape[0],) + quat.shape[1:]))


def _velocity(values: np.ndarray, fps: int) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, axis=0).astype(np.float32) * float(fps)


def _angular_velocity_wxyz(quat: np.ndarray, fps: int) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat.astype(np.float32, copy=False))
    if quat.shape[0] <= 1:
        return np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)

    # Central-difference quaternion derivative. For unit quats, omega = 2 * q_dot * conj(q).
    q_dot = np.gradient(quat, axis=0).astype(np.float32) * float(fps)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    dw, dx, dy, dz = np.moveaxis(q_dot, -1, 0)

    # q_dot * conjugate(q), returning xyz components in world frame convention.
    omega_x = -dw * x + dx * w - dy * z + dz * y
    omega_y = -dw * y + dx * z + dy * w - dz * x
    omega_z = -dw * z - dx * y + dy * x + dz * w
    return (2.0 * np.stack((omega_x, omega_y, omega_z), axis=-1)).astype(np.float32)


def _target_times(num_frames: int, src_fps: int, dst_fps: int) -> np.ndarray:
    duration = (num_frames - 1) / float(src_fps)
    dst_frames = int(np.floor(duration * dst_fps + 1.0e-6)) + 1
    return np.arange(dst_frames, dtype=np.float32) / float(dst_fps)


def downsample_file(src: Path, dst: Path, dst_fps: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    with np.load(src, allow_pickle=False) as data:
        src_fps_values = np.asarray(data["fps"]).reshape(-1)
        if src_fps_values.size == 0:
            raise ValueError(f"{src}: empty fps")
        src_fps = int(src_fps_values[0])
        if src_fps <= 0:
            raise ValueError(f"{src}: fps must be positive, got {src_fps}")

        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        src_t = np.arange(joint_pos.shape[0], dtype=np.float32) / float(src_fps)
        dst_t = _target_times(joint_pos.shape[0], src_fps, dst_fps)

        out = {key: data[key] for key in data.files if key not in {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }}
        out["fps"] = np.asarray([dst_fps], dtype=np.int64)
        out["joint_pos"] = _interp_linear(joint_pos, src_t, dst_t)
        out["body_pos_w"] = _interp_linear(body_pos, src_t, dst_t)
        out["body_quat_w"] = _interp_quat_wxyz(body_quat, src_t, dst_t)
        out["joint_vel"] = _velocity(out["joint_pos"], dst_fps)
        out["body_lin_vel_w"] = _velocity(out["body_pos_w"], dst_fps)
        out["body_ang_vel_w"] = _angular_velocity_wxyz(out["body_quat_w"], dst_fps)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample named LaFAN npz motions to a target fps.")
    parser.add_argument("--input-dir", type=Path, default=Path("bfm/data/named_lafan_10s"))
    parser.add_argument("--output-dir", type=Path, default=Path("bfm/data/named_lafan_10s_30hz"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {args.input_dir}")
    if args.limit is not None:
        files = files[: args.limit]

    for index, src in enumerate(files, start=1):
        dst = args.output_dir / src.name
        downsample_file(src, dst, args.fps, args.overwrite)
        if index == 1 or index % 50 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {src.name} -> {dst}")

    print(f"Done: {len(files)} files written under {args.output_dir}")


if __name__ == "__main__":
    main()
