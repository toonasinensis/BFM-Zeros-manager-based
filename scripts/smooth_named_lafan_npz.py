#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


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


def _smooth_float(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return values.astype(np.float32, copy=True)
    return gaussian_filter1d(values.astype(np.float32, copy=False), sigma=float(sigma), axis=0, mode="nearest").astype(np.float32)


def _smooth_quat_wxyz(quat: np.ndarray, sigma: float) -> np.ndarray:
    quat = _align_quat_signs(_normalize_quat_wxyz(quat.astype(np.float32, copy=False)))
    smoothed = _smooth_float(quat, sigma)
    return _normalize_quat_wxyz(smoothed)


def _velocity(values: np.ndarray, fps: int) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, axis=0).astype(np.float32) * float(fps)


def _angular_velocity_wxyz(quat: np.ndarray, fps: int) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat.astype(np.float32, copy=False))
    if quat.shape[0] <= 1:
        return np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)

    q_dot = np.gradient(quat, axis=0).astype(np.float32) * float(fps)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    dw, dx, dy, dz = np.moveaxis(q_dot, -1, 0)
    omega_x = -dw * x + dx * w - dy * z + dz * y
    omega_y = -dw * y + dx * z + dy * w - dz * x
    omega_z = -dw * z - dx * y + dy * x + dz * w
    return (2.0 * np.stack((omega_x, omega_y, omega_z), axis=-1)).astype(np.float32)


def smooth_file(src: Path, dst: Path, sigma: float, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    with np.load(src, allow_pickle=False) as data:
        fps_values = np.asarray(data["fps"]).reshape(-1)
        if fps_values.size == 0:
            raise ValueError(f"{src}: empty fps")
        fps = int(fps_values[0])
        if fps <= 0:
            raise ValueError(f"{src}: fps must be positive, got {fps}")

        joint_pos = _smooth_float(np.asarray(data["joint_pos"], dtype=np.float32), sigma)
        body_pos = _smooth_float(np.asarray(data["body_pos_w"], dtype=np.float32), sigma)
        body_quat = _smooth_quat_wxyz(np.asarray(data["body_quat_w"], dtype=np.float32), sigma)

        out = {key: data[key] for key in data.files if key not in {
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }}
        out["joint_pos"] = joint_pos
        out["body_pos_w"] = body_pos
        out["body_quat_w"] = body_quat
        out["joint_vel"] = _velocity(joint_pos, fps)
        out["body_lin_vel_w"] = _velocity(body_pos, fps)
        out["body_ang_vel_w"] = _angular_velocity_wxyz(body_quat, fps)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a smoothed 50Hz named LaFAN npz dataset.")
    parser.add_argument("--input-dir", type=Path, default=Path("bfm/data/named_lafan_10s"))
    parser.add_argument("--output-dir", type=Path, default=Path("bfm/data/named_lafan_10s_smooth50_sigma5"))
    parser.add_argument("--sigma", type=float, default=5.0)
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
        smooth_file(src, dst, args.sigma, args.overwrite)
        if index == 1 or index % 50 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] sigma={args.sigma:g} {src.name} -> {dst}")
    print(f"Done: {len(files)} files written under {args.output_dir}")


if __name__ == "__main__":
    main()
