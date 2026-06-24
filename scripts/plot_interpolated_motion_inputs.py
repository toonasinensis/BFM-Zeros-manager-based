#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bfm.manager_envs.g1.motion_provider import BFMZeroMotionProvider
from bfm.manager_envs.g1.observations import compute_humanoid_observations_max
from bfm.manager_envs.g1.spec import load_bfmzero_g1_spec
from bfm.utils.torch_utils import quat_rotate_inverse


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


def _parse_dataset(raw: str) -> DatasetSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Dataset must be NAME=PATH.")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Dataset name is empty.")
    return DatasetSpec(name=name, path=Path(path).expanduser())


def _default_datasets() -> list[DatasetSpec]:
    return [
        DatasetSpec("named50", Path("bfm/data/named_lafan_10s")),
        DatasetSpec("named30", Path("bfm/data/named_lafan_10s_30hz")),
        DatasetSpec("legacy30", Path("bfm/data/lafan_29dof_10s-clipped.pkl")),
    ]


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().float().numpy()


def _make_network_inputs(provider: BFMZeroMotionProvider, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    ref_body_pos = state["rg_pos_t"]
    ref_body_rots_xyzw = state["rg_rot_t_xyzw"]
    ref_body_vels = state["body_vel_t"]
    ref_body_angular_vels = state["body_ang_vel_t"]
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
    ref_dof_pos = state["dof_pos"] - provider.default_joint_pos
    ref_dof_vel = state["dof_vel"]
    ref_ang_vel = ref_body_angular_vels[:, 0]
    projected_gravity = quat_rotate_inverse(
        ref_body_rots_xyzw[:, 0],
        provider.gravity_vec.repeat(privileged_state.shape[0], 1),
        w_last=True,
    )
    policy_state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], dim=-1)
    return {
        "state": policy_state,
        "privileged_state": privileged_state,
        "ref_dof_pos": ref_dof_pos,
        "ref_dof_vel": ref_dof_vel,
        "projected_gravity": projected_gravity,
        "ref_root_ang_vel": ref_ang_vel,
        "local_body_vel": obs_dict["local_body_vel"],
        "local_body_ang_vel": obs_dict["local_body_ang_vel"],
    }


def _sample_motion(provider: BFMZeroMotionProvider, local_motion_id: int, step_dt: float) -> dict[str, np.ndarray]:
    motion_lib = provider.motion_lib
    length = motion_lib._motion_lengths[local_motion_id]
    num_steps = int(np.ceil((length / float(step_dt)).detach().cpu()))
    motion_times = torch.arange(num_steps, dtype=torch.float32, device=provider.device) * float(step_dt)
    motion_ids = torch.full((num_steps,), int(local_motion_id), dtype=torch.long, device=provider.device)
    state = motion_lib.get_motion_state(motion_ids, motion_times)
    network = _make_network_inputs(provider, state)

    body_vel = state["body_vel_t"].float()
    body_ang = state["body_ang_vel_t"].float()
    dof_vel = state["dof_vel"].float()
    root_vel = state["root_vel"].float()
    root_ang = state["root_ang_vel"].float()
    root_pos = state["root_pos"].float()

    def diff_norm(value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] <= 1:
            return torch.zeros((0,), dtype=torch.float32, device=value.device)
        return ((value[1:] - value[:-1]) / float(step_dt)).norm(dim=-1)

    return {
        "time": _as_numpy(motion_times),
        "root_speed": _as_numpy(root_vel.norm(dim=-1)),
        "root_ang_speed": _as_numpy(root_ang.norm(dim=-1)),
        "dof_vel_l2": _as_numpy(dof_vel.norm(dim=-1)),
        "body_speed_mean": _as_numpy(body_vel.norm(dim=-1).mean(dim=-1)),
        "body_ang_speed_mean": _as_numpy(body_ang.norm(dim=-1).mean(dim=-1)),
        "root_height": _as_numpy(root_pos[:, 2]),
        "state_abs_mean": _as_numpy(network["state"].abs().mean(dim=-1)),
        "privileged_abs_mean": _as_numpy(network["privileged_state"].abs().mean(dim=-1)),
        "root_acc": _as_numpy(diff_norm(root_vel)),
        "dof_acc_l2": _as_numpy(diff_norm(dof_vel)),
        "body_acc_mean": _as_numpy(diff_norm(body_vel).mean(dim=-1)),
    }


def _concat_metrics(provider: BFMZeroMotionProvider, step_dt: float, max_motions: int | None) -> dict[str, np.ndarray]:
    motion_lib = provider.motion_lib
    count = motion_lib.num_motions() if max_motions is None else min(int(max_motions), motion_lib.num_motions())
    values: dict[str, list[np.ndarray]] = {}
    for local_motion_id in range(count):
        sampled = _sample_motion(provider, local_motion_id, step_dt)
        for key, value in sampled.items():
            if key == "time":
                continue
            values.setdefault(key, []).append(value.reshape(-1))
    return {key: np.concatenate(parts) for key, parts in values.items() if parts}


def _load_provider(dataset: DatasetSpec, num_envs: int, device: str) -> BFMZeroMotionProvider:
    provider = BFMZeroMotionProvider(
        motion_file=dataset.path,
        spec=load_bfmzero_g1_spec(),
        num_envs=num_envs,
        device=device,
    )
    provider.load_for_training()
    return provider


def _plot_motion_curves(
    samples: dict[str, dict[int, dict[str, np.ndarray]]],
    motion_names: dict[str, dict[int, str]],
    motion_ids: list[int],
    output_path: Path,
) -> None:
    metrics = [
        ("root_speed", "Root Speed"),
        ("root_ang_speed", "Root Angular Speed"),
        ("dof_vel_l2", "DOF Velocity L2"),
        ("body_speed_mean", "Mean Body Speed"),
        ("body_ang_speed_mean", "Mean Body Angular Speed"),
        ("state_abs_mean", "Network State |.| Mean"),
        ("privileged_abs_mean", "Privileged State |.| Mean"),
        ("root_height", "Root Height"),
    ]
    dataset_names = list(samples.keys())
    fig, axes = plt.subplots(len(motion_ids), len(metrics), figsize=(4.2 * len(metrics), 2.8 * len(motion_ids)), squeeze=False)
    colors = {
        "named50": "#3b82f6",
        "named30": "#db2777",
        "legacy30": "#84cc16",
    }
    for row, motion_id in enumerate(motion_ids):
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row][col]
            for dataset_name in dataset_names:
                data = samples[dataset_name][motion_id]
                y = data[metric]
                x = data["time"][: y.shape[0]]
                ax.plot(x, y, label=dataset_name, linewidth=1.1, alpha=0.92, color=colors.get(dataset_name))
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                first_name = next(iter(motion_names.values())).get(motion_id, str(motion_id))
                ax.set_ylabel(f"id {motion_id}\n{first_name[:34]}", fontsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.35)
            if row == len(motion_ids) - 1:
                ax.set_xlabel("time (s)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(dataset_names), frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Interpolated Expert Inputs at Training Control Step (0.02s)", y=0.999, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _plot_distribution(summary: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    metrics = [
        ("root_speed", "Root Speed"),
        ("root_ang_speed", "Root Angular Speed"),
        ("dof_vel_l2", "DOF Velocity L2"),
        ("body_speed_mean", "Mean Body Speed"),
        ("body_ang_speed_mean", "Mean Body Angular Speed"),
        ("root_acc", "Root Acceleration"),
        ("dof_acc_l2", "DOF Acceleration L2"),
        ("body_acc_mean", "Mean Body Acceleration"),
        ("state_abs_mean", "Network State |.| Mean"),
        ("privileged_abs_mean", "Privileged State |.| Mean"),
    ]
    dataset_names = list(summary.keys())
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.2), squeeze=False)
    positions = np.arange(len(dataset_names), dtype=np.float32)
    for ax, (metric, title) in zip(axes.reshape(-1), metrics, strict=True):
        for idx, dataset_name in enumerate(dataset_names):
            values = summary[dataset_name][metric]
            q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
            ax.plot([idx, idx], [q[0], q[4]], color="#9ca3af", linewidth=1.5)
            ax.plot([idx - 0.18, idx + 0.18], [q[2], q[2]], color="#111827", linewidth=2.0)
            ax.fill_between([idx - 0.14, idx + 0.14], q[1], q[3], color="#60a5fa", alpha=0.35)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(positions)
        ax.set_xticklabels(dataset_names, rotation=20)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    fig.suptitle("Distribution After Interpolation to Network/Expert Inputs", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _write_summary_csv(summary: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    rows = []
    for dataset_name, metrics in summary.items():
        for metric, values in metrics.items():
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            row = {
                "dataset": dataset_name,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p05": float(np.quantile(values, 0.05)),
                "p25": float(np.quantile(values, 0.25)),
                "p50": float(np.quantile(values, 0.50)),
                "p75": float(np.quantile(values, 0.75)),
                "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)),
                "count": int(values.size),
            }
            rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot motion data after interpolation to manager training inputs.")
    parser.add_argument("--dataset", action="append", type=_parse_dataset, default=None, help="NAME=PATH. Can be repeated.")
    parser.add_argument("--motion-id", action="append", type=int, default=None, help="Global/local motion id to plot. Can be repeated.")
    parser.add_argument("--step-dt", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-summary-motions", type=int, default=None)
    parser.add_argument("--out-prefix", type=Path, default=Path("docs/interpolated_motion_inputs"))
    args = parser.parse_args()

    datasets = args.dataset or _default_datasets()
    motion_ids = args.motion_id or [0, 25, 120, 300]
    providers = {dataset.name: _load_provider(dataset, args.num_envs, args.device) for dataset in datasets}

    samples: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    motion_names: dict[str, dict[int, str]] = {}
    for dataset_name, provider in providers.items():
        samples[dataset_name] = {}
        motion_names[dataset_name] = {}
        for motion_id in motion_ids:
            if motion_id < 0 or motion_id >= provider.motion_lib.num_motions():
                raise IndexError(f"{dataset_name}: motion id {motion_id} is out of range.")
            samples[dataset_name][motion_id] = _sample_motion(provider, motion_id, args.step_dt)
            motion_names[dataset_name][motion_id] = provider.motion_lib.curr_motion_keys[motion_id]

    summary = {
        dataset_name: _concat_metrics(provider, args.step_dt, args.max_summary_motions)
        for dataset_name, provider in providers.items()
    }

    curves_path = args.out_prefix.with_name(args.out_prefix.name + "_curves.png")
    dist_path = args.out_prefix.with_name(args.out_prefix.name + "_distribution.png")
    csv_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    _plot_motion_curves(samples, motion_names, motion_ids, curves_path)
    _plot_distribution(summary, dist_path)
    _write_summary_csv(summary, csv_path)
    print(f"Wrote {curves_path}")
    print(f"Wrote {dist_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
