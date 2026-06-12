from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

import bfm
from bfm.agents.fb.model import FBModel
from bfm.agents.fb_cpr.model import FBcprModel
from bfm.agents.fb_cpr_aux.model import FBcprAuxModel
from bfm.manager_envs.g1.spec import BFMZERO_ROBOT_CONFIG

if getattr(bfm, "__file__", None) is not None:
    BFM_DIR = Path(bfm.__file__).parent
else:
    BFM_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_FOLDER = Path("/home/thl/wt_wbc/BFM-Zero/results/results/bfmzero-isaac1")
MODEL_NAME_TO_CLASS = {
    "FBModel": FBModel,
    "FBcprModel": FBcprModel,
    "FBcprAuxModel": FBcprAuxModel,
}


def _resolve_checkpoint(model_folder: Path, checkpoint_dir: Path | None) -> Path:
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else Path(model_folder) / "checkpoint"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    return checkpoint


def _resolve_motion_file(model_folder: Path, data_path: Path | None) -> Path:
    if data_path is not None:
        return Path(data_path).expanduser().resolve()
    config_path = Path(model_folder) / "config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)
        configured = config.get("env", {}).get("lafan_tail_path")
        if configured:
            path = Path(configured)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                return path.resolve()
    return (BFM_DIR / "data" / "lafan_29dof_10s-clipped.pkl").resolve()


def _resolve_robot_config(model_folder: Path, robot_config: str | None) -> str:
    if robot_config:
        return robot_config
    config_path = Path(model_folder) / "config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)
        configured = config.get("env", {}).get("robot_config")
        if configured:
            return str(configured)
    return BFMZERO_ROBOT_CONFIG


def _checkpoint_load_device(device: str) -> str:
    device_type = torch.device(device).type
    if device_type == "cuda":
        return "cuda"
    if device_type == "cpu":
        return "cpu"
    raise ValueError(f"BFM-Zero checkpoint loader only supports cpu/cuda devices, got {device!r}.")


def _load_model_from_checkpoint_dir(checkpoint_dir: Path, device: str):
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "model" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint model config not found: {config_path}")
    with config_path.open("r") as f:
        config = json.load(f)

    model_name = config["name"]
    if model_name not in MODEL_NAME_TO_CLASS:
        raise ValueError(f"Unknown checkpoint model name: {model_name}. Available: {list(MODEL_NAME_TO_CLASS)}")
    return MODEL_NAME_TO_CLASS[model_name].load(checkpoint_dir / "model", device=device)


def _load_optional_video_tools():
    raise RuntimeError(
        "--save-mp4 is not available in the manager-only repo. "
        "Run without --save-mp4, or use the full BFM-Zero repo if you need the legacy MuJoCo comparison renderer."
    )


def _tracking_z(model, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    z = model.backward_map(obs)
    for step in range(z.shape[0]):
        end_idx = min(step + 1, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return model.project_z(z)


def _assert_finite_obs(obs: dict[str, torch.Tensor], *, context: str) -> None:
    bad = [key for key, value in obs.items() if torch.is_floating_point(value) and not torch.isfinite(value).all()]
    if bad:
        raise FloatingPointError(f"Non-finite observation tensors in {context}: {bad}")


def _float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def main(
    model_folder: Path = DEFAULT_MODEL_FOLDER,
    checkpoint_dir: Path | None = None,
    data_path: Path | None = None,
    robot_config: str | None = None,
    motion_list: list[int] = [25],
    steps: int = 100,
    headless: bool = True,
    device: str = "cuda:0",
    save_mp4: bool = False,
    output_dir: Path | None = None,
):
    model_folder = Path(model_folder)
    checkpoint = _resolve_checkpoint(model_folder, checkpoint_dir)
    motion_file = _resolve_motion_file(model_folder, data_path)
    resolved_robot_config = _resolve_robot_config(model_folder, robot_config)
    output_dir = Path(output_dir) if output_dir is not None else model_folder / "tracking_inference_manager"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Manager inference config "
        f"model_folder={model_folder} checkpoint={checkpoint} motion_file={motion_file} "
        f"robot_config={resolved_robot_config} device={device} headless={headless}",
        flush=True,
    )

    model = _load_model_from_checkpoint_dir(checkpoint, device=_checkpoint_load_device(device))
    model.to(device)
    model.eval()
    print(
        f"Loaded checkpoint model={model.__class__.__name__} action_dim={model.action_dim} z_dim={model.cfg.archi.z_dim}",
        flush=True,
    )

    from bfm.manager_envs.g1.isaac_app import instantiate_isaac_sim

    instantiate_isaac_sim(num_envs=1, enable_cameras=False, headless=headless)

    from bfm.manager_envs.g1.adapter import BFMZeroManagerBuildConfig, BFMZeroManagerVectorEnvAdapter
    from bfm.manager_envs.g1.spec import assert_model_matches_bfmzero_contract

    adapter = BFMZeroManagerVectorEnvAdapter.build(
        BFMZeroManagerBuildConfig(
            num_envs=1,
            device=device,
            motion_file=motion_file,
            robot_config=resolved_robot_config,
            default_motion_id=int(motion_list[0]),
            episode_length_s=max(float(steps) * 0.02 + 1.0, 10.0),
            render_mode=None,
        )
    )

    try:
        assert_model_matches_bfmzero_contract(model)
        print(f"Manager inference robot_config={resolved_robot_config}", flush=True)
        all_metrics = {}
        for motion_id in motion_list:
            print(f"Preparing manager inference motion={motion_id}", flush=True)
            backward_obs, ref_dict = adapter.get_backward_observation(int(motion_id))
            z = _tracking_z(model, tree_map(lambda x: x[1:].to(device), backward_obs))
            joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")

            observation, reset_info = adapter.reset_to_motion(int(motion_id), frame_id=0)
            adapter.assert_checkpoint_contract(model, observation)
            _assert_finite_obs(observation, context=f"reset motion={motion_id}")
            adapter_obs_shapes = {key: list(value.shape[1:]) for key, value in observation.items()}
            print(f"Adapter checkpoint obs shapes motion={motion_id}: {adapter_obs_shapes}", flush=True)

            reset_joint_error = torch.max(torch.abs(reset_info["joint_pos_abs"] - ref_dict["dof_pos"][0:1])).detach()
            metrics = {
                "motion_id": int(motion_id),
                "requested_steps": int(steps),
                "z_steps": int(z.shape[0]),
                "adapter_obs_shapes": adapter_obs_shapes,
                "reset_max_abs_joint_error": _float(reset_joint_error),
                "completed_steps": 0,
                "terminated_steps": 0,
                "truncated_steps": 0,
                "action_abs_max": 0.0,
                "joint_abs_error_mean": [],
                "body_pos_error_mean": [],
                "body_pos_error_max": [],
            }

            frames = []
            expert_video = None
            rgb_renderer = None
            rollout_steps = min(int(steps), int(z.shape[0]))
            if save_mp4:
                IsaacRendererWithMuJoco, write_video_or_frames = _load_optional_video_tools()
                expert_qpos = np.concatenate(
                    [
                        ref_dict["ref_body_pos"][:, 0].detach().cpu().numpy(),
                        np.roll(ref_dict["ref_body_rots"][:, 0].detach().cpu().numpy(), 1, axis=-1),
                        ref_dict["dof_pos"].detach().cpu().numpy(),
                    ],
                    axis=-1,
                )
                rgb_renderer = IsaacRendererWithMuJoco(render_size=256)
                expert_video = rgb_renderer.from_qpos(expert_qpos[: 1 + rollout_steps])
                frames.append(rgb_renderer.render_qpos(adapter.mujoco_qpos().numpy()[0]))

            with torch.no_grad():
                for step in range(rollout_steps):
                    _assert_finite_obs(observation, context=f"rollout motion={motion_id} step={step}")
                    action = model.act(observation, z[step].reshape(1, -1).repeat(adapter.num_envs, 1), mean=True)
                    if action.shape != (adapter.num_envs, adapter.num_actions):
                        raise AssertionError(f"Action shape mismatch at step {step}: {tuple(action.shape)}")
                    if not torch.isfinite(action).all():
                        raise FloatingPointError(f"Non-finite action at step {step}.")
                    metrics["action_abs_max"] = max(metrics["action_abs_max"], _float(action.abs().max()))
                    observation, reward, terminated, truncated, extras = adapter.step(action)
                    target_idx = min(step + 1, ref_dict["dof_pos"].shape[0] - 1)
                    joint_error = torch.mean(torch.abs(extras["joint_pos_abs"] - ref_dict["dof_pos"][target_idx : target_idx + 1]))
                    metrics["joint_abs_error_mean"].append(_float(joint_error))
                    command_metrics = adapter.motion_command.metrics
                    if "error_body_pos" in command_metrics:
                        metrics["body_pos_error_mean"].append(_float(command_metrics["error_body_pos"]))
                    if "max_error_body_pos" in command_metrics:
                        metrics["body_pos_error_max"].append(_float(command_metrics["max_error_body_pos"]))
                    metrics["terminated_steps"] += int(terminated.sum().item())
                    metrics["truncated_steps"] += int(truncated.sum().item())
                    metrics["completed_steps"] += 1
                    if save_mp4:
                        frames.append(rgb_renderer.render_qpos(adapter.mujoco_qpos().numpy()[0]))
                    if bool(torch.logical_or(terminated, truncated).any().item()):
                        break

            for key in ["joint_abs_error_mean", "body_pos_error_mean", "body_pos_error_max"]:
                values = metrics[key]
                metrics[key] = float(np.mean(values)) if values else 0.0

            if save_mp4 and frames:
                video_path = output_dir / f"tracking_manager_{motion_id}.mp4"
                compare_frames = [np.concatenate([a, b], axis=1) for a, b in zip(expert_video, frames)]
                video_result = write_video_or_frames(video_path, compare_frames, fps=int(round(1.0 / adapter.step_dt)))
                metrics["video"] = video_result
                metrics["video_path"] = video_result["path"]

            all_metrics[str(motion_id)] = metrics
            metrics_path = output_dir / f"metrics_{motion_id}.json"
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            print(f"Manager inference motion={motion_id} metrics={metrics}", flush=True)

        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(all_metrics, f, indent=2, sort_keys=True)
        print(f"Saved manager inference summary to {summary_path}", flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
