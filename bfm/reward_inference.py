from __future__ import annotations

import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")

import functools

import joblib
import mujoco
import numpy as np
import torch
from torch.utils._pytree import tree_map

import bfm
from bfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim, get_idxs
from bfm.agents.buffers.transition import DictBuffer
from bfm.manager_envs.g1.spec import (
    BFMZERO_ROBOT_CONFIG,
    assert_model_matches_bfmzero_contract,
    load_bfmzero_g1_spec,
    resolve_repo_path,
)
from bfm.reward_tasks import RewardFunction, make_from_name
from bfm.tracking_inference_mujoco import (
    LightweightG1MujocoEnv,
    _checkpoint_load_device,
    _export_onnx,
    _load_model_from_checkpoint_dir,
)

if getattr(bfm, "__file__", None) is not None:
    BFM_DIR = Path(bfm.__file__).parent
else:
    BFM_DIR = Path(__file__).resolve().parent


DEFAULT_REWARD_TASKS = [
    # "move-ego-0-0",
    # "move-ego-low0.5-0-0",
    "move-ego-0-0.7",
    "move-ego-0-0.3",
    "move-ego-90-0.3",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "raisearms-l-l",
    "raisearms-l-m",
    "raisearms-m-l",
    "raisearms-m-m",
    "move-arms-0-0.7-m-m",
    "move-arms-90-0.7-m-m",
    "move-arms-180-0.4-m-m",
    "move-arms--90-0.7-m-m",
    "move-arms-0-0.7-l-m",
    "move-arms-90-0.7-l-m",
    "move-arms-180-0.4-l-m",
    "move-arms--90-0.7-l-m",
    "move-arms-0-0.7-m-l",
    "move-arms-90-0.7-m-l",
    "move-arms-180-0.4-m-l",
    "move-arms--90-0.7-m-l",
    "move-arms-0-0.7-l-l",
    "move-arms-90-0.7-l-l",
    "move-arms-180-0.4-l-l",
    "move-arms--90-0.7-l-l",
    "spin-arms-5-l-l",
    "spin-arms--5-l-l",
    "spin-arms-5-l-m",
    "spin-arms--5-l-m",
    "spin-arms-5-m-l",
    "spin-arms--5-m-l",
    "crouch-0",
    "crouch-0.25",
    "sitonground",
]


def _resolve_checkpoint(model_folder: Path, checkpoint_dir: Path | None) -> Path:
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else Path(model_folder) / "checkpoint"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    return checkpoint


def _resolve_robot_config(model_folder: Path, robot_config: str | None) -> str:
    if robot_config:
        return robot_config
    config_path = model_folder / "config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)
        configured = config.get("env", {}).get("robot_config")
        if configured:
            return str(configured)
    return BFMZERO_ROBOT_CONFIG


def _resolve_reward_xml(path: Path | None) -> Path:
    if path is None:
        path = BFM_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_noadditional_actuators.xml"
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = resolve_repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Reward MuJoCo XML does not exist: {path}")
    return path


def _get_next(field: str, data: dict[str, Any]) -> Any:
    if "next" in data and field in data["next"]:
        return data["next"][field]
    next_key = f"next_{field}"
    if next_key in data:
        return data[next_key]
    raise ValueError(f"No next value for {field!r} found in replay data.")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_device_tree(value: Any, device: torch.device | str) -> Any:
    return tree_map(lambda x: x.to(device=device) if isinstance(x, torch.Tensor) else torch.as_tensor(x, device=device), value)


def _load_buffer_honoring_device(path: Path, cls, *, device: str):
    with (path / "config.json").open() as f:
        loaded_config = json.load(f)
    loaded_config.pop("__target__", None)
    idx = loaded_config.pop("_idx", None)
    is_full = loaded_config.pop("_is_full", None)
    loaded_config["device"] = device
    buffer = cls(**loaded_config)
    buffer._idx = idx
    buffer._is_full = is_full
    buffer.load_hdf5(path / "buffer.hdf5")
    return buffer


def _load_replay_buffer(model_folder: Path, *, device: str = "cpu") -> DictBuffer | TrajectoryDictBufferMultiDim:
    reduced = model_folder / "checkpoint" / "buffers" / "train_reduced"
    if reduced.is_dir():
        print(f"Loading reduced replay buffer from {reduced}", flush=True)
        return _load_buffer_honoring_device(reduced, DictBuffer, device=device)
    original = model_folder / "checkpoint" / "buffers" / "train"
    if not original.is_dir():
        raise FileNotFoundError(f"No replay buffer found under {model_folder / 'checkpoint' / 'buffers'}")
    print(f"Loading train replay buffer from {original}", flush=True)
    buffer = _load_buffer_honoring_device(original, TrajectoryDictBufferMultiDim, device=device)
    buffer._get_idxs = get_idxs
    for key in ("qpos", "qvel"):
        if key not in buffer.output_key_tp1:
            buffer.output_key_tp1.append(key)
    return buffer


def _sample_replay(buffer: DictBuffer | TrajectoryDictBufferMultiDim, num_samples: int) -> dict[str, Any]:
    size = int(buffer.size())
    if num_samples >= size and hasattr(buffer, "get_full_buffer"):
        return buffer.get_full_buffer()
    return buffer.sample(num_samples)


def _relabel_worker(
    chunk: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    model: mujoco.MjModel,
    reward_fn: RewardFunction,
) -> np.ndarray:
    qpos, qvel, action = chunk
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos with shape [N, nq], got {qpos.shape}.")
    rewards_np = np.zeros((qpos.shape[0], 1), dtype=np.float32)
    for index in range(qpos.shape[0]):
        rewards_np[index, 0] = reward_fn(model, qpos[index], qvel[index], action[index])
    return rewards_np


def relabel_rewards(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    reward_fn: RewardFunction,
    *,
    max_workers: int = 1,
    executor: Literal["thread", "process"] = "thread",
    process_context: str = "spawn",
) -> np.ndarray:
    if qpos.shape[0] != qvel.shape[0] or qpos.shape[0] != action.shape[0]:
        raise ValueError(f"qpos/qvel/action batch mismatch: {qpos.shape}, {qvel.shape}, {action.shape}")
    if qpos.shape[0] == 0:
        raise ValueError("Cannot relabel an empty replay sample.")
    max_workers = max(1, min(int(max_workers), int(qpos.shape[0])))
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    chunks = [(qpos[i : i + chunk_size], qvel[i : i + chunk_size], action[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    worker = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
    if max_workers == 1:
        result = [worker(chunks[0])]
    elif executor == "process":
        import multiprocessing

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context(process_context)) as pool:
            result = list(pool.map(worker, chunks))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            result = list(pool.map(worker, chunks))
    return np.concatenate(result, axis=0)


@dataclasses.dataclass(kw_only=True)
class RewardInferenceEngine:
    model: Any
    replay_buffer: DictBuffer | TrajectoryDictBufferMultiDim
    reward_model: mujoco.MjModel
    num_samples: int
    device: str
    max_workers: int = 1
    executor: Literal["thread", "process"] = "thread"
    process_context: str = "spawn"

    def infer_task(self, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        data = _sample_replay(self.replay_buffer, self.num_samples)
        qpos = _to_numpy(_get_next("qpos", data)).reshape(-1, self.reward_model.nq)
        qvel = _to_numpy(_get_next("qvel", data)).reshape(-1, self.reward_model.nv)
        action = _to_numpy(data["action"]).reshape(-1, self.reward_model.nu)
        next_obs = _to_device_tree(_get_next("observation", data), self.device)
        reward_fn = make_from_name(task)
        rewards_np = relabel_rewards(
            self.reward_model,
            qpos,
            qvel,
            action,
            reward_fn,
            max_workers=self.max_workers,
            executor=self.executor,
            process_context=self.process_context,
        )
        reward = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)
        z = self.model.reward_wr_inference(next_obs=next_obs, reward=reward).reshape(1, -1)
        summary = {
            "samples": float(rewards_np.shape[0]),
            "reward_mean": float(np.mean(rewards_np)),
            "reward_std": float(np.std(rewards_np)),
            "reward_min": float(np.min(rewards_np)),
            "reward_max": float(np.max(rewards_np)),
            "z_norm": float(torch.linalg.norm(z).detach().cpu().item()),
        }
        return z, summary


def _rollout_reward_z(
    model,
    z: torch.Tensor,
    *,
    robot_config: str,
    mujoco_xml: Path | None,
    steps: int,
    device: str,
    headless: bool,
    real_time: bool,
    real_time_dt: float,
) -> dict[str, float]:
    spec = load_bfmzero_g1_spec(robot_config)
    env = LightweightG1MujocoEnv(
        spec=spec,
        device=device,
        headless=headless,
        show_reference=False,
        mujoco_xml=mujoco_xml,
    )
    try:
        default_qpos = env.model.keyframe("stand").qpos.copy()
        env.data.qpos[:] = default_qpos
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.last_action.zero_()
        env.history.reset()
        obs = env.observation(add_to_history=True)
        z = z.to(device=device).reshape(1, -1)
        action_abs_max = 0.0
        for step in range(int(steps)):
            step_start = time.perf_counter()
            action = model.act(obs, z, mean=True)
            if not torch.isfinite(action).all():
                raise FloatingPointError(f"Non-finite action at reward rollout step {step}.")
            action_abs_max = max(action_abs_max, float(action.abs().max().detach().cpu().item()))
            obs = env.step(action)
            if real_time and not headless:
                sleep_time = float(real_time_dt) - (time.perf_counter() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return {"rollout_steps": float(steps), "action_abs_max": action_abs_max}
    finally:
        env.close()


def main(
    model_folder: Path,
    checkpoint_dir: Path | None = None,
    tasks: list[str] | None = None,
    num_samples: int = 150_000,
    n_inferences: int = 1,
    device: str = "cpu",
    buffer_device: str = "cpu",
    reward_xml: Path | None = None,
    max_workers: int = 1,
    executor: Literal["thread", "process"] = "thread",
    process_context: str = "spawn",
    output_file: Path | None = None,
    skip_rollouts: bool = True,
    rollout_task_limit: int = 1,
    episode_length: int = 500,
    headless: bool = True,
    real_time: bool = True,
    real_time_dt: float = 0.02,
    robot_config: str | None = None,
    rollout_mujoco_xml: Path | None = None,
    export_onnx: bool = True,
) -> None:
    model_folder = Path(model_folder)
    checkpoint = _resolve_checkpoint(model_folder, checkpoint_dir)
    resolved_robot_config = _resolve_robot_config(model_folder, robot_config)
    reward_xml = _resolve_reward_xml(reward_xml)
    tasks = list(DEFAULT_REWARD_TASKS if tasks is None else tasks)
    output_dir = model_folder / "reward_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_file) if output_file is not None else output_dir / "reward_locomotion.pkl"

    model = _load_model_from_checkpoint_dir(checkpoint, device=_checkpoint_load_device(device))
    model.to(device)
    model.eval()
    assert_model_matches_bfmzero_contract(model)

    if export_onnx:
        history = "history_actor" in model.cfg.archi.actor.input_filter.key
        onnx_path = _export_onnx(model, model_folder / "exported", history=history)
        print(f"Exported actor ONNX to {onnx_path}", flush=True)

    start_t = time.time()
    replay_buffer = _load_replay_buffer(model_folder, device=buffer_device)
    print(f"Loaded replay buffer in {time.time() - start_t:.2f}s; size={replay_buffer.size()}", flush=True)
    reward_model = mujoco.MjModel.from_xml_path(str(reward_xml))
    print(
        f"Reward inference config model_folder={model_folder} checkpoint={checkpoint} reward_xml={reward_xml} "
        f"device={device} buffer_device={buffer_device} num_samples={num_samples} tasks={len(tasks)}",
        flush=True,
    )

    engine = RewardInferenceEngine(
        model=model,
        replay_buffer=replay_buffer,
        reward_model=reward_model,
        num_samples=int(num_samples),
        device=device,
        max_workers=int(max_workers),
        executor=executor,
        process_context=process_context,
    )
    z_dict: dict[str, list[torch.Tensor]] = {}
    summary: dict[str, Any] = {
        "model_folder": str(model_folder),
        "checkpoint": str(checkpoint),
        "reward_xml": str(reward_xml),
        "num_samples": int(num_samples),
        "n_inferences": int(n_inferences),
        "tasks": {},
    }

    for inference_idx in range(int(n_inferences)):
        for task in tasks:
            print(f"Reward inference {inference_idx + 1}/{n_inferences} task={task}...", end=" ", flush=True)
            task_start = time.time()
            z, task_summary = engine.infer_task(task)
            z_cpu = z.detach().cpu()
            z_dict.setdefault(task, []).append(z_cpu)
            task_summary["seconds"] = float(time.time() - task_start)
            summary["tasks"].setdefault(task, []).append(task_summary)
            with output_file.open("wb") as f:
                joblib.dump(z_dict, f)
            with (output_dir / "summary.json").open("w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            print(f"done in {task_summary['seconds']:.2f}s reward_mean={task_summary['reward_mean']:.4g} z_norm={task_summary['z_norm']:.4g}", flush=True)

    if not skip_rollouts:
        rollout_tasks = tasks[: max(0, int(rollout_task_limit))]
        rollout_summary: dict[str, Any] = {}
        print(f"Running lightweight MuJoCo rollouts for {len(rollout_tasks)} reward z vectors.", flush=True)
        for task in rollout_tasks:
            rollout_summary[task] = []
            for z in z_dict[task]:
                metrics = _rollout_reward_z(
                    model,
                    z,
                    robot_config=resolved_robot_config,
                    mujoco_xml=rollout_mujoco_xml,
                    steps=int(episode_length),
                    device=device,
                    headless=headless,
                    real_time=real_time,
                    real_time_dt=real_time_dt,
                )
                rollout_summary[task].append(metrics)
        summary["rollouts"] = rollout_summary
        with (output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Saved reward z dict to {output_file}", flush=True)
    print(f"Saved reward inference summary to {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
