from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import mujoco
import mujoco.viewer
import numpy as np
import torch
from torch import nn
from torch.utils._pytree import tree_map

import bfm
from bfm.agents.fb.model import FBModel
from bfm.agents.fb_cpr.model import FBcprModel
from bfm.agents.fb_cpr_aux.model import FBcprAuxModel
from bfm.manager_envs.g1.motion_provider import BFMZeroMotionProvider
from bfm.manager_envs.g1.observations import compute_humanoid_observations_max
from bfm.manager_envs.g1.spec import (
    BFMZERO_BASE_ANG_VEL_OBS_SCALE,
    BFMZERO_DEFAULT_MOTION_FILE,
    BFMZERO_ROBOT_CONFIG,
    BFMZeroG1Spec,
    assert_model_matches_bfmzero_contract,
    load_bfmzero_g1_spec,
    resolve_repo_path,
)
from bfm.utils.torch_utils import quat_rotate_inverse, wxyz_to_xyzw

MODEL_NAME_TO_CLASS = {
    "FBModel": FBModel,
    "FBcprModel": FBcprModel,
    "FBcprAuxModel": FBcprAuxModel,
}

if getattr(bfm, "__file__", None) is not None:
    BFM_DIR = Path(bfm.__file__).parent
else:
    BFM_DIR = Path(__file__).resolve().parent


class _ActorONNXWrapper(nn.Module):
    def __init__(self, model, *, z_dim: int, history: bool) -> None:
        super().__init__()
        self.model = model
        self.z_dim = int(z_dim)
        self.history = bool(history)

    def forward(self, actor_obs):
        obs_part, z = actor_obs[:, : -self.z_dim], actor_obs[:, -self.z_dim :]
        state_end = 64
        action_end = state_end + 29
        obs = {
            "state": obs_part[:, :state_end],
            "last_action": obs_part[:, state_end:action_end],
        }
        if self.history:
            obs["history_actor"] = obs_part[:, action_end:]
        return self.model.act(obs, z)


class BFMZeroHistory:
    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.config = {
            "base_ang_vel": 4,
            "projected_gravity": 4,
            "dof_pos": 4,
            "dof_vel": 4,
            "actions": 4,
        }
        self.order = tuple(sorted(self.config))
        self.dims = {
            "actions": 29,
            "base_ang_vel": 3,
            "dof_pos": 29,
            "dof_vel": 29,
            "projected_gravity": 3,
        }
        self.history = {
            key: torch.zeros(self.num_envs, length, self.dims[key], dtype=torch.float32, device=self.device)
            for key, length in self.config.items()
        }

    def reset(self) -> None:
        for value in self.history.values():
            value.zero_()

    def add(self, terms: dict[str, torch.Tensor]) -> None:
        for key, value in terms.items():
            if key not in self.history:
                continue
            old = self.history[key].clone()
            self.history[key][:, 1:] = old[:, :-1]
            self.history[key][:, 0] = value

    def query_actor(self) -> torch.Tensor:
        values = []
        for key in self.order:
            length = self.config[key]
            values.append(self.history[key][:, :length].reshape(self.num_envs, -1))
        return torch.cat(values, dim=-1)


class LightweightG1MujocoEnv:
    def __init__(
        self,
        *,
        spec: BFMZeroG1Spec,
        device: str = "cpu",
        headless: bool = True,
        show_reference: bool = True,
        mujoco_xml: Path | None = None,
        physics_fps: float = 200.0,
        control_decimation: int = 4,
    ):
        self.spec = spec
        self.device = torch.device(device)
        self.mujoco_xml = Path(mujoco_xml) if mujoco_xml is not None else self._default_mujoco_xml(spec)
        self.model = mujoco.MjModel.from_xml_path(str(self.mujoco_xml))
        self.model.opt.timestep = 1.0 / float(physics_fps)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.show_reference = bool(show_reference)
        self.reference_marker_pos: np.ndarray | None = None
        self.reference_marker_size = 0.035

        self.dt = float(self.model.opt.timestep)
        self.decimation = max(1, int(control_decimation))
        self.control_dt = self.dt * self.decimation
        self.default_joint_pos = torch.tensor(spec.default_joint_pos, dtype=torch.float32, device=self.device).reshape(1, -1)
        self.p_gains = torch.tensor([spec.p_gain_for_joint(name) for name in spec.dof_names], dtype=torch.float32, device=self.device)
        self.d_gains = torch.tensor([spec.d_gain_for_joint(name) for name in spec.dof_names], dtype=torch.float32, device=self.device)
        self.effort_limits = torch.tensor(spec.effort_limits, dtype=torch.float32, device=self.device)
        self.action_rescale = torch.tensor(
            [float(spec.effort_limits[index]) / spec.p_gain_for_joint(name) for index, name in enumerate(spec.dof_names)],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        self.torque_limits = self.effort_limits.clone()
        self.last_action = torch.zeros(1, spec.num_actions, dtype=torch.float32, device=self.device)
        self.history = BFMZeroHistory(1, self.device)
        self.body_ids: np.ndarray | None = None
        self.ctrl_offset = 0
        self.last_step_debug: dict[str, np.ndarray] = {}

        self._assert_order_contract()
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    @staticmethod
    def _default_mujoco_xml(spec: BFMZeroG1Spec) -> Path:
        full_repo_matching_xml = spec.bfm_dir / "data" / "robots" / "g1" / "scene_29dof_freebase_mujoco.xml"
        if full_repo_matching_xml.is_file():
            return full_repo_matching_xml
        return spec.mjcf_path

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _assert_order_contract(self) -> None:
        joint_names = tuple(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, self.model.njnt))
        all_body_names = tuple(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, self.model.nbody))
        body_ids = []
        body_names = []
        for body_id, body_name in enumerate(all_body_names, start=1):
            if "hand" in body_name:
                continue
            body_ids.append(body_id)
            body_names.append(body_name)
        body_names = tuple(body_names)
        if joint_names != self.spec.dof_names:
            raise AssertionError(f"MuJoCo joint order mismatch: {joint_names} != {self.spec.dof_names}")
        if body_names != self.spec.body_names:
            raise AssertionError(f"MuJoCo body order mismatch: {body_names} != {self.spec.body_names}")
        self.body_ids = np.asarray(body_ids, dtype=np.int32)
        self.ctrl_offset = int(self.model.nu - self.spec.num_actions)
        if self.ctrl_offset < 0:
            raise AssertionError(f"MuJoCo model has only {self.model.nu} actuators for {self.spec.num_actions} actions.")
        actuator_joint_names = []
        for actuator_id in range(self.ctrl_offset, self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            actuator_joint_names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        actuator_joint_names = tuple(actuator_joint_names)
        if actuator_joint_names != self.spec.dof_names:
            raise AssertionError(f"MuJoCo actuator joint order mismatch: {actuator_joint_names} != {self.spec.dof_names}")

    def reset_to_reference(self, ref_dict: dict[str, torch.Tensor], frame_id: int = 0) -> dict[str, torch.Tensor]:
        frame_id = int(frame_id)
        root_pos = ref_dict["ref_body_pos"][frame_id, 0].detach().cpu().numpy()
        root_quat_xyzw = ref_dict["ref_body_rots"][frame_id, 0].detach().cpu().numpy()
        dof_pos = ref_dict["dof_pos"][frame_id].detach().cpu().numpy()
        root_lin_vel = ref_dict["ref_body_vels"][frame_id, 0].detach().cpu().numpy()
        root_ang_vel_world = ref_dict["ref_body_angular_vels"][frame_id : frame_id + 1, 0].to(device=self.device)
        root_quat_xyzw_t = ref_dict["ref_body_rots"][frame_id : frame_id + 1, 0].to(device=self.device)
        root_ang_vel = quat_rotate_inverse(root_quat_xyzw_t, root_ang_vel_world, w_last=True).detach().cpu().numpy().reshape(3)
        dof_vel = ref_dict["ref_dof_vel"][frame_id].detach().cpu().numpy()

        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]]
        self.data.qpos[7:] = dof_pos
        self.data.qvel[:3] = root_lin_vel
        self.data.qvel[3:6] = root_ang_vel
        self.data.qvel[6:] = dof_vel
        mujoco.mj_forward(self.model, self.data)

        self.last_action.zero_()
        self.history.reset()
        obs = self.observation(add_to_history=True)
        self._set_reference_markers(ref_dict["ref_body_pos"][frame_id].detach().cpu().numpy())
        self._sync_viewer()
        self.last_step_debug = {
            "reset_qpos": self.data.qpos.copy(),
            "reset_qvel": self.data.qvel.copy(),
            "reset_root_ang_vel_world": root_ang_vel_world.detach().cpu().numpy(),
            "reset_root_ang_vel_local": root_ang_vel.copy(),
        }
        return obs

    def _compute_terms(self) -> dict[str, torch.Tensor]:
        if self.body_ids is None:
            raise RuntimeError("MuJoCo body ids were not initialized.")
        root_quat_xyzw = torch.tensor(self.data.qpos[3:7][[1, 2, 3, 0]], dtype=torch.float32, device=self.device).reshape(1, 4)
        gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=self.device)
        projected_gravity = quat_rotate_inverse(root_quat_xyzw, gravity, w_last=True)
        base_ang_vel = torch.tensor(self.data.qvel[3:6], dtype=torch.float32, device=self.device).reshape(1, 3)
        base_ang_vel = base_ang_vel * BFMZERO_BASE_ANG_VEL_OBS_SCALE

        joint_pos = torch.tensor(self.data.qpos[7:], dtype=torch.float32, device=self.device).reshape(1, -1)
        joint_vel = torch.tensor(self.data.qvel[6:], dtype=torch.float32, device=self.device).reshape(1, -1)
        dof_pos = joint_pos - self.default_joint_pos

        body_pos = torch.tensor(self.data.xpos[self.body_ids], dtype=torch.float32, device=self.device).reshape(1, -1, 3)
        body_quat_xyzw = wxyz_to_xyzw(torch.tensor(self.data.xquat[self.body_ids], dtype=torch.float32, device=self.device)).reshape(1, -1, 4)
        body_vel = torch.tensor(self.data.cvel[self.body_ids, 3:6], dtype=torch.float32, device=self.device).reshape(1, -1, 3)
        body_ang_vel = torch.tensor(self.data.cvel[self.body_ids, 0:3], dtype=torch.float32, device=self.device).reshape(1, -1, 3)

        max_local_self_dict = compute_humanoid_observations_max(
            body_pos,
            body_quat_xyzw,
            body_vel,
            body_ang_vel,
            local_root_obs=True,
            root_height_obs=True,
        )
        privileged_state = torch.cat([value for value in max_local_self_dict.values()], dim=-1)
        action_obs = self.last_action.clone()
        terms = {
            "state": torch.cat([dof_pos, joint_vel, projected_gravity, base_ang_vel], dim=-1),
            "privileged_state": privileged_state,
            "last_action": action_obs,
            "history_actor": self.history.query_actor(),
            "actions": action_obs,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "dof_pos": dof_pos,
            "dof_vel": joint_vel,
        }
        for key, value in terms.items():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite MuJoCo observation term: {key}")
        return terms

    def observation(self, *, add_to_history: bool = False) -> dict[str, torch.Tensor]:
        terms = self._compute_terms()
        observation = {key: terms[key] for key in ("state", "privileged_state", "last_action", "history_actor")}
        if add_to_history:
            self.history.add(
                {
                    "base_ang_vel": terms["base_ang_vel"],
                    "projected_gravity": terms["projected_gravity"],
                    "dof_pos": terms["dof_pos"],
                    "dof_vel": terms["dof_vel"],
                    "actions": terms["actions"],
                }
            )
        return observation

    def _process_action(self, action: torch.Tensor) -> torch.Tensor:
        processed = action * self.spec.action_obs_scale
        return torch.clamp(processed, -self.spec.action_clip_value, self.spec.action_clip_value)

    def _target_pos_from_processed_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        actions_scaled = processed_action * self.spec.action_scale
        if self.spec.action_rescale:
            actions_scaled = actions_scaled * self.action_rescale
        return self.default_joint_pos + actions_scaled

    def step(self, action: torch.Tensor, *, reference_pos: np.ndarray | None = None) -> dict[str, torch.Tensor]:
        action = action.detach().to(device=self.device, dtype=torch.float32)
        if action.shape != (1, self.spec.num_actions):
            raise AssertionError(f"Action shape mismatch: {tuple(action.shape)} != {(1, self.spec.num_actions)}")
        processed_action = self._process_action(action)
        self.last_action = processed_action.clone()
        target_pos = self._target_pos_from_processed_action(processed_action)

        torque = torch.zeros_like(target_pos)
        for _ in range(self.decimation):
            qpos = torch.tensor(self.data.qpos[7:], dtype=torch.float32, device=self.device).reshape(1, -1)
            qvel = torch.tensor(self.data.qvel[6:], dtype=torch.float32, device=self.device).reshape(1, -1)
            torque = self.p_gains * (target_pos - qpos) - self.d_gains * qvel
            torque = torch.clamp(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[self.ctrl_offset :] = torque.detach().cpu().numpy().reshape(-1)
            mujoco.mj_step(self.model, self.data)

        obs = self.observation(add_to_history=True)
        if reference_pos is not None:
            self._set_reference_markers(reference_pos)
        self._sync_viewer()
        self.last_step_debug = {
            "raw_action": action.detach().cpu().numpy(),
            "processed_action": processed_action.detach().cpu().numpy(),
            "target_pos": target_pos.detach().cpu().numpy(),
            "torque": torque.detach().cpu().numpy(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "state": obs["state"].detach().cpu().numpy(),
            "last_action": obs["last_action"].detach().cpu().numpy(),
            "history_actor": obs["history_actor"].detach().cpu().numpy(),
        }
        return obs

    def _set_reference_markers(self, positions: np.ndarray | None, size: float = 0.035) -> None:
        if not self.show_reference or positions is None:
            self.reference_marker_pos = None
            return
        self.reference_marker_pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.reference_marker_size = float(size)

    def _sync_viewer(self) -> None:
        if self.viewer is None:
            return
        if self.reference_marker_pos is not None:
            with self.viewer.lock():
                scene = self.viewer.user_scn
                scene.ngeom = 0
                marker_size = np.array([self.reference_marker_size] * 3, dtype=np.float64)
                marker_mat = np.eye(3, dtype=np.float64).reshape(-1)
                for pos in self.reference_marker_pos[: len(scene.geoms)]:
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        marker_size,
                        pos,
                        marker_mat,
                        np.array([0.1, 0.65, 1.0, 0.65], dtype=np.float32),
                    )
                    scene.ngeom += 1
        self.viewer.sync()


def _resolve_checkpoint(model_folder: Path, checkpoint_dir: Path | None) -> Path:
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else Path(model_folder) / "checkpoint"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    return checkpoint


def _load_model_from_checkpoint_dir(checkpoint_dir: Path, device: str):
    config_path = checkpoint_dir / "model" / "config.json"
    with config_path.open("r") as f:
        config = json.load(f)
    model_name = config["name"]
    if model_name not in MODEL_NAME_TO_CLASS:
        raise ValueError(f"Unknown checkpoint model name: {model_name}. Available: {list(MODEL_NAME_TO_CLASS)}")
    return MODEL_NAME_TO_CLASS[model_name].load(checkpoint_dir / "model", device=device)


def _checkpoint_load_device(device: str) -> str:
    device_type = torch.device(device).type
    if device_type == "cuda":
        return "cuda"
    if device_type == "cpu":
        return "cpu"
    raise ValueError(f"Checkpoint loader only supports cpu/cuda devices, got {device!r}.")


def _resolve_motion_file(model_folder: Path, data_path: Path | None) -> Path:
    if data_path is not None:
        return Path(data_path).expanduser().resolve()
    config_path = model_folder / "config.json"
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)
        configured = config.get("env", {}).get("lafan_tail_path")
        if configured:
            path = resolve_repo_path(configured)
            if path.exists():
                return path
    return resolve_repo_path(BFMZERO_DEFAULT_MOTION_FILE)


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


def _tracking_z(model, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    z = model.backward_map(obs)
    for step in range(z.shape[0]):
        end_idx = min(step + 1, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return model.project_z(z)


def _actor_obs_for_onnx(observation: dict[str, torch.Tensor], z: torch.Tensor, *, history: bool) -> np.ndarray:
    if z.ndim == 1:
        z = z.reshape(1, -1)
    parts = [observation["state"], observation["last_action"]]
    if history:
        parts.append(observation["history_actor"])
    parts.append(z)
    return torch.cat(parts, dim=-1).detach().cpu().numpy().astype(np.float32)


def _export_onnx(model, output_dir: Path, *, history: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model.__class__.__name__}.onnx"
    actor_input_dim = model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim
    wrapper = _ActorONNXWrapper(model, z_dim=model.cfg.archi.z_dim, history=history).eval().cpu()
    torch.onnx.export(
        wrapper,
        torch.randn(1, actor_input_dim, dtype=torch.float32),
        onnx_path,
        verbose=False,
        input_names=["actor_obs"],
        output_names=["action"],
        opset_version=13,
    )
    return onnx_path


def _make_onnx_session(onnx_path: Path, device: str):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if torch.device(device).type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"ONNX Runtime providers: {session.get_providers()}", flush=True)
    return session


def _act_with_onnx(session, observation: dict[str, torch.Tensor], z: torch.Tensor, *, device: str, history: bool) -> torch.Tensor:
    actor_obs = _actor_obs_for_onnx(observation, z, history=history)
    action = session.run(["action"], {"actor_obs": actor_obs})[0]
    return torch.as_tensor(action, dtype=torch.float32, device=device)


def _resolve_mujoco_xml(spec: BFMZeroG1Spec, mujoco_xml: Path | None) -> Path | None:
    if mujoco_xml is None:
        return None
    path = Path(mujoco_xml).expanduser()
    if not path.is_absolute():
        path = resolve_repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo XML does not exist: {path}")
    return path


def _append_trace(trace: dict[str, list[np.ndarray]], prefix: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        trace.setdefault(f"{prefix}_{key}", []).append(array.copy())


def _save_trace(path: Path, trace: dict[str, list[np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = {}
    for key, values in trace.items():
        try:
            packed[key] = np.stack(values)
        except ValueError:
            packed[key] = np.asarray(values, dtype=object)
    np.savez_compressed(path, **packed)


def main(
    model_folder: Path,
    checkpoint_dir: Path | None = None,
    data_path: Path | None = None,
    robot_config: str | None = None,
    mujoco_xml: Path | None = None,
    motion_list: list[int] = [25],
    steps: int = 200,
    device: str = "cpu",
    headless: bool = True,
    policy_runtime: Literal["torch", "onnx"] = "onnx",
    check_onnx_parity: bool = True,
    onnx_parity_atol: float = 1.0e-3,
    show_reference: bool = True,
    real_time: bool = True,
    real_time_dt: float = 0.02,
    progress_every: int = 50,
    full_repo_warmup_step: bool = True,
    debug_trace_path: Path | None = None,
    debug_trace_steps: int = 5,
) -> None:
    model_folder = Path(model_folder)
    checkpoint = _resolve_checkpoint(model_folder, checkpoint_dir)
    motion_file = _resolve_motion_file(model_folder, data_path)
    resolved_robot_config = _resolve_robot_config(model_folder, robot_config)
    spec = load_bfmzero_g1_spec(resolved_robot_config)
    resolved_mujoco_xml = _resolve_mujoco_xml(spec, mujoco_xml)
    provider = BFMZeroMotionProvider(motion_file=motion_file, spec=spec, num_envs=1, device=device)

    model = _load_model_from_checkpoint_dir(checkpoint, device=_checkpoint_load_device(device))
    model.to(device)
    model.eval()
    assert_model_matches_bfmzero_contract(model)
    history = "history_actor" in model.cfg.archi.actor.input_filter.key

    output_dir = model_folder / "tracking_inference_mujoco"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_session = None
    onnx_path = None
    if policy_runtime == "onnx":
        onnx_path = _export_onnx(model, model_folder / "exported", history=history)
        print(f"Exported model to {onnx_path}", flush=True)
        onnx_session = _make_onnx_session(onnx_path, device)

    print(
        "Slim MuJoCo inference config "
        f"model_folder={model_folder} checkpoint={checkpoint} motion_file={motion_file} "
        f"robot_config={resolved_robot_config} device={device} runtime={policy_runtime} headless={headless}",
        flush=True,
    )

    env = LightweightG1MujocoEnv(
        spec=spec,
        device=device,
        headless=headless,
        show_reference=show_reference,
        mujoco_xml=resolved_mujoco_xml,
    )
    print(
        f"Slim MuJoCo env xml={env.mujoco_xml} physics_dt={env.dt:.6f} control_dt={env.control_dt:.6f} "
        f"decimation={env.decimation} ctrl_offset={env.ctrl_offset}",
        flush=True,
    )
    try:
        summary = {}
        for motion_id in motion_list:
            motion_id = int(motion_id)
            backward_obs, ref_dict = provider.reference_backward_observation(
                motion_id,
                step_dt=env.control_dt,
                use_root_height_obs=True,
            )
            z = _tracking_z(model, tree_map(lambda x: x[1:].to(device), backward_obs))
            joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")

            observation = env.reset_to_reference(ref_dict, frame_id=0)
            trace: dict[str, list[np.ndarray]] = {}
            if debug_trace_path is not None:
                _append_trace(trace, "reset", env.last_step_debug)
                _append_trace(trace, "obs", observation)
            if full_repo_warmup_step:
                observation = env.step(
                    torch.zeros((1, env.spec.num_actions), dtype=torch.float32, device=device),
                    reference_pos=ref_dict["ref_body_pos"][0].detach().cpu().numpy(),
                )
                if debug_trace_path is not None:
                    _append_trace(trace, "warmup", env.last_step_debug)
                    _append_trace(trace, "obs", observation)
            assert_model_matches_bfmzero_contract(model, observation)
            rollout_steps = min(int(steps), int(z.shape[0]))
            metrics = {
                "motion_id": motion_id,
                "requested_steps": int(steps),
                "completed_steps": 0,
                "z_steps": int(z.shape[0]),
                "policy_runtime": policy_runtime,
                "onnx_path": str(onnx_path) if onnx_path else "",
                "action_abs_max": 0.0,
                "joint_abs_error_mean": [],
            }
            onnx_parity_checked = False
            print(f"Running slim MuJoCo inference motion={motion_id} steps={rollout_steps}", flush=True)
            for step in range(rollout_steps):
                step_start = time.perf_counter()
                z_step = z[step].reshape(1, -1)
                if policy_runtime == "onnx":
                    action = _act_with_onnx(onnx_session, observation, z_step, device=device, history=history)
                    if check_onnx_parity and not onnx_parity_checked:
                        torch_action = model.act(observation, z_step, mean=True)
                        max_diff = float((torch_action - action).abs().max().detach().cpu().item())
                        print(f"ONNX/Torch action max diff: {max_diff:.6g}", flush=True)
                        if max_diff > float(onnx_parity_atol):
                            raise AssertionError(f"ONNX/Torch action mismatch {max_diff:.6g} > {onnx_parity_atol:.6g}.")
                        onnx_parity_checked = True
                else:
                    action = model.act(observation, z_step, mean=True)

                if not torch.isfinite(action).all():
                    raise FloatingPointError(f"Non-finite action at step {step}.")
                metrics["action_abs_max"] = max(metrics["action_abs_max"], float(action.abs().max().detach().cpu().item()))
                ref_index = min(step + 1, ref_dict["ref_body_pos"].shape[0] - 1)
                observation = env.step(action, reference_pos=ref_dict["ref_body_pos"][ref_index].detach().cpu().numpy())
                if debug_trace_path is not None and step < int(debug_trace_steps):
                    _append_trace(trace, f"step_{step}", env.last_step_debug)
                    _append_trace(trace, f"step_{step}_obs", observation)
                joint_pos = torch.tensor(env.data.qpos[7:], dtype=torch.float32, device=device).reshape(1, -1)
                target_joint = ref_dict["dof_pos"][ref_index : ref_index + 1].to(device)
                metrics["joint_abs_error_mean"].append(float((joint_pos - target_joint).abs().mean().detach().cpu().item()))
                metrics["completed_steps"] += 1
                if progress_every > 0 and (step == 0 or (step + 1) % int(progress_every) == 0 or step + 1 == rollout_steps):
                    print(f"motion={motion_id} step={step + 1}/{rollout_steps}", flush=True)
                if real_time and not headless:
                    sleep_time = float(real_time_dt) - (time.perf_counter() - step_start)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

            metrics["joint_abs_error_mean"] = float(np.mean(metrics["joint_abs_error_mean"])) if metrics["joint_abs_error_mean"] else 0.0
            summary[str(motion_id)] = metrics
            with (output_dir / f"metrics_{motion_id}.json").open("w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            if debug_trace_path is not None:
                trace_path = Path(debug_trace_path)
                if len(motion_list) > 1:
                    trace_path = trace_path.with_name(f"{trace_path.stem}_{motion_id}{trace_path.suffix}")
                _save_trace(trace_path, trace)
                print(f"Saved slim MuJoCo debug trace to {trace_path}", flush=True)
            print(f"Slim MuJoCo inference motion={motion_id} metrics={metrics}", flush=True)

        with (output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Saved slim MuJoCo inference summary to {output_dir / 'summary.json'}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
