from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Any

import bfm.agents.fb_cpr_aux.agent as fb_cpr_aux_agent
from bfm.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from bfm.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from bfm.agents.nn_filters import DictInputFilterConfig
from bfm.agents.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from bfm.agents.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig

if TYPE_CHECKING:
    from bfm.agents.envs.bfmzero_manager_isaac import BFMZeroManagerIsaacConfig
    from bfm.agents.evaluations.bfmzero_manager import BFMZeroManagerTrackingEvaluationConfig


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if raw.lower() in ("none", "null"):
        return None
    return int(raw)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclasses.dataclass(frozen=True)
class BFMZeroManagerTrainSettings:
    device: str = "cuda:0"
    agent_device: str = "cuda"
    buffer_device: str = "cuda"
    work_dir: str = "results/bfmzero-manager-nohead-minimal-cuda0"
    compile_model: bool = False
    seed: int = 4728

    online_parallel_envs: int = 1024
    num_env_steps: int = 384000000
    log_every_updates: int = 10240
    update_agent_every: int = 1024
    num_seed_steps: int = 10240
    num_agent_updates: int = 16
    batch_size: int = 1024
    buffer_size: int = 3000000
    checkpoint_every_steps: int = 1024000
    checkpoint_buffer: bool = True

    enable_eval: bool = True
    eval_every_steps: int = 1024000
    prioritization: bool = True

    use_wandb: bool = True
    wandb_name: str = ""
    wandb_entity: str = "xiechunyang1-hajimi"
    wandb_project: str = "hajimi"
    wandb_group: str = "bfmzero-manager-nohead-minimal"
    disable_tqdm: bool = False

    motion_file: str = "bfm/data/lafan_29dof_10s-clipped.pkl"
    robot_config: str = "g1/g1_29dof_hard_waist_no_head"
    max_episode_length_s: float = 10.0
    base_ang_vel_obs_scale: float = 1.0
    expert_base_ang_vel_obs_scale: float = 1.0
    training_randomize_motions: bool = True
    training_max_num_seqs: int | None = None
    enable_domain_randomization: bool = True

    @classmethod
    def from_env(cls) -> "BFMZeroManagerTrainSettings":
        return cls(
            device=_env_str("BFMZERO_MANAGER_DEVICE", cls.device),
            agent_device=_env_str("BFMZERO_MANAGER_AGENT_DEVICE", cls.agent_device),
            buffer_device=_env_str("BFMZERO_MANAGER_BUFFER_DEVICE", cls.buffer_device),
            work_dir=_env_str("BFMZERO_MANAGER_WORK_DIR", cls.work_dir),
            compile_model=_env_bool("BFMZERO_MANAGER_COMPILE", cls.compile_model),
            seed=_env_int("BFMZERO_MANAGER_SEED", cls.seed),
            online_parallel_envs=_env_int("BFMZERO_MANAGER_ONLINE_ENVS", cls.online_parallel_envs),
            num_env_steps=_env_int("BFMZERO_MANAGER_NUM_ENV_STEPS", cls.num_env_steps),
            log_every_updates=_env_int("BFMZERO_MANAGER_LOG_EVERY_UPDATES", cls.log_every_updates),
            update_agent_every=_env_int("BFMZERO_MANAGER_UPDATE_EVERY", cls.update_agent_every),
            num_seed_steps=_env_int("BFMZERO_MANAGER_NUM_SEED_STEPS", cls.num_seed_steps),
            num_agent_updates=_env_int("BFMZERO_MANAGER_NUM_AGENT_UPDATES", cls.num_agent_updates),
            batch_size=_env_int("BFMZERO_MANAGER_BATCH_SIZE", cls.batch_size),
            buffer_size=_env_int("BFMZERO_MANAGER_BUFFER_SIZE", cls.buffer_size),
            checkpoint_every_steps=_env_int("BFMZERO_MANAGER_CHECKPOINT_EVERY_STEPS", cls.checkpoint_every_steps),
            checkpoint_buffer=_env_bool("BFMZERO_MANAGER_CHECKPOINT_BUFFER", cls.checkpoint_buffer),
            enable_eval=_env_bool("BFMZERO_MANAGER_ENABLE_EVAL", cls.enable_eval),
            eval_every_steps=_env_int("BFMZERO_MANAGER_EVAL_EVERY_STEPS", cls.eval_every_steps),
            prioritization=_env_bool("BFMZERO_MANAGER_PRIORITIZATION", cls.prioritization),
            use_wandb=_env_bool("BFMZERO_MANAGER_USE_WANDB", cls.use_wandb),
            wandb_name=_env_str("BFMZERO_WANDB_NAME", cls.wandb_name),
            wandb_entity=_env_str("BFMZERO_WANDB_ENTITY", cls.wandb_entity),
            wandb_project=_env_str("BFMZERO_WANDB_PROJECT", cls.wandb_project),
            wandb_group=_env_str("BFMZERO_WANDB_GROUP", cls.wandb_group),
            disable_tqdm=_env_bool("BFMZERO_MANAGER_DISABLE_TQDM", cls.disable_tqdm),
            motion_file=_env_str("BFMZERO_MANAGER_MOTION_FILE", cls.motion_file),
            robot_config=_env_str("BFMZERO_MANAGER_ROBOT_CONFIG", cls.robot_config),
            base_ang_vel_obs_scale=_env_float("BFMZERO_MANAGER_BASE_ANG_VEL_OBS_SCALE", cls.base_ang_vel_obs_scale),
            expert_base_ang_vel_obs_scale=_env_float(
                "BFMZERO_MANAGER_EXPERT_BASE_ANG_VEL_OBS_SCALE",
                cls.expert_base_ang_vel_obs_scale,
            ),
            training_randomize_motions=_env_bool("BFMZERO_MANAGER_RANDOMIZE_MOTIONS", cls.training_randomize_motions),
            training_max_num_seqs=_env_optional_int("BFMZERO_MANAGER_TRAINING_MAX_NUM_SEQS", cls.training_max_num_seqs),
            enable_domain_randomization=_env_bool("BFMZERO_MANAGER_ENABLE_DOMAIN_RANDOMIZATION", cls.enable_domain_randomization),
        )


def build_bfmzero_aux_agent_config(*, batch_size: int, compile_model: bool, agent_device: str = "cuda") -> FBcprAuxAgentConfig:
    agent_train_config_cls = getattr(fb_cpr_aux_agent, "FBcprAuxAgent" + "Train" + "Config")
    return FBcprAuxAgentConfig(
        name="FBcprAuxAgent",
        model=FBcprAuxModelConfig(
            name="FBcprAuxModel",
            device=agent_device,
            archi=FBcprAuxModelArchiConfig(
                name="FBcprAuxModelArchiConfig",
                z_dim=256,
                norm_z=True,
                f=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig",
                        key=["state", "privileged_state", "last_action", "history_actor"],
                    ),
                ),
                b=BackwardArchiConfig(
                    name="BackwardArchi",
                    hidden_dim=256,
                    hidden_layers=1,
                    norm=True,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                actor=ActorArchiConfig(
                    name="actor",
                    model="residual",
                    hidden_dim=2048,
                    hidden_layers=6,
                    embedding_layers=2,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "last_action", "history_actor"]),
                ),
                critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig",
                        key=["state", "privileged_state", "last_action", "history_actor"],
                    ),
                ),
                discriminator=DiscriminatorArchiConfig(
                    name="DiscriminatorArchi",
                    hidden_dim=1024,
                    hidden_layers=3,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                aux_critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig",
                        key=["state", "privileged_state", "last_action", "history_actor"],
                    ),
                ),
            ),
            obs_normalizer=ObsNormalizerConfig(
                name="ObsNormalizerConfig",
                normalizers={
                    "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                },
                allow_mismatching_keys=True,
            ),
            inference_batch_size=500000,
            seq_length=8,
            actor_std=0.05,
            amp=False,
            norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
        ),
        train=agent_train_config_cls(
            name="FBcprAuxAgent" + "Train" + "Config",
            lr_f=0.0003,
            lr_b=1e-05,
            lr_actor=0.0003,
            weight_decay=0.0,
            clip_grad_norm=0.0,
            fb_target_tau=0.01,
            ortho_coef=100.0,
            train_goal_ratio=0.2,
            fb_pessimism_penalty=0.0,
            actor_pessimism_penalty=0.5,
            stddev_clip=0.3,
            q_loss_coef=0.0,
            batch_size=batch_size,
            discount=0.98,
            use_mix_rollout=True,
            update_z_every_step=100,
            z_buffer_size=8192,
            rollout_expert_trajectories=True,
            rollout_expert_trajectories_length=250,
            rollout_expert_trajectories_percentage=0.5,
            lr_discriminator=1e-05,
            lr_critic=0.0003,
            critic_target_tau=0.005,
            critic_pessimism_penalty=0.5,
            reg_coeff=0.05,
            scale_reg=True,
            expert_asm_ratio=0.6,
            relabel_ratio=0.8,
            grad_penalty_discriminator=10.0,
            weight_decay_discriminator=0.0,
            lr_aux_critic=0.0003,
            reg_coeff_aux=0.02,
            aux_critic_pessimism_penalty=0.5,
        ),
        aux_rewards=[
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
            "penalty_undesired_contact",
            "penalty_feet_ori",
            "penalty_ankle_roll",
            "penalty_slippage",
        ],
        aux_rewards_scaling={
            "penalty_action_rate": -0.1,
            "penalty_feet_ori": -0.4,
            "penalty_ankle_roll": -4.0,
            "limits_dof_pos": -10.0,
            "penalty_slippage": -2.0,
            "penalty_undesired_contact": -1.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        },
        cudagraphs=False,
        compile=compile_model,
    )


@dataclasses.dataclass(frozen=True)
class StandaloneManagerTrainingConfig:
    agent: FBcprAuxAgentConfig
    env: BFMZeroManagerIsaacConfig
    work_dir: str
    seed: int
    online_parallel_envs: int
    log_every_updates: int
    num_env_steps: int
    update_agent_every: int
    num_seed_steps: int
    num_agent_updates: int
    checkpoint_every_steps: int
    checkpoint_buffer: bool
    prioritization: bool
    prioritization_min_val: float
    prioritization_max_val: float
    prioritization_scale: float
    prioritization_mode: str
    buffer_size: int
    use_wandb: bool
    wandb_name: str | None
    wandb_ename: str | None
    wandb_gname: str | None
    wandb_pname: str | None
    buffer_device: str
    disable_tqdm: bool
    evaluations: list[BFMZeroManagerTrackingEvaluationConfig]
    eval_every_steps: int
    tags: dict[str, Any]
    training_max_num_seqs: int | None
    expert_base_ang_vel_obs_scale: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent.model_dump(),
            "env": self.env.model_dump(),
            "work_dir": self.work_dir,
            "seed": self.seed,
            "online_parallel_envs": self.online_parallel_envs,
            "log_every_updates": self.log_every_updates,
            "num_env_steps": self.num_env_steps,
            "update_agent_every": self.update_agent_every,
            "num_seed_steps": self.num_seed_steps,
            "num_agent_updates": self.num_agent_updates,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "checkpoint_buffer": self.checkpoint_buffer,
            "prioritization": self.prioritization,
            "prioritization_min_val": self.prioritization_min_val,
            "prioritization_max_val": self.prioritization_max_val,
            "prioritization_scale": self.prioritization_scale,
            "prioritization_mode": self.prioritization_mode,
            "buffer_size": self.buffer_size,
            "use_wandb": self.use_wandb,
            "wandb_name": self.wandb_name,
            "wandb_ename": self.wandb_ename,
            "wandb_gname": self.wandb_gname,
            "wandb_pname": self.wandb_pname,
            "buffer_device": self.buffer_device,
            "disable_tqdm": self.disable_tqdm,
            "evaluations": [evaluation.model_dump() for evaluation in self.evaluations],
            "eval_every_steps": self.eval_every_steps,
            "tags": self.tags,
            "training_max_num_seqs": self.training_max_num_seqs,
            "expert_base_ang_vel_obs_scale": self.expert_base_ang_vel_obs_scale,
        }


def build_standalone_manager_train_config(settings: BFMZeroManagerTrainSettings) -> StandaloneManagerTrainingConfig:
    from bfm.agents.envs.bfmzero_manager_isaac import BFMZeroManagerIsaacConfig
    from bfm.agents.evaluations.bfmzero_manager import BFMZeroManagerTrackingEvaluationConfig

    if settings.prioritization and not settings.enable_eval:
        raise ValueError("prioritization=True requires enable_eval=True.")
    if not settings.compile_model:
        os.environ.setdefault("BFM_DISABLE_TORCH_COMPILE", "1")

    evaluations = [
        BFMZeroManagerTrackingEvaluationConfig(
            name="BFMZeroManagerTrackingEvaluationConfig",
            name_in_logs="humanoidverse_tracking_eval",
            env=None,
            num_envs=settings.online_parallel_envs,
            n_episodes_per_motion=1,
        )
    ] if settings.enable_eval else []

    return StandaloneManagerTrainingConfig(
        agent=build_bfmzero_aux_agent_config(
            batch_size=settings.batch_size,
            compile_model=settings.compile_model,
            agent_device=settings.agent_device,
        ),
        env=BFMZeroManagerIsaacConfig(
            name="bfmzero_manager_isaac",
            device=settings.device,
            lafan_tail_path=settings.motion_file,
            robot_config=settings.robot_config,
            enable_cameras=False,
            headless=True,
            max_episode_length_s=settings.max_episode_length_s,
            include_history_actor=True,
            root_height_obs=True,
            default_motion_id=0,
            training_randomize_motions=settings.training_randomize_motions,
            training_max_num_seqs=settings.training_max_num_seqs,
            base_ang_vel_obs_scale=settings.base_ang_vel_obs_scale,
            enable_domain_randomization=settings.enable_domain_randomization,
            render_mode=None,
        ),
        work_dir=settings.work_dir,
        seed=settings.seed,
        online_parallel_envs=settings.online_parallel_envs,
        log_every_updates=settings.log_every_updates,
        num_env_steps=settings.num_env_steps,
        update_agent_every=settings.update_agent_every,
        num_seed_steps=settings.num_seed_steps,
        num_agent_updates=settings.num_agent_updates,
        checkpoint_every_steps=settings.checkpoint_every_steps,
        checkpoint_buffer=settings.checkpoint_buffer,
        prioritization=settings.prioritization,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode="exp",
        buffer_size=settings.buffer_size,
        use_wandb=settings.use_wandb,
        wandb_name=settings.wandb_name,
        wandb_ename=settings.wandb_entity,
        wandb_gname=settings.wandb_group,
        wandb_pname=settings.wandb_project,
        buffer_device=settings.buffer_device,
        disable_tqdm=settings.disable_tqdm,
        evaluations=evaluations,
        eval_every_steps=settings.eval_every_steps,
        tags={
            "manager_no_head": True,
            "standalone_manager_trainer": True,
            "full_train": True,
            "compile_model": settings.compile_model,
            "agent_device": settings.agent_device,
            "buffer_device": settings.buffer_device,
            "online_parallel_envs": settings.online_parallel_envs,
            "log_every_updates": settings.log_every_updates,
            "prioritization": settings.prioritization,
            "enable_eval": settings.enable_eval,
            "disable_tqdm": settings.disable_tqdm,
            "checkpoint_buffer": settings.checkpoint_buffer,
            "training_max_num_seqs": settings.training_max_num_seqs,
            "base_ang_vel_obs_scale": settings.base_ang_vel_obs_scale,
            "expert_base_ang_vel_obs_scale": settings.expert_base_ang_vel_obs_scale,
            "enable_domain_randomization": settings.enable_domain_randomization,
        },
        training_max_num_seqs=settings.training_max_num_seqs,
        expert_base_ang_vel_obs_scale=settings.expert_base_ang_vel_obs_scale,
    )
