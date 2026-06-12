from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainRuntime:
    cfg: Any
    work_dir: Path
    train_env: Any
    agent: Any
    replay_buffer: dict[str, Any]
    evaluations: dict[str, Any]
    eval_loggers: dict[str, Any]
    train_logger: Any
    prioritization_eval_name: str | None
    checkpoint_time: int


@dataclass
class TrainState:
    t: int
    td: Any
    info: dict[str, Any]
    terminated: np.ndarray
    truncated: np.ndarray
    z_context: Any = None
    last_eval_metrics: dict[str, Any] | None = None
    force_truncate_current_transition: bool = False
    pending_update_metrics: list[dict[str, Any]] = field(default_factory=list)
    total_metrics: dict[str, Any] | None = None
    num_metric_updates: int = 0
    start_time: float = field(default_factory=time.time)
    fps_start_time: float = field(default_factory=time.time)


@dataclass
class TrainContext:
    runtime: TrainRuntime
    state: TrainState

    @property
    def cfg(self):
        return self.runtime.cfg

    @property
    def timestep(self) -> int:
        return int(self.state.t)

    def reset_rollout_state(self) -> None:
        td, info = self.runtime.train_env.reset()
        self.state.td = td
        self.state.info = info
        self.state.terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        self.state.truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
