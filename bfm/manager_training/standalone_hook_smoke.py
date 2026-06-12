from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .standalone_context import TrainContext, TrainRuntime, TrainState
from .standalone_hooks import CheckpointHook, EvaluationHook, TrainLogHook


class _FakeEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset_count = 0
        self.priority_updates = []

    def reset(self):
        self.reset_count += 1
        return {"time": np.zeros((self.num_envs, 1), dtype=np.float32)}, {}

    def update_motion_sampling_weights(self, **kwargs):
        self.priority_updates.append(kwargs)


class _FakeModel:
    def __init__(self):
        self.train_modes = []

    def train(self, mode: bool = True):
        self.train_modes.append(mode)


class _FakeAgent:
    def __init__(self):
        self.device = "cpu"
        self._model = _FakeModel()
        self.saved_paths = []

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(path)


class _FakeTrainBuffer:
    def __init__(self):
        self.saved_paths = []

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(path)


class _FakeExpertSlicer:
    def __init__(self):
        self.motion_ids = [10, 11]
        self.file_names = ["a", "b"]
        self.priority_updates = []

    def update_priorities(self, **kwargs):
        self.priority_updates.append(kwargs)


class _FakeEvaluation:
    def __init__(self):
        self.calls = []

    def run(self, *, timestep, agent_or_model, replay_buffer, logger, env):
        del agent_or_model, replay_buffer, logger, env
        self.calls.append(timestep)
        return {
            "m0": {"motion_id": 10, "emd": 0.5},
            "m1": {"motion_id": 11, "emd": 1.0},
        }, {"metric": 1.0}


class _FakeLogger:
    def __init__(self):
        self.rows = []

    def log(self, row):
        self.rows.append(row)


@dataclass
class _Cfg:
    online_parallel_envs: int = 4
    checkpoint_every_steps: int = 8
    checkpoint_buffer: bool = True
    eval_every_steps: int = 8
    prioritization: bool = True
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 2.0
    prioritization_scale: float = 2.0
    prioritization_mode: str = "lin"
    buffer_device: str = "cpu"
    log_every_updates: int = 8
    use_wandb: bool = False


def _make_ctx(tmpdir: Path) -> TrainContext:
    cfg = _Cfg()
    env = _FakeEnv(cfg.online_parallel_envs)
    agent = _FakeAgent()
    replay_buffer = {"train": _FakeTrainBuffer(), "expert_slicer": _FakeExpertSlicer()}
    evaluation = _FakeEvaluation()
    train_logger = _FakeLogger()
    eval_logger = _FakeLogger()
    runtime = TrainRuntime(
        cfg=cfg,
        work_dir=tmpdir,
        train_env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        evaluations={"humanoidverse_tracking_eval": evaluation},
        eval_loggers={"humanoidverse_tracking_eval": eval_logger},
        train_logger=train_logger,
        prioritization_eval_name="humanoidverse_tracking_eval",
        checkpoint_time=0,
    )
    td, info = env.reset()
    state = TrainState(
        t=0,
        td=td,
        info=info,
        terminated=np.zeros(cfg.online_parallel_envs, dtype=bool),
        truncated=np.zeros(cfg.online_parallel_envs, dtype=bool),
    )
    return TrainContext(runtime=runtime, state=state)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(Path(tmp))

        checkpoint_hook = CheckpointHook()
        checkpoint_hook.setup(ctx)
        checkpoint_hook.before_rollout(ctx)
        assert ctx.runtime.agent.saved_paths == []
        ctx.state.t = 8
        checkpoint_hook.before_rollout(ctx)
        assert len(ctx.runtime.agent.saved_paths) == 1
        assert (Path(tmp) / "checkpoint" / "train_status.json").exists()

        eval_hook = EvaluationHook(use_shared_train_env=True)
        eval_hook.setup(ctx)
        ctx.state.t = 0
        eval_hook.before_rollout(ctx)
        assert ctx.runtime.evaluations["humanoidverse_tracking_eval"].calls == [0]
        assert ctx.runtime.train_env.reset_count == 2
        assert len(ctx.runtime.train_env.priority_updates) == 1
        assert len(ctx.runtime.replay_buffer["expert_slicer"].priority_updates) == 1

        log_hook = TrainLogHook()
        log_hook.setup(ctx)
        ctx.state.t = 8
        log_hook.after_update(ctx, {"loss": torch.tensor([1.0, 3.0])})
        log_hook.after_update(ctx, {"loss": torch.tensor([3.0, 5.0])})
        log_hook.on_loop_end(ctx)
        assert ctx.runtime.train_logger.rows[-1]["loss"] == 3.0
        assert ctx.state.total_metrics is None

    print("standalone hook smoke passed")


if __name__ == "__main__":
    main()
