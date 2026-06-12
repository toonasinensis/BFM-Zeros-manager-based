from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bfm.agents.buffers.trajectory import TrajectoryDictBufferMultiDim

CHECKPOINT_DIR_NAME = "checkpoint"


def load_agent_or_build(work_dir: Path, cfg, *, obs_space, action_dim: int):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    if checkpoint_dir.exists():
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = int(train_status["time"])
        print(f"Loading the agent at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(obs_space=obs_space, action_dim=action_dim)
    return agent, checkpoint_time


def load_or_create_train_buffer(work_dir: Path, cfg) -> TrajectoryDictBufferMultiDim:
    buffer_dir = work_dir / CHECKPOINT_DIR_NAME / "buffers" / "train"
    if buffer_dir.exists():
        print("Loading checkpointed buffer")
        buffer = TrajectoryDictBufferMultiDim.load(buffer_dir, device=cfg.buffer_device)
        print(f"Loaded buffer of size {len(buffer)}")
        return buffer

    output_key_t = ["observation", "action", "z", "terminated", "truncated", "step_count", "reward", "aux_rewards"]
    return TrajectoryDictBufferMultiDim(
        capacity=cfg.buffer_size // cfg.online_parallel_envs,
        device=cfg.buffer_device,
        n_dim=2,
        end_key="truncated",
        output_key_t=output_key_t,
        output_key_tp1=["observation", "terminated"],
    )


def save_checkpoint(work_dir: Path, cfg, agent, replay_buffer: dict[str, Any], time: int) -> None:
    print(f"Checkpointing at time {time}")
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    agent.save(str(checkpoint_dir))
    if cfg.checkpoint_buffer:
        replay_buffer["train"].save(checkpoint_dir / "buffers" / "train")
    with (checkpoint_dir / "train_status.json").open("w+") as f:
        json.dump({"time": time}, f, indent=4)
