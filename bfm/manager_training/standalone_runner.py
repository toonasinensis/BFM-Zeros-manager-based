from __future__ import annotations

import torch

from .standalone_config import BFMZeroManagerTrainSettings, build_standalone_manager_train_config


def set_default_cuda_device(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))


def initialize_isaac_sim(settings: BFMZeroManagerTrainSettings) -> None:
    from bfm.manager_envs.mdp.isaac_app import instantiate_isaac_sim

    instantiate_isaac_sim(settings.online_parallel_envs, enable_cameras=False, headless=True)


def run_standalone_manager_training(settings: BFMZeroManagerTrainSettings) -> None:
    set_default_cuda_device(settings.device)
    initialize_isaac_sim(settings)

    from .standalone_trainer import StandaloneManagerTrainer

    cfg = build_standalone_manager_train_config(settings)
    trainer = StandaloneManagerTrainer(cfg)
    trainer.train()
