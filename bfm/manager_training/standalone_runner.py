from __future__ import annotations

import torch

from .standalone_config import BFMZeroManagerTrainSettings, build_standalone_manager_train_config
from .standalone_trainer import StandaloneManagerTrainer


def set_default_cuda_device(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))


def run_standalone_manager_training(settings: BFMZeroManagerTrainSettings) -> None:
    set_default_cuda_device(settings.device)
    cfg = build_standalone_manager_train_config(settings)
    trainer = StandaloneManagerTrainer(cfg)
    trainer.train()
