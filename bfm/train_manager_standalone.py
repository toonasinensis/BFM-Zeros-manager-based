from __future__ import annotations

import tyro

from bfm.manager_training.standalone_config import BFMZeroManagerTrainSettings
from bfm.manager_training.standalone_runner import run_standalone_manager_training


def main(settings: BFMZeroManagerTrainSettings = BFMZeroManagerTrainSettings.from_env()) -> None:
    run_standalone_manager_training(settings)


if __name__ == "__main__":
    tyro.cli(main)
