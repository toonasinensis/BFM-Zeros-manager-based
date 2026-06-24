from __future__ import annotations

import argparse

_ISAAC_SIM_INITIALIZED = False


def instantiate_isaac_sim(num_envs: int, enable_cameras: bool = False, headless: bool = True) -> None:
    global _ISAAC_SIM_INITIALIZED
    if _ISAAC_SIM_INITIALIZED:
        return

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.num_envs = num_envs
    args_cli.enable_cameras = enable_cameras
    args_cli.headless = headless

    app_launcher = AppLauncher(args_cli)
    _ = app_launcher.app
    _ISAAC_SIM_INITIALIZED = True
