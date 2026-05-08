from __future__ import annotations

import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv


ENV_ID = "gymnasium_env/GridWorldCPP-v1"


def register_cpp_env() -> None:
    """Register the CPP environment once."""
    try:
        gym.spec(ENV_ID)
    except gym.error.Error:
        gym.register(
            id=ENV_ID,
            entry_point=GridWorldCPPEnv,
        )


def make_cpp_env(
    size: int,
    obstacles: int,
    max_steps: int,
    view_radius: int = 2,
    render_mode: str = "rgb_array",
) -> gym.Env:
    register_cpp_env()
    return gym.make(
        ENV_ID,
        size=size,
        obs_quantity=obstacles,
        max_steps=max_steps,
        view_radius=view_radius,
        render_mode=render_mode,
    )

