from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from envs.cpp_env import make_cpp_env

if TYPE_CHECKING:
    from sb3_contrib import RecurrentPPO


MODELS_DIR = Path("models")
LOG_DIR = Path("log")
EVAL_DIR = Path("evaluation")


@dataclass
class EnvConfig:
    size: int
    obstacles: int
    max_steps: int
    view_radius: int = 2


def default_max_steps(size: int) -> int:
    return {5: 200, 10: 500, 20: 1000}.get(size, max(200, size * size * 5))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_id(prefix: str, env_cfg: EnvConfig, seed: int) -> str:
    return (
        f"{prefix}_{env_cfg.size}x{env_cfg.size}_obs{env_cfg.obstacles}_"
        f"steps{env_cfg.max_steps}_vr{env_cfg.view_radius}_seed{seed}_{timestamp()}"
    )


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("envs").mkdir(exist_ok=True)
    Path("train").mkdir(exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_ppo(
    model: PPO,
    env: gym.Env,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            total_reward += float(reward)
            steps += 1
        rows.append(
            {
                "episode": ep + 1,
                "full_coverage": bool(terminated and not truncated),
                "coverage": float(info["coverage"]),
                "coverage_percent": float(info["coverage"]) * 100.0,
                "steps": steps,
                "total_reward": total_reward,
            }
        )
    return rows, summarize_eval(rows)


def evaluate_recurrent(
    model: "RecurrentPPO",
    env: gym.Env,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not terminated and not truncated:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            episode_starts = np.array([terminated or truncated], dtype=bool)
            total_reward += float(reward)
            steps += 1

        rows.append(
            {
                "episode": ep + 1,
                "full_coverage": bool(terminated and not truncated),
                "coverage": float(info["coverage"]),
                "coverage_percent": float(info["coverage"]) * 100.0,
                "steps": steps,
                "total_reward": total_reward,
            }
        )
    return rows, summarize_eval(rows)


def summarize_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    covers = np.array([r["coverage"] for r in rows], dtype=np.float32)
    steps = np.array([r["steps"] for r in rows], dtype=np.float32)
    rewards = np.array([r["total_reward"] for r in rows], dtype=np.float32)
    full = int(sum(bool(r["full_coverage"]) for r in rows))
    episodes = len(rows)
    return {
        "episodes": episodes,
        "full_coverage_count": full,
        "full_coverage_rate": full / episodes,
        "full_coverage_rate_percent": 100.0 * full / episodes,
        "average_coverage": float(np.mean(covers)),
        "average_coverage_percent": float(np.mean(covers) * 100.0),
        "average_steps": float(np.mean(steps)),
        "average_reward": float(np.mean(rewards)),
    }


def quick_eval_on_sizes(
    model: PPO | "RecurrentPPO",
    algorithm: str,
    sizes: list[int],
    obstacles_by_size: dict[int, int],
    steps_by_size: dict[int, int],
    episodes: int,
    seed: int,
    view_radius: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {"algorithm": algorithm, "episodes_per_size": episodes, "sizes": {}}
    for size in sizes:
        env = make_cpp_env(
            size=size,
            obstacles=obstacles_by_size[size],
            max_steps=steps_by_size[size],
            view_radius=view_radius,
        )
        if algorithm == "ppo":
            rows, summary = evaluate_ppo(model, env, episodes=episodes, seed=seed + size)
        else:
            rows, summary = evaluate_recurrent(model, env, episodes=episodes, seed=seed + size)
        env.close()
        summaries["sizes"][f"{size}x{size}"] = summary
        for row in rows:
            all_rows.append({"size": size, **row})
    return all_rows, summaries

