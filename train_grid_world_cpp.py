#
# PPO training/evaluation for the Coverage Path Planning GridWorld.
#
# Examples:
#   python train_grid_world_cpp.py train 5 3 200 500000
#   python train_grid_world_cpp.py curriculum 10 12 500 500000 --model-path data/model_5x5.zip
#   python train_grid_world_cpp.py test 10 12 500 --model-path data/model_10x10.zip
#   python train_grid_world_cpp.py run 5 3 200 --model-path data/model_5x5.zip
#

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor


ENV_ID = "gymnasium_env/GridWorldCPP-v0"
DATA_DIR = Path("data")
LOG_DIR = Path("log")


def print_action(action: int) -> str:
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")


def register_env() -> None:
    try:
        gym.spec(ENV_ID)
    except gym.error.Error:
        gym.register(
            id=ENV_ID,
            entry_point=GridWorldCPPEnv,
        )


def default_max_steps(dim: int) -> int:
    return {
        5: 200,
        10: 500,
        20: 1000,
    }.get(dim, max(200, dim * dim * 5))


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "run"


def make_run_id(args: argparse.Namespace, suffix: str = "") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    view_size = (2 * args.view_radius) + 1
    if args.train_env_specs:
        run_id = f"ppo_cpp_mixed_view{view_size}_seed{args.seed}_{timestamp}"
    else:
        run_id = (
            f"ppo_cpp_{args.dim}x{args.dim}_obs{args.obstacles}"
            f"_steps{args.max_steps}_view{view_size}_seed{args.seed}_{timestamp}"
        )
    if args.run_name:
        run_id = f"{run_id}_{sanitize_name(args.run_name)}"
    if suffix:
        run_id = f"{run_id}_{sanitize_name(suffix)}"
    return run_id


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


def parse_train_envs(value: str) -> list[dict[str, Any]]:
    if not value:
        return []

    specs = []
    for raw_spec in value.split(","):
        parts = raw_spec.strip().split(":")
        if len(parts) not in {3, 4}:
            raise ValueError(
                "Invalid --train-envs item. Use size:obstacles:max_steps[:weight], "
                "for example 5:3:200:1,10:12:500:2,20:48:1500:6."
            )
        size = int(parts[0])
        obstacles = int(parts[1])
        max_steps = int(parts[2])
        weight = float(parts[3]) if len(parts) == 4 else 1.0
        if weight <= 0:
            raise ValueError("All --train-envs weights must be positive.")
        specs.append(
            {
                "size": size,
                "obs_quantity": obstacles,
                "max_steps": max_steps,
                "weight": weight,
            }
        )
    return specs


class MixedGridWorldCPPEnv(gym.Env):
    metadata = GridWorldCPPEnv.metadata

    def __init__(
        self,
        env_specs: list[dict[str, Any]],
        render_mode: str = "rgb_array",
        view_radius: int = 2,
    ):
        if not env_specs:
            raise ValueError("MixedGridWorldCPPEnv requires at least one environment spec.")

        self.env_specs = env_specs
        self.render_mode = render_mode
        self.view_radius = view_radius
        self.current_env: GridWorldCPPEnv | None = None
        self.current_env_index = 0
        self.weights = np.array([spec["weight"] for spec in env_specs], dtype=np.float64)
        self.weights = self.weights / np.sum(self.weights)

        first = env_specs[0]
        prototype = GridWorldCPPEnv(
            render_mode=render_mode,
            size=first["size"],
            obs_quantity=first["obs_quantity"],
            max_steps=first["max_steps"],
            view_radius=view_radius,
        )
        self.observation_space = prototype.observation_space
        self.action_space = prototype.action_space
        prototype.close()

    def _make_env(self, index: int) -> GridWorldCPPEnv:
        spec = self.env_specs[index]
        return GridWorldCPPEnv(
            render_mode=self.render_mode,
            size=spec["size"],
            obs_quantity=spec["obs_quantity"],
            max_steps=spec["max_steps"],
            view_radius=self.view_radius,
        )

    def _annotate_info(self, info: dict[str, Any]) -> dict[str, Any]:
        spec = self.env_specs[self.current_env_index]
        return {
            **info,
            "mixed_training": True,
            "training_size": spec["size"],
            "training_obstacles": spec["obs_quantity"],
            "training_max_steps": spec["max_steps"],
            "training_weight": spec["weight"],
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.current_env is not None:
            self.current_env.close()

        self.current_env_index = int(
            self.np_random.choice(len(self.env_specs), p=self.weights)
        )
        self.current_env = self._make_env(self.current_env_index)
        env_seed = int(self.np_random.integers(0, 2**31 - 1))
        observation, info = self.current_env.reset(seed=env_seed, options=options)
        return observation, self._annotate_info(info)

    def step(self, action):
        if self.current_env is None:
            raise RuntimeError("Call reset before step.")
        observation, reward, terminated, truncated, info = self.current_env.step(action)
        return observation, reward, terminated, truncated, self._annotate_info(info)

    def render(self):
        if self.current_env is None:
            return None
        return self.current_env.render()

    def close(self):
        if self.current_env is not None:
            self.current_env.close()
            self.current_env = None


def make_env(args: argparse.Namespace, render_mode: str = "rgb_array") -> gym.Env:
    return gym.make(
        ENV_ID,
        size=args.dim,
        obs_quantity=args.obstacles,
        max_steps=args.max_steps,
        view_radius=args.view_radius,
        render_mode=render_mode,
    )


def make_training_env(args: argparse.Namespace, render_mode: str = "rgb_array") -> gym.Env:
    if args.train_env_specs:
        return MixedGridWorldCPPEnv(
            args.train_env_specs,
            render_mode=render_mode,
            view_radius=args.view_radius,
        )
    return make_env(args, render_mode=render_mode)


def model_path_from_args(args: argparse.Namespace) -> Path:
    if args.model_path:
        return Path(args.model_path)

    if args.model_name:
        filename = args.model_name if args.model_name.endswith(".zip") else f"{args.model_name}.zip"
        return DATA_DIR / filename

    pattern = f"ppo_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_*.zip"
    candidates = sorted(DATA_DIR.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No model found for pattern data/{pattern}. Use --model-path or --model-name."
        )
    return candidates[0]


def action_to_int(action: Any) -> int:
    return int(np.asarray(action).item())


def evaluate_model(
    model: PPO,
    env: gym.Env,
    episodes: int,
    deterministic: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        steps = 0
        total_reward = 0.0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action_to_int(action))
            total_reward += float(reward)
            steps += 1

        rows.append(
            {
                "episode": episode + 1,
                "full_coverage": bool(terminated and not truncated),
                "coverage": float(info["coverage"]),
                "coverage_percent": float(info["coverage"]) * 100.0,
                "steps": steps,
                "total_reward": total_reward,
                "visited_cells": int(info["visited_cells"]),
                "total_free_cells": int(info["total_free_cells"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    coverages = np.array([row["coverage"] for row in rows], dtype=np.float32)
    steps_array = np.array([row["steps"] for row in rows], dtype=np.float32)
    rewards = np.array([row["total_reward"] for row in rows], dtype=np.float32)
    full_coverage_count = int(sum(row["full_coverage"] for row in rows))

    summary = {
        "episodes": episodes,
        "deterministic": deterministic,
        "full_coverage_count": full_coverage_count,
        "full_coverage_rate": full_coverage_count / episodes if episodes else 0.0,
        "full_coverage_rate_percent": (full_coverage_count / episodes * 100.0) if episodes else 0.0,
        "average_coverage": float(np.mean(coverages)) if episodes else 0.0,
        "average_coverage_percent": float(np.mean(coverages) * 100.0) if episodes else 0.0,
        "std_coverage_percent": float(np.std(coverages) * 100.0) if episodes else 0.0,
        "min_coverage_percent": float(np.min(coverages) * 100.0) if episodes else 0.0,
        "max_coverage_percent": float(np.max(coverages) * 100.0) if episodes else 0.0,
        "average_steps": float(np.mean(steps_array)) if episodes else 0.0,
        "std_steps": float(np.std(steps_array)) if episodes else 0.0,
        "min_steps": int(np.min(steps_array)) if episodes else 0,
        "max_steps": int(np.max(steps_array)) if episodes else 0,
        "average_reward": float(np.mean(rewards)) if episodes else 0.0,
    }
    return rows, summary


def save_evaluation(
    run_id: str,
    label: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    extra: dict[str, Any],
) -> tuple[Path, Path]:
    csv_path = DATA_DIR / f"{run_id}_{label}_episodes.csv"
    summary_path = DATA_DIR / f"{run_id}_{label}_summary.json"
    write_csv(csv_path, rows)
    save_json(summary_path, {**extra, **summary})
    return csv_path, summary_path


def print_summary(summary: dict[str, Any]) -> None:
    print("\n--- Evaluation Finished ---")
    print(
        "Full Coverage Rate: "
        f"{summary['full_coverage_rate_percent']:.2f}% "
        f"({summary['full_coverage_count']}/{summary['episodes']})"
    )
    print(
        "Average Coverage: "
        f"{summary['average_coverage_percent']:.2f}% "
        f"Std: {summary['std_coverage_percent']:.2f}% "
        f"Min: {summary['min_coverage_percent']:.2f}% "
        f"Max: {summary['max_coverage_percent']:.2f}%"
    )
    print(
        "Average Steps: "
        f"{summary['average_steps']:.1f} "
        f"Std: {summary['std_steps']:.1f} "
        f"Min: {summary['min_steps']} "
        f"Max: {summary['max_steps']}"
    )


class CoverageEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        run_id: str,
        eval_freq: int,
        episodes: int,
        deterministic: bool,
        seed: int,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.run_id = run_id
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.deterministic = deterministic
        self.seed = seed
        self.best_score = (-1.0, -1.0, 0.0)
        self.best_model_path = DATA_DIR / f"{run_id}_best.zip"

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        rows, summary = evaluate_model(
            self.model,
            self.eval_env,
            episodes=self.episodes,
            deterministic=self.deterministic,
            seed=self.seed + self.n_calls,
        )
        score = (
            summary["full_coverage_rate"],
            summary["average_coverage"],
            -summary["average_steps"],
        )

        save_evaluation(
            self.run_id,
            f"step{self.num_timesteps}_eval",
            rows,
            summary,
            {
                "model_path": str(self.best_model_path),
                "num_timesteps": self.num_timesteps,
                "is_best_so_far": score > self.best_score,
            },
        )

        if score > self.best_score:
            self.best_score = score
            self.model.save(self.best_model_path)
            save_json(
                DATA_DIR / f"{self.run_id}_best_summary.json",
                {
                    **summary,
                    "num_timesteps": self.num_timesteps,
                    "model_path": str(self.best_model_path),
                },
            )
            print(
                "New best checkpoint saved: "
                f"{self.best_model_path} | "
                f"full coverage {summary['full_coverage_rate_percent']:.2f}% | "
                f"avg coverage {summary['average_coverage_percent']:.2f}%"
            )

        return True


def policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    layers = [args.net_width] * args.net_layers
    return {
        "activation_fn": th.nn.ReLU,
        "net_arch": {
            "pi": layers,
            "vf": layers,
        },
    }


def config_payload(args: argparse.Namespace, run_id: str, model_path: Path, log_dir: Path) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": args.mode,
        "model_path": str(model_path),
        "log_dir": str(log_dir),
        "environment": {
            "id": ENV_ID,
            "dim": args.dim,
            "obstacles": args.obstacles,
            "max_steps": args.max_steps,
            "view_radius": args.view_radius,
            "view_size": (2 * args.view_radius) + 1,
            "observation": "agent position/coverage plus local 5x5 matrix when view_radius=2",
            "mixed_training_envs": args.train_env_specs,
        },
        "ppo": {
            "total_timesteps": args.total_timesteps,
            "ent_coef": args.ent_coef,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "net_width": args.net_width,
            "net_layers": args.net_layers,
            "device": args.device,
            "seed": args.seed,
            "eval_freq": args.eval_freq,
            "eval_episodes_during_train": args.eval_episodes_during_train,
        },
    }


def train(args: argparse.Namespace) -> None:
    run_id = make_run_id(args)
    log_dir = LOG_DIR / run_id
    model_path = DATA_DIR / f"{run_id}.zip"

    if args.check_env:
        checked_env = make_training_env(args)
        check_env(checked_env, warn=True)
        checked_env.close()

    env = Monitor(make_training_env(args))
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs(args),
        device=args.device,
        seed=args.seed,
    )

    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    save_json(DATA_DIR / f"{run_id}_config.json", config_payload(args, run_id, model_path, log_dir))

    callback_env = None
    callback = None
    if args.eval_freq > 0:
        callback_env = make_env(args)
        callback = CoverageEvalCallback(
            callback_env,
            run_id=run_id,
            eval_freq=args.eval_freq,
            episodes=args.eval_episodes_during_train,
            deterministic=not args.stochastic,
            seed=args.seed + 40_000,
        )

    print(f"Starting PPO learning with {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    if callback is not None:
        print(f"Best checkpoint path: {callback.best_model_path}")
    print(f"Training logs saved to {log_dir}")
    if callback_env is not None:
        callback_env.close()

    if args.eval_episodes > 0:
        eval_env = make_env(args)
        rows, summary = evaluate_model(
            model,
            eval_env,
            episodes=args.eval_episodes,
            deterministic=not args.stochastic,
            seed=args.seed + 10_000,
        )
        csv_path, summary_path = save_evaluation(
            run_id,
            "eval",
            rows,
            summary,
            {
                "model_path": str(model_path),
                "dim": args.dim,
                "obstacles": args.obstacles,
                "max_steps": args.max_steps,
                "view_radius": args.view_radius,
                "mixed_training_envs": args.train_env_specs,
            },
        )
        eval_env.close()
        print_summary(summary)
        print(f"Evaluation episodes saved to {csv_path}")
        print(f"Evaluation summary saved to {summary_path}")

    env.close()


def curriculum(args: argparse.Namespace) -> None:
    source_model_path = model_path_from_args(args)
    run_id = make_run_id(args, suffix="curriculum")
    log_dir = LOG_DIR / run_id
    target_model_path = DATA_DIR / f"{run_id}.zip"

    env = Monitor(make_training_env(args))
    model = PPO.load(source_model_path, env=env, device=args.device)

    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    save_json(DATA_DIR / f"{run_id}_config.json", config_payload(args, run_id, target_model_path, log_dir))

    callback_env = None
    callback = None
    if args.eval_freq > 0:
        callback_env = make_env(args)
        callback = CoverageEvalCallback(
            callback_env,
            run_id=run_id,
            eval_freq=args.eval_freq,
            episodes=args.eval_episodes_during_train,
            deterministic=not args.stochastic,
            seed=args.seed + 50_000,
        )

    print(f"Continuing PPO training from {source_model_path}")
    print(f"Target environment: {args.dim}x{args.dim}, {args.obstacles} obstacles")
    model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False, callback=callback)
    model.save(target_model_path)
    print(f"Curriculum model saved to {target_model_path}")
    if callback is not None:
        print(f"Best checkpoint path: {callback.best_model_path}")
    print(f"Training logs saved to {log_dir}")
    if callback_env is not None:
        callback_env.close()

    if args.eval_episodes > 0:
        eval_env = make_env(args)
        rows, summary = evaluate_model(
            model,
            eval_env,
            episodes=args.eval_episodes,
            deterministic=not args.stochastic,
            seed=args.seed + 20_000,
        )
        csv_path, summary_path = save_evaluation(
            run_id,
            "eval",
            rows,
            summary,
            {
                "source_model_path": str(source_model_path),
                "model_path": str(target_model_path),
                "dim": args.dim,
                "obstacles": args.obstacles,
                "max_steps": args.max_steps,
                "view_radius": args.view_radius,
                "mixed_training_envs": args.train_env_specs,
            },
        )
        eval_env.close()
        print_summary(summary)
        print(f"Evaluation episodes saved to {csv_path}")
        print(f"Evaluation summary saved to {summary_path}")

    env.close()


def test(args: argparse.Namespace) -> None:
    source_model_path = model_path_from_args(args)
    run_id = make_run_id(args, suffix="test")

    print(f"Loading model from {source_model_path}")
    model = PPO.load(source_model_path, device=args.device)
    env = make_env(args)
    rows, summary = evaluate_model(
        model,
        env,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        seed=args.seed + 30_000,
    )
    csv_path, summary_path = save_evaluation(
        run_id,
        "eval",
        rows,
        summary,
        {
            "model_path": str(source_model_path),
            "dim": args.dim,
            "obstacles": args.obstacles,
            "max_steps": args.max_steps,
            "view_radius": args.view_radius,
        },
    )
    env.close()
    print_summary(summary)
    print(f"Evaluation episodes saved to {csv_path}")
    print(f"Evaluation summary saved to {summary_path}")


def run_once(args: argparse.Namespace) -> None:
    source_model_path = model_path_from_args(args)
    print(f"Loading model from {source_model_path}")

    model = PPO.load(source_model_path, device=args.device)
    env = make_env(args, render_mode="human")
    obs, info = env.reset(seed=args.seed)
    terminated = False
    truncated = False
    steps = 0
    total_reward = 0.0

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        action_int = action_to_int(action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += float(reward)
        steps += 1
        print(
            f"Step: {steps}, Action: {print_action(action_int)}, "
            f"Reward: {reward:.2f}, Coverage: {info['coverage']:.1%}, "
            f"Done: {terminated}, Truncated: {truncated}"
        )

    print(
        "--- Run Finished --- "
        f"Total reward: {total_reward:.2f}, Coverage: {info['coverage']:.1%}"
    )
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PPO for GridWorld CPP.")
    parser.add_argument("mode", choices=["train", "test", "run", "curriculum"])
    parser.add_argument("dim", type=int, help="Grid dimension, e.g. 5, 10, 20.")
    parser.add_argument("obstacles", type=int, help="Number of obstacles.")
    parser.add_argument("max_steps", type=int, nargs="?", help="Max steps per episode.")
    parser.add_argument("total_timesteps", type=int, nargs="?", help="Training timesteps.")
    parser.add_argument("--model-path", help="Path to a .zip model file.")
    parser.add_argument("--model-name", help="Model filename inside data/, with or without .zip.")
    parser.add_argument("--run-name", default="", help="Optional suffix for output files.")
    parser.add_argument("--view-radius", type=int, default=2, help="2 means a 5x5 local observation.")
    parser.add_argument(
        "--train-envs",
        default="",
        help=(
            "Mixed training environments as size:obstacles:max_steps[:weight], "
            "comma-separated. Example: 5:3:200:1,10:12:500:2,20:48:1500:6."
        ),
    )
    parser.add_argument("--episodes", type=int, default=100, help="Episodes used by test mode.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes after training.")
    parser.add_argument("--eval-freq", type=int, default=0, help="Save best checkpoint every N train steps.")
    parser.add_argument("--eval-episodes-during-train", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--net-width", type=int, default=256)
    parser.add_argument("--net-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions in eval/test/run.")
    parser.add_argument("--no-check-env", action="store_true", help="Skip Gymnasium environment checks.")
    args = parser.parse_args()

    args.max_steps = args.max_steps if args.max_steps is not None else default_max_steps(args.dim)
    if args.total_timesteps is None:
        args.total_timesteps = 500_000 if args.mode in {"train", "curriculum"} else 0
    args.check_env = not args.no_check_env
    args.train_env_specs = parse_train_envs(args.train_envs)

    return args


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    register_env()
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "curriculum":
        curriculum(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "run":
        run_once(args)


if __name__ == "__main__":
    main()
