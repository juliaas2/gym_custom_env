from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.cpp_env import make_cpp_env
from train.common import (
    EnvConfig,
    EVAL_DIR,
    LOG_DIR,
    MODELS_DIR,
    default_max_steps,
    ensure_dirs,
    evaluate_ppo,
    run_id,
    save_json,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: PPO baseline on 5x5 CPP.")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-radius", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    max_steps = args.max_steps if args.max_steps > 0 else default_max_steps(args.size)
    cfg = EnvConfig(args.size, args.obstacles, max_steps, args.view_radius)
    exp_id = run_id("ppo_baseline", cfg, args.seed)

    env = Monitor(make_cpp_env(cfg.size, cfg.obstacles, cfg.max_steps, cfg.view_radius))
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps)

    model_path = MODELS_DIR / f"{exp_id}.zip"
    model.save(model_path)

    eval_env = make_cpp_env(cfg.size, cfg.obstacles, cfg.max_steps, cfg.view_radius)
    rows, summary = evaluate_ppo(model, eval_env, episodes=args.eval_episodes, seed=args.seed + 10_000)
    eval_env.close()
    env.close()

    write_csv(EVAL_DIR / f"{exp_id}_episodes.csv", rows)
    save_json(EVAL_DIR / f"{exp_id}_summary.json", summary)
    save_json(
        MODELS_DIR / f"{exp_id}_config.json",
        {
            "algorithm": "ppo",
            "model_path": str(model_path),
            "log_dir": str(LOG_DIR / exp_id),
            "env": cfg.__dict__,
            "train": vars(args),
        },
    )
    print(f"Saved model: {model_path}")
    print(f"Eval summary: {EVAL_DIR / f'{exp_id}_summary.json'}")


if __name__ == "__main__":
    main()

