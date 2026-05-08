from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.cpp_env import make_cpp_env
from train.common import (
    EVAL_DIR,
    MODELS_DIR,
    ensure_dirs,
    evaluate_recurrent,
    save_json,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Curriculum with RecurrentPPO.")
    parser.add_argument("--source-model", required=True, help="Path to a recurrent model trained on 5x5.")
    parser.add_argument("--target-size", type=int, default=10)
    parser.add_argument("--target-obstacles", type=int, default=12)
    parser.add_argument("--target-max-steps", type=int, default=500)
    parser.add_argument("--view-radius", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=700_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    source_path = Path(args.source_model)
    if not source_path.exists():
        raise FileNotFoundError(f"Source model not found: {source_path}")

    exp_id = f"recurrent_curriculum_{args.target_size}x{args.target_size}_{args.target_obstacles}obs_seed{args.seed}"
    env = Monitor(
        make_cpp_env(
            size=args.target_size,
            obstacles=args.target_obstacles,
            max_steps=args.target_max_steps,
            view_radius=args.view_radius,
        )
    )
    model = RecurrentPPO.load(source_path, env=env, device=args.device)
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)

    model_path = MODELS_DIR / f"{exp_id}.zip"
    model.save(model_path)

    eval_env = make_cpp_env(
        size=args.target_size,
        obstacles=args.target_obstacles,
        max_steps=args.target_max_steps,
        view_radius=args.view_radius,
    )
    rows, summary = evaluate_recurrent(
        model, eval_env, episodes=args.eval_episodes, seed=args.seed + 20_000
    )
    eval_env.close()
    env.close()

    write_csv(EVAL_DIR / f"{exp_id}_episodes.csv", rows)
    save_json(
        EVAL_DIR / f"{exp_id}_summary.json",
        {
            **summary,
            "source_model": str(source_path),
            "target_size": args.target_size,
            "target_obstacles": args.target_obstacles,
            "target_max_steps": args.target_max_steps,
        },
    )
    print(f"Saved curriculum model: {model_path}")


if __name__ == "__main__":
    main()

