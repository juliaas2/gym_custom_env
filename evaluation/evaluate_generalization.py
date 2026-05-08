from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.common import EVAL_DIR, ensure_dirs, quick_eval_on_sizes, save_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: Evaluate generalization on 5x5, 10x10 and 20x20.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--algorithm", choices=["ppo", "recurrent_ppo"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-radius", type=int, default=2)
    parser.add_argument("--suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.algorithm == "ppo":
        model = PPO.load(model_path, device="cpu")
    else:
        model = RecurrentPPO.load(model_path, device="cpu")

    all_rows, summary = quick_eval_on_sizes(
        model=model,
        algorithm="ppo" if args.algorithm == "ppo" else "recurrent_ppo",
        sizes=[5, 10, 20],
        obstacles_by_size={5: 3, 10: 12, 20: 48},
        steps_by_size={5: 200, 10: 500, 20: 1000},
        episodes=args.episodes,
        seed=args.seed,
        view_radius=args.view_radius,
    )
    stem = model_path.stem
    suffix = f"_{args.suffix}" if args.suffix else ""
    csv_path = EVAL_DIR / f"{stem}_generalization{suffix}_episodes.csv"
    summary_path = EVAL_DIR / f"{stem}_generalization{suffix}_summary.json"
    write_csv(csv_path, all_rows)
    save_json(summary_path, summary)
    print(f"Saved episodes: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

