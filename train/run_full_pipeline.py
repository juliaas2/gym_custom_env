from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.common import EVAL_DIR, MODELS_DIR, ensure_dirs, save_json



def run_cmd(args: list[str]) -> None:
    print("\n>>>", " ".join(args))
    subprocess.run(args, check=True, cwd=ROOT)


def find_summary_by_prefix(prefix: str) -> Path:
    candidates = sorted(EVAL_DIR.glob(f"{prefix}*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No summary found with prefix: {prefix}")
    return candidates[0]


def find_model_by_prefix(prefix: str) -> Path:
    candidates = sorted(MODELS_DIR.glob(f"{prefix}*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No model found with prefix: {prefix}")
    return candidates[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def sweep_recurrent_5x5(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for lr in args.sweep_lr:
        for ent in args.sweep_ent_coef:
            for hidden in args.sweep_lstm_hidden:
                run_cmd(
                    [
                        sys.executable,
                        "train/train_recurrent_ppo.py",
                        "--size",
                        "5",
                        "--obstacles",
                        str(args.obs_5),
                        "--max-steps",
                        str(args.steps_5),
                        "--timesteps",
                        str(args.timesteps_recurrent_5),
                        "--seed",
                        str(args.seed),
                        "--view-radius",
                        str(args.view_radius),
                        "--eval-episodes",
                        str(args.eval_episodes),
                        "--learning-rate",
                        str(lr),
                        "--ent-coef",
                        str(ent),
                        "--n-steps",
                        str(args.n_steps),
                        "--batch-size",
                        str(args.batch_size),
                        "--lstm-hidden-size",
                        str(hidden),
                        "--lstm-layers",
                        str(args.lstm_layers),
                        "--device",
                        args.device,
                    ]
                )
                summary_path = find_summary_by_prefix("recurrent_ppo_5x5")
                summary = load_json(summary_path)
                score = (
                    float(summary.get("full_coverage_rate", 0.0)),
                    float(summary.get("average_coverage", 0.0)),
                    -float(summary.get("average_steps", 1e9)),
                )
                model_stem = summary_path.stem.replace("_summary", "")
                model_path = MODELS_DIR / f"{model_stem}.zip"
                candidates.append(
                    {
                        "learning_rate": lr,
                        "ent_coef": ent,
                        "lstm_hidden_size": hidden,
                        "summary_path": str(summary_path),
                        "model_path": str(model_path),
                        "score": score,
                        "summary": summary,
                    }
                )

    if not candidates:
        raise RuntimeError("Sweep did not produce any recurrent candidates.")
    best = max(candidates, key=lambda x: x["score"])
    best_model = Path(best["model_path"])
    best_summary = Path(best["summary_path"])
    if not best_model.exists():
        best_model = find_model_by_prefix("recurrent_ppo_5x5")
    return best_model, best_summary, {"candidates": candidates, "best": best}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete PPO vs RecurrentPPO pipeline with optional sweep.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view-radius", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--generalization-episodes", type=int, default=100)
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")

    parser.add_argument("--obs-5", type=int, default=3)
    parser.add_argument("--obs-10", type=int, default=12)
    parser.add_argument("--obs-20", type=int, default=48)
    parser.add_argument("--steps-5", type=int, default=200)
    parser.add_argument("--steps-10", type=int, default=500)
    parser.add_argument("--steps-20", type=int, default=1000)

    parser.add_argument("--timesteps-ppo-5", type=int, default=500_000)
    parser.add_argument("--timesteps-recurrent-5", type=int, default=700_000)
    parser.add_argument("--timesteps-curriculum-10", type=int, default=700_000)

    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)

    parser.add_argument("--sweep-lr", type=float, nargs="+", default=[3e-4, 1e-4])
    parser.add_argument("--sweep-ent-coef", type=float, nargs="+", default=[0.01, 0.02])
    parser.add_argument("--sweep-lstm-hidden", type=int, nargs="+", default=[128, 256])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    run_tag = f"seed{args.seed}_vr{args.view_radius}"

    ppo_generalization_summary: Path | None = None
    if not args.skip_ppo:
        run_cmd(
            [
                sys.executable,
                "train/train_ppo_baseline.py",
                "--size",
                "5",
                "--obstacles",
                str(args.obs_5),
                "--max-steps",
                str(args.steps_5),
                "--timesteps",
                str(args.timesteps_ppo_5),
                "--seed",
                str(args.seed),
                "--view-radius",
                str(args.view_radius),
                "--eval-episodes",
                str(args.eval_episodes),
                "--device",
                args.device,
            ]
        )
        ppo_model = find_model_by_prefix("ppo_baseline_5x5")
        run_cmd(
            [
                sys.executable,
                "evaluation/evaluate_generalization.py",
                "--model-path",
                str(ppo_model),
                "--algorithm",
                "ppo",
                "--episodes",
                str(args.generalization_episodes),
                "--seed",
                str(args.seed),
                "--view-radius",
                str(args.view_radius),
                "--suffix",
                run_tag,
            ]
        )
        ppo_generalization_summary = find_summary_by_prefix(f"{ppo_model.stem}_generalization_{run_tag}")

    if args.skip_sweep:
        run_cmd(
            [
                sys.executable,
                "train/train_recurrent_ppo.py",
                "--size",
                "5",
                "--obstacles",
                str(args.obs_5),
                "--max-steps",
                str(args.steps_5),
                "--timesteps",
                str(args.timesteps_recurrent_5),
                "--seed",
                str(args.seed),
                "--view-radius",
                str(args.view_radius),
                "--eval-episodes",
                str(args.eval_episodes),
                "--learning-rate",
                "0.0003",
                "--ent-coef",
                "0.01",
                "--n-steps",
                str(args.n_steps),
                "--batch-size",
                str(args.batch_size),
                "--lstm-hidden-size",
                "128",
                "--lstm-layers",
                str(args.lstm_layers),
                "--device",
                args.device,
            ]
        )
        recurrent_5_model = find_model_by_prefix("recurrent_ppo_5x5")
        recurrent_5_summary = find_summary_by_prefix("recurrent_ppo_5x5")
        sweep_payload = {"skip_sweep": True, "selected_model": str(recurrent_5_model)}
    else:
        recurrent_5_model, recurrent_5_summary, sweep_payload = sweep_recurrent_5x5(args)

    run_cmd(
        [
            sys.executable,
            "train/curriculum_recurrent_ppo.py",
            "--source-model",
            str(recurrent_5_model),
            "--target-size",
            "10",
            "--target-obstacles",
            str(args.obs_10),
            "--target-max-steps",
            str(args.steps_10),
            "--view-radius",
            str(args.view_radius),
            "--timesteps",
            str(args.timesteps_curriculum_10),
            "--seed",
            str(args.seed),
            "--eval-episodes",
            str(args.eval_episodes),
            "--device",
            args.device,
        ]
    )
    recurrent_curriculum_model = find_model_by_prefix("recurrent_curriculum_10x10")

    run_cmd(
        [
            sys.executable,
            "evaluation/evaluate_generalization.py",
            "--model-path",
            str(recurrent_curriculum_model),
            "--algorithm",
            "recurrent_ppo",
            "--episodes",
            str(args.generalization_episodes),
            "--seed",
            str(args.seed),
            "--view-radius",
            str(args.view_radius),
            "--suffix",
            run_tag,
        ]
    )
    recurrent_generalization_summary = find_summary_by_prefix(
        f"{recurrent_curriculum_model.stem}_generalization_{run_tag}"
    )

    report_path = None
    if ppo_generalization_summary is not None:
        report_path = ROOT / "reports" / f"ppo_vs_lstm_{run_tag}.md"
        run_cmd(
            [
                sys.executable,
                "reports/compare_ppo_vs_lstm.py",
                "--ppo-summary",
                str(ppo_generalization_summary),
                "--lstm-summary",
                str(recurrent_generalization_summary),
                "--output",
                str(report_path),
            ]
        )

    payload = {
        "seed": args.seed,
        "view_radius": args.view_radius,
        "ppo_generalization_summary": str(ppo_generalization_summary) if ppo_generalization_summary else None,
        "recurrent_stage1_model": str(recurrent_5_model),
        "recurrent_stage1_summary": str(recurrent_5_summary),
        "recurrent_curriculum_model": str(recurrent_curriculum_model),
        "recurrent_generalization_summary": str(recurrent_generalization_summary),
        "report_path": str(report_path) if report_path else None,
        "sweep": sweep_payload,
    }
    out_path = ROOT / "reports" / f"pipeline_run_{run_tag}.json"
    save_json(out_path, payload)
    print("\nPipeline completed.")
    print(f"Run metadata: {out_path}")
    if report_path:
        print(f"Comparison report: {report_path}")


if __name__ == "__main__":
    main()

