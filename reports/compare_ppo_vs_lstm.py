from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def extract_size_metrics(payload: dict) -> dict[str, dict]:
    return payload.get("sizes", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text report comparing PPO and PPO-LSTM.")
    parser.add_argument("--ppo-summary", required=True, help="Generalization summary JSON from PPO.")
    parser.add_argument("--lstm-summary", required=True, help="Generalization summary JSON from RecurrentPPO.")
    parser.add_argument("--output", default="reports/ppo_vs_lstm_report.md")
    args = parser.parse_args()

    ppo = load_json(Path(args.ppo_summary))
    lstm = load_json(Path(args.lstm_summary))
    ppo_sizes = extract_size_metrics(ppo)
    lstm_sizes = extract_size_metrics(lstm)

    lines = [
        "# PPO vs RecurrentPPO (LSTM)",
        "",
        "## Full Coverage Rate",
        "",
    ]
    for size in ["5x5", "10x10", "20x20"]:
        ppo_rate = ppo_sizes.get(size, {}).get("full_coverage_rate_percent", 0.0)
        lstm_rate = lstm_sizes.get(size, {}).get("full_coverage_rate_percent", 0.0)
        lines.append(f"- {size}: PPO={ppo_rate:.2f}% | RecurrentPPO={lstm_rate:.2f}%")

    lines += ["", "## Average Coverage", ""]
    for size in ["5x5", "10x10", "20x20"]:
        ppo_cov = ppo_sizes.get(size, {}).get("average_coverage_percent", 0.0)
        lstm_cov = lstm_sizes.get(size, {}).get("average_coverage_percent", 0.0)
        lines.append(f"- {size}: PPO={ppo_cov:.2f}% | RecurrentPPO={lstm_cov:.2f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()

