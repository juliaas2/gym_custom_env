import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def latest_log_dir():
    dirs = [path for path in Path("log").glob("*") if path.is_dir()]
    if not dirs:
        raise FileNotFoundError("No log directories found.")
    return max(dirs, key=lambda path: path.stat().st_mtime)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = [line for line in file if not line.startswith("#")]
    return list(csv.DictReader(rows))


def as_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def moving_average(values, window):
    if not values:
        return []
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start:index + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def plot_monitor_curve(log_dir, output_path, window):
    monitor_path = Path(log_dir) / "monitor.csv"
    if not monitor_path.exists():
        raise FileNotFoundError(f"Could not find {monitor_path}")

    rows = read_csv(monitor_path)
    episodes = list(range(1, len(rows) + 1))
    rewards = [as_float(row, "r") for row in rows]
    lengths = [as_float(row, "l") for row in rows]
    coverages = [as_float(row, "coverage") for row in rows]

    rewards = [value for value in rewards if value is not None]
    lengths = [value for value in lengths if value is not None]
    coverages = [value for value in coverages if value is not None]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"CPP PPO learning curves\n{Path(log_dir).name}")

    axes[0].plot(episodes[:len(rewards)], rewards, alpha=0.25, label="episode reward")
    axes[0].plot(episodes[:len(rewards)], moving_average(rewards, window), label=f"MA{window}")
    axes[0].set_ylabel("reward")
    axes[0].legend()

    if coverages:
        axes[1].plot(episodes[:len(coverages)], [100 * value for value in coverages], alpha=0.25, label="coverage")
        axes[1].plot(
            episodes[:len(coverages)],
            [100 * value for value in moving_average(coverages, window)],
            label=f"MA{window}",
        )
        axes[1].axhline(90, color="red", linestyle="--", linewidth=1, label="90% target")
    axes[1].set_ylabel("coverage (%)")
    axes[1].legend()

    axes[2].plot(episodes[:len(lengths)], lengths, alpha=0.25, label="episode length")
    axes[2].plot(episodes[:len(lengths)], moving_average(lengths, window), label=f"MA{window}")
    axes[2].set_ylabel("steps")
    axes[2].set_xlabel("episode")
    axes[2].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_progress_curve(log_dir, output_path):
    progress_path = Path(log_dir) / "progress.csv"
    if not progress_path.exists():
        raise FileNotFoundError(f"Could not find {progress_path}")

    rows = read_csv(progress_path)
    steps = [as_float(row, "time/total_timesteps") for row in rows]
    rewards = [as_float(row, "rollout/ep_rew_mean") for row in rows]
    lengths = [as_float(row, "rollout/ep_len_mean") for row in rows]

    points = [(step, reward, length) for step, reward, length in zip(steps, rewards, lengths) if step is not None]
    if not points:
        raise ValueError("No plottable progress rows found.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"CPP PPO progress\n{Path(log_dir).name}")

    axes[0].plot([p[0] for p in points], [p[1] for p in points])
    axes[0].set_ylabel("mean reward")

    axes[1].plot([p[0] for p in points], [p[2] for p in points])
    axes[1].set_ylabel("mean steps")
    axes[1].set_xlabel("timesteps")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CPP learning curves from SB3 logs.")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory. Defaults to the newest directory in log/.")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path.")
    parser.add_argument("--window", type=int, default=25, help="Moving-average window in episodes.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else latest_log_dir()
    output_path = Path(args.output) if args.output else log_dir / "learning_curve.png"

    try:
        plot_monitor_curve(log_dir, output_path, args.window)
    except FileNotFoundError:
        plot_progress_curve(log_dir, output_path)

    print(f"Saved learning curve to {output_path}")


if __name__ == "__main__":
    main()
