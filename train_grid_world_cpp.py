#
# Usage examples:
#   python train_grid_world_cpp.py test 20 48
#   python train_grid_world_cpp.py run 10 12 --agent frontier
#   python train_grid_world_cpp.py train 5 3 200 500000
#   python train_grid_world_cpp.py test 10 12 --agent ppo --model data/model.zip
#

import argparse
import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from utils.cpp_coverage_agent import LocalCoverageAgent
from utils.cpp_q_learning_agent import QLearningCoverageAgent, summarize_training_log


def register_env():
    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP-v0",
            entry_point=GridWorldCPPEnv,
        )
    except gym.error.Error:
        pass


def default_max_steps(size):
    if size <= 5:
        return 200
    if size <= 10:
        return 400
    return max(800, 2 * size * size)


def print_action(action):
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(int(action), "unknown")


def resolve_model_path(model):
    if model is None:
        return None

    path = Path(model)
    if path.exists():
        return path

    data_path = Path("data") / model
    if data_path.exists():
        return data_path

    if path.suffix != ".zip":
        zipped = Path(f"{model}.zip")
        data_zipped = Path("data") / f"{model}.zip"
        if zipped.exists():
            return zipped
        if data_zipped.exists():
            return data_zipped

    return path


def default_tabular_model_path(args, agent_name):
    candidates = sorted(
        Path("data").glob(
            f"{agent_name}_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_"
            f"view{args.view_size}_*.zip"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def default_ppo_model_path(args):
    patterns = [
        f"recurrent_ppo_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_view{args.view_size}_*.zip",
        f"ppo_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_view{args.view_size}_*.zip",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(Path("data").glob(pattern))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def make_env(args, render_mode):
    return gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=args.dim,
        obs_quantity=args.obstacles,
        max_steps=args.max_steps or default_max_steps(args.dim),
        view_size=args.view_size,
        render_mode=render_mode,
    )


def make_single_env(args, render_mode, seed_offset=0):
    def _init():
        env = gym.make(
            "gymnasium_env/GridWorldCPP-v0",
            size=args.dim,
            obs_quantity=args.obstacles,
            max_steps=args.max_steps or default_max_steps(args.dim),
            view_size=args.view_size,
            render_mode=render_mode,
        )
        if args.seed is not None:
            env.reset(seed=args.seed + seed_offset)
        return env

    return _init


def make_training_env(args, log_dir=None):
    n_envs = max(1, int(args.n_envs))
    env = DummyVecEnv([make_single_env(args, "rgb_array", idx) for idx in range(n_envs)])
    monitor_file = None if log_dir is None else str(Path(log_dir) / "monitor.csv")
    return VecMonitor(env, filename=monitor_file, info_keywords=("coverage",))


class PPOPolicy:
    def __init__(self, model, recurrent=False, deterministic=True):
        self.model = model
        self.recurrent = recurrent
        self.deterministic = deterministic
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)

    def reset(self, size=None):
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)

    def act(self, observation):
        if self.recurrent:
            action, self.lstm_states = self.model.predict(
                observation,
                state=self.lstm_states,
                episode_start=self.episode_start,
                deterministic=self.deterministic,
            )
            self.episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return int(action)

    def save(self, path, metrics=None):
        self.model.save(path)
        if metrics:
            metadata_path = Path(path).with_suffix(".json")
            metadata_path.write_text(
                json.dumps(
                    {
                        "description": "Evaluation metadata for PPO/Recurrent PPO model.",
                        "summary": metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return Path(path)


def load_ppo(model_path, env=None, deterministic=True):
    resolved_path = resolve_model_path(model_path)
    try:
        from sb3_contrib import RecurrentPPO

        return PPOPolicy(
            RecurrentPPO.load(str(resolved_path), env=env, device="cpu"),
            recurrent=True,
            deterministic=deterministic,
        )
    except Exception:
        from stable_baselines3 import PPO

        return PPOPolicy(
            PPO.load(str(resolved_path), env=env, device="cpu"),
            recurrent=False,
            deterministic=deterministic,
        )


def train_ppo(args):
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure

    check_env(make_env(args, render_mode="rgb_array"))
    model_prefix = "recurrent_ppo" if args.ppo_kind == "recurrent" else "ppo"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("log") / (
        f"{model_prefix}_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_"
        f"view{args.view_size}_steps{args.total_timesteps}_{timestamp}"
    )
    if args.model_output:
        model_path = Path(args.model_output)
    else:
        model_path = Path("data") / (
            f"{model_prefix}_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_"
            f"view{args.view_size}_steps{args.total_timesteps}.zip"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_training_env(args, log_dir=log_dir)

    if args.ppo_kind == "recurrent":
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            verbose=1,
            ent_coef=args.entropy_coef,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            device="cpu",
        )
    else:
        from stable_baselines3 import PPO

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            ent_coef=args.entropy_coef,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            device="cpu",
        )

    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    callback = None
    if args.checkpoint_freq > 0:
        checkpoint_dir = log_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // max(1, int(args.n_envs))),
            save_path=str(checkpoint_dir),
            name_prefix=model_prefix,
        )
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(model_path)
    env.close()

    print(f"{model_prefix} model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")


def curriculum_ppo(args):
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure

    if not args.model:
        raise ValueError("curriculum mode needs --model with a pretrained PPO zip")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("log") / (
        f"ppo_cpp_continue_{args.dim}x{args.dim}_obs{args.obstacles}_"
        f"view{args.view_size}_steps{args.total_timesteps}_{timestamp}"
    )
    if args.model_output:
        model_path = Path(args.model_output)
    else:
        model_path = Path("data") / (
            f"recurrent_ppo_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_"
            f"view{args.view_size}_steps_continue{args.total_timesteps}.zip"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    env = make_training_env(args, log_dir=log_dir)
    model = load_ppo(args.model, env=env).model
    model.ent_coef = args.entropy_coef

    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    callback = None
    if args.checkpoint_freq > 0:
        checkpoint_dir = log_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // max(1, int(args.n_envs))),
            save_path=str(checkpoint_dir),
            name_prefix="recurrent_ppo_continue",
        )
    model.learn(total_timesteps=args.total_timesteps, reset_num_timesteps=False, callback=callback)
    model.save(model_path)
    env.close()

    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")


def train_tabular(args):
    env = make_env(args, render_mode="rgb_array")
    agent = QLearningCoverageAgent(
        size=args.dim,
        algorithm=args.algorithm,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )
    training_log = []

    for episode in range(args.episodes):
        agent.reset(size=args.dim)
        seed = None if args.seed is None else args.seed + episode
        observation, _ = env.reset(seed=seed)
        state = agent.state_from_observation(observation)
        action = agent.select_action(state, training=True)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        info = {}

        while not terminated and not truncated:
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.last_action = action
            next_state = agent.state_from_observation(next_observation)

            if args.algorithm == "sarsa":
                next_action = agent.select_action(next_state, training=True)
                agent.update_sarsa(state, action, reward, next_state, next_action, done)
                action = next_action
            else:
                agent.update_qlearning(state, action, reward, next_state, done)
                action = agent.select_action(next_state, training=True)

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        agent.training_episodes += 1
        entry = {
            "episode": episode + 1,
            "reward": float(total_reward),
            "coverage": float(info.get("coverage", 0.0)),
            "steps": int(steps),
            "epsilon": float(agent.epsilon),
        }
        training_log.append(entry)

        if args.print_every and ((episode + 1) % args.print_every == 0 or episode == 0):
            print(
                f"Episode {episode + 1:5d}/{args.episodes}: "
                f"coverage={entry['coverage']:6.1%}, reward={entry['reward']:8.2f}, "
                f"steps={entry['steps']:4d}, epsilon={entry['epsilon']:.3f}, "
                f"states={len(agent.q_table)}"
            )

    env.close()

    training_summary = summarize_training_log(training_log)
    output_path = resolve_tabular_model_output(args)
    saved_path = agent.save(output_path, metrics=training_summary, training_log=training_log)
    save_training_log_csv(args, training_log, saved_path)
    print(f"{args.algorithm} model saved to {saved_path}")
    print(f"Learned states: {len(agent.q_table)}")
    print(f"Last-window training coverage: {100 * training_summary['last_window_average_coverage']:.2f}%")


def resolve_tabular_model_output(args):
    if args.model_output:
        return Path(args.model_output)

    seed_label = "random" if args.seed is None else str(args.seed)
    return Path("data") / (
        f"{args.algorithm}_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_"
        f"view{args.view_size}_seed{seed_label}_episodes{args.episodes}.zip"
    )


def save_training_log_csv(args, training_log, model_path):
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = log_dir / f"{Path(model_path).stem}_training.csv"
    lines = ["episode,reward,coverage,steps,epsilon"]
    for entry in training_log:
        lines.append(
            f"{entry['episode']},{entry['reward']:.6f},{entry['coverage']:.6f},"
            f"{entry['steps']},{entry['epsilon']:.6f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Training log saved to {output_path}")


def make_policy(args, env):
    if args.agent == "frontier":
        if args.model:
            return LocalCoverageAgent.load(resolve_model_path(args.model))
        return LocalCoverageAgent(size=args.dim)
    if args.agent in {"qlearning", "sarsa"}:
        model_path = resolve_model_path(args.model) if args.model else default_tabular_model_path(args, args.agent)
        if model_path is None:
            raise ValueError(f"{args.agent} test/run needs --model with a trained Q-table zip")
        args.model = str(model_path)
        return QLearningCoverageAgent.load(model_path)
    if args.agent == "ppo":
        model_path = resolve_model_path(args.model) if args.model else default_ppo_model_path(args)
        if model_path is None:
            raise ValueError("PPO test/run needs --model with a trained PPO zip")
        args.model = str(model_path)
        return load_ppo(model_path, deterministic=not args.ppo_stochastic)
    return None


def select_action(args, policy, env, observation):
    if args.agent == "frontier":
        return int(policy.act(observation))
    if args.agent in {"qlearning", "sarsa"}:
        return int(policy.act(observation, training=False))
    if args.agent == "ppo":
        return policy.act(observation)
    return int(env.action_space.sample())


def evaluate(args, render_mode):
    env = make_env(args, render_mode=render_mode)
    policy = make_policy(args, env)

    full_coverage_count = 0
    good_coverage_count = 0
    coverages = []
    steps_list = []

    for episode in range(args.episodes):
        if args.agent in {"frontier", "qlearning", "sarsa", "ppo"}:
            policy.reset(size=args.dim)

        seed = None if args.seed is None else args.seed + episode
        observation, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not terminated and not truncated:
            action = select_action(args, policy, env, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if args.command == "run":
                print(
                    f"Step {steps:4d} | Action: {print_action(action):5s} | "
                    f"Reward: {reward:6.2f} | Coverage: {info['coverage']:6.1%}"
                )

        coverage = float(info["coverage"])
        coverages.append(coverage)
        steps_list.append(steps)
        full_coverage_count += int(terminated and not truncated)
        good_coverage_count += int(coverage >= args.coverage_target)

        print(
            f"Episode {episode + 1:3d}: coverage={coverage:6.1%}, "
            f"steps={steps:4d}, reward={total_reward:7.2f}, "
            f"full_coverage={'yes' if terminated and not truncated else 'no'}"
        )

    env.close()

    coverages_np = np.array(coverages)
    steps_np = np.array(steps_list)
    summary = {
        "agent": args.agent,
        "model": args.model,
        "grid_size": args.dim,
        "obstacles": args.obstacles,
        "max_steps": args.max_steps or default_max_steps(args.dim),
        "view_size": args.view_size,
        "episodes": args.episodes,
        "seed": args.seed,
        "coverage_target": args.coverage_target,
        "full_coverage_count": full_coverage_count,
        "good_coverage_count": good_coverage_count,
        "full_coverage_rate": full_coverage_count / args.episodes,
        "good_coverage_rate": good_coverage_count / args.episodes,
        "average_coverage": float(coverages_np.mean()),
        "coverage_std": float(coverages_np.std()),
        "min_coverage": float(coverages_np.min()),
        "max_coverage": float(coverages_np.max()),
        "average_steps": float(steps_np.mean()),
        "steps_std": float(steps_np.std()),
        "min_steps": int(steps_np.min()),
        "max_steps_observed": int(steps_np.max()),
    }

    print("\n--- Test Finished ---")
    print(f"Agent: {summary['agent']}")
    print(f"Grid: {summary['grid_size']}x{summary['grid_size']}, obstacles: {summary['obstacles']}, max_steps: {summary['max_steps']}")
    print(f"Full Coverage Rate: {100 * summary['full_coverage_rate']:.2f}% ({full_coverage_count}/{args.episodes})")
    print(f"Coverage >= {100 * args.coverage_target:.0f}% Rate: {100 * summary['good_coverage_rate']:.2f}% ({good_coverage_count}/{args.episodes})")
    print(
        f"Average Coverage: {100 * summary['average_coverage']:.2f}% "
        f"Std: {100 * summary['coverage_std']:.2f}% "
        f"Min: {100 * summary['min_coverage']:.2f}% "
        f"Max: {100 * summary['max_coverage']:.2f}%"
    )
    print(
        f"Average Steps: {summary['average_steps']:.1f} "
        f"Std: {summary['steps_std']:.1f} "
        f"Min: {summary['min_steps']} "
        f"Max: {summary['max_steps_observed']}"
    )

    if args.save_results:
        save_results(args, summary)

    if args.save_model:
        save_evaluated_model(args, policy, summary)

    return summary


def save_results(args, summary):
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.results_path:
        output_path = Path(args.results_path)
    else:
        seed_label = "random" if args.seed is None else str(args.seed)
        output_path = output_dir / (
            f"{args.agent}_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_"
            f"view{args.view_size}_seed{seed_label}_episodes{args.episodes}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "GridWorld CPP evaluation artifact with model configuration and metrics.",
        "summary": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved to {output_path}")


def save_evaluated_model(args, policy, summary):
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_output:
        output_path = Path(args.model_output)
    else:
        seed_label = "random" if args.seed is None else str(args.seed)
        output_path = output_dir / (
            f"{args.agent}_cpp_{args.dim}x{args.dim}_obs{args.obstacles}_"
            f"view{args.view_size}_seed{seed_label}_episodes{args.episodes}.zip"
        )

    saved_path = policy.save(output_path, metrics=summary)
    print(f"{args.agent} model saved to {saved_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train/test agents for GridWorld CPP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["train", "curriculum"]:
        subparser = subparsers.add_parser(command)
        add_env_args(subparser, needs_timesteps=True)
        add_ppo_args(subparser)
        if command == "curriculum":
            subparser.add_argument("--model", type=str, required=True)

    qtrain = subparsers.add_parser("qtrain")
    add_tabular_train_args(qtrain)

    for command in ["test", "run"]:
        subparser = subparsers.add_parser(command)
        add_env_args(subparser, needs_timesteps=False)
        subparser.add_argument("--agent", choices=["ppo", "qlearning", "sarsa", "frontier", "random"], default="ppo")
        subparser.add_argument("--model", type=str, default=None)
        subparser.add_argument("--episodes", type=int, default=100 if command == "test" else 1)
        subparser.add_argument("--seed", type=int, default=123)
        subparser.add_argument("--coverage-target", type=float, default=0.90)
        subparser.add_argument("--save-results", action="store_true")
        subparser.add_argument("--results-path", type=str, default=None)
        subparser.add_argument("--save-model", action="store_true")
        subparser.add_argument("--model-output", type=str, default=None)
        subparser.add_argument("--ppo-stochastic", action="store_true")

    return parser


def add_env_args(parser, needs_timesteps):
    parser.add_argument("dim", type=int, help="Grid size, e.g. 5, 10 or 20")
    parser.add_argument("obstacles", type=int, help="Number of obstacles")
    parser.add_argument("max_steps", type=int, nargs="?", default=None)
    parser.add_argument("total_timesteps", type=int, nargs="?", default=None)
    parser.add_argument("--view-size", type=int, default=5, help="Odd local observation size. The project uses 5.")

    if needs_timesteps:
        parser.set_defaults(needs_timesteps=True)
    else:
        parser.set_defaults(needs_timesteps=False)


def add_ppo_args(parser):
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ppo-kind", choices=["recurrent", "vanilla"], default="recurrent")
    parser.add_argument("--model-output", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint-freq", type=int, default=100000)


def add_tabular_train_args(parser):
    parser.add_argument("dim", type=int, help="Grid size, e.g. 5, 10 or 20")
    parser.add_argument("obstacles", type=int, help="Number of obstacles")
    parser.add_argument("max_steps", type=int, nargs="?", default=None)
    parser.add_argument("episodes", type=int, nargs="?", default=800)
    parser.add_argument("--algorithm", choices=["qlearning", "sarsa"], default="qlearning")
    parser.add_argument("--view-size", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.80)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.992)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--model-output", type=str, default=None)
    parser.add_argument("--print-every", type=int, default=50)


def validate_args(args):
    if args.command in ["train", "curriculum"]:
        if args.max_steps is None or args.total_timesteps is None:
            raise ValueError(
                f"{args.command} usage: python train_grid_world_cpp.py {args.command} "
                "dim obstacles max_steps total_timesteps"
            )

    if args.max_steps is None:
        args.max_steps = default_max_steps(args.dim)


def main():
    args = build_parser().parse_args()
    validate_args(args)
    register_env()

    if args.command == "train":
        train_ppo(args)
    elif args.command == "curriculum":
        curriculum_ppo(args)
    elif args.command == "qtrain":
        train_tabular(args)
    elif args.command == "test":
        evaluate(args, render_mode="rgb_array")
    elif args.command == "run":
        evaluate(args, render_mode="human")


if __name__ == "__main__":
    main()
