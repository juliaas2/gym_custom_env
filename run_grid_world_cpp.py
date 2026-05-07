import argparse

import gymnasium as gym

from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from utils.cpp_coverage_agent import LocalCoverageAgent
from utils.cpp_q_learning_agent import QLearningCoverageAgent
from train_grid_world_cpp import load_ppo


def get_direction(action):
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local frontier CPP agent.")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--obstacles", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--view-size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--agent", choices=["ppo", "qlearning", "frontier"], default="ppo")
    parser.add_argument(
        "--model",
        type=str,
        default="data/recurrent_ppo_cpp_20x20_obs48_view5_steps250000.zip",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP-v0",
            entry_point=GridWorldCPPEnv,
        )
    except gym.error.Error:
        pass

    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        render_mode="human" if args.render else "rgb_array",
        size=args.size,
        obs_quantity=args.obstacles,
        max_steps=args.max_steps,
        view_size=args.view_size,
    )
    if args.agent == "ppo":
        agent = load_ppo(args.model)
    elif args.agent == "qlearning":
        agent = QLearningCoverageAgent.load(args.model)
    else:
        agent = LocalCoverageAgent(size=args.size)

    for episode in range(args.episodes):
        agent.reset(size=args.size)
        observation, info = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0

        print(f"\nEpisode {episode + 1}")
        print(f"Free cells to cover: {info['total_free_cells']}")

        while not terminated and not truncated:
            if args.agent == "ppo":
                action = agent.act(observation)
            elif args.agent == "qlearning":
                action = agent.act(observation, training=False)
            else:
                action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            print(
                f"Step {step:4d} | Action: {get_direction(action):5s} | "
                f"Reward: {reward:+6.2f} | Coverage: {info['coverage']:6.1%} | "
                f"Visited: {info['visited_cells']}/{info['total_free_cells']}"
            )

        print(
            f"Episode finished: steps={step}, reward={total_reward:.2f}, "
            f"coverage={info['coverage']:.1%}, full_coverage={'yes' if terminated else 'no'}"
        )

    env.close()


if __name__ == "__main__":
    main()
