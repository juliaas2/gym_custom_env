import gymnasium as gym
from gymnasium_env.grid_world_obstacles import GridWorldRenderEnv
from stable_baselines3.common.env_checker import check_env

def get_direction(action):
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down"
    }.get(action, "unknown")

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldRenderEnv,
)

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human", size=5, obs_quantity=4)

(state, _) = env.reset()
print(f"Initial State: {state}")
done = False
truncated = False
steps = 1
while not done:
    action = env.action_space.sample()
    (next_state, reward, terminated, truncated, info) = env.step(action)
    print(f"Step: {steps} Action: {get_direction(action)}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
    done = terminated or truncated
    steps += 1
    print(f"Next State: {next_state}")

env.close()
