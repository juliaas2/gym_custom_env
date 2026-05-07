from typing import Optional
import numpy as np
import gymnasium as gym

#
# This code is based on the example available at:
# https://gymnasium.farama.org/introduction/create_custom_env/
#
# The example above was adapted to create a 3D grid environment.
# Note: Rendering requires tkinter (python3-tk) to be installed.
#

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, size: int = 5, max_steps: int = 100):
        # The size of the square grid
        self.size = size
        self.count_steps = 0
        self.max_steps = max_steps

        # Rendering setup
        self.render_mode = render_mode
        self.fig = None
        self.ax = None

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^3
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(3,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(3,), dtype=int),
            }
        )

        # We have 6 actions, corresponding to "right", "up", "left", "down", "forward", "backward"
        self.action_space = gym.spaces.Discrete(6)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0, 0]),  # right
            1: np.array([0, 1, 0]),  # up
            2: np.array([-1, 0, 0]),  # left
            3: np.array([0, -1, 0]),  # down
            4: np.array([0, 0, 1]),  # forward
            5: np.array([0, 0, -1]),  # backward
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "size": self.size
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.count_steps = 0

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=3, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=3, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3,4,5}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        reward = 0
        self.count_steps += 1

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        if terminated:
            reward = 1.0

        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward = -1.0
        else:
            truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Warning: matplotlib or tkinter not available for rendering.")
            return

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.show(block=False)
            try:
                self.fig.canvas.manager.window.lift()
            except:
                try:
                    self.fig.canvas.manager.window.raise_()
                except:
                    pass

        self.ax.clear()

        # Set axis labels and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([-0.5, self.size - 0.5])
        self.ax.set_ylim([-0.5, self.size - 0.5])
        self.ax.set_zlim([-0.5, self.size - 0.5])

        # Draw grid lines
        for i in range(self.size):
            for j in range(self.size):
                self.ax.plot([i, i], [j, j], [0, self.size - 1], 'gray', alpha=0.2)
                self.ax.plot([i, i], [0, self.size - 1], [j, j], 'gray', alpha=0.2)
                self.ax.plot([0, self.size - 1], [i, i], [j, j], 'gray', alpha=0.2)

        # Draw grid boundaries
        vertices = np.array([
            [0, 0, 0], [self.size - 1, 0, 0], [self.size - 1, self.size - 1, 0], [0, self.size - 1, 0],
            [0, 0, self.size - 1], [self.size - 1, 0, self.size - 1],
            [self.size - 1, self.size - 1, self.size - 1], [0, self.size - 1, self.size - 1]
        ])
        edges = [
            [vertices[0], vertices[1]], [vertices[1], vertices[2]],
            [vertices[2], vertices[3]], [vertices[3], vertices[0]],
            [vertices[4], vertices[5]], [vertices[5], vertices[6]],
            [vertices[6], vertices[7]], [vertices[7], vertices[4]],
            [vertices[0], vertices[4]], [vertices[1], vertices[5]],
            [vertices[2], vertices[6]], [vertices[3], vertices[7]]
        ]
        for edge in edges:
            self.ax.plot3D(*zip(*edge), color='black', linewidth=2)

        # Draw agent (blue sphere)
        self.ax.scatter(self._agent_location[0], self._agent_location[1],
                        self._agent_location[2], color='blue', s=200, label='Agent')

        # Draw target (red star)
        self.ax.scatter(self._target_location[0], self._target_location[1],
                        self._target_location[2], color='red', marker='*', s=200, label='Target')

        # Add legend
        self.ax.legend()

        # Set title with current positions
        self.ax.set_title(f'Agent: {tuple(self._agent_location)}, Target: {tuple(self._target_location)}')

        # Adjust the view angle for better visibility
        self.ax.view_init(elev=30, azim=45)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1)

    def close(self):
        if self.fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.fig)
            except:
                pass
            self.fig = None
            self.ax = None
