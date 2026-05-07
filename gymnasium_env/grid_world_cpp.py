from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym

#
# Coverage Path Planning (CPP) environment based on GridWorld with obstacles.
#
# The agent must visit as many free cells as possible while avoiding obstacles.
# The reward function is designed to encourage exploration of new cells and
# discourage revisiting already-visited cells.
#
# Reward function (inspired by deep RL approaches to patrolling/coverage problems):
#   - +1.0 for visiting a new (unvisited) cell
#   - -0.08 for revisiting an already-visited cell
#   - -0.02 step penalty to encourage efficiency
#   - +20.0 bonus for achieving full coverage (all free cells visited)
#   - -5.0 penalty when max steps reached without full coverage
#
# The observation space includes:
#   - Agent's (x, y) location (normalized)
#   - Coverage ratio (proportion of free cells visited)
#   - A local odd-sized matrix of neighboring cells centered on the agent,
#     where the center is the agent's position and each cell is:
#       0 = free (not yet visited), 1 = obstacle or wall (including out-of-bounds),
#       2 = already visited position.
#     Cells outside the grid boundaries are treated as walls (1).
#
# The episode ends when all free cells are visited or max steps is reached.
#

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        size: int = 5,
        obs_quantity: int = 3,
        max_steps: int = 200,
        view_size: int = 5,
        ensure_connected: bool = True,
    ):
        if view_size < 3 or view_size % 2 == 0:
            raise ValueError("view_size must be an odd integer >= 3")
        if obs_quantity >= size * size:
            raise ValueError("obs_quantity must leave at least one free cell")

        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.view_size = view_size
        self.ensure_connected = ensure_connected
        self.obstacles_locations = []
        self.obstacles = set()
        self.count_steps = 0
        self.max_steps = max_steps

        # Track visited cells
        self.visited = set()

        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.zeros((self.view_size, self.view_size), dtype=int)

        # Observation: agent info (x, y, coverage) and local centered map.
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((self.view_size, self.view_size), dtype=np.float32),
                high=np.full((self.view_size, self.view_size), 2.0, dtype=np.float32),
                dtype=np.float32
            ),
        })

        # 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    @property
    def total_free_cells(self):
        return self.size * self.size - len(self.obstacles)

    @property
    def coverage_ratio(self):
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _get_obs(self):
        return {
            "agent": np.array([
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
            ], dtype=np.float32),
            "neighbors": self._neighbors.astype(np.float32),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
            "view_size": self.view_size,
        }

    def _is_obstacle(self, location):
        return tuple(location) in self.obstacles

    def _random_cell(self):
        return tuple(self.np_random.integers(0, self.size, size=2, dtype=int))

    def _is_free_space_connected(self, obstacles):
        start = tuple(self._agent_location)
        if start in obstacles:
            return False

        total_free = self.size * self.size - len(obstacles)
        queue = deque([start])
        seen = {start}

        while queue:
            x, y = queue.popleft()
            for dx, dy in self._action_to_direction.values():
                nx = x + int(dx)
                ny = y + int(dy)
                nxt = (nx, ny)
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and nxt not in obstacles
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    queue.append(nxt)

        return len(seen) == total_free

    def _place_obstacles(self):
        target = min(self.obs_quantity, self.size * self.size - 1)
        obstacles = set()

        # Rejection sampling is fast for the obstacle densities used in the APS
        # and prevents impossible episodes with disconnected free regions.
        attempts = 0
        max_attempts = max(500, target * self.size * self.size * 4)
        while len(obstacles) < target and attempts < max_attempts:
            attempts += 1
            candidate = self._random_cell()
            if candidate == tuple(self._agent_location) or candidate in obstacles:
                continue

            proposed = obstacles | {candidate}
            if not self.ensure_connected or self._is_free_space_connected(proposed):
                obstacles = proposed

        self.obstacles = obstacles
        self.obstacles_locations = [
            np.array(location, dtype=int) for location in sorted(self.obstacles)
        ]

    def set_neighbors(self):
        # Create a local matrix centered on the agent's location.
        # Row i corresponds to agent_y + (i-radius), col j to agent_x + (j-radius).
        # 0 = free (not yet visited), 1 = obstacle or wall (out-of-bounds), 2 = already visited.
        radius = self.view_size // 2
        matrix = np.zeros((self.view_size, self.view_size), dtype=int)
        for i in range(self.view_size):
            for j in range(self.view_size):
                nx = self._agent_location[0] + (j - radius)
                ny = self._agent_location[1] + (i - radius)
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    matrix[i][j] = 1
                elif (nx, ny) in self.obstacles:
                    matrix[i][j] = 1
                elif (nx, ny) in self.visited:
                    matrix[i][j] = 2
        self._neighbors = matrix

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self.visited = set()

        # Place agent randomly
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place obstacles
        self._place_obstacles()

        # Mark starting position as visited
        self.visited.add(tuple(self._agent_location))

        self.set_neighbors()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[int(action)]
        old_location = self._agent_location.copy()
        target_location = self._agent_location + direction

        # Move only when the target is inside the grid and not blocked.
        if (
            0 <= target_location[0] < self.size
            and 0 <= target_location[1] < self.size
            and not self._is_obstacle(target_location)
        ):
            self._agent_location = target_location

        self.count_steps += 1

        # --- CPP Reward Function ---
        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)

        # Base step penalty
        reward = -0.02

        if stayed_in_place:
            # Hitting wall or obstacle
            reward -= 0.75
        elif is_new_cell:
            # Reward for exploring new cell
            reward += 1.0
            self.visited.add(current_pos)
        else:
            # Penalty for revisiting
            reward -= 0.08

        self.set_neighbors()

        # Check if full coverage achieved
        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage

        if full_coverage:
            reward += 20.0

        # Truncation on max steps
        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0
        else:
            truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw visited cells in light green
        for cell in self.visited:
            cell_arr = np.array(cell)
            pygame.draw.rect(
                canvas,
                (144, 238, 144),  # light green
                pygame.Rect(
                    pix_square_size * cell_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw obstacles in black
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw agent as blue circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw coverage info text
        font = pygame.font.SysFont(None, 24)
        coverage_text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
            True, (0, 0, 0)
        )
        canvas.blit(coverage_text, (5, 5))

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
