import json
import random
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from utils.cpp_coverage_agent import LocalCoverageAgent


class QLearningCoverageAgent(LocalCoverageAgent):
    """Tabular Q-Learning/SARSA agent for the CPP environment.

    The observation is still local and centered on the agent. The agent keeps a
    belief map of cells already observed during the episode and uses that belief
    to build a compact discrete state for the Q-table.
    """

    def __init__(
        self,
        size=None,
        algorithm="qlearning",
        alpha=0.25,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.995,
        q_table=None,
        training_episodes=0,
    ):
        super().__init__(size=size)
        if algorithm not in {"qlearning", "sarsa"}:
            raise ValueError("algorithm must be 'qlearning' or 'sarsa'")

        self.algorithm = algorithm
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.q_table = q_table or {}
        self.training_episodes = int(training_episodes)
        self.last_action = 4

    def reset(self, size=None):
        super().reset(size=size)
        self.last_action = 4

    def state_from_observation(self, observation):
        self.update(observation)
        local_map = observation["neighbors"].astype(int).flatten()
        local_code = "".join(str(int(value)) for value in local_map)

        unvisited_mask = self._action_mask(self._is_unvisited_free)
        blocked_mask = self._blocked_mask()
        hint_action = self._frontier_hint_action()
        coverage_bucket = min(10, int(float(observation["agent"][2]) * 10))

        row_parity = self.position[1] % 2
        col_parity = self.position[0] % 2

        return "|".join(
            [
                local_code,
                str(unvisited_mask),
                str(blocked_mask),
                str(hint_action),
                str(coverage_bucket),
                str(row_parity),
                str(col_parity),
                str(self.last_action),
            ]
        )

    def act(self, observation, training=False):
        state = self.state_from_observation(observation)
        action = self.select_action(state, training=training)
        self.last_action = action
        return action

    def select_action(self, state, training=False):
        if training and random.random() < self.epsilon:
            return random.choice(self._valid_actions_from_state(state))

        q_values = self._q_values(state)
        valid_actions = self._valid_actions_from_state(state)
        best_value = max(q_values[action] for action in valid_actions)
        best_actions = [action for action in valid_actions if q_values[action] == best_value]

        hint_action = self._hint_from_state(state)
        if hint_action in best_actions:
            return hint_action
        return min(best_actions)

    def update_qlearning(self, state, action, reward, next_state, done):
        old_value = self._q_values(state)[action]
        next_max = 0.0 if done else max(self._q_values(next_state))
        target = reward + self.gamma * next_max
        self.q_table[state][action] = old_value + self.alpha * (target - old_value)

    def update_sarsa(self, state, action, reward, next_state, next_action, done):
        old_value = self._q_values(state)[action]
        next_value = 0.0 if done else self._q_values(next_state)[next_action]
        target = reward + self.gamma * next_value
        self.q_table[state][action] = old_value + self.alpha * (target - old_value)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = self._initial_q_values(state)
        return self.q_table[state]

    def _initial_q_values(self, state):
        values = [0.0, 0.0, 0.0, 0.0]
        unvisited_mask = self._unvisited_mask_from_state(state)
        blocked_mask = self._blocked_mask_from_state(state)
        hint_action = self._hint_from_state(state)

        for action in range(4):
            if blocked_mask & (1 << action):
                values[action] = -2.0
            elif unvisited_mask & (1 << action):
                values[action] = 0.75

        values[hint_action] += 1.0
        return values

    def _frontier_hint_action(self):
        direct_action = self._best_adjacent_unvisited_action()
        if direct_action is not None:
            return direct_action

        path = self._path_to_nearest(self._is_unvisited_free)
        if path:
            return self._action_for_next_cell(path[0])

        path = self._path_to_nearest(self._has_unknown_neighbor)
        if path:
            return self._action_for_next_cell(path[0])

        fallback = self._first_known_free_neighbor()
        if fallback is not None:
            return fallback
        return 0

    def _action_mask(self, predicate):
        mask = 0
        for action in range(4):
            if predicate(self._cell_after_action(action)):
                mask |= 1 << action
        return mask

    def _blocked_mask(self):
        mask = 0
        for action in range(4):
            cell = self._cell_after_action(action)
            if not self._inside(cell) or cell in self.blocked_cells:
                mask |= 1 << action
        return mask

    def _valid_actions_from_state(self, state):
        blocked_mask = self._blocked_mask_from_state(state)
        valid_actions = [action for action in range(4) if not (blocked_mask & (1 << action))]
        return valid_actions or [self._hint_from_state(state)]

    def _split_state(self, state):
        return state.split("|")

    def _unvisited_mask_from_state(self, state):
        return int(self._split_state(state)[1])

    def _blocked_mask_from_state(self, state):
        return int(self._split_state(state)[2])

    def _hint_from_state(self, state):
        return int(self._split_state(state)[3])

    def save(self, path, metrics=None, training_log=None):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": "TabularCPPAgent",
            "algorithm": self.algorithm,
            "size": self.size,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "training_episodes": self.training_episodes,
            "num_states": len(self.q_table),
            "q_table": self.q_table,
            "metrics": metrics or {},
            "training_log": training_log or [],
        }

        with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("model.json", json.dumps(payload, indent=2))

        return output_path

    @classmethod
    def load(cls, path):
        with ZipFile(Path(path), "r") as archive:
            payload = json.loads(archive.read("model.json").decode("utf-8"))

        return cls(
            size=payload.get("size"),
            algorithm=payload.get("algorithm", "qlearning"),
            alpha=payload.get("alpha", 0.25),
            gamma=payload.get("gamma", 0.98),
            epsilon=0.0,
            epsilon_min=payload.get("epsilon_min", 0.02),
            epsilon_decay=payload.get("epsilon_decay", 0.995),
            q_table={key: list(map(float, value)) for key, value in payload.get("q_table", {}).items()},
            training_episodes=payload.get("training_episodes", 0),
        )


def summarize_training_log(training_log):
    if not training_log:
        return {}

    coverages = np.array([entry["coverage"] for entry in training_log], dtype=float)
    rewards = np.array([entry["reward"] for entry in training_log], dtype=float)
    steps = np.array([entry["steps"] for entry in training_log], dtype=float)
    window = min(50, len(training_log))

    return {
        "training_episodes": len(training_log),
        "average_training_coverage": float(coverages.mean()),
        "last_window_average_coverage": float(coverages[-window:].mean()),
        "average_training_reward": float(rewards.mean()),
        "last_window_average_reward": float(rewards[-window:].mean()),
        "average_training_steps": float(steps.mean()),
        "last_window_average_steps": float(steps[-window:].mean()),
    }
