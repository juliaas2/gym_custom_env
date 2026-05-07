from collections import deque
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


class LocalCoverageAgent:
    """Online CPP agent that only uses the centered local observation plus memory.

    The policy is deliberately simple: keep a map of cells observed so far, move
    to adjacent unvisited cells whenever possible, and otherwise use BFS through
    known free cells to reach the closest discovered frontier.
    """

    ACTION_TO_DELTA = {
        0: (1, 0),    # right
        1: (0, -1),   # up
        2: (-1, 0),   # left
        3: (0, 1),    # down
    }
    DELTA_TO_ACTION = {delta: action for action, delta in ACTION_TO_DELTA.items()}

    def __init__(self, size=None, unknown_weight=3.0, unvisited_weight=1.0):
        self.size = size
        self.unknown_weight = float(unknown_weight)
        self.unvisited_weight = float(unvisited_weight)
        self.position = None
        self.free_cells = set()
        self.blocked_cells = set()
        self.visited_cells = set()

    def reset(self, size=None):
        if size is not None:
            self.size = size
        self.position = None
        self.free_cells = set()
        self.blocked_cells = set()
        self.visited_cells = set()

    def act(self, observation):
        self.update(observation)

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

    def update(self, observation):
        agent_vector = observation["agent"]
        local_map = observation["neighbors"]
        size = self.size or self._infer_size_from_local_map(local_map)

        # The environment exposes the current position, but the map itself is
        # only the centered local window. The agent never receives the full map.
        x = min(size - 1, max(0, int(round(float(agent_vector[0]) * size))))
        y = min(size - 1, max(0, int(round(float(agent_vector[1]) * size))))
        self.position = (x, y)
        self.visited_cells.add(self.position)
        self.free_cells.add(self.position)

        radius = local_map.shape[0] // 2
        for row in range(local_map.shape[0]):
            for col in range(local_map.shape[1]):
                cell = (x + col - radius, y + row - radius)
                value = int(local_map[row][col])
                if not self._inside(cell):
                    self.blocked_cells.add(cell)
                elif value == 1:
                    self.blocked_cells.add(cell)
                    self.free_cells.discard(cell)
                else:
                    self.free_cells.add(cell)
                    self.blocked_cells.discard(cell)
                    if value == 2:
                        self.visited_cells.add(cell)

    def _infer_size_from_local_map(self, local_map):
        if self.size is None:
            raise ValueError("LocalCoverageAgent needs the grid size")
        return self.size

    def _inside(self, cell):
        if self.size is None:
            return True
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_passable(self, cell):
        return self._inside(cell) and cell in self.free_cells and cell not in self.blocked_cells

    def _is_unvisited_free(self, cell):
        return self._is_passable(cell) and cell not in self.visited_cells

    def _has_unknown_neighbor(self, cell):
        if not self._is_passable(cell):
            return False
        for neighbor in self._neighbors(cell):
            if self._inside(neighbor) and neighbor not in self.free_cells and neighbor not in self.blocked_cells:
                return True
        return False

    def _neighbors(self, cell):
        x, y = cell
        for dx, dy in self.ACTION_TO_DELTA.values():
            yield (x + dx, y + dy)

    def _ordered_actions(self, position=None):
        x, y = position or self.position
        if y % 2 == 0:
            preferred = [0, 3, 2, 1]  # right, down, left, up
        else:
            preferred = [2, 3, 0, 1]  # left, down, right, up

        # Near the bottom, prefer climbing back up so the agent clears cells
        # above the initial row without wasting the whole step budget.
        if self.size is not None and y >= self.size - 1:
            preferred = [2, 1, 0, 3] if y % 2 == 0 else [0, 1, 2, 3]
        return preferred

    def _best_adjacent_unvisited_action(self):
        candidates = []
        for priority, action in enumerate(self._ordered_actions()):
            next_cell = self._cell_after_action(action)
            if self._is_unvisited_free(next_cell):
                candidates.append((self._exploration_score(next_cell), priority, action))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    def _exploration_score(self, cell):
        unknown = 0
        unvisited = 0
        for neighbor in self._neighbors(cell):
            if not self._inside(neighbor) or neighbor in self.blocked_cells:
                continue
            if neighbor not in self.free_cells:
                unknown += 1
            elif neighbor not in self.visited_cells:
                unvisited += 1
        return (unknown * self.unknown_weight) + (unvisited * self.unvisited_weight)

    def _path_to_nearest(self, predicate):
        queue = deque([self.position])
        parent = {self.position: None}

        while queue:
            current = queue.popleft()
            if current != self.position and predicate(current):
                return self._reconstruct_path(parent, current)

            for action in self._ordered_actions(current):
                neighbor = self._cell_after_action(action, current)
                if neighbor in parent or not self._is_passable(neighbor):
                    continue
                parent[neighbor] = current
                queue.append(neighbor)

        return []

    def _reconstruct_path(self, parent, target):
        path = []
        current = target
        while parent[current] is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path

    def _action_for_next_cell(self, next_cell):
        dx = next_cell[0] - self.position[0]
        dy = next_cell[1] - self.position[1]
        return self.DELTA_TO_ACTION[(dx, dy)]

    def _cell_after_action(self, action, position=None):
        x, y = position or self.position
        dx, dy = self.ACTION_TO_DELTA[action]
        return (x + dx, y + dy)

    def _first_known_free_neighbor(self):
        for action in self._ordered_actions():
            if self._is_passable(self._cell_after_action(action)):
                return action
        return None

    def to_dict(self, metrics=None):
        return {
            "model_type": "LocalCoverageAgent",
            "description": "Deterministic frontier-coverage policy for GridWorld CPP.",
            "size": self.size,
            "weights": {
                "unknown_weight": self.unknown_weight,
                "unvisited_weight": self.unvisited_weight,
            },
            "action_to_delta": self.ACTION_TO_DELTA,
            "metrics": metrics or {},
        }

    def save(self, path, metrics=None):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(metrics=metrics)

        with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("model.json", json.dumps(payload, indent=2))

        return output_path

    @classmethod
    def load(cls, path):
        with ZipFile(Path(path), "r") as archive:
            payload = json.loads(archive.read("model.json").decode("utf-8"))

        weights = payload.get("weights", {})
        return cls(
            size=payload.get("size"),
            unknown_weight=weights.get("unknown_weight", 3.0),
            unvisited_weight=weights.get("unvisited_weight", 1.0),
        )
