"""Grid World environment for Engram agents."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

# Cell types
EMPTY = 0
WALL = 1
REWARD = 2
HAZARD = 3

# Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

CELL_CHARS = {EMPTY: ".", WALL: "#", REWARD: "$", HAZARD: "!"}


class GridWorldEnv:
    """A simple grid world environment for testing continual learning.

    The agent navigates a grid with walls, rewards, and hazards.
    The observation includes the agent's position, target position,
    and the cell types in the 4 cardinal directions.

    Args:
        size: Grid dimensions (size x size)
        num_walls: Number of wall cells to place
        num_hazards: Number of hazard cells
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        size: int = 12,
        num_walls: int = 15,
        num_hazards: int = 5,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.num_walls = num_walls
        self.num_hazards = num_hazards
        self.rng = random.Random(seed)

        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.agent_x = 1
        self.agent_y = 1
        self.target_x = size - 2
        self.target_y = size - 2
        self.steps = 0
        self.max_steps = size * size * 2
        self.total_reward = 0.0

        self._build_grid()

    def _build_grid(self) -> None:
        """Generate a random grid layout."""
        self.grid.fill(EMPTY)

        # Border walls
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        # Random walls
        placed = 0
        while placed < self.num_walls:
            x = self.rng.randint(1, self.size - 2)
            y = self.rng.randint(1, self.size - 2)
            if (x, y) == (self.agent_x, self.agent_y):
                continue
            if (x, y) == (self.target_x, self.target_y):
                continue
            if self.grid[y, x] == EMPTY:
                self.grid[y, x] = WALL
                placed += 1

        # Random hazards
        placed = 0
        while placed < self.num_hazards:
            x = self.rng.randint(1, self.size - 2)
            y = self.rng.randint(1, self.size - 2)
            if (x, y) == (self.agent_x, self.agent_y):
                continue
            if (x, y) == (self.target_x, self.target_y):
                continue
            if self.grid[y, x] == EMPTY:
                self.grid[y, x] = HAZARD
                placed += 1

        # Place target reward
        self.grid[self.target_y, self.target_x] = REWARD

    def reset(self, seed: Optional[int] = None) -> list[float]:
        """Reset the environment."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.agent_x = 1
        self.agent_y = 1
        self.steps = 0
        self.total_reward = 0.0
        self._build_grid()
        return self._get_observation()

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        """Take an action and return (observation, reward, done, info)."""
        dx, dy = ACTION_DELTAS[action % 4]
        nx = self.agent_x + dx
        ny = self.agent_y + dy

        reward = -0.01  # small step cost
        done = False

        if 0 <= nx < self.size and 0 <= ny < self.size:
            cell = self.grid[ny, nx]
            if cell == WALL:
                reward = -0.1  # bump into wall
            elif cell == HAZARD:
                self.agent_x = nx
                self.agent_y = ny
                reward = -1.0
            elif cell == REWARD:
                self.agent_x = nx
                self.agent_y = ny
                reward = 10.0
                done = True
            else:
                self.agent_x = nx
                self.agent_y = ny
        else:
            reward = -0.1  # out of bounds

        self.steps += 1
        self.total_reward += reward

        if self.steps >= self.max_steps:
            done = True

        info = {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "agent_pos": (self.agent_x, self.agent_y),
            "target_pos": (self.target_x, self.target_y),
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> list[float]:
        """Build observation vector (8 dimensions)."""
        gs = float(self.size)
        obs = [
            self.agent_x / gs,
            self.agent_y / gs,
            self.target_x / gs,
            self.target_y / gs,
        ]
        # Cardinal neighbors
        for dx, dy in ACTION_DELTAS:
            nx = self.agent_x + dx
            ny = self.agent_y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                obs.append(self.grid[ny, nx] / 3.0)
            else:
                obs.append(1.0)  # wall
        return obs

    def render(self) -> str:
        """Render the grid as an ANSI string."""
        lines = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if x == self.agent_x and y == self.agent_y:
                    row.append("A")
                elif x == self.target_x and y == self.target_y:
                    row.append("$")
                else:
                    row.append(CELL_CHARS.get(self.grid[y, x], "?"))
            lines.append(" ".join(row))
        return "\n".join(lines)

    def get_grid_flat(self) -> list[int]:
        """Get grid as flat list (for WASM/dashboard)."""
        return self.grid.flatten().tolist()
