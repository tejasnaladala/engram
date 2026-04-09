"""Procedural maze environment for navigation learning."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class MazeEnv:
    """Procedurally generated maze for testing navigation learning.

    Uses recursive backtracking to generate solvable mazes.
    The agent must learn to navigate from start (top-left) to
    goal (bottom-right) through corridors.

    Observation (10 dims):
      [agent_x/w, agent_y/h, goal_x/w, goal_y/h,
       wall_up, wall_right, wall_down, wall_left,
       dist_to_goal/max_dist, steps/max_steps]

    Rewards:
      +10.0  reaching the goal
      -0.01  each step (encourages efficiency)
      -0.5   hitting a wall
      +0.1   getting closer to goal
      -0.1   getting further from goal
    """

    def __init__(
        self,
        width: int = 7,
        height: int = 7,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.maze_w = width * 2 + 1
        self.maze_h = height * 2 + 1
        self.rng = random.Random(seed)
        self.grid: np.ndarray = np.zeros((self.maze_h, self.maze_w), dtype=np.uint8)
        self.agent_x = 1
        self.agent_y = 1
        self.goal_x = self.maze_w - 2
        self.goal_y = self.maze_h - 2
        self.steps = 0
        self.max_steps = self.maze_w * self.maze_h * 2
        self.total_reward = 0.0
        self.prev_dist = 0.0
        self._generate_maze()

    def _generate_maze(self) -> None:
        """Generate a maze using recursive backtracking."""
        self.grid.fill(1)  # all walls

        # Carve passages
        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))
        self.grid[1, 1] = 0  # start cell

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))

            if neighbors:
                nx, ny, dx, dy = self.rng.choice(neighbors)
                # Remove wall between current and neighbor
                wall_x = cx * 2 + 1 + dx
                wall_y = cy * 2 + 1 + dy
                self.grid[wall_y, wall_x] = 0
                # Carve neighbor cell
                self.grid[ny * 2 + 1, nx * 2 + 1] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        # Ensure start and goal are open
        self.grid[1, 1] = 0
        self.grid[self.goal_y, self.goal_x] = 0

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.agent_x = 1
        self.agent_y = 1
        self.steps = 0
        self.total_reward = 0.0
        self._generate_maze()
        self.prev_dist = self._dist_to_goal()
        return self._get_observation()

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        dx, dy = ACTION_DELTAS[action % 4]
        nx = self.agent_x + dx
        ny = self.agent_y + dy

        reward = -0.01  # step cost
        done = False

        if 0 <= nx < self.maze_w and 0 <= ny < self.maze_h and self.grid[ny, nx] == 0:
            self.agent_x = nx
            self.agent_y = ny

            # Distance-based shaping
            dist = self._dist_to_goal()
            if dist < self.prev_dist:
                reward += 0.1
            elif dist > self.prev_dist:
                reward -= 0.1
            self.prev_dist = dist

            # Goal reached
            if nx == self.goal_x and ny == self.goal_y:
                reward = 10.0
                done = True
        else:
            reward = -0.5  # wall hit

        self.steps += 1
        self.total_reward += reward
        if self.steps >= self.max_steps:
            done = True

        info = {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "reached_goal": nx == self.goal_x and ny == self.goal_y,
            "agent_pos": (self.agent_x, self.agent_y),
        }
        return self._get_observation(), reward, done, info

    def _dist_to_goal(self) -> float:
        return abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)

    def _get_observation(self) -> list[float]:
        w = float(self.maze_w)
        h = float(self.maze_h)
        obs = [
            self.agent_x / w,
            self.agent_y / h,
            self.goal_x / w,
            self.goal_y / h,
        ]
        # Wall sensors in 4 directions
        for dx, dy in ACTION_DELTAS:
            nx, ny = self.agent_x + dx, self.agent_y + dy
            if 0 <= nx < self.maze_w and 0 <= ny < self.maze_h:
                obs.append(float(self.grid[ny, nx]))
            else:
                obs.append(1.0)
        # Distance and progress
        max_dist = float(self.maze_w + self.maze_h)
        obs.append(self._dist_to_goal() / max_dist)
        obs.append(self.steps / self.max_steps)
        return obs

    def render(self) -> str:
        lines = []
        for y in range(self.maze_h):
            row = []
            for x in range(self.maze_w):
                if x == self.agent_x and y == self.agent_y:
                    row.append("A")
                elif x == self.goal_x and y == self.goal_y:
                    row.append("G")
                elif self.grid[y, x] == 1:
                    row.append("#")
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)
