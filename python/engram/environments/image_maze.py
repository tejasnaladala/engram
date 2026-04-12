"""Load a maze from an image file and solve it with Engram.

Supports PNG/JPG maze images where:
- Black/dark pixels = walls
- White/light pixels = paths
- Green pixel (optional) = start position
- Red pixel (optional) = goal position

If no start/goal markers found, uses top-left and bottom-right open cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def load_maze_from_image(path: str, max_size: int = 40) -> np.ndarray:
    """Load a maze image and convert to a binary grid.

    Args:
        path: Path to maze image (PNG, JPG, BMP)
        max_size: Maximum grid dimension (downscales if larger)

    Returns:
        2D numpy array: 1 = wall, 0 = path
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for image maze loading: pip install Pillow")

    img = Image.open(path).convert('L')  # grayscale

    # Downscale if too large
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.NEAREST)

    arr = np.array(img)
    # Threshold: dark = wall (1), light = path (0)
    threshold = arr.mean()
    grid = (arr < threshold).astype(np.uint8)
    return grid


def find_start_goal(grid: np.ndarray, image_path: str | None = None):
    """Find start and goal positions.

    Tries to find green (start) and red (goal) pixels in color image.
    Falls back to first and last open cells.
    """
    h, w = grid.shape
    start = None
    goal = None

    # Try color detection if image path provided
    if image_path:
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img_arr = np.array(img.resize((w, h), Image.NEAREST))
            for y in range(h):
                for x in range(w):
                    r, g, b = img_arr[y, x]
                    if g > 150 and r < 100 and b < 100 and start is None:
                        start = (x, y)
                    if r > 150 and g < 100 and b < 100 and goal is None:
                        goal = (x, y)
        except Exception:
            pass

    # Fallback: first open cell from top-left, last from bottom-right
    if start is None:
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 0:
                    start = (x, y)
                    break
            if start:
                break

    if goal is None:
        for y in range(h - 1, -1, -1):
            for x in range(w - 1, -1, -1):
                if grid[y, x] == 0:
                    goal = (x, y)
                    break
            if goal:
                break

    return start or (1, 1), goal or (w - 2, h - 2)


class ImageMazeEnv:
    """Environment that loads a maze from an image file.

    The agent must navigate from start to goal through the maze corridors.

    Observation (10 dims):
      [agent_x/w, agent_y/h, goal_x/w, goal_y/h,
       wall_up, wall_right, wall_down, wall_left,
       dist_to_goal/max_dist, steps/max_steps]
    """

    def __init__(self, image_path: str, max_size: int = 30):
        self.image_path = image_path
        self.grid = load_maze_from_image(image_path, max_size)
        self.h, self.w = self.grid.shape
        start, goal = find_start_goal(self.grid, image_path)
        self.start_x, self.start_y = start
        self.goal_x, self.goal_y = goal
        self.agent_x = self.start_x
        self.agent_y = self.start_y
        self.steps = 0
        self.max_steps = self.w * self.h * 3
        self.total_reward = 0.0
        self.prev_dist = self._dist()
        self.path_history: list[tuple[int, int]] = []

    def _dist(self) -> float:
        return abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)

    def reset(self, seed: int | None = None) -> list[float]:
        self.agent_x = self.start_x
        self.agent_y = self.start_y
        self.steps = 0
        self.total_reward = 0.0
        self.prev_dist = self._dist()
        self.path_history = [(self.agent_x, self.agent_y)]
        return self._obs()

    def step(self, action: int) -> tuple[list[float], float, bool, dict]:
        dx, dy = ACTION_DELTAS[action % 4]
        nx, ny = self.agent_x + dx, self.agent_y + dy

        reward = -0.01
        done = False

        if 0 <= nx < self.w and 0 <= ny < self.h and self.grid[ny, nx] == 0:
            self.agent_x, self.agent_y = nx, ny
            dist = self._dist()
            reward += 0.1 if dist < self.prev_dist else -0.1
            self.prev_dist = dist
            if nx == self.goal_x and ny == self.goal_y:
                reward = 10.0
                done = True
        else:
            reward = -0.5

        self.steps += 1
        self.total_reward += reward
        self.path_history.append((self.agent_x, self.agent_y))
        if self.steps >= self.max_steps:
            done = True

        info = {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'reached_goal': self.agent_x == self.goal_x and self.agent_y == self.goal_y,
            'agent_pos': (self.agent_x, self.agent_y),
            'path': self.path_history,
        }
        return self._obs(), reward, done, info

    def _obs(self) -> list[float]:
        obs = [
            self.agent_x / max(self.w, 1),
            self.agent_y / max(self.h, 1),
            self.goal_x / max(self.w, 1),
            self.goal_y / max(self.h, 1),
        ]
        for dx, dy in ACTION_DELTAS:
            nx, ny = self.agent_x + dx, self.agent_y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                obs.append(float(self.grid[ny, nx]))
            else:
                obs.append(1.0)
        max_dist = float(self.w + self.h)
        obs.append(self._dist() / max(max_dist, 1))
        obs.append(self.steps / max(self.max_steps, 1))
        return obs

    def render(self) -> str:
        lines = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                if x == self.agent_x and y == self.agent_y:
                    row.append('A')
                elif x == self.goal_x and y == self.goal_y:
                    row.append('G')
                elif self.grid[y, x] == 1:
                    row.append('#')
                elif (x, y) in self.path_history:
                    row.append('.')
                else:
                    row.append(' ')
            lines.append(''.join(row))
        return '\n'.join(lines)

    def render_image(self, scale: int = 10) -> np.ndarray:
        """Render maze as RGB image array for video/display.

        Returns numpy array of shape (h*scale, w*scale, 3).
        """
        img = np.zeros((self.h * scale, self.w * scale, 3), dtype=np.uint8)

        for y in range(self.h):
            for x in range(self.w):
                color = (20, 20, 30) if self.grid[y, x] == 1 else (40, 50, 70)
                img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = color

        # Draw path history
        for px, py in self.path_history[:-1]:
            cx, cy = px * scale + scale // 2, py * scale + scale // 2
            r = scale // 4
            img[max(0,cy-r):cy+r, max(0,cx-r):cx+r] = (40, 100, 80)

        # Goal
        gx, gy = self.goal_x * scale, self.goal_y * scale
        img[gy:gy+scale, gx:gx+scale] = (50, 200, 80)

        # Agent
        ax, ay = self.agent_x * scale, self.agent_y * scale
        img[ay:ay+scale, ax:ax+scale] = (60, 220, 240)

        return img
