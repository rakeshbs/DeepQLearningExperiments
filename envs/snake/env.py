import os
import random
from collections import deque

import numpy as np
import pygame

from envs.base import BaseEnv

SCREEN_WIDTH = 512
SCREEN_HEIGHT = 512
FPS = 15
PIXEL_OBS_SIZE = 84
PIXEL_OBS_STACK = 4

GRID_SIZE = 20          # number of cells along each axis
CELL = SCREEN_WIDTH // GRID_SIZE  # pixel size of each cell (25 px)

# Colours
BG_COLOR = (15, 15, 20)
GRID_COLOR = (25, 25, 32)
SNAKE_HEAD_COLOR = (80, 220, 100)
SNAKE_BODY_COLOR = (50, 160, 70)
FOOD_COLOR = (230, 80, 80)

# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# Direction vectors (dx, dy) in grid coordinates
_DIR = {
    ACTION_UP:    (0, -1),
    ACTION_DOWN:  (0,  1),
    ACTION_LEFT:  (-1, 0),
    ACTION_RIGHT: (1,  0),
}

# Opposite actions — used to prevent 180° turns
_OPPOSITE = {
    ACTION_UP: ACTION_DOWN,
    ACTION_DOWN: ACTION_UP,
    ACTION_LEFT: ACTION_RIGHT,
    ACTION_RIGHT: ACTION_LEFT,
}

MAX_STEPS_NO_FOOD = GRID_SIZE * GRID_SIZE * 2  # reset if stuck without eating


class SnakeEnv(BaseEnv):
    """
    Snake environment for reinforcement learning.

    The snake lives on a GRID_SIZE × GRID_SIZE grid.  The agent picks one of
    four cardinal directions each step; turning 180° is silently ignored (the
    snake keeps its current direction instead).

    obs_type:
        "pixels" — 4 stacked 84×84 grayscale frames, shape (4, 84, 84).
                   Requires pygame even when not rendering visually.
        "state"  — flat feature vector of length 11 (danger flags, direction
                   one-hot, food direction flags).  Fast; does not need pygame.

    action space: 0=UP  1=DOWN  2=LEFT  3=RIGHT
    """

    action_dim = 4

    def __init__(self, render_mode: bool = False, obs_type: str = "pixels"):
        self.render_mode = render_mode
        self.obs_type = obs_type

        needs_pygame = render_mode or obs_type == "pixels"
        self._pygame_active = needs_pygame
        if needs_pygame:
            if not render_mode:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            if render_mode:
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Snake RL")
            else:
                pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        if obs_type == "pixels":
            self.obs_shape = (PIXEL_OBS_STACK, PIXEL_OBS_SIZE, PIXEL_OBS_SIZE)
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.uint8)
        else:
            self.obs_shape = (11,)

        # Game state (populated by reset)
        self.grid_size: int = GRID_SIZE  # exposed so reward shapers can read it

        self.snake: deque[tuple[int, int]] = deque()
        self.direction: int = ACTION_RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.score: int = 0
        self.steps: int = 0
        self.steps_since_food: int = 0

        self.reset()

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        # Start near centre, length 3, heading right
        mid = GRID_SIZE // 2
        self.snake = deque([(mid - 2, mid), (mid - 1, mid), (mid, mid)])
        self.direction = ACTION_RIGHT
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._place_food()

        if self.obs_type == "pixels":
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.uint8)
            self._draw_frame()
            frame = self._capture_frame()
            for i in range(PIXEL_OBS_STACK):
                self._frame_stack[i] = frame

        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        # Ignore 180° reversal
        if action == _OPPOSITE[self.direction]:
            action = self.direction
        self.direction = action

        dx, dy = _DIR[self.direction]
        hx, hy = self.snake[-1]
        new_head = (hx + dx, hy + dy)

        self.steps += 1
        self.steps_since_food += 1

        done = False
        reward = 0.0

        # Wall collision
        nx, ny = new_head
        if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
            done = True
            reward = -1.0
        # Self collision (ignore tail tip because it will move away)
        elif new_head in list(self.snake)[:-1]:
            done = True
            reward = -1.0
        else:
            ate_food = new_head == self.food
            self.snake.append(new_head)

            if ate_food:
                self.score += 1
                reward = 1.0
                self.steps_since_food = 0
                if len(self.snake) == GRID_SIZE * GRID_SIZE:
                    done = True  # board full — perfect game
                else:
                    self._place_food()
            else:
                self.snake.popleft()  # remove tail to keep length constant

            # Starvation: too many steps without eating
            if self.steps_since_food >= MAX_STEPS_NO_FOOD:
                done = True
                reward = -1.0

        if self.render_mode:
            self.render()
        elif self.obs_type == "pixels":
            self._draw_frame()

        if self.obs_type == "pixels":
            frame = self._capture_frame()
            self._frame_stack = np.roll(self._frame_stack, shift=-1, axis=0)
            self._frame_stack[-1] = frame

        return self._get_obs(), reward, done, {"score": self.score}

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            return self._frame_stack.copy()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        11-dimensional state vector:
          [0-2]  danger straight / right / left  (1 if collision one step ahead)
          [3-6]  direction one-hot: up / down / left / right
          [7-10] food: up / down / left / right  (1 if food is in that relative direction)
        """
        hx, hy = self.snake[-1]
        dx, dy = _DIR[self.direction]

        def _danger(ddx, ddy):
            nx, ny = hx + ddx, hy + ddy
            if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
                return 1.0
            if (nx, ny) in set(self.snake):
                return 1.0
            return 0.0

        # Relative directions (straight / turn right / turn left)
        # Rotating (dx, dy) 90° clockwise → (dy, -dx); counter-clockwise → (-dy, dx)
        straight = (dx, dy)
        right_turn = (dy, -dx)
        left_turn = (-dy, dx)

        fx, fy = self.food
        state = [
            _danger(*straight),
            _danger(*right_turn),
            _danger(*left_turn),
            float(self.direction == ACTION_UP),
            float(self.direction == ACTION_DOWN),
            float(self.direction == ACTION_LEFT),
            float(self.direction == ACTION_RIGHT),
            float(fy < hy),   # food is above
            float(fy > hy),   # food is below
            float(fx < hx),   # food is to the left
            float(fx > hx),   # food is to the right
        ]
        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_frame(self) -> None:
        if self.screen is None:
            return
        pygame.event.pump()

        self.screen.fill(BG_COLOR)

        # Subtle grid lines
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, GRID_COLOR, (i * CELL, 0), (i * CELL, SCREEN_HEIGHT))
            pygame.draw.line(self.screen, GRID_COLOR, (0, i * CELL), (SCREEN_WIDTH, i * CELL))

        # Food
        fx, fy = self.food
        food_rect = pygame.Rect(fx * CELL + 2, fy * CELL + 2, CELL - 4, CELL - 4)
        pygame.draw.ellipse(self.screen, FOOD_COLOR, food_rect)

        # Snake body
        snake_list = list(self.snake)
        for i, (sx, sy) in enumerate(snake_list):
            color = SNAKE_HEAD_COLOR if i == len(snake_list) - 1 else SNAKE_BODY_COLOR
            margin = 1 if i < len(snake_list) - 1 else 0
            rect = pygame.Rect(sx * CELL + margin, sy * CELL + margin, CELL - margin * 2, CELL - margin * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Score HUD
        font = pygame.font.SysFont(None, 28)
        score_surf = font.render(f"Score: {self.score}", True, (200, 200, 200))
        self.screen.blit(score_surf, (8, 8))

    def _capture_frame(self) -> np.ndarray:
        raw = pygame.surfarray.array3d(self.screen)   # (W, H, 3)
        raw = np.transpose(raw, (1, 0, 2))             # (H, W, 3)
        gray = 0.299 * raw[:, :, 0] + 0.587 * raw[:, :, 1] + 0.114 * raw[:, :, 2]
        h, w = gray.shape
        th = tw = PIXEL_OBS_SIZE
        row_idx = (np.arange(th) * h // th).astype(np.int32)
        col_idx = (np.arange(tw) * w // tw).astype(np.int32)
        resized = gray[np.ix_(row_idx, col_idx)]
        return resized.astype(np.uint8)

    def render(self) -> None:
        if not self.render_mode:
            return
        self._draw_frame()
        pygame.display.flip()
        self.clock.tick(FPS)

    def capture_frame(self):
        """Return current frame as RGB (H, W, 3) array, or None."""
        if self.screen is None:
            return None
        frame = pygame.surfarray.array3d(self.screen)
        return frame.transpose(1, 0, 2)

    def close(self) -> None:
        if self._pygame_active:
            pygame.quit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _place_food(self) -> None:
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            if (x, y) not in occupied
        ]
        self.food = random.choice(free)
