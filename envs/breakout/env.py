import os
import random

import numpy as np
import pygame

from envs.base import BaseEnv

SCREEN_WIDTH = 512
SCREEN_HEIGHT = 512
FPS = 60
PIXEL_OBS_SIZE = 84
PIXEL_OBS_STACK = 4

PADDLE_WIDTH = 88
PADDLE_HEIGHT = 14
PADDLE_Y = SCREEN_HEIGHT - 40
PADDLE_SPEED = 8

BALL_SIZE = 10
BALL_SPEED = 5.0
BALL_SPEEDUP = 1.02
MAX_BALL_SPEED = 9.0

BRICK_ROWS = 6
BRICK_COLS = 10
BRICK_HEIGHT = 20
BRICK_GAP = 4
BRICK_TOP = 60
BRICK_MARGIN_X = 0
MAX_STEPS = 5_000

BRICK_COLORS = [
    (239, 83, 80),
    (255, 167, 38),
    (255, 238, 88),
    (102, 187, 106),
    (66, 165, 245),
    (171, 71, 188),
]


class BreakoutEnv(BaseEnv):
    """
    Single-player Breakout environment.

    obs_type:
        "state"  — 6-dim float vector:
                   (ball_x, ball_y, ball_vx, ball_vy, paddle_x, bricks_remaining)
        "pixels" — 4 stacked 84x84 grayscale frames, shape (4, 84, 84)

    action space: 0 = stay, 1 = move left, 2 = move right

    Rewards:
        +1 for each brick destroyed
        -1 when the ball is lost
        +10 when the board is cleared

    Episode ends when the ball is lost, all bricks are cleared, or max_steps is reached.
    """

    action_dim = 3

    def __init__(self, render_mode: bool = False, obs_type: str = "state"):
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        needs_pygame = render_mode or obs_type == "pixels"
        self._pygame_active = needs_pygame
        if needs_pygame:
            if not render_mode:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            if render_mode:
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Breakout RL")
            else:
                pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        if obs_type == "pixels":
            self.obs_shape = (PIXEL_OBS_STACK, PIXEL_OBS_SIZE, PIXEL_OBS_SIZE)
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.float32)
        else:
            self.obs_shape = (6,)

        self.brick_width = (
            SCREEN_WIDTH - 2 * BRICK_MARGIN_X - (BRICK_COLS - 1) * BRICK_GAP
        ) / BRICK_COLS
        self.total_bricks = BRICK_ROWS * BRICK_COLS

        self.paddle_x = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.bricks = []
        self.score = 0
        self.steps = 0

        self.reset()

    def reset(self) -> np.ndarray:
        self.paddle_x = (SCREEN_WIDTH - PADDLE_WIDTH) / 2
        self.score = 0
        self.steps = 0
        self.bricks = self._build_bricks()
        self._reset_ball(launch_upward=True)

        if self.obs_type == "pixels":
            self._frame_stack = np.zeros(self.obs_shape, dtype=np.float32)
            self._draw_frame()
            frame = self._capture_frame()
            for i in range(PIXEL_OBS_STACK):
                self._frame_stack[i] = frame

        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.steps += 1

        if action == 1:
            self.paddle_x -= PADDLE_SPEED
        elif action == 2:
            self.paddle_x += PADDLE_SPEED
        self.paddle_x = float(np.clip(self.paddle_x, 0, SCREEN_WIDTH - PADDLE_WIDTH))

        reward = 0.0
        done = False

        steps = max(
            1, int(np.ceil(max(abs(self.ball_vx), abs(self.ball_vy)) / BALL_SIZE))
        )
        sub_vx = self.ball_vx / steps
        sub_vy = self.ball_vy / steps

        for _ in range(steps):
            prev_x = self.ball_x
            prev_y = self.ball_y
            self.ball_x += sub_vx
            self.ball_y += sub_vy

            if self.ball_x <= 0:
                self.ball_x = 0
                self.ball_vx = abs(self.ball_vx)
                sub_vx = abs(sub_vx)
            elif self.ball_x + BALL_SIZE >= SCREEN_WIDTH:
                self.ball_x = SCREEN_WIDTH - BALL_SIZE
                self.ball_vx = -abs(self.ball_vx)
                sub_vx = -abs(sub_vx)

            if self.ball_y <= 0:
                self.ball_y = 0
                self.ball_vy = abs(self.ball_vy)
                sub_vy = abs(sub_vy)

            paddle_rect = pygame.Rect(
                self.paddle_x, PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT
            )
            ball_rect = pygame.Rect(self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE)

            if ball_rect.colliderect(paddle_rect) and self.ball_vy > 0:
                self.ball_y = PADDLE_Y - BALL_SIZE - 1
                self.ball_vy = -abs(self.ball_vy)
                hit_pos = (
                    (self.ball_x + BALL_SIZE / 2) - (self.paddle_x + PADDLE_WIDTH / 2)
                ) / (PADDLE_WIDTH / 2)
                self.ball_vx += float(hit_pos * 2.2)
                self._speed_up_ball()
                self._clamp_ball_speed(min_vertical_speed=3.0)
                sub_vx = self.ball_vx / steps
                sub_vy = self.ball_vy / steps
                ball_rect = pygame.Rect(self.ball_x, self.ball_y, BALL_SIZE, BALL_SIZE)

            brick_hit = None
            for brick in self.bricks:
                brick_rect = brick["rect"]
                if ball_rect.colliderect(brick_rect):
                    brick_hit = brick
                    overlap_left = ball_rect.right - brick_rect.left
                    overlap_right = brick_rect.right - ball_rect.left
                    overlap_top = ball_rect.bottom - brick_rect.top
                    overlap_bottom = brick_rect.bottom - ball_rect.top
                    min_overlap = min(
                        overlap_left, overlap_right, overlap_top, overlap_bottom
                    )

                    if min_overlap in (overlap_left, overlap_right):
                        self.ball_vx = -self.ball_vx
                        if overlap_left < overlap_right:
                            self.ball_x = brick_rect.left - BALL_SIZE - 1
                        else:
                            self.ball_x = brick_rect.right + 1
                        sub_vx = self.ball_vx / steps
                    else:
                        self.ball_vy = -self.ball_vy
                        if overlap_top < overlap_bottom:
                            self.ball_y = brick_rect.top - BALL_SIZE - 1
                        else:
                            self.ball_y = brick_rect.bottom + 1
                        sub_vy = self.ball_vy / steps
                    break

            if brick_hit is not None:
                self.bricks.remove(brick_hit)
                self.score += 1
                reward += 1.0
                self._speed_up_ball()
                self._clamp_ball_speed(min_vertical_speed=2.5)
                sub_vx = self.ball_vx / steps
                sub_vy = self.ball_vy / steps

            if not self.bricks:
                reward += 10.0
                done = True
                break

            if self.ball_y > SCREEN_HEIGHT:
                reward -= 1.0
                done = True
                break

        if self.steps >= MAX_STEPS:
            done = True

        if self.render_mode:
            self.render()
        elif self.obs_type == "pixels":
            self._draw_frame()

        if self.obs_type == "pixels":
            frame = self._capture_frame()
            self._frame_stack = np.roll(self._frame_stack, shift=-1, axis=0)
            self._frame_stack[-1] = frame

        return self._get_obs(), reward, done, {"score": self.score}

    def _build_bricks(self) -> list[dict]:
        bricks = []
        for row in range(BRICK_ROWS):
            color = BRICK_COLORS[row % len(BRICK_COLORS)]
            for col in range(BRICK_COLS):
                x = BRICK_MARGIN_X + col * (self.brick_width + BRICK_GAP)
                y = BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
                bricks.append(
                    {
                        "rect": pygame.Rect(
                            int(round(x)),
                            int(round(y)),
                            int(round(self.brick_width)),
                            BRICK_HEIGHT,
                        ),
                        "color": color,
                    }
                )
        return bricks

    def _reset_ball(self, launch_upward: bool = False) -> None:
        self.ball_x = self.paddle_x + PADDLE_WIDTH / 2 - BALL_SIZE / 2
        self.ball_y = PADDLE_Y - BALL_SIZE - 4
        angle = random.uniform(35, 65) * random.choice([-1, 1])
        rad = np.deg2rad(angle)
        self.ball_vx = BALL_SPEED * np.sin(rad)
        self.ball_vy = -BALL_SPEED * np.cos(rad)
        if not launch_upward and self.ball_vy > 0:
            self.ball_vy = -self.ball_vy

    def _clamp_ball_speed(self, min_vertical_speed: float = 0.0) -> None:
        speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
        if speed > MAX_BALL_SPEED:
            scale = MAX_BALL_SPEED / speed
            self.ball_vx *= scale
            self.ball_vy *= scale

        if min_vertical_speed > 0 and abs(self.ball_vy) < min_vertical_speed:
            self.ball_vy = np.sign(self.ball_vy or -1.0) * min_vertical_speed
            speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
            if speed > MAX_BALL_SPEED:
                scale = MAX_BALL_SPEED / speed
                self.ball_vx *= scale
                self.ball_vy *= scale

    def _speed_up_ball(self) -> None:
        speed = np.sqrt(self.ball_vx**2 + self.ball_vy**2)
        if speed == 0:
            return
        new_speed = min(MAX_BALL_SPEED, speed * BALL_SPEEDUP)
        scale = new_speed / speed
        self.ball_vx *= scale
        self.ball_vy *= scale

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            return self._frame_stack.copy()
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array(
            [
                (self.ball_x + BALL_SIZE / 2) / SCREEN_WIDTH,
                (self.ball_y + BALL_SIZE / 2) / SCREEN_HEIGHT,
                self.ball_vx / MAX_BALL_SPEED,
                self.ball_vy / MAX_BALL_SPEED,
                (self.paddle_x + PADDLE_WIDTH / 2) / SCREEN_WIDTH,
                len(self.bricks) / self.total_bricks,
            ],
            dtype=np.float32,
        )

    def _draw_frame(self) -> None:
        if self.screen is None:
            return
        pygame.event.pump()

        self.screen.fill((14, 20, 30))

        for brick in self.bricks:
            pygame.draw.rect(
                self.screen, brick["color"], brick["rect"], border_radius=3
            )

        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            (int(self.paddle_x), PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT),
            border_radius=4,
        )
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE),
        )

        font = pygame.font.SysFont(None, 32)
        score_text = font.render(f"Score: {self.score}", True, (230, 230, 230))
        bricks_text = font.render(f"Bricks: {len(self.bricks)}", True, (180, 200, 220))
        self.screen.blit(score_text, (16, 14))
        self.screen.blit(bricks_text, (SCREEN_WIDTH - bricks_text.get_width() - 16, 14))

    def _capture_frame(self) -> np.ndarray:
        raw = pygame.surfarray.array3d(self.screen)
        raw = np.transpose(raw, (1, 0, 2))
        gray = 0.299 * raw[:, :, 0] + 0.587 * raw[:, :, 1] + 0.114 * raw[:, :, 2]
        h, w = gray.shape
        th = tw = PIXEL_OBS_SIZE
        row_idx = (np.arange(th) * h // th).astype(np.int32)
        col_idx = (np.arange(tw) * w // tw).astype(np.int32)
        resized = gray[np.ix_(row_idx, col_idx)]
        return (resized / 255.0).astype(np.float32)

    def render(self) -> None:
        if not self.render_mode:
            return
        self._draw_frame()
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self) -> None:
        if self._pygame_active:
            pygame.quit()
