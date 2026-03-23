# Breakout Environment

Atari-inspired Breakout implementation built with pygame.

![Breakout](../../gifs/breakout.gif)

## Rules

- `5` lives per game
- `2` brick walls per game
- Max score: `864`
- Brick values by row: `7, 7, 4, 4, 1, 1`

## Observation Modes

- `state`: 8-dim vector `[ball_x, ball_y, ball_vx, ball_vy, paddle_x, next_brick_x, next_brick_y, bricks_remaining]` (normalised)
- `pixels`: 4 stacked `84×84` grayscale frames, shape `(4, 84, 84)`

## Action Space

| Action | Meaning |
|--------|---------|
| `0` | NOOP |
| `1` | FIRE |
| `2` | RIGHT |
| `3` | LEFT |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obs_type` | `"state"` | `"state"` or `"pixels"` |
| `frame_skip` | `1` | Repeat each action for N frames |
| `render_mode` | `False` | Display a pygame window |
| `terminal_on_life_loss` | `True` | End episode on life loss (training) vs full 5-life game (eval) |

## Experiments

| Experiment | Command |
|-----------|---------|
| Pixel CNN DQN | `python -m experiments.breakout.cnn_dqn` |

Test flags: `--test`, `--best`, `--best-score`, `--full-game`, `--no-render`, `--episodes=N`, `--epsilon=N`, `--record`

## Manual Play

```bash
python play.py --game breakout
```

Controls: `Space` serve · `A`/`D` or `Left`/`Right` move · `R` reset · `Esc`/`Q` quit
