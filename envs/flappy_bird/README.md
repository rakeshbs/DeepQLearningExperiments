# Flappy Bird Environment

Custom Flappy Bird implementation built with pygame.

![Flappy Bird](../../gifs/flappy.gif)

## Observation Modes

- `state`: 5-dim vector `[bird_y, bird_velocity, next_pipe_x, next_pipe_top, next_pipe_bottom]` (normalised)
- `pixels`: 4 stacked `84×84` grayscale frames, shape `(4, 84, 84)`

## Action Space

| Action | Meaning |
|--------|---------|
| `0` | do nothing |
| `1` | flap |

## Reward Structure

| Event | Reward |
|-------|--------|
| Survive one step | `+0.1` |
| Pass a pipe | `+10` (clipped to `+1`) |
| Die (ground / ceiling / pipe) | `0` (terminal) |

All rewards are clipped to `[-1, +1]`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipe_gap` | `100` | Vertical gap height in pixels the bird must fly through |
| `obs_type` | `"state"` | `"state"` or `"pixels"` |
| `render_mode` | `False` | Display a pygame window |

## Experiments

| Experiment | Command |
|-----------|---------|
| State DQN | `python -m experiments.flappy.dqn` |
| State Double DQN | `python -m experiments.flappy.double_dqn` |
| Pixel CNN DQN | `python -m experiments.flappy.cnn_dqn` |

Test flags: `--test`, `--best`, `--record`

## Manual Play

```bash
python play.py --game flappy
```

Controls: `Space` flap · `R` reset · `Esc`/`Q` quit
