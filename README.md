# MLX Reinforcement Learning

Small reinforcement learning playground built around custom game environments and MLX-based DQN variants.

Current environments:
- `Flappy Bird`
- `Breakout`

Current training setup:
- `DQN`
- `DoubleDQN`
- `Prioritized Experience Replay`
- `ParallelRunner` with Ape-X style actor/learner layout

## Requirements

This project is written for Python 3.10 and uses `MLX`, so it is intended for Apple Silicon machines.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install mlx
pip install -r requirements.txt
```

`requirements.txt` currently contains the non-MLX runtime dependencies:
- `pygame`
- `numpy`

## Project Layout

```text
algorithms/
  base.py
  buffers.py
  dqn.py
  double_dqn.py
envs/
  base.py
  flappy_bird/
  breakout/
experiments/
  flappy_dqn.py
  flappy_double_dqn.py
  flappy_cnn_dqn.py
  breakout_cnn_dqn.py
training/
  runner.py
  parallel_runner.py
  checkpoint.py
play.py
```

## Environments

### Flappy Bird

`envs/flappy_bird/env.py`

Observation modes:
- `state`: 5-dim vector
- `pixels`: 4 stacked `84x84` grayscale frames

Action space:
- `0`: do nothing
- `1`: flap

### Breakout

`envs/breakout/env.py`

Atari-inspired rules:
- `5` lives
- `2` brick walls per game
- max score `864`
- brick values by row: `7, 7, 4, 4, 1, 1`

Observation modes:
- `state`: 8-dim vector
- `pixels`: 4 stacked `84x84` grayscale frames

Action space:
- `0`: NOOP
- `1`: FIRE
- `2`: RIGHT
- `3`: LEFT

## Training Experiments

### Flappy Bird

State-vector DQN:

```bash
python -m experiments.flappy_dqn
```

State-vector DoubleDQN:

```bash
python -m experiments.flappy_double_dqn
```

Pixel-based CNN experiment:

```bash
python -m experiments.flappy_cnn_dqn
```

Note: despite the filename, `flappy_cnn_dqn.py` currently uses `DoubleDQN` with a CNN encoder.

### Breakout

Pixel-based CNN DQN:

```bash
python -m experiments.breakout_cnn_dqn
```

## Testing Saved Checkpoints

Latest checkpoint:

```bash
python -m experiments.breakout_cnn_dqn --test
```

Best checkpoint:

```bash
python -m experiments.breakout_cnn_dqn --test --best
```

Equivalent `--test` / `--best` flags work for the Flappy Bird experiments as well.

## Manual Play

Run Breakout:

```bash
python play.py --game breakout
```

Run Flappy Bird:

```bash
python play.py --game flappy
```

Controls:

Breakout:
- `Space`: serve
- `A` / `D` or `Left` / `Right`: move paddle
- `R`: reset
- `Esc` or `Q`: quit

Flappy Bird:
- `Space`: flap
- `R`: reset
- `Esc` or `Q`: quit

## Checkpoints

Checkpoints are written under `checkpoints/<experiment_name>/` as:
- `latest.npz` / `latest.json`
- `best.npz` / `best.json`

For the parallel runner:
- `latest` is always the most recent learner state
- `best` is selected using the best rolling `Avg100`, not the best single episode

## Notes

- The pixel experiments are much heavier than the state-vector ones.
- `ParallelRunner` uses multiple actor processes, so repeated `pygame` startup lines during launch are normal.
- You may see a `pkg_resources` deprecation warning from `pygame`; that is upstream package noise, not a project-specific error.
