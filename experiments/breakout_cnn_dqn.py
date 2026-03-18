"""
Experiment: Breakout + CNN DQN (parallel, pixel observations)

Obs: 4 stacked 84x84 grayscale frames — shape (4, 84, 84)
Net: CNNQNetwork (3 conv layers + MLP head, DeepMind Atari architecture)
Algo: Standard DQN

The environment follows the Atari Breakout scoring and match structure:
  - 4 actions: NOOP, FIRE, RIGHT, LEFT
  - 5 lives
  - 2 walls total per game
  - brick rewards by row: 7, 7, 4, 4, 1, 1

Train:  python -m experiments.breakout_cnn_dqn
Test:   python -m experiments.breakout_cnn_dqn --test [--best]
"""

import os
import sys

from algorithms.dqn import CNNQNetwork, DQN, DQNConfig
from envs.breakout import BreakoutEnv
from training.parallel_runner import ParallelRunner
from training.runner import RunnerConfig

OBS_SHAPE = (4, 84, 84)
ACTION_DIM = 4


def make_algo():
    return DQN(
        DQNConfig(
            action_dim=ACTION_DIM,
            network_factory=lambda: CNNQNetwork(obs_shape=OBS_SHAPE, action_dim=ACTION_DIM),
            lr=1e-4,
            gamma=0.99,
            target_update_freq=2_500,
        )
    )


runner = ParallelRunner(
    env_factory=BreakoutEnv,
    algo=make_algo(),
    algo_factory=make_algo,
    config=RunnerConfig(
        buffer_size=100_000,
        batch_size=32,
        train_start=5_000,
        max_episodes=1_000_000,
        render_every=200,
        ckpt_dir=os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "breakout_cnn_dqn"
        ),
        log_every=50,
    ),
    num_actors=8,
    updates_per_drain=4,
    weight_sync_freq=200,
    epsilon_alpha=4.0,
    epsilon_base=0.4,
    epsilon_base_decay=1.0,
    epsilon_base_min=0.4,
    per_alpha=0.6,
    per_beta=0.4,
    env_kwargs={"obs_type": "pixels"},
)

if __name__ == "__main__":
    if "--test" in sys.argv:
        runner.test(best="--best" in sys.argv)
    else:
        runner.train()
