from abc import ABC, abstractmethod

import numpy as np


class BaseEnv(ABC):
    """
    Minimal interface every game environment must implement.

    Mirrors the OpenAI Gym API so algorithms and runners are game-agnostic.
    Concrete subclasses declare obs_shape and action_dim as class attributes
    so callers can inspect them before instantiating the network.
    """

    obs_shape: tuple  # e.g. (5,) for a state vector, (4, 84, 84) for a pixel stack
    action_dim: int   # number of discrete actions available to the agent

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        Returns the first observation as a numpy array of shape obs_shape.
        """
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply action to the environment for one time step.
        Returns (next_obs, reward, done, info).
          - next_obs: numpy array of shape obs_shape
          - reward:   scalar float
          - done:     True if the episode has ended
          - info:     dict with auxiliary data (e.g. {"score": int})
        """
        ...

    def render(self) -> None:
        """Optional: draw the current frame to a display window."""

    def capture_frame(self) -> "np.ndarray | None":
        """
        Optional: return the current frame as an RGB numpy array of shape (H, W, 3).
        Returns None if the environment does not support frame capture.
        Used by the test runner for video recording (--record).
        """
        return None

    def close(self) -> None:
        """Optional: release display or other OS resources when the episode ends."""
