from abc import ABC, abstractmethod

import numpy as np


class BaseAlgorithm(ABC):
    """
    Interface every RL algorithm must implement.

    Defines a minimal contract so runners, checkpointers, and parallel actors
    can operate without knowing any algorithm-specific details. All weight
    transfer methods use plain numpy dicts so they are serialisable across
    process boundaries (multiprocessing queues cannot ship MLX arrays directly).
    """

    @abstractmethod
    def select_action(self, state: np.ndarray) -> int:
        """
        Return the greedy action for the given state (no exploration).

        Actors call this during inference. Exploration (epsilon-greedy) is
        handled by the runner, not here, so this always returns argmax Q.
        """
        ...

    @abstractmethod
    def update(self, batch: tuple, weights: np.ndarray | None = None) -> tuple:
        """
        Update the algorithm with a batch of transitions.
        Returns (loss: float, td_errors: np.ndarray).

        td_errors are per-sample |Q - target| values used for PER priority
        updates — the buffer needs them to re-rank which transitions to replay.
        weights: per-sample importance-sampling weights (None = uniform).
            PER passes IS weights here to correct for the non-uniform sampling
            bias introduced by prioritised replay.
        """
        ...

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Save learnable parameters to disk (as .npz via MLX)."""
        ...

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Restore learnable parameters from disk and sync target network."""
        ...

    @abstractmethod
    def get_weights(self) -> dict:
        """
        Return online network weights as a flat numpy dict.

        Used by the learner to package weights for broadcast to actor processes.
        Numpy arrays (not MLX) are required so the dict can cross process
        boundaries through a multiprocessing Queue.
        """
        ...

    @abstractmethod
    def set_weights(self, weights: dict) -> None:
        """
        Load online network weights from a flat numpy dict.

        Called by actor processes when the learner broadcasts updated weights.
        The dict must first be converted back to MLX arrays before being
        applied to the network.
        """
        ...
