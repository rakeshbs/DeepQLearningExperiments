import mlx.core as mx

from .dqn import DQN, DQNConfig


class DoubleDQN(DQN):
    """
    Double DQN (van Hasselt et al., 2015) — fixes DQN's Q-value overestimation bias.

    Standard DQN:
        target = r + γ * max_a [ Q_target(s', a) ]
        The target net does both action selection AND evaluation. Because it
        always picks the highest predicted Q-value, estimation noise causes
        the selected value to be an overestimate in expectation.

    Double DQN:
        a* = argmax_a [ Q_online(s', a) ]   ← online net selects the action
        target = r + γ * Q_target(s', a*)   ← target net evaluates it

    Because selection and evaluation use different networks with uncorrelated
    noise, the same noise spike cannot inflate both simultaneously, eliminating
    the upward bias. Everything else (network, buffer, target sync,
    checkpointing) is identical to DQN — only _compute_targets() is overridden.
    """

    def _compute_targets(self, rewards, next_states, dones):
        # Online net picks the best next action (argmax over Q values)
        best_actions = mx.argmax(self.online(next_states), axis=1)
        # Target net evaluates the Q-value of that specific action
        target_q = self.target(next_states)
        # Index target Q-values at the actions chosen by the online net
        next_q   = target_q[mx.arange(len(best_actions)), best_actions]
        # (1 - dones) zeroes out the future return for terminal transitions
        targets  = rewards + self.config.gamma * next_q * (1.0 - dones)
        mx.eval(targets)  # materialise before returning so gradients don't flow through targets
        return targets
