import multiprocessing as mp
import random
import resource
import threading
import time
from typing import Any, Callable, Dict, Optional

import mlx.core as mx
import numpy as np

from algorithms.buffers import PrioritizedReplayBuffer, ReplayBuffer
from training.checkpoint import Checkpointer
from training.runner import RunnerConfig


def _ape_x_epsilons(num_actors: int, base: float = 0.4, alpha: float = 7.0) -> list:
    """
    Ape-X epsilon schedule: actor i gets ε_i = base ^ (1 + α * i / (N - 1)).

    This formula spreads actors across a wide exploration range in a single
    parameter (base). Actor 0 is the most exploratory (base^1 = base) and
    actor N-1 is nearly greedy (base^(1+alpha) → 0 for large alpha). The
    diversity of experience collected across actors improves sample efficiency
    compared to all actors sharing the same epsilon. When epsilon_base decays
    over training, the entire spread shifts toward more greedy behaviour.
    """
    if num_actors == 1:
        return [base]
    return [base ** (1.0 + alpha * i / (num_actors - 1)) for i in range(num_actors)]


def _actor_fn(
    actor_id: int,
    env_factory,
    algo_factory,
    reward_shaper,
    weight_queue,
    transition_queue,
    epsilon: float,
    train_start: int,
    env_kwargs: dict,
    initial_weights: dict,
):
    """
    Runs in a child process. Each actor has its own independent MLX Metal context.

    The actor runs an infinite episode loop, collecting (s, a, r, s', done)
    transitions and pushing them into the shared transition_queue for the
    learner process to consume. If the queue is full, transitions are dropped
    rather than blocking — actor throughput is more important than perfect
    capture rate.

    Weight updates arrive as (weights_dict, new_epsilon) tuples on weight_queue.
    Draining the queue before each episode ensures the actor uses the latest
    learner weights without blocking the episode loop mid-step.

    epsilon starts at the Ape-X assigned value and is updated by the learner
    via weight_queue messages, which are (weights_dict, new_epsilon) tuples.
    """
    # Import mlx inside the child process — each spawned process gets a fresh
    # Metal GPU context, which is why 'spawn' (not 'fork') is required.
    import mlx.core as mx

    algo = algo_factory()
    algo.set_weights(initial_weights)
    env = env_factory(**env_kwargs)
    total_steps = 0

    while True:
        # Drain weight_queue before each episode — only keep the most recent
        # message so stale weight broadcasts don't accumulate. Using get_nowait
        # means this never blocks the actor.
        latest_weights = None
        while not weight_queue.empty():
            try:
                msg = weight_queue.get_nowait()
                if isinstance(msg, tuple):
                    latest_weights, epsilon = msg  # unpack (weights, new_epsilon)
                else:
                    latest_weights = msg  # legacy scalar message (backward compat)
            except Exception:
                break
        if latest_weights is not None:
            algo.set_weights(latest_weights)

        state = env.reset()
        while True:
            # Epsilon-greedy: random action during warm-up or with prob epsilon
            if random.random() < epsilon or total_steps < train_start:
                action = random.randint(0, env.action_dim - 1)
            else:
                action = algo.select_action(state)

            next_state, reward, done, info = env.step(action)

            if reward_shaper is not None:
                reward = reward_shaper(env, reward, done)

            # Only send the score on terminal transitions — the learner uses it
            # to track episode boundaries and update the best-score tracker.
            score = info.get("score") if done else None
            transition = (state, action, float(reward), next_state, bool(done), score)
            if done:
                # Terminal transitions carry episode score — block until the learner
                # drains the queue rather than silently dropping them.  A brief block
                # here is acceptable (once per episode) and is far less harmful than
                # losing all score signals (which prevents any logging or checkpointing).
                try:
                    transition_queue.put(transition, timeout=5.0)
                except Exception:
                    pass  # learner appears dead; keep actor alive
            else:
                try:
                    # put_nowait: never block — drop the transition if the queue is full.
                    # A blocked actor would stall its episode loop, which is worse than
                    # occasionally losing a non-terminal transition.
                    transition_queue.put_nowait(transition)
                except Exception:
                    pass  # queue full — drop transition, never block
            state = next_state
            total_steps += 1

            if done:
                break

        # Release GPU memory between episodes; each episode's computation graph
        # has already been materialised so clearing the cache is safe here.
        mx.clear_cache()


class ParallelRunner:
    """
    Ape-X style parallel training loop.

    Architecture:
      - N actor processes run inference-only loops, each with its own MLX
        Metal GPU context (spawned via 'spawn', not 'fork', to avoid shared
        Metal state). Actors push transitions into a shared multiprocessing Queue.
      - One learner (this process) owns the PrioritizedReplayBuffer, optimizer,
        and target network. It drains the queue in batches, runs gradient updates,
        and periodically broadcasts updated weights back to actors.

    Communication:
      - transition_queue: actors → learner (s, a, r, s', done, score)
      - weight_queues[i]: learner → actor i ((weights_dict, epsilon) tuple)
        Each actor has its own queue (maxsize=2) so the learner never blocks
        waiting for a slow actor to consume weights.

    algo_factory must be a module-level named callable (not a lambda) so it
    can be pickled for the child processes. Example:

        def make_algo():
            return DQN(DQNConfig(
                action_dim=2,
                network_factory=lambda: MLPQNetwork(5, 128, 2),
                ...
            ))

        runner = ParallelRunner(env_factory=..., algo=make_algo(), algo_factory=make_algo, ...)

    Note: network_factory inside make_algo CAN be a lambda — it only runs
    inside each child process and never needs to be pickled directly.
    """

    def __init__(
        self,
        env_factory,
        algo,
        algo_factory: Callable,
        config: RunnerConfig,
        num_actors: int = 6,
        updates_per_drain: int = 4,  # gradient steps per queue-drain cycle
        weight_sync_freq: int = 100,  # broadcast weights every N learner updates
        epsilon_base: float = 0.4,
        epsilon_alpha: float = 7.0,
        reward_shaper=None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        epsilon_base_decay: float = 1.0,  # multiply epsilon_base by this each weight sync
        epsilon_base_min: float = 0.01,  # floor for epsilon_base after decay
    ):
        self.env_factory = env_factory
        self.algo = algo
        self.algo_factory = algo_factory
        self.config = config
        self.num_actors = num_actors
        self.updates_per_drain = updates_per_drain
        self.weight_sync_freq = weight_sync_freq
        self.epsilon_base = epsilon_base
        self.epsilon_alpha = epsilon_alpha
        self.reward_shaper = reward_shaper
        self.env_kwargs = env_kwargs or {}
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.epsilon_base_decay = epsilon_base_decay
        self.epsilon_base_min = epsilon_base_min
        self.checkpointer = Checkpointer(config.ckpt_dir)

    def train(self):
        """
        Launch actor processes and run the learner loop.

        The learner alternates between two phases in a tight loop:
          1. Drain: pull up to 512 transitions from the queue into the PER buffer.
             Episode boundaries (done=True) trigger checkpointing and logging.
          2. Update: run `updates_per_drain` gradient steps if the buffer is
             large enough. After `weight_sync_freq` updates, broadcast fresh
             weights and decayed epsilons to all actors.

        A daemon heartbeat thread prints debug state every 10 s so hangs
        can be diagnosed without attaching a debugger.
        """
        # Raise the open-file-descriptor limit; spawning N actors + queues can
        # hit the default limit (256 on macOS) when num_actors is large.
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 8192), hard))

        cfg = self.config

        # Restore learner state from disk if a previous run was checkpointed
        meta = self.checkpointer.load(self.algo)
        if meta:
            start_ep = meta["episode"] + 1
            total_steps = meta["total_steps"]
            best_score = meta.get("best_score", -1)
            best_avg100 = meta.get("best_avg100")
            best_avg100_text = (
                f"{best_avg100:.2f}" if best_avg100 is not None else "n/a"
            )
            print(
                f"Resuming from checkpoint — "
                f"episode={start_ep}  "
                f"total_steps={total_steps}  "
                f"peak_score={best_score}  best_avg100={best_avg100_text}"
            )
        else:
            print("No checkpoint found — starting fresh.")
            start_ep = 1
            total_steps = 0
            best_score = -1
            best_avg100 = None

        # Compute the initial per-actor epsilon spread using the Ape-X formula
        current_epsilon_base = self.epsilon_base
        actor_epsilons = _ape_x_epsilons(
            self.num_actors, current_epsilon_base, self.epsilon_alpha
        )
        print("Actor epsilons: " + "  ".join(f"{e:.4f}" for e in actor_epsilons))

        buffer = PrioritizedReplayBuffer(
            cfg.buffer_size, alpha=self.per_alpha, beta=self.per_beta
        )
        print(
            f"Using PrioritizedReplayBuffer (alpha={self.per_alpha}, beta={self.per_beta})"
        )

        # 'spawn' context creates fresh interpreter per process — required for MLX
        # because Metal GPU contexts cannot be safely inherited by forked processes.
        ctx = mp.get_context("spawn")
        # maxsize=2 per actor queue: prevents the learner from queuing up stale
        # weight messages faster than actors can consume them.
        weight_queues = [ctx.Queue(maxsize=2) for _ in range(self.num_actors)]
        # Large enough to buffer bursts from all actors without dropping too many transitions
        transition_queue = ctx.Queue(maxsize=self.num_actors * 2_000)

        actors = []
        initial_weights = self.algo.get_weights()
        for i in range(self.num_actors):
            p = ctx.Process(
                target=_actor_fn,
                args=(
                    i,
                    self.env_factory,
                    self.algo_factory,
                    self.reward_shaper,
                    weight_queues[i],
                    transition_queue,
                    actor_epsilons[i],
                    cfg.train_start,
                    self.env_kwargs,
                    initial_weights,
                ),
                daemon=True,  # actors die automatically if the learner exits
            )
            p.start()
            actors.append(p)

        print(f"Started {self.num_actors} actor processes.")

        learner_updates = 0
        episode = start_ep
        recent_scores: list = []  # rolling window (last 100) for mean score display

        # Heartbeat: prints debug state every 10s so we can see where it's stuck
        # without the overhead of printing every iteration of the tight inner loop.
        _dbg = {"phase": "starting", "drained": 0, "updates": 0, "last_ep": episode}
        _stop_hb = threading.Event()

        def _heartbeat():
            while not _stop_hb.is_set():
                alive = sum(1 for p in actors if p.is_alive())
                print(
                    f"[HB] phase={_dbg['phase']}  "
                    f"buf={len(buffer)}  drained={_dbg['drained']}  "
                    f"updates={_dbg['updates']}  ep={_dbg['last_ep']}  "
                    f"actors_alive={alive}/{self.num_actors}"
                )
                _stop_hb.wait(10)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        hb.start()

        try:
            while episode < start_ep + cfg.max_episodes:
                _dbg["phase"] = "drain"
                # Drain the transition queue in batches of up to 512 at a time.
                # Capping the drain prevents the learner from spending all its time
                # ingesting data and starving the gradient update step.
                drained = 0
                while drained < 512:
                    try:
                        s, a, r, ns, done, score = transition_queue.get_nowait()
                        buffer.push(s, a, r, ns, done)
                        total_steps += 1
                        drained += 1
                        _dbg["drained"] = drained

                        if done and score is not None:
                            # Track peak score separately from the checkpoint
                            # criterion, which is the rolling Avg100 once the
                            # window is fully populated.
                            if score > best_score:
                                best_score = score

                            recent_scores.append(score)
                            if len(recent_scores) > 100:
                                recent_scores.pop(0)  # keep only last 100

                            avg = sum(recent_scores) / len(recent_scores)
                            avg_ready = len(recent_scores) == 100
                            is_best = avg_ready and (
                                best_avg100 is None or avg > best_avg100
                            )
                            if is_best:
                                best_avg100 = avg

                            if episode % cfg.log_every == 0:
                                best_avg_text = (
                                    f"{best_avg100:6.1f}"
                                    if best_avg100 is not None
                                    else "   n/a"
                                )
                                print(
                                    f"Episode {episode:6d} | "
                                    f"Score: {score:4d} | "
                                    f"Avg100: {avg:6.1f} | "
                                    f"BestAvg100: {best_avg_text} | "
                                    f"Peak: {best_score:4d} | "
                                    f"Buf: {len(buffer):6d} | "
                                    f"Upd: {learner_updates}"
                                )

                            _dbg["phase"] = "checkpoint"
                            self.checkpointer.save(
                                self.algo,
                                meta={
                                    "episode": episode,
                                    "total_steps": total_steps,
                                    "best_score": best_score,
                                    "best_avg100": round(best_avg100, 4)
                                    if best_avg100 is not None
                                    else None,
                                },
                                is_best=is_best,
                            )
                            episode += 1
                            _dbg["last_ep"] = episode
                            _dbg["phase"] = "drain"

                    except Exception:
                        break  # queue is empty — stop draining

                # Train the learner — run multiple gradient steps per drain cycle
                # so learning keeps pace with the actors' data production rate.
                if len(buffer) >= cfg.train_start:
                    _dbg["phase"] = "update"
                    for _ in range(self.updates_per_drain):
                        batch, tree_indices, is_weights = buffer.sample(cfg.batch_size)
                        _, td_errors = self.algo.update(batch, weights=is_weights)
                        # Feed the fresh TD errors back into PER so priorities
                        # reflect the current network's surprisal, not stale values.
                        buffer.update_priorities(tree_indices, td_errors)
                        learner_updates += 1
                        _dbg["updates"] = learner_updates

                    # Periodically push latest weights (and decayed epsilons) to actors
                    if learner_updates % self.weight_sync_freq == 0:
                        _dbg["phase"] = "weight_sync"
                        # Decay epsilon_base toward epsilon_base_min so actors
                        # gradually become more greedy as the policy matures.
                        current_epsilon_base = max(
                            self.epsilon_base_min,
                            current_epsilon_base * self.epsilon_base_decay,
                        )
                        actor_epsilons = _ape_x_epsilons(
                            self.num_actors, current_epsilon_base, self.epsilon_alpha
                        )
                        weights_np = (
                            self.algo.get_weights()
                        )  # numpy dict, safe to queue
                        for wq, eps in zip(weight_queues, actor_epsilons):
                            # Flush any unconsumed stale weight message before pushing
                            # the new one — maxsize=2 queues can still hold one stale msg.
                            while not wq.empty():
                                try:
                                    wq.get_nowait()
                                except Exception:
                                    break
                            try:
                                wq.put_nowait((weights_np, eps))
                            except Exception:
                                pass  # actor queue full — skip this sync cycle

                    # Release GPU memory pool periodically to avoid fragmentation
                    if learner_updates % 50 == 0:
                        _dbg["phase"] = "clear_cache"
                        mx.clear_cache()
                else:
                    # Buffer not warm yet — yield CPU time so actors can fill it faster
                    _dbg["phase"] = "waiting_buffer"
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopped by user. Latest checkpoint is saved.")
        finally:
            # Signal heartbeat thread to stop, then clean up all child processes and queues
            _stop_hb.set()
            for p in actors:
                p.terminate()
            for p in actors:
                p.join()
            # join_thread() flushes the queue's background feeder thread before closing
            transition_queue.close()
            transition_queue.join_thread()
            for wq in weight_queues:
                wq.close()
                wq.join_thread()

        best_avg100_text = f"{best_avg100:.2f}" if best_avg100 is not None else "n/a"
        print(
            f"\nCycle complete. Peak score: {best_score} | "
            f"BestAvg100: {best_avg100_text}"
        )

    def test(self, best: bool = False):
        """
        Run the greedy policy with rendering in a single process until KeyboardInterrupt.

        No actors are spawned — inference runs directly in the learner process
        so the display is accessible. Epsilon is 0 (always greedy).
        """
        if best:
            meta = self.checkpointer.load_best(self.algo)
            tag = "best"
        else:
            meta = self.checkpointer.load(self.algo)
            tag = "latest"

        if meta is None:
            print(f"No {tag} checkpoint found.")
            return

        print(
            f"Loaded {tag} checkpoint — "
            f"trained for {meta['episode']} episodes, "
            f"best_avg100: {meta.get('best_avg100', 'n/a')}  "
            f"peak_score: {meta.get('best_score', 'n/a')}"
        )

        episode = 0
        try:
            while True:
                env = self.env_factory(render_mode=True, **self.env_kwargs)
                state = env.reset()
                episode += 1
                while True:
                    action = self.algo.select_action(state)
                    state, _, done, info = env.step(action)
                    if done:
                        print(f"Episode {episode} — Score: {info['score']}")
                        break
                env.close()
        except KeyboardInterrupt:
            pass
