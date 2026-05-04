"""
Cognitive Architecture Adapters

Wrappers that adapt existing ML objects (numpy arrays, RL agents, etc.)
to the DevelopableEntity protocol, enabling teleological tracking.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from .core import ActualizationMetrics, ActualizationTracker, Telos
from .entity import SimpleGene


class NumpyArrayEntity:
    """Adapts a numpy weight array as a DevelopableEntity.

    Use this to track neural network weights or any numpy array
    through the actualization framework.
    """

    def __init__(
        self,
        weights: np.ndarray,
        name: str = "array_entity",
        flatten: bool = True,
    ):
        self.name = name
        self._weights = weights.flatten().copy() if flatten else weights.copy()
        self.genes: List[SimpleGene] = [
            SimpleGene(weight=float(np.clip(v, 0.0, 1.0))) for v in self._normalize(self._weights)
        ]
        self.fitness: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
        self._weight_history: List[np.ndarray] = [self._weights.copy()]
        self.age: int = 0

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr, dtype=float)
        return (arr - mn) / (mx - mn)

    @property
    def state(self) -> np.ndarray:
        return self._normalize(self._weights)

    def get_state_vector(self) -> np.ndarray:
        return self.state

    def get_active_genes(self) -> List[SimpleGene]:
        return [g for g in self.genes if g.active]

    def update_weights(self, new_weights: np.ndarray, loss: Optional[float] = None) -> None:
        """Update the underlying weight array (e.g., after a training step)."""
        self._weights = new_weights.flatten().copy()
        normalized = self._normalize(self._weights)
        self.genes = [SimpleGene(weight=float(v)) for v in normalized]
        self._weight_history.append(self._weights.copy())
        self.age += 1
        if loss is not None:
            self.metadata["loss"] = loss
            self.metadata.setdefault("loss_history", []).append(loss)
            # Set fitness as negative loss (higher = better)
            self.fitness = float(np.exp(-abs(loss)))

    @property
    def raw_weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def weight_history(self) -> List[np.ndarray]:
        return list(self._weight_history)


class RLAgentEntity:
    """Adapts a reinforcement learning agent's policy as a DevelopableEntity.

    Tracks reward trajectory as the actualization signal.
    Attractor state encodes the optimal policy.
    """

    def __init__(
        self,
        policy_vector: np.ndarray,
        name: str = "rl_agent",
    ):
        self.name = name
        self._policy = policy_vector.copy()
        self.genes: List[SimpleGene] = [
            SimpleGene(weight=float(np.clip(v, 0.0, 1.0))) for v in self._policy
        ]
        self.fitness: Optional[float] = None
        self.metadata: Dict[str, Any] = {
            "reward_history": [],
            "episode": 0,
        }
        self.age: int = 0

    @property
    def state(self) -> np.ndarray:
        return np.clip(self._policy, 0.0, 1.0)

    def get_state_vector(self) -> np.ndarray:
        return self.state

    def get_active_genes(self) -> List[SimpleGene]:
        return [g for g in self.genes if g.active]

    def record_episode(
        self, total_reward: float, policy_update: Optional[np.ndarray] = None
    ) -> None:
        """Record an episode's reward and optionally update the policy."""
        self.metadata["reward_history"].append(total_reward)
        self.metadata["episode"] += 1
        self.age += 1

        # Update fitness as running average reward
        history = self.metadata["reward_history"]
        window = history[-20:]  # last 20 episodes
        avg_reward = np.mean(window)
        self.fitness = float(np.tanh(avg_reward / max(abs(avg_reward) + 1e-10, 1.0)) * 0.5 + 0.5)

        if policy_update is not None:
            self._policy = np.clip(policy_update, 0.0, 1.0)
            self.genes = [SimpleGene(weight=float(v)) for v in self._policy]

    @property
    def cumulative_reward(self) -> float:
        return float(sum(self.metadata.get("reward_history", [])))

    @property
    def recent_avg_reward(self) -> float:
        history = self.metadata.get("reward_history", [])
        if not history:
            return 0.0
        return float(np.mean(history[-20:]))


def create_training_observer(
    entity: NumpyArrayEntity,
    telos: Telos,
    tracker: Optional[ActualizationTracker] = None,
) -> Iterator[ActualizationMetrics]:
    """Generator that yields metrics after each weight update.

    Usage:
        observer = create_training_observer(entity, telos)
        for step in range(100):
            # do training step
            entity.update_weights(new_weights, loss=current_loss)
            metrics = next(observer)
            print(f"Step {step}: actualization={metrics.actualization:.3f}")
    """
    if tracker is None:
        tracker = ActualizationTracker()

    while True:
        metrics = tracker.compute_metrics(entity, telos)
        yield metrics
