"""
Emergence - Stateful detection of emergent phenomena.
"""

from typing import List

import numpy as np


class EmergenceDetector:
    """Stateful detector for emergent phenomena using sliding-window statistics."""

    def __init__(self, window_size: int = 20, novelty_threshold: float = 0.3) -> None:
        self.window_size = window_size
        self.novelty_threshold = novelty_threshold
        self._state_history: List[np.ndarray] = []
        self._entropy_history: List[float] = []

    def record(self, state: np.ndarray) -> None:
        """Record a new state observation."""
        self._state_history.append(state.copy())
        entropy = self._compute_entropy(state)
        self._entropy_history.append(entropy)

    def _compute_entropy(self, state: np.ndarray) -> float:
        """Compute approximate Shannon entropy of state distribution."""
        hist, _ = np.histogram(state, bins=10, range=(0.0, 1.0), density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        return float(-np.sum(hist * np.log(hist)))

    def entropy_change(self) -> float:
        """Rate of change in entropy over the window."""
        if len(self._entropy_history) < 2:
            return 0.0
        window = self._entropy_history[-self.window_size :]
        if len(window) < 2:
            return 0.0
        changes = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
        return float(np.mean(changes))

    def lyapunov_estimate(self) -> float:
        """Estimate local Lyapunov exponent (divergence rate)."""
        if len(self._state_history) < 3:
            return 0.0
        window = self._state_history[-self.window_size :]
        if len(window) < 3:
            return 0.0
        divergences = []
        for i in range(1, len(window)):
            d0 = np.linalg.norm(window[i] - window[i - 1])
            if i > 1:
                d_prev = np.linalg.norm(window[i - 1] - window[i - 2])
                if d_prev > 1e-10:
                    divergences.append(np.log(d0 / d_prev))
        if not divergences:
            return 0.0
        return float(np.mean(divergences))

    def novelty_score(self, state: np.ndarray) -> float:
        """How novel is this state relative to all seen states?"""
        if len(self._state_history) < 2:
            return 1.0
        past = self._state_history[:-1]
        distances = [np.linalg.norm(state - s) for s in past]
        min_dist = min(distances)
        return float(np.tanh(min_dist))

    def emergence_score(self) -> float:
        """Composite emergence score combining entropy, Lyapunov and novelty signals."""
        entropy_signal = np.tanh(self.entropy_change() * 5.0)
        lyapunov_signal = np.tanh(max(0.0, self.lyapunov_estimate()))

        if self._state_history:
            novelty_signal = self.novelty_score(self._state_history[-1])
        else:
            novelty_signal = 0.0

        return float(entropy_signal * 0.3 + lyapunov_signal * 0.4 + novelty_signal * 0.3)

    def reset(self) -> None:
        """Clear all recorded history."""
        self._state_history.clear()
        self._entropy_history.clear()
