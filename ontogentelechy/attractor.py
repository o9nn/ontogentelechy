"""
Attractor Dynamics — Multi-basin landscape topology for developmental attractors.

Provides richer attractor mechanics than the DevelopmentalAttractor in core.py,
including multiple basins, repulsors, annealing, and saddle-point detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .core import Telos


@dataclass
class Basin:
    """A single basin of attraction in the landscape."""

    state: np.ndarray
    strength: float = 1.0
    radius: float = 0.5
    label: str = ""

    def distance_to(self, point: np.ndarray) -> float:
        min_len = min(len(self.state), len(point))
        return float(np.linalg.norm(self.state[:min_len] - point[:min_len]))

    def gradient_toward(self, point: np.ndarray) -> np.ndarray:
        min_len = min(len(self.state), len(point))
        target = self.state[:min_len]
        current = point[:min_len]
        distance = np.linalg.norm(target - current)
        if distance < 1e-10:
            return np.zeros_like(current)
        direction = (target - current) / distance
        pull = np.tanh(distance / self.radius) * self.strength
        return direction * pull


@dataclass
class Repulsor:
    """A repulsor state — an anti-pattern to avoid."""

    state: np.ndarray
    strength: float = 1.0
    radius: float = 0.3
    label: str = ""

    def gradient_away(self, point: np.ndarray) -> np.ndarray:
        min_len = min(len(self.state), len(point))
        source = self.state[:min_len]
        current = point[:min_len]
        distance = np.linalg.norm(current - source)
        if distance < 1e-10:
            random_dir = np.random.randn(min_len)
            random_dir /= np.linalg.norm(random_dir) + 1e-10
            return random_dir * self.strength
        direction = (current - source) / distance
        push = self.strength * np.exp(-distance / self.radius)
        return direction * push


class AttractorLandscape:
    """Multi-basin attractor landscape with repulsors and annealing."""

    def __init__(
        self,
        basins: Optional[List[Basin]] = None,
        repulsors: Optional[List[Repulsor]] = None,
        annealing_rate: float = 0.0,
    ):
        self.basins: List[Basin] = basins or []
        self.repulsors: List[Repulsor] = repulsors or []
        self.annealing_rate = annealing_rate
        self._step_count: int = 0
        self._initial_strengths: List[float] = [b.strength for b in self.basins]

    def add_basin(self, basin: Basin) -> None:
        self.basins.append(basin)
        self._initial_strengths.append(basin.strength)

    def add_repulsor(self, repulsor: Repulsor) -> None:
        self.repulsors.append(repulsor)

    def step(self) -> None:
        """Advance one developmental time step (for annealing)."""
        self._step_count += 1
        if self.annealing_rate > 0:
            decay = np.exp(-self.annealing_rate * self._step_count)
            for basin, initial in zip(self.basins, self._initial_strengths):
                basin.strength = initial * decay

    def net_gradient(self, point: np.ndarray) -> np.ndarray:
        """Compute net gradient as superposition of all basins and repulsors."""
        if not self.basins and not self.repulsors:
            return np.zeros_like(point)

        total = np.zeros(len(point))

        for basin in self.basins:
            grad = basin.gradient_toward(point)
            min_len = min(len(total), len(grad))
            total[:min_len] += grad[:min_len]

        for repulsor in self.repulsors:
            grad = repulsor.gradient_away(point)
            min_len = min(len(total), len(grad))
            total[:min_len] += grad[:min_len]

        return total

    def nearest_basin(self, point: np.ndarray) -> Optional[Basin]:
        """Find the basin nearest to point."""
        if not self.basins:
            return None
        return min(self.basins, key=lambda b: b.distance_to(point))

    def is_in_any_basin(self, point: np.ndarray) -> bool:
        """Check if point is within any basin's radius."""
        return any(b.distance_to(point) < b.radius for b in self.basins)

    def detect_saddle_points(
        self,
        n_samples: int = 1000,
        dim: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Detect approximate saddle points by finding near-zero gradient regions between basins.

        Uses random sampling in the convex hull of basins, looks for points where
        gradient magnitude is very small but not at any basin center.
        """
        if len(self.basins) < 2:
            return []

        if dim is None:
            dim = len(self.basins[0].state)

        saddle_points = []
        gradient_threshold = 0.05

        for _ in range(n_samples):
            weights = np.random.dirichlet(np.ones(len(self.basins)))
            point = sum(w * b.state[:dim] for w, b in zip(weights, self.basins))

            grad = self.net_gradient(point)
            grad_magnitude = np.linalg.norm(grad)

            if grad_magnitude < gradient_threshold and not self.is_in_any_basin(point):
                saddle_points.append(point.copy())

        return saddle_points

    def visualize(
        self,
        entity_states: Optional[List[np.ndarray]] = None,
        title: str = "Attractor Landscape",
        figsize: Tuple[int, int] = (10, 8),
    ):
        """Generate 2D PCA projection of the attractor landscape.

        Requires matplotlib. Returns matplotlib Figure object.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            )

        all_points = [b.state for b in self.basins] + [r.state for r in self.repulsors]
        if entity_states:
            all_points += entity_states

        if len(all_points) < 2:
            raise ValueError("Need at least 2 points for PCA projection")

        max_dim = max(len(p) for p in all_points)
        padded = np.array([np.pad(p, (0, max_dim - len(p))) for p in all_points])

        centered = padded - padded.mean(axis=0)
        if centered.shape[1] >= 2:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            coords_2d = centered @ Vt[:2].T
        else:
            coords_2d = np.hstack([centered, np.zeros((len(centered), 2 - centered.shape[1]))])

        fig, ax = plt.subplots(figsize=figsize)

        n_basins = len(self.basins)
        n_repulsors = len(self.repulsors)

        for i, (basin, coord) in enumerate(zip(self.basins, coords_2d[:n_basins])):
            ax.scatter(
                *coord,
                s=200,
                marker="*",
                color="gold",
                zorder=5,
                label=f"Basin: {basin.label or i}",
            )
            circle = Circle(
                coord, basin.radius * 0.5, fill=False, color="green", alpha=0.5, linestyle="--"
            )
            ax.add_patch(circle)
            ax.annotate(
                basin.label or f"B{i}", coord, textcoords="offset points", xytext=(5, 5), fontsize=9
            )

        for i, (rep, coord) in enumerate(
            zip(self.repulsors, coords_2d[n_basins : n_basins + n_repulsors])
        ):
            ax.scatter(
                *coord,
                s=200,
                marker="X",
                color="red",
                zorder=5,
                label=f"Repulsor: {rep.label or i}",
            )
            circle = Circle(
                coord, rep.radius * 0.5, fill=False, color="red", alpha=0.5, linestyle="--"
            )
            ax.add_patch(circle)

        if entity_states:
            entity_coords = coords_2d[n_basins + n_repulsors :]
            ax.plot(
                entity_coords[:, 0],
                entity_coords[:, 1],
                "b-o",
                alpha=0.6,
                markersize=4,
                label="Entity trajectory",
            )
            ax.scatter(*entity_coords[0], s=100, marker="o", color="blue", label="Start", zorder=6)
            ax.scatter(*entity_coords[-1], s=100, marker="s", color="navy", label="End", zorder=6)

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @classmethod
    def from_telos(cls, telos: "Telos", annealing_rate: float = 0.0) -> "AttractorLandscape":
        """Create an AttractorLandscape from a Telos's attractor_state."""
        from .core import Telos as _Telos  # noqa: F401 — keep local import for type check

        attractor = telos.attractor_state
        if isinstance(attractor, dict):
            numeric_vals = [v for v in attractor.values() if isinstance(v, (int, float))]
            state = np.array(numeric_vals) if numeric_vals else np.zeros(1)
        else:
            state = np.array(attractor)

        basin = Basin(state=state, strength=1.0, radius=0.5, label=telos.name)
        return cls(basins=[basin], annealing_rate=annealing_rate)
