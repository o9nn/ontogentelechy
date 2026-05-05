"""
Entity - Reference implementations of DevelopableEntity.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

import numpy as np


@dataclass
class SimpleGene:
    """Minimal gene implementation."""

    weight: float = 0.5
    active: bool = True


@dataclass
class SimpleEntity:
    """Reference implementation of DevelopableEntity."""

    state: np.ndarray
    genes: List[SimpleGene] = field(default_factory=list)
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    age: int = 0

    def get_state_vector(self) -> np.ndarray:
        """Return primary state vector, falling back to gene weights."""
        if self.state is not None and len(self.state) > 0:
            return cast(np.ndarray, np.array(self.state, dtype=np.float64))
        if self.genes:
            return cast(np.ndarray, np.array([g.weight for g in self.genes], dtype=np.float64))
        return cast(np.ndarray, np.array([], dtype=np.float64))

    def update_state(self, new_state: np.ndarray) -> None:
        """Update state and sync gene weights."""
        self.state = new_state
        if len(self.genes) == len(new_state):
            for gene, val in zip(self.genes, new_state):
                gene.weight = float(np.clip(val, 0.0, 1.0))
        self.age += 1

    @classmethod
    def random(cls, dim: int = 10) -> "SimpleEntity":
        """Create a SimpleEntity with random state."""
        state = np.random.uniform(0.0, 1.0, dim)
        genes = [SimpleGene(weight=float(v)) for v in state]
        return cls(state=state, genes=genes)

    @classmethod
    def from_weights(cls, weights: List[float]) -> "SimpleEntity":
        """Create a SimpleEntity from a list of weights."""
        state = np.array(weights)
        genes = [SimpleGene(weight=w) for w in weights]
        return cls(state=state, genes=genes)
