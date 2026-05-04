"""
Protocols - Formal interfaces for ontogentelechy components.

Defines typing.Protocol classes to replace external cogprime dependencies.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FitnessFunction(Protocol):
    """Protocol for fitness evaluation functions."""

    def evaluate(self, individual: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate fitness of an individual.

        Args:
            individual: The individual to evaluate
            context: Optional evaluation context

        Returns:
            Fitness score (0.0 to 1.0)
        """
        ...


@runtime_checkable
class Individual(Protocol):
    """Protocol for evolvable individuals."""

    genes: List[Any]
    fitness: Optional[float]
    metadata: Dict[str, Any]

    def get_active_genes(self) -> List[Any]:
        """Return only active genes."""
        ...


@runtime_checkable
class AtomSpace(Protocol):
    """Minimal protocol for an AtomSpace container."""

    def get_atoms(self) -> List[Any]:
        """Return all atoms in the space."""
        ...

    def add_atom(self, atom: Any) -> Any:
        """Add an atom and return it."""
        ...


@runtime_checkable
class Gene(Protocol):
    """Protocol for evolvable gene units."""

    weight: float
    active: bool


@runtime_checkable
class OntogeneticState(Protocol):
    """Protocol for an entity's developmental state."""

    maturity: float
    stage: Any
    development_history: List[Dict[str, Any]]


@runtime_checkable
class DevelopableEntity(Protocol):
    """Formal protocol for entities that can undergo teleological development."""

    @property
    def genes(self) -> List[Gene]:
        """List of genes."""
        ...

    @property
    def fitness(self) -> Optional[float]:
        """Current fitness value."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Arbitrary metadata dictionary."""
        ...

    def get_state_vector(self) -> "np.ndarray":
        """Return the entity's state as a 1-D numpy array."""
        ...
