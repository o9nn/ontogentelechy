"""
Telos Module - Teleological Development and Actualization

This module implements ontogentelechy: the integration of ontogenesis (development),
teleology (purpose), and entelechy (actualization) for creating self-organizing,
purpose-driven cognitive systems.

Key Concepts:
- Telos: Intrinsic purpose or final cause
- Actualization: Progress toward realizing potential
- Developmental Attractor: Stable configuration toward which development tends
- Phase Transitions: Qualitative changes in developmental state
- Emergence: New properties arising from development

Usage:
    from ontogentelechy import Telos, Criterion, TeleologicalFitness

    # Define a telos
    telos = Telos(
        name="semantic_coherence",
        description="Achieve coherent semantic representation",
        actualization_criteria=[
            Criterion("coherence", "Semantic coherence", 0.5, evaluate_coherence),
            Criterion("integration", "Component integration", 0.3, evaluate_integration),
            Criterion("emergence", "Emergent properties", 0.2, evaluate_emergence)
        ],
        attractor_state={'weights': [0.8, 0.7, 0.9]}
    )

    # Use in fitness evaluation
    fitness = TeleologicalFitness(atomspace, telos)
    score = fitness.evaluate(individual)
"""

__version__ = "0.3.0"

from .attractor import AttractorLandscape, Basin, Repulsor
from .core import (
    ActualizationMetrics,
    ActualizationPhase,
    ActualizationTracker,
    Criterion,
    DevelopmentalAttractor,
    Telos,
)
from .emergence import EmergenceDetector
from .entity import SimpleEntity, SimpleGene
from .evolution import (
    EvoIndividual,
    IslandModel,
    OntogeneticStage,
    ParetoEvolution,
    TelosPopulation,
)
from .fitness import (
    MultiTelosFitness,
    TeleologicalFitness,
)
from .protocols import (
    AtomSpace,
    DevelopableEntity,
    FitnessFunction,
    Gene,
    Individual,
    OntogeneticState,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Telos",
    "Criterion",
    "ActualizationPhase",
    "ActualizationMetrics",
    "ActualizationTracker",
    "DevelopmentalAttractor",
    # Fitness functions
    "TeleologicalFitness",
    "MultiTelosFitness",
    # Protocol classes
    "FitnessFunction",
    "Individual",
    "AtomSpace",
    "Gene",
    "OntogeneticState",
    "DevelopableEntity",
    # Entity implementations
    "SimpleEntity",
    "SimpleGene",
    # Emergence detection
    "EmergenceDetector",
    # Attractor dynamics
    "Basin",
    "Repulsor",
    "AttractorLandscape",
    # Evolution engine
    "OntogeneticStage",
    "EvoIndividual",
    "TelosPopulation",
    "IslandModel",
    "ParetoEvolution",
]
