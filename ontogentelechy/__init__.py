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
    from cogprime.telos import Telos, Criterion, TeleologicalFitness
    
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

from .core import (
    Telos,
    Criterion,
    ActualizationPhase,
    ActualizationMetrics,
    ActualizationTracker,
    DevelopmentalAttractor,
)

# Fitness functions require full package installation
# from .fitness import (
#     TeleologicalFitness,
#     MultiTelosFitness,
# )

__all__ = [
    # Core classes
    "Telos",
    "Criterion",
    "ActualizationPhase",
    "ActualizationMetrics",
    "ActualizationTracker",
    "DevelopmentalAttractor",
    # Fitness functions (require full package installation)
    # "TeleologicalFitness",
    # "MultiTelosFitness",
]
