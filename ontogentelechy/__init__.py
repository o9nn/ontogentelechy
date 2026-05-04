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

__version__ = "1.0.0"

from .adapters import NumpyArrayEntity, RLAgentEntity, create_training_observer
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
from .llm_teloi import (
    create_factual_coherence_telos,
    create_instruction_following_telos,
    create_llm_telos,
    create_safety_alignment_telos,
    create_stylistic_consistency_telos,
    list_llm_teloi,
)
from .meta import (
    AutopoieticSystem,
    SwarmMetrics,
    SwarmTeleology,
    TelosDiscoveryConfig,
    TelosHierarchy,
    TelosNode,
    crossover_teloi,
    discover_telos,
    evolve_teloi,
    mutate_telos,
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
    # Adapters (Phase 4)
    "NumpyArrayEntity",
    "RLAgentEntity",
    "create_training_observer",
    # LLM Teloi (Phase 4)
    "create_instruction_following_telos",
    "create_factual_coherence_telos",
    "create_stylistic_consistency_telos",
    "create_safety_alignment_telos",
    "create_llm_telos",
    "list_llm_teloi",
    # Meta-Teleology (Phase 5)
    "TelosDiscoveryConfig",
    "discover_telos",
    "TelosNode",
    "TelosHierarchy",
    "mutate_telos",
    "crossover_teloi",
    "evolve_teloi",
    "AutopoieticSystem",
    "SwarmMetrics",
    "SwarmTeleology",
    # Benchmarks (Phase 6)
    "BenchmarkResult",
    "CoherentClusterBenchmark",
    "PhaseSeparationBenchmark",
    "EmergenceTrackingBenchmark",
    "run_benchmark",
    "list_benchmarks",
    # Registry (Phase 6)
    "registry",
    "TelosRegistry",
]
