"""
Example Teloi - Common Cognitive Purposes

This module provides pre-defined teloi for common cognitive purposes
that can be used as templates or starting points.
"""

import numpy as np
from typing import Any
from .core import Telos, Criterion


def create_semantic_coherence_telos() -> Telos:
    """Create a telos for semantic coherence.
    
    Purpose: Achieve coherent semantic representation where concepts
    are meaningfully related and integrated.
    
    Returns:
        Telos for semantic coherence
    """
    def evaluate_coherence(entity: Any) -> float:
        """Evaluate semantic coherence."""
        if hasattr(entity, 'genes'):
            # Check gene weight consistency
            weights = [g.weight for g in entity.genes if hasattr(g, 'weight')]
            if weights:
                # Low variance = high coherence
                variance = np.var(weights)
                return 1.0 - min(variance, 1.0)
        return 0.5
    
    def evaluate_integration(entity: Any) -> float:
        """Evaluate component integration."""
        if hasattr(entity, 'genes'):
            active_count = sum(1 for g in entity.genes if getattr(g, 'active', True))
            total_count = len(entity.genes)
            if total_count > 0:
                return active_count / total_count
        return 0.5
    
    def evaluate_meaning(entity: Any) -> float:
        """Evaluate meaningful connections."""
        if hasattr(entity, 'to_atomspace_structure'):
            try:
                atoms = entity.to_atomspace_structure()
                # More atoms = more meaning
                return min(len(atoms) / 10.0, 1.0)
            except:
                pass
        return 0.5
    
    return Telos(
        name="semantic_coherence",
        description="Achieve coherent semantic representation with meaningful relationships",
        actualization_criteria=[
            Criterion("coherence", "Semantic coherence", 0.4, evaluate_coherence, 1.0),
            Criterion("integration", "Component integration", 0.3, evaluate_integration, 1.0),
            Criterion("meaning", "Meaningful connections", 0.3, evaluate_meaning, 1.0)
        ],
        attractor_state={'weights': [0.8] * 5, 'coherence': 0.9}
    )


def create_adaptive_learning_telos() -> Telos:
    """Create a telos for adaptive learning.
    
    Purpose: Continuously learn and adapt to new information while
    maintaining stability.
    
    Returns:
        Telos for adaptive learning
    """
    def evaluate_adaptability(entity: Any) -> float:
        """Evaluate adaptability."""
        if hasattr(entity, 'metadata') and 'fitness_history' in entity.metadata:
            history = entity.metadata['fitness_history']
            if len(history) > 1:
                # Positive trend = good adaptability
                improvements = sum(1 for i in range(1, len(history)) 
                                 if history[i] > history[i-1])
                return improvements / (len(history) - 1)
        return 0.5
    
    def evaluate_stability(entity: Any) -> float:
        """Evaluate stability."""
        if hasattr(entity, 'genes'):
            weights = [g.weight for g in entity.genes if hasattr(g, 'weight')]
            if weights:
                # Weights in middle range = stable
                avg_weight = np.mean(weights)
                stability = 1.0 - abs(avg_weight - 0.5) * 2.0
                return max(0.0, stability)
        return 0.5
    
    def evaluate_learning_rate(entity: Any) -> float:
        """Evaluate learning rate."""
        if hasattr(entity, 'age'):
            # Optimal learning rate decreases with age
            optimal_rate = 1.0 / (1.0 + entity.age * 0.1)
            return optimal_rate
        return 0.5
    
    return Telos(
        name="adaptive_learning",
        description="Continuously learn and adapt while maintaining stability",
        actualization_criteria=[
            Criterion("adaptability", "Ability to adapt", 0.4, evaluate_adaptability, 1.0),
            Criterion("stability", "Maintain stability", 0.3, evaluate_stability, 1.0),
            Criterion("learning_rate", "Optimal learning rate", 0.3, evaluate_learning_rate, 1.0)
        ],
        attractor_state={'weights': [0.5] * 5, 'adaptability': 0.8, 'stability': 0.7}
    )


def create_complexity_emergence_telos() -> Telos:
    """Create a telos for complexity emergence.
    
    Purpose: Develop increasingly complex structures and capabilities
    through emergence.
    
    Returns:
        Telos for complexity emergence
    """
    def evaluate_complexity(entity: Any) -> float:
        """Evaluate structural complexity."""
        if hasattr(entity, 'genes'):
            # More genes = more complexity
            complexity = min(len(entity.genes) / 20.0, 1.0)
            return complexity
        return 0.5
    
    def evaluate_emergence(entity: Any) -> float:
        """Evaluate emergent properties."""
        if hasattr(entity, 'ontogenetic_state') and hasattr(entity.ontogenetic_state, 'development_history'):
            history = entity.ontogenetic_state.development_history
            # Count emergence events
            emergence_events = [e for e in history 
                              if e.get('type') in ['property_emergence', 'new_capability']]
            return min(len(emergence_events) / 5.0, 1.0)
        return 0.3
    
    def evaluate_integration(entity: Any) -> float:
        """Evaluate integration of complexity."""
        if hasattr(entity, 'genes'):
            active_genes = [g for g in entity.genes if getattr(g, 'active', True)]
            if entity.genes:
                # High ratio of active genes = good integration
                return len(active_genes) / len(entity.genes)
        return 0.5
    
    return Telos(
        name="complexity_emergence",
        description="Develop complex structures through emergence and integration",
        actualization_criteria=[
            Criterion("complexity", "Structural complexity", 0.3, evaluate_complexity, 1.0),
            Criterion("emergence", "Emergent properties", 0.4, evaluate_emergence, 1.0),
            Criterion("integration", "Integrated complexity", 0.3, evaluate_integration, 1.0)
        ],
        attractor_state={'weights': np.linspace(0.3, 0.9, 10).tolist(), 'complexity': 0.8}
    )


def create_efficient_computation_telos() -> Telos:
    """Create a telos for efficient computation.
    
    Purpose: Achieve computational efficiency while maintaining
    effectiveness.
    
    Returns:
        Telos for efficient computation
    """
    def evaluate_efficiency(entity: Any) -> float:
        """Evaluate computational efficiency."""
        if hasattr(entity, 'grip_metrics'):
            return entity.grip_metrics.efficiency
        if hasattr(entity, 'genes'):
            # Fewer active genes = more efficient
            active_count = sum(1 for g in entity.genes if getattr(g, 'active', True))
            if entity.genes:
                efficiency = 1.0 - (active_count / len(entity.genes))
                return max(0.3, efficiency)  # Minimum 0.3
        return 0.5
    
    def evaluate_effectiveness(entity: Any) -> float:
        """Evaluate computational effectiveness."""
        if hasattr(entity, 'fitness'):
            return entity.fitness or 0.5
        if hasattr(entity, 'genome') and hasattr(entity.genome, 'fitness'):
            return entity.genome.fitness
        return 0.5
    
    def evaluate_stability(entity: Any) -> float:
        """Evaluate numerical stability."""
        if hasattr(entity, 'grip_metrics'):
            return entity.grip_metrics.stability
        if hasattr(entity, 'coefficients'):
            # Check for numerical issues
            coeffs = entity.coefficients
            if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
                return 0.0
            # Moderate values = stable
            max_coeff = np.max(np.abs(coeffs))
            if max_coeff < 100.0:
                return 1.0
            else:
                return 1.0 / (1.0 + np.log10(max_coeff / 100.0))
        return 0.5
    
    return Telos(
        name="efficient_computation",
        description="Achieve computational efficiency while maintaining effectiveness",
        actualization_criteria=[
            Criterion("efficiency", "Computational efficiency", 0.4, evaluate_efficiency, 1.0),
            Criterion("effectiveness", "Computational effectiveness", 0.4, evaluate_effectiveness, 1.0),
            Criterion("stability", "Numerical stability", 0.2, evaluate_stability, 1.0)
        ],
        attractor_state={'weights': [0.6] * 3, 'efficiency': 0.9}
    )


def create_knowledge_integration_telos() -> Telos:
    """Create a telos for knowledge integration.
    
    Purpose: Integrate diverse knowledge into a coherent whole.
    
    Returns:
        Telos for knowledge integration
    """
    def evaluate_diversity(entity: Any) -> float:
        """Evaluate knowledge diversity."""
        if hasattr(entity, 'genes'):
            # Different atom types = diverse knowledge
            types = set(g.atom.atom_type for g in entity.genes if hasattr(g, 'atom'))
            if entity.genes:
                return len(types) / len(entity.genes)
        return 0.5
    
    def evaluate_coherence(entity: Any) -> float:
        """Evaluate knowledge coherence."""
        if hasattr(entity, 'genes'):
            weights = [g.weight for g in entity.genes if hasattr(g, 'weight')]
            if weights:
                # Consistent weights = coherent
                variance = np.var(weights)
                return 1.0 - min(variance, 1.0)
        return 0.5
    
    def evaluate_connectivity(entity: Any) -> float:
        """Evaluate knowledge connectivity."""
        if hasattr(entity, 'to_atomspace_structure'):
            try:
                atoms = entity.to_atomspace_structure()
                # Count links (connections)
                links = [a for a in atoms if a.is_link()]
                if atoms:
                    return len(links) / len(atoms)
            except:
                pass
        return 0.5
    
    return Telos(
        name="knowledge_integration",
        description="Integrate diverse knowledge into coherent whole",
        actualization_criteria=[
            Criterion("diversity", "Knowledge diversity", 0.3, evaluate_diversity, 1.0),
            Criterion("coherence", "Knowledge coherence", 0.4, evaluate_coherence, 1.0),
            Criterion("connectivity", "Knowledge connectivity", 0.3, evaluate_connectivity, 1.0)
        ],
        attractor_state={'weights': [0.7] * 7, 'integration': 0.85}
    )


# Dictionary of example teloi for easy access
EXAMPLE_TELOI = {
    'semantic_coherence': create_semantic_coherence_telos,
    'adaptive_learning': create_adaptive_learning_telos,
    'complexity_emergence': create_complexity_emergence_telos,
    'efficient_computation': create_efficient_computation_telos,
    'knowledge_integration': create_knowledge_integration_telos,
}


def get_example_telos(name: str) -> Telos:
    """Get an example telos by name.
    
    Args:
        name: Name of the telos
        
    Returns:
        The requested telos
        
    Raises:
        KeyError: If telos name not found
    """
    if name not in EXAMPLE_TELOI:
        available = ', '.join(EXAMPLE_TELOI.keys())
        raise KeyError(f"Unknown telos '{name}'. Available: {available}")
    
    return EXAMPLE_TELOI[name]()


def list_example_teloi() -> list[str]:
    """List all available example teloi.
    
    Returns:
        List of telos names
    """
    return list(EXAMPLE_TELOI.keys())
