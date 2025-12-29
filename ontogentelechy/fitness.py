"""
Teleological Fitness - Purpose-Driven Fitness Evaluation

This module implements fitness functions that evaluate individuals based on
their progress toward actualization of intrinsic purpose (telos).
"""

import numpy as np
from typing import Dict, Any, Optional, List
try:
    from ..evolution.moses_engine import FitnessFunction, Individual
    from ..atomspace import AtomSpace
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evolution.moses_engine import FitnessFunction, Individual
    from atomspace import AtomSpace
from .core import Telos, ActualizationTracker, ActualizationMetrics


class TeleologicalFitness(FitnessFunction):
    """Fitness evaluation based on telos actualization.
    
    This fitness function evaluates individuals not just on semantic coherence
    and diversity, but on their progress toward actualizing their intrinsic purpose.
    """
    
    def __init__(self, atomspace: AtomSpace, telos: Telos):
        """Initialize teleological fitness function.
        
        Args:
            atomspace: The AtomSpace for semantic context
            telos: The telos defining the purpose
        """
        self.atomspace = atomspace
        self.telos = telos
        self.tracker = ActualizationTracker()
        
        # Fitness component weights
        self.base_weight = 0.25  # Base semantic fitness
        self.telos_weight = 0.30  # Telos alignment
        self.actualization_weight = 0.25  # Actualization progress
        self.emergence_weight = 0.20  # Emergent properties
    
    def evaluate(self, individual: Individual, context: Dict[str, Any] = None) -> float:
        """Evaluate fitness based on teleological criteria.
        
        Args:
            individual: The individual to evaluate
            context: Optional context for evaluation
            
        Returns:
            Fitness score (0.0 to 1.0)
        """
        if not individual.genes:
            return 0.0
        
        # Store previous fitness for emergence detection
        if 'previous_fitness' not in individual.metadata:
            individual.metadata['previous_fitness'] = individual.fitness or 0.0
        
        # 1. Base semantic fitness
        base_fitness = self._evaluate_base_fitness(individual)
        
        # 2. Telos alignment score
        telos_score = self._evaluate_telos_alignment(individual)
        
        # 3. Actualization progress
        actualization_score = self._evaluate_actualization(individual)
        
        # 4. Emergent properties
        emergence_score = self._evaluate_emergence(individual)
        
        # Combine components
        total_fitness = (
            self.base_weight * base_fitness +
            self.telos_weight * telos_score +
            self.actualization_weight * actualization_score +
            self.emergence_weight * emergence_score
        )
        
        # Update metadata
        individual.metadata['previous_fitness'] = total_fitness
        individual.metadata['fitness_components'] = {
            'base': base_fitness,
            'telos': telos_score,
            'actualization': actualization_score,
            'emergence': emergence_score
        }
        
        return max(0.0, min(1.0, total_fitness))
    
    def _evaluate_base_fitness(self, individual: Individual) -> float:
        """Evaluate base semantic fitness.
        
        This is similar to the original SemanticAwarenessFitness.
        """
        active_genes = individual.get_active_genes()
        
        if not active_genes:
            return 0.0
        
        # Base fitness from gene weights and activity
        base_fitness = sum(gene.weight for gene in active_genes) / len(active_genes)
        
        # Semantic coherence
        semantic_score = self._evaluate_semantic_coherence(active_genes)
        
        # Diversity bonus
        diversity_score = self._evaluate_diversity(active_genes)
        
        # Combine
        fitness = (0.4 * base_fitness + 
                  0.4 * semantic_score + 
                  0.2 * diversity_score)
        
        return fitness
    
    def _evaluate_semantic_coherence(self, genes: List) -> float:
        """Evaluate semantic coherence of genes."""
        if len(genes) <= 1:
            return 1.0
        
        coherence = 0.0
        comparisons = 0
        
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene1, gene2 = genes[i], genes[j]
                
                # Check for semantic relationships
                similarity = self._compute_atom_similarity(gene1.atom, gene2.atom)
                coherence += similarity
                comparisons += 1
        
        return coherence / comparisons if comparisons > 0 else 0.0
    
    def _evaluate_diversity(self, genes: List) -> float:
        """Evaluate diversity of gene types."""
        if not genes:
            return 0.0
        
        atom_types = set(gene.atom.atom_type for gene in genes)
        return len(atom_types) / len(genes)
    
    def _compute_atom_similarity(self, atom1, atom2) -> float:
        """Compute similarity between two atoms."""
        if atom1.atom_type == atom2.atom_type:
            if atom1.is_node() and atom2.is_node():
                return 0.8 if atom1.name == atom2.name else 0.5
            elif atom1.is_link() and atom2.is_link():
                if len(atom1.outgoing_set) == len(atom2.outgoing_set):
                    return 0.7
                else:
                    return 0.3
            return 0.5
        return 0.2
    
    def _evaluate_telos_alignment(self, individual: Individual) -> float:
        """Evaluate alignment with telos.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Telos alignment score (0.0 to 1.0)
        """
        # Use tracker to compute alignment
        metrics = self.tracker.compute_metrics(individual, self.telos)
        return metrics.telos_alignment
    
    def _evaluate_actualization(self, individual: Individual) -> float:
        """Evaluate actualization progress.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Actualization score (0.0 to 1.0)
        """
        # Use tracker to measure actualization
        metrics = self.tracker.compute_metrics(individual, self.telos)
        
        # Combine actualization with integration
        actualization = (
            0.6 * metrics.actualization +
            0.4 * metrics.integration
        )
        
        return actualization
    
    def _evaluate_emergence(self, individual: Individual) -> float:
        """Evaluate emergent properties.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Emergence score (0.0 to 1.0)
        """
        # Use tracker to measure emergence
        metrics = self.tracker.compute_metrics(individual, self.telos)
        
        # Check for phase transitions
        transition = self.tracker.detect_phase_transition()
        
        # Base emergence score
        emergence = metrics.emergence
        
        # Bonus for phase transitions
        if transition is not None:
            emergence *= 1.5  # 50% bonus
        
        # Bonus for high potentiality (unexplored potential is valuable)
        emergence += metrics.potentiality * 0.2
        
        return min(emergence, 1.0)
    
    def get_actualization_metrics(self, individual: Individual) -> ActualizationMetrics:
        """Get detailed actualization metrics for an individual.
        
        Args:
            individual: Individual to analyze
            
        Returns:
            Complete actualization metrics
        """
        return self.tracker.compute_metrics(individual, self.telos)
    
    def get_phase_transitions(self) -> List[Dict[str, Any]]:
        """Get history of detected phase transitions.
        
        Returns:
            List of phase transition events
        """
        return self.tracker.phase_transitions


class MultiTelosFitness(FitnessFunction):
    """Fitness evaluation with multiple teloi (plural of telos).
    
    This allows an individual to pursue multiple purposes simultaneously,
    with different weights for each telos.
    """
    
    def __init__(self, atomspace: AtomSpace, teloi: List[tuple[Telos, float]]):
        """Initialize multi-telos fitness function.
        
        Args:
            atomspace: The AtomSpace for semantic context
            teloi: List of (telos, weight) tuples
        """
        self.atomspace = atomspace
        self.teloi = teloi
        self.fitness_functions = [
            TeleologicalFitness(atomspace, telos) 
            for telos, _ in teloi
        ]
        
        # Normalize weights
        total_weight = sum(weight for _, weight in teloi)
        self.weights = [weight / total_weight for _, weight in teloi]
    
    def evaluate(self, individual: Individual, context: Dict[str, Any] = None) -> float:
        """Evaluate fitness across multiple teloi.
        
        Args:
            individual: The individual to evaluate
            context: Optional context for evaluation
            
        Returns:
            Combined fitness score (0.0 to 1.0)
        """
        # Evaluate each telos
        scores = []
        for fitness_fn, weight in zip(self.fitness_functions, self.weights):
            score = fitness_fn.evaluate(individual, context)
            weighted_score = score * weight
            scores.append(weighted_score)
        
        # Combine scores
        total_fitness = sum(scores)
        
        # Store component scores in metadata
        individual.metadata['multi_telos_scores'] = {
            f'telos_{i}': score 
            for i, score in enumerate(scores)
        }
        
        return max(0.0, min(1.0, total_fitness))
    
    def get_dominant_telos(self, individual: Individual) -> tuple[Telos, float]:
        """Determine which telos the individual is most aligned with.
        
        Args:
            individual: Individual to analyze
            
        Returns:
            Tuple of (dominant telos, alignment score)
        """
        best_score = 0.0
        best_telos = None
        
        for fitness_fn, (telos, _) in zip(self.fitness_functions, self.teloi):
            score = fitness_fn.evaluate(individual)
            if score > best_score:
                best_score = score
                best_telos = telos
        
        return best_telos, best_score
