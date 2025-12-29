"""
Telos Core - Intrinsic Purpose and Actualization

This module implements the core concepts of teleology (purpose), entelechy (actualization),
and developmental attractors for the ontogentelechy framework.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class ActualizationPhase(Enum):
    """Phases of actualization from potential to realized."""
    POTENTIAL = "potential"  # 0.0-0.2: Unrealized possibilities
    EMERGENT = "emergent"  # 0.2-0.4: New properties appearing
    DEVELOPING = "developing"  # 0.4-0.6: Integration and growth
    ACTUALIZING = "actualizing"  # 0.6-0.8: Refinement and optimization
    ACTUALIZED = "actualized"  # 0.8-1.0: Full realization


@dataclass
class Criterion:
    """Criterion for measuring actualization progress."""
    name: str
    description: str
    weight: float  # Importance weight (0.0 to 1.0)
    evaluator: Callable[[Any], float]  # Function to evaluate entity
    target_value: float = 1.0  # Target value for full actualization
    
    def evaluate(self, entity: Any) -> float:
        """Evaluate this criterion for an entity."""
        try:
            raw_score = self.evaluator(entity)
            # Normalize to 0-1 range
            normalized = min(max(raw_score, 0.0), 1.0)
            return normalized
        except Exception as e:
            # If evaluation fails, return 0
            return 0.0


@dataclass
class Telos:
    """Represents an intrinsic purpose or final cause.
    
    A telos defines what something is meant to become, its actualized form,
    and the criteria for measuring progress toward that actualization.
    """
    name: str
    description: str
    actualization_criteria: List[Criterion]
    attractor_state: Dict[str, Any]  # Target configuration
    current_actualization: float = 0.0  # Overall actualization (0.0 to 1.0)
    phase: ActualizationPhase = ActualizationPhase.POTENTIAL
    
    def evaluate_actualization(self, entity: Any) -> float:
        """Evaluate overall actualization for an entity.
        
        Args:
            entity: The entity to evaluate
            
        Returns:
            Actualization score (0.0 to 1.0)
        """
        if not self.actualization_criteria:
            return 0.0
        
        # Evaluate all criteria
        scores = []
        total_weight = 0.0
        
        for criterion in self.actualization_criteria:
            score = criterion.evaluate(entity)
            weighted_score = score * criterion.weight
            scores.append(weighted_score)
            total_weight += criterion.weight
        
        # Compute weighted average
        if total_weight > 0:
            actualization = sum(scores) / total_weight
        else:
            actualization = 0.0
        
        # Apply non-linear scaling (sigmoid) to emphasize middle progress
        # This makes progress more significant in the 0.3-0.7 range
        actualization = 1.0 / (1.0 + np.exp(-10.0 * (actualization - 0.5)))
        
        # Update current actualization
        self.current_actualization = actualization
        
        # Update phase based on actualization
        self._update_phase()
        
        return actualization
    
    def _update_phase(self) -> None:
        """Update actualization phase based on current progress."""
        if self.current_actualization < 0.2:
            self.phase = ActualizationPhase.POTENTIAL
        elif self.current_actualization < 0.4:
            self.phase = ActualizationPhase.EMERGENT
        elif self.current_actualization < 0.6:
            self.phase = ActualizationPhase.DEVELOPING
        elif self.current_actualization < 0.8:
            self.phase = ActualizationPhase.ACTUALIZING
        else:
            self.phase = ActualizationPhase.ACTUALIZED
    
    def get_phase_description(self) -> str:
        """Get description of current phase."""
        descriptions = {
            ActualizationPhase.POTENTIAL: "Exploring possibilities and discovering potential",
            ActualizationPhase.EMERGENT: "New properties and structures emerging",
            ActualizationPhase.DEVELOPING: "Integrating components and growing capabilities",
            ActualizationPhase.ACTUALIZING: "Refining and optimizing toward ideal form",
            ActualizationPhase.ACTUALIZED: "Fully realized and transcending to enable higher-order"
        }
        return descriptions.get(self.phase, "Unknown phase")


@dataclass
class ActualizationMetrics:
    """Comprehensive metrics for tracking actualization."""
    potentiality: float = 0.0  # Unrealized potential (0.0 to 1.0)
    emergence: float = 0.0  # New properties appearing (0.0 to 1.0)
    integration: float = 0.0  # Component coherence (0.0 to 1.0)
    actualization: float = 0.0  # Overall progress (0.0 to 1.0)
    telos_alignment: float = 0.0  # Alignment with purpose (0.0 to 1.0)
    
    @property
    def overall_health(self) -> float:
        """Compute overall developmental health."""
        return (self.potentiality * 0.1 +
                self.emergence * 0.2 +
                self.integration * 0.3 +
                self.actualization * 0.3 +
                self.telos_alignment * 0.1)


class ActualizationTracker:
    """Tracks actualization progress for entities."""
    
    def __init__(self):
        """Initialize tracker."""
        self.history: List[ActualizationMetrics] = []
        self.phase_transitions: List[Dict[str, Any]] = []
    
    def measure_potentiality(self, entity: Any) -> float:
        """Measure unrealized potential.
        
        Potentiality is high when entity has many unexplored capabilities
        and decreases as capabilities are actualized.
        """
        # Check for unexplored genetic variation
        if hasattr(entity, 'genome') and hasattr(entity.genome, 'genes'):
            total_genes = len(entity.genome.genes)
            active_genes = sum(1 for g in entity.genome.genes 
                             if getattr(g, 'active', True))
            if total_genes > 0:
                unused_potential = (total_genes - active_genes) / total_genes
                return unused_potential
        
        # Check for developmental maturity
        if hasattr(entity, 'ontogenetic_state'):
            # High potentiality in early stages
            maturity = entity.ontogenetic_state.maturity
            return 1.0 - maturity
        
        # Default: moderate potentiality
        return 0.5
    
    def measure_emergence(self, entity: Any) -> float:
        """Measure emerging properties.
        
        Emergence is high when new capabilities or properties are appearing.
        """
        # Check development history for recent changes
        if hasattr(entity, 'ontogenetic_state') and hasattr(entity.ontogenetic_state, 'development_history'):
            history = entity.ontogenetic_state.development_history
            if len(history) > 0:
                # Count recent events (last 10)
                recent_events = history[-10:]
                emergence_events = [e for e in recent_events 
                                  if e.get('type') in ['stage_transition', 'new_capability', 'property_emergence']]
                emergence_rate = len(emergence_events) / 10.0
                return min(emergence_rate, 1.0)
        
        # Check for fitness improvements
        if hasattr(entity, 'genome') and hasattr(entity.genome, 'fitness'):
            # If fitness is improving rapidly, emergence is happening
            if hasattr(entity, 'metadata') and 'previous_fitness' in entity.metadata:
                prev_fitness = entity.metadata['previous_fitness']
                current_fitness = entity.genome.fitness
                improvement = current_fitness - prev_fitness
                # Normalize improvement to 0-1
                emergence = np.tanh(improvement * 5.0)  # Scale and bound
                return max(0.0, emergence)
        
        # Default: low emergence
        return 0.2
    
    def measure_integration(self, entity: Any) -> float:
        """Measure component integration and coherence.
        
        Integration is high when components work together harmoniously.
        """
        # Check gene coherence
        if hasattr(entity, 'genes'):
            genes = entity.genes
            if len(genes) > 1:
                # Measure weight variance (lower is more integrated)
                weights = [g.weight for g in genes if hasattr(g, 'weight')]
                if weights:
                    variance = np.var(weights)
                    # Low variance = high integration
                    integration = 1.0 - min(variance, 1.0)
                    return integration
        
        # Check structural coherence
        if hasattr(entity, 'to_atomspace_structure'):
            try:
                atoms = entity.to_atomspace_structure()
                if atoms:
                    # More atoms with consistent types = better integration
                    types = [a.atom_type for a in atoms]
                    unique_types = len(set(types))
                    if len(types) > 0:
                        type_diversity = unique_types / len(types)
                        # Moderate diversity indicates good integration
                        # Too low = monotonous, too high = chaotic
                        integration = 1.0 - abs(type_diversity - 0.5) * 2.0
                        return max(0.0, integration)
            except:
                pass
        
        # Default: moderate integration
        return 0.5
    
    def measure_actualization(self, entity: Any, telos: Optional[Telos] = None) -> float:
        """Measure overall actualization progress.
        
        Args:
            entity: Entity to measure
            telos: Optional telos for guided measurement
            
        Returns:
            Actualization score (0.0 to 1.0)
        """
        if telos is not None:
            # Use telos criteria for precise measurement
            return telos.evaluate_actualization(entity)
        
        # Generic actualization based on fitness and maturity
        actualization = 0.0
        count = 0
        
        # Fitness contribution
        if hasattr(entity, 'fitness') and entity.fitness is not None:
            actualization += entity.fitness
            count += 1
        elif hasattr(entity, 'genome') and hasattr(entity.genome, 'fitness'):
            actualization += entity.genome.fitness
            count += 1
        
        # Maturity contribution
        if hasattr(entity, 'ontogenetic_state'):
            actualization += entity.ontogenetic_state.maturity
            count += 1
        
        # Development stage contribution
        if hasattr(entity, 'ontogenetic_state'):
            stage_values = {
                'embryonic': 0.2,
                'juvenile': 0.4,
                'mature': 0.8,
                'senescent': 0.6  # Decline after peak
            }
            stage = entity.ontogenetic_state.stage
            stage_name = stage.value if hasattr(stage, 'value') else str(stage).lower()
            actualization += stage_values.get(stage_name, 0.5)
            count += 1
        
        return actualization / count if count > 0 else 0.0
    
    def compute_metrics(self, entity: Any, telos: Optional[Telos] = None) -> ActualizationMetrics:
        """Compute comprehensive actualization metrics.
        
        Args:
            entity: Entity to measure
            telos: Optional telos for guided measurement
            
        Returns:
            Complete actualization metrics
        """
        metrics = ActualizationMetrics(
            potentiality=self.measure_potentiality(entity),
            emergence=self.measure_emergence(entity),
            integration=self.measure_integration(entity),
            actualization=self.measure_actualization(entity, telos)
        )
        
        # Compute telos alignment if telos provided
        if telos is not None:
            metrics.telos_alignment = self._compute_telos_alignment(entity, telos)
        else:
            metrics.telos_alignment = metrics.actualization  # Default to actualization
        
        # Record in history
        self.history.append(metrics)
        
        return metrics
    
    def _compute_telos_alignment(self, entity: Any, telos: Telos) -> float:
        """Compute how well entity aligns with its telos.
        
        Args:
            entity: Entity to evaluate
            telos: The telos to align with
            
        Returns:
            Alignment score (0.0 to 1.0)
        """
        # Get entity state
        if hasattr(entity, 'get_state_vector'):
            current_state = entity.get_state_vector()
        elif hasattr(entity, 'genes'):
            # Use gene weights as state
            current_state = np.array([g.weight for g in entity.genes if hasattr(g, 'weight')])
        else:
            # Can't compute alignment without state
            return 0.5
        
        # Get target state from telos
        target_state = telos.attractor_state
        
        # If target state is a dict, extract relevant values
        if isinstance(target_state, dict):
            # Try to match dimensions
            if 'weights' in target_state:
                target_values = np.array(target_state['weights'])
            elif 'state_vector' in target_state:
                target_values = np.array(target_state['state_vector'])
            else:
                # Use all numeric values
                target_values = np.array([v for v in target_state.values() 
                                        if isinstance(v, (int, float))])
        else:
            target_values = np.array(target_state)
        
        # Ensure same dimensions
        min_len = min(len(current_state), len(target_values))
        if min_len == 0:
            return 0.5
        
        current_state = current_state[:min_len]
        target_values = target_values[:min_len]
        
        # Compute cosine similarity
        dot_product = np.dot(current_state, target_values)
        norm_current = np.linalg.norm(current_state)
        norm_target = np.linalg.norm(target_values)
        
        if norm_current > 0 and norm_target > 0:
            similarity = dot_product / (norm_current * norm_target)
            # Convert from [-1, 1] to [0, 1]
            alignment = (similarity + 1.0) / 2.0
            return alignment
        
        return 0.5
    
    def detect_phase_transition(self) -> Optional[Dict[str, Any]]:
        """Detect if a phase transition occurred recently.
        
        Returns:
            Transition details if detected, None otherwise
        """
        if len(self.history) < 2:
            return None
        
        # Get recent metrics
        current = self.history[-1]
        previous = self.history[-2]
        
        # Check for significant changes
        actualization_jump = current.actualization - previous.actualization
        emergence_spike = current.emergence - previous.emergence
        integration_change = current.integration - previous.integration
        
        # Transition criteria
        if (actualization_jump > 0.15 or 
            emergence_spike > 0.3 or
            abs(integration_change) > 0.2):
            
            transition = {
                'type': 'phase_transition',
                'actualization_jump': actualization_jump,
                'emergence_spike': emergence_spike,
                'integration_change': integration_change,
                'from_metrics': previous,
                'to_metrics': current
            }
            
            self.phase_transitions.append(transition)
            return transition
        
        return None


@dataclass
class DevelopmentalAttractor:
    """Represents an attractor state in developmental space.
    
    An attractor is a stable configuration toward which development naturally tends.
    """
    telos: Telos
    basin_strength: float = 1.0  # Strength of attraction
    basin_radius: float = 0.5  # Radius of basin of attraction
    
    def compute_gradient(self, current_state: np.ndarray) -> np.ndarray:
        """Compute gradient toward attractor.
        
        Args:
            current_state: Current state vector
            
        Returns:
            Gradient vector pointing toward attractor
        """
        # Get target state
        target_state = self.telos.attractor_state
        
        # Convert to array if needed
        if isinstance(target_state, dict):
            if 'weights' in target_state:
                target_array = np.array(target_state['weights'])
            elif 'state_vector' in target_state:
                target_array = np.array(target_state['state_vector'])
            else:
                target_array = np.array([v for v in target_state.values() 
                                       if isinstance(v, (int, float))])
        else:
            target_array = np.array(target_state)
        
        # Ensure same dimensions
        min_len = min(len(current_state), len(target_array))
        current_state = current_state[:min_len]
        target_array = target_array[:min_len]
        
        # Compute distance
        distance = np.linalg.norm(target_array - current_state)
        
        # Compute direction
        if distance > 0:
            direction = (target_array - current_state) / distance
        else:
            # Already at attractor
            return np.zeros_like(current_state)
        
        # Compute strength (stronger when further, weaker when close)
        # Use tanh for smooth falloff
        strength = np.tanh(distance / self.basin_radius) * self.basin_strength
        
        # Gradient points toward attractor
        gradient = direction * strength
        
        return gradient
    
    def apply_pull(self, entity: Any, strength: float = 1.0) -> None:
        """Apply attractive force toward telos.
        
        Args:
            entity: Entity to pull toward attractor
            strength: Multiplier for pull strength
        """
        # Get current state
        if hasattr(entity, 'genes'):
            current_state = np.array([g.weight for g in entity.genes if hasattr(g, 'weight')])
            
            # Compute gradient
            gradient = self.compute_gradient(current_state)
            
            # Apply gradient to gene weights
            for i, gene in enumerate(entity.genes):
                if i < len(gradient) and hasattr(gene, 'weight'):
                    # Move weight toward attractor
                    gene.weight += gradient[i] * strength * 0.1  # Scale for stability
                    # Keep in valid range
                    gene.weight = max(0.0, min(1.0, gene.weight))
    
    def is_in_basin(self, entity: Any) -> bool:
        """Check if entity is within basin of attraction.
        
        Args:
            entity: Entity to check
            
        Returns:
            True if within basin, False otherwise
        """
        if hasattr(entity, 'genes'):
            current_state = np.array([g.weight for g in entity.genes if hasattr(g, 'weight')])
            gradient = self.compute_gradient(current_state)
            distance = np.linalg.norm(gradient)
            return distance < self.basin_radius
        
        return False
