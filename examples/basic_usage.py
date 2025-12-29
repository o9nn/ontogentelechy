#!/usr/bin/env python3
"""
Basic Usage Example for Ontogentelechy

This example demonstrates the core concepts of the ontogentelechy framework:
- Creating a telos (intrinsic purpose)
- Tracking actualization progress
- Using developmental attractors
- Phase transitions
"""

import numpy as np
from ontogentelechy import (
    Telos,
    Criterion,
    ActualizationTracker,
    DevelopmentalAttractor,
)
from ontogentelechy.examples import (
    create_semantic_coherence_telos,
    list_example_teloi,
)


def main():
    print("=" * 70)
    print("ONTOGENTELECHY - BASIC USAGE EXAMPLE")
    print("=" * 70)
    
    # 1. List available example teloi
    print("\n[1] Available Example Teloi:")
    for telos_name in list_example_teloi():
        print(f"  - {telos_name}")
    
    # 2. Create a telos
    print("\n[2] Creating Semantic Coherence Telos...")
    telos = create_semantic_coherence_telos()
    print(f"  Name: {telos.name}")
    print(f"  Description: {telos.description}")
    print(f"  Criteria: {len(telos.actualization_criteria)}")
    print(f"  Phase: {telos.phase.value}")
    
    # 3. Create a mock entity for demonstration
    print("\n[3] Creating Mock Entity...")
    
    class MockEntity:
        def __init__(self):
            self.fitness = 0.5
            self.age = 0
            self.metadata = {}
            self.genes = []
    
    class MockGene:
        def __init__(self, weight):
            self.weight = weight
            self.active = True
    
    entity = MockEntity()
    entity.genes = [MockGene(w) for w in [0.5, 0.6, 0.7, 0.8, 0.9]]
    print(f"  Entity created with {len(entity.genes)} genes")
    
    # 4. Track actualization
    print("\n[4] Tracking Actualization...")
    tracker = ActualizationTracker()
    
    metrics = tracker.compute_metrics(entity, telos)
    print(f"  Potentiality: {metrics.potentiality:.3f}")
    print(f"  Emergence: {metrics.emergence:.3f}")
    print(f"  Integration: {metrics.integration:.3f}")
    print(f"  Actualization: {metrics.actualization:.3f}")
    print(f"  Telos Alignment: {metrics.telos_alignment:.3f}")
    print(f"  Overall Health: {metrics.overall_health:.3f}")
    
    # 5. Use developmental attractor
    print("\n[5] Using Developmental Attractor...")
    attractor = DevelopmentalAttractor(telos)
    
    current_state = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    gradient = attractor.compute_gradient(current_state)
    
    print(f"  Current state: {current_state}")
    print(f"  Gradient: {gradient}")
    print(f"  Gradient norm: {np.linalg.norm(gradient):.3f}")
    
    in_basin = attractor.is_in_basin(entity)
    print(f"  In basin of attraction: {in_basin}")
    
    # 6. Simulate development with phase transitions
    print("\n[6] Simulating Development...")
    
    for step in range(10):
        # Gradually improve fitness
        entity.metadata['previous_fitness'] = entity.fitness
        entity.fitness += 0.05
        
        # Track metrics
        metrics = tracker.compute_metrics(entity, telos)
        
        # Check for phase transition
        transition = tracker.detect_phase_transition()
        
        print(f"  Step {step + 1}: "
              f"Fitness={entity.fitness:.2f}, "
              f"Actualization={metrics.actualization:.3f}, "
              f"Phase={telos.phase.value}")
        
        if transition:
            print(f"    → Phase transition detected!")
            print(f"      Actualization jump: {transition['actualization_jump']:.3f}")
    
    # 7. Custom telos example
    print("\n[7] Creating Custom Telos...")
    
    def evaluate_creativity(entity):
        # Higher fitness = more creative
        return entity.fitness
    
    def evaluate_stability(entity):
        # Check gene weight variance
        if hasattr(entity, 'genes') and entity.genes:
            weights = [g.weight for g in entity.genes]
            variance = np.var(weights)
            return 1.0 - min(variance, 1.0)
        return 0.5
    
    custom_telos = Telos(
        name="creative_stability",
        description="Balance creativity with stability",
        actualization_criteria=[
            Criterion("creativity", "Creative output", 0.6, evaluate_creativity, 1.0),
            Criterion("stability", "System stability", 0.4, evaluate_stability, 1.0)
        ],
        attractor_state={'creativity': 0.9, 'stability': 0.7}
    )
    
    print(f"  Custom telos created: {custom_telos.name}")
    
    # Evaluate with custom telos
    custom_metrics = tracker.compute_metrics(entity, custom_telos)
    print(f"  Actualization: {custom_metrics.actualization:.3f}")
    print(f"  Phase: {custom_telos.phase.value}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Created and used teloi (purposes)")
    print("✓ Tracked actualization metrics")
    print("✓ Computed developmental attractor gradients")
    print("✓ Detected phase transitions")
    print("✓ Created custom telos")
    print("\n🎯 Ontogentelechy framework operational!")
    print("=" * 70)


if __name__ == "__main__":
    main()
