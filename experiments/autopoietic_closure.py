#!/usr/bin/env python3
"""
Experiment: Autopoietic Closure
"""

import numpy as np


def main() -> None:
    from ontogentelechy.examples import create_adaptive_learning_telos
    from ontogentelechy.meta import AutopoieticSystem
    from ontogentelechy.entity import SimpleEntity, SimpleGene
    
    print("=" * 70)
    print("EXPERIMENT: Autopoietic Closure")
    print("=" * 70)
    
    initial_telos = create_adaptive_learning_telos()
    system = AutopoieticSystem(initial_telos, complexity_growth=1.3, max_levels=4)
    
    dim = 10
    state = np.random.uniform(0.1, 0.3, dim)
    entity = SimpleEntity(state=state, genes=[SimpleGene(weight=float(v)) for v in state])
    entity.fitness = 0.3
    
    print(f"\nInitial telos: {system.current_telos.name}")
    print(f"Max levels: {system.max_levels}")
    print("\nRunning 80 steps...\n")
    
    for step in range(80):
        state += np.random.normal(0.025, 0.01, dim)
        state = np.clip(state, 0.0, 1.0)
        entity.update_state(state)
        entity.fitness = float(np.mean(state))
        entity.metadata['fitness_history'] = entity.metadata.get('fitness_history', []) + [entity.fitness]
        entity.age = step + 1
        
        metrics, closure = system.step(entity)
        
        if closure:
            print(f"Step {step:2d}: *** AUTOPOIETIC CLOSURE *** → {system.current_telos.name}")
            print(f"         Criteria: {len(system.current_telos.actualization_criteria)}, "
                  f"Level: {system.level}")
        
        if step % 10 == 0:
            print(f"Step {step:2d}: act={metrics.actualization:.3f} "
                  f"phase={system.current_telos.phase.value} "
                  f"telos={system.current_telos.name}")
    
    print(f"\nTelos lineage: {' → '.join(system.telos_lineage)}")
    print(f"Final level: {system.level}")
    print(f"Final telos criteria: {len(system.current_telos.actualization_criteria)}")
    print("\n✓ Autopoietic experiment complete")


if __name__ == "__main__":
    main()
