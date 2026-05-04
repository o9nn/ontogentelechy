#!/usr/bin/env python3
"""
Experiment: Swarm Collective Actualization
"""

import numpy as np


def main() -> None:
    from ontogentelechy.examples import create_semantic_coherence_telos
    from ontogentelechy.meta import SwarmTeleology
    from ontogentelechy.entity import SimpleEntity, SimpleGene
    
    print("=" * 70)
    print("EXPERIMENT: Swarm Collective Actualization")
    print("=" * 70)
    
    telos = create_semantic_coherence_telos()
    swarm = SwarmTeleology(shared_telos=telos, stigmergy_strength=0.15)
    n_agents = 10
    dim = 8
    
    for i in range(n_agents):
        state = np.random.uniform(0.1, 0.4, dim)
        entity = SimpleEntity(
            state=state,
            genes=[SimpleGene(weight=float(v)) for v in state],
        )
        swarm.register_agent(f"agent_{i}", entity)
    
    print(f"\nSwarm: {n_agents} agents, dim={dim}")
    print(f"Stigmergy strength: {swarm.stigmergy_strength}")
    print("\nRunning 30 steps...\n")
    
    for step in range(30):
        metrics = swarm.step()
        
        attractor = swarm.stigmergic_attractor
        for agent_id, entity in swarm._agents.items():
            state = entity.get_state_vector()
            if attractor is not None and len(attractor) > 0:
                min_len = min(len(state), len(attractor))
                direction = attractor[:min_len] - state[:min_len]
                noise = np.random.normal(0, 0.03, len(state))
                new_state = np.clip(state + direction * 0.05 + noise, 0.0, 1.0)
            else:
                new_state = np.clip(state + np.random.normal(0.01, 0.02, len(state)), 0.0, 1.0)
            entity.update_state(new_state)
        
        if step % 5 == 0 or step == 29:
            print(f"Step {step:2d}: collective={metrics.collective_actualization:.3f} "
                  f"coherence={metrics.coherence:.3f} "
                  f"diversity={metrics.diversity:.3f} "
                  f"phase={metrics.emergent_phase.value}")
    
    final = swarm.collective_history[-1]
    print(f"\nFinal collective actualization: {final.collective_actualization:.3f}")
    print(f"Final phase: {final.emergent_phase.value}")
    print(f"Total steps tracked: {len(swarm.collective_history)}")
    print("\n✓ Swarm experiment complete")


if __name__ == "__main__":
    main()
