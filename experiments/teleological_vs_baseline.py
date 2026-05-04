#!/usr/bin/env python3
"""
Experiment: Teleological Fitness vs Baseline Optimization
"""

import numpy as np
import time
from typing import List, Dict


def run_teleological(dim: int, n_generations: int, seed: int) -> Dict:
    """Run teleological evolution."""
    from ontogentelechy.benchmarks import CoherentClusterBenchmark
    rng = np.random.RandomState(seed)
    target = rng.uniform(0.3, 0.7, dim)
    bench = CoherentClusterBenchmark(dim=dim, target_centroid=target, population_size=50, seed=seed)
    result = bench.run(n_generations=n_generations)
    return {
        'method': 'teleological',
        'final_fitness': result.final_fitness,
        'convergence_gen': result.convergence_generation,
        'time': result.time_seconds,
        'fitness_history': result.max_fitness_history,
    }


def run_baseline(dim: int, n_generations: int, seed: int) -> Dict:
    """Baseline: random search without teleological guidance."""
    from ontogentelechy.evolution import Individual, _gaussian_mutation
    rng = np.random.RandomState(seed)
    target = rng.uniform(0.3, 0.7, dim)
    
    population = [Individual.random(dim) for _ in range(50)]
    
    def score(ind: Individual) -> float:
        dist = float(np.linalg.norm(ind.state[:dim] - target[:dim]))
        return float(np.exp(-dist))
    
    max_history = []
    start = time.time()
    
    best = max(population, key=score)
    best_score = score(best)
    
    for gen in range(n_generations):
        new_pop = []
        for ind in population:
            mutant = _gaussian_mutation(ind, mutation_rate=0.3, mutation_sigma=0.1)
            new_pop.append(mutant if score(mutant) > score(ind) else ind)
        population = new_pop
        current_best = max(population, key=score)
        current_score = score(current_best)
        if current_score > best_score:
            best_score = current_score
            best = current_best
        max_history.append(current_score)
    
    conv_gen = next((i for i, f in enumerate(max_history) if f >= 0.8), None)
    return {
        'method': 'baseline',
        'final_fitness': max_history[-1] if max_history else 0.0,
        'convergence_gen': conv_gen,
        'time': time.time() - start,
        'fitness_history': max_history,
    }


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT: Teleological Fitness vs Baseline")
    print("=" * 70)
    
    dim = 20
    n_generations = 100
    n_trials = 3
    
    tele_results = []
    base_results = []
    
    for seed in range(n_trials):
        print(f"\nTrial {seed + 1}/{n_trials} (seed={seed})")
        
        t_res = run_teleological(dim, n_generations, seed)
        b_res = run_baseline(dim, n_generations, seed)
        
        tele_results.append(t_res)
        base_results.append(b_res)
        
        print(f"  Teleological: final={t_res['final_fitness']:.4f}, "
              f"conv={t_res['convergence_gen'] or 'N/A'}, "
              f"time={t_res['time']:.2f}s")
        print(f"  Baseline:     final={b_res['final_fitness']:.4f}, "
              f"conv={b_res['convergence_gen'] or 'N/A'}, "
              f"time={b_res['time']:.2f}s")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    tele_final = np.mean([r['final_fitness'] for r in tele_results])
    base_final = np.mean([r['final_fitness'] for r in base_results])
    
    tele_conv = [r['convergence_gen'] for r in tele_results if r['convergence_gen'] is not None]
    base_conv = [r['convergence_gen'] for r in base_results if r['convergence_gen'] is not None]
    
    print(f"Teleological mean final fitness: {tele_final:.4f}")
    print(f"Baseline mean final fitness:     {base_final:.4f}")
    print(f"Improvement: {(tele_final - base_final) / max(base_final, 1e-10) * 100:.1f}%")
    
    if tele_conv:
        print(f"Teleological mean convergence gen: {np.mean(tele_conv):.1f}")
    if base_conv:
        print(f"Baseline mean convergence gen:     {np.mean(base_conv):.1f}")
    
    print("\n✓ Experiment complete")


if __name__ == "__main__":
    main()
