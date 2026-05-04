"""
Teleological Evolution Engine

Self-contained evolutionary optimizer driven by teleological fitness.
No external dependencies beyond numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .attractor import AttractorLandscape
from .core import ActualizationPhase, ActualizationTracker, Telos
from .entity import SimpleEntity, SimpleGene


class OntogeneticStage(Enum):
    EMBRYONIC = "embryonic"
    JUVENILE = "juvenile"
    MATURE = "mature"
    SENESCENT = "senescent"


@dataclass
class Individual:
    """An individual in a teleological population."""

    state: np.ndarray
    genes: List[SimpleGene] = field(default_factory=list)
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    ontogenetic_stage: OntogeneticStage = OntogeneticStage.EMBRYONIC
    metadata: Dict[str, Any] = field(default_factory=dict)
    _tracker: ActualizationTracker = field(default_factory=ActualizationTracker, repr=False)

    def __post_init__(self) -> None:
        if not self.genes:
            self.genes = [SimpleGene(weight=float(v)) for v in self.state]

    def get_active_genes(self) -> List[SimpleGene]:
        return [g for g in self.genes if g.active]

    def get_state_vector(self) -> np.ndarray:
        return self.state.copy()

    def develop(self) -> None:
        """Age this individual one step and update ontogenetic stage."""
        self.age += 1
        if self.age < 5:
            self.ontogenetic_stage = OntogeneticStage.EMBRYONIC
        elif self.age < 15:
            self.ontogenetic_stage = OntogeneticStage.JUVENILE
        elif self.age < 30:
            self.ontogenetic_stage = OntogeneticStage.MATURE
        else:
            self.ontogenetic_stage = OntogeneticStage.SENESCENT

    def is_senescent(self) -> bool:
        return self.ontogenetic_stage == OntogeneticStage.SENESCENT

    @classmethod
    def random(cls, dim: int, generation: int = 0) -> "Individual":
        state = np.random.uniform(0.0, 1.0, dim)
        return cls(state=state, generation=generation)


def _tournament_select(
    population: List[Individual],
    tournament_size: int = 3,
) -> Individual:
    """Tournament selection: pick best of k random individuals."""
    contestants = np.random.choice(
        len(population), size=min(tournament_size, len(population)), replace=False
    )
    return max((population[i] for i in contestants), key=lambda ind: ind.fitness)


def _fitness_proportionate_select(population: List[Individual]) -> Individual:
    """Fitness-proportionate (roulette wheel) selection."""
    fitnesses = np.array([max(ind.fitness, 1e-10) for ind in population])
    probs = fitnesses / fitnesses.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


def _developmental_crossover(
    parent1: Individual,
    parent2: Individual,
    landscape: Optional[AttractorLandscape] = None,
) -> Tuple[Individual, Individual]:
    """Crossover guided by attractor gradients.

    Offspring states are biased toward both parents' attractor gradients.
    """
    dim = len(parent1.state)

    if landscape is not None:
        grad1 = landscape.net_gradient(parent1.state)
        grad2 = landscape.net_gradient(parent2.state)
        avg_gradient = (grad1 + grad2) / 2.0
    else:
        avg_gradient = np.zeros(dim)

    alpha = np.random.uniform(0.3, 0.7, dim)

    child1_state = alpha * parent1.state + (1 - alpha) * parent2.state
    child2_state = (1 - alpha) * parent1.state + alpha * parent2.state

    step = 0.05
    child1_state = np.clip(child1_state + avg_gradient * step, 0.0, 1.0)
    child2_state = np.clip(child2_state + avg_gradient * step, 0.0, 1.0)

    next_gen = max(parent1.generation, parent2.generation) + 1
    child1 = Individual(state=child1_state, generation=next_gen)
    child2 = Individual(state=child2_state, generation=next_gen)

    return child1, child2


def _gaussian_mutation(
    individual: Individual,
    mutation_rate: float = 0.1,
    mutation_sigma: float = 0.05,
) -> Individual:
    """Gaussian mutation on state vector."""
    state = individual.state.copy()
    mask = np.random.random(len(state)) < mutation_rate
    noise = np.random.normal(0, mutation_sigma, len(state))
    state[mask] += noise[mask]
    state = np.clip(state, 0.0, 1.0)
    return Individual(
        state=state,
        generation=individual.generation,
        genes=[SimpleGene(weight=float(v)) for v in state],
    )


class TelosPopulation:
    """Population of individuals evolving under a Telos.

    Implements:
    - Tournament and fitness-proportionate selection
    - Gradient-guided developmental crossover
    - Gaussian mutation
    - Ontogenetic stage progression and senescent culling
    - Trajectory quality selection (not just endpoint)
    """

    def __init__(
        self,
        telos: Telos,
        population_size: int = 50,
        dim: int = 10,
        landscape: Optional[AttractorLandscape] = None,
        selection_method: str = "tournament",
        tournament_size: int = 3,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.05,
        elitism: int = 2,
        cull_senescent: bool = True,
        trajectory_weight: float = 0.3,
    ):
        self.telos = telos
        self.population_size = population_size
        self.dim = dim
        self.landscape = landscape or AttractorLandscape.from_telos(telos)
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.elitism = elitism
        self.cull_senescent = cull_senescent
        self.trajectory_weight = trajectory_weight

        self.generation: int = 0
        self.population: List[Individual] = [Individual.random(dim) for _ in range(population_size)]
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict[str, Any]] = []
        self._tracker = ActualizationTracker()

    def _evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate teleological fitness for an individual."""
        entity = SimpleEntity(
            state=individual.state,
            genes=individual.genes,
            fitness=individual.fitness,
            metadata=individual.metadata,
        )
        metrics = self._tracker.compute_metrics(entity, self.telos)

        trajectory_bonus = 0.0
        if len(self._tracker.history) > 1 and self.trajectory_weight > 0:
            recent = self._tracker.history[-min(5, len(self._tracker.history)) :]
            improvements = sum(
                1
                for i in range(1, len(recent))
                if recent[i].actualization > recent[i - 1].actualization
            )
            trajectory_bonus = (improvements / max(len(recent) - 1, 1)) * self.trajectory_weight

        endpoint = metrics.overall_health * (1.0 - self.trajectory_weight)
        return float(np.clip(endpoint + trajectory_bonus, 0.0, 1.0))

    def evaluate_population(self) -> None:
        """Evaluate all individuals in the population."""
        for individual in self.population:
            individual.fitness = self._evaluate_fitness(individual)

        best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or best.fitness > self.best_individual.fitness:
            self.best_individual = Individual(
                state=best.state.copy(),
                fitness=best.fitness,
                generation=best.generation,
            )

    def _select(self) -> Individual:
        """Select one individual using configured method."""
        if self.selection_method == "tournament":
            return _tournament_select(self.population, self.tournament_size)
        else:
            return _fitness_proportionate_select(self.population)

    def step(self) -> Dict[str, Any]:
        """Run one generation of evolution. Returns generation statistics."""
        for ind in self.population:
            ind.develop()

        self.evaluate_population()

        if self.cull_senescent:
            viable = [ind for ind in self.population if not ind.is_senescent()]
            culled = len(self.population) - len(viable)
            self.population = viable
        else:
            culled = 0

        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        elites = self.population[: self.elitism]

        next_gen: List[Individual] = list(elites)

        while len(next_gen) < self.population_size:
            if len(self.population) < 2:
                next_gen.append(Individual.random(self.dim, self.generation + 1))
                continue

            if np.random.random() < self.crossover_rate and len(self.population) >= 2:
                p1 = self._select()
                p2 = self._select()
                child1, child2 = _developmental_crossover(p1, p2, self.landscape)
                next_gen.extend([child1, child2])
            else:
                parent = self._select()
                child = _gaussian_mutation(parent, self.mutation_rate, self.mutation_sigma)
                next_gen.append(child)

        self.population = next_gen[: self.population_size]

        self.landscape.step()

        fitnesses = [ind.fitness for ind in self.population]
        stats: Dict[str, Any] = {
            "generation": self.generation,
            "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
            "max_fitness": float(np.max(fitnesses)) if fitnesses else 0.0,
            "min_fitness": float(np.min(fitnesses)) if fitnesses else 0.0,
            "std_fitness": float(np.std(fitnesses)) if fitnesses else 0.0,
            "culled": culled,
            "telos_phase": self.telos.phase.value,
        }
        self.history.append(stats)
        self.generation += 1

        return stats

    def run(self, n_generations: int, verbose: bool = False) -> List[Dict[str, Any]]:
        """Run for n_generations. Returns full history."""
        for gen in range(n_generations):
            stats = self.step()
            if verbose:
                print(
                    f"Gen {gen:4d}: mean={stats['mean_fitness']:.4f} "
                    f"max={stats['max_fitness']:.4f} "
                    f"phase={stats['telos_phase']}"
                )
        return self.history

    def best_state(self) -> Optional[np.ndarray]:
        return self.best_individual.state if self.best_individual else None


class IslandModel:
    """Island model with multiple sub-populations and migration.

    Migration is gated by telos-alignment similarity:
    migrants are only accepted if they are similar in alignment
    to the receiving island's telos.
    """

    def __init__(
        self,
        islands: List[TelosPopulation],
        migration_interval: int = 10,
        migration_size: int = 2,
        alignment_threshold: float = 0.3,
    ):
        self.islands = islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.alignment_threshold = alignment_threshold
        self.generation: int = 0

    def step(self) -> List[Dict[str, Any]]:
        """Run one generation on all islands, with migration if due."""
        stats = [island.step() for island in self.islands]

        if self.generation % self.migration_interval == 0 and self.generation > 0:
            self._migrate()

        self.generation += 1
        return stats

    def _migrate(self) -> None:
        """Migrate best individuals between islands."""
        for i, source_island in enumerate(self.islands):
            if not source_island.population:
                continue

            sorted_pop = sorted(source_island.population, key=lambda ind: ind.fitness, reverse=True)
            migrants = sorted_pop[: self.migration_size]

            target_island = self.islands[(i + 1) % len(self.islands)]

            for migrant in migrants:
                target_entity = SimpleEntity(
                    state=migrant.state,
                    genes=migrant.genes,
                    fitness=migrant.fitness,
                    metadata=migrant.metadata,
                )
                tracker = ActualizationTracker()
                metrics = tracker.compute_metrics(target_entity, target_island.telos)

                if metrics.telos_alignment >= self.alignment_threshold:
                    if target_island.population:
                        worst_idx = min(
                            range(len(target_island.population)),
                            key=lambda j: target_island.population[j].fitness,
                        )
                        target_island.population[worst_idx] = Individual(
                            state=migrant.state.copy(),
                            generation=migrant.generation,
                        )

    def run(self, n_generations: int, verbose: bool = False) -> None:
        for gen in range(n_generations):
            stats = self.step()
            if verbose:
                means = [f"Island {i}: {s['mean_fitness']:.4f}" for i, s in enumerate(stats)]
                print(f"Gen {gen:4d}: " + " | ".join(means))


class ParetoEvolution:
    """Multi-telos co-evolution with Pareto-front selection.

    Each individual is evaluated against multiple teloi.
    Selection uses non-dominated sorting (NSGA-II style).
    """

    def __init__(
        self,
        teloi: List[Telos],
        population_size: int = 50,
        dim: int = 10,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.05,
        crossover_rate: float = 0.7,
    ):
        self.teloi = teloi
        self.population_size = population_size
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.generation: int = 0
        self.population: List[Individual] = [Individual.random(dim) for _ in range(population_size)]
        self._tracker = ActualizationTracker()

    def _evaluate_multi_objective(self, individual: Individual) -> List[float]:
        """Return list of objective scores (one per telos)."""
        entity = SimpleEntity(
            state=individual.state,
            genes=individual.genes,
            fitness=individual.fitness,
            metadata=individual.metadata,
        )
        scores = []
        for telos in self.teloi:
            metrics = self._tracker.compute_metrics(entity, telos)
            scores.append(metrics.overall_health)
        return scores

    def _dominates(self, scores_a: List[float], scores_b: List[float]) -> bool:
        """Return True if a dominates b (a is no worse on all, better on at least one)."""
        at_least_as_good = all(a >= b for a, b in zip(scores_a, scores_b))
        strictly_better = any(a > b for a, b in zip(scores_a, scores_b))
        return at_least_as_good and strictly_better

    def _non_dominated_sort(
        self,
        population: List[Individual],
        all_scores: List[List[float]],
    ) -> List[List[int]]:
        """NSGA-II non-dominated sorting. Returns list of Pareto fronts (lists of indices)."""
        n = len(population)
        domination_count = [0] * n
        dominated_by: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(all_scores[i], all_scores[j]):
                    dominated_by[i].append(j)
                elif self._dominates(all_scores[j], all_scores[i]):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front: List[int] = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front += 1

        return [f for f in fronts if f]

    def pareto_front(self) -> List[Individual]:
        """Return the current Pareto front."""
        all_scores = [self._evaluate_multi_objective(ind) for ind in self.population]
        fronts = self._non_dominated_sort(self.population, all_scores)
        if not fronts:
            return []
        return [self.population[i] for i in fronts[0]]

    def step(self) -> Dict[str, Any]:
        """One generation of Pareto evolution."""
        all_scores = [self._evaluate_multi_objective(ind) for ind in self.population]

        for ind, scores in zip(self.population, all_scores):
            ind.fitness = float(np.mean(scores))
            ind.metadata["objective_scores"] = scores

        fronts = self._non_dominated_sort(self.population, all_scores)

        next_gen: List[Individual] = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.population_size:
                next_gen.extend(self.population[i] for i in front)
            else:
                needed = self.population_size - len(next_gen)
                front_inds = [self.population[i] for i in front[:needed]]
                next_gen.extend(front_inds)
                break

        while len(next_gen) < self.population_size:
            if len(self.population) >= 2 and np.random.random() < self.crossover_rate:
                p1 = self.population[np.random.randint(len(self.population))]
                p2 = self.population[np.random.randint(len(self.population))]
                child1, child2 = _developmental_crossover(p1, p2)
                next_gen.extend([child1, child2])
            else:
                parent = self.population[np.random.randint(len(self.population))]
                child = _gaussian_mutation(parent, self.mutation_rate, self.mutation_sigma)
                next_gen.append(child)

        self.population = next_gen[: self.population_size]
        self.generation += 1

        pareto = self.pareto_front()
        return {
            "generation": self.generation,
            "pareto_front_size": len(pareto),
            "mean_fitness": float(np.mean([ind.fitness for ind in self.population])),
            "n_fronts": len(fronts),
        }

    def run(self, n_generations: int, verbose: bool = False) -> None:
        for gen in range(n_generations):
            stats = self.step()
            if verbose:
                print(
                    f"Gen {gen:4d}: pareto={stats['pareto_front_size']:3d} "
                    f"mean={stats['mean_fitness']:.4f}"
                )


# Alias for clean export alongside protocols.Individual
EvoIndividual = Individual
