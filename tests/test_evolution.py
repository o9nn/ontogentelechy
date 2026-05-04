"""Tests for ontogentelechy.evolution (Phase 3)."""

import numpy as np
import pytest

from ontogentelechy.attractor import AttractorLandscape, Basin
from ontogentelechy.core import Criterion, Telos
from ontogentelechy.evolution import (
    Individual,
    IslandModel,
    OntogeneticStage,
    ParetoEvolution,
    TelosPopulation,
    _developmental_crossover,
    _gaussian_mutation,
    _tournament_select,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_telos(name: str = "test") -> Telos:
    return Telos(
        name=name,
        description="Test telos",
        actualization_criteria=[
            Criterion("c1", "crit1", 1.0, lambda e: float(np.mean(e.get_state_vector()))),
        ],
        attractor_state={"x": 0.8, "y": 0.6},
    )


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


class TestIndividual:
    def test_creation_from_state(self):
        ind = Individual(state=np.array([0.1, 0.5, 0.9]))
        assert len(ind.state) == 3
        assert len(ind.genes) == 3

    def test_random_creates_valid_individual(self):
        ind = Individual.random(dim=5)
        assert len(ind.state) == 5
        assert all(0.0 <= v <= 1.0 for v in ind.state)

    def test_develop_increments_age(self):
        ind = Individual.random(dim=3)
        ind.develop()
        assert ind.age == 1

    def test_develop_embryonic_stage(self):
        ind = Individual.random(dim=3)
        ind.develop()  # age = 1
        assert ind.ontogenetic_stage == OntogeneticStage.EMBRYONIC

    def test_develop_juvenile_stage(self):
        ind = Individual.random(dim=3)
        for _ in range(7):
            ind.develop()  # age = 7
        assert ind.ontogenetic_stage == OntogeneticStage.JUVENILE

    def test_develop_mature_stage(self):
        ind = Individual.random(dim=3)
        for _ in range(20):
            ind.develop()
        assert ind.ontogenetic_stage == OntogeneticStage.MATURE

    def test_develop_senescent_stage(self):
        ind = Individual.random(dim=3)
        for _ in range(31):
            ind.develop()
        assert ind.ontogenetic_stage == OntogeneticStage.SENESCENT

    def test_is_senescent(self):
        ind = Individual.random(dim=3)
        for _ in range(31):
            ind.develop()
        assert ind.is_senescent()

    def test_not_senescent_early(self):
        ind = Individual.random(dim=3)
        ind.develop()
        assert not ind.is_senescent()

    def test_get_state_vector(self):
        state = np.array([0.2, 0.4, 0.6])
        ind = Individual(state=state)
        np.testing.assert_array_equal(ind.get_state_vector(), state)

    def test_get_active_genes(self):
        ind = Individual.random(dim=4)
        active = ind.get_active_genes()
        assert len(active) == 4  # all genes active by default


# ---------------------------------------------------------------------------
# Selection tests
# ---------------------------------------------------------------------------


class TestTournamentSelect:
    def test_returns_individual(self):
        pop = [Individual.random(dim=3) for _ in range(10)]
        for i, ind in enumerate(pop):
            ind.fitness = float(i) / 10.0
        result = _tournament_select(pop, tournament_size=3)
        assert isinstance(result, Individual)

    def test_returns_from_population(self):
        pop = [Individual.random(dim=3) for _ in range(5)]
        result = _tournament_select(pop, tournament_size=2)
        assert any(result is ind for ind in pop)


# ---------------------------------------------------------------------------
# Crossover / mutation tests
# ---------------------------------------------------------------------------


class TestDevelopmentalCrossover:
    def test_returns_two_individuals(self):
        p1 = Individual.random(dim=5)
        p2 = Individual.random(dim=5)
        c1, c2 = _developmental_crossover(p1, p2)
        assert isinstance(c1, Individual)
        assert isinstance(c2, Individual)

    def test_children_states_in_unit_range(self):
        p1 = Individual.random(dim=5)
        p2 = Individual.random(dim=5)
        c1, c2 = _developmental_crossover(p1, p2)
        assert all(0.0 <= v <= 1.0 for v in c1.state)
        assert all(0.0 <= v <= 1.0 for v in c2.state)

    def test_children_generation_incremented(self):
        p1 = Individual.random(dim=5, generation=3)
        p2 = Individual.random(dim=5, generation=5)
        c1, _ = _developmental_crossover(p1, p2)
        assert c1.generation == 6

    def test_with_landscape(self):
        landscape = AttractorLandscape(basins=[Basin(state=np.array([0.5, 0.5, 0.5, 0.5, 0.5]))])
        p1 = Individual.random(dim=5)
        p2 = Individual.random(dim=5)
        c1, c2 = _developmental_crossover(p1, p2, landscape=landscape)
        assert all(0.0 <= v <= 1.0 for v in c1.state)
        assert all(0.0 <= v <= 1.0 for v in c2.state)


class TestGaussianMutation:
    def test_returns_individual(self):
        ind = Individual.random(dim=5)
        mutated = _gaussian_mutation(ind)
        assert isinstance(mutated, Individual)

    def test_state_stays_in_unit_range(self):
        ind = Individual.random(dim=10)
        for _ in range(20):
            ind = _gaussian_mutation(ind, mutation_rate=1.0, mutation_sigma=0.5)
        assert all(0.0 <= v <= 1.0 for v in ind.state)

    def test_zero_mutation_rate_unchanged(self):
        ind = Individual.random(dim=5)
        mutated = _gaussian_mutation(ind, mutation_rate=0.0)
        np.testing.assert_array_equal(mutated.state, ind.state)


# ---------------------------------------------------------------------------
# TelosPopulation tests
# ---------------------------------------------------------------------------


class TestTelosPopulation:
    def test_creation(self):
        telos = _make_telos()
        pop = TelosPopulation(telos=telos, population_size=10, dim=5)
        assert len(pop.population) == 10
        assert pop.generation == 0

    def test_evaluate_population_sets_fitness(self):
        telos = _make_telos()
        pop = TelosPopulation(telos=telos, population_size=5, dim=4)
        pop.evaluate_population()
        assert all(ind.fitness >= 0.0 for ind in pop.population)

    def test_step_returns_stats_dict(self):
        telos = _make_telos()
        pop = TelosPopulation(telos=telos, population_size=10, dim=4)
        stats = pop.step()
        expected_keys = {
            "generation",
            "mean_fitness",
            "max_fitness",
            "min_fitness",
            "std_fitness",
            "culled",
            "telos_phase",
        }
        assert expected_keys.issubset(stats.keys())

    def test_step_increments_generation(self):
        telos = _make_telos()
        pop = TelosPopulation(telos=telos, population_size=10, dim=4)
        pop.step()
        assert pop.generation == 1

    def test_run_returns_history(self):
        telos = _make_telos()
        pop = TelosPopulation(telos=telos, population_size=10, dim=4)
        history = pop.run(n_generations=3)
        assert isinstance(history, list)
        assert len(history) == 3

    def test_best_state_not_none_after_run(self):
        telos = _make_telos()
        pop = TelosPopulation(telos=telos, population_size=10, dim=4)
        pop.run(n_generations=2)
        assert pop.best_state() is not None

    def test_fitness_proportionate_selection(self):
        telos = _make_telos()
        pop = TelosPopulation(
            telos=telos, population_size=10, dim=4, selection_method="fitness_proportionate"
        )
        stats = pop.step()
        assert "mean_fitness" in stats

    def test_run_without_senescent_culling(self):
        telos = _make_telos()
        pop = TelosPopulation(telos=telos, population_size=10, dim=4, cull_senescent=False)
        pop.run(n_generations=5)
        assert pop.generation == 5


# ---------------------------------------------------------------------------
# IslandModel tests
# ---------------------------------------------------------------------------


class TestIslandModel:
    def test_creation(self):
        t1 = _make_telos("t1")
        t2 = _make_telos("t2")
        islands = [
            TelosPopulation(t1, population_size=8, dim=4),
            TelosPopulation(t2, population_size=8, dim=4),
        ]
        model = IslandModel(islands=islands, migration_interval=5)
        assert len(model.islands) == 2

    def test_step_returns_stats_list(self):
        t1 = _make_telos("t1")
        t2 = _make_telos("t2")
        islands = [
            TelosPopulation(t1, population_size=6, dim=4),
            TelosPopulation(t2, population_size=6, dim=4),
        ]
        model = IslandModel(islands=islands)
        stats = model.step()
        assert isinstance(stats, list)
        assert len(stats) == 2

    def test_run_completes(self):
        t1 = _make_telos("t1")
        t2 = _make_telos("t2")
        islands = [
            TelosPopulation(t1, population_size=6, dim=4),
            TelosPopulation(t2, population_size=6, dim=4),
        ]
        model = IslandModel(islands=islands, migration_interval=3)
        model.run(n_generations=5)
        assert model.generation == 5

    def test_migration_triggered_at_interval(self):
        t1 = _make_telos("t1")
        t2 = _make_telos("t2")
        islands = [
            TelosPopulation(t1, population_size=6, dim=4),
            TelosPopulation(t2, population_size=6, dim=4),
        ]
        model = IslandModel(islands=islands, migration_interval=2, migration_size=1)
        # Run enough steps to trigger migration
        model.run(n_generations=4)
        assert model.generation == 4


# ---------------------------------------------------------------------------
# ParetoEvolution tests
# ---------------------------------------------------------------------------


class TestParetoEvolution:
    def test_creation(self):
        teloi = [_make_telos("t1"), _make_telos("t2")]
        evo = ParetoEvolution(teloi=teloi, population_size=10, dim=4)
        assert len(evo.population) == 10

    def test_step_returns_dict_with_pareto_front_size(self):
        teloi = [_make_telos("t1"), _make_telos("t2")]
        evo = ParetoEvolution(teloi=teloi, population_size=10, dim=4)
        stats = evo.step()
        assert "pareto_front_size" in stats
        assert "mean_fitness" in stats
        assert "n_fronts" in stats

    def test_pareto_front_returns_list(self):
        teloi = [_make_telos("t1"), _make_telos("t2")]
        evo = ParetoEvolution(teloi=teloi, population_size=10, dim=4)
        evo.step()
        front = evo.pareto_front()
        assert isinstance(front, list)
        assert len(front) > 0

    def test_pareto_front_size_leq_population(self):
        teloi = [_make_telos("t1"), _make_telos("t2")]
        evo = ParetoEvolution(teloi=teloi, population_size=10, dim=4)
        front = evo.pareto_front()
        assert len(front) <= len(evo.population)

    def test_run_completes(self):
        teloi = [_make_telos("t1"), _make_telos("t2")]
        evo = ParetoEvolution(teloi=teloi, population_size=10, dim=4)
        evo.run(n_generations=3)
        assert evo.generation == 3

    def test_single_telos_pareto(self):
        teloi = [_make_telos("single")]
        evo = ParetoEvolution(teloi=teloi, population_size=8, dim=4)
        stats = evo.step()
        assert stats["pareto_front_size"] >= 1
