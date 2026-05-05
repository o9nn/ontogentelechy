"""Tests for the benchmark suite."""

import pytest
import numpy as np

from ontogentelechy.benchmarks import (
    BenchmarkResult,
    CoherentClusterBenchmark,
    PhaseSeparationBenchmark,
    EmergenceTrackingBenchmark,
    run_benchmark,
    list_benchmarks,
    BENCHMARKS,
)


class TestBenchmarkResult:
    def test_convergence_generation_found(self):
        result = BenchmarkResult(
            benchmark_name="test",
            n_generations=5,
            final_fitness=0.9,
            mean_fitness_history=[0.1, 0.5, 0.8, 0.9, 0.95],
            max_fitness_history=[0.2, 0.6, 0.85, 0.92, 0.97],
            time_seconds=0.1,
            telos_phase_history=["potential"] * 5,
        )
        assert result.convergence_generation == 2

    def test_convergence_generation_not_found(self):
        result = BenchmarkResult(
            benchmark_name="test",
            n_generations=3,
            final_fitness=0.5,
            mean_fitness_history=[0.1, 0.3, 0.5],
            max_fitness_history=[0.2, 0.4, 0.6],
            time_seconds=0.1,
            telos_phase_history=["potential"] * 3,
        )
        assert result.convergence_generation is None

    def test_summary_contains_name(self):
        result = BenchmarkResult(
            benchmark_name="my_bench",
            n_generations=10,
            final_fitness=0.7,
            mean_fitness_history=[0.7],
            max_fitness_history=[0.7],
            time_seconds=1.0,
            telos_phase_history=["potential"],
        )
        summary = result.summary()
        assert "my_bench" in summary
        assert "10" in summary


class TestCoherentClusterBenchmark:
    def test_run_returns_result(self):
        bench = CoherentClusterBenchmark(dim=5, population_size=10, seed=42)
        result = bench.run(n_generations=5)
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "coherent_cluster"
        assert result.n_generations == 5
        assert len(result.max_fitness_history) == 5
        assert len(result.mean_fitness_history) == 5
        assert len(result.telos_phase_history) == 5
        assert result.time_seconds >= 0.0

    def test_fitness_values_in_range(self):
        bench = CoherentClusterBenchmark(dim=5, population_size=10, seed=0)
        result = bench.run(n_generations=5)
        for f in result.max_fitness_history:
            assert 0.0 <= f <= 1.0

    def test_custom_centroid(self):
        centroid = np.ones(5) * 0.5
        bench = CoherentClusterBenchmark(dim=5, target_centroid=centroid, population_size=10, seed=1)
        result = bench.run(n_generations=3)
        assert result.final_fitness >= 0.0

    def test_metadata_contains_dim_and_seed(self):
        bench = CoherentClusterBenchmark(dim=5, population_size=10, seed=7)
        result = bench.run(n_generations=2)
        assert result.metadata['dim'] == 5
        assert result.metadata['seed'] == 7


class TestPhaseSeparationBenchmark:
    def test_run_returns_result(self):
        bench = PhaseSeparationBenchmark(dim=5, population_size=10, seed=42)
        result = bench.run(n_generations=5)
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "phase_separation"
        assert len(result.max_fitness_history) == 5

    def test_fitness_non_negative(self):
        bench = PhaseSeparationBenchmark(dim=5, population_size=10, seed=0)
        result = bench.run(n_generations=5)
        for f in result.max_fitness_history:
            assert f >= 0.0


class TestEmergenceTrackingBenchmark:
    def test_run_returns_result(self):
        bench = EmergenceTrackingBenchmark(dim=4, n_steps=20)
        result = bench.run()
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "emergence_tracking"
        assert result.n_generations == 20

    def test_metadata_has_phase_transitions(self):
        bench = EmergenceTrackingBenchmark(dim=4, n_steps=20)
        result = bench.run()
        assert 'phase_transitions_detected' in result.metadata
        assert isinstance(result.metadata['phase_transitions_detected'], int)

    def test_phase_history_length(self):
        bench = EmergenceTrackingBenchmark(dim=4, n_steps=15)
        result = bench.run()
        assert len(result.telos_phase_history) == 15


class TestRunBenchmark:
    def test_run_coherent_cluster(self):
        result = run_benchmark('coherent_cluster', dim=5, population_size=10, seed=0, n_generations=5)
        assert result.benchmark_name == "coherent_cluster"

    def test_run_phase_separation(self):
        result = run_benchmark('phase_separation', dim=5, population_size=10, seed=0, n_generations=5)
        assert result.benchmark_name == "phase_separation"

    def test_run_emergence_tracking(self):
        result = run_benchmark('emergence_tracking', dim=4)
        assert result.benchmark_name == "emergence_tracking"

    def test_unknown_benchmark_raises(self):
        with pytest.raises(KeyError):
            run_benchmark('nonexistent_benchmark')

    def test_list_benchmarks(self):
        names = list_benchmarks()
        assert 'coherent_cluster' in names
        assert 'phase_separation' in names
        assert 'emergence_tracking' in names
        assert len(names) == 3
