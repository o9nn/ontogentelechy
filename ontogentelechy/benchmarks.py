"""
Benchmark Suite for Ontogentelechy

Standard synthetic tasks for evaluating teleological frameworks.
Each benchmark provides:
- A ground-truth Telos
- An entity generator
- A scoring function
- A reproducible experiment runner
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import time

from .core import Telos, Criterion, ActualizationTracker
from .entity import SimpleEntity, SimpleGene
from .evolution import TelosPopulation, Individual
from .attractor import AttractorLandscape, Basin


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""
    benchmark_name: str
    n_generations: int
    final_fitness: float
    mean_fitness_history: List[float]
    max_fitness_history: List[float]
    time_seconds: float
    telos_phase_history: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def convergence_generation(self) -> Optional[int]:
        """Generation where max_fitness first exceeded 0.8."""
        for i, f in enumerate(self.max_fitness_history):
            if f >= 0.8:
                return i
        return None
    
    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"Generations: {self.n_generations}",
            f"Final fitness: {self.final_fitness:.4f}",
            f"Time: {self.time_seconds:.2f}s",
            f"Convergence at gen: {self.convergence_generation or 'N/A'}",
            f"Final phase: {self.telos_phase_history[-1] if self.telos_phase_history else 'N/A'}",
        ]
        return "\n".join(lines)


class CoherentClusterBenchmark:
    """Find a coherent cluster in high-dimensional space."""
    
    NAME = "coherent_cluster"
    
    def __init__(
        self,
        dim: int = 20,
        target_centroid: Optional[np.ndarray] = None,
        population_size: int = 50,
        seed: int = 42,
    ):
        self.dim = dim
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.target_centroid = target_centroid if target_centroid is not None else rng.uniform(0.3, 0.7, dim)
        self.population_size = population_size
    
    def _create_telos(self) -> Telos:
        centroid = self.target_centroid
        
        def evaluate_proximity(entity: Any) -> float:
            sv = _get_state(entity)
            if sv is None:
                return 0.0
            min_len = min(len(sv), len(centroid))
            dist = float(np.linalg.norm(sv[:min_len] - centroid[:min_len]))
            return float(np.exp(-dist))
        
        def evaluate_coherence(entity: Any) -> float:
            sv = _get_state(entity)
            if sv is None:
                return 0.0
            variance = float(np.var(sv))
            return 1.0 - min(variance, 1.0)
        
        def evaluate_alignment(entity: Any) -> float:
            sv = _get_state(entity)
            if sv is None or sv.sum() == 0:
                return 0.0
            min_len = min(len(sv), len(centroid))
            dot = float(np.dot(sv[:min_len], centroid[:min_len]))
            norm = float(np.linalg.norm(sv[:min_len]) * np.linalg.norm(centroid[:min_len]))
            if norm < 1e-10:
                return 0.0
            return float((dot / norm + 1.0) / 2.0)
        
        return Telos(
            name="coherent_cluster",
            description="Converge to coherent cluster around target centroid",
            actualization_criteria=[
                Criterion("proximity", "Proximity to centroid", 0.5, evaluate_proximity, 1.0),
                Criterion("coherence", "State coherence", 0.3, evaluate_coherence, 1.0),
                Criterion("alignment", "Directional alignment", 0.2, evaluate_alignment, 1.0),
            ],
            attractor_state={'weights': self.target_centroid.tolist()},
        )
    
    def run(self, n_generations: int = 100, verbose: bool = False) -> BenchmarkResult:
        telos = self._create_telos()
        landscape = AttractorLandscape.from_telos(telos)
        pop = TelosPopulation(
            telos=telos,
            population_size=self.population_size,
            dim=self.dim,
            landscape=landscape,
        )
        
        start = time.time()
        mean_history = []
        max_history = []
        phase_history = []
        
        for _ in range(n_generations):
            stats = pop.step()
            mean_history.append(stats['mean_fitness'])
            max_history.append(stats['max_fitness'])
            phase_history.append(stats['telos_phase'])
            if verbose:
                print(f"Gen {stats['generation']:4d}: max={stats['max_fitness']:.4f}")
        
        elapsed = time.time() - start
        
        return BenchmarkResult(
            benchmark_name=self.NAME,
            n_generations=n_generations,
            final_fitness=max_history[-1] if max_history else 0.0,
            mean_fitness_history=mean_history,
            max_fitness_history=max_history,
            time_seconds=elapsed,
            telos_phase_history=phase_history,
            metadata={'dim': self.dim, 'seed': self.seed},
        )


class PhaseSeparationBenchmark:
    """Multi-basin phase separation task."""
    
    NAME = "phase_separation"
    
    def __init__(
        self,
        dim: int = 10,
        population_size: int = 40,
        seed: int = 42,
    ):
        self.dim = dim
        self.population_size = population_size
        self.seed = seed
        rng = np.random.RandomState(seed)
        self.basin1_center = rng.uniform(0.1, 0.4, dim)
        self.basin2_center = rng.uniform(0.6, 0.9, dim)
    
    def _create_telos(self) -> Telos:
        b1 = self.basin1_center
        b2 = self.basin2_center
        
        def evaluate_bimodal(entity: Any) -> float:
            sv = _get_state(entity)
            if sv is None:
                return 0.0
            min_len = min(len(sv), len(b1), len(b2))
            dist1 = float(np.linalg.norm(sv[:min_len] - b1[:min_len]))
            dist2 = float(np.linalg.norm(sv[:min_len] - b2[:min_len]))
            return float(np.exp(-min(dist1, dist2)))
        
        def evaluate_stability(entity: Any) -> float:
            sv = _get_state(entity)
            if sv is None:
                return 0.0
            return max(0.0, 1.0 - float(np.var(sv)))
        
        return Telos(
            name="phase_separation",
            description="Separate into two distinct phases/niches",
            actualization_criteria=[
                Criterion("bimodal_proximity", "Proximity to nearest basin", 0.7, evaluate_bimodal, 1.0),
                Criterion("stability", "State stability", 0.3, evaluate_stability, 1.0),
            ],
            attractor_state={'weights': b1.tolist()},
        )
    
    def run(self, n_generations: int = 100, verbose: bool = False) -> BenchmarkResult:
        from .attractor import Basin
        telos = self._create_telos()
        landscape = AttractorLandscape(basins=[
            Basin(state=self.basin1_center, label="basin1"),
            Basin(state=self.basin2_center, label="basin2"),
        ])
        pop = TelosPopulation(
            telos=telos,
            population_size=self.population_size,
            dim=self.dim,
            landscape=landscape,
        )
        
        start = time.time()
        mean_history, max_history, phase_history = [], [], []
        
        for _ in range(n_generations):
            stats = pop.step()
            mean_history.append(stats['mean_fitness'])
            max_history.append(stats['max_fitness'])
            phase_history.append(stats['telos_phase'])
        
        elapsed = time.time() - start
        return BenchmarkResult(
            benchmark_name=self.NAME,
            n_generations=n_generations,
            final_fitness=max_history[-1] if max_history else 0.0,
            mean_fitness_history=mean_history,
            max_fitness_history=max_history,
            time_seconds=elapsed,
            telos_phase_history=phase_history,
            metadata={'dim': self.dim, 'seed': self.seed},
        )


class EmergenceTrackingBenchmark:
    """Track emergence events during development."""
    
    NAME = "emergence_tracking"
    
    def __init__(self, dim: int = 8, n_steps: int = 50):
        self.dim = dim
        self.n_steps = n_steps
    
    def run(self, verbose: bool = False) -> BenchmarkResult:
        from .core import Telos, Criterion, ActualizationTracker
        from .emergence import EmergenceDetector
        
        def eval_fitness(entity: Any) -> float:
            sv = _get_state(entity)
            return float(np.mean(sv)) if sv is not None else 0.0
        
        telos = Telos(
            name="emergence_tracking",
            description="Track emergence during rapid development",
            actualization_criteria=[
                Criterion("fitness", "Mean activation", 1.0, eval_fitness, 1.0),
            ],
            attractor_state={'weights': [0.8] * self.dim},
        )
        
        tracker = ActualizationTracker()
        detector = EmergenceDetector(window_size=10)
        
        state = np.random.uniform(0.1, 0.3, self.dim)
        entity = SimpleEntity(state=state)
        
        phase_transitions_detected = 0
        phase_history = []
        max_history = []
        mean_history = []
        
        start = time.time()
        for step in range(self.n_steps):
            if step == 20:
                state = np.random.uniform(0.5, 0.7, self.dim)
            elif step == 40:
                state = np.random.uniform(0.75, 0.95, self.dim)
            else:
                state += np.random.normal(0, 0.02, self.dim)
                state = np.clip(state, 0.0, 1.0)
            
            entity.update_state(state)
            detector.record(state)
            metrics = tracker.compute_metrics(entity, telos)
            
            transition = tracker.detect_phase_transition()
            if transition:
                phase_transitions_detected += 1
            
            emergence_score = detector.emergence_score()
            phase_history.append(telos.phase.value)
            max_history.append(metrics.actualization)
            mean_history.append(emergence_score)
        
        elapsed = time.time() - start
        return BenchmarkResult(
            benchmark_name=self.NAME,
            n_generations=self.n_steps,
            final_fitness=max_history[-1] if max_history else 0.0,
            mean_fitness_history=mean_history,
            max_fitness_history=max_history,
            time_seconds=elapsed,
            telos_phase_history=phase_history,
            metadata={
                'phase_transitions_detected': phase_transitions_detected,
                'dim': self.dim,
            },
        )


def _get_state(entity: Any) -> Optional[np.ndarray]:
    """Helper to extract state vector from any entity."""
    if hasattr(entity, 'get_state_vector'):
        return entity.get_state_vector()
    if hasattr(entity, 'state'):
        return np.asarray(entity.state)
    if hasattr(entity, 'genes'):
        weights = [g.weight for g in entity.genes if hasattr(g, 'weight')]
        return np.array(weights) if weights else None
    return None


BENCHMARKS: Dict[str, Any] = {
    'coherent_cluster': CoherentClusterBenchmark,
    'phase_separation': PhaseSeparationBenchmark,
    'emergence_tracking': EmergenceTrackingBenchmark,
}


def run_benchmark(name: str, **kwargs: Any) -> BenchmarkResult:
    """Run a named benchmark."""
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {list(BENCHMARKS.keys())}")
    bench_cls = BENCHMARKS[name]
    bench = bench_cls(**{k: v for k, v in kwargs.items() if k != 'n_generations'})
    n_gen = kwargs.get('n_generations', 50)
    if name == 'emergence_tracking':
        return bench.run()
    return bench.run(n_generations=n_gen)


def list_benchmarks() -> List[str]:
    return list(BENCHMARKS.keys())
