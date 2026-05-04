"""
Meta-Teleology — Systems that discover and evolve their own teloi.

Implements:
- Telos discovery from labeled trajectories
- Hierarchical teloi (DAG structure)
- Telos mutation and meta-evolution
- Autopoietic closure
- Collective/swarm teleology
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .core import ActualizationMetrics, ActualizationPhase, ActualizationTracker, Criterion, Telos
from .entity import SimpleEntity

# ─── Telos Discovery ────────────────────────────────────────────────────────


@dataclass
class TelosDiscoveryConfig:
    n_criteria: int = 3
    n_generations: int = 50
    population_size: int = 20
    mutation_rate: float = 0.1


def discover_telos(
    successful_trajectories: List[List[np.ndarray]],
    failed_trajectories: List[List[np.ndarray]],
    config: Optional[TelosDiscoveryConfig] = None,
    name: str = "discovered_telos",
) -> Telos:
    """Learn a Telos from labeled trajectories using evolutionary search.

    Args:
        successful_trajectories: List of state sequences that led to success.
        failed_trajectories: List of state sequences that led to failure.
        config: Discovery configuration.
        name: Name for the discovered telos.

    Returns:
        A learned Telos that separates successful from failed trajectories.
    """
    if config is None:
        config = TelosDiscoveryConfig()

    n_criteria = config.n_criteria

    def extract_features(trajectory: List[np.ndarray]) -> np.ndarray:
        """Extract summary features from a trajectory."""
        if not trajectory:
            return np.zeros(4)
        states = np.array([s for s in trajectory if s is not None])
        if len(states) == 0:
            return np.zeros(4)
        return np.array(
            [
                float(np.mean(states)),
                float(np.std(states)),
                float(np.mean(np.diff(states.mean(axis=-1)))) if len(states) > 1 else 0.0,
                float(states[-1].mean() - states[0].mean()) if len(states) > 1 else 0.0,
            ]
        )

    success_features = (
        np.array([extract_features(t) for t in successful_trajectories])
        if successful_trajectories
        else np.empty((0, 4))
    )
    failure_features = (
        np.array([extract_features(t) for t in failed_trajectories])
        if failed_trajectories
        else np.empty((0, 4))
    )

    n_features = 4

    def evaluate_criterion_weights(weights: np.ndarray) -> float:
        """Fitness: maximize separation of success vs failure features."""
        if len(success_features) == 0 or len(failure_features) == 0:
            return float(np.mean(weights))

        success_scores = (
            success_features @ weights[:n_features]
            if len(success_features) > 0
            else np.array([0.0])
        )
        failure_scores = (
            failure_features @ weights[:n_features]
            if len(failure_features) > 0
            else np.array([0.0])
        )

        success_mean = float(np.mean(success_scores))
        failure_mean = float(np.mean(failure_scores))
        separation = success_mean - failure_mean
        return float(np.tanh(separation))

    best_weights = np.ones(n_features) / n_features
    best_fitness = evaluate_criterion_weights(best_weights)

    for gen in range(config.n_generations):
        for _ in range(config.population_size):
            candidate = best_weights + np.random.normal(0, config.mutation_rate, n_features)
            candidate = np.clip(candidate, 0.01, 1.0)
            candidate /= candidate.sum()

            candidate_fitness = evaluate_criterion_weights(candidate)
            if candidate_fitness > best_fitness:
                best_fitness = candidate_fitness
                best_weights = candidate.copy()

    if len(success_features) > 0:
        attractor_values = success_features.mean(axis=0).tolist()
    else:
        attractor_values = [0.5] * n_features

    criteria_weights = np.abs(best_weights) / (np.abs(best_weights).sum() + 1e-10)

    feature_names = ["mean_activation", "activation_variance", "improvement_rate", "net_progress"]

    def make_evaluator(feature_idx: int, target_value: float) -> Callable[[Any], float]:
        def evaluator(entity: Any) -> float:
            if hasattr(entity, "get_state_vector"):
                sv = entity.get_state_vector()
            elif hasattr(entity, "genes"):
                sv = np.array([g.weight for g in entity.genes if hasattr(g, "weight")])
            else:
                return 0.5
            if len(sv) == 0:
                return 0.5
            features = np.array(
                [
                    float(np.mean(sv)),
                    float(np.std(sv)),
                    0.0,
                    float(sv.mean()),
                ]
            )
            return float(1.0 - min(abs(features[feature_idx] - target_value), 1.0))

        return evaluator

    criteria = [
        Criterion(
            name=feature_names[i % len(feature_names)],
            description=f"Discovered criterion {i}",
            weight=float(criteria_weights[i % len(criteria_weights)]),
            evaluator=make_evaluator(i % n_features, attractor_values[i % len(attractor_values)]),
            target_value=1.0,
        )
        for i in range(n_criteria)
    ]

    return Telos(
        name=name,
        description=f"Discovered telos (fitness={best_fitness:.3f})",
        actualization_criteria=criteria,
        attractor_state={"weights": attractor_values},
    )


# ─── Hierarchical Teloi ──────────────────────────────────────────────────────


@dataclass
class TelosNode:
    """A node in the hierarchical telos DAG."""

    telos: Telos
    children: List["TelosNode"] = field(default_factory=list)
    parent: Optional["TelosNode"] = None
    activation_threshold: float = 0.8
    is_active: bool = True

    def add_child(self, child: "TelosNode") -> None:
        child.parent = self
        self.children.append(child)

    def check_completion(self) -> bool:
        return self.telos.current_actualization >= self.activation_threshold

    def activate_children(self) -> List["TelosNode"]:
        """Activate child teloi when this node completes."""
        if self.check_completion():
            for child in self.children:
                child.is_active = True
            return self.children
        return []


class TelосHierarchy:
    """DAG-structured hierarchy of teloi.

    Lower-level teloi serve higher-level ones.
    Completion of a lower telos triggers emergence at the next level.
    """

    def __init__(self, root: TelosNode):
        self.root = root
        self._all_nodes: List[TelosNode] = []
        self._collect_nodes(root)

    def _collect_nodes(self, node: TelosNode) -> None:
        self._all_nodes.append(node)
        for child in node.children:
            self._collect_nodes(child)

    @property
    def active_teloi(self) -> List[Telos]:
        return [n.telos for n in self._all_nodes if n.is_active]

    def evaluate(self, entity: Any) -> Dict[str, float]:
        """Evaluate entity against all active teloi and trigger transitions."""
        results = {}
        tracker = ActualizationTracker()

        for node in self._all_nodes:
            if not node.is_active:
                continue
            metrics = tracker.compute_metrics(entity, node.telos)
            results[node.telos.name] = metrics.actualization

            activated = node.activate_children()
            for child in activated:
                results[f"{child.telos.name}_activated"] = 1.0

        return results

    def depth(self) -> int:
        def _depth(node: TelosNode) -> int:
            if not node.children:
                return 1
            return 1 + max(_depth(c) for c in node.children)

        return _depth(self.root)


# Alias with correct ASCII spelling
TelosHierarchy = TelосHierarchy


# ─── Telos Evolution ─────────────────────────────────────────────────────────


def mutate_telos(telos: Telos, mutation_rate: float = 0.1) -> Telos:
    """Create a mutated copy of a Telos.

    Mutations: perturb criterion weights, perturb attractor state.
    """
    new_telos = copy.deepcopy(telos)

    for criterion in new_telos.actualization_criteria:
        if np.random.random() < mutation_rate:
            criterion.weight = float(
                np.clip(criterion.weight + np.random.normal(0, 0.1), 0.01, 1.0)
            )

    total = sum(c.weight for c in new_telos.actualization_criteria)
    if total > 0:
        for criterion in new_telos.actualization_criteria:
            criterion.weight /= total

    if isinstance(new_telos.attractor_state, dict):
        for key, val in new_telos.attractor_state.items():
            if isinstance(val, (int, float)) and np.random.random() < mutation_rate:
                new_telos.attractor_state[key] = float(
                    np.clip(val + np.random.normal(0, 0.05), 0.0, 1.0)
                )
            elif isinstance(val, list) and np.random.random() < mutation_rate:
                new_telos.attractor_state[key] = [
                    float(np.clip(v + np.random.normal(0, 0.05), 0.0, 1.0)) for v in val
                ]

    new_telos.name = f"{telos.name}_mutant"
    return new_telos


def crossover_teloi(telos_a: Telos, telos_b: Telos) -> Telos:
    """Create an offspring Telos by crossing over two parent teloi.

    Combines criteria from both parents (uniform crossover on weights).
    """
    new_telos = copy.deepcopy(telos_a)

    for i, criterion in enumerate(new_telos.actualization_criteria):
        if i < len(telos_b.actualization_criteria):
            alpha = np.random.uniform(0.3, 0.7)
            criterion.weight = (
                alpha * criterion.weight + (1 - alpha) * telos_b.actualization_criteria[i].weight
            )

    total = sum(c.weight for c in new_telos.actualization_criteria)
    if total > 0:
        for criterion in new_telos.actualization_criteria:
            criterion.weight /= total

    new_telos.name = f"{telos_a.name}_{telos_b.name}_offspring"
    return new_telos


def evolve_teloi(
    seed_teloi: List[Telos],
    evaluation_entities: List[Any],
    n_generations: int = 20,
    population_size: int = 10,
    mutation_rate: float = 0.1,
) -> Telos:
    """Meta-evolution: improve telos quality over generations.

    Fitness of a Telos = mean actualization score across evaluation entities.

    Args:
        seed_teloi: Initial pool of teloi
        evaluation_entities: Entities used to score each telos
        n_generations: Number of meta-evolution generations
        population_size: Population size for meta-evolution
        mutation_rate: Mutation rate for telos genes

    Returns:
        Best telos found
    """
    population = list(seed_teloi)

    while len(population) < population_size:
        seed = population[np.random.randint(len(population))]
        population.append(mutate_telos(seed, mutation_rate))

    def score_telos(telos: Telos) -> float:
        tracker = ActualizationTracker()
        scores = []
        for entity in evaluation_entities:
            try:
                metrics = tracker.compute_metrics(entity, telos)
                scores.append(metrics.overall_health)
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    best_telos = max(population, key=score_telos)
    best_score = score_telos(best_telos)

    for gen in range(n_generations):
        scores = [(score_telos(t), t) for t in population]
        scores.sort(key=lambda x: x[0], reverse=True)

        survivors = [t for _, t in scores[: max(1, population_size // 2)]]

        offspring = []
        while len(offspring) < population_size - len(survivors):
            if len(survivors) >= 2 and np.random.random() < 0.5:
                pa = survivors[np.random.randint(len(survivors))]
                pb = survivors[np.random.randint(len(survivors))]
                child = crossover_teloi(pa, pb)
            else:
                parent = survivors[np.random.randint(len(survivors))]
                child = mutate_telos(parent, mutation_rate)
            offspring.append(child)

        population = survivors + offspring

        current_best = scores[0][1]
        current_score = scores[0][0]
        if current_score > best_score:
            best_score = current_score
            best_telos = current_best

    best_telos.name = f"{best_telos.name}_evolved"
    return best_telos


# ─── Autopoietic Closure ─────────────────────────────────────────────────────


class AutopoieticSystem:
    """A system that generates a new, more complex Telos when fully actualized.

    Implements Whitehead's 'concrescence' — growing together into unity,
    then generating a new, higher-order purpose.

    When the current telos reaches ACTUALIZED phase, the system generates
    a new telos by composing/extending the current one.
    """

    def __init__(
        self,
        initial_telos: Telos,
        complexity_growth: float = 1.5,
        max_levels: int = 5,
    ):
        self.current_telos = initial_telos
        self.complexity_growth = complexity_growth
        self.max_levels = max_levels
        self.level: int = 0
        self.history: List[Telos] = [initial_telos]
        self.tracker = ActualizationTracker()

    def _generate_higher_telos(self, actualized_telos: Telos) -> Telos:
        """Generate a new, more complex telos from the actualized one."""
        new_telos = copy.deepcopy(actualized_telos)
        n_new_criteria = max(
            1, int(len(actualized_telos.actualization_criteria) * (self.complexity_growth - 1))
        )

        for i in range(n_new_criteria):
            source = actualized_telos.actualization_criteria[
                i % len(actualized_telos.actualization_criteria)
            ]

            source_eval = source.evaluator
            higher_target = min(1.0, source.target_value * 1.1)

            def make_higher_evaluator(base_eval: Callable, target: float) -> Callable[[Any], float]:
                def higher_eval(entity: Any) -> float:
                    base_score = base_eval(entity)
                    return float(max(0.0, base_score - (1.0 - target)))

                return higher_eval

            new_criterion = Criterion(
                name=f"{source.name}_higher_{i}",
                description=f"Elevated {source.description} (level {self.level + 1})",
                weight=source.weight / n_new_criteria,
                evaluator=make_higher_evaluator(source_eval, higher_target),
                target_value=higher_target,
            )
            new_telos.actualization_criteria.append(new_criterion)

        total = sum(c.weight for c in new_telos.actualization_criteria)
        if total > 0:
            for c in new_telos.actualization_criteria:
                c.weight /= total

        if isinstance(new_telos.attractor_state, dict):
            for key, val in new_telos.attractor_state.items():
                if isinstance(val, (int, float)):
                    new_telos.attractor_state[key] = min(1.0, float(val) * 1.05)
                elif isinstance(val, list):
                    new_telos.attractor_state[key] = [min(1.0, v * 1.05) for v in val]

        new_telos.name = f"{actualized_telos.name}_L{self.level + 1}"
        new_telos.current_actualization = 0.0
        new_telos.phase = ActualizationPhase.POTENTIAL
        return new_telos

    def step(self, entity: Any) -> Tuple[ActualizationMetrics, bool]:
        """Evaluate entity and potentially trigger autopoietic closure.

        Returns:
            (metrics, closure_triggered) where closure_triggered is True if
            a new telos was generated.
        """
        metrics = self.tracker.compute_metrics(entity, self.current_telos)

        closure_triggered = False
        if (
            self.current_telos.phase == ActualizationPhase.ACTUALIZED
            and self.level < self.max_levels - 1
        ):
            new_telos = self._generate_higher_telos(self.current_telos)
            self.history.append(new_telos)
            self.current_telos = new_telos
            self.level += 1
            closure_triggered = True
            self.tracker = ActualizationTracker()

        return metrics, closure_triggered

    @property
    def telos_lineage(self) -> List[str]:
        return [t.name for t in self.history]


# ─── Collective/Swarm Teleology ──────────────────────────────────────────────


@dataclass
class SwarmMetrics:
    """Group-level actualization metrics."""

    individual_scores: List[float] = field(default_factory=list)
    collective_actualization: float = 0.0
    coherence: float = 0.0
    diversity: float = 0.0
    emergent_phase: ActualizationPhase = ActualizationPhase.POTENTIAL

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.individual_scores)) if self.individual_scores else 0.0

    @property
    def std_score(self) -> float:
        return float(np.std(self.individual_scores)) if self.individual_scores else 0.0


class SwarmTeleology:
    """Multiple agents sharing a distributed Telos.

    Tracks both individual and collective actualization.
    Stigmergic attractor fields: each agent's state influences
    the shared attractor landscape.
    """

    def __init__(
        self,
        shared_telos: Telos,
        stigmergy_strength: float = 0.1,
    ):
        self.shared_telos = shared_telos
        self.stigmergy_strength = stigmergy_strength
        self._agents: Dict[str, Any] = {}
        self._trackers: Dict[str, ActualizationTracker] = {}
        self._shared_attractor: Optional[np.ndarray] = None
        self.collective_history: List[SwarmMetrics] = []

    def register_agent(self, agent_id: str, entity: Any) -> None:
        self._agents[agent_id] = entity
        self._trackers[agent_id] = ActualizationTracker()

    def _update_stigmergic_attractor(self) -> None:
        """Update shared attractor based on high-performing agents (stigmergy)."""
        if not self._agents:
            return

        states = []
        fitnesses = []
        for agent_id, entity in self._agents.items():
            tracker = self._trackers[agent_id]
            if tracker.history:
                last_metrics = tracker.history[-1]
                score = last_metrics.overall_health
            else:
                score = 0.0

            if hasattr(entity, "get_state_vector"):
                sv = entity.get_state_vector()
            elif hasattr(entity, "genes"):
                sv = np.array([g.weight for g in entity.genes if hasattr(g, "weight")])
            else:
                continue

            states.append(sv)
            fitnesses.append(score)

        if not states:
            return

        fitnesses_arr = np.array(fitnesses)
        total = fitnesses_arr.sum()
        if total < 1e-10:
            return

        weights = fitnesses_arr / total

        max_dim = max(len(s) for s in states)
        padded = np.array([np.pad(s, (0, max_dim - len(s))) for s in states])

        stigmergic_center = np.average(padded, axis=0, weights=weights)

        if self._shared_attractor is None:
            self._shared_attractor = stigmergic_center
        else:
            alpha = self.stigmergy_strength
            min_len = min(len(self._shared_attractor), len(stigmergic_center))
            self._shared_attractor[:min_len] = (1 - alpha) * self._shared_attractor[
                :min_len
            ] + alpha * stigmergic_center[:min_len]

    def step(self) -> SwarmMetrics:
        """Evaluate all agents and update collective metrics."""
        individual_scores = []
        all_states = []

        for agent_id, entity in self._agents.items():
            tracker = self._trackers[agent_id]
            metrics = tracker.compute_metrics(entity, self.shared_telos)
            individual_scores.append(metrics.overall_health)

            if hasattr(entity, "get_state_vector"):
                all_states.append(entity.get_state_vector())

        self._update_stigmergic_attractor()

        collective = float(np.mean(individual_scores)) if individual_scores else 0.0

        coherence = 1.0 - float(np.std(individual_scores)) if individual_scores else 0.0
        coherence = max(0.0, coherence)

        diversity = 0.5
        if len(all_states) > 1:
            try:
                states_matrix = np.array(
                    [np.pad(s, (0, max(len(x) for x in all_states) - len(s))) for s in all_states]
                )
                pairwise_distances = []
                for i in range(len(states_matrix)):
                    for j in range(i + 1, len(states_matrix)):
                        d = float(np.linalg.norm(states_matrix[i] - states_matrix[j]))
                        pairwise_distances.append(d)
                diversity = float(np.tanh(np.mean(pairwise_distances)))
            except Exception:
                diversity = 0.5

        if collective < 0.2:
            phase = ActualizationPhase.POTENTIAL
        elif collective < 0.4:
            phase = ActualizationPhase.EMERGENT
        elif collective < 0.6:
            phase = ActualizationPhase.DEVELOPING
        elif collective < 0.8:
            phase = ActualizationPhase.ACTUALIZING
        else:
            phase = ActualizationPhase.ACTUALIZED

        swarm_metrics = SwarmMetrics(
            individual_scores=individual_scores,
            collective_actualization=collective,
            coherence=coherence,
            diversity=diversity,
            emergent_phase=phase,
        )
        self.collective_history.append(swarm_metrics)
        return swarm_metrics

    @property
    def stigmergic_attractor(self) -> Optional[np.ndarray]:
        return self._shared_attractor.copy() if self._shared_attractor is not None else None
