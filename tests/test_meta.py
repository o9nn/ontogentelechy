"""Tests for ontogentelechy/meta.py (Phase 5)."""

import numpy as np
import pytest

from ontogentelechy.core import ActualizationPhase, ActualizationTracker, Criterion, Telos
from ontogentelechy.entity import SimpleEntity, SimpleGene
from ontogentelechy.examples import create_semantic_coherence_telos
from ontogentelechy.meta import (
    AutopoieticSystem,
    SwarmMetrics,
    SwarmTeleology,
    TelosDiscoveryConfig,
    TelosHierarchy,
    TelosNode,
    TelосHierarchy,
    crossover_teloi,
    discover_telos,
    evolve_teloi,
    mutate_telos,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_telos(name: str = "test", n_criteria: int = 3) -> Telos:
    criteria = [
        Criterion(
            name=f"c{i}",
            description=f"criterion {i}",
            weight=1.0 / n_criteria,
            evaluator=lambda e, i=i: float(np.mean(e.get_state_vector())),
            target_value=1.0,
        )
        for i in range(n_criteria)
    ]
    return Telos(
        name=name,
        description="test telos",
        actualization_criteria=criteria,
        attractor_state={"target": 0.8},
    )


def make_entity(dim: int = 5, state_value: float = 0.5) -> SimpleEntity:
    state = np.full(dim, state_value)
    genes = [SimpleGene(weight=state_value) for _ in range(dim)]
    return SimpleEntity(state=state, genes=genes, fitness=state_value)


def make_trajectories(n: int = 3, length: int = 5) -> list:
    return [[np.random.uniform(0.5, 1.0, 4) for _ in range(length)] for _ in range(n)]


def make_failed_trajectories(n: int = 3, length: int = 5) -> list:
    return [[np.random.uniform(0.0, 0.3, 4) for _ in range(length)] for _ in range(n)]


# ─── discover_telos ───────────────────────────────────────────────────────────


class TestDiscoverTelos:
    def test_returns_telos(self):
        success = make_trajectories()
        failure = make_failed_trajectories()
        t = discover_telos(success, failure, name="learned")
        assert isinstance(t, Telos)

    def test_correct_name(self):
        t = discover_telos(make_trajectories(), make_failed_trajectories(), name="my_telos")
        assert t.name == "my_telos"

    def test_correct_n_criteria(self):
        config = TelosDiscoveryConfig(n_criteria=4, n_generations=5, population_size=5)
        t = discover_telos(make_trajectories(), make_failed_trajectories(), config=config)
        assert len(t.actualization_criteria) == 4

    def test_default_n_criteria(self):
        config = TelosDiscoveryConfig(n_criteria=3, n_generations=5, population_size=5)
        t = discover_telos(make_trajectories(), make_failed_trajectories(), config=config)
        assert len(t.actualization_criteria) == 3

    def test_empty_successful_trajectories(self):
        config = TelosDiscoveryConfig(n_generations=5, population_size=5)
        t = discover_telos([], make_failed_trajectories(), config=config)
        assert isinstance(t, Telos)
        assert len(t.actualization_criteria) == config.n_criteria

    def test_empty_failed_trajectories(self):
        config = TelosDiscoveryConfig(n_generations=5, population_size=5)
        t = discover_telos(make_trajectories(), [], config=config)
        assert isinstance(t, Telos)

    def test_both_empty(self):
        config = TelosDiscoveryConfig(n_generations=5, population_size=5)
        t = discover_telos([], [], config=config)
        assert isinstance(t, Telos)

    def test_criteria_can_evaluate_entity(self):
        config = TelosDiscoveryConfig(n_generations=5, population_size=5)
        t = discover_telos(make_trajectories(), make_failed_trajectories(), config=config)
        entity = make_entity()
        for c in t.actualization_criteria:
            score = c.evaluate(entity)
            assert 0.0 <= score <= 1.0

    def test_attractor_state_has_weights(self):
        config = TelosDiscoveryConfig(n_generations=5, population_size=5)
        t = discover_telos(make_trajectories(), make_failed_trajectories(), config=config)
        assert "weights" in t.attractor_state

    def test_config_default(self):
        t = discover_telos(
            make_trajectories(2, 3),
            make_failed_trajectories(2, 3),
            config=TelosDiscoveryConfig(n_generations=5, population_size=5),
        )
        assert isinstance(t, Telos)


# ─── TelosNode ────────────────────────────────────────────────────────────────


class TestTelosNode:
    def test_creation(self):
        telos = make_telos("root")
        node = TelosNode(telos=telos)
        assert node.telos is telos
        assert node.children == []
        assert node.parent is None
        assert node.is_active is True
        assert node.activation_threshold == pytest.approx(0.8)

    def test_check_completion_false_when_low(self):
        telos = make_telos()
        telos.current_actualization = 0.3
        node = TelosNode(telos=telos)
        assert not node.check_completion()

    def test_check_completion_true_when_high(self):
        telos = make_telos()
        telos.current_actualization = 0.9
        node = TelosNode(telos=telos)
        assert node.check_completion()

    def test_check_completion_at_threshold(self):
        telos = make_telos()
        telos.current_actualization = 0.8
        node = TelosNode(telos=telos, activation_threshold=0.8)
        assert node.check_completion()

    def test_add_child(self):
        parent_telos = make_telos("parent")
        child_telos = make_telos("child")
        parent = TelosNode(telos=parent_telos)
        child = TelosNode(telos=child_telos)

        parent.add_child(child)

        assert child in parent.children
        assert child.parent is parent

    def test_activate_children_when_complete(self):
        parent_telos = make_telos("parent")
        parent_telos.current_actualization = 0.9

        child_telos = make_telos("child")
        parent = TelosNode(telos=parent_telos)
        child = TelosNode(telos=child_telos, is_active=False)
        parent.add_child(child)

        activated = parent.activate_children()
        assert child in activated
        assert child.is_active is True

    def test_activate_children_when_incomplete(self):
        parent_telos = make_telos("parent")
        parent_telos.current_actualization = 0.3

        child_telos = make_telos("child")
        parent = TelosNode(telos=parent_telos)
        child = TelosNode(telos=child_telos, is_active=False)
        parent.add_child(child)

        activated = parent.activate_children()
        assert activated == []
        assert child.is_active is False

    def test_multiple_children(self):
        root = TelosNode(telos=make_telos("root"))
        c1 = TelosNode(telos=make_telos("c1"))
        c2 = TelosNode(telos=make_telos("c2"))
        root.add_child(c1)
        root.add_child(c2)
        assert len(root.children) == 2


# ─── TelosHierarchy / TelосHierarchy ─────────────────────────────────────────


class TestTelosHierarchy:
    def _make_hierarchy(self) -> TelosHierarchy:
        root_telos = make_telos("root")
        child_telos = make_telos("child")
        grandchild_telos = make_telos("grandchild")

        root_node = TelosNode(telos=root_telos)
        child_node = TelosNode(telos=child_telos)
        grandchild_node = TelosNode(telos=grandchild_telos)

        root_node.add_child(child_node)
        child_node.add_child(grandchild_node)

        return TelosHierarchy(root_node)

    def test_creation(self):
        h = self._make_hierarchy()
        assert h.root is not None

    def test_depth_single_node(self):
        node = TelosNode(telos=make_telos())
        h = TelosHierarchy(node)
        assert h.depth() == 1

    def test_depth_three_levels(self):
        h = self._make_hierarchy()
        assert h.depth() == 3

    def test_active_teloi(self):
        h = self._make_hierarchy()
        active = h.active_teloi
        assert len(active) == 3  # all active by default

    def test_active_teloi_with_inactive(self):
        root_telos = make_telos("root")
        child_telos = make_telos("child")
        root_node = TelosNode(telos=root_telos)
        child_node = TelosNode(telos=child_telos, is_active=False)
        root_node.add_child(child_node)
        h = TelosHierarchy(root_node)
        active = h.active_teloi
        assert len(active) == 1  # only root

    def test_evaluate_returns_dict(self):
        h = self._make_hierarchy()
        entity = make_entity()
        results = h.evaluate(entity)
        assert isinstance(results, dict)
        assert "root" in results
        assert "child" in results

    def test_evaluate_scores_in_range(self):
        h = self._make_hierarchy()
        entity = make_entity()
        results = h.evaluate(entity)
        for key, val in results.items():
            assert 0.0 <= val <= 1.0

    def test_alias_same_as_cyrillic(self):
        """TelosHierarchy (ASCII alias) is the same class as TelосHierarchy (Cyrillic)."""
        assert TelosHierarchy is TelосHierarchy

    def test_collect_all_nodes(self):
        h = self._make_hierarchy()
        assert len(h._all_nodes) == 3


# ─── mutate_telos ─────────────────────────────────────────────────────────────


class TestMutateTelos:
    def test_returns_telos(self):
        t = make_telos()
        mutated = mutate_telos(t)
        assert isinstance(mutated, Telos)

    def test_returns_different_name(self):
        t = make_telos("original")
        mutated = mutate_telos(t)
        assert mutated.name == "original_mutant"

    def test_original_unchanged(self):
        t = make_telos()
        original_weights = [c.weight for c in t.actualization_criteria]
        _ = mutate_telos(t, mutation_rate=1.0)
        current_weights = [c.weight for c in t.actualization_criteria]
        assert original_weights == current_weights

    def test_weights_sum_to_one(self):
        t = make_telos()
        for _ in range(10):
            mutated = mutate_telos(t, mutation_rate=0.5)
            total = sum(c.weight for c in mutated.actualization_criteria)
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_weights_are_positive(self):
        t = make_telos()
        for _ in range(10):
            mutated = mutate_telos(t, mutation_rate=1.0)
            for c in mutated.actualization_criteria:
                assert c.weight > 0

    def test_high_mutation_rate_changes_weights(self):
        np.random.seed(42)
        t = make_telos("t", n_criteria=5)
        mutated = mutate_telos(t, mutation_rate=1.0)
        original_weights = [c.weight for c in t.actualization_criteria]
        mutated_weights = [c.weight for c in mutated.actualization_criteria]
        # At mutation_rate=1.0 all should mutate — at least one should differ
        assert any(abs(o - m) > 1e-9 for o, m in zip(original_weights, mutated_weights))

    def test_mutates_attractor_state(self):
        t = make_telos()
        t.attractor_state = {"key": 0.5}
        mutated = mutate_telos(t, mutation_rate=1.0)
        # Should still be a dict
        assert isinstance(mutated.attractor_state, dict)


# ─── crossover_teloi ──────────────────────────────────────────────────────────


class TestCrossoverTeloi:
    def test_returns_telos(self):
        ta = make_telos("a")
        tb = make_telos("b")
        offspring = crossover_teloi(ta, tb)
        assert isinstance(offspring, Telos)

    def test_name_combines_parents(self):
        ta = make_telos("alpha")
        tb = make_telos("beta")
        offspring = crossover_teloi(ta, tb)
        assert "alpha" in offspring.name and "beta" in offspring.name

    def test_weights_sum_to_one(self):
        ta = make_telos("a", n_criteria=3)
        tb = make_telos("b", n_criteria=3)
        for _ in range(10):
            offspring = crossover_teloi(ta, tb)
            total = sum(c.weight for c in offspring.actualization_criteria)
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_same_number_of_criteria(self):
        ta = make_telos("a", n_criteria=4)
        tb = make_telos("b", n_criteria=4)
        offspring = crossover_teloi(ta, tb)
        assert len(offspring.actualization_criteria) == 4

    def test_parents_unchanged(self):
        ta = make_telos("a")
        tb = make_telos("b")
        wa = [c.weight for c in ta.actualization_criteria]
        wb = [c.weight for c in tb.actualization_criteria]
        _ = crossover_teloi(ta, tb)
        assert [c.weight for c in ta.actualization_criteria] == wa
        assert [c.weight for c in tb.actualization_criteria] == wb


# ─── evolve_teloi ─────────────────────────────────────────────────────────────


class TestEvolveTeloi:
    def test_returns_telos(self):
        seeds = [make_telos("s1"), make_telos("s2")]
        entities = [make_entity() for _ in range(3)]
        result = evolve_teloi(seeds, entities, n_generations=3, population_size=4)
        assert isinstance(result, Telos)

    def test_name_ends_with_evolved(self):
        seeds = [make_telos("base")]
        entities = [make_entity()]
        result = evolve_teloi(seeds, entities, n_generations=2, population_size=4)
        assert result.name.endswith("_evolved")

    def test_zero_generations(self):
        seeds = [make_telos("base")]
        entities = [make_entity()]
        result = evolve_teloi(seeds, entities, n_generations=0, population_size=4)
        assert isinstance(result, Telos)

    def test_single_seed(self):
        seeds = [make_telos("single")]
        entities = [make_entity()]
        result = evolve_teloi(seeds, entities, n_generations=3, population_size=4)
        assert isinstance(result, Telos)

    def test_multiple_entities(self):
        seeds = [make_telos("s1"), make_telos("s2")]
        entities = [make_entity(dim=5, state_value=v) for v in [0.3, 0.5, 0.7, 0.9]]
        result = evolve_teloi(seeds, entities, n_generations=5, population_size=6)
        assert isinstance(result, Telos)

    def test_weights_normalized(self):
        seeds = [make_telos("s1")]
        entities = [make_entity()]
        result = evolve_teloi(seeds, entities, n_generations=3, population_size=4)
        total = sum(c.weight for c in result.actualization_criteria)
        assert total == pytest.approx(1.0, abs=1e-5)


# ─── AutopoieticSystem ────────────────────────────────────────────────────────


class TestAutopoieticSystem:
    def test_creation(self):
        telos = make_telos("initial")
        system = AutopoieticSystem(telos)
        assert system.current_telos is telos
        assert system.level == 0
        assert len(system.history) == 1
        assert system.max_levels == 5

    def test_step_returns_metrics_and_bool(self):
        telos = make_telos()
        system = AutopoieticSystem(telos)
        entity = make_entity()
        result = system.step(entity)
        assert isinstance(result, tuple)
        assert len(result) == 2
        metrics, closure = result
        assert hasattr(metrics, "actualization")
        assert isinstance(closure, bool)

    def test_no_closure_without_actualization(self):
        telos = make_telos()
        telos.current_actualization = 0.3
        telos.phase = ActualizationPhase.POTENTIAL
        system = AutopoieticSystem(telos)
        entity = make_entity(state_value=0.1)

        # Force phase to non-ACTUALIZED
        system.current_telos.phase = ActualizationPhase.POTENTIAL
        _, closure = system.step(entity)
        assert closure is False

    def test_closure_triggers_when_actualized(self):
        telos = make_telos("initial")
        telos.phase = ActualizationPhase.ACTUALIZED
        telos.current_actualization = 0.95
        system = AutopoieticSystem(telos, max_levels=5)

        entity = make_entity(state_value=0.9)
        # Manually force ACTUALIZED state
        system.current_telos.phase = ActualizationPhase.ACTUALIZED

        _, closure = system.step(entity)
        assert closure is True
        assert system.level == 1
        assert len(system.history) == 2

    def test_closure_generates_higher_telos(self):
        telos = make_telos("base", n_criteria=2)
        telos.phase = ActualizationPhase.ACTUALIZED
        system = AutopoieticSystem(telos, complexity_growth=1.5, max_levels=5)
        system.current_telos.phase = ActualizationPhase.ACTUALIZED

        entity = make_entity()
        _, closure = system.step(entity)

        if closure:
            new_telos = system.current_telos
            assert new_telos.name != "base"
            # More criteria than original
            assert len(new_telos.actualization_criteria) > len(telos.actualization_criteria)

    def test_max_levels_prevents_closure(self):
        telos = make_telos("top")
        telos.phase = ActualizationPhase.ACTUALIZED
        system = AutopoieticSystem(telos, max_levels=1)
        system.current_telos.phase = ActualizationPhase.ACTUALIZED

        entity = make_entity()
        _, closure = system.step(entity)
        assert closure is False  # already at max level (0 < 1-1=0 is False)

    def test_telos_lineage(self):
        telos = make_telos("start")
        system = AutopoieticSystem(telos)
        lineage = system.telos_lineage
        assert lineage == ["start"]

    def test_telos_lineage_grows_on_closure(self):
        telos = make_telos("start")
        telos.phase = ActualizationPhase.ACTUALIZED
        system = AutopoieticSystem(telos, max_levels=5)
        system.current_telos.phase = ActualizationPhase.ACTUALIZED

        entity = make_entity()
        _, closure = system.step(entity)

        if closure:
            assert len(system.telos_lineage) == 2


# ─── SwarmTeleology ───────────────────────────────────────────────────────────


class TestSwarmTeleology:
    def _make_swarm(self, n_agents: int = 3) -> SwarmTeleology:
        telos = make_telos("shared")
        swarm = SwarmTeleology(telos, stigmergy_strength=0.2)
        for i in range(n_agents):
            entity = make_entity(state_value=np.random.uniform(0.3, 0.9))
            swarm.register_agent(f"agent_{i}", entity)
        return swarm

    def test_creation(self):
        telos = make_telos()
        swarm = SwarmTeleology(telos)
        assert swarm.shared_telos is telos
        assert swarm.stigmergy_strength == pytest.approx(0.1)
        assert len(swarm._agents) == 0

    def test_register_agent(self):
        telos = make_telos()
        swarm = SwarmTeleology(telos)
        entity = make_entity()
        swarm.register_agent("ag1", entity)
        assert "ag1" in swarm._agents
        assert "ag1" in swarm._trackers

    def test_step_returns_swarm_metrics(self):
        swarm = self._make_swarm(3)
        metrics = swarm.step()
        assert isinstance(metrics, SwarmMetrics)

    def test_step_metrics_have_correct_fields(self):
        swarm = self._make_swarm(3)
        metrics = swarm.step()
        assert hasattr(metrics, "individual_scores")
        assert hasattr(metrics, "collective_actualization")
        assert hasattr(metrics, "coherence")
        assert hasattr(metrics, "diversity")
        assert hasattr(metrics, "emergent_phase")

    def test_step_individual_scores_count(self):
        swarm = self._make_swarm(4)
        metrics = swarm.step()
        assert len(metrics.individual_scores) == 4

    def test_step_collective_in_range(self):
        swarm = self._make_swarm(3)
        metrics = swarm.step()
        assert 0.0 <= metrics.collective_actualization <= 1.0

    def test_step_coherence_in_range(self):
        swarm = self._make_swarm(3)
        metrics = swarm.step()
        assert 0.0 <= metrics.coherence <= 1.0

    def test_stigmergic_attractor_none_before_step(self):
        telos = make_telos()
        swarm = SwarmTeleology(telos)
        swarm.register_agent("a", make_entity())
        assert swarm.stigmergic_attractor is None

    def test_stigmergic_attractor_not_none_after_step(self):
        swarm = self._make_swarm(3)
        swarm.step()
        assert swarm.stigmergic_attractor is not None

    def test_stigmergic_attractor_returns_copy(self):
        swarm = self._make_swarm(2)
        swarm.step()
        att1 = swarm.stigmergic_attractor
        att2 = swarm.stigmergic_attractor
        att1[0] = 999.0
        assert swarm._shared_attractor[0] != 999.0

    def test_collective_history_grows(self):
        swarm = self._make_swarm(2)
        swarm.step()
        swarm.step()
        assert len(swarm.collective_history) == 2

    def test_mean_score_property(self):
        m = SwarmMetrics(individual_scores=[0.4, 0.6, 0.8])
        assert m.mean_score == pytest.approx(0.6)

    def test_std_score_property(self):
        m = SwarmMetrics(individual_scores=[0.5, 0.5, 0.5])
        assert m.std_score == pytest.approx(0.0)

    def test_empty_swarm_step(self):
        telos = make_telos()
        swarm = SwarmTeleology(telos)
        metrics = swarm.step()
        assert metrics.collective_actualization == pytest.approx(0.0)
        assert metrics.individual_scores == []

    def test_phase_determined_by_collective(self):
        swarm = self._make_swarm(0)
        # Manually create high-scoring entities
        telos = make_telos()
        swarm2 = SwarmTeleology(telos)
        for i in range(3):
            e = make_entity(state_value=0.9)
            e.fitness = 0.95
            swarm2.register_agent(f"a{i}", e)
        metrics = swarm2.step()
        assert isinstance(metrics.emergent_phase, ActualizationPhase)

    def test_single_agent_step(self):
        telos = make_telos()
        swarm = SwarmTeleology(telos)
        swarm.register_agent("solo", make_entity())
        metrics = swarm.step()
        assert len(metrics.individual_scores) == 1
        # With one agent diversity defaults to 0.5
        assert metrics.diversity == pytest.approx(0.5)
