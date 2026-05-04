"""Tests for ontogentelechy/adapters.py (Phase 4)."""

import numpy as np
import pytest

from ontogentelechy.adapters import NumpyArrayEntity, RLAgentEntity, create_training_observer
from ontogentelechy.core import ActualizationTracker
from ontogentelechy.examples import create_semantic_coherence_telos

# ─── NumpyArrayEntity ────────────────────────────────────────────────────────


class TestNumpyArrayEntity:
    def test_creation_basic(self):
        weights = np.array([0.1, 0.5, 0.9, 0.3])
        entity = NumpyArrayEntity(weights, name="test_entity")
        assert entity.name == "test_entity"
        assert len(entity.genes) == 4
        assert entity.fitness is None
        assert entity.age == 0

    def test_creation_2d_flattened(self):
        weights = np.ones((3, 4))
        entity = NumpyArrayEntity(weights, flatten=True)
        assert len(entity.genes) == 12

    def test_creation_no_flatten(self):
        weights = np.array([0.1, 0.5, 0.9, 0.3])
        entity = NumpyArrayEntity(weights, flatten=False)
        # shape preserved (1D stays 1D)
        assert entity._weights.shape == (4,)
        assert len(entity.genes) == 4

    def test_state_in_01(self):
        weights = np.array([-5.0, 0.0, 5.0, 10.0])
        entity = NumpyArrayEntity(weights)
        state = entity.state
        assert state.min() >= 0.0 - 1e-9
        assert state.max() <= 1.0 + 1e-9

    def test_state_uniform_array(self):
        weights = np.ones(5) * 3.0
        entity = NumpyArrayEntity(weights)
        # All same → normalized to 0
        assert np.allclose(entity.state, 0.0)

    def test_get_state_vector(self):
        weights = np.array([1.0, 2.0, 3.0])
        entity = NumpyArrayEntity(weights)
        sv = entity.get_state_vector()
        assert sv.shape == (3,)
        assert np.allclose(sv, entity.state)

    def test_genes_weights_in_01(self):
        weights = np.array([100.0, -100.0, 50.0])
        entity = NumpyArrayEntity(weights)
        for gene in entity.genes:
            assert 0.0 <= gene.weight <= 1.0

    def test_update_weights_grows_history(self):
        weights = np.array([0.1, 0.2, 0.3])
        entity = NumpyArrayEntity(weights)
        assert len(entity.weight_history) == 1

        entity.update_weights(np.array([0.4, 0.5, 0.6]))
        assert len(entity.weight_history) == 2

        entity.update_weights(np.array([0.7, 0.8, 0.9]))
        assert len(entity.weight_history) == 3

    def test_update_weights_updates_genes(self):
        weights = np.array([0.0, 0.0, 0.0])
        entity = NumpyArrayEntity(weights)
        old_gene_weights = [g.weight for g in entity.genes]

        entity.update_weights(np.array([1.0, 2.0, 3.0]))
        new_gene_weights = [g.weight for g in entity.genes]

        # Genes should reflect new normalized weights (distinct from old)
        # Old were all 0 (uniform), new are valid [0,1]
        assert len(entity.genes) == 3
        assert all(0.0 <= g.weight <= 1.0 for g in entity.genes)

    def test_update_weights_increments_age(self):
        entity = NumpyArrayEntity(np.array([0.1, 0.2]))
        entity.update_weights(np.array([0.3, 0.4]))
        assert entity.age == 1

    def test_update_weights_with_loss(self):
        entity = NumpyArrayEntity(np.array([0.1, 0.5, 0.9]))
        entity.update_weights(np.array([0.2, 0.4, 0.8]), loss=0.5)
        assert entity.fitness is not None
        assert 0.0 <= entity.fitness <= 1.0
        assert "loss" in entity.metadata
        assert "loss_history" in entity.metadata
        assert entity.metadata["loss_history"] == [0.5]

    def test_update_weights_loss_history_accumulates(self):
        entity = NumpyArrayEntity(np.array([0.1, 0.5]))
        for loss in [1.0, 0.8, 0.5, 0.3]:
            entity.update_weights(np.random.rand(2), loss=loss)
        assert entity.metadata["loss_history"] == [1.0, 0.8, 0.5, 0.3]

    def test_raw_weights_returns_copy(self):
        weights = np.array([0.1, 0.2, 0.3])
        entity = NumpyArrayEntity(weights)
        raw = entity.raw_weights
        raw[0] = 999.0
        assert entity._weights[0] != 999.0

    def test_weight_history_returns_copy(self):
        entity = NumpyArrayEntity(np.array([0.5, 0.5]))
        history = entity.weight_history
        assert isinstance(history, list)

    def test_get_active_genes(self):
        entity = NumpyArrayEntity(np.array([0.1, 0.5, 0.9]))
        active = entity.get_active_genes()
        assert len(active) == len(entity.genes)  # all active by default


# ─── RLAgentEntity ───────────────────────────────────────────────────────────


class TestRLAgentEntity:
    def test_creation(self):
        policy = np.array([0.2, 0.4, 0.6, 0.8])
        agent = RLAgentEntity(policy, name="test_agent")
        assert agent.name == "test_agent"
        assert len(agent.genes) == 4
        assert agent.fitness is None
        assert agent.age == 0
        assert agent.metadata["episode"] == 0

    def test_state_clipped(self):
        policy = np.array([-1.0, 0.5, 2.0])
        agent = RLAgentEntity(policy)
        state = agent.state
        assert state.min() >= 0.0
        assert state.max() <= 1.0

    def test_get_state_vector(self):
        policy = np.array([0.3, 0.6])
        agent = RLAgentEntity(policy)
        sv = agent.get_state_vector()
        assert np.allclose(sv, agent.state)

    def test_record_episode_updates_fitness(self):
        agent = RLAgentEntity(np.array([0.5, 0.5]))
        assert agent.fitness is None
        agent.record_episode(1.0)
        assert agent.fitness is not None
        assert 0.0 <= agent.fitness <= 1.0

    def test_record_episode_increments_episode_and_age(self):
        agent = RLAgentEntity(np.array([0.5, 0.5]))
        agent.record_episode(1.0)
        assert agent.metadata["episode"] == 1
        assert agent.age == 1

    def test_record_episode_accumulates_history(self):
        agent = RLAgentEntity(np.array([0.5, 0.5]))
        for r in [1.0, 2.0, 3.0]:
            agent.record_episode(r)
        assert agent.metadata["reward_history"] == [1.0, 2.0, 3.0]

    def test_record_episode_with_policy_update(self):
        policy = np.array([0.1, 0.1])
        agent = RLAgentEntity(policy)
        new_policy = np.array([0.9, 0.9])
        agent.record_episode(5.0, policy_update=new_policy)
        # Policy should have updated
        assert np.allclose(agent._policy, new_policy)

    def test_cumulative_reward(self):
        agent = RLAgentEntity(np.array([0.5]))
        for r in [1.0, 2.0, 3.0]:
            agent.record_episode(r)
        assert agent.cumulative_reward == pytest.approx(6.0)

    def test_recent_avg_reward_empty(self):
        agent = RLAgentEntity(np.array([0.5]))
        assert agent.recent_avg_reward == 0.0

    def test_recent_avg_reward(self):
        agent = RLAgentEntity(np.array([0.5]))
        for r in [2.0, 4.0]:
            agent.record_episode(r)
        assert agent.recent_avg_reward == pytest.approx(3.0)

    def test_get_active_genes(self):
        agent = RLAgentEntity(np.array([0.3, 0.6, 0.9]))
        active = agent.get_active_genes()
        assert len(active) == 3


# ─── create_training_observer ────────────────────────────────────────────────


class TestCreateTrainingObserver:
    def test_yields_metrics(self):
        entity = NumpyArrayEntity(np.array([0.1, 0.5, 0.9]))
        telos = create_semantic_coherence_telos()
        observer = create_training_observer(entity, telos)
        metrics = next(observer)
        assert hasattr(metrics, "actualization")
        assert hasattr(metrics, "overall_health")
        assert 0.0 <= metrics.actualization <= 1.0

    def test_yields_multiple_times(self):
        entity = NumpyArrayEntity(np.array([0.1, 0.5, 0.9]))
        telos = create_semantic_coherence_telos()
        observer = create_training_observer(entity, telos)
        for _ in range(5):
            metrics = next(observer)
            assert metrics is not None

    def test_metrics_update_after_weight_change(self):
        entity = NumpyArrayEntity(np.zeros(10))
        telos = create_semantic_coherence_telos()
        observer = create_training_observer(entity, telos)

        m1 = next(observer)
        entity.update_weights(np.ones(10) * 0.8, loss=0.1)
        m2 = next(observer)

        # Both should be valid metrics
        assert 0.0 <= m1.actualization <= 1.0
        assert 0.0 <= m2.actualization <= 1.0

    def test_with_custom_tracker(self):
        from ontogentelechy.core import ActualizationTracker

        entity = NumpyArrayEntity(np.array([0.3, 0.7]))
        telos = create_semantic_coherence_telos()
        tracker = ActualizationTracker()
        observer = create_training_observer(entity, telos, tracker=tracker)
        metrics = next(observer)
        assert metrics is not None


# ─── Integration: NumpyArrayEntity with ActualizationTracker ─────────────────


class TestNumpyArrayEntityIntegration:
    def test_compute_metrics_returns_valid(self):
        entity = NumpyArrayEntity(np.array([0.3, 0.5, 0.7, 0.4, 0.6]))
        telos = create_semantic_coherence_telos()
        tracker = ActualizationTracker()
        metrics = tracker.compute_metrics(entity, telos)

        assert 0.0 <= metrics.actualization <= 1.0
        assert 0.0 <= metrics.potentiality <= 1.0
        assert 0.0 <= metrics.emergence <= 1.0
        assert 0.0 <= metrics.overall_health <= 1.0

    def test_compute_metrics_with_loss_history(self):
        entity = NumpyArrayEntity(np.array([0.2, 0.4, 0.6]))
        entity.metadata["loss_history"] = [1.0, 0.8, 0.6, 0.4, 0.2]
        entity.fitness = 0.8
        telos = create_semantic_coherence_telos()
        tracker = ActualizationTracker()
        metrics = tracker.compute_metrics(entity, telos)
        assert metrics is not None
        assert 0.0 <= metrics.overall_health <= 1.0
