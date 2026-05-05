"""Tests for ontogentelechy.core module."""

import pytest

from ontogentelechy.core import (
    ActualizationMetrics,
    ActualizationPhase,
    ActualizationTracker,
    Criterion,
    DevelopmentalAttractor,
    Telos,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleEntity:
    def __init__(self, genes=None, fitness=0.5):
        self.genes = genes or []
        self.fitness = fitness
        self.metadata = {}


class GeneStub:
    def __init__(self, weight=0.5, active=True):
        self.weight = weight
        self.active = active


# ---------------------------------------------------------------------------
# ActualizationPhase
# ---------------------------------------------------------------------------


class TestActualizationPhase:
    def test_values(self):
        assert ActualizationPhase.POTENTIAL.value == "potential"
        assert ActualizationPhase.EMERGENT.value == "emergent"
        assert ActualizationPhase.DEVELOPING.value == "developing"
        assert ActualizationPhase.ACTUALIZING.value == "actualizing"
        assert ActualizationPhase.ACTUALIZED.value == "actualized"

    def test_all_phases_present(self):
        phases = list(ActualizationPhase)
        assert len(phases) == 5


# ---------------------------------------------------------------------------
# Criterion
# ---------------------------------------------------------------------------


class TestCriterion:
    def test_creation(self):
        c = Criterion("test", "A test criterion", 0.5, lambda e: 0.7)
        assert c.name == "test"
        assert c.weight == 0.5

    def test_evaluate_normal(self):
        c = Criterion("c", "d", 0.5, lambda e: 0.8)
        assert c.evaluate(object()) == pytest.approx(0.8)

    def test_evaluate_clamps_above_one(self):
        c = Criterion("c", "d", 0.5, lambda e: 5.0)
        assert c.evaluate(object()) == pytest.approx(1.0)

    def test_evaluate_clamps_below_zero(self):
        c = Criterion("c", "d", 0.5, lambda e: -1.0)
        assert c.evaluate(object()) == pytest.approx(0.0)

    def test_evaluate_raising_callable_returns_zero(self):
        def bad(e):
            raise ValueError("boom")

        c = Criterion("c", "d", 0.5, bad)
        assert c.evaluate(object()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Telos
# ---------------------------------------------------------------------------


class TestTelos:
    def _make_telos(self, scores=None):
        scores = scores or [0.5, 0.5, 0.5]
        criteria = [
            Criterion(f"c{i}", f"desc{i}", 1 / len(scores), lambda e, s=s: s)
            for i, s in enumerate(scores)
        ]
        return Telos(
            name="test_telos",
            description="a telos",
            actualization_criteria=criteria,
            attractor_state={"weights": [0.8] * 3},
        )

    def test_creation(self):
        t = self._make_telos()
        assert t.name == "test_telos"
        assert len(t.actualization_criteria) == 3

    def test_evaluate_actualization_returns_float(self):
        t = self._make_telos([0.5, 0.5, 0.5])
        score = t.evaluate_actualization(object())
        assert 0.0 <= score <= 1.0

    def test_phase_transition_at_boundaries(self):
        t = self._make_telos()

        # Force actualization values and check phase
        t.current_actualization = 0.1
        t._update_phase()
        assert t.phase == ActualizationPhase.POTENTIAL

        t.current_actualization = 0.25
        t._update_phase()
        assert t.phase == ActualizationPhase.EMERGENT

        t.current_actualization = 0.45
        t._update_phase()
        assert t.phase == ActualizationPhase.DEVELOPING

        t.current_actualization = 0.65
        t._update_phase()
        assert t.phase == ActualizationPhase.ACTUALIZING

        t.current_actualization = 0.85
        t._update_phase()
        assert t.phase == ActualizationPhase.ACTUALIZED

    def test_get_phase_description_returns_string(self):
        t = self._make_telos()
        desc = t.get_phase_description()
        assert isinstance(desc, str)
        assert len(desc) > 0


# ---------------------------------------------------------------------------
# ActualizationMetrics
# ---------------------------------------------------------------------------


class TestActualizationMetrics:
    def test_defaults(self):
        m = ActualizationMetrics()
        assert m.potentiality == 0.0
        assert m.emergence == 0.0
        assert m.integration == 0.0
        assert m.actualization == 0.0
        assert m.telos_alignment == 0.0

    def test_overall_health_weighted(self):
        m = ActualizationMetrics(
            potentiality=1.0,
            emergence=1.0,
            integration=1.0,
            actualization=1.0,
            telos_alignment=1.0,
        )
        # weights: 0.1 + 0.2 + 0.3 + 0.3 + 0.1 = 1.0
        assert m.overall_health == pytest.approx(1.0)

    def test_overall_health_partial(self):
        m = ActualizationMetrics(actualization=1.0)
        assert m.overall_health == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# ActualizationTracker
# ---------------------------------------------------------------------------


class TestActualizationTracker:
    def test_measure_potentiality_no_genes(self):
        tracker = ActualizationTracker()
        entity = SimpleEntity()
        p = tracker.measure_potentiality(entity)
        assert 0.0 <= p <= 1.0

    def test_measure_potentiality_with_ontogenetic_state(self):
        class State:
            maturity = 0.6

        entity = SimpleEntity()
        entity.ontogenetic_state = State()
        tracker = ActualizationTracker()
        p = tracker.measure_potentiality(entity)
        assert p == pytest.approx(0.4)

    def test_measure_emergence_no_history(self):
        tracker = ActualizationTracker()
        entity = SimpleEntity()
        e = tracker.measure_emergence(entity)
        assert 0.0 <= e <= 1.0

    def test_measure_emergence_with_history(self):
        class State:
            development_history = [
                {"type": "stage_transition"},
                {"type": "new_capability"},
                {"type": "other"},
            ]

        class Entity:
            ontogenetic_state = State()

        tracker = ActualizationTracker()
        e = tracker.measure_emergence(Entity())
        assert e == pytest.approx(2 / 10.0)

    def test_measure_integration_with_genes(self):
        genes = [GeneStub(0.5), GeneStub(0.5), GeneStub(0.5)]
        entity = SimpleEntity(genes=genes)
        tracker = ActualizationTracker()
        i = tracker.measure_integration(entity)
        assert i == pytest.approx(1.0)  # zero variance → full integration

    def test_measure_integration_with_to_atomspace_structure(self):
        class Atom:
            def __init__(self, t):
                self.atom_type = t

            def is_link(self):
                return False

        class Entity:
            def to_atomspace_structure(self):
                return [Atom("A"), Atom("A"), Atom("B")]

        tracker = ActualizationTracker()
        val = tracker.measure_integration(Entity())
        assert 0.0 <= val <= 1.0

    def test_measure_actualization_with_telos(self):
        tracker = ActualizationTracker()
        telos = Telos(
            name="t",
            description="d",
            actualization_criteria=[Criterion("c", "d", 1.0, lambda e: 0.6)],
            attractor_state={},
        )
        val = tracker.measure_actualization(object(), telos)
        assert 0.0 <= val <= 1.0

    def test_measure_actualization_without_telos(self):
        tracker = ActualizationTracker()
        entity = SimpleEntity(fitness=0.7)
        val = tracker.measure_actualization(entity, None)
        assert val == pytest.approx(0.7)

    def test_compute_metrics_stores_history(self):
        tracker = ActualizationTracker()
        entity = SimpleEntity()
        tracker.compute_metrics(entity)
        assert len(tracker.history) == 1
        tracker.compute_metrics(entity)
        assert len(tracker.history) == 2

    def test_detect_phase_transition_returns_none_with_less_than_two(self):
        tracker = ActualizationTracker()
        assert tracker.detect_phase_transition() is None
        tracker.history.append(ActualizationMetrics(actualization=0.1))
        assert tracker.detect_phase_transition() is None

    def test_detect_phase_transition_returns_transition_on_large_jump(self):
        tracker = ActualizationTracker()
        tracker.history.append(ActualizationMetrics(actualization=0.1))
        tracker.history.append(ActualizationMetrics(actualization=0.4))
        result = tracker.detect_phase_transition()
        assert result is not None
        assert result["actualization_jump"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# DevelopmentalAttractor
# ---------------------------------------------------------------------------


class TestDevelopmentalAttractor:
    def _make_attractor(self, weights=None):
        weights = weights or [0.8, 0.8, 0.8]
        telos = Telos(
            name="t",
            description="d",
            actualization_criteria=[],
            attractor_state={"weights": weights},
        )
        return DevelopmentalAttractor(telos=telos)

    def test_compute_gradient_toward_target(self):
        import numpy as np

        attractor = self._make_attractor([0.8, 0.8, 0.8])
        current = np.array([0.0, 0.0, 0.0])
        grad = attractor.compute_gradient(current)
        assert len(grad) == 3
        assert all(g > 0 for g in grad)

    def test_compute_gradient_returns_zeros_at_target(self):
        import numpy as np

        attractor = self._make_attractor([0.8, 0.8, 0.8])
        current = np.array([0.8, 0.8, 0.8])
        grad = attractor.compute_gradient(current)
        assert all(g == pytest.approx(0.0) for g in grad)

    def test_apply_pull_modifies_gene_weights(self):
        attractor = self._make_attractor([0.9, 0.9, 0.9])
        entity = SimpleEntity(genes=[GeneStub(0.0), GeneStub(0.0), GeneStub(0.0)])
        attractor.apply_pull(entity)
        # Weights should have moved toward 0.9
        assert any(g.weight > 0.0 for g in entity.genes)

    def test_apply_pull_clips_to_0_1(self):
        attractor = self._make_attractor([1.0, 1.0, 1.0])
        entity = SimpleEntity(genes=[GeneStub(0.99), GeneStub(0.99), GeneStub(0.99)])
        attractor.apply_pull(entity, strength=100.0)
        for g in entity.genes:
            assert 0.0 <= g.weight <= 1.0

    def test_is_in_basin(self):
        attractor = self._make_attractor([0.8, 0.8, 0.8])
        entity = SimpleEntity(genes=[GeneStub(0.8), GeneStub(0.8), GeneStub(0.8)])
        assert attractor.is_in_basin(entity)
