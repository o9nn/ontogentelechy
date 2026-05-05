"""Tests for ontogentelechy.fitness module."""

import pytest

from ontogentelechy.core import ActualizationMetrics, Criterion, Telos
from ontogentelechy.fitness import MultiTelosFitness, TeleologicalFitness

# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockAtom:
    def __init__(self):
        self.atom_type = "ConceptNode"
        self.name = "concept"
        self.outgoing_set = []

    def is_node(self):
        return True

    def is_link(self):
        return False


class MockGene:
    def __init__(self, weight=0.5):
        self.weight = weight
        self.active = True
        self.atom = MockAtom()


class MockIndividual:
    def __init__(self, n_genes=5, fitness=0.5):
        self.genes = [MockGene(0.5 + i * 0.1) for i in range(n_genes)]
        self.fitness = fitness
        self.metadata = {}

    def get_active_genes(self):
        return [g for g in self.genes if g.active]


class MockAtomSpace:
    def get_atoms(self):
        return []

    def add_atom(self, atom):
        return atom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_telos(name="test_telos"):
    return Telos(
        name=name,
        description="A test telos",
        actualization_criteria=[
            Criterion("c1", "desc1", 0.5, lambda e: 0.6),
            Criterion("c2", "desc2", 0.5, lambda e: 0.4),
        ],
        attractor_state={"weights": [0.7, 0.7, 0.7, 0.7, 0.7]},
    )


# ---------------------------------------------------------------------------
# TeleologicalFitness
# ---------------------------------------------------------------------------


class TestTeleologicalFitness:
    def test_construction(self):
        tf = TeleologicalFitness(MockAtomSpace(), _make_telos())
        assert tf.telos is not None

    def test_evaluate_returns_float(self):
        tf = TeleologicalFitness(MockAtomSpace(), _make_telos())
        ind = MockIndividual()
        score = tf.evaluate(ind)
        assert isinstance(score, float)

    def test_evaluate_in_range(self):
        tf = TeleologicalFitness(MockAtomSpace(), _make_telos())
        ind = MockIndividual()
        score = tf.evaluate(ind)
        assert 0.0 <= score <= 1.0

    def test_evaluate_stores_fitness_components(self):
        tf = TeleologicalFitness(MockAtomSpace(), _make_telos())
        ind = MockIndividual()
        tf.evaluate(ind)
        assert "fitness_components" in ind.metadata
        components = ind.metadata["fitness_components"]
        for key in ("base", "telos", "actualization", "emergence"):
            assert key in components

    def test_evaluate_empty_genes_returns_zero(self):
        tf = TeleologicalFitness(MockAtomSpace(), _make_telos())
        ind = MockIndividual(n_genes=0)
        assert tf.evaluate(ind) == pytest.approx(0.0)

    def test_get_actualization_metrics(self):
        tf = TeleologicalFitness(MockAtomSpace(), _make_telos())
        ind = MockIndividual()
        metrics = tf.get_actualization_metrics(ind)
        assert isinstance(metrics, ActualizationMetrics)

    def test_get_phase_transitions_empty_initially(self):
        tf = TeleologicalFitness(MockAtomSpace(), _make_telos())
        assert tf.get_phase_transitions() == []


# ---------------------------------------------------------------------------
# MultiTelosFitness
# ---------------------------------------------------------------------------


class TestMultiTelosFitness:
    def test_construction_with_two_teloi(self):
        teloi = [(_make_telos("t1"), 0.6), (_make_telos("t2"), 0.4)]
        mf = MultiTelosFitness(MockAtomSpace(), teloi)
        assert len(mf.fitness_functions) == 2

    def test_weights_normalized(self):
        teloi = [(_make_telos("t1"), 2.0), (_make_telos("t2"), 3.0)]
        mf = MultiTelosFitness(MockAtomSpace(), teloi)
        assert sum(mf.weights) == pytest.approx(1.0)

    def test_evaluate_returns_float_in_range(self):
        teloi = [(_make_telos("t1"), 0.5), (_make_telos("t2"), 0.5)]
        mf = MultiTelosFitness(MockAtomSpace(), teloi)
        ind = MockIndividual()
        score = mf.evaluate(ind)
        assert 0.0 <= score <= 1.0

    def test_get_dominant_telos_returns_telos(self):
        t1 = _make_telos("t1")
        t2 = _make_telos("t2")
        mf = MultiTelosFitness(MockAtomSpace(), [(t1, 0.5), (t2, 0.5)])
        ind = MockIndividual()
        dominant, score = mf.get_dominant_telos(ind)
        assert isinstance(dominant, Telos)
        assert 0.0 <= score <= 1.0
