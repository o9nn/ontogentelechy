"""Tests for ontogentelechy.examples module."""

import pytest

from ontogentelechy.core import Telos
from ontogentelechy.examples import (
    EXAMPLE_TELOI,
    create_adaptive_learning_telos,
    create_complexity_emergence_telos,
    create_efficient_computation_telos,
    create_knowledge_integration_telos,
    create_semantic_coherence_telos,
    get_example_telos,
    list_example_teloi,
)


class MockGene:
    def __init__(self, weight=0.5):
        self.weight = weight
        self.active = True


class MockEntity:
    def __init__(self):
        self.genes = [MockGene(0.5 + i * 0.05) for i in range(5)]
        self.fitness = 0.6
        self.metadata = {}
        self.age = 3


CREATORS = [
    (create_semantic_coherence_telos, "semantic_coherence", 3),
    (create_adaptive_learning_telos, "adaptive_learning", 3),
    (create_complexity_emergence_telos, "complexity_emergence", 3),
    (create_efficient_computation_telos, "efficient_computation", 3),
    (create_knowledge_integration_telos, "knowledge_integration", 3),
]


@pytest.mark.parametrize("creator,name,n_criteria", CREATORS)
class TestExampleTelos:
    def test_returns_telos_instance(self, creator, name, n_criteria):
        t = creator()
        assert isinstance(t, Telos)

    def test_correct_name(self, creator, name, n_criteria):
        t = creator()
        assert t.name == name

    def test_correct_number_of_criteria(self, creator, name, n_criteria):
        t = creator()
        assert len(t.actualization_criteria) == n_criteria

    def test_weights_sum_to_one(self, creator, name, n_criteria):
        t = creator()
        total = sum(c.weight for c in t.actualization_criteria)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_evaluate_actualization_does_not_crash(self, creator, name, n_criteria):
        t = creator()
        entity = MockEntity()
        score = t.evaluate_actualization(entity)
        assert 0.0 <= score <= 1.0


class TestGetExampleTelos:
    def test_returns_telos_by_name(self):
        t = get_example_telos("semantic_coherence")
        assert isinstance(t, Telos)
        assert t.name == "semantic_coherence"

    def test_raises_key_error_for_unknown(self):
        with pytest.raises(KeyError):
            get_example_telos("nonexistent_telos_xyz")


class TestListExampleTeloi:
    def test_returns_list_of_five(self):
        names = list_example_teloi()
        assert isinstance(names, list)
        assert len(names) == 5

    def test_contains_expected_names(self):
        names = list_example_teloi()
        assert "semantic_coherence" in names
        assert "adaptive_learning" in names
        assert "complexity_emergence" in names
        assert "efficient_computation" in names
        assert "knowledge_integration" in names
