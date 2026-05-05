"""Tests for ontogentelechy/llm_teloi.py (Phase 4)."""

import numpy as np
import pytest

from ontogentelechy.adapters import NumpyArrayEntity
from ontogentelechy.llm_teloi import (
    LLM_TELOI,
    create_factual_coherence_telos,
    create_instruction_following_telos,
    create_llm_telos,
    create_safety_alignment_telos,
    create_stylistic_consistency_telos,
    list_llm_teloi,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def make_entity(n: int = 10, fitness: float = 0.7) -> NumpyArrayEntity:
    weights = np.random.uniform(0.2, 0.8, n)
    entity = NumpyArrayEntity(weights)
    entity.fitness = fitness
    entity.metadata["loss_history"] = [1.0, 0.8, 0.6, 0.4, 0.2]
    return entity


# ─── Individual Telos Factories ───────────────────────────────────────────────


class TestInstructionFollowingTelos:
    def test_returns_telos(self):
        from ontogentelechy.core import Telos

        t = create_instruction_following_telos()
        assert isinstance(t, Telos)

    def test_correct_name(self):
        t = create_instruction_following_telos()
        assert t.name == "instruction_following"

    def test_has_three_criteria(self):
        t = create_instruction_following_telos()
        assert len(t.actualization_criteria) == 3

    def test_criteria_weights_sum_to_one(self):
        t = create_instruction_following_telos()
        total = sum(c.weight for c in t.actualization_criteria)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_evaluates_on_entity(self):
        t = create_instruction_following_telos()
        entity = make_entity()
        for criterion in t.actualization_criteria:
            score = criterion.evaluate(entity)
            assert 0.0 <= score <= 1.0

    def test_attractor_state(self):
        t = create_instruction_following_telos()
        assert isinstance(t.attractor_state, dict)
        assert "loss" in t.attractor_state


class TestFactualCoherenceTelos:
    def test_returns_telos(self):
        from ontogentelechy.core import Telos

        t = create_factual_coherence_telos()
        assert isinstance(t, Telos)

    def test_correct_name(self):
        t = create_factual_coherence_telos()
        assert t.name == "factual_coherence"

    def test_has_three_criteria(self):
        t = create_factual_coherence_telos()
        assert len(t.actualization_criteria) == 3

    def test_criteria_weights_sum_to_one(self):
        t = create_factual_coherence_telos()
        total = sum(c.weight for c in t.actualization_criteria)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_evaluates_on_entity(self):
        t = create_factual_coherence_telos()
        entity = make_entity()
        # Add weight history for representation_stability criterion
        entity.update_weights(np.random.rand(10))
        entity.update_weights(np.random.rand(10))
        for criterion in t.actualization_criteria:
            score = criterion.evaluate(entity)
            assert 0.0 <= score <= 1.0

    def test_evaluates_without_weight_history(self):
        t = create_factual_coherence_telos()
        entity = make_entity()
        for criterion in t.actualization_criteria:
            score = criterion.evaluate(entity)
            assert 0.0 <= score <= 1.0


class TestStylisticConsistencyTelos:
    def test_returns_telos(self):
        from ontogentelechy.core import Telos

        t = create_stylistic_consistency_telos()
        assert isinstance(t, Telos)

    def test_correct_name(self):
        t = create_stylistic_consistency_telos()
        assert t.name == "stylistic_consistency"

    def test_has_three_criteria(self):
        t = create_stylistic_consistency_telos()
        assert len(t.actualization_criteria) == 3

    def test_criteria_weights_sum_to_one(self):
        t = create_stylistic_consistency_telos()
        total = sum(c.weight for c in t.actualization_criteria)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_evaluates_on_entity(self):
        t = create_stylistic_consistency_telos()
        entity = make_entity()
        for criterion in t.actualization_criteria:
            score = criterion.evaluate(entity)
            assert 0.0 <= score <= 1.0


class TestSafetyAlignmentTelos:
    def test_returns_telos(self):
        from ontogentelechy.core import Telos

        t = create_safety_alignment_telos()
        assert isinstance(t, Telos)

    def test_correct_name(self):
        t = create_safety_alignment_telos()
        assert t.name == "safety_alignment"

    def test_has_three_criteria(self):
        t = create_safety_alignment_telos()
        assert len(t.actualization_criteria) == 3

    def test_criteria_weights_sum_to_one(self):
        t = create_safety_alignment_telos()
        total = sum(c.weight for c in t.actualization_criteria)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_evaluates_on_entity(self):
        t = create_safety_alignment_telos()
        entity = make_entity()
        for criterion in t.actualization_criteria:
            score = criterion.evaluate(entity)
            assert 0.0 <= score <= 1.0

    def test_extreme_weights_penalized(self):
        t = create_safety_alignment_telos()
        # Entity with extreme weights (near 0 and 1)
        extreme_weights = np.array([0.01, 0.99, 0.01, 0.99])
        entity = NumpyArrayEntity(extreme_weights)
        # Force gene weights to be extreme
        from ontogentelechy.entity import SimpleGene

        entity.genes = [
            SimpleGene(weight=0.01),
            SimpleGene(weight=0.99),
            SimpleGene(weight=0.01),
            SimpleGene(weight=0.99),
        ]
        # no_extremes criterion should give low score
        no_extremes_crit = next(c for c in t.actualization_criteria if c.name == "no_extremes")
        score = no_extremes_crit.evaluate(entity)
        assert score < 0.5  # penalized for extremes

    def test_moderate_weights_rewarded(self):
        t = create_safety_alignment_telos()
        # Entity with moderate weights
        moderate_weights = np.array([0.3, 0.5, 0.7, 0.4])
        entity = NumpyArrayEntity(moderate_weights)
        from ontogentelechy.entity import SimpleGene

        entity.genes = [SimpleGene(weight=w) for w in moderate_weights]
        no_extremes_crit = next(c for c in t.actualization_criteria if c.name == "no_extremes")
        score = no_extremes_crit.evaluate(entity)
        assert score == pytest.approx(1.0)


# ─── create_llm_telos ─────────────────────────────────────────────────────────


class TestCreateLlmTelos:
    def test_creates_by_name(self):
        from ontogentelechy.core import Telos

        for name in LLM_TELOI:
            t = create_llm_telos(name)
            assert isinstance(t, Telos)

    def test_raises_for_unknown(self):
        with pytest.raises(KeyError, match="Unknown LLM telos"):
            create_llm_telos("nonexistent_telos")

    def test_instruction_following_by_name(self):
        t = create_llm_telos("instruction_following")
        assert t.name == "instruction_following"

    def test_factual_coherence_by_name(self):
        t = create_llm_telos("factual_coherence")
        assert t.name == "factual_coherence"

    def test_stylistic_consistency_by_name(self):
        t = create_llm_telos("stylistic_consistency")
        assert t.name == "stylistic_consistency"

    def test_safety_alignment_by_name(self):
        t = create_llm_telos("safety_alignment")
        assert t.name == "safety_alignment"


# ─── list_llm_teloi ───────────────────────────────────────────────────────────


class TestListLlmTeloi:
    def test_returns_four_names(self):
        names = list_llm_teloi()
        assert len(names) == 4

    def test_contains_expected_names(self):
        names = list_llm_teloi()
        expected = [
            "instruction_following",
            "factual_coherence",
            "stylistic_consistency",
            "safety_alignment",
        ]
        for name in expected:
            assert name in names

    def test_returns_list(self):
        assert isinstance(list_llm_teloi(), list)


# ─── Each telos evaluates without error on mock NumpyArrayEntity ──────────────


class TestAllTeloiEvaluate:
    @pytest.mark.parametrize(
        "name",
        ["instruction_following", "factual_coherence", "stylistic_consistency", "safety_alignment"],
    )
    def test_evaluate_on_mock_entity(self, name: str):
        from ontogentelechy.core import ActualizationTracker

        telos = create_llm_telos(name)
        entity = make_entity(n=20)

        tracker = ActualizationTracker()
        metrics = tracker.compute_metrics(entity, telos)

        assert 0.0 <= metrics.actualization <= 1.0
        assert 0.0 <= metrics.overall_health <= 1.0

    @pytest.mark.parametrize(
        "name",
        ["instruction_following", "factual_coherence", "stylistic_consistency", "safety_alignment"],
    )
    def test_evaluate_on_empty_entity(self, name: str):
        """Teloi should degrade gracefully on minimal entities."""
        from ontogentelechy.core import ActualizationTracker
        from ontogentelechy.entity import SimpleEntity

        telos = create_llm_telos(name)
        entity = SimpleEntity(state=np.array([0.5]), genes=[])
        entity.metadata = {}

        tracker = ActualizationTracker()
        metrics = tracker.compute_metrics(entity, telos)
        assert metrics is not None
