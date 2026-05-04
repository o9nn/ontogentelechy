"""
LLM Teleology — Purpose-driven development for language models.

Provides Telos presets targeting common fine-tuning goals:
- Instruction-following fidelity
- Factual coherence
- Stylistic consistency
- Safety alignment
"""

from typing import Any, List

import numpy as np

from .core import Criterion, Telos


def create_instruction_following_telos() -> Telos:
    """Telos: maximize instruction-following fidelity.

    Evaluates via loss_history trend and fitness score.
    """

    def evaluate_loss_trend(entity: Any) -> float:
        history = entity.metadata.get("loss_history", [])
        if len(history) < 2:
            return 0.3
        improvements = sum(1 for i in range(1, len(history)) if history[i] < history[i - 1])
        return improvements / (len(history) - 1)

    def evaluate_convergence(entity: Any) -> float:
        history = entity.metadata.get("loss_history", [])
        if not history:
            return 0.3
        last_loss = history[-1]
        return float(np.exp(-abs(last_loss)))

    def evaluate_gradient_stability(entity: Any) -> float:
        if hasattr(entity, "genes") and entity.genes:
            weights = [g.weight for g in entity.genes]
            variance = float(np.var(weights))
            return 1.0 - min(variance, 1.0)
        return 0.5

    return Telos(
        name="instruction_following",
        description="Maximize instruction-following fidelity through stable loss minimization",
        actualization_criteria=[
            Criterion("loss_trend", "Loss improvement trend", 0.4, evaluate_loss_trend, 1.0),
            Criterion("convergence", "Convergence toward low loss", 0.4, evaluate_convergence, 1.0),
            Criterion("stability", "Gradient stability", 0.2, evaluate_gradient_stability, 1.0),
        ],
        attractor_state={"loss": 0.0, "stability": 0.9, "convergence": 1.0},
    )


def create_factual_coherence_telos() -> Telos:
    """Telos: maintain factual coherence and consistency."""

    def evaluate_weight_coherence(entity: Any) -> float:
        if hasattr(entity, "genes") and entity.genes:
            weights = [g.weight for g in entity.genes]
            if len(weights) > 1:
                variance = float(np.var(weights))
                return 1.0 - min(variance, 1.0)
        return 0.5

    def evaluate_representation_stability(entity: Any) -> float:
        if hasattr(entity, "weight_history") and len(entity.weight_history) > 1:
            history = entity.weight_history
            recent = history[-min(10, len(history)) :]
            if len(recent) < 2:
                return 0.5
            changes = [
                float(np.linalg.norm(recent[i] - recent[i - 1])) for i in range(1, len(recent))
            ]
            avg_change = np.mean(changes)
            return float(np.exp(-avg_change))
        return 0.5

    def evaluate_fitness_score(entity: Any) -> float:
        if hasattr(entity, "fitness") and entity.fitness is not None:
            return float(entity.fitness)
        return 0.5

    return Telos(
        name="factual_coherence",
        description="Maintain factual coherence and stable knowledge representations",
        actualization_criteria=[
            Criterion(
                "weight_coherence", "Weight space coherence", 0.3, evaluate_weight_coherence, 1.0
            ),
            Criterion(
                "representation_stability",
                "Representation stability",
                0.4,
                evaluate_representation_stability,
                1.0,
            ),
            Criterion("fitness", "Overall fitness", 0.3, evaluate_fitness_score, 1.0),
        ],
        attractor_state={"coherence": 0.9, "stability": 0.8},
    )


def create_stylistic_consistency_telos() -> Telos:
    """Telos: maintain consistent stylistic patterns."""

    def evaluate_consistency(entity: Any) -> float:
        if hasattr(entity, "genes") and entity.genes:
            weights = [g.weight for g in entity.genes]
            if weights:
                mean_w = float(np.mean(weights))
                std_w = float(np.std(weights))
                # High mean, low std = consistent strong activations
                return mean_w * (1.0 - min(std_w, 1.0))
        return 0.5

    def evaluate_style_fitness(entity: Any) -> float:
        if hasattr(entity, "fitness") and entity.fitness is not None:
            return float(entity.fitness)
        return 0.5

    def evaluate_regularity(entity: Any) -> float:
        if hasattr(entity, "genes") and len(entity.genes) > 4:
            weights = [g.weight for g in entity.genes]
            arr = np.array(weights)
            if len(arr) > 2:
                diffs = np.diff(arr)
                variance_of_diffs = float(np.var(diffs))
                return 1.0 - min(variance_of_diffs, 1.0)
        return 0.5

    return Telos(
        name="stylistic_consistency",
        description="Maintain consistent stylistic patterns in generated content",
        actualization_criteria=[
            Criterion("consistency", "Activation consistency", 0.4, evaluate_consistency, 1.0),
            Criterion("style_fitness", "Style fitness", 0.3, evaluate_style_fitness, 1.0),
            Criterion("regularity", "Pattern regularity", 0.3, evaluate_regularity, 1.0),
        ],
        attractor_state={"consistency": 0.85, "regularity": 0.8},
    )


def create_safety_alignment_telos() -> Telos:
    """Telos: achieve safety alignment — avoid degenerate/harmful patterns."""

    def evaluate_weight_boundedness(entity: Any) -> float:
        if hasattr(entity, "genes") and entity.genes:
            weights = [g.weight for g in entity.genes]
            in_range = sum(1 for w in weights if 0.1 <= w <= 0.9)
            return in_range / len(weights)
        return 0.5

    def evaluate_no_extremes(entity: Any) -> float:
        if hasattr(entity, "genes") and entity.genes:
            weights = [g.weight for g in entity.genes]
            extremes = sum(1 for w in weights if w < 0.05 or w > 0.95)
            return 1.0 - (extremes / len(weights))
        return 0.5

    def evaluate_alignment_fitness(entity: Any) -> float:
        if hasattr(entity, "fitness") and entity.fitness is not None:
            return float(entity.fitness)
        return 0.5

    return Telos(
        name="safety_alignment",
        description="Achieve safety alignment by avoiding degenerate activation patterns",
        actualization_criteria=[
            Criterion("boundedness", "Weight boundedness", 0.35, evaluate_weight_boundedness, 1.0),
            Criterion("no_extremes", "Avoidance of extremes", 0.35, evaluate_no_extremes, 1.0),
            Criterion(
                "alignment_fitness", "Alignment fitness", 0.3, evaluate_alignment_fitness, 1.0
            ),
        ],
        attractor_state={"boundedness": 1.0, "safety": 0.95},
    )


LLM_TELOI = {
    "instruction_following": create_instruction_following_telos,
    "factual_coherence": create_factual_coherence_telos,
    "stylistic_consistency": create_stylistic_consistency_telos,
    "safety_alignment": create_safety_alignment_telos,
}


def create_llm_telos(name: str) -> Telos:
    if name not in LLM_TELOI:
        raise KeyError(f"Unknown LLM telos '{name}'. Available: {list(LLM_TELOI.keys())}")
    return LLM_TELOI[name]()


def list_llm_teloi() -> List[str]:
    return list(LLM_TELOI.keys())
