"""Tests for ontogentelechy.attractor (Phase 2)."""

import numpy as np
import pytest

from ontogentelechy.attractor import AttractorLandscape, Basin, Repulsor
from ontogentelechy.core import ActualizationPhase, Criterion, Telos

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_telos() -> Telos:
    return Telos(
        name="test_telos",
        description="A telos for testing",
        actualization_criteria=[
            Criterion("c1", "crit1", 1.0, lambda e: 0.5),
        ],
        attractor_state={"a": 0.8, "b": 0.6},
    )


# ---------------------------------------------------------------------------
# Basin tests
# ---------------------------------------------------------------------------


class TestBasin:
    def test_distance_to_center(self):
        basin = Basin(state=np.array([0.5, 0.5]))
        assert basin.distance_to(np.array([0.5, 0.5])) == pytest.approx(0.0)

    def test_distance_to_point(self):
        basin = Basin(state=np.array([0.0, 0.0]))
        dist = basin.distance_to(np.array([3.0, 4.0]))
        assert dist == pytest.approx(5.0)

    def test_distance_handles_different_lengths(self):
        basin = Basin(state=np.array([0.0, 0.0, 0.0]))
        dist = basin.distance_to(np.array([1.0, 0.0]))
        assert dist == pytest.approx(1.0)

    def test_gradient_toward_center_is_zero(self):
        basin = Basin(state=np.array([0.5, 0.5]))
        grad = basin.gradient_toward(np.array([0.5, 0.5]))
        assert np.allclose(grad, 0.0)

    def test_gradient_toward_points_toward_center(self):
        basin = Basin(state=np.array([1.0, 1.0]), strength=1.0, radius=0.5)
        point = np.array([0.0, 0.0])
        grad = basin.gradient_toward(point)
        # Gradient should point in positive direction (toward [1,1])
        assert grad[0] > 0
        assert grad[1] > 0

    def test_gradient_toward_magnitude_nonzero_away_from_center(self):
        basin = Basin(state=np.array([1.0, 0.0]))
        grad = basin.gradient_toward(np.array([0.0, 0.0]))
        assert np.linalg.norm(grad) > 0


# ---------------------------------------------------------------------------
# Repulsor tests
# ---------------------------------------------------------------------------


class TestRepulsor:
    def test_gradient_away_at_center_has_nonzero_magnitude(self):
        rep = Repulsor(state=np.array([0.5, 0.5]), strength=1.0)
        # At exact center, random direction is returned
        grad = rep.gradient_away(np.array([0.5, 0.5]))
        assert np.linalg.norm(grad) > 0

    def test_gradient_away_points_away(self):
        rep = Repulsor(state=np.array([0.0, 0.0]), strength=1.0, radius=1.0)
        point = np.array([1.0, 0.0])
        grad = rep.gradient_away(point)
        # Should point away (positive x direction)
        assert grad[0] > 0

    def test_gradient_away_decays_with_distance(self):
        rep = Repulsor(state=np.array([0.0, 0.0]), strength=1.0, radius=0.5)
        near = rep.gradient_away(np.array([0.1, 0.0]))
        far = rep.gradient_away(np.array([5.0, 0.0]))
        assert np.linalg.norm(near) > np.linalg.norm(far)


# ---------------------------------------------------------------------------
# AttractorLandscape tests
# ---------------------------------------------------------------------------


class TestAttractorLandscape:
    def test_empty_landscape_returns_zero_gradient(self):
        landscape = AttractorLandscape()
        grad = landscape.net_gradient(np.array([0.5, 0.5]))
        assert np.allclose(grad, 0.0)

    def test_add_basin(self):
        landscape = AttractorLandscape()
        basin = Basin(state=np.array([0.5, 0.5]))
        landscape.add_basin(basin)
        assert len(landscape.basins) == 1

    def test_add_repulsor(self):
        landscape = AttractorLandscape()
        rep = Repulsor(state=np.array([0.0, 0.0]))
        landscape.add_repulsor(rep)
        assert len(landscape.repulsors) == 1

    def test_net_gradient_superposition(self):
        basin1 = Basin(state=np.array([1.0, 0.0]), strength=1.0)
        basin2 = Basin(state=np.array([-1.0, 0.0]), strength=1.0)
        landscape = AttractorLandscape(basins=[basin1, basin2])
        # At the midpoint the x-components should mostly cancel
        grad = landscape.net_gradient(np.array([0.0, 0.0]))
        assert abs(grad[0]) < 0.2  # near-cancellation

    def test_nearest_basin(self):
        b1 = Basin(state=np.array([0.0, 0.0]), label="left")
        b2 = Basin(state=np.array([10.0, 0.0]), label="right")
        landscape = AttractorLandscape(basins=[b1, b2])
        nearest = landscape.nearest_basin(np.array([1.0, 0.0]))
        assert nearest is b1

    def test_nearest_basin_empty_returns_none(self):
        landscape = AttractorLandscape()
        assert landscape.nearest_basin(np.array([0.0])) is None

    def test_is_in_any_basin(self):
        basin = Basin(state=np.array([0.5, 0.5]), radius=1.0)
        landscape = AttractorLandscape(basins=[basin])
        assert landscape.is_in_any_basin(np.array([0.5, 0.5]))
        assert not landscape.is_in_any_basin(np.array([10.0, 10.0]))

    def test_annealing_decays_strength(self):
        basin = Basin(state=np.array([0.5, 0.5]), strength=1.0)
        landscape = AttractorLandscape(basins=[basin], annealing_rate=0.5)
        initial = basin.strength
        landscape.step()
        assert basin.strength < initial

    def test_no_annealing_keeps_strength(self):
        basin = Basin(state=np.array([0.5, 0.5]), strength=1.0)
        landscape = AttractorLandscape(basins=[basin], annealing_rate=0.0)
        for _ in range(10):
            landscape.step()
        assert basin.strength == pytest.approx(1.0)

    def test_detect_saddle_points_requires_two_basins(self):
        landscape = AttractorLandscape(basins=[Basin(state=np.array([0.0, 0.0]))])
        result = landscape.detect_saddle_points()
        assert result == []

    def test_detect_saddle_points_returns_list(self):
        b1 = Basin(state=np.array([0.0, 0.0]), strength=1.0, radius=0.3)
        b2 = Basin(state=np.array([1.0, 0.0]), strength=1.0, radius=0.3)
        landscape = AttractorLandscape(basins=[b1, b2])
        result = landscape.detect_saddle_points(n_samples=200)
        assert isinstance(result, list)

    def test_from_telos_dict_attractor(self):
        telos = _simple_telos()
        landscape = AttractorLandscape.from_telos(telos)
        assert len(landscape.basins) == 1
        assert landscape.basins[0].label == "test_telos"
        assert len(landscape.basins[0].state) == 2  # two numeric values in dict

    def test_from_telos_array_attractor(self):
        telos = Telos(
            name="arr_telos",
            description="Array attractor",
            actualization_criteria=[],
            attractor_state=[0.3, 0.6, 0.9],
        )
        landscape = AttractorLandscape.from_telos(telos)
        assert len(landscape.basins) == 1
        assert len(landscape.basins[0].state) == 3

    def test_from_telos_with_annealing_rate(self):
        telos = _simple_telos()
        landscape = AttractorLandscape.from_telos(telos, annealing_rate=0.1)
        assert landscape.annealing_rate == pytest.approx(0.1)

    def test_net_gradient_with_repulsor(self):
        basin = Basin(state=np.array([1.0, 0.0]), strength=2.0)
        repulsor = Repulsor(state=np.array([-1.0, 0.0]), strength=1.0, radius=0.5)
        landscape = AttractorLandscape(basins=[basin], repulsors=[repulsor])
        # At origin, basin pulls right and repulsor pushes right → both positive x
        grad = landscape.net_gradient(np.array([0.0, 0.0]))
        assert grad[0] > 0
