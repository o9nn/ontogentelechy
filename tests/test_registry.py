"""Tests for the Telos registry."""

import pytest

from ontogentelechy.registry import TelosRegistry, registry, RegistryEntry
from ontogentelechy.core import Telos, Criterion


def _make_telos(name: str = "test") -> Telos:
    return Telos(
        name=name,
        description=f"Test telos: {name}",
        actualization_criteria=[
            Criterion("c1", "Criterion 1", 1.0, lambda e: 0.5, 1.0),
        ],
        attractor_state={'weights': [0.5]},
    )


class TestTelosRegistry:
    def test_register_and_get(self):
        reg = TelosRegistry()
        reg.register("my_telos", lambda: _make_telos("my_telos"))
        telos = reg.get("my_telos")
        assert isinstance(telos, Telos)
        assert telos.name == "my_telos"

    def test_get_unknown_raises(self):
        reg = TelosRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_names(self):
        reg = TelosRegistry()
        reg.register("a", lambda: _make_telos("a"))
        reg.register("b", lambda: _make_telos("b"))
        names = reg.list_names()
        assert "a" in names
        assert "b" in names

    def test_list_by_tag(self):
        reg = TelosRegistry()
        reg.register("tagged", lambda: _make_telos("tagged"), tags=["cognitive"])
        reg.register("other", lambda: _make_telos("other"), tags=["efficiency"])
        assert "tagged" in reg.list_by_tag("cognitive")
        assert "other" not in reg.list_by_tag("cognitive")

    def test_describe(self):
        reg = TelosRegistry()
        reg.register("desc_telos", lambda: _make_telos(), description="A test telos", tags=["test"])
        desc = reg.describe("desc_telos")
        assert "desc_telos" in desc
        assert "test" in desc

    def test_describe_unknown_raises(self):
        reg = TelosRegistry()
        with pytest.raises(KeyError):
            reg.describe("missing")

    def test_contains(self):
        reg = TelosRegistry()
        reg.register("present", lambda: _make_telos())
        assert "present" in reg
        assert "absent" not in reg

    def test_len(self):
        reg = TelosRegistry()
        assert len(reg) == 0
        reg.register("a", lambda: _make_telos())
        reg.register("b", lambda: _make_telos())
        assert len(reg) == 2

    def test_register_decorator(self):
        reg = TelosRegistry()

        @reg.register_decorator(name="decorated", description="Decorated telos", tags=["demo"])
        def my_factory():
            return _make_telos("decorated")

        assert "decorated" in reg
        telos = reg.get("decorated")
        assert telos.name == "decorated"

    def test_register_decorator_uses_function_name(self):
        reg = TelosRegistry()

        @reg.register_decorator()
        def my_auto_named_telos():
            return _make_telos()

        assert "my_auto_named_telos" in reg


class TestDefaultRegistry:
    def test_default_registry_populated(self):
        assert len(registry) >= 5

    def test_default_registry_has_example_teloi(self):
        assert "semantic_coherence" in registry
        assert "adaptive_learning" in registry
        assert "complexity_emergence" in registry
        assert "efficient_computation" in registry
        assert "knowledge_integration" in registry

    def test_default_registry_get_returns_telos(self):
        telos = registry.get("semantic_coherence")
        assert isinstance(telos, Telos)

    def test_default_registry_tags(self):
        cognitive = registry.list_by_tag("cognitive")
        assert len(cognitive) >= 1
