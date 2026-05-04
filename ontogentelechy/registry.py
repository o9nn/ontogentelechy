"""
Telos Registry — Plugin system for registering and discovering Telos implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .core import Telos


@dataclass
class RegistryEntry:
    """An entry in the Telos registry."""
    name: str
    factory: Callable[..., Telos]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelosRegistry:
    """Registry for Telos implementations.
    
    Allows registering, discovering, and instantiating Telos by name.
    Supports tags for categorization and filtering.
    """
    
    def __init__(self) -> None:
        self._entries: Dict[str, RegistryEntry] = {}
    
    def register(
        self,
        name: str,
        factory: Callable[..., Telos],
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a Telos factory by name."""
        self._entries[name] = RegistryEntry(
            name=name,
            factory=factory,
            description=description,
            tags=tags or [],
            metadata=metadata or {},
        )
    
    def register_decorator(
        self,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Callable:
        """Decorator to register a Telos factory."""
        def decorator(fn: Callable[..., Telos]) -> Callable[..., Telos]:
            registry_name = name or fn.__name__
            self.register(registry_name, fn, description=description, tags=tags or [])
            return fn
        return decorator
    
    def get(self, name: str, **kwargs: Any) -> Telos:
        """Instantiate a Telos by name."""
        if name not in self._entries:
            available = list(self._entries.keys())
            raise KeyError(f"Telos '{name}' not found. Available: {available}")
        return self._entries[name].factory(**kwargs)
    
    def list_names(self) -> List[str]:
        """List all registered Telos names."""
        return list(self._entries.keys())
    
    def list_by_tag(self, tag: str) -> List[str]:
        """List Telos names matching a tag."""
        return [name for name, entry in self._entries.items() if tag in entry.tags]
    
    def describe(self, name: str) -> str:
        """Return the description of a registered Telos."""
        if name not in self._entries:
            raise KeyError(f"Telos '{name}' not found.")
        entry = self._entries[name]
        tags_str = ", ".join(entry.tags) if entry.tags else "none"
        return f"{name}: {entry.description} [tags: {tags_str}]"
    
    def __contains__(self, name: str) -> bool:
        return name in self._entries
    
    def __len__(self) -> int:
        return len(self._entries)


# Global default registry, pre-populated with example teloi
registry = TelosRegistry()

def _populate_default_registry() -> None:
    from .examples import (
        create_semantic_coherence_telos,
        create_adaptive_learning_telos,
        create_complexity_emergence_telos,
        create_efficient_computation_telos,
        create_knowledge_integration_telos,
    )
    registry.register(
        "semantic_coherence", create_semantic_coherence_telos,
        description="Achieve coherent semantic representation",
        tags=["cognitive", "semantic"],
    )
    registry.register(
        "adaptive_learning", create_adaptive_learning_telos,
        description="Continuously learn and adapt",
        tags=["cognitive", "learning"],
    )
    registry.register(
        "complexity_emergence", create_complexity_emergence_telos,
        description="Foster emergence of complex patterns",
        tags=["emergence", "complexity"],
    )
    registry.register(
        "efficient_computation", create_efficient_computation_telos,
        description="Optimize computational efficiency",
        tags=["efficiency", "computation"],
    )
    registry.register(
        "knowledge_integration", create_knowledge_integration_telos,
        description="Integrate knowledge across domains",
        tags=["cognitive", "knowledge"],
    )

_populate_default_registry()
