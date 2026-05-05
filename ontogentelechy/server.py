"""
Ontogentelechy Experiment Server

FastAPI server for tracking entity actualization in real-time.
Install with: pip install ontogentelechy[server]

Usage:
    uvicorn ontogentelechy.server:app --reload
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .core import ActualizationTracker, Telos
from .entity import SimpleEntity, SimpleGene
from .examples import EXAMPLE_TELOI

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


def _require_fastapi() -> None:
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install ontogentelechy[server]"
        )


if FASTAPI_AVAILABLE:

    class StateUpdate(BaseModel):
        entity_id: str
        state: List[float]
        fitness: Optional[float] = None
        metadata: Dict[str, Any] = {}

    class MetricsResponse(BaseModel):
        entity_id: str
        potentiality: float
        emergence: float
        integration: float
        actualization: float
        telos_alignment: float
        overall_health: float
        phase: str

    class TelosRegistration(BaseModel):
        name: str
        preset: Optional[str] = None

    class ServerStatus(BaseModel):
        version: str
        registered_teloi: List[str]
        tracked_entities: List[str]
        available_presets: List[str]

    app = FastAPI(
        title="Ontogentelechy Server",
        description="Real-time teleological actualization tracking",
        version="0.5.0",
    )

    # In-memory state
    _teloi: Dict[str, Telos] = {}
    _trackers: Dict[str, ActualizationTracker] = {}
    _entities: Dict[str, SimpleEntity] = {}

    # Register default presets
    for preset_name in EXAMPLE_TELOI:
        _teloi[preset_name] = EXAMPLE_TELOI[preset_name]()

    @app.get("/", response_model=ServerStatus)
    async def status():
        from . import __version__

        return ServerStatus(
            version=__version__,
            registered_teloi=list(_teloi.keys()),
            tracked_entities=list(_entities.keys()),
            available_presets=list(EXAMPLE_TELOI.keys()),
        )

    @app.post("/teloi/register")
    async def register_telos(reg: TelosRegistration):
        if reg.preset:
            if reg.preset not in EXAMPLE_TELOI:
                raise HTTPException(400, f"Unknown preset: {reg.preset}")
            _teloi[reg.name] = EXAMPLE_TELOI[reg.preset]()
        else:
            raise HTTPException(400, "Must provide 'preset' for now")
        return {"status": "registered", "name": reg.name}

    @app.get("/teloi")
    async def list_teloi():
        return {"teloi": list(_teloi.keys())}

    @app.post("/entities/{entity_id}/update", response_model=MetricsResponse)
    async def update_entity(
        entity_id: str, update: StateUpdate, telos_name: str = "semantic_coherence"
    ):
        state = np.array(update.state, dtype=float)

        if entity_id not in _entities:
            _entities[entity_id] = SimpleEntity(
                state=state,
                genes=[SimpleGene(weight=float(np.clip(v, 0.0, 1.0))) for v in state],
                fitness=update.fitness,
                metadata=update.metadata,
            )
            _trackers[entity_id] = ActualizationTracker()
        else:
            _entities[entity_id].update_state(state)
            if update.fitness is not None:
                _entities[entity_id].fitness = update.fitness
            _entities[entity_id].metadata.update(update.metadata)

        if telos_name not in _teloi:
            raise HTTPException(404, f"Telos '{telos_name}' not registered")

        telos = _teloi[telos_name]
        tracker = _trackers[entity_id]
        entity = _entities[entity_id]

        metrics = tracker.compute_metrics(entity, telos)

        return MetricsResponse(
            entity_id=entity_id,
            potentiality=metrics.potentiality,
            emergence=metrics.emergence,
            integration=metrics.integration,
            actualization=metrics.actualization,
            telos_alignment=metrics.telos_alignment,
            overall_health=metrics.overall_health,
            phase=telos.phase.value,
        )

    @app.get("/entities/{entity_id}/history")
    async def entity_history(entity_id: str):
        if entity_id not in _trackers:
            raise HTTPException(404, f"Entity '{entity_id}' not found")
        tracker = _trackers[entity_id]
        return {
            "entity_id": entity_id,
            "history": [
                {
                    "potentiality": m.potentiality,
                    "emergence": m.emergence,
                    "integration": m.integration,
                    "actualization": m.actualization,
                    "telos_alignment": m.telos_alignment,
                    "overall_health": m.overall_health,
                }
                for m in tracker.history
            ],
            "phase_transitions": len(tracker.phase_transitions),
        }

    @app.delete("/entities/{entity_id}")
    async def delete_entity(entity_id: str):
        for store in (_entities, _trackers):
            store.pop(entity_id, None)
        return {"status": "deleted", "entity_id": entity_id}

else:
    # Stub so the module can be imported without fastapi
    app = None  # type: ignore
