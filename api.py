"""
api.py — Module C (Part 2): The FastAPI Application Server
===========================================================
Exposes two endpoints consumed by the React frontend:

  GET  /api/probe        → Returns the 3D topology JSON for a single model
                           file.  Called before merge to populate the canvas.

  GET  /api/probe-pair   → Returns topologies for both models + common keys.

  POST /api/validate     → Quick sanity-check (file exists, is .safetensors).

  WS   /ws/merge         → Streams per-layer LayerEvent + MergeCompleteEvent
                           JSON objects in real-time as the engine runs.

Running the server
------------------
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

CORS
----
Origins are open (*) for local development.  Restrict in production.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from probe import probe, probe_pair, ModelTopology, LayerBlock, TensorMeta
from engine import MergeConfig, stream_merge
from tensor_math import MergeAlgorithm
from hub_utils import resolve_path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlab.api")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LayerLab API",
    description=(
        "Out-of-Core Neural Surgery Engine — "
        "stream-merges billion-parameter LLMs on 8 GB RAM."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic I/O schemas
# ---------------------------------------------------------------------------

# ── Probe ──────────────────────────────────────────────────────────────────

class ProbeRequest(BaseModel):
    path: str = Field(..., description="Absolute path to a .safetensors file on the server host.")


class TensorMetaOut(BaseModel):
    key:        str
    shape:      list[int]
    dtype:      str
    num_params: int


class LayerBlockOut(BaseModel):
    index:        int
    label:        str
    block_type:   str
    tensors:      list[TensorMetaOut]
    total_params: int


class TopologyOut(BaseModel):
    path:         str
    num_layers:   int
    blocks:       list[LayerBlockOut]
    all_keys:     list[str]
    total_params: int
    metadata:     dict[str, str]


class ProbePairRequest(BaseModel):
    path_a: str
    path_b: str


class ProbePairOut(BaseModel):
    topology_a:  TopologyOut
    topology_b:  TopologyOut
    common_keys: list[str]
    common_count: int


# ── Validate ───────────────────────────────────────────────────────────────

class ValidateRequest(BaseModel):
    path: str


class ValidateResponse(BaseModel):
    valid:         bool
    path:          str
    size_bytes:    int
    error_message: str = ""


# ── Merge (WebSocket payload) ──────────────────────────────────────────────

class MergeRequest(BaseModel):
    """
    JSON payload sent by the frontend over the WebSocket immediately after
    the connection is established.  The engine then starts streaming events.
    """
    path_a:          str
    path_b:          str
    output_path:     str
    algorithm:       str = "lerp"
    global_alpha:    float = Field(0.5, ge=0.0, le=1.0)
    path_base:       str | None = None
    per_layer_alpha: dict[str, float] = Field(default_factory=dict)
    trim_fraction:   float = Field(0.80, ge=0.0, lt=1.0)
    warn_threshold:  float = Field(0.70, ge=0.0, le=1.0)

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        valid = {a.value for a in MergeAlgorithm}
        if v not in valid:
            raise ValueError(f"algorithm must be one of {valid}, got '{v}'")
        return v

    @field_validator("per_layer_alpha")
    @classmethod
    def validate_alphas(cls, d: dict[str, float]) -> dict[str, float]:
        for k, v in d.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"per_layer_alpha['{k}'] = {v} is not in [0, 1]")
        return d


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_tensor_meta(t: TensorMeta) -> TensorMetaOut:
    return TensorMetaOut(
        key=t.key,
        shape=t.shape,
        dtype=t.dtype,
        num_params=t.num_params,
    )


def _serialise_block(b: LayerBlock) -> LayerBlockOut:
    return LayerBlockOut(
        index=b.index,
        label=b.label,
        block_type=b.block_type,
        tensors=[_serialise_tensor_meta(t) for t in b.tensors],
        total_params=b.total_params,
    )


def _serialise_topology(topo: ModelTopology) -> TopologyOut:
    return TopologyOut(
        path=topo.path,
        num_layers=topo.num_layers,
        blocks=[_serialise_block(b) for b in topo.blocks],
        all_keys=topo.all_keys,
        total_params=topo.total_params,
        metadata=topo.metadata,
    )


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness check."""
    return {"status": "ok", "service": "LayerLab API"}


@app.post("/api/probe", response_model=TopologyOut)
async def api_probe(req: ProbeRequest) -> TopologyOut:
    """
    Probe a single .safetensors file and return its Transformer topology.
    No weight data is loaded — only the JSON header is parsed.
    """
    try:
        path = resolve_path(req.path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    if path.suffix != ".safetensors":
        raise HTTPException(
            status_code=400,
            detail=f"Expected .safetensors extension, got '{path.suffix}'",
        )
    try:
        topo = probe(path)
    except Exception as exc:
        logger.exception(f"Probe error for {req.path}")
        raise HTTPException(status_code=422, detail=str(exc))

    return _serialise_topology(topo)


@app.post("/api/probe-pair", response_model=ProbePairOut)
async def api_probe_pair(req: ProbePairRequest) -> ProbePairOut:
    """
    Probe both models simultaneously and return their topologies plus the
    sorted list of tensor keys common to both (eligible for merging).
    """
    try:
        resolved_a = resolve_path(req.path_a)
        resolved_b = resolve_path(req.path_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    for p, resolved_p in [("A", resolved_a), ("B", resolved_b)]:
        if not resolved_p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {p}")
        if resolved_p.suffix != ".safetensors":
            raise HTTPException(status_code=400, detail=f"Not a .safetensors file: {p}")

    try:
        topo_a, topo_b, common_keys = probe_pair(resolved_a, resolved_b)
    except Exception as exc:
        logger.exception("Probe-pair error")
        raise HTTPException(status_code=422, detail=str(exc))

    return ProbePairOut(
        topology_a=_serialise_topology(topo_a),
        topology_b=_serialise_topology(topo_b),
        common_keys=common_keys,
        common_count=len(common_keys),
    )


@app.post("/api/validate", response_model=ValidateResponse)
async def api_validate(req: ValidateRequest) -> ValidateResponse:
    """
    Quick file validation: exists, .safetensors extension, readable header.
    Used by the frontend to highlight invalid path inputs before running a merge.
    """
    try:
        path = resolve_path(req.path)
    except Exception as e:
        return ValidateResponse(valid=False, path=req.path, size_bytes=0, error_message=str(e))

    if not path.exists():
        return ValidateResponse(valid=False, path=req.path, size_bytes=0,
                                error_message="File not found.")
    if path.suffix != ".safetensors":
        return ValidateResponse(valid=False, path=req.path, size_bytes=0,
                                error_message=f"Expected .safetensors, got '{path.suffix}'.")
    try:
        probe(path)     # header-only, fast
        return ValidateResponse(valid=True, path=req.path, size_bytes=path.stat().st_size)
    except Exception as exc:
        return ValidateResponse(valid=False, path=req.path, size_bytes=0,
                                error_message=str(exc))


# ---------------------------------------------------------------------------
# WebSocket Endpoint — /ws/merge
# ---------------------------------------------------------------------------

@app.websocket("/ws/merge")
async def ws_merge(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for the streaming merge pipeline.

    Protocol
    --------
    1. Client connects.
    2. Server sends:  {"event_type": "connected", "message": "Send merge config."}
    3. Client sends:  MergeRequest JSON (single message).
    4. Server streams: LayerEvent JSON objects (one per layer, two per key:
       layer_start/layer_warning then layer_done).
    5. Server sends:  MergeCompleteEvent JSON and closes the connection.

    Error handling
    --------------
    If the client sends malformed JSON or invalid config, the server sends an
    error event and closes gracefully — it does NOT crash the process.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection accepted from {websocket.client}")

    # --- Handshake -------------------------------------------------------
    await websocket.send_json({
        "event_type": "connected",
        "message":    "LayerLab WebSocket ready. Send your MergeRequest JSON.",
    })

    # --- Receive config --------------------------------------------------
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        merge_req = MergeRequest(**data)
    except json.JSONDecodeError as exc:
        await websocket.send_json({
            "event_type":  "error",
            "error_msg":   f"Invalid JSON payload: {exc}",
        })
        await websocket.close(code=1003)
        return
    except Exception as exc:
        await websocket.send_json({
            "event_type":  "error",
            "error_msg":   f"Config validation error: {exc}",
        })
        await websocket.close(code=1003)
        return

    # --- Validate input paths before starting the engine ----------------
    resolved_paths = {}
    try:
        resolved_paths["path_a"] = resolve_path(merge_req.path_a)
        resolved_paths["path_b"] = resolve_path(merge_req.path_b)
        if merge_req.path_base:
            resolved_paths["path_base"] = resolve_path(merge_req.path_base)
        else:
            resolved_paths["path_base"] = None
    except Exception as e:
        await websocket.send_json({"event_type": "error", "error_msg": str(e)})
        await websocket.close(code=1003)
        return

    for label, p in [("path_a", resolved_paths["path_a"]), ("path_b", resolved_paths["path_b"])]:
        if not p.exists():
            await websocket.send_json({
                "event_type": "error",
                "error_msg":  f"{label} not found on disk after resolution.",
            })
            await websocket.close(code=1003)
            return

    if resolved_paths.get("path_base"):
        if not resolved_paths["path_base"].exists():
            await websocket.send_json({
                "event_type": "error",
                "error_msg":  f"path_base not found on disk after resolution.",
            })
            await websocket.close(code=1003)
            return

    # --- Build engine config ---------------------------------------------
    config = MergeConfig(
        path_a          = resolved_paths["path_a"],
        path_b          = resolved_paths["path_b"],
        output_path     = merge_req.output_path,
        algorithm       = MergeAlgorithm(merge_req.algorithm),
        global_alpha    = merge_req.global_alpha,
        path_base       = resolved_paths["path_base"],
        per_layer_alpha = merge_req.per_layer_alpha,
        trim_fraction   = merge_req.trim_fraction,
        warn_threshold  = merge_req.warn_threshold,
    )

    logger.info(
        f"Starting merge: alg={config.algorithm.value}, α={config.global_alpha}, "
        f"A={config.path_a}, B={config.path_b} → {config.output_path}"
    )

    await websocket.send_json({
        "event_type": "merge_start",
        "message":    f"Engine started. Algorithm: {config.algorithm.value.upper()}, alpha={config.global_alpha}",
        "config": {
            "algorithm":    config.algorithm.value,
            "global_alpha": config.global_alpha,
            "path_a":       str(config.path_a),
            "path_b":       str(config.path_b),
            "output_path":  str(config.output_path),
            "warn_threshold": config.warn_threshold,
        },
    })

    # --- Stream engine events to client ----------------------------------
    try:
        async for event in stream_merge(config):
            try:
                await websocket.send_json(event)
            except WebSocketDisconnect:
                logger.warning("Client disconnected mid-stream; aborting.")
                return

            # If this is the final completion event, close cleanly.
            if event.get("event_type") == "merge_complete":
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client.")
        return
    except Exception as exc:
        logger.exception(f"Unexpected WebSocket error: {exc}")
        try:
            await websocket.send_json({
                "event_type": "error",
                "error_msg":  f"Server-side crash: {exc}",
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket connection closed.")


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
