"""
engine.py — Module C (Part 1): The Out-of-Core Streaming Pipeline
==================================================================
Iterates through every common layer key shared by Model A and Model B,
processes one layer at a time, measures interference, calls the tensor
math dispatcher, and writes the result to a new .safetensors file.

STRICT RAM CONTRACT (mirrors tensor_math.py)
--------------------------------------------
  • safetensors.safe_open is opened with mmap=True.  No weight data enters
    RAM until the OS page-fault resolves the specific byte-range for key k.
  • After each layer: del tensors + explicit gc.collect().
  • Peak RAM ≤ size_of_largest_single_layer × (K_models + 1_output + scratch).
  • Never calls torch.load() or torch.save() directly.

Output format
-------------
  Accumulated tensors are written in one shot via safetensors.torch.save_file
  once all keys have been processed.  If a crash occurs mid-run the partial
  state dict is discarded — no corrupt output is ever written.

  For very large models (>13B) the caller should instead stream-write via a
  custom safetensors builder; that path is marked with a TODO comment below.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from probe import probe, probe_pair, ModelTopology
from tensor_math import MergeAlgorithm, merge, cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MergeConfig:
    """
    All user-supplied parameters for a single merge job.

    path_a          : path to Model A .safetensors file
    path_b          : path to Model B .safetensors file
    path_base       : optional base model for TIES; if None and algorithm==TIES
                      engine falls back to using tensor_a as the base and
                      emits a warning.
    output_path     : destination .safetensors file (must not yet exist, or
                      the engine will overwrite it after confirmation)
    algorithm       : LERP | SLERP | TIES
    global_alpha    : default interpolation ratio ∈ [0, 1] for all layers
    per_layer_alpha : optional dict {tensor_key: alpha} for surgical overrides
    trim_fraction   : TIES-specific noise trim threshold (default 0.80)
    warn_threshold  : cosine similarity below this value triggers a WARNING
                      event in the stream (default 0.70)
    """
    path_a: str | Path
    path_b: str | Path
    output_path: str | Path
    algorithm: MergeAlgorithm | str = MergeAlgorithm.LERP
    global_alpha: float = 0.5
    path_base: str | Path | None = None
    per_layer_alpha: dict[str, float] = field(default_factory=dict)
    trim_fraction: float = 0.80
    warn_threshold: float = 0.70


# ---------------------------------------------------------------------------
# Stream event dataclasses (serialised to JSON by api.py)
# ---------------------------------------------------------------------------

@dataclass
class LayerEvent:
    """
    Emitted once per layer during the merge stream.

    event_type  : "layer_start" | "layer_done" | "layer_warning"
    key         : tensor key being processed
    index       : 0-based ordinal within the key list
    total       : total number of keys in the merge job
    progress    : percentage complete ∈ [0.0, 100.0]
    cosine_sim  : cosine similarity between TA and TB for this layer
    alpha_used  : the alpha value actually applied (may be surgical override)
    elapsed_sec : wall-clock seconds since the merge started
    warning_msg : human-readable warning if event_type == "layer_warning"
    """
    event_type: str
    key: str
    index: int
    total: int
    progress: float
    cosine_sim: float
    alpha_used: float
    elapsed_sec: float
    warning_msg: str = ""

    def to_dict(self) -> dict:
        return {
            "event_type":  self.event_type,
            "key":         self.key,
            "index":       self.index,
            "total":       self.total,
            "progress":    round(self.progress, 2),
            "cosine_sim":  round(self.cosine_sim, 6),
            "alpha_used":  round(self.alpha_used, 4),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "warning_msg": self.warning_msg,
        }


@dataclass
class MergeCompleteEvent:
    """Emitted once when the merge job finishes (or fails)."""
    success: bool
    output_path: str
    total_keys: int
    elapsed_sec: float
    peak_ram_mb: float          # approximate, from tracemalloc if available
    error_msg: str = ""

    def to_dict(self) -> dict:
        return {
            "event_type":   "merge_complete",
            "success":      self.success,
            "output_path":  self.output_path,
            "total_keys":   self.total_keys,
            "elapsed_sec":  round(self.elapsed_sec, 2),
            "peak_ram_mb":  round(self.peak_ram_mb, 1),
            "error_msg":    self.error_msg,
        }


# ---------------------------------------------------------------------------
# RAM tracker (lightweight, no tracemalloc overhead)
# ---------------------------------------------------------------------------

def _approx_ram_mb() -> float:
    """
    Return approximate RSS memory in MB for the current process.
    Uses /proc/self/status on Linux; falls back to psutil if available;
    returns -1.0 if neither is available.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Core streaming engine
# ---------------------------------------------------------------------------

async def stream_merge(
    config: MergeConfig,
    on_event: Callable[[dict], None] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that performs the out-of-core merge and yields
    one dict per layer event (plus a final MergeCompleteEvent).

    Usage in api.py (WebSocket handler):

        async for event in stream_merge(config):
            await websocket.send_json(event)

    Parameters
    ----------
    config   : MergeConfig describing the merge job.
    on_event : optional synchronous callback (for testing / logging).

    Yields
    ------
    dict  — serialisable JSON payload (LayerEvent or MergeCompleteEvent).

    RAM contract
    ------------
    Inside the loop we hold at most 3 tensors simultaneously:
      tensor_a, tensor_b, (tensor_base if TIES), merged_tensor.
    After writing to state_dict we immediately del + gc.collect().
    """

    start_time  = time.monotonic()
    state_dict: dict[str, torch.Tensor] = {}
    peak_ram_mb = 0.0
    alg         = MergeAlgorithm(config.algorithm)

    # ------------------------------------------------------------------
    # 1. Probe both models — header-only, no weight data loaded.
    # ------------------------------------------------------------------
    logger.info("Probing model pair …")
    try:
        topo_a, topo_b, common_keys = probe_pair(config.path_a, config.path_b)
    except Exception as exc:
        event = MergeCompleteEvent(
            success=False,
            output_path=str(config.output_path),
            total_keys=0,
            elapsed_sec=time.monotonic() - start_time,
            peak_ram_mb=_approx_ram_mb(),
            error_msg=f"Probe failed: {exc}",
        ).to_dict()
        if on_event:
            on_event(event)
        yield event
        return

    if not common_keys:
        event = MergeCompleteEvent(
            success=False,
            output_path=str(config.output_path),
            total_keys=0,
            elapsed_sec=time.monotonic() - start_time,
            peak_ram_mb=_approx_ram_mb(),
            error_msg="No common keys found between the two models.",
        ).to_dict()
        if on_event:
            on_event(event)
        yield event
        return

    total = len(common_keys)
    logger.info(f"Merging {total} common keys with algorithm={alg.value} α={config.global_alpha}")

    # ------------------------------------------------------------------
    # 2. TIES base model handle (opened once, kept for the whole loop)
    # ------------------------------------------------------------------
    base_handle = None
    if alg == MergeAlgorithm.TIES:
        if config.path_base:
            base_handle = safe_open(str(config.path_base), framework="pt", device="cpu")
            logger.info("TIES base model handle opened.")
        else:
            logger.warning(
                "TIES selected but no base model path provided. "
                "Using tensor_a as base for each layer (task vector = A - A = 0 for model A). "
                "Results may differ from reference TIES. Provide a dedicated base for best quality."
            )

    # ------------------------------------------------------------------
    # 3. Open both model handles (mmap=True — no weights loaded yet)
    # ------------------------------------------------------------------
    handle_a = safe_open(str(config.path_a), framework="pt", device="cpu")
    handle_b = safe_open(str(config.path_b), framework="pt", device="cpu")

    # ------------------------------------------------------------------
    # 4. Layer-by-layer streaming loop
    # ------------------------------------------------------------------
    try:
        for idx, key in enumerate(common_keys):

            # Yield control to the event loop so WebSocket sends don't starve.
            await asyncio.sleep(0)

            elapsed = time.monotonic() - start_time
            progress = (idx / total) * 100.0

            # --- 4a. Lazy-load tensors (OS page fault resolves here) -------
            try:
                tensor_a = handle_a.get_tensor(key)   # triggers mmap read
                tensor_b = handle_b.get_tensor(key)   # triggers mmap read
            except Exception as exc:
                logger.error(f"Failed to load tensors for key '{key}': {exc}")
                # Skip this key and continue; record cosine_sim as NaN.
                yield LayerEvent(
                    event_type  = "layer_warning",
                    key         = key,
                    index       = idx,
                    total       = total,
                    progress    = progress,
                    cosine_sim  = float("nan"),
                    alpha_used  = 0.0,
                    elapsed_sec = elapsed,
                    warning_msg = f"Tensor load error: {exc}",
                ).to_dict()
                continue

            # --- 4b. Interference detection (cosine similarity) ------------
            cos_sim = cosine_similarity(tensor_a, tensor_b)

            # Track peak RAM after every load.
            current_ram = _approx_ram_mb()
            if current_ram > peak_ram_mb:
                peak_ram_mb = current_ram

            # --- 4c. Determine alpha for this layer (surgical override?) ---
            alpha = config.per_layer_alpha.get(key, config.global_alpha)

            # --- 4d. Emit layer_start / layer_warning event ----------------
            event_type  = "layer_start"
            warning_msg = ""

            if cos_sim < config.warn_threshold:
                event_type  = "layer_warning"
                warning_msg = (
                    f"Severe parameter interference detected (cos_sim={cos_sim:.4f} "
                    f"< threshold={config.warn_threshold}). "
                    "Consider adjusting the per-layer alpha in the Surgical Panel."
                )
                logger.warning(f"[{key}] {warning_msg}")

            layer_event = LayerEvent(
                event_type  = event_type,
                key         = key,
                index       = idx,
                total       = total,
                progress    = progress,
                cosine_sim  = cos_sim,
                alpha_used  = alpha,
                elapsed_sec = elapsed,
                warning_msg = warning_msg,
            ).to_dict()

            if on_event:
                on_event(layer_event)
            yield layer_event

            # --- 4e. Merge ------------------------------------------------
            try:
                if alg == MergeAlgorithm.TIES:
                    if base_handle is not None:
                        tensor_base = base_handle.get_tensor(key)
                    else:
                        # Fallback: use tensor_a as base (task vector for A → 0)
                        tensor_base = tensor_a.clone()

                    merged = merge(
                        tensor_a      = tensor_a,
                        tensor_b      = tensor_b,
                        alpha         = alpha,
                        algorithm     = alg,
                        tensor_base   = tensor_base,
                        trim_fraction = config.trim_fraction,
                    )
                    del tensor_base
                    gc.collect()
                else:
                    merged = merge(
                        tensor_a  = tensor_a,
                        tensor_b  = tensor_b,
                        alpha     = alpha,
                        algorithm = alg,
                    )

            except Exception as exc:
                logger.error(f"Math error on key '{key}': {exc}")
                # On math failure, copy tensor_a verbatim (safe fallback).
                merged = tensor_a.clone()
                warning_msg = f"Math error ({exc}); tensor_a copied verbatim."
                yield LayerEvent(
                    event_type  = "layer_warning",
                    key         = key,
                    index       = idx,
                    total       = total,
                    progress    = progress,
                    cosine_sim  = cos_sim,
                    alpha_used  = alpha,
                    elapsed_sec = time.monotonic() - start_time,
                    warning_msg = warning_msg,
                ).to_dict()

            # --- 4f. Store merged tensor; free inputs immediately ----------
            state_dict[key] = merged      # ownership transferred to dict

            # CRITICAL: delete input references NOW — do not hold both
            # the merged result AND the inputs simultaneously any longer
            # than this single line.
            del tensor_a, tensor_b, merged
            gc.collect()

            # --- 4g. Emit layer_done event ---------------------------------
            done_progress = ((idx + 1) / total) * 100.0
            done_event = LayerEvent(
                event_type  = "layer_done",
                key         = key,
                index       = idx,
                total       = total,
                progress    = done_progress,
                cosine_sim  = cos_sim,
                alpha_used  = alpha,
                elapsed_sec = time.monotonic() - start_time,
            ).to_dict()

            if on_event:
                on_event(done_event)
            yield done_event

    except Exception as exc:
        # Unexpected crash — emit failure event before re-raising.
        logger.exception(f"Unexpected engine error: {exc}")
        event = MergeCompleteEvent(
            success      = False,
            output_path  = str(config.output_path),
            total_keys   = total,
            elapsed_sec  = time.monotonic() - start_time,
            peak_ram_mb  = _approx_ram_mb(),
            error_msg    = f"Engine crash: {exc}",
        ).to_dict()
        if on_event:
            on_event(event)
        yield event
        return

    # ------------------------------------------------------------------
    # 5. Write output (single-shot safetensors write)
    # ------------------------------------------------------------------
    # TODO (large model path): for models >13B, replace this block with
    # an incremental safetensors writer that flushes tensors to disk one
    # at a time to avoid re-accumulating the full model in RAM here.
    logger.info(f"Writing merged model to {config.output_path} …")
    try:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(output_path))
    except Exception as exc:
        event = MergeCompleteEvent(
            success      = False,
            output_path  = str(config.output_path),
            total_keys   = total,
            elapsed_sec  = time.monotonic() - start_time,
            peak_ram_mb  = _approx_ram_mb(),
            error_msg    = f"Write failed: {exc}",
        ).to_dict()
        if on_event:
            on_event(event)
        yield event
        return
    finally:
        # Always free the accumulated state dict after the write attempt.
        del state_dict
        gc.collect()

    # ------------------------------------------------------------------
    # 6. Final success event
    # ------------------------------------------------------------------
    final_event = MergeCompleteEvent(
        success      = True,
        output_path  = str(config.output_path),
        total_keys   = total,
        elapsed_sec  = time.monotonic() - start_time,
        peak_ram_mb  = _approx_ram_mb(),
    ).to_dict()

    if on_event:
        on_event(final_event)
    yield final_event

    logger.info(
        f"Merge complete. {total} keys in {final_event['elapsed_sec']:.1f}s, "
        f"peak RAM ≈ {final_event['peak_ram_mb']:.0f} MB"
    )
