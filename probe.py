"""
probe.py — Module A: The Metadata Probe
========================================
Reads ONLY the JSON header of a .safetensors file.  Zero binary weight
data is ever touched.  Returns a structured Transformer topology so the
frontend can render the 3D graph before any merge begins.

Key guarantee: this module never imports torch and never triggers any
memory allocation proportional to model size.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# safetensors lets us read ONLY the header (8-byte length prefix + JSON).
# We do this manually to avoid even opening a safe_open context.
# safe_open itself memory-maps the whole file descriptor; for probing we
# just need the first N bytes.
from safetensors import safe_open  # used only for dtype cross-check


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TensorMeta:
    """Lightweight descriptor for a single weight tensor."""
    key: str
    shape: list[int]
    dtype: str
    num_params: int = field(init=False)

    def __post_init__(self) -> None:
        p = 1
        for s in self.shape:
            p *= s
        self.num_params = p


@dataclass
class LayerBlock:
    """
    A logical block of tensors that belong to one Transformer layer index.
    E.g. all tensors whose key contains 'layers.14'.
    """
    index: int                           # numeric layer index (or -1 for special)
    label: str                           # human-readable, e.g. "Layer 14"
    block_type: str                      # "attention" | "mlp" | "embedding" | "lm_head" | "norm" | "other"
    tensors: list[TensorMeta] = field(default_factory=list)
    total_params: int = field(init=False, default=0)

    def finalise(self) -> None:
        self.total_params = sum(t.num_params for t in self.tensors)


@dataclass
class ModelTopology:
    """Full parsed topology of a model file."""
    path: str
    num_layers: int
    blocks: list[LayerBlock]
    all_keys: list[str]
    total_params: int
    metadata: dict[str, str]


# ---------------------------------------------------------------------------
# Internal: read only the safetensors file header (no mmap of weights)
# ---------------------------------------------------------------------------

_HEADER_LENGTH_BYTES = 8  # first 8 bytes encode header JSON length as uint64-LE


def _read_raw_header(path: Path) -> dict[str, Any]:
    """
    Manually parse the safetensors binary header without opening the full
    memory-map.  Only reads the first (8 + header_len) bytes of the file.
    """
    with open(path, "rb") as fh:
        raw_len = fh.read(_HEADER_LENGTH_BYTES)
        if len(raw_len) < _HEADER_LENGTH_BYTES:
            raise ValueError(f"File too small to be a valid safetensors file: {path}")

        header_len: int = struct.unpack("<Q", raw_len)[0]

        if header_len > 100 * 1024 * 1024:  # sanity: 100 MB header cap
            raise ValueError(
                f"Suspiciously large header ({header_len} bytes). "
                "File may be corrupt or not a safetensors file."
            )

        raw_header = fh.read(header_len)
        if len(raw_header) < header_len:
            raise ValueError("Truncated header — file may be corrupt.")

    return json.loads(raw_header.decode("utf-8"))


# ---------------------------------------------------------------------------
# Internal: topology parser — maps flat key list → structured LayerBlock list
# ---------------------------------------------------------------------------

# Patterns that identify common Transformer tensor naming conventions.
# Supports: LLaMA / Mistral / Falcon / GPT-NeoX / Phi style keys.
_LAYER_INDEX_RE = re.compile(
    r"(?:layers|blocks|h|encoder\.layer|decoder\.layer)[._](\d+)"
)
_ATTENTION_KEYWORDS = {"q_proj", "k_proj", "v_proj", "o_proj", "query", "key",
                       "value", "out_proj", "q_attn", "c_attn", "attn"}
_MLP_KEYWORDS       = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2",
                       "mlp", "ffn", "dense_h_to_4h", "dense_4h_to_h", "c_fc",
                       "c_proj"}
_EMBED_KEYWORDS     = {"embed_tokens", "wte", "word_embeddings",
                       "token_embeddings", "embed_in"}
_LMHEAD_KEYWORDS    = {"lm_head", "embed_out", "output.weight"}
_NORM_KEYWORDS      = {"norm", "ln", "layer_norm", "layernorm", "rmsnorm"}


def _classify_block_type(key: str) -> str:
    k = key.lower()
    if any(kw in k for kw in _EMBED_KEYWORDS):
        return "embedding"
    if any(kw in k for kw in _LMHEAD_KEYWORDS):
        return "lm_head"
    if any(kw in k for kw in _ATTENTION_KEYWORDS):
        return "attention"
    if any(kw in k for kw in _MLP_KEYWORDS):
        return "mlp"
    if any(kw in k for kw in _NORM_KEYWORDS):
        return "norm"
    return "other"


def _parse_topology(
    header: dict[str, Any]
) -> tuple[list[LayerBlock], list[str]]:
    """
    Convert the flat key→{dtype, shape, data_offsets} header into a list of
    LayerBlock objects ordered: embedding → layers 0..N → norm → lm_head.
    """
    # '__metadata__' is a reserved key in safetensors headers.
    all_keys = sorted(k for k in header if k != "__metadata__")

    # Group keys by their layer index (or None for global tensors).
    index_map: dict[int | str, list[TensorMeta]] = {}

    for key in all_keys:
        info = header[key]
        meta = TensorMeta(
            key=key,
            shape=info.get("shape", []),
            dtype=info.get("dtype", "unknown"),
        )

        m = _LAYER_INDEX_RE.search(key)
        if m:
            idx: int | str = int(m.group(1))
        else:
            # No numeric layer index — classify as special.
            block_type = _classify_block_type(key)
            idx = block_type  # string sentinel, e.g. "embedding"

        index_map.setdefault(idx, []).append(meta)

    # Build LayerBlock list — numeric indices first, then special blocks.
    numeric_indices = sorted(k for k in index_map if isinstance(k, int))
    special_keys    = [k for k in index_map if isinstance(k, str)]

    blocks: list[LayerBlock] = []

    # --- Embedding block(s) ---
    for sk in ("embedding", "other"):
        if sk in special_keys:
            blk = LayerBlock(
                index=-1,
                label="Embeddings" if sk == "embedding" else "Global / Other",
                block_type=sk,
                tensors=index_map[sk],
            )
            blk.finalise()
            blocks.append(blk)

    # --- Transformer layers ---
    for idx in numeric_indices:
        tensors = index_map[idx]
        # Infer dominant block type from tensor keys.
        types = [_classify_block_type(t.key) for t in tensors]
        dominant = max(set(types), key=types.count)
        blk = LayerBlock(
            index=idx,
            label=f"Layer {idx}",
            block_type=dominant,
            tensors=tensors,
        )
        blk.finalise()
        blocks.append(blk)

    # --- Norm & LM Head ---
    for sk in ("norm", "lm_head"):
        if sk in special_keys:
            blk = LayerBlock(
                index=-1,
                label="Final Norm" if sk == "norm" else "LM Head",
                block_type=sk,
                tensors=index_map[sk],
            )
            blk.finalise()
            blocks.append(blk)

    return blocks, all_keys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe(path: str | Path) -> ModelTopology:
    """
    Parse the metadata of a .safetensors file and return a ModelTopology.

    This function:
      1. Opens only the first ~N bytes of the file (header).
      2. Parses the JSON to extract key names, shapes, dtypes.
      3. Reconstructs the Transformer topology into named LayerBlocks.
      4. Returns structured data suitable for 3D rendering on the frontend.

    No weight tensors are loaded.  Memory usage is O(number_of_keys).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix not in (".safetensors",):
        raise ValueError(f"Expected a .safetensors file, got: {path.suffix}")

    raw_header = _read_raw_header(path)

    # Extract safetensors file-level metadata (optional dict under __metadata__).
    file_metadata: dict[str, str] = raw_header.get("__metadata__", {}) or {}

    blocks, all_keys = _parse_topology(raw_header)

    # Count transformer layers (blocks with numeric index).
    numeric_layers = [b for b in blocks if b.index >= 0]
    num_layers = len(numeric_layers)

    total_params = sum(b.total_params for b in blocks)

    return ModelTopology(
        path=str(path),
        num_layers=num_layers,
        blocks=blocks,
        all_keys=all_keys,
        total_params=total_params,
        metadata=file_metadata,
    )


def probe_pair(
    path_a: str | Path,
    path_b: str | Path,
) -> tuple[ModelTopology, ModelTopology, list[str]]:
    """
    Probe both models and return their topologies plus the list of keys
    that are common to both (eligible for merging).
    """
    topo_a = probe(path_a)
    topo_b = probe(path_b)

    set_a = set(topo_a.all_keys)
    set_b = set(topo_b.all_keys)
    common_keys = sorted(set_a & set_b)

    return topo_a, topo_b, common_keys
