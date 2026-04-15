"""
tensor_math.py — Module B: The Tensor Math Engine
===================================================
Implements three model-merging algorithms for pairs of weight tensors.

STRICT RAM CONTRACT
-------------------
Every function in this module MUST:
  1. Accept tensors as arguments (caller owns allocation).
  2. Produce a result tensor.
  3. Delete ALL intermediate tensors with `del` before returning.
  4. Call `gc.collect()` once at the end.
  5. NEVER return a reference to any intermediate.

The caller (engine.py) is responsible for deleting the input tensors and
calling gc.collect() after the result has been written to disk.

No function here ever holds more than ~3× the size of a single layer in RAM
at any moment.
"""

from __future__ import annotations

import gc
import math
from enum import Enum
from typing import Literal

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Algorithm enum
# ---------------------------------------------------------------------------

class MergeAlgorithm(str, Enum):
    LERP  = "lerp"
    SLERP = "slerp"
    TIES  = "ties"


# ---------------------------------------------------------------------------
# Utility: dtype-safe operations
# ---------------------------------------------------------------------------

def _to_float32(t: torch.Tensor) -> torch.Tensor:
    """
    Upcast to float32 for arithmetic precision.
    We never upcast in-place — always return a new tensor so the caller
    can del the original separately.
    """
    if t.dtype == torch.float32:
        return t
    return t.float()


# ---------------------------------------------------------------------------
# Algorithm 1: Linear Interpolation (LERP)
# ---------------------------------------------------------------------------

def lerp(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Standard weighted average:  result = (1 - α) * A  +  α * B

    α = 0.0 → pure Model A
    α = 1.0 → pure Model B

    Memory cost: 1 extra tensor (the result), no intermediates.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # torch.lerp does not upcast; we upcast manually for BF16/FP16 models.
    a_f = _to_float32(tensor_a)
    b_f = _to_float32(tensor_b)

    result_f = torch.lerp(a_f, b_f, alpha)

    # Cast back to original dtype to preserve the model's native precision.
    result = result_f.to(tensor_a.dtype)

    del a_f, b_f, result_f
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Algorithm 2: SLERP (Spherical Linear Interpolation)
# ---------------------------------------------------------------------------

_SLERP_EPS = 1e-8   # numerical stability guard


def slerp(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Interpolates along the great circle on the weight hypersphere.

    Procedure:
      1. Flatten both tensors into 1D vectors.
      2. L2-normalise each vector.
      3. Compute cosine of angle Ω between them.
      4. If |cos Ω| ≈ 1 (nearly parallel/anti-parallel), fall back to LERP
         to avoid division-by-zero in sin(Ω).
      5. result = sin((1-α)Ω)/sin(Ω) * A  +  sin(αΩ)/sin(Ω) * B
      6. Rescale the result to the geometric mean of the original norms.
      7. Reshape back to original shape.

    Reference: "Editing Models with Task Arithmetic" (Ilharco et al., 2022)
    and common SLERP quaternion formulations adapted for high-dim tensors.

    Memory cost: 4 intermediates (flat_a, flat_b, norm_a, norm_b) + result.
    All are deleted before return.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    original_shape = tensor_a.shape
    original_dtype = tensor_a.dtype

    # --- Step 1: flatten and upcast ---
    flat_a = _to_float32(tensor_a).flatten()   # shape: [N]
    flat_b = _to_float32(tensor_b).flatten()   # shape: [N]

    # --- Step 2: compute per-vector L2 norms (for rescaling later) ---
    norm_a = torch.linalg.norm(flat_a)
    norm_b = torch.linalg.norm(flat_b)

    # Geometric mean of norms — used to rescale after interpolation.
    # If either model has a zero-norm layer (e.g. untrained head), fall back.
    if norm_a < _SLERP_EPS or norm_b < _SLERP_EPS:
        result = lerp(tensor_a, tensor_b, alpha)
        del flat_a, flat_b, norm_a, norm_b
        gc.collect()
        return result

    # --- Step 3: normalise to unit sphere ---
    unit_a = flat_a / norm_a   # shape: [N]
    unit_b = flat_b / norm_b   # shape: [N]

    del flat_a, flat_b         # immediately free; no longer needed
    gc.collect()

    # --- Step 4: angle Ω ---
    # clamp to [-1, 1] to guard against floating-point drift beyond unit sphere
    cos_omega = torch.clamp(torch.dot(unit_a, unit_b), -1.0, 1.0)
    omega = torch.acos(cos_omega)                 # scalar tensor
    sin_omega = torch.sin(omega)                  # scalar tensor

    # --- Step 5: LERP fallback for near-parallel / near-antiparallel ---
    if sin_omega.item() < _SLERP_EPS:
        # Vectors are nearly co-linear; linear interpolation is numerically
        # identical to SLERP here and avoids 0/0.
        target_norm = ((1.0 - alpha) * norm_a + alpha * norm_b)
        result_unit = (1.0 - alpha) * unit_a + alpha * unit_b
        result_f    = result_unit * target_norm
        result      = result_f.reshape(original_shape).to(original_dtype)

        del unit_a, unit_b, norm_a, norm_b, cos_omega, omega, sin_omega
        del result_unit, result_f
        gc.collect()
        return result

    # --- Step 5 (normal path): SLERP coefficients ---
    coeff_a = torch.sin((1.0 - alpha) * omega) / sin_omega   # scalar
    coeff_b = torch.sin(alpha          * omega) / sin_omega   # scalar

    # Interpolated unit vector on the hypersphere
    interp_unit = coeff_a * unit_a + coeff_b * unit_b         # shape: [N]

    del unit_a, unit_b, coeff_a, coeff_b, cos_omega, omega, sin_omega
    gc.collect()

    # --- Step 6: rescale to geometric mean of original norms ---
    target_norm = torch.exp(
        (1.0 - alpha) * torch.log(norm_a) + alpha * torch.log(norm_b)
    )
    result_f = interp_unit * target_norm           # shape: [N]

    del interp_unit, norm_a, norm_b, target_norm
    gc.collect()

    # --- Step 7: reshape and cast ---
    result = result_f.reshape(original_shape).to(original_dtype)

    del result_f
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Algorithm 3: TIES-Merging (Trim, Elect, Select)
# ---------------------------------------------------------------------------

_TIES_DEFAULT_TRIM_FRACTION = 0.80   # discard bottom 80% by magnitude


def ties(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_base: torch.Tensor,
    alpha: float,
    trim_fraction: float = _TIES_DEFAULT_TRIM_FRACTION,
) -> torch.Tensor:
    """
    TIES-Merging: Trim, Elect, Select.

    Reference: "TIES-Merging: Resolving Interference When Merging Models"
    (Yadav et al., NeurIPS 2023).

    Steps
    -----
    1. **Task Vectors**  — δ_A = A − base,  δ_B = B − base
    2. **Trim**          — Zero out the bottom `trim_fraction` of each task
                           vector by absolute magnitude (per tensor, not global).
    3. **Elect**         — For each parameter position, take the sign with the
                           greater total magnitude across task vectors.
    4. **Select**        — Keep only parameters whose task-vector sign agrees
                           with the elected sign.  Average the keepers.
    5. **Reconstruct**   — merged = base + α * disjoint_average

    If tensor_base is None, this function raises ValueError.  Callers should
    pass tensor_a as the base when no explicit base is available (with a
    warning emitted by engine.py).

    Memory cost: up to ~7× layer size during computation.  Every intermediate
    tensor is explicitly deleted and gc.collect() called at each major phase.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not (0.0 <= trim_fraction < 1.0):
        raise ValueError(f"trim_fraction must be in [0, 1), got {trim_fraction}")
    if tensor_base is None:
        raise ValueError(
            "TIES-Merging requires a base tensor.  "
            "Pass tensor_a as base if no dedicated base model is available."
        )

    original_shape = tensor_a.shape
    original_dtype = tensor_a.dtype

    # Upcast everything for arithmetic.
    a_f    = _to_float32(tensor_a)
    b_f    = _to_float32(tensor_b)
    base_f = _to_float32(tensor_base)

    # -----------------------------------------------------------------------
    # Phase 1: Task vectors
    # -----------------------------------------------------------------------
    delta_a = a_f - base_f   # δ_A
    delta_b = b_f - base_f   # δ_B

    del a_f, b_f             # inputs no longer needed
    gc.collect()

    # -----------------------------------------------------------------------
    # Phase 2: Trim — zero out bottom trim_fraction by |magnitude|
    # -----------------------------------------------------------------------
    def _trim(delta: torch.Tensor) -> torch.Tensor:
        flat     = delta.flatten()
        abs_flat = flat.abs()
        k        = int(math.ceil(trim_fraction * flat.numel()))
        # torch.topk is memory-cheap for finding the threshold value.
        # We want the (1 - trim_fraction) fraction of LARGEST values.
        # Equivalently: zero everything below the k-th smallest.
        if k >= flat.numel():
            # Edge case: trim everything → return zeros
            result = torch.zeros_like(delta)
            del flat, abs_flat
            gc.collect()
            return result

        threshold_val, _ = torch.kthvalue(abs_flat, k)
        mask   = abs_flat >= threshold_val          # keep top (1 - trim_fraction)
        trimmed = (flat * mask).reshape(delta.shape)

        del flat, abs_flat, mask, threshold_val
        gc.collect()
        return trimmed

    delta_a_trimmed = _trim(delta_a)
    del delta_a
    gc.collect()

    delta_b_trimmed = _trim(delta_b)
    del delta_b
    gc.collect()

    # -----------------------------------------------------------------------
    # Phase 3: Elect — consensus sign via weighted magnitude vote
    # -----------------------------------------------------------------------
    # sign_score > 0 → elect positive; < 0 → elect negative; == 0 → abstain
    sign_score    = delta_a_trimmed + delta_b_trimmed    # shape: original_shape
    elected_sign  = torch.sign(sign_score)               # -1, 0, +1

    del sign_score
    gc.collect()

    # -----------------------------------------------------------------------
    # Phase 4: Select — keep only sign-consistent parameters, disjoint average
    # -----------------------------------------------------------------------
    # For each position, a task vector "agrees" with the elected sign iff
    # its own sign matches elected_sign (or elected_sign == 0 → skip).
    agree_a = (torch.sign(delta_a_trimmed) == elected_sign).float()
    agree_b = (torch.sign(delta_b_trimmed) == elected_sign).float()

    # Disjoint average: sum of sign-consistent deltas / number of agreeing vectors
    # Add small eps in denominator to avoid 0/0 at positions where nobody agrees.
    _EPS = 1e-9
    disjoint_sum   = delta_a_trimmed * agree_a + delta_b_trimmed * agree_b
    agreement_count = agree_a + agree_b + _EPS
    disjoint_avg    = disjoint_sum / agreement_count

    del delta_a_trimmed, delta_b_trimmed
    del agree_a, agree_b, elected_sign, disjoint_sum, agreement_count
    gc.collect()

    # -----------------------------------------------------------------------
    # Phase 5: Reconstruct — merged = base + α * disjoint_avg
    # -----------------------------------------------------------------------
    merged_f = base_f + alpha * disjoint_avg

    del base_f, disjoint_avg
    gc.collect()

    result = merged_f.to(original_dtype).reshape(original_shape)

    del merged_f
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def merge(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    alpha: float,
    algorithm: MergeAlgorithm | str,
    tensor_base: torch.Tensor | None = None,
    trim_fraction: float = _TIES_DEFAULT_TRIM_FRACTION,
) -> torch.Tensor:
    """
    Unified merge dispatcher.  Called by engine.py for each layer.

    Parameters
    ----------
    tensor_a, tensor_b : weight tensors for the two models.
    alpha              : interpolation ratio ∈ [0, 1].
    algorithm          : one of MergeAlgorithm.{LERP, SLERP, TIES}.
    tensor_base        : required for TIES; ignored otherwise.
    trim_fraction      : TIES-specific fraction to trim (default 0.80).

    Returns
    -------
    Merged tensor with the same dtype and shape as tensor_a.
    """
    alg = MergeAlgorithm(algorithm)

    if alg == MergeAlgorithm.LERP:
        return lerp(tensor_a, tensor_b, alpha)

    if alg == MergeAlgorithm.SLERP:
        return slerp(tensor_a, tensor_b, alpha)

    if alg == MergeAlgorithm.TIES:
        if tensor_base is None:
            raise ValueError(
                "TIES requires tensor_base.  "
                "engine.py should pass tensor_a as base if none is configured."
            )
        return ties(tensor_a, tensor_b, tensor_base, alpha, trim_fraction)

    raise NotImplementedError(f"Unknown algorithm: {algorithm}")  # unreachable


# ---------------------------------------------------------------------------
# Interference metric (used by engine.py before calling merge())
# ---------------------------------------------------------------------------

def cosine_similarity(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
) -> float:
    """
    Compute the cosine similarity between two weight tensors.

    Flattens both tensors to 1D, upcasts to float32, and computes:
        cos_sim = (A · B) / (||A|| * ||B||)

    Returns a Python float in [-1, 1].
    A score near 1.0 → weights are aligned (safe to merge).
    A score near 0.0 → orthogonal (moderate interference).
    A score near −1 → anti-aligned (severe interference).

    All intermediates are deleted before return.
    """
    a_f = _to_float32(tensor_a).flatten().unsqueeze(0)  # [1, N]
    b_f = _to_float32(tensor_b).flatten().unsqueeze(0)  # [1, N]

    sim = F.cosine_similarity(a_f, b_f, dim=1).item()   # Python float

    del a_f, b_f
    gc.collect()

    return float(sim)
