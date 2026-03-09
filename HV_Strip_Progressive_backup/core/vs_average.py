"""
Vs30 and Weighted-Average Vs Calculator
========================================

Computes time-averaged shear-wave velocity to a target depth (Vs30 by
default) or a thickness-weighted average over all finite layers.

Supports:
- ``SoilProfile`` objects
- Raw model files (HVf format: n_layers header + thickness Vp Vs density rows)
- Lists of (thickness, vs) pairs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class VsAverageResult:
    """Result container for Vs averaging calculations.

    Attributes
    ----------
    vs_avg : float
        Thickness-weighted harmonic mean over the requested depth.
    target_depth : float
        Depth to which the average was computed (metres).
    actual_depth : float
        Actual finite-layer depth available (metres).
    extrapolated : bool
        ``True`` when the half-space was used to fill remaining depth.
    layer_contributions : list[dict]
        Per-layer travel-time breakdown.
    """

    vs_avg: float
    target_depth: float
    actual_depth: float
    extrapolated: bool
    layer_contributions: List[dict]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_vs_average(
    layers: Sequence[Tuple[float, float]],
    *,
    target_depth: float = 30.0,
    use_halfspace: bool = True,
) -> VsAverageResult:
    """Compute time-averaged Vs to *target_depth* (Vs30 by default).

    The standard Vs30 formula (EN 1998-1 / NEHRP) is the *harmonic mean*
    weighted by travel time::

        VsZ = Z / Σ(hᵢ / Vsᵢ)

    Parameters
    ----------
    layers : sequence of (thickness, vs)
        Each element is ``(thickness_m, vs_m_s)``.
        A thickness of ``0`` marks the half-space (infinite layer).
    target_depth : float
        Depth to average over (default 30 m for Vs30).
    use_halfspace : bool
        If ``True`` and the finite layers do not reach *target_depth*,
        fill the remaining depth with the half-space Vs.

    Returns
    -------
    VsAverageResult
    """
    contributions: List[dict] = []
    remaining = target_depth
    actual_depth = 0.0
    hs_vs: Optional[float] = None
    total_travel_time = 0.0

    for thick, vs in layers:
        if vs <= 0:
            continue
        if thick <= 0:
            hs_vs = vs
            continue

        h_use = min(thick, remaining)
        tt = h_use / vs
        total_travel_time += tt
        actual_depth += h_use
        remaining -= h_use
        contributions.append({
            "thickness": h_use,
            "vs": vs,
            "travel_time": tt,
        })
        if remaining <= 0:
            break

    extrapolated = False
    if remaining > 0 and use_halfspace and hs_vs is not None and hs_vs > 0:
        tt = remaining / hs_vs
        total_travel_time += tt
        contributions.append({
            "thickness": remaining,
            "vs": hs_vs,
            "travel_time": tt,
            "extrapolated": True,
        })
        remaining = 0.0
        extrapolated = True

    depth_used = target_depth - remaining
    vs_avg = depth_used / total_travel_time if total_travel_time > 0 else 0.0

    return VsAverageResult(
        vs_avg=round(vs_avg, 2),
        target_depth=target_depth,
        actual_depth=actual_depth,
        extrapolated=extrapolated,
        layer_contributions=contributions,
    )


def compute_vs_weighted(
    layers: Sequence[Tuple[float, float]],
) -> float:
    """Thickness-weighted arithmetic mean of Vs over finite layers only.

    Parameters
    ----------
    layers : sequence of (thickness, vs)

    Returns
    -------
    float
        Weighted arithmetic mean Vs (m/s).  Returns 0 if no finite layers.
    """
    total_h = 0.0
    weighted_sum = 0.0
    for thick, vs in layers:
        if thick <= 0 or vs <= 0:
            continue
        total_h += thick
        weighted_sum += thick * vs
    return round(weighted_sum / total_h, 2) if total_h > 0 else 0.0


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def vs_average_from_model_file(
    path: Union[str, Path],
    *,
    target_depth: float = 30.0,
    use_halfspace: bool = True,
) -> VsAverageResult:
    """Load an HVf model file and compute Vs average.

    Parameters
    ----------
    path : str or Path
        Path to model file (n header + thickness Vp Vs density rows).
    target_depth : float
        Depth for averaging.
    use_halfspace : bool
        Whether to extrapolate into the half-space.

    Returns
    -------
    VsAverageResult
    """
    layers = _parse_model_file(path)
    return compute_vs_average(layers, target_depth=target_depth,
                              use_halfspace=use_halfspace)


def vs_average_from_profile(
    profile,
    *,
    target_depth: float = 30.0,
    use_halfspace: bool = True,
) -> VsAverageResult:
    """Compute Vs average from a ``SoilProfile`` instance.

    Parameters
    ----------
    profile : SoilProfile
        Soil profile object.
    target_depth : float
        Depth for averaging.
    use_halfspace : bool
        Whether to extrapolate into the half-space.

    Returns
    -------
    VsAverageResult
    """
    layers = [(l.thickness, l.vs) for l in profile.layers]
    return compute_vs_average(layers, target_depth=target_depth,
                              use_halfspace=use_halfspace)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_model_file(path: Union[str, Path]) -> List[Tuple[float, float]]:
    """Read (thickness, vs) pairs from an HVf model file."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    try:
        n = int(lines[0])
    except ValueError:
        return []
    result: List[Tuple[float, float]] = []
    for line in lines[1: n + 1]:
        parts = line.split()
        if len(parts) < 3:
            continue
        thick = float(parts[0])
        vs = float(parts[2])
        result.append((thick, vs))
    return result


__all__ = [
    "VsAverageResult",
    "compute_vs_average",
    "compute_vs_weighted",
    "vs_average_from_model_file",
    "vs_average_from_profile",
]
