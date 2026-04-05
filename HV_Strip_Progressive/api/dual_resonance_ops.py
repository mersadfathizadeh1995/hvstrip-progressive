"""
Dual Resonance Operations — f0/f1 extraction from stripping results.

Wraps :mod:`core.dual_resonance` with a thin API façade.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.dual_resonance import (
    extract_dual_resonance as core_extract,
    theoretical_frequency,
    compute_batch_statistics as core_batch_stats,
    DualResonanceResult as CoreDualResult,
    BatchDualResonanceStats,
)

from .config import DualResonanceConfig, PeakDetectionConfig

logger = logging.getLogger(__name__)


def extract_dual_resonance(
    strip_dir: str,
    config: Optional[DualResonanceConfig] = None,
    peak_config: Optional[PeakDetectionConfig] = None,
    profile_name: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract f0 (deep) and f1 (shallow) resonance frequencies.

    Parameters
    ----------
    strip_dir : str
        Path to the stripping output directory containing step folders.
    config : DualResonanceConfig, optional
    peak_config : PeakDetectionConfig, optional
    profile_name, profile_path : str, optional
        Metadata passed through to the result.

    Returns
    -------
    dict
        Serialised :class:`DualResonanceResult` with keys ``f0``, ``f1``,
        ``freq_ratio``, ``separation_success``, etc.
    """
    if config is None:
        config = DualResonanceConfig()

    peak_cfg = peak_config.to_core_config() if peak_config else None

    result: CoreDualResult = core_extract(
        strip_dir=strip_dir,
        peak_config=peak_cfg,
        profile_name=profile_name or "",
        profile_path=profile_path or "",
    )

    return _dual_result_to_dict(result)


def compute_theoretical_frequencies(
    layers: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute theoretical f0 and f1 from layer properties.

    Parameters
    ----------
    layers : list of dict
        Each dict has ``thickness`` and ``vs``.

    Returns
    -------
    dict
        ``{"f0": float, "f1": float, "ratio": float}``
    """
    layer_tuples = [
        (l["thickness"], l["vs"]) for l in layers
    ]
    f0, f1 = theoretical_frequency(layer_tuples)
    ratio = f1 / f0 if f0 > 0 else 0.0
    return {"f0": f0, "f1": f1, "ratio": ratio}


def validate_dual_resonance(
    measured: Dict[str, float],
    theoretical: Dict[str, float],
) -> Dict[str, Any]:
    """Compare measured dual-resonance against theoretical prediction.

    Parameters
    ----------
    measured : dict
        ``{"f0": float, "f1": float}``
    theoretical : dict
        ``{"f0": float, "f1": float}``

    Returns
    -------
    dict
        Agreement metrics.
    """
    m_f0 = measured.get("f0", 0)
    m_f1 = measured.get("f1", 0)
    t_f0 = theoretical.get("f0", 0)
    t_f1 = theoretical.get("f1", 0)

    f0_error = abs(m_f0 - t_f0) / t_f0 if t_f0 else float("inf")
    f1_error = abs(m_f1 - t_f1) / t_f1 if t_f1 else float("inf")

    return {
        "f0_measured": m_f0,
        "f0_theoretical": t_f0,
        "f0_relative_error": round(f0_error, 4),
        "f1_measured": m_f1,
        "f1_theoretical": t_f1,
        "f1_relative_error": round(f1_error, 4),
        "agreement": "good" if max(f0_error, f1_error) < 0.15 else "poor",
    }


def compute_batch_dual_statistics(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate dual-resonance statistics across profiles.

    Parameters
    ----------
    results : list of dict
        Each from :func:`extract_dual_resonance`.

    Returns
    -------
    dict
        Aggregated statistics.
    """
    # Reconstruct core objects for the core stats function
    core_results = []
    for r in results:
        cr = CoreDualResult(
            profile_name=r.get("profile_name", ""),
            profile_path=r.get("profile_path", ""),
            success=r.get("success", False),
            f0=r.get("f0", 0),
            a0=r.get("a0", 0),
            f0_theoretical=r.get("f0_theoretical", 0),
            f1=r.get("f1", 0),
            a1=r.get("a1", 0),
            f1_theoretical=r.get("f1_theoretical", 0),
            freq_ratio=r.get("freq_ratio", 0),
            max_freq_shift=r.get("max_freq_shift", 0),
            separation_success=r.get("separation_success", False),
        )
        core_results.append(cr)

    stats: BatchDualResonanceStats = core_batch_stats(core_results)

    return {
        "n_profiles": stats.n_profiles,
        "n_successful": stats.n_successful,
        "success_rate": stats.success_rate,
        "f0_mean": stats.f0_mean,
        "f0_std": stats.f0_std,
        "f1_mean": stats.f1_mean,
        "f1_std": stats.f1_std,
        "freq_ratio_mean": stats.freq_ratio_mean,
        "freq_ratio_std": stats.freq_ratio_std,
        "separation_success_rate": stats.separation_success_rate,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dual_result_to_dict(result: CoreDualResult) -> Dict[str, Any]:
    """Convert a core DualResonanceResult to a JSON-safe dict."""
    return {
        "profile_name": result.profile_name,
        "profile_path": result.profile_path,
        "success": result.success,
        "n_layers": result.n_layers,
        "total_depth": result.total_depth,
        "f0": result.f0,
        "a0": result.a0,
        "f0_theoretical": result.f0_theoretical,
        "f1": result.f1,
        "a1": result.a1,
        "f1_theoretical": result.f1_theoretical,
        "freq_ratio": result.freq_ratio,
        "max_freq_shift": result.max_freq_shift,
        "controlling_step": result.controlling_step,
        "separation_success": result.separation_success,
        "error_message": result.error_message,
        "freq_per_step": list(result.freq_per_step),
        "amp_per_step": list(result.amp_per_step),
    }
