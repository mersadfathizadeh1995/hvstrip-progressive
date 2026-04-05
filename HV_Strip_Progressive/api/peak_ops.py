"""
Peak Operations — detection, presets, and manual peak management.

Wraps :mod:`core.peak_detection` with a clean API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.peak_detection import (
    detect_peak as core_detect_peak,
    find_all_peaks as core_find_all_peaks,
    get_peak_detection_preset as core_get_preset,
    PEAK_DETECTION_PRESETS,
)

from .config import PeakDetectionConfig, AutoPeakConfig, PEAK_PRESET_NAMES
from .forward_engine import PeakInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_peak(
    freqs: np.ndarray,
    amps: np.ndarray,
    config: Optional[PeakDetectionConfig] = None,
) -> PeakInfo:
    """Detect the primary peak on an HV curve.

    Parameters
    ----------
    freqs, amps : array-like
    config : PeakDetectionConfig, optional

    Returns
    -------
    PeakInfo
    """
    if config is None:
        config = PeakDetectionConfig()

    core_cfg = config.to_core_config()
    f0, a0, idx = core_detect_peak(
        np.asarray(freqs), np.asarray(amps), core_cfg
    )
    return PeakInfo(
        frequency=float(f0),
        amplitude=float(a0),
        index=int(idx),
        label="f0",
        source="auto",
    )


def detect_all_peaks(
    freqs: np.ndarray,
    amps: np.ndarray,
    config: Optional[PeakDetectionConfig] = None,
    max_peaks: int = 10,
) -> List[PeakInfo]:
    """Detect all peaks above the configured thresholds.

    Parameters
    ----------
    freqs, amps : array-like
    config : PeakDetectionConfig, optional
    max_peaks : int
        Maximum number of peaks to return.

    Returns
    -------
    list of PeakInfo
        Sorted by frequency (ascending).
    """
    if config is None:
        config = PeakDetectionConfig()

    all_raw = core_find_all_peaks(
        np.asarray(freqs),
        np.asarray(amps),
        prominence=config.prominence,
        distance=config.distance,
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        min_amplitude=config.min_amplitude,
    )

    # Sort by frequency
    all_raw.sort(key=lambda x: x[0])

    peaks: List[PeakInfo] = []
    for i, (f, a, idx) in enumerate(all_raw[:max_peaks]):
        peaks.append(PeakInfo(
            frequency=float(f),
            amplitude=float(a),
            index=int(idx),
            label=f"f{i}",
            source="auto",
        ))

    return peaks


def detect_peaks_with_ranges(
    freqs: np.ndarray,
    amps: np.ndarray,
    ranges: List[Tuple[float, float]],
    config: Optional[PeakDetectionConfig] = None,
) -> List[PeakInfo]:
    """Detect one peak per frequency range.

    This is the range-constrained mode used by the auto-peak GUI.

    Parameters
    ----------
    freqs, amps : array-like
    ranges : list of (fmin, fmax)
        Each tuple defines a frequency window to search in.
    config : PeakDetectionConfig, optional

    Returns
    -------
    list of PeakInfo
        One entry per range (may be empty if no peak found).
    """
    if config is None:
        config = PeakDetectionConfig()

    freqs = np.asarray(freqs)
    amps = np.asarray(amps)
    peaks: List[PeakInfo] = []

    for i, (fmin, fmax) in enumerate(ranges):
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not mask.any():
            continue

        sub_freqs = freqs[mask]
        sub_amps = amps[mask]
        sub_indices = np.where(mask)[0]

        try:
            local_cfg = PeakDetectionConfig(
                preset="custom",
                method=config.method,
                select=config.select,
                prominence=config.prominence,
                distance=config.distance,
                freq_min=fmin,
                freq_max=fmax,
                min_amplitude=config.min_amplitude,
                min_rel_height=config.min_rel_height,
            )
            f0, a0, local_idx = core_detect_peak(
                sub_freqs, sub_amps, local_cfg.to_core_config()
            )
            global_idx = int(sub_indices[local_idx])
            peaks.append(PeakInfo(
                frequency=float(f0),
                amplitude=float(a0),
                index=global_idx,
                label=f"f{i}",
                source="auto",
            ))
        except Exception:
            # No peak found in this range
            continue

    return peaks


def set_manual_peak(
    freqs: np.ndarray,
    amps: np.ndarray,
    target_frequency: float,
    label: str = "f0",
) -> PeakInfo:
    """Snap a manual frequency pick to the nearest curve point.

    Parameters
    ----------
    freqs, amps : array-like
    target_frequency : float
        User-selected frequency.
    label : str

    Returns
    -------
    PeakInfo
    """
    freqs = np.asarray(freqs)
    amps = np.asarray(amps)
    idx = int(np.argmin(np.abs(freqs - target_frequency)))
    return PeakInfo(
        frequency=float(freqs[idx]),
        amplitude=float(amps[idx]),
        index=idx,
        label=label,
        source="manual",
    )


# ---------------------------------------------------------------------------
# Preset management
# ---------------------------------------------------------------------------


def get_preset(name: str) -> Dict[str, Any]:
    """Return the full config dict for a named preset.

    Parameters
    ----------
    name : str
        One of ``"default"``, ``"forward_modeling"``, ``"conservative"``,
        ``"forward_modeling_sharp"``, ``"custom"``.

    Returns
    -------
    dict
    """
    return core_get_preset(name)


def list_presets() -> List[Dict[str, Any]]:
    """Return metadata for all available peak-detection presets."""
    result = []
    for name in PEAK_PRESET_NAMES:
        try:
            preset = core_get_preset(name)
            result.append({
                "name": name,
                "method": preset.get("method", ""),
                "select": preset.get("select", ""),
                "prominence": preset.get("find_peaks_params", {}).get(
                    "prominence", 0
                ),
                "min_amplitude": preset.get("min_amplitude"),
                "description": _preset_descriptions.get(name, ""),
            })
        except Exception:
            continue
    return result


_preset_descriptions = {
    "default": "General-purpose peak detection (prominence 0.2, leftmost peak)",
    "forward_modeling": (
        "Tuned for forward models — lower prominence (0.1), "
        "clarity ratio check, leftmost peak"
    ),
    "conservative": "High-confidence peaks only (prominence 0.5, max amplitude)",
    "forward_modeling_sharp": (
        "Forward models with sharpness bias — "
        "leftmost of high-prominence peaks"
    ),
    "custom": "Fully user-configurable preset",
}
