"""
Post-processing utilities for Rayleigh ellipticity curves.

Handles singularity clipping, peak extraction, and result packaging.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EllipticityResult:
    """Post-processed ellipticity result.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency array in Hz.
    amplitudes : np.ndarray
        Clipped / normalised H/V-proxy amplitudes.
    raw_amplitudes : np.ndarray
        Original amplitudes before clipping (may contain singularities).
    peaks : List[float]
        Detected peak frequencies in Hz (from sign changes).
    mode : int
        Rayleigh mode number used (0 = fundamental).
    all_modes : Dict[int, Tuple[np.ndarray, np.ndarray]]
        Raw per-mode curves from gpell.
    metadata : Dict
        Additional info (model path, config summary, etc.).
    """

    frequencies: np.ndarray
    amplitudes: np.ndarray
    raw_amplitudes: np.ndarray
    peaks: List[float]
    mode: int = 0
    all_modes: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


def detect_peaks_from_sign_change(
    freqs: np.ndarray,
    raw_amps: np.ndarray,
) -> List[float]:
    """Find peak frequencies where signed ellipticity crosses zero.

    At the resonance frequency, Rayleigh ellipticity passes through
    a singularity, so the sign flips.  The peak frequency is
    interpolated between the two bracketing samples.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    raw_amps : np.ndarray
        Signed (or absolute) amplitude array.

    Returns
    -------
    List[float]
        Detected peak frequencies.
    """
    if len(raw_amps) < 2:
        return []

    signs = np.sign(raw_amps)
    change_idx = np.where(np.diff(signs) != 0)[0]

    peaks = []
    for ci in change_idx:
        f1, f2 = freqs[ci], freqs[ci + 1]
        a1, a2 = abs(raw_amps[ci]), abs(raw_amps[ci + 1])
        if a1 + a2 > 0:
            w = a1 / (a1 + a2)
            f_peak = f1 + w * (f2 - f1)
        else:
            f_peak = 0.5 * (f1 + f2)
        peaks.append(float(f_peak))

    return peaks


def detect_peaks_from_maxima(
    freqs: np.ndarray,
    amps: np.ndarray,
    prominence_factor: float = 2.0,
) -> List[float]:
    """Find peak frequencies from local maxima in absolute amplitudes.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    amps : np.ndarray
        Absolute amplitude array.
    prominence_factor : float
        Minimum peak height relative to median amplitude.

    Returns
    -------
    List[float]
        Detected peak frequencies sorted by amplitude (descending).
    """
    abs_amps = np.abs(amps)
    median_amp = np.median(abs_amps)
    threshold = prominence_factor * median_amp

    peaks = []
    for i in range(1, len(abs_amps) - 1):
        if abs_amps[i] > abs_amps[i - 1] and abs_amps[i] > abs_amps[i + 1]:
            if abs_amps[i] > threshold:
                peaks.append((freqs[i], abs_amps[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in peaks]


def clip_amplitudes(
    amps: np.ndarray,
    clip_factor: float = 50.0,
) -> np.ndarray:
    """Clip singularity amplitudes to a reasonable maximum.

    Parameters
    ----------
    amps : np.ndarray
        Raw amplitudes (may contain very large values at singularities).
    clip_factor : float
        Clip at ``clip_factor * median(|amps|)``. Set to 0 to disable.

    Returns
    -------
    np.ndarray
        Clipped amplitudes.
    """
    if clip_factor <= 0:
        return amps.copy()

    abs_amps = np.abs(amps)
    positive = abs_amps[abs_amps > 0]
    median_val = np.median(positive) if len(positive) > 0 else 1.0
    clip_max = clip_factor * median_val

    return np.clip(abs_amps, 0, clip_max)


def postprocess_curve(
    freqs: np.ndarray,
    amps: np.ndarray,
    clip_factor: float = 50.0,
    all_modes: Optional[Dict] = None,
    model_path: str = "",
    config: object = None,
) -> EllipticityResult:
    """Full post-processing pipeline.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array from gpell.
    amps : np.ndarray
        Amplitude array from gpell (signed or absolute).
    clip_factor : float
        Singularity clipping factor.
    all_modes : Dict, optional
        All mode curves from gpell.
    model_path : str
        Source model file path.
    config : EllipticityConfig, optional
        Configuration used for this run.

    Returns
    -------
    EllipticityResult
    """
    raw = amps.copy()

    peaks = detect_peaks_from_sign_change(freqs, amps)
    if not peaks:
        peaks = detect_peaks_from_maxima(freqs, amps)

    clipped = clip_amplitudes(amps, clip_factor)

    metadata = {
        "model_path": model_path,
        "n_points": len(freqs),
        "n_peaks": len(peaks),
        "peak_frequencies_hz": peaks,
    }
    if config is not None:
        metadata["config"] = {
            "fmin": config.fmin,
            "fmax": config.fmax,
            "n_samples": config.n_samples,
            "n_modes": config.n_modes,
            "sampling": config.sampling,
            "absolute": config.absolute,
            "love_alpha": config.love_alpha,
            "auto_q": config.auto_q,
        }

    return EllipticityResult(
        frequencies=freqs,
        amplitudes=clipped,
        raw_amplitudes=raw,
        peaks=peaks,
        mode=0,
        all_modes=all_modes or {},
        metadata=metadata,
    )


__all__ = [
    "EllipticityResult",
    "postprocess_curve",
    "clip_amplitudes",
    "detect_peaks_from_sign_change",
    "detect_peaks_from_maxima",
]
