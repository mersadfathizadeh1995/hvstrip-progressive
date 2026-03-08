"""
postprocess.py
==============
Post-processing utilities for SH transfer function curves.

Handles F0 peak detection, singularity clipping, result packaging,
and the Convergence Index (CI) used in multi-method HVSR comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SHTFResult:
    """Post-processed SH transfer function result.

    Attributes
    ----------
    frequencies : np.ndarray, shape (F,)
        Frequency vector [Hz].
    amplitudes : np.ndarray, shape (F, M)
        TF amplitude at each frequency for each profile (M profiles).
        If ``clip_tf > 0`` was set, singularities are clipped.
    f0 : list[float]
        Fundamental resonance frequency [Hz] for each profile.
        Detected as the peak of the TF within the search window.
    peaks : list[list[float]]
        All detected TF peaks [Hz] per profile (sorted by amplitude).
    convergence_index : float or None
        Convergence Index comparing this engine's F0 to other methods.
        Populated externally by :func:`compute_convergence_index`.
    model : SHTFModel
        The input layered earth model.
    config : SHTFConfig
        Configuration used for this run.
    metadata : dict
        Auxiliary information (source, n_layers, etc.).
    """

    frequencies:        np.ndarray
    amplitudes:         np.ndarray
    f0:                 List[float]
    peaks:              List[List[float]]
    convergence_index:  Optional[float]
    model:              object   # SHTFModel (avoid circular import)
    config:             object   # SHTFConfig
    metadata:           Dict     = field(default_factory=dict)


# ---------------------------------------------------------------------------
# F0 / peak detection
# ---------------------------------------------------------------------------

def detect_peaks(
    freq: np.ndarray,
    tf:   np.ndarray,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> List[float]:
    """Detect TF peak frequencies within a search window.

    Finds all local maxima, then returns them sorted by amplitude
    (highest first).  The first entry is the fundamental frequency F0.

    Parameters
    ----------
    freq : np.ndarray, shape (F,)
        Frequency vector [Hz].
    tf : np.ndarray, shape (F,)
        TF amplitude for one profile.
    fmin, fmax : float or None
        Search window [Hz]. Defaults to full freq range.

    Returns
    -------
    list[float]
        Peak frequencies [Hz], sorted by amplitude (descending).
    """
    lo = fmin if fmin is not None else freq[0]
    hi = fmax if fmax is not None else freq[-1]
    mask = (freq >= lo) & (freq <= hi)

    f_win  = freq[mask]
    tf_win = tf[mask]

    if len(tf_win) < 3:
        return [float(f_win[tf_win.argmax()])] if len(tf_win) > 0 else []

    peaks = []
    for i in range(1, len(tf_win) - 1):
        if tf_win[i] > tf_win[i - 1] and tf_win[i] > tf_win[i + 1]:
            # Quadratic refinement between neighbours
            a, b, c = tf_win[i - 1], tf_win[i], tf_win[i + 1]
            denom = a - 2*b + c
            if denom != 0:
                delta = 0.5 * (a - c) / denom
                delta = max(-0.5, min(0.5, delta))
            else:
                delta = 0.0
            # Linear interpolation in freq space
            df = f_win[i] - f_win[i - 1] if i > 0 else 0.0
            f_peak = f_win[i] + delta * df
            peaks.append((float(f_peak), float(b)))

    if not peaks:
        # Fallback: global maximum
        idx = tf_win.argmax()
        return [float(f_win[idx])]

    peaks.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in peaks]


def detect_f0(
    freq: np.ndarray,
    tf:   np.ndarray,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> float:
    """Return the fundamental frequency F0 (highest peak within window).

    Parameters
    ----------
    freq : np.ndarray, shape (F,)
    tf   : np.ndarray, shape (F,)
    fmin, fmax : float or None
        Search window [Hz].

    Returns
    -------
    float
        F0 in Hz.
    """
    pk = detect_peaks(freq, tf, fmin=fmin, fmax=fmax)
    return pk[0] if pk else float("nan")


# ---------------------------------------------------------------------------
# Convergence Index
# ---------------------------------------------------------------------------

def compute_convergence_index(
    *f0_values: float,
) -> float:
    """Compute the Convergence Index (CI) for F0 estimates from N methods.

    CI = 1 - (max(F0) - min(F0)) / mean(F0)

    Range:
      1.0  → perfect agreement across all methods
      0.0  → total disagreement (spread equals the mean)
      < 0  → extreme disagreement (spread exceeds the mean)

    Parameters
    ----------
    *f0_values : float
        F0 estimates [Hz] from each method. Pass at least two values.
        NaN values are ignored.

    Returns
    -------
    float
        Convergence Index.

    Examples
    --------
    >>> compute_convergence_index(9.30, 9.15, 9.40)
    0.9730...
    """
    vals = np.array([v for v in f0_values if not np.isnan(v)], dtype=float)
    if len(vals) < 2:
        raise ValueError("Need at least two non-NaN F0 estimates to compute CI")
    mean_f0 = vals.mean()
    if mean_f0 == 0:
        raise ValueError("Mean F0 is zero — CI is undefined")
    return float(1.0 - (vals.max() - vals.min()) / mean_f0)


# ---------------------------------------------------------------------------
# Amplitude clipping
# ---------------------------------------------------------------------------

def clip_amplitudes(
    tf: np.ndarray,
    clip_tf: float = 0.0,
) -> np.ndarray:
    """Clip TF amplitudes above ``clip_tf``.

    Parameters
    ----------
    tf : np.ndarray
        TF amplitude array.
    clip_tf : float
        Hard maximum. 0 = disabled.

    Returns
    -------
    np.ndarray
        Clipped amplitudes.
    """
    if clip_tf <= 0:
        return tf.copy()
    return np.minimum(tf, clip_tf)


# ---------------------------------------------------------------------------
# Full post-processing pipeline
# ---------------------------------------------------------------------------

def postprocess_tf(
    freq:       np.ndarray,
    tf:         np.ndarray,
    model:      object,
    config:     object,
) -> SHTFResult:
    """Package raw TF arrays into an ``SHTFResult``.

    Parameters
    ----------
    freq : np.ndarray, shape (F,)
        Frequency vector [Hz].
    tf : np.ndarray, shape (F, M)
        Raw TF amplitudes.
    model : SHTFModel
        Input model.
    config : SHTFConfig
        Run configuration.

    Returns
    -------
    SHTFResult
    """
    # Clip if requested
    tf_out = clip_amplitudes(tf, clip_tf=config.clip_tf)

    lo = config.f0_search_fmin
    hi = config.f0_search_fmax

    n_profiles = tf_out.shape[1] if tf_out.ndim > 1 else 1
    tf_2d = tf_out if tf_out.ndim > 1 else tf_out[:, np.newaxis]

    f0_list:   List[float]       = []
    peak_list: List[List[float]] = []

    for k in range(n_profiles):
        pks = detect_peaks(freq, tf_2d[:, k], fmin=lo, fmax=hi)
        peak_list.append(pks)
        f0_list.append(pks[0] if pks else float("nan"))

    metadata = {
        "source":   getattr(model, "source", ""),
        "n_layers": getattr(model, "n_layers", None),
        "n_profiles": n_profiles,
        "f0_hz":    f0_list,
        "fmin":     float(freq[0]),
        "fmax":     float(freq[-1]),
        "n_freq":   len(freq),
    }

    return SHTFResult(
        frequencies=freq,
        amplitudes=tf_out,
        f0=f0_list,
        peaks=peak_list,
        convergence_index=None,
        model=model,
        config=config,
        metadata=metadata,
    )


__all__ = [
    "SHTFResult",
    "detect_f0",
    "detect_peaks",
    "clip_amplitudes",
    "postprocess_tf",
    "compute_convergence_index",
]
