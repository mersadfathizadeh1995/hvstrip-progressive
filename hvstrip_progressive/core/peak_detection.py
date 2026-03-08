"""
Peak detection module for HVSR analysis.

Provides configurable peak detection with multiple presets and methods:
- find_peaks: scipy-based local peak detection with prominence filtering
- max: global maximum amplitude selection
- manual: user-specified frequency

Supports frequency windowing, minimum amplitude, relative height,
clarity ratio checks, and multiple selection strategies (leftmost,
sharpest, leftmost_sharpest, max).
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences


PEAK_DETECTION_PRESETS = {
    "default": {
        "method": "find_peaks",
        "select": "leftmost",
        "find_peaks_params": {"prominence": 0.2, "distance": 3},
        "freq_min": 0.5,
        "freq_max": None,
        "min_rel_height": 0.25,
        "exclude_first_n": 1,
        "min_amplitude": None,
    },
    "forward_modeling": {
        "method": "find_peaks",
        "select": "leftmost",
        "find_peaks_params": {"prominence": 0.1, "distance": 2},
        "freq_min": 0.3,
        "freq_max": None,
        "min_rel_height": 0.15,
        "exclude_first_n": 1,
        "min_amplitude": 1.5,
        "check_clarity_ratio": True,
        "clarity_ratio_threshold": 1.5,
    },
    "conservative": {
        "method": "find_peaks",
        "select": "max",
        "find_peaks_params": {"prominence": 0.5, "distance": 5},
        "freq_min": 0.5,
        "freq_max": None,
        "min_rel_height": 0.3,
        "exclude_first_n": 1,
        "min_amplitude": 2.0,
    },
    "forward_modeling_sharp": {
        "method": "find_peaks",
        "select": "leftmost_sharpest",
        "find_peaks_params": {"prominence": 0.1, "distance": 2},
        "freq_min": 0.3,
        "freq_max": None,
        "min_rel_height": 0.15,
        "exclude_first_n": 1,
        "min_amplitude": 1.5,
    },
}


def get_peak_detection_preset(preset_name: str) -> Dict:
    """
    Get peak detection configuration by preset name.

    Parameters
    ----------
    preset_name : str
        One of 'default', 'forward_modeling', 'conservative',
        'forward_modeling_sharp', or 'custom'.

    Returns
    -------
    Dict
        Peak detection configuration dictionary.
    """
    if preset_name == "custom":
        return {
            "method": "find_peaks",
            "select": "leftmost",
            "find_peaks_params": {"prominence": 0.2, "distance": 3},
            "freq_min": None,
            "freq_max": None,
            "min_rel_height": 0.0,
            "exclude_first_n": 0,
            "min_amplitude": None,
        }
    return PEAK_DETECTION_PRESETS.get(
        preset_name, PEAK_DETECTION_PRESETS["default"]
    ).copy()


def _apply_freq_window(
    idxs: np.ndarray,
    freqs: np.ndarray,
    fmin: float = None,
    fmax: float = None,
) -> np.ndarray:
    """Filter peak indices by frequency bounds."""
    if idxs is None or len(idxs) == 0:
        return idxs
    mask = np.ones(len(idxs), dtype=bool)
    if fmin is not None:
        mask &= freqs[idxs] >= float(fmin)
    if fmax is not None:
        mask &= freqs[idxs] <= float(fmax)
    return idxs[mask]


def _apply_min_amplitude(
    idxs: np.ndarray,
    amps: np.ndarray,
    min_amp: float = None,
) -> np.ndarray:
    """Filter peak indices by minimum absolute amplitude."""
    if min_amp is None or idxs is None or len(idxs) == 0:
        return idxs
    return idxs[amps[idxs] >= float(min_amp)]


def _check_clarity(
    freqs: np.ndarray,
    amps: np.ndarray,
    peak_idx: int,
    clarity_threshold: float = 1.5,
) -> bool:
    """Check if peak amplitude exceeds threshold * amplitude at f0/2 and 2*f0."""
    peak_freq = freqs[peak_idx]
    peak_amp = amps[peak_idx]
    half_freq = peak_freq / 2.0
    double_freq = peak_freq * 2.0

    if half_freq >= freqs[0]:
        idx_half = np.argmin(np.abs(freqs - half_freq))
        if peak_amp < clarity_threshold * amps[idx_half]:
            return False

    if double_freq <= freqs[-1]:
        idx_double = np.argmin(np.abs(freqs - double_freq))
        if peak_amp < clarity_threshold * amps[idx_double]:
            return False

    return True


def _select_peak(
    peaks: np.ndarray,
    freqs: np.ndarray,
    amps: np.ndarray,
    select: str,
) -> int:
    """
    Choose one peak index from candidates using the given strategy.

    Parameters
    ----------
    peaks : np.ndarray
        Candidate peak indices.
    freqs : np.ndarray
        Full frequency array.
    amps : np.ndarray
        Full amplitude array.
    select : str
        Strategy: 'leftmost', 'sharpest', 'leftmost_sharpest', or 'max'.

    Returns
    -------
    int
        Selected peak index into freqs/amps.
    """
    select_lower = str(select).lower()

    if select_lower == "leftmost":
        return int(peaks[np.argmin(freqs[peaks])])

    if select_lower == "sharpest":
        proms, _, _ = peak_prominences(amps, peaks)
        return int(peaks[np.argmax(proms)])

    if select_lower == "leftmost_sharpest":
        proms, _, _ = peak_prominences(amps, peaks)
        max_prom = np.max(proms) if len(proms) > 0 else 0
        sharp_peaks = peaks[proms >= max_prom * 0.5]
        if len(sharp_peaks) > 0:
            return int(sharp_peaks[np.argmin(freqs[sharp_peaks])])
        return int(peaks[np.argmin(freqs[peaks])])

    # 'max' or fallback: highest amplitude
    return int(peaks[np.argmax(amps[peaks])])


def detect_peak(
    freqs: np.ndarray,
    amps: np.ndarray,
    config: Dict,
) -> Tuple[float, float, int]:
    """
    Detect the primary peak in an HVSR curve.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    amps : np.ndarray
        Amplitude array.
    config : Dict
        Configuration dict; reads ``config['peak_detection']``.

    Returns
    -------
    Tuple[float, float, int]
        (peak_frequency, peak_amplitude, peak_index)
    """
    peak_cfg = config.get("peak_detection", {})

    # Resolve preset
    preset = peak_cfg.get("preset", None)
    if preset and preset != "custom":
        resolved = get_peak_detection_preset(preset)
        for key, val in peak_cfg.items():
            if key != "preset" and val is not None:
                resolved[key] = val
        peak_cfg = resolved

    method = peak_cfg.get("method", "max")
    select = peak_cfg.get("select", "max")
    fmin = peak_cfg.get("freq_min", None)
    fmax = peak_cfg.get("freq_max", None)
    min_rel = float(peak_cfg.get("min_rel_height", 0.0) or 0.0)
    excl_n = int(peak_cfg.get("exclude_first_n", 0) or 0)
    min_amp = peak_cfg.get("min_amplitude", None)
    check_clar = peak_cfg.get("check_clarity_ratio", False)
    clarity_thr = float(peak_cfg.get("clarity_ratio_threshold", 1.5) or 1.5)

    # --- Manual mode ---
    if method == "manual" and peak_cfg.get("manual_frequency"):
        f_manual = float(peak_cfg["manual_frequency"])
        idx = int(np.argmin(np.abs(freqs - f_manual)))
        return float(freqs[idx]), float(amps[idx]), idx

    # --- find_peaks mode ---
    if method == "find_peaks":
        params = peak_cfg.get("find_peaks_params", {})
        peaks, _ = find_peaks(amps, **params)
        peaks = np.array(peaks, dtype=int)

        if excl_n > 0 and peaks.size > 0:
            peaks = peaks[peaks >= excl_n]

        peaks = _apply_freq_window(peaks, freqs, fmin, fmax)

        if min_rel > 0 and peaks is not None and len(peaks) > 0:
            amax = float(np.max(amps)) if len(amps) else 0.0
            peaks = peaks[amps[peaks] >= amax * min_rel]

        peaks = _apply_min_amplitude(peaks, amps, min_amp)

        if check_clar and peaks is not None and len(peaks) > 0:
            valid = [p for p in peaks if _check_clarity(freqs, amps, p, clarity_thr)]
            peaks = np.array(valid, dtype=int) if valid else np.array([], dtype=int)

        if peaks is not None and len(peaks) > 0:
            idx = _select_peak(peaks, freqs, amps, select)
            return float(freqs[idx]), float(amps[idx]), idx
        # Fallback to global max below

    # --- Global max mode (also fallback) ---
    if fmin is not None or fmax is not None or excl_n > 0 or min_rel > 0:
        mask = np.ones_like(freqs, dtype=bool)
        if fmin is not None:
            mask &= freqs >= float(fmin)
        if fmax is not None:
            mask &= freqs <= float(fmax)
        if excl_n > 0:
            mask &= np.arange(len(freqs)) >= excl_n
        cand = np.where(mask)[0]
        if cand.size > 0:
            if min_rel > 0:
                amax = float(np.max(amps)) if len(amps) else 0.0
                cand2 = cand[amps[cand] >= amax * min_rel]
                cand = cand2 if cand2.size > 0 else cand
            cand = _apply_min_amplitude(cand, amps, min_amp)
            if cand is not None and len(cand) > 0:
                idx_local = int(cand[np.argmax(amps[cand])])
                return float(freqs[idx_local]), float(amps[idx_local]), idx_local

    idx = int(np.argmax(amps))
    return float(freqs[idx]), float(amps[idx]), idx


def find_all_peaks(
    freqs: np.ndarray,
    amps: np.ndarray,
    prominence: float = 0.1,
    distance: int = 2,
    freq_min: float = None,
    freq_max: float = None,
) -> List[Tuple[float, float, int]]:
    """
    Find all significant peaks in an HVSR curve.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    amps : np.ndarray
        Amplitude array.
    prominence : float
        Minimum peak prominence.
    distance : int
        Minimum sample distance between peaks.
    freq_min : float, optional
        Minimum frequency bound.
    freq_max : float, optional
        Maximum frequency bound.

    Returns
    -------
    List[Tuple[float, float, int]]
        List of (frequency, amplitude, index) for each peak,
        sorted by frequency ascending.
    """
    peaks, _ = find_peaks(amps, prominence=prominence, distance=distance)
    peaks = _apply_freq_window(np.array(peaks, dtype=int), freqs, freq_min, freq_max)

    if peaks is None or len(peaks) == 0:
        return []

    result = [(float(freqs[p]), float(amps[p]), int(p)) for p in peaks]
    result.sort(key=lambda x: x[0])
    return result


__all__ = [
    "PEAK_DETECTION_PRESETS",
    "get_peak_detection_preset",
    "detect_peak",
    "find_all_peaks",
]
