"""
Dual-resonance (two-resonance) analysis for HVSR progressive stripping.

Extracts deep (f0) and shallow (f1) resonance frequencies from stripping
step results.  Works for both single-profile and batch workflows.

Typical usage::

    from hvstrip_progressive.core.dual_resonance import (
        extract_dual_resonance,
        compute_batch_statistics,
    )

    dr = extract_dual_resonance(strip_dir)
    stats = compute_batch_statistics([dr1, dr2, ...])
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hv_postprocess import DEFAULT_CONFIG, detect_peak, read_hv_csv, read_model


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DualResonanceResult:
    """Result of dual-resonance extraction for one profile."""

    profile_name: str
    profile_path: str
    success: bool

    # Layer geometry
    n_layers: int = 0
    total_depth: float = 0.0
    layer_thicknesses: List[float] = field(default_factory=list)
    layer_vs: List[float] = field(default_factory=list)

    # Deep resonance (original full model — Step 0)
    f0: float = 0.0
    a0: float = 0.0
    f0_theoretical: float = 0.0

    # Shallow resonance (after removing deepest layer — Step 1)
    f1: float = 0.0
    a1: float = 0.0
    f1_theoretical: float = 0.0

    # Per-step peak tracking
    freq_per_step: List[float] = field(default_factory=list)
    amp_per_step: List[float] = field(default_factory=list)

    # Derived metrics
    freq_ratio: float = 0.0       # f1 / f0
    max_freq_shift: float = 0.0   # largest |Δf| between consecutive steps
    controlling_step: int = 0     # step index with largest shift
    separation_success: bool = False

    error_message: str = ""


@dataclass
class BatchDualResonanceStats:
    """Aggregate statistics from a batch of dual-resonance analyses."""

    n_profiles: int = 0
    n_successful: int = 0
    success_rate: float = 0.0

    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0

    f1_mean: float = 0.0
    f1_std: float = 0.0
    f1_min: float = 0.0
    f1_max: float = 0.0

    freq_ratio_mean: float = 0.0
    freq_ratio_std: float = 0.0

    f0_theoretical_correlation: float = 0.0
    f1_theoretical_correlation: float = 0.0

    separation_success_rate: float = 0.0

    mean_freq_shift_per_step: List[float] = field(default_factory=list)
    std_freq_shift_per_step: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Theoretical helpers
# ---------------------------------------------------------------------------

def theoretical_frequency(layers: List[Dict]) -> Tuple[float, float]:
    """Compute theoretical f0 (full model) and f1 (top layer only).

    Uses the quarter-wavelength approximation ``f = Vs_avg / (4 * H)``.

    Parameters
    ----------
    layers : list of dict
        Each dict must have ``thickness`` and ``vs`` keys.

    Returns
    -------
    (f0_full, f1_shallow)
    """
    finite = [l for l in layers if l["thickness"] > 0]
    total_h = sum(l["thickness"] for l in finite)
    if total_h <= 0:
        return 0.0, 0.0

    travel_time = sum(l["thickness"] / l["vs"] for l in finite if l["vs"] > 0)
    vs_avg = total_h / travel_time if travel_time > 0 else 0.0
    f0 = vs_avg / (4.0 * total_h)

    f1 = 0.0
    if finite and finite[0]["thickness"] > 0 and finite[0]["vs"] > 0:
        f1 = finite[0]["vs"] / (4.0 * finite[0]["thickness"])

    return f0, f1


# ---------------------------------------------------------------------------
# Single-profile extraction
# ---------------------------------------------------------------------------

SEPARATION_RATIO_THRESHOLD = 1.2
SEPARATION_SHIFT_THRESHOLD = 0.3  # Hz


def extract_dual_resonance(
    strip_dir: str,
    peak_config: Optional[Dict] = None,
    profile_name: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> DualResonanceResult:
    """Extract f0/f1 from an already-computed stripping output folder.

    Parameters
    ----------
    strip_dir : str
        Path to the ``strip/`` directory that contains ``Step*_*-layer/``
        sub-folders, each holding ``model_*.txt`` and ``hv_curve.csv``.
    peak_config : dict, optional
        Peak-detection config passed to ``detect_peak``.
    profile_name : str, optional
        Human-readable label (defaults to folder name).
    profile_path : str, optional
        Original model file path for bookkeeping.

    Returns
    -------
    DualResonanceResult
    """
    strip_path = Path(strip_dir)
    if profile_name is None:
        profile_name = strip_path.parent.name
    if profile_path is None:
        profile_path = str(strip_path)

    cfg = peak_config or DEFAULT_CONFIG

    fail = DualResonanceResult(
        profile_name=profile_name,
        profile_path=profile_path,
        success=False,
    )

    # Discover step folders
    step_folders = sorted(
        (d for d in strip_path.iterdir()
         if d.is_dir() and d.name.startswith("Step") and "-layer" in d.name),
        key=lambda p: int(p.name.split("_")[0].replace("Step", "")),
    )
    if not step_folders:
        fail.error_message = "No step folders found"
        return fail

    # Read original model from Step 0
    step0 = step_folders[0]
    model_files = list(step0.glob("model_*.txt"))
    if not model_files:
        fail.error_message = f"No model file in {step0.name}"
        return fail

    model = read_model(model_files[0])
    layers = model["layers"]
    thicknesses = [l["thickness"] for l in layers]
    vs_values = [l["vs"] for l in layers]
    total_depth = sum(t for t in thicknesses if t > 0)

    f0_theo, f1_theo = theoretical_frequency(layers)

    # Collect per-step peak frequencies
    freq_per_step: List[float] = []
    amp_per_step: List[float] = []

    for sf in step_folders:
        hv_csv = sf / "hv_curve.csv"
        if not hv_csv.exists():
            continue
        freqs, amps = read_hv_csv(hv_csv)
        f_peak, a_peak, _ = detect_peak(freqs, amps, cfg)
        freq_per_step.append(f_peak)
        amp_per_step.append(a_peak)

    if len(freq_per_step) < 2:
        fail.error_message = "Not enough steps for dual-resonance analysis"
        return fail

    f0 = freq_per_step[0]
    a0 = amp_per_step[0]
    f1 = freq_per_step[1]
    a1 = amp_per_step[1]

    # Step-to-step frequency shifts
    shifts = [
        abs(freq_per_step[i] - freq_per_step[i - 1])
        for i in range(1, len(freq_per_step))
    ]
    max_shift = max(shifts) if shifts else 0.0
    ctrl_step = (shifts.index(max_shift) + 1) if shifts else 0

    ratio = f1 / f0 if f0 > 0 else 0.0
    sep_ok = (ratio > SEPARATION_RATIO_THRESHOLD
              and max_shift > SEPARATION_SHIFT_THRESHOLD)

    return DualResonanceResult(
        profile_name=profile_name,
        profile_path=profile_path,
        success=True,
        n_layers=len(layers),
        total_depth=total_depth,
        layer_thicknesses=thicknesses,
        layer_vs=vs_values,
        f0=f0,
        a0=a0,
        f0_theoretical=f0_theo,
        f1=f1,
        a1=a1,
        f1_theoretical=f1_theo,
        freq_per_step=freq_per_step,
        amp_per_step=amp_per_step,
        freq_ratio=ratio,
        max_freq_shift=max_shift,
        controlling_step=ctrl_step,
        separation_success=sep_ok,
    )


# ---------------------------------------------------------------------------
# Batch statistics
# ---------------------------------------------------------------------------

def compute_batch_statistics(
    results: List[DualResonanceResult],
) -> BatchDualResonanceStats:
    """Compute aggregate statistics from a list of profile results."""
    successful = [r for r in results if r.success]
    n_ok = len(successful)

    if n_ok == 0:
        return BatchDualResonanceStats(
            n_profiles=len(results),
            n_successful=0,
            success_rate=0.0,
        )

    f0s = np.array([r.f0 for r in successful])
    f1s = np.array([r.f1 for r in successful])
    f0t = np.array([r.f0_theoretical for r in successful])
    f1t = np.array([r.f1_theoretical for r in successful])
    ratios = np.array([r.freq_ratio for r in successful])

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    # Per-step shift distributions
    max_steps = max(len(r.freq_per_step) for r in successful)
    shift_lists: List[List[float]] = [[] for _ in range(max(max_steps - 1, 0))]
    for r in successful:
        for i in range(1, len(r.freq_per_step)):
            if i - 1 < len(shift_lists):
                shift_lists[i - 1].append(
                    r.freq_per_step[i] - r.freq_per_step[i - 1]
                )

    mean_shifts = [float(np.mean(s)) if s else 0.0 for s in shift_lists]
    std_shifts = [float(np.std(s)) if s else 0.0 for s in shift_lists]
    sep_count = sum(1 for r in successful if r.separation_success)

    return BatchDualResonanceStats(
        n_profiles=len(results),
        n_successful=n_ok,
        success_rate=n_ok / len(results) * 100 if results else 0.0,
        f0_mean=float(np.mean(f0s)),
        f0_std=float(np.std(f0s)),
        f0_min=float(np.min(f0s)),
        f0_max=float(np.max(f0s)),
        f1_mean=float(np.mean(f1s)),
        f1_std=float(np.std(f1s)),
        f1_min=float(np.min(f1s)),
        f1_max=float(np.max(f1s)),
        freq_ratio_mean=float(np.mean(ratios)),
        freq_ratio_std=float(np.std(ratios)),
        f0_theoretical_correlation=_safe_corr(f0s, f0t),
        f1_theoretical_correlation=_safe_corr(f1s, f1t),
        separation_success_rate=sep_count / n_ok * 100 if n_ok else 0.0,
        mean_freq_shift_per_step=mean_shifts,
        std_freq_shift_per_step=std_shifts,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_results_csv(
    results: List[DualResonanceResult],
    output_path: str,
) -> None:
    """Save batch results to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "profile_name", "success", "n_layers", "total_depth_m",
        "f0_Hz", "a0", "f0_theoretical_Hz",
        "f1_Hz", "a1", "f1_theoretical_Hz",
        "freq_ratio", "max_freq_shift_Hz",
        "controlling_step", "separation_success", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "profile_name": r.profile_name,
                "success": r.success,
                "n_layers": r.n_layers,
                "total_depth_m": r.total_depth,
                "f0_Hz": r.f0,
                "a0": r.a0,
                "f0_theoretical_Hz": r.f0_theoretical,
                "f1_Hz": r.f1,
                "a1": r.a1,
                "f1_theoretical_Hz": r.f1_theoretical,
                "freq_ratio": r.freq_ratio,
                "max_freq_shift_Hz": r.max_freq_shift,
                "controlling_step": r.controlling_step,
                "separation_success": r.separation_success,
                "error": r.error_message,
            })


def save_statistics_json(
    stats: BatchDualResonanceStats,
    output_path: str,
) -> None:
    """Serialise batch statistics to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(stats), fh, indent=2)


__all__ = [
    "DualResonanceResult",
    "BatchDualResonanceStats",
    "theoretical_frequency",
    "extract_dual_resonance",
    "compute_batch_statistics",
    "save_results_csv",
    "save_statistics_json",
    "SEPARATION_RATIO_THRESHOLD",
    "SEPARATION_SHIFT_THRESHOLD",
]
