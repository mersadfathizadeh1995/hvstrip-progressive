"""
Single-profile dual-resonance extraction.

Extracts deep (f0) and shallow (f1) resonance frequencies from stripping
step results.  Supports user-supplied peaks and configurable step pairs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..hv_postprocess import DEFAULT_CONFIG, detect_peak, read_hv_csv, read_model
from .data_structures import DualResonanceResult


# ---------------------------------------------------------------------------
# Theoretical helpers
# ---------------------------------------------------------------------------

SEPARATION_RATIO_THRESHOLD = 1.2
SEPARATION_SHIFT_THRESHOLD = 0.3  # Hz


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
# Step folder discovery
# ---------------------------------------------------------------------------

def discover_step_folders(strip_path: Path) -> List[Path]:
    """Return sorted list of step folders in a strip directory."""
    return sorted(
        (d for d in strip_path.iterdir()
         if d.is_dir() and d.name.startswith("Step") and "-layer" in d.name),
        key=lambda p: int(p.name.split("_")[0].replace("Step", "")),
    )


# ---------------------------------------------------------------------------
# Single-profile extraction
# ---------------------------------------------------------------------------

def extract_dual_resonance(
    strip_dir: str,
    peak_config: Optional[Dict] = None,
    profile_name: Optional[str] = None,
    profile_path: Optional[str] = None,
    user_peaks: Optional[Dict[str, Tuple[float, float]]] = None,
    step_pair: Optional[Tuple[int, int]] = None,
) -> DualResonanceResult:
    """Extract f0/f1 from an already-computed stripping output folder.

    Parameters
    ----------
    strip_dir : str
        Path to the ``strip/`` directory that contains ``Step*_*-layer/``
        sub-folders, each holding ``model_*.txt`` and ``hv_curve.csv``.
    peak_config : dict, optional
        Peak-detection config passed to ``detect_peak`` for auto-detection.
    profile_name : str, optional
        Human-readable label (defaults to folder name).
    profile_path : str, optional
        Original model file path for bookkeeping.
    user_peaks : dict, optional
        Mapping of step folder name → ``(frequency, amplitude)`` from
        the HV Strip Wizard.  When provided for a step, these peaks are
        used instead of auto-detection.
    step_pair : tuple of (int, int), optional
        Which step indices to use as ``(deep, shallow)`` for the f0/f1
        comparison.  Default is ``(0, 1)``.

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
    pair = step_pair or (0, 1)

    fail = DualResonanceResult(
        profile_name=profile_name,
        profile_path=profile_path,
        success=False,
    )

    # Discover step folders
    step_folders = discover_step_folders(strip_path)
    if not step_folders:
        fail.error_message = "No step folders found"
        return fail

    step_names = [sf.name for sf in step_folders]

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
        sname = sf.name
        hv_csv = sf / "hv_curve.csv"

        # Check for user-supplied peak first
        if user_peaks and sname in user_peaks:
            up = user_peaks[sname]
            if up is not None and len(up) >= 2:
                freq_per_step.append(float(up[0]))
                amp_per_step.append(float(up[1]))
                continue

        # Auto-detect
        if not hv_csv.exists():
            continue
        freqs, amps = read_hv_csv(hv_csv)
        f_peak, a_peak, _ = detect_peak(freqs, amps, cfg)
        freq_per_step.append(f_peak)
        amp_per_step.append(a_peak)

    if len(freq_per_step) < 2:
        fail.error_message = "Not enough steps for dual-resonance analysis"
        return fail

    # Validate step pair indices
    deep_idx, shallow_idx = pair
    if deep_idx >= len(freq_per_step) or shallow_idx >= len(freq_per_step):
        fail.error_message = (
            f"Step pair ({deep_idx}, {shallow_idx}) out of range "
            f"(only {len(freq_per_step)} steps available)")
        return fail

    f0 = freq_per_step[deep_idx]
    a0 = amp_per_step[deep_idx]
    f1 = freq_per_step[shallow_idx]
    a1 = amp_per_step[shallow_idx]

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
        step_names=step_names,
        step_pair=pair,
        freq_ratio=ratio,
        max_freq_shift=max_shift,
        controlling_step=ctrl_step,
        separation_success=sep_ok,
    )
