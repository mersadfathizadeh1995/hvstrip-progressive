"""Plot observed HV median (+/- deviation) together with HVf forward-model curves."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from HVSR_Extraction.hv_curve_extractor import extract_hv_curve

# ---------------------------------------------------------------------------
# Configuration (edit to match your data)
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"D:\Research\Analysis_tools\HV_In")
MAT_FILE = BASE_DIR / "HVSR_Extraction" / "HVSR_Median_120Sec_HV1_B2.mat"
BATCH_CURVES_FILE = BASE_DIR / "example" / "batch_hv_curves_10.txt"
MODEL_IDS: Optional[Iterable[int]] = None  # e.g. [211, 320]; None -> use all
MAX_MODELS: Optional[int] = 10             # limit number of curves for clarity
FMIN: Optional[float] = 1
FMAX: Optional[float] = 50.0
NUM_POINTS: Optional[int] = None           # resample observed curve (None retains native sampling)
SPACING: str = "linear"                    # "linear" or "log"
PREFER_PERCENTILES = False                 # use HVStd unless missing

PLOT_TITLE = "Observed vs Forward-Model HV Curves"
OUTPUT_FIG = BASE_DIR / "example" / "comparison_hv_curves.png"
PLOT_X_MIN: Optional[float] = 1.0
PLOT_X_MAX: Optional[float] = 10
PLOT_Y_MIN: Optional[float] = None
PLOT_Y_MAX: Optional[float] = None

# ---------------------------------------------------------------------------


def _native_path(path: Path) -> Path:
    if os.name == "posix":
        raw = str(path)
        if len(raw) >= 2 and raw[1] == ":":
            drive = raw[0].lower()
            rest = raw[2:].replace("\\", "/").lstrip("/")
            return Path(f"/mnt/{drive}/{rest}")
    return path


def load_observed_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curve = extract_hv_curve(
        _native_path(MAT_FILE),
        fmin=FMIN,
        fmax=FMAX,
        num_points=NUM_POINTS,
        spacing=SPACING,
        prefer_percentiles=PREFER_PERCENTILES,
    )
    return curve.frequency, curve.median, curve.plus, curve.minus


def parse_batch_curves() -> list[tuple[int, np.ndarray, np.ndarray]]:
    path = _native_path(BATCH_CURVES_FILE)
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    model_id: Optional[int] = None
    freq: list[float] = []
    amp: list[float] = []

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# Model"):
                if model_id is not None and freq:
                    curves.append((model_id, np.array(freq, dtype=float), np.array(amp, dtype=float)))
                parts = line.split()
                model_id = int(parts[2])
                freq = []
                amp = []
            elif line.startswith("#"):
                continue
            else:
                f_val, a_val = map(float, line.split())
                freq.append(f_val)
                amp.append(a_val)
    if model_id is not None and freq:
        curves.append((model_id, np.array(freq, dtype=float), np.array(amp, dtype=float)))
    return curves


def plot_comparison(observed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    modeled: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    freq_obs, median_obs, plus_obs, minus_obs = observed

    mask_obs = np.ones_like(freq_obs, dtype=bool)
    if PLOT_X_MIN is not None:
        mask_obs &= freq_obs >= PLOT_X_MIN
    if PLOT_X_MAX is not None:
        mask_obs &= freq_obs <= PLOT_X_MAX
    if mask_obs.any():
        freq_obs_p = freq_obs[mask_obs]
        median_obs_p = median_obs[mask_obs]
        plus_obs_p = plus_obs[mask_obs]
        minus_obs_p = minus_obs[mask_obs]
    else:
        freq_obs_p, median_obs_p, plus_obs_p, minus_obs_p = freq_obs, median_obs, plus_obs, minus_obs

    plt.figure(figsize=(9, 5))
    plt.fill_between(freq_obs_p, minus_obs_p, plus_obs_p, color="tab:orange", alpha=0.25, label="Observed ± deviation")
    plt.semilogx(freq_obs_p, median_obs_p, color="tab:blue", linewidth=2.0, label="Observed median")

    for mid, freq, amp in modeled:
        mask = np.ones_like(freq, dtype=bool)
        if PLOT_X_MIN is not None:
            mask &= freq >= PLOT_X_MIN
        if PLOT_X_MAX is not None:
            mask &= freq <= PLOT_X_MAX
        freq_p = freq[mask]
        amp_p = amp[mask]
        if freq_p.size == 0:
            freq_p, amp_p = freq, amp
        plt.semilogx(freq_p, amp_p, linewidth=1.0, alpha=0.8, label=f"Model {mid}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("H/V amplitude")
    plt.title(PLOT_TITLE)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(fontsize=8, ncol=2)

    if PLOT_X_MIN is not None or PLOT_X_MAX is not None:
        plt.xlim(left=PLOT_X_MIN, right=PLOT_X_MAX)
    if PLOT_Y_MIN is not None or PLOT_Y_MAX is not None:
        plt.ylim(bottom=PLOT_Y_MIN, top=PLOT_Y_MAX)
    plt.tight_layout()

    out_path = _native_path(OUTPUT_FIG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved comparison plot to {out_path}")


def main() -> None:
    observed = load_observed_curve()
    modeled = parse_batch_curves()

    if MODEL_IDS is not None:
        id_set = {mid for mid in MODEL_IDS}
        modeled = [curve for curve in modeled if curve[0] in id_set]
    if MAX_MODELS is not None:
        modeled = modeled[:MAX_MODELS]

    if not modeled:
        raise RuntimeError("No modeled curves selected to plot.")

    plot_comparison(observed, modeled)


if __name__ == "__main__":
    main()
