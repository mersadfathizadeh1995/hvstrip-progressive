"""Convenience runner for generating and plotting H/V curves for multiple models."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import HV_In_Batch_Modeling as batch

# ---------------------------------------------------------------------------
# Hard-coded configuration. Adjust these to match your project structure.
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"D:\Research\Analysis_tools\HV_In")
VS_FILE = BASE_DIR / "example" / "Indep_Powerplant_B6_vs.txt"
VP_FILE = BASE_DIR / "example" / "Indep_Powerplant_B6_vp.txt"
RHO_FILE = BASE_DIR / "example" / "Indep_Powerplant_B6_rho.txt"
HVF_EXE = BASE_DIR / "HVf.exe"

# Batch run parameters
MODEL_COUNT = 10
START_INDEX = 0
FMIN = 1
FMAX = 10
NF = 71
NMR = 10
NML = 10
NKS = 10

OUTPUT_TXT = BASE_DIR / "example" / "batch_hv_curves_10.txt"
OUTPUT_FIG = BASE_DIR / "example" / "batch_hv_curves_10.png"
MODEL_DIR = BASE_DIR / "example" / "batch_models"

# Plot configuration
X_MIN = 1       # e.g., 0.5
X_MAX = 5       # e.g., 50
Y_MIN = None       # e.g., 0.5
Y_MAX = None       # e.g., 4.0
X_SCALE = "linear"   # "log" or "linear"
Y_SCALE = "linear"  # "log" or "linear"


def _native_path(path: Path) -> Path:
    """Convert Windows-style paths to the host platform (handles WSL)."""
    if os.name == "posix":
        raw = str(path)
        if len(raw) >= 2 and raw[1] == ":":
            drive = raw[0].lower()
            rest = raw[2:].replace("\\", "/").lstrip("/")
            return Path(f"/mnt/{drive}/{rest}")
    return path


def run_batch() -> None:
    vs_path = _native_path(VS_FILE)
    vp_path = _native_path(VP_FILE)
    rho_path = _native_path(RHO_FILE)
    hvf_path = _native_path(HVF_EXE)
    out_txt = _native_path(OUTPUT_TXT)
    model_dir = _native_path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--vs-file", str(vs_path),
        "--vp-file", str(vp_path),
        "--rho-file", str(rho_path),
        "--hvf-exe", str(hvf_path),
        "--output", str(out_txt),
        "--model-dir", str(model_dir),
        "--count", str(MODEL_COUNT),
        "--start", str(START_INDEX),
        "--fmin", str(FMIN),
        "--fmax", str(FMAX),
        "--nf", str(NF),
        "--nmr", str(NMR),
        "--nml", str(NML),
        "--nks", str(NKS),
    ]
    exit_code = batch.main(args)
    if exit_code != 0:
        raise SystemExit(exit_code)


def plot_curves() -> None:
    out_txt = _native_path(OUTPUT_TXT)
    out_fig = _native_path(OUTPUT_FIG)
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    current_id: int | None = None
    freq: list[float] = []
    amp: list[float] = []

    with out_txt.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# Model"):
                if current_id is not None and freq:
                    curves.append((current_id, np.array(freq, dtype=float), np.array(amp, dtype=float)))
                parts = line.split()
                current_id = int(parts[2])
                freq = []
                amp = []
            elif line.startswith("#"):
                continue
            else:
                f_val, a_val = map(float, line.split())
                freq.append(f_val)
                amp.append(a_val)

    if current_id is not None and freq:
        curves.append((current_id, np.array(freq, dtype=float), np.array(amp, dtype=float)))

    if not curves:
        raise RuntimeError(f"No curves found in {out_txt}")

    plt.figure(figsize=(9, 5))
    for model_id, frequency, amplitude in curves:
        mask = np.ones_like(frequency, dtype=bool)
        if X_MIN is not None:
            mask &= frequency >= X_MIN
        if X_MAX is not None:
            mask &= frequency <= X_MAX
        freq_plot = frequency[mask]
        amp_plot = amplitude[mask]
        if freq_plot.size == 0:
            continue
        if X_SCALE == "log":
            plt.semilogx(freq_plot, amp_plot, label=f"Model {model_id}")
        else:
            plt.plot(freq_plot, amp_plot, label=f"Model {model_id}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("H/V amplitude")
    plt.title(f"H/V curves for first {len(curves)} models")
    plt.grid(True, which="both" if X_SCALE == "log" else "major", ls=":")
    plt.legend(fontsize=8, ncol=2)

    plt.xscale(X_SCALE)
    plt.yscale(Y_SCALE)

    if X_MIN is not None or X_MAX is not None:
        plt.xlim(left=X_MIN, right=X_MAX)
    if Y_MIN is not None or Y_MAX is not None:
        plt.ylim(bottom=Y_MIN, top=Y_MAX)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=180)
    plt.close()
    print(f"Saved plot to {out_fig}")


def main() -> None:
    run_batch()
    plot_curves()


if __name__ == "__main__":
    main()
