"""
Publication-quality plots for dual-resonance (f0/f1) analysis.

Provides:
- ``plot_resonance_separation`` — side-by-side HV curves + Vs profile
- ``plot_frequency_distribution`` — histograms for f0 and f1
- ``plot_theoretical_validation`` — measured vs. theoretical scatter
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ..core.soil_profile import compute_halfspace_display_depth


# ---------------------------------------------------------------------------
# Single-profile resonance separation figure
# ---------------------------------------------------------------------------

def draw_resonance_separation(
    strip_dir: str,
    fig: "matplotlib.figure.Figure",
    **kw,
) -> bool:
    """Draw dual-resonance panels into an existing matplotlib *fig*.

    Left panel: original vs. stripped HV curves with f0/f1 annotations.
    Right panel: Vs-depth profile coloured by retained / removed layers.

    Parameters
    ----------
    strip_dir : str
        Path to the ``strip/`` directory with Step folders.
    fig : matplotlib.figure.Figure
        Target figure (cleared before drawing).
    **kw
        Optional overrides: ``f0_offset``, ``f1_offset`` (Δx, Δy tuples),
        ``show_stripped``, ``linewidth``, ``grid``, ``hs_ratio``, ``font_size``.

    Returns
    -------
    bool
        ``True`` if the figure was drawn successfully.
    """
    strip_path = Path(strip_dir)
    data = _load_separation_data(strip_path)
    if data is None:
        return False

    fig.clear()
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.3)
    ax_hv = fig.add_subplot(gs[0])
    ax_vs = fig.add_subplot(gs[1])

    _draw_hv_panel(ax_hv, *data["hv"], **kw)
    _draw_vs_panel(ax_vs, data["layers"], data["n_kept"], **kw)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return True


def plot_resonance_separation(
    strip_dir: str,
    output_path: str,
    dpi: int = 300,
    figsize: tuple = (15, 9),
) -> Optional[str]:
    """Generate a two-panel figure showing resonance mode separation.

    Left panel: original vs. stripped HV curves with f0/f1 annotations.
    Right panel: Vs-depth profile coloured by retained / removed layers.

    Parameters
    ----------
    strip_dir : str
        Path to the ``strip/`` directory with Step folders.
    output_path : str
        Destination image file path.
    dpi : int
        Figure resolution.
    figsize : tuple
        Figure size in inches (width, height).

    Returns
    -------
    str or None
        Path to saved figure, or ``None`` if data was insufficient.
    """
    fig = plt.figure(figsize=figsize)
    ok = draw_resonance_separation(strip_dir, fig)
    if not ok:
        plt.close(fig)
        return None

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Batch statistical plots
# ---------------------------------------------------------------------------

def plot_frequency_distribution(
    f0_values: List[float],
    f1_values: List[float],
    output_path: str,
    dpi: int = 300,
) -> str:
    """Histogram of deep (f0) and shallow (f1) resonance frequencies."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(f0_values, bins="auto", color="#2A9D8F", edgecolor="white", alpha=0.85)
    if f0_values:
        axes[0].axvline(np.mean(f0_values), color="black", ls="--", lw=1.5,
                        label=f"Mean = {np.mean(f0_values):.2f} Hz")
    axes[0].set_xlabel("Frequency (Hz)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Deep Resonance $f_0$", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)

    axes[1].hist(f1_values, bins="auto", color="#E76F51", edgecolor="white", alpha=0.85)
    if f1_values:
        axes[1].axvline(np.mean(f1_values), color="black", ls="--", lw=1.5,
                        label=f"Mean = {np.mean(f1_values):.2f} Hz")
    axes[1].set_xlabel("Frequency (Hz)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Shallow Resonance $f_1$", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return str(out)


def plot_theoretical_validation(
    measured: List[float],
    theoretical: List[float],
    output_path: str,
    label: str = "f",
    dpi: int = 300,
) -> str:
    """Scatter plot of measured vs. theoretical frequencies with 1:1 line."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(theoretical, measured, s=40, alpha=0.7, edgecolors="black", linewidths=0.5)
    lo = min(min(theoretical, default=0), min(measured, default=0))
    hi = max(max(theoretical, default=1), max(measured, default=1))
    margin = (hi - lo) * 0.1
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", lw=1, label="1 : 1")
    ax.set_xlabel(f"Theoretical ${label}$ (Hz)", fontsize=12)
    ax.set_ylabel(f"Measured ${label}$ (Hz)", fontsize=12)
    ax.set_title(f"Theoretical vs Measured ${label}$", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_separation_data(strip_path: Path) -> Optional[Dict]:
    """Load all data needed for a resonance-separation figure."""
    step0_dirs = list(strip_path.glob("Step0_*"))
    if not step0_dirs:
        return None
    step0 = step0_dirs[0]
    hv0 = step0 / "hv_curve.csv"
    if not hv0.exists():
        return None
    freqs0, amps0 = _load_hv_csv(hv0)

    step1_dirs = list(strip_path.glob("Step1_*"))
    if step1_dirs:
        step1 = step1_dirs[0]
    else:
        others = sorted(strip_path.glob("Step*_*-layer"))
        if len(others) < 2:
            return None
        step1 = others[1]

    hv1 = step1 / "hv_curve.csv"
    if not hv1.exists():
        return None
    freqs1, amps1 = _load_hv_csv(hv1)

    model_files = list(step0.glob("model_*.txt"))
    layers = _read_model_layers(model_files[0]) if model_files else []

    f0, a0 = _detect_first_peak(freqs0, amps0)
    f1, a1 = _detect_primary_peak(freqs1, amps1)
    n_kept = _parse_kept_layers(step1.name)

    return {
        "hv": (freqs0, amps0, freqs1, amps1, f0, a0, f1, a1),
        "layers": layers,
        "n_kept": n_kept,
    }


def _load_hv_csv(path: Path):
    """Load (freqs, amps) from a two-column CSV with header."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def _read_model_layers(path: Path) -> List[Dict]:
    """Read HVf model file into a list of layer dicts with depth info."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    layers: List[Dict] = []
    depth = 0.0
    try:
        n = int(lines[0])
        for line in lines[1: n + 1]:
            parts = [float(x) for x in line.split()]
            if len(parts) < 3:
                continue
            thick = parts[0]
            vs = parts[2]
            display_thick = thick if thick > 0 else compute_halfspace_display_depth(depth)
            layers.append({
                "top": depth,
                "bot": depth + display_thick,
                "vs": vs,
                "thick": display_thick,
            })
            depth += display_thick
    except (ValueError, IndexError):
        pass
    return layers


def _detect_first_peak(freqs, amps):
    """Detect the first significant peak (deep resonance f0)."""
    smooth = gaussian_filter1d(amps, sigma=1)
    prom = 0.3 * (np.max(smooth) - np.min(smooth))
    height = np.mean(smooth) + 0.2 * np.std(smooth)
    peaks, _ = find_peaks(smooth, prominence=prom * 0.3, height=height * 0.5, distance=3)
    if len(peaks) >= 1:
        idx = peaks[0]
        return float(freqs[idx]), float(amps[idx])
    idx = int(np.argmax(amps))
    return float(freqs[idx]), float(amps[idx])


def _detect_primary_peak(freqs, amps):
    """Detect the primary (highest amplitude) peak."""
    idx = int(np.argmax(amps))
    return float(freqs[idx]), float(amps[idx])


def _parse_kept_layers(step_name: str) -> int:
    """Extract the number of kept finite layers from a step folder name."""
    try:
        parts = step_name.split("_")
        return int(parts[1].split("-")[0])
    except (IndexError, ValueError):
        return 1


def _draw_hv_panel(ax, freqs0, amps0, freqs1, amps1, f0, a0, f1, a1, **kw):
    """Draw the HV-curve overlay panel."""
    lw = kw.get("linewidth", 3)
    show_stripped = kw.get("show_stripped", True)
    fs = kw.get("font_size", 13)
    grid = kw.get("grid", True)

    ax.semilogx(freqs0, amps0, color="black", lw=lw, label="Original Model", zorder=3)
    if show_stripped:
        ax.semilogx(freqs1, amps1, color="#E63946", lw=max(lw - 0.5, 1), ls="--",
                     label="Stripped Model", zorder=4)

    max_amp = max(a0, a1, float(np.max(amps0)))
    ax.set_ylim(0, max_amp * 1.35)

    f0_off = kw.get("f0_offset", (0.0, 0.0))
    f1_off = kw.get("f1_offset", (0.0, 0.0))

    ax.annotate(
        f"Deep Resonance ($f_0$)\n{f0:.2f} Hz",
        xy=(f0, a0),
        xytext=(f0 * 0.45 + f0_off[0], max_amp * 1.15 + f0_off[1]),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
        fontsize=max(fs - 2, 8), fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
    )
    ax.annotate(
        f"Shallow Resonance ($f_1$)\n{f1:.2f} Hz",
        xy=(f1, a1),
        xytext=(f1 * 1.6 + f1_off[0], max_amp * 1.15 + f1_off[1]),
        arrowprops=dict(facecolor="#E63946", shrink=0.05, width=1.5),
        fontsize=max(fs - 2, 8), fontweight="bold", color="#E63946", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E63946", alpha=0.9),
    )
    ax.plot(f0, a0, "o", color="black", ms=7, zorder=5)
    ax.plot(f1, a1, "o", color="#E63946", ms=7, zorder=5)

    ax.set_xlabel("Frequency (Hz)", fontsize=fs, fontweight="bold")
    ax.set_ylabel("H/V Amplitude", fontsize=fs, fontweight="bold")
    ax.set_title("(a) Resonance Mode Separation", fontsize=fs + 2, fontweight="bold", pad=15)
    if grid:
        ax.grid(True, which="both", alpha=0.3)
    else:
        ax.grid(False)
    ax.legend(fontsize=max(fs - 2, 8), loc="lower left", frameon=True, framealpha=0.95)
    ax.set_xlim(0.2, 20.0)


def _draw_vs_panel(ax, layers: List[Dict], n_kept: int, **kw):
    """Draw the Vs-depth profile panel."""
    if not layers:
        return
    fs = kw.get("font_size", 13)
    grid = kw.get("grid", True)
    max_vs = max(l["vs"] for l in layers)
    max_depth = layers[-1]["bot"]

    for i, layer in enumerate(layers):
        is_retained = i < n_kept
        color = "#E63946" if is_retained else "#B0B0B0"
        alpha = 0.6 if is_retained else 0.4

        rect = Rectangle(
            (0, layer["top"]), layer["vs"], layer["thick"],
            facecolor=color, alpha=alpha, edgecolor="black", lw=1,
        )
        ax.add_patch(rect)

        y_mid = (layer["top"] + layer["bot"]) / 2
        ax.annotate(
            f"$V_s$={int(layer['vs'])}",
            xy=(layer["vs"], y_mid),
            xytext=(max_vs * 1.15, y_mid),
            ha="left", va="center", fontsize=max(fs - 2, 8), fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    # "REMOVED" bracket for stripped layers
    if n_kept < len(layers):
        removed_top = layers[n_kept]["top"]
        removed_bot = layers[-1]["bot"]
        bx = max_vs * 1.6
        mid_y = (removed_top + removed_bot) / 2

        ax.annotate(
            "REMOVED", xy=(bx, mid_y),
            xytext=(bx + max_vs * 0.1, mid_y),
            rotation=90, va="center", ha="center",
            fontweight="bold", color="#666666", fontsize=max(fs - 1, 8),
        )
        ax.plot([bx, bx], [removed_top, removed_bot], color="#666666", lw=2)
        tick = max_vs * 0.05
        ax.plot([bx - tick, bx], [removed_top, removed_top], color="#666666", lw=2)
        ax.plot([bx - tick, bx], [removed_bot, removed_bot], color="#666666", lw=2)

    ax.invert_yaxis()
    ax.set_xlabel("Shear Wave Velocity, $V_s$ (m/s)", fontsize=fs, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=fs, fontweight="bold")
    ax.set_title("(b) Progressive Stripping", fontsize=fs + 2, fontweight="bold", pad=15)
    if grid:
        ax.grid(True, alpha=0.3, which="major", ls="--")
    else:
        ax.grid(False)
    ax.set_xlim(0, max_vs * 2.0)
    ax.set_ylim(max_depth, 0)


__all__ = [
    "draw_resonance_separation",
    "plot_resonance_separation",
    "plot_frequency_distribution",
    "plot_theoretical_validation",
]
