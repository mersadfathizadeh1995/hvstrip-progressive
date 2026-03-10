"""Publication-quality figure generators for the All Profiles View.

Each function produces a single figure type suitable for journal papers.
All functions accept pre-computed data so they have no dependency on the
view class itself.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure

from .save_helpers import (
    build_depth_vs,
    get_median_f0,
    resolve_peak,
    save_figure_pair,
)
from .statistics import compute_stats, get_colors
from .drawing import _apply_legend


def _legend_cfg_or_default(legend_cfg: Optional[dict] = None) -> dict:
    """Return a legend config dict, falling back to publication defaults."""
    if legend_cfg is not None:
        return legend_cfg
    return dict(mode="Full", loc="upper right", fontsize=7,
                ncol=2, alpha=0.9, frame=True)


# ── 1. Combined HV + Vs side-by-side ──────────────────────────

def save_hv_vs_combined(
    out_dir: Path,
    computed: list,
    results: list,
    median_peaks: dict,
    palette_name: str,
    dpi: int = 300,
    fmt: str = "png",
    legend_cfg: Optional[dict] = None,
) -> None:
    """Two-panel figure: all HV curves (left) + all Vs profiles (right)."""
    n = len(computed)
    colors = get_colors(palette_name, n)

    fig = Figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Left: HV curves
    for i, r in enumerate(computed):
        ax1.plot(r.freqs, r.amps, color=colors[i % n], lw=0.8,
                 alpha=0.5, label=r.name)

    med_f, med_a, std = compute_stats(results)
    if med_f is not None:
        ax1.plot(med_f, med_a, color="black", lw=2.5,
                 label="Median", zorder=10)
        if std is not None:
            ax1.fill_between(med_f, med_a - std, med_a + std,
                             alpha=0.15, color="gray", label="±1σ")

        f0m = get_median_f0(median_peaks, med_f, med_a)
        if f0m:
            ax1.plot(f0m[0], f0m[1], "*", color="red", ms=14,
                     zorder=11, markeredgecolor="darkred",
                     markeredgewidth=0.8,
                     label=f"f0 = {f0m[0]:.3f} Hz")
            ax1.axvline(f0m[0], color="red", ls="--", lw=0.8, alpha=0.4)
        for j, sp in enumerate(median_peaks.get("secondary", [])):
            ax1.plot(sp[0], sp[1], "*", color="green", ms=12,
                     zorder=11, markeredgecolor="darkgreen",
                     markeredgewidth=0.6,
                     label=f"Sec.{j+1} ({sp[0]:.2f} Hz)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Frequency (Hz)", fontsize=11)
    ax1.set_ylabel("H/V Amplitude Ratio", fontsize=11)
    ax1.set_title("All Profiles — H/V Curves", fontsize=12,
                  fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")
    lcfg = _legend_cfg_or_default(legend_cfg)
    _apply_legend(ax1, lcfg, summary_labels={"Median", "±1σ"})

    # Right: Vs profiles
    for i, r in enumerate(computed):
        if not r.profile:
            continue
        depths, vs, _fin, _hs = build_depth_vs(r.profile)
        if depths:
            ax2.plot(vs, depths, color=colors[i % n], lw=0.8,
                     alpha=0.6, label=r.name)

    ax2.invert_yaxis()
    ax2.set_xlabel("Vs (m/s)", fontsize=11)
    ax2.set_ylabel("Depth (m)", fontsize=11)
    ax2.set_title("Vs Profiles", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    try:
        ax2.axhline(30.0, color="blue", lw=0.8, ls="-.",
                    alpha=0.6, label="Vs30 (30 m)")
    except Exception:
        pass
    _apply_legend(ax2, _legend_cfg_or_default(legend_cfg),
                  summary_labels={"Median", "Vs30"})

    fig.tight_layout()
    save_figure_pair(fig, out_dir, "publication_hv_vs_combined", dpi, fmt)


# ── 2. Median-only ± σ ────────────────────────────────────────

def save_median_hv(
    out_dir: Path,
    results: list,
    median_peaks: dict,
    dpi: int = 300,
    fmt: str = "png",
    legend_cfg: Optional[dict] = None,
) -> None:
    """Clean figure showing only the median curve with ± 1σ band."""
    med_f, med_a, std = compute_stats(results)
    if med_f is None:
        return

    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.plot(med_f, med_a, color="black", lw=2.5, label="Median H/V")
    if std is not None:
        ax.fill_between(med_f, med_a - std, med_a + std,
                        alpha=0.2, color="gray", label="±1σ")

    f0m = get_median_f0(median_peaks, med_f, med_a)
    if f0m:
        ax.plot(f0m[0], f0m[1], "*", color="red", ms=16,
                zorder=10, markeredgecolor="darkred",
                markeredgewidth=0.8,
                label=f"f0 = {f0m[0]:.3f} Hz (A = {f0m[1]:.2f})")
        ax.axvline(f0m[0], color="red", ls="--", lw=0.9, alpha=0.4)

    for j, sp in enumerate(median_peaks.get("secondary", [])):
        ax.plot(sp[0], sp[1], "*", color="green", ms=13,
                zorder=9, markeredgecolor="black",
                markeredgewidth=0.5,
                label=f"Secondary ({sp[0]:.2f} Hz, A={sp[1]:.2f})")
        ax.axvline(sp[0], color="green", ls=":", lw=0.8, alpha=0.4)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("H/V Amplitude Ratio", fontsize=12)
    ax.set_title("Median H/V Spectral Ratio", fontsize=14,
                 fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    lcfg = _legend_cfg_or_default(legend_cfg)
    lcfg_med = {**lcfg, "fontsize": max(lcfg["fontsize"], 10)}
    _apply_legend(ax, lcfg_med)
    fig.tight_layout()
    save_figure_pair(fig, out_dir, "publication_median_hv", dpi, fmt)


# ── 3. Summary tables (CSV + LaTeX) ───────────────────────────

def save_summary_tables(
    out_dir: Path,
    computed: list,
    results: list,
    peak_data: dict,
    median_peaks: dict,
) -> None:
    """Write summary_table.csv and summary_table.tex."""
    med_f, med_a, _std = compute_stats(results)

    rows = []
    for r in computed:
        f0, sec = resolve_peak(peak_data, r)
        vs30_str = "—"
        if r.profile:
            try:
                from ...core.vs_average import (
                    vs_average_from_profile,
                )
                res30 = vs_average_from_profile(r.profile, target_depth=30.0)
                vs30_str = f"{res30.vs_avg:.0f}"
            except Exception:
                pass
        f0_str = f"{f0[0]:.3f}" if f0 else "—"
        amp_str = f"{f0[1]:.2f}" if f0 else "—"
        sec_str = "; ".join(f"{s[0]:.2f}" for s in sec)
        rows.append((r.name, f0_str, amp_str, vs30_str, sec_str))

    # CSV
    with open(out_dir / "summary_table.csv", "w") as fh:
        fh.write("Profile,f0 (Hz),Amplitude,Vs30 (m/s),Secondary Peaks\n")
        for name, f0_s, amp_s, vs_s, sec_s in rows:
            fh.write(f"{name},{f0_s},{amp_s},{vs_s},{sec_s}\n")
        if med_f is not None:
            f0m = get_median_f0(median_peaks, med_f, med_a)
            if f0m:
                fh.write(f"Median,{f0m[0]:.3f},{f0m[1]:.2f},—,\n")

    # LaTeX
    with open(out_dir / "summary_table.tex", "w") as fh:
        fh.write("\\begin{table}[htbp]\n\\centering\n")
        fh.write("\\caption{HVSR Analysis Summary}\n")
        fh.write("\\begin{tabular}{lcccc}\n\\hline\n")
        fh.write("Profile & $f_0$ (Hz) & Amplitude & $V_{s30}$ (m/s) "
                 "& Secondary Peaks \\\\\n\\hline\n")
        for name, f0_s, amp_s, vs_s, sec_s in rows:
            tex_name = name.replace("_", "\\_")
            fh.write(f"{tex_name} & {f0_s} & {amp_s} & "
                     f"{vs_s} & {sec_s} \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n\\end{table}\n")


# ── 4. Vs comparison (standalone) ──────────────────────────────

def save_vs_comparison(
    out_dir: Path,
    computed: list,
    palette_name: str,
    dpi: int = 300,
    fmt: str = "png",
    legend_cfg: Optional[dict] = None,
) -> None:
    """All Vs profiles with median Vs overlay and Vs30 line."""
    profiles = [r for r in computed if r.profile]
    if not profiles:
        return

    n = len(profiles)
    colors = get_colors(palette_name, n)
    fig = Figure(figsize=(8, 10))
    ax = fig.add_subplot(111)

    all_d, all_v = [], []
    for i, r in enumerate(profiles):
        depths, vs, _fin, _hs = build_depth_vs(r.profile)
        if depths:
            ax.plot(vs, depths, color=colors[i % n], lw=1.0,
                    alpha=0.6, label=r.name)
            all_d.append(depths)
            all_v.append(vs)

    # Median Vs
    if len(all_d) >= 2:
        max_d = max(max(d) for d in all_d)
        common = np.linspace(0, max_d, 200)
        interps = []
        for d, v in zip(all_d, all_v):
            if len(d) >= 2:
                interps.append(np.interp(common, d, v))
        if len(interps) >= 2:
            med_vs = np.median(np.array(interps), axis=0)
            ax.plot(med_vs, common, color="black", lw=2.5,
                    label="Median Vs", zorder=10)

    ax.invert_yaxis()
    ax.set_xlabel("Vs (m/s)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title("Vs Profile Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    try:
        ax.axhline(30.0, color="blue", lw=0.8, ls="-.",
                   alpha=0.6, label="Vs30 (30 m)")
    except Exception:
        pass
    _apply_legend(ax, _legend_cfg_or_default(legend_cfg),
                  summary_labels={"Median", "Vs30"})
    fig.tight_layout()
    save_figure_pair(fig, out_dir, "publication_vs_comparison", dpi, fmt)


# ── 5. Normalized HV ──────────────────────────────────────────

def save_normalized_hv(
    out_dir: Path,
    computed: list,
    results: list,
    palette_name: str,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    fmt: str = "png",
    legend_cfg: Optional[dict] = None,
) -> None:
    """All curves normalised to peak amplitude = 1."""
    if not computed:
        return
    n = len(computed)
    colors = get_colors(palette_name, n)
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for i, r in enumerate(computed):
        if r.freqs is None:
            continue
        peak = np.max(r.amps) if np.max(r.amps) > 0 else 1.0
        ax.plot(r.freqs, r.amps / peak, color=colors[i % n],
                lw=0.8, alpha=0.5, label=r.name)

    med_f, med_a, _ = compute_stats(results)
    if med_f is not None and np.max(med_a) > 0:
        ax.plot(med_f, med_a / np.max(med_a), color="black",
                lw=2.5, label="Median", zorder=10)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Normalized H/V (peak = 1)", fontsize=12)
    ax.set_title("Normalized H/V Comparison", fontsize=14,
                 fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    _apply_legend(ax, _legend_cfg_or_default(legend_cfg),
                  summary_labels={"Median"})
    fig.tight_layout()
    save_figure_pair(fig, out_dir, "publication_normalized_hv", dpi, fmt)


# ── 6. f0 histogram ───────────────────────────────────────────

def save_f0_histogram(
    out_dir: Path,
    computed: list,
    peak_data: dict,
    dpi: int = 300,
    fmt: str = "png",
    legend_cfg: Optional[dict] = None,
) -> None:
    """Bar chart of f0 values across all profiles."""
    f0_vals = []
    for r in computed:
        f0, _ = resolve_peak(peak_data, r)
        if f0:
            f0_vals.append(f0[0])
    if len(f0_vals) < 2:
        return

    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    n_bins = min(max(len(f0_vals) // 2, 5), 20)
    ax.hist(f0_vals, bins=n_bins, color="steelblue",
            edgecolor="black", alpha=0.8)
    ax.axvline(np.median(f0_vals), color="red", lw=2, ls="--",
               label=f"Median = {np.median(f0_vals):.3f} Hz")
    ax.axvline(np.mean(f0_vals), color="orange", lw=2, ls=":",
               label=f"Mean = {np.mean(f0_vals):.3f} Hz")
    ax.set_xlabel("Fundamental Frequency f0 (Hz)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("f0 Distribution Across Profiles", fontsize=14,
                 fontweight="bold")
    lcfg = _legend_cfg_or_default(legend_cfg)
    lcfg_hist = {**lcfg, "fontsize": max(lcfg["fontsize"], 10)}
    _apply_legend(ax, lcfg_hist)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure_pair(fig, out_dir, "publication_f0_histogram", dpi, fmt)


# ── 7. f0 vs Vs30 scatter ─────────────────────────────────────

def save_f0_vs_vs30(
    out_dir: Path,
    computed: list,
    peak_data: dict,
    dpi: int = 300,
    fmt: str = "png",
) -> None:
    """Scatter plot of fundamental frequency vs Vs30."""
    pairs: list = []
    for r in computed:
        f0, _ = resolve_peak(peak_data, r)
        if f0 and r.profile:
            try:
                from ...core.vs_average import (
                    vs_average_from_profile,
                )
                res30 = vs_average_from_profile(r.profile, target_depth=30.0)
                pairs.append((f0[0], res30.vs_avg, r.name))
            except Exception:
                pass
    if len(pairs) < 2:
        return

    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    freqs = [p[0] for p in pairs]
    vs30s = [p[1] for p in pairs]
    ax.scatter(freqs, vs30s, s=80, c="steelblue",
               edgecolors="black", zorder=5, alpha=0.8)
    for f, v, name in pairs:
        ax.annotate(name, (f, v), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.7)
    ax.set_xlabel("Fundamental Frequency f0 (Hz)", fontsize=12)
    ax.set_ylabel("Vs30 (m/s)", fontsize=12)
    ax.set_title("f0 vs Vs30", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure_pair(fig, out_dir, "publication_f0_vs_vs30", dpi, fmt)


# ── 8. Spectral ratio matrix ──────────────────────────────────

def save_spectral_matrix(
    out_dir: Path,
    computed: list,
    peak_data: dict,
    dpi: int = 300,
    fmt: str = "png",
) -> None:
    """Small-multiples grid of individual HV curves."""
    n = len(computed)
    if n < 1:
        return

    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig = Figure(figsize=(4 * ncols, 3 * nrows))

    for i, r in enumerate(computed):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        if r.freqs is not None:
            ax.plot(r.freqs, r.amps, color="steelblue", lw=1.2)
            f0, _ = resolve_peak(peak_data, r)
            if f0:
                ax.axvline(f0[0], color="red", lw=0.8, ls="--", alpha=0.6)
                ax.plot(f0[0], f0[1], "rv", ms=8, zorder=5)
        ax.set_xscale("log")
        ax.set_title(r.name, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(labelsize=7)
        if i % ncols == 0:
            ax.set_ylabel("H/V", fontsize=8)
        if i >= n - ncols:
            ax.set_xlabel("Freq (Hz)", fontsize=8)

    fig.tight_layout()
    save_figure_pair(fig, out_dir, "publication_spectral_matrix", dpi, fmt)
