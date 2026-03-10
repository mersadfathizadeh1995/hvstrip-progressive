"""Combined / master save orchestration for the All Profiles View.

Functions here write CSV summaries, median curves, per-profile peak
updates, and delegate to :mod:`save_profiles` / :mod:`save_publication`
for individual figure types.  They accept plain data so they are
testable without a live QWidget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .save_helpers import (
    build_depth_vs,
    get_median_f0,
    resolve_peak,
    save_figure_pair,
    write_peak_info,
)
from .save_profiles import save_profile_figure, save_vs_figure, save_vs_info
from .statistics import compute_stats


# ── Combined CSV + median curve output ─────────────────────────

def save_combined_csv(
    out_dir: Path,
    results: list,
    peak_data: dict,
    median_peaks: dict,
) -> None:
    """Write ``combined_summary.csv`` and ``median_hv_curve.csv``."""
    computed = [r for r in results if r.computed]
    med_f, med_a, std = compute_stats(results)

    with open(out_dir / "combined_summary.csv", "w") as fh:
        fh.write("Profile,f0_Hz,f0_Amplitude,Secondary_Peaks,"
                 "Vs30_m_s,VsAvg_m_s\n")
        for r in results:
            f0, sec = resolve_peak(peak_data, r)
            vs30_str = ""
            if r.profile:
                try:
                    from ...core.vs_average import (
                        vs_average_from_profile,
                    )
                    res30 = vs_average_from_profile(
                        r.profile, target_depth=30.0)
                    vs30_str = f"{res30.vs_avg:.2f}"
                except Exception:
                    pass
            if r.computed and f0:
                sec_str = "; ".join(f"{s[0]:.3f} Hz" for s in sec)
                fh.write(
                    f'{r.name},{f0[0]:.6f},{f0[1]:.6f},"{sec_str}",'
                    f"{vs30_str},\n"
                )
            else:
                fh.write(f"{r.name},,,,{vs30_str},\n")

        if med_f is not None:
            idx = int(np.argmax(med_a))
            fh.write(f"Median,{med_f[idx]:.6f},{med_a[idx]:.6f},,,\n")

    # Median HV curve CSV
    if med_f is not None and std is not None:
        with open(out_dir / "median_hv_curve.csv", "w") as fh:
            fh.write("frequency,median_amplitude,std\n")
            for freq, amp, s in zip(med_f, med_a, std):
                fh.write(f"{freq},{amp},{s}\n")

        # Median peak info
        f0m = get_median_f0(median_peaks, med_f, med_a)
        with open(out_dir / "median_peak_info.txt", "w") as fh:
            if f0m:
                fh.write(f"Median_f0_Frequency_Hz,{f0m[0]:.6f}\n")
                fh.write(f"Median_f0_Amplitude,{f0m[1]:.6f}\n")
            for j, sp in enumerate(median_peaks.get("secondary", [])):
                fh.write(
                    f"Median_Secondary_{j+1}_Frequency_Hz,{sp[0]:.6f}\n")
                fh.write(
                    f"Median_Secondary_{j+1}_Amplitude,{sp[1]:.6f}\n")


def save_canvas_snapshots(
    out_dir: Path,
    hv_figure,
    vs_figure,
    vs_visible: bool,
    dpi: int = 300,
    fmt: str = "png",
) -> None:
    """Save the current HV and Vs canvases to disk."""
    if hv_figure is not None:
        save_figure_pair(hv_figure, out_dir, "combined_hv_curves",
                         dpi, fmt, close=False)
    if vs_visible and vs_figure is not None:
        save_figure_pair(vs_figure, out_dir, "vs_profiles_overlay",
                         dpi, fmt, close=False)


# ── Per-profile peak info update ───────────────────────────────

def update_peak_files(
    base_dir: Path,
    results: list,
    peak_data: dict,
) -> None:
    """Overwrite ``peak_info.txt`` for every computed profile."""
    for r in results:
        if not r.computed:
            continue
        f0, sec = resolve_peak(peak_data, r)
        if f0:
            prof_dir = base_dir / r.name
            if prof_dir.exists():
                write_peak_info(prof_dir / "peak_info.txt", f0, sec)


# ── Per-profile re-save helpers ────────────────────────────────

def resave_hv_csv(base_dir: Path, results: list) -> None:
    """Re-write per-profile ``hv_curve.csv``."""
    for r in results:
        if not r.computed or r.freqs is None:
            continue
        prof_dir = base_dir / r.name
        prof_dir.mkdir(exist_ok=True)
        with open(prof_dir / "hv_curve.csv", "w") as fh:
            fh.write("frequency,amplitude\n")
            for freq, amp in zip(r.freqs, r.amps):
                fh.write(f"{freq},{amp}\n")


def resave_hv_figures(
    base_dir: Path,
    results: list,
    peak_data: dict,
    figsize: Tuple[int, int],
    dpi: int,
    fmt: str,
    legend_cfg: Optional[dict] = None,
) -> None:
    """Re-save per-profile HV forward-curve figures."""
    for r in results:
        if not r.computed:
            continue
        prof_dir = base_dir / r.name
        prof_dir.mkdir(exist_ok=True)
        f0, sec = resolve_peak(peak_data, r)
        save_profile_figure(r, f0, sec, prof_dir, figsize, dpi, fmt,
                            legend_cfg=legend_cfg)


def resave_vs_figures(
    base_dir: Path,
    results: list,
    dpi: int = 300,
) -> None:
    """Re-save per-profile Vs figures and info files."""
    for r in results:
        if not r.computed or not r.profile:
            continue
        prof_dir = base_dir / r.name
        prof_dir.mkdir(exist_ok=True)
        try:
            save_vs_figure(r, prof_dir, dpi)
        except Exception:
            pass
        save_vs_info(r, prof_dir)
