"""Drawing / rendering functions for the All Profiles View.

All public functions accept explicit data so they can be tested without
a live QWidget.  The ``redraw_hv`` and ``redraw_vs`` helpers read widget
state through a thin *view* reference whose attributes are documented in
each docstring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .constants import MARKER_SHAPES
from .save_helpers import build_depth_vs
from .statistics import compute_stats, get_colors


# ── Utilities ──────────────────────────────────────────────────

def safe_tight_layout(fig) -> None:
    """Call ``fig.tight_layout()`` swallowing singular-matrix errors."""
    try:
        fig.tight_layout()
    except Exception:
        pass


def apply_smart_ylim(ax, method: str, results: list) -> None:
    """Clip Y-axis using *method* (e.g. '95th Percentile')."""
    all_amps = []
    for r in results:
        if r.amps is not None:
            all_amps.extend(r.amps.tolist())
    if not all_amps:
        return
    arr = np.array(all_amps)
    ymin = max(0, float(np.min(arr)) * 0.9)
    if method == "95th Percentile":
        ymax = float(np.percentile(arr, 95)) * 1.1
    elif method == "Mean + 3σ":
        ymax = float(np.mean(arr) + 3 * np.std(arr))
    elif method == "Mean + 2×IQR":
        q1, q3 = np.percentile(arr, [25, 75])
        ymax = float(np.mean(arr) + 2 * (q3 - q1))
    else:
        return
    ax.set_ylim(ymin, ymax)


# ── HV curve redraw ───────────────────────────────────────────

def redraw_hv(view) -> None:
    """Repaint the HV overlay canvas.

    *view* must expose:
        _results, _hv_plot, _peak_data, _median_peaks,
        _median_ann_positions, _palette, _alpha, _lw,
        _f0_marker, _f0_size, _sec_marker, _sec_size,
        _ann_font, _chk_annotations, _chk_primary, _chk_secondary,
        _chk_median, _med_lw, _chk_sigma, _sigma_alpha,
        _chk_grid, _grid_alpha, _xlabel_edit, _ylabel_edit,
        _title_edit, _chk_ylim_auto, _ylim_method, _fig_size
    """
    results = view._results
    if not results:
        return

    fig = view._hv_plot.figure
    fig.clear()
    ax = fig.add_subplot(111)

    n = len(results)
    palette_name = view._palette.currentText()
    colors = get_colors(palette_name, n)
    alpha = view._alpha.value()
    lw = view._lw.value()
    f0_mk = MARKER_SHAPES.get(view._f0_marker.currentText(), "*")
    f0_ms = view._f0_size.value()
    sec_mk = MARKER_SHAPES.get(view._sec_marker.currentText(), "D")
    sec_ms = view._sec_size.value()
    ann_fs = view._ann_font.value()
    show_ann = view._chk_annotations.isChecked()

    # Individual curves
    for i, r in enumerate(results):
        c = colors[i % n]
        ax.plot(r.freqs, r.amps, color=c, lw=lw, alpha=alpha, label=r.name)

        pk = view._peak_data.get(r.name, {})
        f0 = pk.get("f0")
        if f0 and view._chk_primary.isChecked():
            ax.plot(f0[0], f0[1], f0_mk, color=c, ms=f0_ms, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.3)
            if show_ann:
                ax.annotate(f"{f0[0]:.3f}", xy=(f0[0], f0[1]),
                            xytext=(4, 6), textcoords="offset points",
                            fontsize=max(ann_fs - 2, 5), color=c,
                            fontweight="bold")

        if view._chk_secondary.isChecked():
            for s in pk.get("secondary", []):
                ax.plot(s[0], s[1], sec_mk, color=c, ms=sec_ms,
                        zorder=5, alpha=0.7,
                        markeredgecolor="black", markeredgewidth=0.3)
                if show_ann:
                    ax.annotate(f"{s[0]:.2f}", xy=(s[0], s[1]),
                                xytext=(4, -8), textcoords="offset points",
                                fontsize=max(ann_fs - 3, 4), color=c, alpha=0.8)

    # Median overlay
    if view._chk_median.isChecked() and n >= 2:
        med_f, med_a, std = compute_stats(results)
        if med_f is not None:
            ax.plot(med_f, med_a, color="black", lw=view._med_lw.value(),
                    label="Median", zorder=10)

            mp = view._median_peaks
            f0m = mp.get("f0")
            if f0m is None:
                idx = int(np.argmax(med_a))
                f0m = (med_f[idx], med_a[idx], idx)

            ax.plot(f0m[0], f0m[1], "*", color="red", ms=14, zorder=11,
                    markeredgecolor="darkred", markeredgewidth=0.8)
            ax.axvline(f0m[0], color="red", ls="--", lw=0.8, alpha=0.4)

            if show_ann:
                _annotate_median_peak(ax, f0m, "Median f0", "red", ann_fs,
                                      view._median_ann_positions)

            for j, sp in enumerate(mp.get("secondary", [])):
                ax.plot(sp[0], sp[1], "*", color="green", ms=12, zorder=11,
                        markeredgecolor="darkgreen", markeredgewidth=0.6)
                ax.axvline(sp[0], color="green", ls=":", lw=0.7, alpha=0.4)
                if show_ann:
                    _annotate_secondary(ax, sp, j, ann_fs,
                                        view._median_ann_positions)

            if view._chk_sigma.isChecked() and std is not None:
                ax.fill_between(med_f, med_a - std, med_a + std,
                                alpha=view._sigma_alpha.value(),
                                color="gray", label="±1σ")

    # Axis decoration
    ax.set_xscale("log")
    ax.set_xlabel(view._xlabel_edit.text())
    ax.set_ylabel(view._ylabel_edit.text())
    title = view._title_edit.text() or f"All Profiles ({n})"
    ax.set_title(title, fontsize=12, fontweight="bold")

    if view._chk_grid.isChecked():
        ax.grid(True, alpha=view._grid_alpha.value(), which="both")

    if not view._chk_ylim_auto.isChecked():
        pass  # manual limits (future spinbox)
    else:
        method = view._ylim_method.currentText()
        if method != "Auto":
            apply_smart_ylim(ax, method, results)

    if n <= 15:
        ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.8)
    fig.tight_layout()
    view._hv_plot.refresh()


# ── Vs profile redraw ─────────────────────────────────────────

def redraw_vs(view) -> None:
    """Repaint the Vs overlay canvas.

    *view* must expose:
        _results, _vs_plot, _palette, _chk_vs_median, _chk_vs30
    """
    profiles = [r for r in view._results if r.profile]
    fig = view._vs_plot.figure

    size = fig.get_size_inches()
    if size[0] < 0.1 or size[1] < 0.1:
        return

    if not profiles:
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No Vs data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        safe_tight_layout(fig)
        view._vs_plot.refresh()
        return

    fig.clear()
    ax = fig.add_subplot(111)

    n = len(profiles)
    palette_name = view._palette.currentText()
    colors = get_colors(palette_name, n)
    all_depths_list, all_vs_list = [], []
    any_data = False

    for i, r in enumerate(profiles):
        depths, vs, _fin, _hs = build_depth_vs(r.profile)
        if depths:
            ax.plot(vs, depths, color=colors[i % n],
                    lw=1.0, alpha=0.6, label=r.name)
            any_data = True
        all_depths_list.append(depths)
        all_vs_list.append(vs)

    if not any_data:
        ax.text(0.5, 0.5, "No Vs data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        safe_tight_layout(fig)
        view._vs_plot.refresh()
        return

    # Median Vs
    if view._chk_vs_median.isChecked() and len(profiles) >= 2:
        try:
            max_depth = max(max(d) for d in all_depths_list if d)
            common_d = np.linspace(0, max_depth, 200)
            interps = []
            for depths, vs in zip(all_depths_list, all_vs_list):
                if len(depths) >= 2:
                    interps.append(np.interp(common_d, depths, vs))
            if len(interps) >= 2:
                med_vs = np.median(np.array(interps), axis=0)
                ax.plot(med_vs, common_d, color="black", lw=2.5,
                        label="Median Vs", zorder=10)
        except Exception:
            pass

    ax.invert_yaxis()
    ax.set_xlabel("Vs (m/s)", fontsize=8)
    ax.set_ylabel("Depth (m)", fontsize=8)
    ax.set_title("Vs Profiles", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    fig.canvas.draw()

    if view._chk_vs30.isChecked():
        try:
            ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.6)
            xlim = ax.get_xlim()
            ax.annotate(
                "Vs30 (30 m)",
                xy=(xlim[0] + (xlim[1] - xlim[0]) * 0.05, 30.0),
                fontsize=7, color="blue", fontweight="bold",
                xytext=(0, -8), textcoords="offset points")
        except Exception:
            pass

    if n <= 10:
        ax.legend(fontsize=6, loc="lower right")
    safe_tight_layout(fig)
    view._vs_plot.refresh()


# ── Private annotation helpers ─────────────────────────────────

def _annotate_median_peak(ax, peak, label_prefix, color, fontsize,
                          ann_positions):
    """Draw an annotation for a median peak (f0 or secondary)."""
    ann_key = f"{peak[0]:.6f}"
    ann_pos = ann_positions.get(ann_key)
    text = f"{label_prefix} = {peak[0]:.3f} Hz"
    if ann_pos:
        ax.annotate(
            text, xy=(peak[0], peak[1]), xycoords="data",
            xytext=ann_pos, textcoords="data",
            fontsize=fontsize, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=color, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
    else:
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        y_off = -20 if peak[1] > ylim[0] + 0.8 * y_range else 10
        ax.annotate(
            text, xy=(peak[0], peak[1]),
            xytext=(10, y_off), textcoords="offset points",
            fontsize=fontsize, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=color, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))


def _annotate_secondary(ax, sp, j, fontsize, ann_positions):
    """Draw an annotation for secondary peak *j*."""
    ann_key = f"{sp[0]:.6f}"
    ann_pos = ann_positions.get(ann_key)
    text = f"Sec.{j+1} ({sp[0]:.2f} Hz)"
    if ann_pos:
        ax.annotate(
            text, xy=(sp[0], sp[1]), xycoords="data",
            xytext=ann_pos, textcoords="data",
            fontsize=max(fontsize - 1, 5), color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="green", alpha=0.8))
    else:
        ax.annotate(
            text, xy=(sp[0], sp[1]),
            xytext=(8, -14), textcoords="offset points",
            fontsize=max(fontsize - 1, 5), color="green")
