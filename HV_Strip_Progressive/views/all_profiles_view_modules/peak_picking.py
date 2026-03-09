"""Peak-picking interaction mixin for AllProfilesView.

Provides primary / secondary peak selection on the median HV curve via
mouse press-drag-release, as well as undo, clear, and right-click removal.

Usage::

    class AllProfilesView(PeakPickingMixin, QWidget):
        ...
"""

from __future__ import annotations

import numpy as np


class PeakPickingMixin:
    """Mixin that manages interactive peak picking state.

    Assumes the host class has the following attributes:

    * ``_picking_mode``  – ``None | "f0" | "secondary"``
    * ``_median_peaks``  – ``{"f0": tuple|None, "secondary": list}``
    * ``_drag_start``    – ``tuple|None``
    * ``_drag_temp_marker`` – matplotlib artist or ``None``
    * ``_median_ann_positions`` – ``dict``
    * ``_hv_plot``       – ``FigureCanvas`` (has ``.figure``, ``.canvas``)
    * ``_btn_pick_f0``   – QPushButton (toggle)
    * ``_btn_pick_sec``  – QPushButton (toggle)
    * ``_results``       – list of ProfileResult
    * ``_redraw()``      – callable to refresh the HV canvas
    * ``_compute_stats()`` – callable → (med_f, med_a, std)
    """

    # ── State initialisation (call from __init__) ──────────────

    def _init_peak_picking_state(self):
        """Initialise mixin state.  Call from the host ``__init__``."""
        self._picking_mode = None
        self._drag_start = None
        self._drag_temp_marker = None
        self._median_peaks: dict = {"f0": None, "secondary": []}
        self._median_ann_positions: dict = {}

    # ── Toggle buttons ─────────────────────────────────────────

    def _toggle_pick_f0(self, on: bool):
        """Enable / disable primary peak picking."""
        if on:
            self._btn_pick_sec.setChecked(False)
        self._picking_mode = "f0" if on else None
        self._btn_pick_f0.setStyleSheet(
            "background-color: #FFB3B3;" if on else "")

    def _toggle_pick_sec(self, on: bool):
        """Enable / disable secondary peak picking."""
        if on:
            self._btn_pick_f0.setChecked(False)
        self._picking_mode = "secondary" if on else None
        self._btn_pick_sec.setStyleSheet(
            "background-color: #FFDAB3;" if on else "")

    # ── Undo / clear ───────────────────────────────────────────

    def _undo_last_secondary(self):
        """Remove the most recently added secondary peak."""
        if self._median_peaks["secondary"]:
            removed = self._median_peaks["secondary"].pop()
            self._median_ann_positions.pop(f"{removed[0]:.6f}", None)
            self._redraw()

    def _clear_median_peaks(self):
        """Remove all user-selected median peaks."""
        self._median_peaks = {"f0": None, "secondary": []}
        self._drag_start = None
        if self._drag_temp_marker is not None:
            try:
                self._drag_temp_marker.remove()
            except Exception:
                pass
            self._drag_temp_marker = None
        self._median_ann_positions.clear()
        self._redraw()

    # ── Mouse handlers ─────────────────────────────────────────

    def _on_press(self, event):
        """Mouse press: snap to median curve and show a temporary marker."""
        if self._picking_mode is None or event.inaxes is None:
            return
        if not self._results:
            return

        # Right-click removes nearest peak
        if event.button == 3:
            self._remove_nearest_median_peak(event.xdata)
            return

        if event.button != 1:
            return

        med_f, med_a, _ = self._compute_stats()
        if med_f is None:
            return

        cx = event.xdata
        if cx is None:
            return

        idx = int(np.argmin(np.abs(med_f - cx)))
        self._drag_start = (float(med_f[idx]), float(med_a[idx]), idx)

        color = "red" if self._picking_mode == "f0" else "green"
        ax = (self._hv_plot.figure.axes[0]
              if self._hv_plot.figure.axes else None)
        if ax:
            self._drag_temp_marker, = ax.plot(
                self._drag_start[0], self._drag_start[1], "*",
                color=color, ms=14, markeredgecolor="black",
                markeredgewidth=0.8, zorder=20)
            self._hv_plot.canvas.draw_idle()

    def _on_release(self, event):
        """Mouse release: commit the peak and position the annotation."""
        if self._drag_start is None or event.button != 1:
            return

        freq, amp, idx = self._drag_start
        self._drag_start = None

        # Remove temporary marker
        if self._drag_temp_marker is not None:
            try:
                self._drag_temp_marker.remove()
            except Exception:
                pass
            self._drag_temp_marker = None

        peak = (freq, amp, idx)
        if self._picking_mode == "f0":
            self._median_peaks["f0"] = peak
        elif self._picking_mode == "secondary":
            self._median_peaks["secondary"].append(peak)

        # Store annotation position (release point)
        if event.inaxes is not None and event.xdata is not None:
            self._median_ann_positions[f"{freq:.6f}"] = (
                event.xdata, event.ydata)

        self._redraw()

    # ── Right-click removal ────────────────────────────────────

    def _remove_nearest_median_peak(self, xdata):
        """Remove the closest peak (f0 or secondary) to *xdata*."""
        if xdata is None:
            return

        all_peaks: list = []
        f0 = self._median_peaks.get("f0")
        if f0:
            all_peaks.append(("f0", 0, f0))
        for i, sp in enumerate(self._median_peaks.get("secondary", [])):
            all_peaks.append(("sec", i, sp))
        if not all_peaks:
            return

        dists = [abs(xdata - p[2][0]) for p in all_peaks]
        nearest = all_peaks[int(np.argmin(dists))]
        if nearest[0] == "f0":
            self._median_peaks["f0"] = None
        else:
            self._median_peaks["secondary"].pop(nearest[1])

        key = f"{nearest[2][0]:.6f}"
        self._median_ann_positions.pop(key, None)
        self._redraw()
