"""``HVInteractiveFigure`` — the interactive HV curve with manual peak
picking (spec 002 FR-1/FR-2/FR-3; the legacy app's headline capability).

Ported (copy-not-import) from ``gui/views/strip_wizard_view.py``'s
press/motion/release machinery and marker/annotation drawing, rebased on
:class:`MplFigureWidget` (PySide6 ``backend_qtagg``) and enhanced:

* the CLICK-vs-DRAG duality, verbatim: a short click places the peak at
  the EXACT clicked frequency with the amplitude interpolated from the
  curve; a drag beyond ``2 %`` of the (linear) frequency span snaps to
  the **argmax inside the dragged band**, previewed live as a
  translucent ``axvspan`` (red for f0, orange for secondary);
* f0 is one-shot (the arm auto-releases), secondaries accumulate;
* the UNIFORM verb set (FR-2 — the legacy asymmetry resolved):
  right-click deletes the nearest peak, Undo pops the last secondary,
  Clear wipes the current scope; every verb everywhere;
* picking is suppressed while the toolbar's pan/zoom is armed;
* annotation labels are draggable AND their dragged position persists
  per peak (FR-3 — data-coordinate ``label_pos``, re-applied with a
  leader arrow on every redraw; the legacy strip wizard lost them);
* peaks are plain api dicts (``frequency``/``amplitude``/``label``/
  ``source``/``label_pos``) so the owner pushes them straight into the
  AppState → facade store.

The widget is presentation-only: no AppState, no disk — the owner feeds
``set_curve``/``set_peaks`` and harvests ``get_peaks()`` on
``peaks_changed``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton

from HV_Strip_Progressive.gui.v2.canvas.mpl_widget import MplFigureWidget

#: Legacy secondary-marker color cycle (strip_wizard_view.SEC_COLORS).
SEC_COLORS = ["green", "purple", "orange", "brown", "teal"]

#: Drag-vs-click threshold, relative to the LINEAR frequency span
#: (legacy ``DRAG_THRESHOLD``).
DRAG_THRESHOLD = 0.02

_DEFAULT_STYLE = {
    "show_markers": True,
    "show_annotations": True,
    "f0_shape": "*",
    "f0_size": 14.0,
    "secondary_shape": "*",
    "secondary_size": 11.0,
    "annotation_fontsize": 8,
}


class HVInteractiveFigure(MplFigureWidget):
    """One HV curve + manual peak picking (see module docstring)."""

    #: any pick / delete / undo / clear / label-drag — harvest
    #: :meth:`get_peaks` and push to the AppState.
    peaks_changed = Signal()
    #: "" | "f0" | "secondary"
    pick_mode_changed = Signal(str)

    def __init__(self, parent=None, *, log_x: bool = True) -> None:
        super().__init__(parent, figsize=(8.0, 4.6), toolbar=True)
        self._log_x = log_x
        self._freqs: Optional[np.ndarray] = None
        self._amps: Optional[np.ndarray] = None
        self._title = ""
        self._f0: Optional[Dict[str, Any]] = None
        self._secondary: List[Dict[str, Any]] = []
        self._style = dict(_DEFAULT_STYLE)
        self._markers_visible = True

        self._drag_start_x: Optional[float] = None
        self._drag_rect = None
        self._annotations: List[tuple] = []      # (peak_dict, annotation)

        self._build_verb_row()

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self.draw_placeholder("No curve")

    # ------------------------------------------------------------------
    #  The verb row (arm f0 / arm secondary / undo / clear)
    # ------------------------------------------------------------------
    def _build_verb_row(self) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(6, 2, 6, 4)
        self._btn_f0 = QPushButton("📍 Select f0")
        self._btn_f0.setCheckable(True)
        self._btn_f0.toggled.connect(self._on_f0_toggled)
        self._btn_sec = QPushButton("🔶 Select Secondary")
        self._btn_sec.setCheckable(True)
        self._btn_sec.toggled.connect(self._on_sec_toggled)
        undo_btn = QPushButton("↩ Undo Sec.")
        undo_btn.clicked.connect(self.undo_secondary)
        clear_btn = QPushButton("✕ Clear Peaks")
        clear_btn.clicked.connect(self.clear_peaks)
        for b in (self._btn_f0, self._btn_sec, undo_btn, clear_btn):
            row.addWidget(b)
        row.addStretch(1)
        self._sel_label = QLabel("")
        self._sel_label.setProperty("role", "caption")
        row.addWidget(self._sel_label)
        self.layout().addLayout(row)

    def _on_f0_toggled(self, on: bool) -> None:
        if on:
            self._btn_sec.setChecked(False)      # mutually exclusive
        self.pick_mode_changed.emit(self.pick_mode)

    def _on_sec_toggled(self, on: bool) -> None:
        if on:
            self._btn_f0.setChecked(False)
        self.pick_mode_changed.emit(self.pick_mode)

    @property
    def pick_mode(self) -> str:
        if self._btn_f0.isChecked():
            return "f0"
        if self._btn_sec.isChecked():
            return "secondary"
        return ""

    def arm(self, mode: str) -> None:
        """Programmatic arming: ``"f0"`` | ``"secondary"`` | ``""``."""
        self._btn_f0.setChecked(mode == "f0")
        self._btn_sec.setChecked(mode == "secondary")

    # ------------------------------------------------------------------
    #  Data in / out
    # ------------------------------------------------------------------
    def set_curve(self, freqs, amps, title: str = "") -> None:
        self._freqs = np.asarray(freqs, dtype=float)
        self._amps = np.asarray(amps, dtype=float)
        self._title = title
        self._redraw()

    def set_peaks(self, f0: Optional[Dict[str, Any]],
                  secondary: Optional[List[Dict[str, Any]]] = None) -> None:
        """Feed api-shaped peak dicts (copies are kept)."""
        self._f0 = dict(f0) if f0 else None
        self._secondary = [dict(s) for s in (secondary or [])]
        self._redraw()

    def get_peaks(self) -> Dict[str, Any]:
        """Current picks: ``{"f0": dict|None, "secondary": [dicts]}``."""
        return {
            "f0": dict(self._f0) if self._f0 else None,
            "secondary": [dict(s) for s in self._secondary],
        }

    def set_marker_style(self, style: Dict[str, Any]) -> None:
        """Merge marker/annotation style (the ``config.markers`` shape)."""
        self._style.update({k: v for k, v in (style or {}).items()
                            if k in _DEFAULT_STYLE})
        self._redraw()

    def set_markers_visible(self, visible: bool) -> None:
        """The layer-tree 'Peak markers' toggle (spec 002 FR-6)."""
        self._markers_visible = bool(visible)
        self._redraw()

    # ------------------------------------------------------------------
    #  Verbs
    # ------------------------------------------------------------------
    def undo_secondary(self) -> None:
        if not self._secondary:
            return
        self._secondary.pop()
        self._redraw()
        self.peaks_changed.emit()

    def clear_peaks(self) -> None:
        if self._f0 is None and not self._secondary:
            return
        self._f0 = None
        self._secondary = []
        self._redraw()
        self.peaks_changed.emit()

    def delete_nearest(self, x: float) -> bool:
        """Right-click verb: delete the peak nearest *x* (data coords)."""
        candidates: List[tuple] = []
        if self._f0:
            candidates.append((abs(x - float(self._f0["frequency"])),
                               "f0", 0))
        for i, s in enumerate(self._secondary):
            candidates.append((abs(x - float(s["frequency"])),
                               "secondary", i))
        if not candidates:
            return False
        _, kind, idx = min(candidates, key=lambda c: c[0])
        if kind == "f0":
            self._f0 = None
        else:
            self._secondary.pop(idx)
        self._redraw()
        self.peaks_changed.emit()
        return True

    # ------------------------------------------------------------------
    #  The picking machinery (legacy-verbatim semantics)
    # ------------------------------------------------------------------
    def _toolbar_active(self) -> bool:
        return bool(self.toolbar is not None and self.toolbar.mode)

    def _on_press(self, event) -> None:
        if event.inaxes is None or self._freqs is None:
            return
        if self._toolbar_active():
            return
        if event.button == 3:                    # right-click delete
            if event.xdata is not None:
                self.delete_nearest(float(event.xdata))
            return
        if not self.pick_mode:
            return
        self._drag_start_x = event.xdata

    def _on_motion(self, event) -> None:
        if self._drag_start_x is None or event.inaxes is None:
            return
        if not self.pick_mode:
            return
        ax = event.inaxes
        if self._drag_rect is not None:
            try:
                self._drag_rect.remove()
            except (ValueError, NotImplementedError):
                pass
            self._drag_rect = None
        x0 = min(self._drag_start_x, event.xdata)
        x1 = max(self._drag_start_x, event.xdata)
        color = "red" if self.pick_mode == "f0" else "orange"
        self._drag_rect = ax.axvspan(x0, x1, alpha=0.15, color=color)
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        try:
            self._harvest_label_positions()
            if (event.inaxes is None or self._freqs is None
                    or event.xdata is None):
                return
            if self._toolbar_active() or not self.pick_mode:
                return
            if event.button == 3:
                return
            freqs, amps = self._freqs, self._amps
            if self._drag_start_x is not None:
                drag_dist = abs(event.xdata - self._drag_start_x)
                freq_range = float(freqs[-1] - freqs[0])
                if drag_dist > freq_range * DRAG_THRESHOLD:
                    # Drag: the true local maximum inside the band.
                    x0 = min(self._drag_start_x, event.xdata)
                    x1 = max(self._drag_start_x, event.xdata)
                    mask = (freqs >= x0) & (freqs <= x1)
                    if np.any(mask):
                        masked = np.where(mask, amps, -np.inf)
                        idx = int(np.argmax(masked))
                        self._commit_pick(float(freqs[idx]),
                                          float(amps[idx]), idx)
                    return
            # Click: the EXACT clicked frequency, amplitude interpolated.
            cx = float(event.xdata)
            amp = float(np.interp(cx, freqs, amps))
            idx = int(np.argmin(np.abs(freqs - cx)))
            self._commit_pick(cx, amp, idx)
        finally:
            self._drag_start_x = None
            if self._drag_rect is not None:
                try:
                    self._drag_rect.remove()
                except (ValueError, NotImplementedError):
                    pass
                self._drag_rect = None
                self.canvas.draw_idle()

    def _commit_pick(self, freq: float, amp: float, idx: int) -> None:
        peak = {"frequency": freq, "amplitude": amp, "index": idx,
                "source": "manual", "label_pos": None}
        if self.pick_mode == "f0":
            peak["label"] = "f0"
            self._f0 = peak
            self._btn_f0.setChecked(False)       # one-shot
        else:
            peak["label"] = f"sec{len(self._secondary) + 1}"
            self._secondary.append(peak)
        self._redraw()
        self.peaks_changed.emit()

    def _harvest_label_positions(self) -> None:
        """After any release, persist dragged annotation positions
        (data coords) back into the peak dicts (spec 002 FR-3)."""
        moved = False
        for peak, ann in self._annotations:
            pos = [float(ann.xyann[0]), float(ann.xyann[1])]
            if peak.get("label_pos") != pos:
                peak["label_pos"] = pos
                moved = True
        if moved:
            self.peaks_changed.emit()

    # ------------------------------------------------------------------
    #  Drawing (legacy marker/annotation styling + persisted labels)
    # ------------------------------------------------------------------
    def _default_label_pos(self, x: float, y: float, k: int) -> List[float]:
        """A deterministic data-coord label spot near the peak (log-aware)."""
        fx = 1.18 if self._log_x else 1.0
        dx = x * fx - x if self._log_x else 0.04 * (
            float(self._freqs[-1] - self._freqs[0]) if self._freqs is not None
            else 1.0)
        dy = (0.06 + 0.05 * (k % 3)) * max(abs(y), 1.0)
        sign = 1.0 if k % 2 == 0 else -1.0
        return [x + dx, y + sign * dy]

    def _redraw(self) -> None:
        self._annotations = []
        if self._freqs is None or not len(self._freqs):
            self.draw_placeholder("No curve")
            return
        p = self._palette
        curve_color = p.accent if p else "#C87A20"
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self._log_x:
            ax.set_xscale("log")
        ax.plot(self._freqs, self._amps, color=curve_color, lw=1.8,
                label="H/V")
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("H/V amplitude", fontsize=9)
        if self._title:
            ax.set_title(self._title, fontsize=10, fontweight="bold")

        st = self._style
        if self._markers_visible and st.get("show_markers", True):
            if self._f0:
                self._draw_peak(ax, self._f0, "red", "darkred",
                                st["f0_shape"], float(st["f0_size"]),
                                f"f0 = {self._f0['frequency']:.4f} Hz\n"
                                f"A = {self._f0['amplitude']:.2f}", 0)
            for j, s in enumerate(self._secondary):
                sc = SEC_COLORS[j % len(SEC_COLORS)]
                self._draw_peak(ax, s, sc, "black",
                                st["secondary_shape"],
                                float(st["secondary_size"]),
                                f"Sec.{j + 1}: {s['frequency']:.3f} Hz "
                                f"({s['amplitude']:.2f})", j + 1)
        self.style_axes(ax)
        self._update_sel_label()
        self.canvas.draw_idle()

    def _draw_peak(self, ax, peak: Dict[str, Any], color: str,
                   edge: str, shape: str, size: float,
                   text: str, k: int) -> None:
        x = float(peak["frequency"])
        y = float(peak["amplitude"])
        ax.plot(x, y, shape, color=color, ms=size, zorder=10,
                markeredgecolor=edge, markeredgewidth=0.8)
        ax.axvline(x, color=color, ls="--" if k == 0 else ":",
                   lw=0.8 if k == 0 else 0.7, alpha=0.4)
        if not self._style.get("show_annotations", True):
            return
        pos = peak.get("label_pos") or self._default_label_pos(x, y, k)
        ann = ax.annotate(
            text, xy=(x, y), xytext=tuple(pos), textcoords="data",
            fontsize=int(self._style["annotation_fontsize"]),
            color=color, fontweight="bold" if k == 0 else "normal",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc=(self._palette.card_alt if self._palette
                          else "white"),
                      ec=color, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )
        ann.draggable(True)
        self._annotations.append((peak, ann))

    def _update_sel_label(self) -> None:
        parts = []
        if self._f0:
            parts.append(f"f0 = {self._f0['frequency']:.4f} Hz "
                         f"(A = {self._f0['amplitude']:.2f})")
        for j, s in enumerate(self._secondary):
            parts.append(f"Sec.{j + 1} = {s['frequency']:.3f} Hz")
        self._sel_label.setText(
            " | ".join(parts) if parts
            else "Arm a button, then click or drag on the curve.")

    def apply_theme(self, palette) -> None:  # keep markers on retheme
        self._palette = palette
        self.figure.set_facecolor(palette.card)
        self._redraw()


__all__ = ["HVInteractiveFigure", "SEC_COLORS", "DRAG_THRESHOLD"]
