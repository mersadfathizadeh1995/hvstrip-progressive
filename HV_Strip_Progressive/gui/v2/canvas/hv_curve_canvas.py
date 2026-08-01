"""PyQtGraph-based HV ratio curve canvas.

Used by the Data, Bounds, and Inversion panels to display observed
curves, synthetic overlays and detected peaks.

Design notes
------------
- The x-axis is rendered as ``log10(freq)`` internally, and a custom
  :class:`LogFreqAxis` converts tick positions back to Hz for display.
  This avoids pyqtgraph's ``setLogMode`` quirks with ``FillBetweenItem``.
- Live updates use ``PlotDataItem.setData`` (no full redraw) so the
  widget remains responsive at 60 fps even for long inversions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .constants import (
    BAND_ALPHA,
    MULTI_COLORS,
    OBSERVED_BAND,
    OBSERVED_COLOR,
    OBS_LINE_WIDTH,
    PEAK_COLOR,
    SYNTHETIC_COLOR,
    SYN_LINE_WIDTH,
)
from .log_freq_axis import LogFreqAxis


def _to_log_freq(freqs: np.ndarray) -> np.ndarray:
    """Convert linear Hz to log10 space, clipping non-positive values."""
    f = np.asarray(freqs, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, np.log10(f), -10.0)


class HVCurveCanvas(QWidget):
    """PyQtGraph widget for observed + synthetic HV ratio curves.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    Interaction (T030 — ported, not imported, from ``gui_v2``'s ``HVSRCanvas``):
    when :meth:`set_pick_mode` is on, a left-click **adds** a peak snapped to the
    nearest observed frequency (amplitude read off the curve) and emits
    :attr:`peak_added`; a right-click near an existing peak **removes** it
    (:attr:`peak_remove_requested`); a right-click while holding no peak nearby
    is a no-op.  Double-left-click near a peak **promotes** it to primary
    (:attr:`peak_promote_requested`).  The canvas never mutates state itself —
    the Data panel routes the signal through ``AppState``.
    """

    #: emitted on a left-click add — (frequency Hz, amplitude).
    peak_added = Signal(float, float)
    #: emitted on a right-click near a peak — (frequency Hz).
    peak_remove_requested = Signal(float)
    #: emitted on a double-click near a peak — (frequency Hz).
    peak_promote_requested = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._gl = pg.GraphicsLayoutWidget()
        layout.addWidget(self._gl)

        self._plot: pg.PlotItem = self._gl.addPlot(
            row=0, col=0,
            axisItems={"bottom": LogFreqAxis(orientation="bottom")},
        )
        self._plot.setLabel("bottom", "Frequency (Hz)")
        self._plot.setLabel("left", "H/V Ratio")
        self._plot.setTitle("HVSR Curve")
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        # Legend (top-right, created lazily when items are added)
        self._legend = self._plot.addLegend(
            offset=(-10, 10),
            labelTextSize="9pt",
        )

        # Plot items held for fast updates / removal
        self._obs_line: Optional[pg.PlotDataItem] = None
        self._sigma_fill: Optional[pg.FillBetweenItem] = None
        self._sigma_low: Optional[pg.PlotDataItem] = None
        self._sigma_high: Optional[pg.PlotDataItem] = None
        self._syn_line: Optional[pg.PlotDataItem] = None
        self._syn_multi: List[pg.PlotDataItem] = []
        #: keyed, individually-addressable synthetic curves (Results per-scenario)
        #: — toggled / re-penned by key (see set_item_hv).
        self._items: Dict[str, pg.PlotDataItem] = {}
        self._peak_scatter: Optional[pg.ScatterPlotItem] = None
        self._peak_labels: List[pg.TextItem] = []

        # Theme + interaction state
        self._obs_color = OBSERVED_COLOR
        self._syn_color = SYNTHETIC_COLOR
        self._peak_color = PEAK_COLOR
        self._obs_freqs: Optional[np.ndarray] = None
        self._obs_hv: Optional[np.ndarray] = None
        self._peak_freqs: List[float] = []
        self._pick_mode = False
        self._plot.scene().sigMouseClicked.connect(self._on_scene_click)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def plot_item(self) -> pg.PlotItem:
        """Expose the underlying PlotItem for constraint overlays."""
        return self._plot

    # ------------------------------------------------------------------
    # Theme (T028) — repaint chrome + semantic pens from a Palette
    # ------------------------------------------------------------------
    def apply_theme(self, palette) -> None:
        """Repaint background, axes and semantic pens from *palette*.

        Chrome (bg/axes/text/grid) follows the card surface so the plot sits on
        the same colour as the surrounding dock; the observed curve uses
        ``palette.fg`` (legible on light **and** dark), and the synthetic/best
        overlays use the ``palette.accent`` violet family.
        """
        self._obs_color = palette.fg
        self._syn_color = palette.accent
        self._peak_color = palette.danger
        self._gl.setBackground(palette.card)
        axis_pen = pg.mkPen(palette.muted, width=1)
        text_pen = pg.mkPen(palette.fg)
        for name in ("left", "bottom", "right", "top"):
            ax = self._plot.getAxis(name)
            if ax is not None:
                ax.setPen(axis_pen)
                ax.setTextPen(text_pen)
        self._plot.getViewBox().setBackgroundColor(palette.card)
        if hasattr(self._plot, "titleLabel") and self._plot.titleLabel is not None:
            self._plot.setTitle(self._plot.titleLabel.text, color=palette.fg)
        # Re-pen the live items so a mode switch mid-run recolours instantly.
        if self._obs_line is not None:
            self._obs_line.setPen(pg.mkPen(self._obs_color, width=OBS_LINE_WIDTH))
        if self._syn_line is not None:
            self._syn_line.setPen(pg.mkPen(self._syn_color, width=SYN_LINE_WIDTH))

    def set_pick_mode(self, enabled: bool) -> None:
        """Enable/disable interactive peak picking on this canvas."""
        self._pick_mode = bool(enabled)

    def set_peak_freqs(self, freqs) -> None:
        """Remember the current peak frequencies (for snap-to-remove)."""
        self._peak_freqs = [float(f) for f in (freqs or [])]

    # ------------------------------------------------------------------
    # Mouse interaction (peak pick)
    # ------------------------------------------------------------------
    def _on_scene_click(self, ev) -> None:
        if not self._pick_mode or self._obs_freqs is None:
            return
        try:
            vb = self._plot.getViewBox()
            pt = vb.mapSceneToView(ev.scenePos())
        except Exception:
            return
        log_x = float(pt.x())
        try:
            freq = float(10.0 ** log_x)
        except OverflowError:
            return
        if freq <= 0:
            return
        button = ev.button()
        if button == Qt.RightButton:
            near = self._nearest_peak(freq)
            if near is not None:
                self.peak_remove_requested.emit(near)
            ev.accept()
            return
        if button == Qt.LeftButton:
            idx = int(np.abs(self._obs_freqs - freq).argmin())
            f0 = float(self._obs_freqs[idx])
            a0 = float(self._obs_hv[idx]) if self._obs_hv is not None else 0.0
            if getattr(ev, "double", lambda: False)():
                near = self._nearest_peak(f0)
                if near is not None:
                    self.peak_promote_requested.emit(near)
                    ev.accept()
                    return
            self.peak_added.emit(f0, a0)
            ev.accept()

    def _nearest_peak(self, freq: float, max_log_dist: float = 0.08):
        """Return the nearest picked peak frequency within a log-distance."""
        if not self._peak_freqs or freq <= 0:
            return None
        lf = np.log10(freq)
        best, best_d = None, max_log_dist
        for pf in self._peak_freqs:
            if pf <= 0:
                continue
            d = abs(np.log10(pf) - lf)
            if d < best_d:
                best_d, best = d, pf
        return best

    # ------------------------------------------------------------------
    # Observed curve
    # ------------------------------------------------------------------

    def plot_observed(
        self,
        freqs: np.ndarray,
        hv: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        peaks: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Plot the observed HV curve with optional ±1σ band and peak markers.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array in Hz (linear scale).
        hv : np.ndarray
            HV ratio values.
        sigma : np.ndarray, optional
            Per-frequency standard deviation.
        peaks : list of (freq, amplitude), optional
            Detected peaks to mark with downward triangles.
        """
        self._remove_observed_items()

        freqs = np.asarray(freqs, dtype=float)
        hv = np.asarray(hv, dtype=float)
        self._obs_freqs = freqs
        self._obs_hv = hv
        log_f = _to_log_freq(freqs)

        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            # Skip band if sigma is all ones (no real uncertainty data)
            has_real_sigma = not np.allclose(sigma, 1.0)
            if has_real_sigma:
                band_color = QColor(*OBSERVED_BAND)
                band_color.setAlpha(BAND_ALPHA)
                # Faint boundary lines so the band edges are visible
                edge_color = QColor(*OBSERVED_BAND)
                edge_color.setAlpha(min(BAND_ALPHA + 60, 255))
                edge_pen = pg.mkPen(edge_color, width=0.8)
                self._sigma_low = pg.PlotDataItem(
                    log_f, hv - sigma, pen=edge_pen,
                )
                self._sigma_high = pg.PlotDataItem(
                    log_f, hv + sigma, pen=edge_pen,
                )
                self._sigma_fill = pg.FillBetweenItem(
                    self._sigma_low, self._sigma_high,
                    brush=pg.mkBrush(band_color),
                )
                # Z-order: fill above grid (>0), observed line on top (>1)
                self._sigma_low.setZValue(1)
                self._sigma_high.setZValue(1)
                self._sigma_fill.setZValue(0.5)
                self._plot.addItem(self._sigma_low)
                self._plot.addItem(self._sigma_high)
                self._plot.addItem(self._sigma_fill)

        self._obs_line = self._plot.plot(
            log_f, hv,
            pen=pg.mkPen(self._obs_color, width=OBS_LINE_WIDTH),
            name="Observed",
        )
        self._obs_line.setZValue(2)  # observed line always on top

        if peaks:
            self._peak_freqs = [float(p[0]) for p in peaks]
            pf = np.array([p[0] for p in peaks], dtype=float)
            pa = np.array([p[1] for p in peaks], dtype=float)
            self._peak_scatter = pg.ScatterPlotItem(
                _to_log_freq(pf), pa,
                symbol="t", size=12,
                brush=pg.mkBrush(self._peak_color),
                pen=pg.mkPen("white", width=1),
                name="Peaks",
            )
            self._plot.addItem(self._peak_scatter)
            self._peak_scatter.setZValue(3)
            for f, a in peaks:
                txt = pg.TextItem(
                    f"{f:.2f} Hz", color=self._peak_color, anchor=(0.5, 1.2),
                )
                txt.setPos(float(np.log10(f) if f > 0 else -10.0), float(a))
                self._plot.addItem(txt)
                self._peak_labels.append(txt)
        else:
            self._peak_freqs = []

        self._plot.enableAutoRange()

    # ------------------------------------------------------------------
    # Synthetic (single) overlay
    # ------------------------------------------------------------------

    def _save_view_state(self) -> Tuple[bool, bool, Tuple, Tuple]:
        """Capture auto-range flags and current ranges per axis.

        When the user pans/zooms, pyqtgraph disables auto-range on the
        affected axis; we use that signal to preserve their view across
        live updates instead of snapping back to data extents.
        """
        vb = self._plot.getViewBox()
        auto_x, auto_y = vb.autoRangeEnabled()
        (x_range, y_range) = vb.viewRange()
        return bool(auto_x), bool(auto_y), tuple(x_range), tuple(y_range)

    def _restore_view_state(
        self, state: Tuple[bool, bool, Tuple, Tuple],
    ) -> None:
        """Re-apply saved ranges for axes whose auto-range was disabled."""
        auto_x, auto_y, x_range, y_range = state
        vb = self._plot.getViewBox()
        if not auto_x:
            vb.setXRange(*x_range, padding=0)
        if not auto_y:
            vb.setYRange(*y_range, padding=0)

    def update_synthetic(
        self, freqs: np.ndarray, hv_syn: np.ndarray
    ) -> None:
        """Create or update the single synthetic overlay curve.

        Fast: calls ``setData`` on an existing item, no full redraw.
        """
        freqs = np.asarray(freqs, dtype=float)
        hv_syn = np.asarray(hv_syn, dtype=float)
        log_f = _to_log_freq(freqs)

        state = self._save_view_state()
        # When multi-model mode was active, swap to single-model mode.
        if self._syn_multi:
            self._remove_multi_synthetic()

        if self._syn_line is None:
            self._syn_line = self._plot.plot(
                log_f, hv_syn,
                pen=pg.mkPen(self._syn_color, width=SYN_LINE_WIDTH),
                name="Synthetic",
            )
        else:
            self._syn_line.setData(log_f, hv_syn)
        self._restore_view_state(state)

    def clear_synthetic(self) -> None:
        """Remove any synthetic overlay (single or multi-model)."""
        if self._syn_line is not None:
            self._plot.removeItem(self._syn_line)
            self._syn_line = None
        self._remove_multi_synthetic()

    # ------------------------------------------------------------------
    # Multi-model overlay
    # ------------------------------------------------------------------

    def update_multi_synthetic(
        self, freqs: np.ndarray, models: List[Dict],
    ) -> None:
        """Overlay multiple synthetic HV curves with a shared legend.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array in Hz.
        models : list of dict
            Each model has ``hv_syn`` (sequence) and optional ``misfit``.
            The first model (presumed best) is drawn thicker and less
            transparent; the next four get labelled legend entries.
        """
        freqs = np.asarray(freqs, dtype=float)
        log_f = _to_log_freq(freqs)

        state = self._save_view_state()

        # Single-model line is no longer representative - drop it.
        if self._syn_line is not None:
            self._plot.removeItem(self._syn_line)
            self._syn_line = None

        self._remove_multi_synthetic()

        for i, md in enumerate(models):
            hv = md.get("hv_syn")
            if hv is None or len(hv) != len(freqs):
                continue
            color = QColor(MULTI_COLORS[i % len(MULTI_COLORS)])
            lw = 2.2 if i == 0 else 1.0
            alpha = 230 if i == 0 else 150
            color.setAlpha(alpha)
            misfit = md.get("misfit", 0.0)
            label = f"#{i+1} ({misfit:.4f})" if i < 5 else None
            pen = pg.mkPen(color, width=lw)
            item = self._plot.plot(
                log_f, np.asarray(hv, dtype=float),
                pen=pen, name=label,
            )
            self._syn_multi.append(item)

        self._restore_view_state(state)

    def _remove_multi_synthetic(self) -> None:
        for it in self._syn_multi:
            self._plot.removeItem(it)
        self._syn_multi.clear()

    # ------------------------------------------------------------------
    # Keyed items — individually-addressable synthetics (Results view)
    # ------------------------------------------------------------------
    def set_item_hv(
        self, key: str, freqs: np.ndarray, hv_syn: np.ndarray, pen,
    ) -> pg.PlotDataItem:
        """Create or update a keyed synthetic H/V curve (per-scenario model).

        Existing keys are fast-updated via ``setData`` + ``setPen``; a new key is
        plotted (log-freq x) and stored.  No view reset.
        """
        log_f = _to_log_freq(np.asarray(freqs, dtype=float))
        hv = np.asarray(hv_syn, dtype=float)
        item = self._items.get(key)
        if item is None:
            item = self._plot.plot(log_f, hv, pen=pen)
            item.setZValue(1)          # below the observed line (z=2)
            self._items[key] = item
        else:
            item.setData(log_f, hv)
            item.setPen(pen)
        return item

    def set_item_visible(self, key: str, visible: bool) -> None:
        item = self._items.get(key)
        if item is not None:
            item.setVisible(bool(visible))

    def set_item_pen(self, key: str, pen) -> None:
        item = self._items.get(key)
        if item is not None:
            item.setPen(pen)

    def remove_item(self, key: str) -> None:
        item = self._items.pop(key, None)
        if item is not None:
            self._plot.removeItem(item)

    def item_keys(self) -> List[str]:
        return list(self._items.keys())

    def clear_items(self) -> None:
        for item in self._items.values():
            self._plot.removeItem(item)
        self._items.clear()

    # ------------------------------------------------------------------
    # Cleanup helpers
    # ------------------------------------------------------------------

    def _remove_observed_items(self) -> None:
        """Clear observed curve, sigma band, peak markers and labels."""
        if self._obs_line is not None:
            self._plot.removeItem(self._obs_line)
            self._obs_line = None
        for itm in (self._sigma_fill, self._sigma_low, self._sigma_high):
            if itm is not None:
                self._plot.removeItem(itm)
        self._sigma_fill = None
        self._sigma_low = None
        self._sigma_high = None
        if self._peak_scatter is not None:
            self._plot.removeItem(self._peak_scatter)
            self._peak_scatter = None
        for t in self._peak_labels:
            self._plot.removeItem(t)
        self._peak_labels.clear()

    def clear_all(self) -> None:
        """Remove every plot item and reset the view."""
        self._remove_observed_items()
        self.clear_synthetic()
        self.clear_items()
        self._plot.enableAutoRange()
