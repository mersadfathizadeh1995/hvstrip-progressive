"""PyQtGraph-based Vs-depth staircase canvas.

Public API: :class:`VsProfileCanvas`. Depth increases downward
(y-axis inverted).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .constants import (
    BEST_LINE_WIDTH,
    BEST_MODEL_COLOR,
    ENSEMBLE_LINE_WIDTH,
    HALFSPACE_EXTENSION_M,
    MULTI_COLORS,
)


def _layers_to_staircase(
    layers: List[Dict],
    halfspace_extension: float = HALFSPACE_EXTENSION_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of layer dicts into staircase (vs, depth) arrays.

    Parameters
    ----------
    layers : list of dict
        Each dict must have a ``vs`` key and one of ``thickness`` / ``h``
        for the layer thickness (the last / halfspace layer's thickness
        is replaced by ``halfspace_extension``).
    halfspace_extension : float
        Visual extension below the last real interface (metres).

    Returns
    -------
    vs_vals, depths : np.ndarray
        Arrays suitable for a pyqtgraph stepped line plot.
    """
    depths: List[float] = []
    vs_vals: List[float] = []
    z = 0.0
    n = len(layers)
    for i, lay in enumerate(layers):
        vs = float(lay["vs"])
        h = float(lay.get("thickness", lay.get("h", 0.0)))
        if i < n - 1:
            depths.extend([z, z + h])
            vs_vals.extend([vs, vs])
            z += h
        else:
            depths.extend([z, z + halfspace_extension])
            vs_vals.extend([vs, vs])
    return np.asarray(vs_vals), np.asarray(depths)


class VsProfileCanvas(QWidget):
    """PyQtGraph widget for Vs-depth staircase profiles.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._gl = pg.GraphicsLayoutWidget()
        layout.addWidget(self._gl)

        self._plot: pg.PlotItem = self._gl.addPlot(row=0, col=0)
        self._plot.setLabel("bottom", "Vs (m/s)")
        self._plot.setLabel("left", "Depth (m)")
        self._plot.setTitle("Vs Profile")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.invertY(True)  # depth downward

        self._legend = self._plot.addLegend(
            offset=(-10, 10),
            labelTextSize="9pt",
        )

        self._profile_line: Optional[pg.PlotDataItem] = None
        self._multi_lines: List[pg.PlotDataItem] = []
        #: keyed, individually-addressable staircases (Results per-scenario +
        #: the median) — each toggled / re-penned by its key (see set_item).
        self._items: Dict[str, pg.PlotDataItem] = {}
        self._best_color = BEST_MODEL_COLOR

    @property
    def plot_item(self) -> pg.PlotItem:
        """Expose the underlying PlotItem for constraint overlays."""
        return self._plot

    # ------------------------------------------------------------------
    # Theme (T028)
    # ------------------------------------------------------------------
    def apply_theme(self, palette) -> None:
        """Repaint chrome + the best-model pen from *palette* (accent violet)."""
        self._best_color = palette.accent
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
        if self._profile_line is not None:
            self._profile_line.setPen(pg.mkPen(self._best_color, width=BEST_LINE_WIDTH))

    # ------------------------------------------------------------------
    # Single profile
    # ------------------------------------------------------------------

    def _save_view_state(self) -> Tuple[bool, bool, Tuple, Tuple]:
        """Capture auto-range flags and current ranges per axis.

        Returns ``(auto_x, auto_y, x_range, y_range)``. If auto-range
        is enabled on an axis, the stored range for that axis is not
        meaningful and will be ignored on restore.
        """
        vb = self._plot.getViewBox()
        auto_x, auto_y = vb.autoRangeEnabled()
        (x_range, y_range) = vb.viewRange()
        return bool(auto_x), bool(auto_y), tuple(x_range), tuple(y_range)

    def _restore_view_state(
        self, state: Tuple[bool, bool, Tuple, Tuple],
    ) -> None:
        """Re-apply saved ranges for axes whose auto-range was disabled.

        An axis that had auto-range enabled is left on auto-range so
        the first-ever plot still fits the data.
        """
        auto_x, auto_y, x_range, y_range = state
        vb = self._plot.getViewBox()
        if not auto_x:
            vb.setXRange(*x_range, padding=0)
        if not auto_y:
            vb.setYRange(*y_range, padding=0)

    def plot_profile(
        self,
        layers: List[Dict],
        color: Optional[str] = None,
        label: str = "Best model",
    ) -> None:
        """Replace the current single profile with a new one."""
        state = self._save_view_state()
        self._clear_single()
        vs_vals, depths = _layers_to_staircase(layers)
        pen = pg.mkPen(color or self._best_color, width=BEST_LINE_WIDTH)
        self._profile_line = self._plot.plot(
            vs_vals, depths, pen=pen, name=label,
        )
        self._restore_view_state(state)

    def update_profile(
        self,
        layers: List[Dict],
        color: Optional[str] = None,
    ) -> None:
        """Fast update of an existing profile line; falls back to plot_profile."""
        vs_vals, depths = _layers_to_staircase(layers)
        if self._profile_line is not None:
            self._profile_line.setData(vs_vals, depths)
        else:
            self.plot_profile(layers, color=color)

    # ------------------------------------------------------------------
    # Multi-model overlay
    # ------------------------------------------------------------------

    def plot_multi_profiles(self, models: List[Dict]) -> None:
        """Overlay multiple Vs profiles (best model drawn thicker).

        Parameters
        ----------
        models : list of dict
            Each model has ``layers`` (list of layer dicts) and
            optionally ``misfit``.
        """
        state = self._save_view_state()
        self._clear_single()
        self._clear_multi()

        for i, md in enumerate(models):
            layers = md.get("layers")
            if not layers:
                continue
            vs_vals, depths = _layers_to_staircase(layers)
            color = QColor(MULTI_COLORS[i % len(MULTI_COLORS)])
            lw = BEST_LINE_WIDTH if i == 0 else ENSEMBLE_LINE_WIDTH
            alpha = 230 if i == 0 else 130
            color.setAlpha(alpha)
            misfit = md.get("misfit", 0.0)
            label = f"#{i+1} ({misfit:.4f})" if i < 5 else None
            pen = pg.mkPen(color, width=lw)
            item = self._plot.plot(vs_vals, depths, pen=pen, name=label)
            self._multi_lines.append(item)
        self._restore_view_state(state)

    # ------------------------------------------------------------------
    # Keyed items — individually-addressable staircases (Results view)
    # ------------------------------------------------------------------
    def set_item(
        self, key: str, layers: List[Dict], pen, *, on_top: bool = False,
    ) -> pg.PlotDataItem:
        """Create or update a keyed staircase item (per-scenario model / median).

        Existing keys are fast-updated via ``setData`` + ``setPen`` (no redraw,
        no view reset); a new key is plotted and stored.  ``on_top`` raises the
        z-order (used for the median so it rides above the ensemble).
        """
        vs_vals, depths = _layers_to_staircase(layers)
        item = self._items.get(key)
        if item is None:
            item = self._plot.plot(vs_vals, depths, pen=pen)
            item.setZValue(3 if on_top else 1)
            self._items[key] = item
        else:
            item.setData(vs_vals, depths)
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
    # Cleanup
    # ------------------------------------------------------------------

    def _clear_single(self) -> None:
        if self._profile_line is not None:
            self._plot.removeItem(self._profile_line)
            self._profile_line = None

    def _clear_multi(self) -> None:
        for itm in self._multi_lines:
            self._plot.removeItem(itm)
        self._multi_lines.clear()

    def clear_all(self) -> None:
        self._clear_single()
        self._clear_multi()
        self.clear_items()
        self._plot.enableAutoRange()
