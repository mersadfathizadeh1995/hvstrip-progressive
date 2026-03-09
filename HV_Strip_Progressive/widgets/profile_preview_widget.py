"""Vs-depth profile preview widget."""
import os
os.environ["QT_API"] = "pyqt5"

import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    _HAS_CANVAS = True
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        _HAS_CANVAS = True
    except Exception:
        _HAS_CANVAS = False


class ProfilePreviewWidget(QWidget):
    """Compact Vs vs depth step-function preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._profile = None

        if _HAS_CANVAS:
            self.figure = Figure(figsize=(3, 4), tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            layout.addWidget(self.canvas)
            self._draw_empty()
        else:
            self.figure = None
            self.canvas = None
            lbl = QLabel("Matplotlib canvas unavailable")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: gray;")
            layout.addWidget(lbl)

    # -- public API --
    def set_profile(self, profile):
        self._profile = profile
        self._redraw()

    def refresh(self):
        self._redraw()

    # -- drawing --
    def _draw_empty(self):
        if not self.figure:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "No profile loaded", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def _redraw(self):
        if not self.figure:
            return
        if self._profile is None or not self._profile.layers:
            self._draw_empty()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        depths, vs_vals = [], []
        z = 0.0
        finite_layers = [L for L in self._profile.layers if not L.is_halfspace]
        hs = [L for L in self._profile.layers if L.is_halfspace]

        for L in finite_layers:
            depths.append(z)
            vs_vals.append(L.vs)
            z += L.thickness
            depths.append(z)
            vs_vals.append(L.vs)

        total_finite = z
        if hs:
            hs_depth = total_finite * 0.25
            depths.append(z)
            vs_vals.append(hs[0].vs)
            z += hs_depth
            depths.append(z)
            vs_vals.append(hs[0].vs)

        ax.plot(vs_vals, depths, color="teal", linewidth=1.8)
        ax.axhline(total_finite, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=8)
        ax.set_ylabel("Depth (m)", fontsize=8)
        ax.set_title(f"{len(finite_layers)}L", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        self.canvas.draw_idle()
