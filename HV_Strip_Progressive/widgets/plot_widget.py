"""Reusable matplotlib canvas widget for PyQt5."""
import os as _os
# Ensure matplotlib binds to PyQt5, not PySide6 (both may be installed)
_os.environ["QT_API"] = "pyqt5"

import matplotlib
try:
    matplotlib.use("Qt5Agg")
except Exception:
    pass

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


class MatplotlibWidget(QWidget):
    """Embedded matplotlib figure with navigation toolbar."""

    def __init__(self, figsize=(8, 6), parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def get_figure(self):
        return self.figure

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def refresh(self):
        self.canvas.draw_idle()
