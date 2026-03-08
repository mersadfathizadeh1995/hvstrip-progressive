"""
Matplotlib canvas widget for PyQt5.

Provides a reusable FigureCanvas + NavigationToolbar combination that
can be embedded in any panel or dialog.
"""

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QWidget, QVBoxLayout


class PlotCanvas(QWidget):
    """Embeddable matplotlib figure with navigation toolbar."""

    def __init__(self, parent=None, figsize=(10, 6), dpi=100,
                 toolbar: bool = True):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None

        layout.addWidget(self.canvas)

    # Convenience methods

    def add_subplot(self, *args, **kwargs):
        """Create and return a new Axes on the figure."""
        return self.figure.add_subplot(*args, **kwargs)

    def clear(self):
        """Clear all axes from the figure."""
        self.figure.clear()

    def draw(self):
        """Redraw the canvas."""
        self.canvas.draw_idle()

    def tight_layout(self, **kwargs):
        """Apply tight layout to the figure."""
        try:
            self.figure.tight_layout(**kwargs)
        except Exception:
            pass
