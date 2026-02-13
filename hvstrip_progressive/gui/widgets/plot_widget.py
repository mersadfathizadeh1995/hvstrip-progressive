"""
Matplotlib widget for embedding plots in the GUI.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class MatplotlibWidget(QWidget):
    """Widget for displaying matplotlib plots with toolbar."""

    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Set size policy
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setMinimumHeight(400)

    def get_figure(self):
        """Get the matplotlib figure."""
        return self.figure

    def clear(self):
        """Clear the current plot."""
        self.figure.clear()
        self.canvas.draw()

    def plot(self, *args, **kwargs):
        """Convenience method to create a simple plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(*args, **kwargs)
        self.canvas.draw()
        return ax

    def refresh(self):
        """Refresh the canvas."""
        self.canvas.draw()
