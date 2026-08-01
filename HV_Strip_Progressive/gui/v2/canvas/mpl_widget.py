"""``MplFigureWidget`` — the house matplotlib canvas (Round-2 foundation).

The user's call: figures go COMPLETELY matplotlib, ported from the legacy
GUI's components (which carry the mpl navigation toolbar + interactive
picking) rather than re-skinning the pyqtgraph views.  This widget is the
base every v2 mpl figure builds on: ``FigureCanvasQTAgg`` +
``NavigationToolbar2QT`` (home/back/pan/zoom/save — the legacy toolbar), a
theme hook that recolors the figure for light/gray/dark, and offscreen
safety.  PySide6 ``backend_qtagg`` ONLY — never ``matplotlib.use("Qt5Agg")``
(the legacy poison).
"""

from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MplFigureWidget(QWidget):
    """A themed matplotlib figure + the navigation toolbar.

    Subclasses draw into :attr:`figure` and call :meth:`style_axes` on each
    axes after (re)drawing so theme colors apply; :meth:`apply_theme`
    restyles live axes and is safe to call any time.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        figsize=(8.0, 5.0),
        toolbar: bool = True,
    ) -> None:
        super().__init__(parent)
        self._palette = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.figure = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar: Optional[NavigationToolbar2QT] = None
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setProperty("role", "mplToolbar")
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    # ------------------------------------------------------------------
    #  Theme
    # ------------------------------------------------------------------
    @property
    def palette_(self):
        """The last applied house palette (``None`` before first theme)."""
        return self._palette

    def apply_theme(self, palette) -> None:
        """Recolor the figure + every live axes from a house ``Palette``."""
        self._palette = palette
        self.figure.set_facecolor(palette.card)
        for ax in self.figure.axes:
            self.style_axes(ax)
        self.canvas.draw_idle()

    def style_axes(self, ax) -> None:
        """Apply the current palette to one axes (no-op before a theme)."""
        p = self._palette
        if p is None:
            return
        ax.set_facecolor(p.card)
        ax.tick_params(colors=p.muted, labelcolor=p.fg)
        for spine in ax.spines.values():
            spine.set_color(p.border_strong)
        ax.xaxis.label.set_color(p.fg)
        ax.yaxis.label.set_color(p.fg)
        ax.title.set_color(p.fg)
        ax.grid(True, alpha=0.3, color=p.muted)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(p.card_alt)
            legend.get_frame().set_edgecolor(p.border)
            for text in legend.get_texts():
                text.set_color(p.fg)

    # ------------------------------------------------------------------
    def draw_placeholder(self, text: str) -> None:
        """One centred muted message (empty states)."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        color = self._palette.muted if self._palette else "gray"
        ax.text(0.5, 0.5, text, ha="center", va="center",
                transform=ax.transAxes, color=color, fontsize=10)
        ax.set_axis_off()
        self.canvas.draw_idle()


__all__ = ["MplFigureWidget"]
