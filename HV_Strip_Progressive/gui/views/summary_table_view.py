"""Summary Table View — canvas tab for multi-profile summary table.

Shows a tabular summary of all computed profiles including f0, amplitude,
secondary peaks, and status.  Uses a matplotlib table for rendering.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from ..widgets.plot_widget import MatplotlibWidget


class SummaryTableView(QWidget):
    """Canvas view for displaying a summary table of multiple profiles."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._results = []
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self._plot = MatplotlibWidget(figsize=(14, 6))
        lay.addWidget(self._plot, 1)
        self._info = QLabel("")
        self._info.setStyleSheet("font-size: 11px; color: #555;")
        lay.addWidget(self._info)
        self._draw_empty()

    def _draw_empty(self):
        fig = self._plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No results yet\nCompute profiles to see summary",
                ha="center", va="center", color="gray", fontsize=14,
                transform=ax.transAxes)
        self._plot.refresh()

    def set_results(self, results):
        """Populate the table from a list of ProfileResult objects."""
        self._results = results
        fig = self._plot.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")

        headers = ["Profile", "f0 (Hz)", "Amplitude", "Secondary Peaks", "Status"]
        rows = []
        for r in results:
            if r.computed and r.f0:
                sec = ", ".join(f"{s[0]:.2f}" for s in r.secondary_peaks) or "—"
                rows.append([r.name, f"{r.f0[0]:.3f}", f"{r.f0[1]:.2f}", sec, "✓"])
            else:
                rows.append([r.name, "—", "—", "—", "✗"])

        if rows:
            table = ax.table(
                cellText=rows, colLabels=headers,
                loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.4)
            for j in range(len(headers)):
                table[0, j].set_facecolor("#2E86AB")
                table[0, j].set_text_props(color="white", fontweight="bold")
            for i in range(len(rows)):
                bg = "#f9f9f9" if i % 2 == 0 else "white"
                for j in range(len(headers)):
                    table[i + 1, j].set_facecolor(bg)

        n_ok = sum(1 for r in results if r.computed)
        ax.set_title(
            f"Multi-Profile Summary — {n_ok}/{len(results)} computed",
            fontsize=12, pad=20)
        fig.tight_layout()
        self._plot.refresh()
        self._info.setText(f"{n_ok} of {len(results)} profiles computed")
