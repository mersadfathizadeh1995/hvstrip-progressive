"""ResearchComparisonView — engine-by-engine HV curve comparison display."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class ResearchComparisonView(QWidget):
    """Canvas view showing per-profile engine comparison.

    Top: navigation bar (prev/next, profile selector).
    Center: matplotlib plot of HV curves from all engines for selected profile.
    Bottom: summary table of results for current profile.
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._dataset = None
        self._current_idx = 0
        self._engine_colors = {
            "diffuse_field": "#2196F3",
            "sh_wave": "#FF9800",
            "ellipticity": "#4CAF50",
        }
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # Navigation
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_prev.clicked.connect(self._go_prev)
        nav.addWidget(self._btn_prev)

        self._profile_combo = QComboBox()
        self._profile_combo.currentIndexChanged.connect(self._on_combo_change)
        nav.addWidget(self._profile_combo, 1)

        self._btn_next = QPushButton("Next ▶")
        self._btn_next.clicked.connect(self._go_next)
        nav.addWidget(self._btn_next)

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #888;")
        nav.addWidget(self._count_label)
        lay.addLayout(nav)

        # Splitter: plot on top, table on bottom
        splitter = QSplitter(Qt.Vertical)

        # Plot area
        if HAS_MPL:
            self._figure = Figure(figsize=(8, 5), dpi=100)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self._ax = self._figure.add_subplot(111)
            splitter.addWidget(self._canvas)
        else:
            self._canvas = None
            lbl = QLabel("matplotlib not available — install it for plots")
            lbl.setAlignment(Qt.AlignCenter)
            splitter.addWidget(lbl)

        # Results table for current profile
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            "Engine", "Success", "f₀ (Hz)", "N Peaks", "Time (s)",
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setMaximumHeight(200)
        splitter.addWidget(self._table)

        lay.addWidget(splitter)

        self._status = QLabel("Run a comparison to see results here.")
        self._status.setStyleSheet("color: #888;")
        lay.addWidget(self._status)

    # ══════════════════════════════════════════════════════════════
    #  DATA
    # ══════════════════════════════════════════════════════════════

    def set_dataset(self, dataset_dict):
        """Accept comparison dataset (as dict from runner results).

        Expected structure: comparisons[], engine_names[], total_runs, etc.
        """
        self._dataset = dataset_dict
        comparisons = dataset_dict.get("comparisons", [])
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()
        for i, comp in enumerate(comparisons):
            name = comp.get("profile_name", f"Profile {i}")
            cat = comp.get("category", "")
            label = f"{name} [{cat}]" if cat else name
            self._profile_combo.addItem(label)
        self._profile_combo.blockSignals(False)

        if comparisons:
            self._current_idx = 0
            self._profile_combo.setCurrentIndex(0)
            self._show_profile(0)
        self._count_label.setText(f"{len(comparisons)} profiles")
        self._status.setText(
            f"Comparison complete: {dataset_dict.get('successful_runs', 0)}/"
            f"{dataset_dict.get('total_runs', 0)} runs")

    def _show_profile(self, idx):
        """Display HV curves and results for profile at index."""
        if not self._dataset:
            return
        comparisons = self._dataset.get("comparisons", [])
        if idx < 0 or idx >= len(comparisons):
            return

        self._current_idx = idx
        comp = comparisons[idx]
        engine_results = comp.get("engine_results", {})

        # Update plot
        if self._canvas:
            self._ax.clear()
            for eng_name, er in engine_results.items():
                if not er.get("success"):
                    continue
                freqs = er.get("frequencies", [])
                amps = er.get("amplitudes", [])
                if freqs and amps:
                    color = self._engine_colors.get(eng_name, "#999")
                    label = eng_name.replace("_", " ").title()
                    self._ax.semilogx(freqs, amps, color=color,
                                      linewidth=1.5, label=label)

            self._ax.set_xlabel("Frequency (Hz)")
            self._ax.set_ylabel("H/V Amplitude")
            self._ax.set_title(
                comp.get("profile_name", f"Profile {idx}"))
            self._ax.legend(fontsize=9)
            self._ax.grid(True, alpha=0.3)
            self._figure.tight_layout()
            self._canvas.draw()

        # Update table
        eng_list = sorted(engine_results.keys())
        self._table.setRowCount(len(eng_list))
        for i, eng in enumerate(eng_list):
            er = engine_results[eng]
            self._table.setItem(i, 0, QTableWidgetItem(eng))
            success = er.get("success", False)
            self._table.setItem(i, 1, QTableWidgetItem(
                "✅" if success else "❌"))
            peaks = er.get("peaks", [])
            f0 = peaks[0].get("frequency", 0) if peaks else 0
            self._table.setItem(i, 2, QTableWidgetItem(
                f"{f0:.2f}" if f0 else "—"))
            self._table.setItem(i, 3, QTableWidgetItem(str(len(peaks))))
            self._table.setItem(i, 4, QTableWidgetItem(
                f"{er.get('elapsed_seconds', 0):.2f}"))

    # ── Navigation ──────────────────────────────────────────────

    def _go_prev(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._profile_combo.setCurrentIndex(self._current_idx)

    def _go_next(self):
        if self._dataset:
            n = len(self._dataset.get("comparisons", []))
            if self._current_idx < n - 1:
                self._current_idx += 1
                self._profile_combo.setCurrentIndex(self._current_idx)

    def _on_combo_change(self, idx):
        if idx >= 0:
            self._show_profile(idx)

    # Called by strip_window.update_research_results()
    def on_comparison_complete(self, result):
        """Handle comparison phase completion — expects dataset-like dict."""
        # The full dataset should be available from the runner
        self._status.setText(
            f"Comparison: {result.get('successful_runs', 0)}/"
            f"{result.get('total_runs', 0)} successful")
