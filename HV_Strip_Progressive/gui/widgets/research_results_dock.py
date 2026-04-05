"""ResearchResultsDock — right-side dock with engine stats and agreement tables."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QLabel, QHeaderView,
)


class ResearchResultsDock(QDockWidget):
    """Dock widget showing research comparison metrics.

    Has two tabs:
    1. Engine Statistics — per-engine success/f0/runtime
    2. Pairwise Agreement — f0 differences, correlation, agreement rate
    """

    def __init__(self, parent=None):
        super().__init__("Research Results", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetMovable)

        self._tabs = QTabWidget()
        self._build_engine_tab()
        self._build_agreement_tab()
        self._build_summary_tab()
        self.setWidget(self._tabs)

    # ── Engine Stats Tab ────────────────────────────────────────

    def _build_engine_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        self._engine_label = QLabel("No data yet")
        self._engine_label.setStyleSheet("color: #888;")
        lay.addWidget(self._engine_label)

        self._engine_table = QTableWidget(0, 6)
        self._engine_table.setHorizontalHeaderLabels([
            "Engine", "Success %", "f₀ Mean", "f₀ Std",
            "Peaks", "Time (s)",
        ])
        self._engine_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._engine_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._engine_table.setAlternatingRowColors(True)
        lay.addWidget(self._engine_table)

        self._tabs.addTab(w, "📊 Engine Stats")

    # ── Pairwise Agreement Tab ──────────────────────────────────

    def _build_agreement_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        self._agree_label = QLabel("No data yet")
        self._agree_label.setStyleSheet("color: #888;")
        lay.addWidget(self._agree_label)

        self._agree_table = QTableWidget(0, 7)
        self._agree_table.setHorizontalHeaderLabels([
            "Engine A", "Engine B", "N", "Δf₀ (Hz)",
            "Ratio", "Corr.", "Agree %",
        ])
        self._agree_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._agree_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._agree_table.setAlternatingRowColors(True)
        lay.addWidget(self._agree_table)

        self._tabs.addTab(w, "📏 Agreement")

    # ── Summary Tab ─────────────────────────────────────────────

    def _build_summary_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        self._summary_text = QLabel(
            "Run a comparison study to see results here.")
        self._summary_text.setWordWrap(True)
        self._summary_text.setAlignment(Qt.AlignTop)
        lay.addWidget(self._summary_text)

        self._tabs.addTab(w, "📋 Summary")

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API — called by strip_window.update_research_results()
    # ══════════════════════════════════════════════════════════════

    def update_phase(self, phase_name, result):
        """Update dock content based on completed pipeline phase."""
        if phase_name == "metrics":
            self._populate_metrics(result)
        elif phase_name == "comparison":
            self._update_summary_comparison(result)
        elif phase_name == "profiles":
            self._update_summary_profiles(result)
        elif phase_name == "report":
            self._update_summary_report(result)

    def _populate_metrics(self, result):
        """Populate engine stats and agreement tables from metrics result."""
        # Engine stats
        stats = result.get("engine_stats", [])
        self._engine_table.setRowCount(len(stats))
        for i, es in enumerate(stats):
            name = es.get("engine_name", "?")
            self._engine_table.setItem(i, 0, QTableWidgetItem(name))
            self._engine_table.setItem(
                i, 1, QTableWidgetItem(f"{es.get('success_rate', 0):.0%}"))
            self._engine_table.setItem(
                i, 2, QTableWidgetItem(f"{es.get('f0_mean', 0):.2f}"))
            self._engine_table.setItem(
                i, 3, QTableWidgetItem(f"{es.get('f0_std', 0):.2f}"))
            self._engine_table.setItem(
                i, 4, QTableWidgetItem(f"{es.get('mean_n_peaks', 0):.1f}"))
            self._engine_table.setItem(
                i, 5, QTableWidgetItem(f"{es.get('mean_time', 0):.2f}"))

        self._engine_label.setText(f"{len(stats)} engines analyzed")

        # Note: pairwise agreement would need full metrics object
        n_pa = result.get("n_peak_agreements", 0)
        self._agree_label.setText(
            f"{n_pa} pairwise comparisons computed")

    def _update_summary_comparison(self, result):
        total = result.get("total_runs", 0)
        success = result.get("successful_runs", 0)
        elapsed = result.get("elapsed_seconds", 0)
        self._summary_text.setText(
            f"Comparison: {success}/{total} successful runs\n"
            f"Elapsed: {elapsed:.1f}s")

    def _update_summary_profiles(self, result):
        n = result.get("n_profiles", 0)
        self._summary_text.setText(f"Profiles loaded: {n}")

    def _update_summary_report(self, result):
        n = result.get("n_files", 0)
        self._summary_text.setText(
            f"Report generated: {n} files\n"
            f"See Figures and Report tabs for details.")

    def set_agreement_data(self, agreements):
        """Populate the pairwise agreement table directly."""
        self._agree_table.setRowCount(len(agreements))
        for i, pa in enumerate(agreements):
            self._agree_table.setItem(i, 0, QTableWidgetItem(
                pa.get("engine_a", "")))
            self._agree_table.setItem(i, 1, QTableWidgetItem(
                pa.get("engine_b", "")))
            self._agree_table.setItem(i, 2, QTableWidgetItem(
                str(pa.get("n_both_have_peaks", 0))))
            self._agree_table.setItem(i, 3, QTableWidgetItem(
                f"{pa.get('mean_freq_difference', 0):.3f}"))
            self._agree_table.setItem(i, 4, QTableWidgetItem(
                f"{pa.get('mean_freq_ratio', 0):.3f}"))
            self._agree_table.setItem(i, 5, QTableWidgetItem(
                f"{pa.get('correlation', 0):.3f}"))
            self._agree_table.setItem(i, 6, QTableWidgetItem(
                f"{pa.get('agreement_rate', 0):.1%}"))

        self._agree_label.setText(f"{len(agreements)} pairs")
