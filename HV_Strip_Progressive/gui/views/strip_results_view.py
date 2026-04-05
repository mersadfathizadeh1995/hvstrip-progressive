"""Strip Results View — Tabular summary of stripping workflow results.

Right-panel canvas tab showing a results table (step, layers, peak freq,
Vs30) with summary statistics and CSV export.
"""
import os
import csv

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QTextEdit,
)

from ..widgets.style_constants import (
    MONOSPACE_PREVIEW, BUTTON_PRIMARY, SECONDARY_LABEL, EMOJI,
)


class StripResultsView(QWidget):
    """Canvas view showing stripping results as a table + summary."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._result = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        header = QLabel(f"<b>{EMOJI['report']} Stripping Results Summary</b>")
        header.setStyleSheet("font-size: 13px; padding: 4px;")
        lay.addWidget(header)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "Step", "Layers", "Peak Freq (Hz)", "Peak Amp",
            "Vs30 (m/s)", "Status"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        lay.addWidget(self._table)

        # Summary
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(120)
        self._summary.setStyleSheet(MONOSPACE_PREVIEW)
        self._summary.setPlaceholderText("Run a stripping workflow to see results...")
        lay.addWidget(self._summary)

        # Export button
        btn_row = QHBoxLayout()
        btn_csv = QPushButton(f"{EMOJI['export']} Export CSV")
        btn_csv.setStyleSheet(BUTTON_PRIMARY)
        btn_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_csv)
        btn_row.addStretch()
        lay.addLayout(btn_row)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def set_results(self, result_dict):
        self._result = result_dict
        self._populate_table()
        self._update_summary()

    # ══════════════════════════════════════════════════════════════
    #  TABLE POPULATION
    # ══════════════════════════════════════════════════════════════
    def _populate_table(self):
        self._table.setRowCount(0)
        if not self._result:
            return

        step_results = self._result.get("step_results", {})
        rows = sorted(step_results.items())
        self._table.setRowCount(len(rows))

        for i, (name, data) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(name))

            # Parse layer count from name (e.g., "Step0_5-layer" → 5)
            n_layers = "?"
            for part in name.split("_"):
                if "-layer" in part:
                    try:
                        n_layers = part.split("-")[0]
                    except (ValueError, IndexError):
                        pass
            self._table.setItem(i, 1, QTableWidgetItem(str(n_layers)))

            pf = data.get("peak_frequency", 0)
            pa = data.get("peak_amplitude", 0)
            vs30 = data.get("vs30", "—")
            self._table.setItem(i, 2, QTableWidgetItem(f"{pf:.4f}" if pf else "—"))
            self._table.setItem(i, 3, QTableWidgetItem(f"{pa:.3f}" if pa else "—"))
            self._table.setItem(i, 4, QTableWidgetItem(
                f"{vs30:.1f}" if isinstance(vs30, (int, float)) else str(vs30)))
            self._table.setItem(i, 5, QTableWidgetItem("OK"))

    def _update_summary(self):
        if not self._result:
            return
        sr = self._result.get("step_results", {})
        n = len(sr)
        freqs = [d.get("peak_frequency", 0) for d in sr.values() if d.get("peak_frequency")]
        strip_dir = self._result.get("strip_directory", "N/A")

        lines = [
            f"Steps completed: {n}",
            f"Output: {strip_dir}",
        ]
        if freqs:
            lines.append(f"Frequency range: {min(freqs):.3f} – {max(freqs):.3f} Hz")
            lines.append(f"Initial f0: {freqs[0]:.4f} Hz")
            if len(freqs) > 1:
                shift = abs(freqs[-1] - freqs[0]) / freqs[0] * 100
                lines.append(f"Total shift: {shift:.1f}%")

        report = self._result.get("report_files", {})
        if report:
            lines.append(f"Report files: {len(report)} generated")

        self._summary.setText("\n".join(lines))

    # ══════════════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════════════
    def _export_csv(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "strip_results.csv",
            "CSV Files (*.csv);;All (*)")
        if not path:
            return

        sr = self._result.get("step_results", {})
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Step", "Peak_Frequency_Hz", "Peak_Amplitude",
                         "Vs30_m_s", "N_Frequencies"])
            for name, data in sorted(sr.items()):
                w.writerow([
                    name,
                    f"{data.get('peak_frequency', ''):.6f}",
                    f"{data.get('peak_amplitude', ''):.6f}",
                    data.get("vs30", ""),
                    data.get("n_frequencies", ""),
                ])
        if self._mw:
            self._mw.log(f"Results exported to {path}")
