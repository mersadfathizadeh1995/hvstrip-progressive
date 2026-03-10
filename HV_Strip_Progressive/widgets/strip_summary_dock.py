"""Strip Summary Dock — right-panel results table for Strip Single mode.

Displays a step-by-step table with peak frequencies, Vs30, VsAvg,
bedrock depth, and status.  Includes summary statistics and CSV export.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QPushButton, QFileDialog,
)

from .style_constants import EMOJI, BUTTON_PRIMARY


COLUMNS = [
    "Step", "Layers", "f0 (Hz)", "Amp", "Vs30", "VsAvg",
    "Bedrock (m)", "Status",
]


class StripSummaryDock(QDockWidget):
    """Dockable summary panel for HV strip results."""

    def __init__(self, parent=None):
        super().__init__("Strip Summary", parent)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # Title
        title = QLabel(f"{EMOJI.get('report', '📋')} Strip Results Summary")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        lay.addWidget(title)

        # Table
        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.verticalHeader().setDefaultSectionSize(22)
        lay.addWidget(self._table, 1)

        # Summary text
        self._summary = QLabel("")
        self._summary.setStyleSheet("font-size: 10px; color: #444;")
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        # Export button
        btn_row = QHBoxLayout()
        self._btn_export = QPushButton(f"{EMOJI.get('save', '💾')} Export CSV")
        self._btn_export.clicked.connect(self._export_csv)
        btn_row.addWidget(self._btn_export)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.setWidget(w)
        self.setMinimumWidth(280)

    # ── Public API ─────────────────────────────────────────────
    def set_results(self, result_dict, peak_data=None):
        """Populate table from workflow result + optional wizard peak data.

        Parameters
        ----------
        result_dict : dict
            From run_complete_workflow(): contains step_results, strip_directory.
        peak_data : dict or None
            From StripWizardView.get_peak_data():
            {step_name: {"f0": tuple, "secondary": [...],
                         "bedrock_depth": float, "vs30": float, "vsavg": float}}
        """
        self._result = result_dict
        self._peak_data = peak_data or {}
        self._populate_table()
        self._update_summary()

    def update_step(self, step_name, data):
        """Live-update a single step row (e.g., during wizard navigation)."""
        for row in range(self._table.rowCount()):
            if self._table.item(row, 0) and \
               self._table.item(row, 0).text() == step_name:
                self._fill_row(row, step_name, data)
                break

    # ── Table population ──────────────────────────────────────
    def _populate_table(self):
        step_results = self._result.get("step_results", {})
        names = sorted(step_results.keys())
        self._table.setRowCount(len(names))

        for i, name in enumerate(names):
            data = step_results[name]
            pk = self._peak_data.get(name, {})
            merged = {**data, **pk}
            self._fill_row(i, name, merged)

    def _fill_row(self, row, name, data):
        # Step name
        self._table.setItem(row, 0, QTableWidgetItem(name))

        # Layers
        n_layers = "?"
        for part in name.split("_"):
            if "-layer" in part:
                try:
                    n_layers = part.split("-")[0]
                except (ValueError, IndexError):
                    pass
        self._table.setItem(row, 1, QTableWidgetItem(str(n_layers)))

        # f0
        f0 = data.get("f0")
        if isinstance(f0, tuple):
            self._table.setItem(row, 2, QTableWidgetItem(f"{f0[0]:.4f}"))
            self._table.setItem(row, 3, QTableWidgetItem(f"{f0[1]:.3f}"))
        else:
            pf = data.get("peak_frequency")
            pa = data.get("peak_amplitude")
            self._table.setItem(row, 2, QTableWidgetItem(
                f"{pf:.4f}" if pf else "—"))
            self._table.setItem(row, 3, QTableWidgetItem(
                f"{pa:.3f}" if pa else "—"))

        # Vs30
        vs30 = data.get("vs30")
        self._table.setItem(row, 4, QTableWidgetItem(
            f"{vs30:.0f}" if isinstance(vs30, (int, float)) and vs30 else "—"))

        # VsAvg
        vsavg = data.get("vsavg")
        self._table.setItem(row, 5, QTableWidgetItem(
            f"{vsavg:.0f}" if isinstance(vsavg, (int, float)) and vsavg else "—"))

        # Bedrock depth
        bd = data.get("bedrock_depth")
        self._table.setItem(row, 6, QTableWidgetItem(
            f"{bd:.1f}" if isinstance(bd, (int, float)) and bd else "—"))

        # Status
        self._table.setItem(row, 7, QTableWidgetItem("OK"))

    def _update_summary(self):
        sr = self._result.get("step_results", {})
        n = len(sr)
        strip_dir = self._result.get("strip_directory", "N/A")

        # Collect frequencies (from peak_data first, then step_results)
        freqs, vs30_vals = [], []
        for name in sorted(sr.keys()):
            pk = self._peak_data.get(name, {})
            f0 = pk.get("f0")
            if isinstance(f0, tuple):
                freqs.append(f0[0])
            elif sr[name].get("peak_frequency"):
                freqs.append(sr[name]["peak_frequency"])
            # Collect Vs30
            v30 = pk.get("vs30") or sr[name].get("vs30")
            if isinstance(v30, (int, float)) and v30:
                vs30_vals.append(v30)

        lines = [f"Steps: {n}"]
        if freqs:
            lines.append(f"f0 range: {min(freqs):.3f} – {max(freqs):.3f} Hz")
            lines.append(f"Initial f0: {freqs[0]:.4f} Hz")
            if len(freqs) > 1:
                shift = abs(freqs[-1] - freqs[0]) / freqs[0] * 100
                lines.append(f"Total shift: {shift:.1f}%")
        if vs30_vals:
            lines.append(f"Vs30 range: {min(vs30_vals):.0f} – "
                         f"{max(vs30_vals):.0f} m/s")

        report = self._result.get("report_files", {})
        if report:
            lines.append(f"Report files: {len(report)}")

        self._summary.setText("\n".join(lines))

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "strip_results.csv",
            "CSV Files (*.csv)")
        if not path:
            return

        with open(path, "w") as f:
            f.write(",".join(COLUMNS) + "\n")
            for row in range(self._table.rowCount()):
                cells = []
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    cells.append(item.text() if item else "")
                f.write(",".join(cells) + "\n")
