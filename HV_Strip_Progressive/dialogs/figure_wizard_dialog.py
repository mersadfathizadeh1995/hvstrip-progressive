"""Figure Wizard Dialog — preview and export report figures.

Shows generated figures from analysis results with export controls.
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QTabWidget, QMessageBox, QGroupBox, QFormLayout,
)

from ..widgets.plot_widget import MatplotlibWidget


class FigureWizardDialog(QDialog):
    """Preview and export analysis report figures."""

    def __init__(self, result=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Wizard — Report Export")
        self.resize(1000, 700)
        self._result = result or {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Top — settings
        settings = QGroupBox("Export Settings")
        sform = QFormLayout(settings)
        self._dpi = QSpinBox(); self._dpi.setRange(72, 600); self._dpi.setValue(300)
        self._width = QDoubleSpinBox(); self._width.setRange(4, 20); self._width.setValue(10)
        self._height = QDoubleSpinBox(); self._height.setRange(3, 15); self._height.setValue(6)
        self._fmt = QComboBox(); self._fmt.addItems(["PNG", "PDF", "SVG", "EPS"])
        sform.addRow("DPI:", self._dpi)
        sform.addRow("Width (in):", self._width)
        sform.addRow("Height (in):", self._height)
        sform.addRow("Format:", self._fmt)

        self._chk_png = QCheckBox("PNG"); self._chk_png.setChecked(True)
        self._chk_pdf = QCheckBox("PDF"); self._chk_pdf.setChecked(True)
        self._chk_svg = QCheckBox("SVG")
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(self._chk_png)
        fmt_row.addWidget(self._chk_pdf)
        fmt_row.addWidget(self._chk_svg)
        sform.addRow("Formats:", fmt_row)
        layout.addWidget(settings)

        # Preview tabs
        self._tabs = QTabWidget()
        figure_types = [
            "HV Curve", "Vs Profile", "HV Overlay", "Peak Evolution",
            "Summary Report", "Combined Strip",
        ]
        self._plots = {}
        for name in figure_types:
            pw = MatplotlibWidget(figsize=(10, 6))
            self._plots[name] = pw
            self._tabs.addTab(pw, name)
        layout.addWidget(self._tabs)

        # Generate preview
        self._generate_previews()

        # Buttons
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Previews")
        btn_refresh.clicked.connect(self._generate_previews)
        btn_export = QPushButton("Export All Figures")
        btn_export.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        btn_export.clicked.connect(self._export_all)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_refresh)
        btn_row.addStretch()
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _generate_previews(self):
        """Generate preview figures from result data."""
        for name, pw in self._plots.items():
            fig = pw.get_figure()
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"{name}\n(Preview will appear after analysis)",
                    ha="center", va="center", fontsize=11, color="gray",
                    transform=ax.transAxes)
            ax.set_title(name)
            fig.tight_layout()
            pw.refresh()

    def _export_all(self):
        d = QFileDialog.getExistingDirectory(self, "Export All Figures To")
        if not d:
            return
        dpi = self._dpi.value()
        formats = []
        if self._chk_png.isChecked(): formats.append("png")
        if self._chk_pdf.isChecked(): formats.append("pdf")
        if self._chk_svg.isChecked(): formats.append("svg")
        if not formats:
            formats = [self._fmt.currentText().lower()]

        count = 0
        for name, pw in self._plots.items():
            for fmt in formats:
                path = os.path.join(d, f"{name.replace(' ', '_').lower()}.{fmt}")
                pw.get_figure().savefig(path, dpi=dpi, bbox_inches="tight")
                count += 1

        QMessageBox.information(self, "Exported", f"Exported {count} figures to {d}")
