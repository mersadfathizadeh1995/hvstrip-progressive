"""Batch Settings Dialog — 4-tab pre-batch configuration.

Shown before batch processing to confirm/adjust settings.
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QPushButton, QLabel,
)


class BatchSettingsDialog(QDialog):
    """Pre-batch configuration dialog."""

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Settings")
        self.setMinimumWidth(480)
        self._config = config or {}
        self._build_ui()
        self._load_config(self._config)

    def get_config(self):
        return {
            "frequency": {
                "fmin": self.fmin.value(),
                "fmax": self.fmax.value(),
                "nf": self.nf.value(),
                "engine": self.engine_combo.currentText(),
            },
            "options": {
                "generate_report": self.chk_report.isChecked(),
                "dual_resonance": self.chk_dual.isChecked(),
            },
            "figure_defaults": {
                "dpi": self.dpi.value(),
                "width": self.fig_width.value(),
                "height": self.fig_height.value(),
                "font_size": self.font_size.value(),
                "palette": self.palette_combo.currentText(),
                "log_x": self.chk_log_x.isChecked(),
                "log_y": self.chk_log_y.isChecked(),
                "grid": self.chk_grid.isChecked(),
                "save_png": self.chk_png.isChecked(),
                "save_pdf": self.chk_pdf.isChecked(),
                "save_svg": self.chk_svg.isChecked(),
            },
            "peak_detection": {
                "preset": self.peak_preset.currentText(),
                "method": self.peak_method.currentText(),
                "select": self.peak_select.currentText(),
            },
        }

    def _build_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(self._build_freq_tab(), "Frequency")
        tabs.addTab(self._build_options_tab(), "Options")
        tabs.addTab(self._build_figure_tab(), "Figure Defaults")
        tabs.addTab(self._build_peak_tab(), "Peak Detection")

        btn_row = QHBoxLayout()
        btn_run = QPushButton("Run Batch")
        btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        btn_run.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_run)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def _build_freq_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        self.fmin = QDoubleSpinBox(); self.fmin.setRange(0.01, 10); self.fmin.setValue(0.5)
        self.fmax = QDoubleSpinBox(); self.fmax.setRange(1, 100); self.fmax.setValue(20.0)
        self.nf = QSpinBox(); self.nf.setRange(50, 2000); self.nf.setValue(500)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["diffuse_field", "sh_wave", "ellipticity"])
        form.addRow("Freq Min (Hz):", self.fmin)
        form.addRow("Freq Max (Hz):", self.fmax)
        form.addRow("Points:", self.nf)
        form.addRow("Engine:", self.engine_combo)
        return w

    def _build_options_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        self.chk_report = QCheckBox("Generate comprehensive report")
        self.chk_report.setChecked(True)
        self.chk_dual = QCheckBox("Run dual-resonance analysis")
        form.addRow(self.chk_report)
        form.addRow(self.chk_dual)
        return w

    def _build_figure_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        self.dpi = QSpinBox(); self.dpi.setRange(72, 600); self.dpi.setValue(150)
        self.fig_width = QDoubleSpinBox(); self.fig_width.setRange(4, 20); self.fig_width.setValue(8)
        self.fig_height = QDoubleSpinBox(); self.fig_height.setRange(3, 15); self.fig_height.setValue(6)
        self.font_size = QSpinBox(); self.font_size.setRange(6, 24); self.font_size.setValue(12)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems([
            "tab10", "Set1", "Set2", "Dark2", "Pastel1", "viridis", "plasma",
            "inferno", "magma", "cividis", "coolwarm", "Spectral"])
        form.addRow("DPI:", self.dpi)
        form.addRow("Width (in):", self.fig_width)
        form.addRow("Height (in):", self.fig_height)
        form.addRow("Font Size:", self.font_size)
        form.addRow("Palette:", self.palette_combo)

        self.chk_log_x = QCheckBox("Log X"); self.chk_log_x.setChecked(True)
        self.chk_log_y = QCheckBox("Log Y")
        self.chk_grid = QCheckBox("Grid"); self.chk_grid.setChecked(True)
        form.addRow(self.chk_log_x)
        form.addRow(self.chk_log_y)
        form.addRow(self.chk_grid)

        form.addRow(QLabel("Save formats:"))
        self.chk_png = QCheckBox("PNG"); self.chk_png.setChecked(True)
        self.chk_pdf = QCheckBox("PDF"); self.chk_pdf.setChecked(True)
        self.chk_svg = QCheckBox("SVG")
        form.addRow(self.chk_png)
        form.addRow(self.chk_pdf)
        form.addRow(self.chk_svg)
        return w

    def _build_peak_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        self.peak_preset = QComboBox()
        self.peak_preset.addItems(["default", "forward_modeling", "forward_modeling_sharp", "conservative", "custom"])
        self.peak_method = QComboBox()
        self.peak_method.addItems(["find_peaks", "max", "manual"])
        self.peak_select = QComboBox()
        self.peak_select.addItems(["leftmost", "max", "sharpest", "leftmost_sharpest"])
        form.addRow("Preset:", self.peak_preset)
        form.addRow("Method:", self.peak_method)
        form.addRow("Selection:", self.peak_select)
        return w

    def _load_config(self, cfg):
        freq = cfg.get("frequency", {})
        if "fmin" in freq: self.fmin.setValue(freq["fmin"])
        if "fmax" in freq: self.fmax.setValue(freq["fmax"])
        if "nf" in freq: self.nf.setValue(freq["nf"])
