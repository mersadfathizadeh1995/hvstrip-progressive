"""
Batch Settings Dialog.

Comprehensive pre-batch configuration popup that lets users set
frequency parameters, engine, analysis options, figure defaults,
and peak detection before launching the batch workflow.
"""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from .dual_resonance_settings_dialog import DualResonanceSettingsDialog


class BatchSettingsDialog(QDialog):
    """Pre-batch settings popup — all config in one place."""

    def __init__(self, parent=None, defaults=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Settings")
        self.setMinimumWidth(520)
        d = defaults or {}

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_frequency_tab(d), "Frequency")
        self.tabs.addTab(self._build_options_tab(d), "Options")
        self.tabs.addTab(self._build_figure_tab(d), "Figure Defaults")
        self.tabs.addTab(self._build_peak_tab(d), "Peak Detection")
        layout.addWidget(self.tabs)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("Run Batch")
        btn_ok.setStyleSheet(
            "background-color: #107c10; color: white; "
            "padding: 8px 24px; font-weight: bold;"
        )
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ tabs

    def _build_frequency_tab(self, d) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.fmin_spin = QDoubleSpinBox()
        self.fmin_spin.setRange(0.01, 10.0)
        self.fmin_spin.setValue(d.get("fmin", 0.5))
        self.fmin_spin.setSingleStep(0.1)
        self.fmin_spin.setSuffix(" Hz")
        form.addRow("Freq Min:", self.fmin_spin)

        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(1.0, 100.0)
        self.fmax_spin.setValue(d.get("fmax", 20.0))
        self.fmax_spin.setSingleStep(1.0)
        self.fmax_spin.setSuffix(" Hz")
        form.addRow("Freq Max:", self.fmax_spin)

        self.nf_spin = QSpinBox()
        self.nf_spin.setRange(50, 2000)
        self.nf_spin.setValue(d.get("nf", 500))
        form.addRow("Num Points:", self.nf_spin)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["diffuse_field", "sh_wave", "ellipticity"])
        self.engine_combo.setCurrentText(d.get("engine", "diffuse_field"))
        form.addRow("Engine:", self.engine_combo)

        return w

    def _build_options_tab(self, d) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self.chk_report = QCheckBox("Generate comprehensive report")
        self.chk_report.setChecked(d.get("generate_report", True))
        layout.addWidget(self.chk_report)

        dr_row = QHBoxLayout()
        self.chk_dual_resonance = QCheckBox("Run dual-resonance analysis")
        self.chk_dual_resonance.setChecked(
            d.get("dual_resonance", {}).get("enable", False)
        )
        dr_row.addWidget(self.chk_dual_resonance)

        self._dr_ratio = d.get("dual_resonance", {}).get(
            "separation_ratio_threshold", 1.2
        )
        self._dr_shift = d.get("dual_resonance", {}).get(
            "separation_shift_threshold", 0.3
        )
        btn_dr = QPushButton("\u2699")
        btn_dr.setMaximumWidth(30)
        btn_dr.setToolTip("Dual-resonance thresholds")
        btn_dr.clicked.connect(self._open_dr_settings)
        dr_row.addWidget(btn_dr)
        dr_row.addStretch()
        layout.addLayout(dr_row)

        layout.addStretch()
        return w

    def _build_figure_tab(self, d) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        fig = d.get("figure", {})

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(fig.get("dpi", 300))
        form.addRow("DPI:", self.dpi_spin)

        self.fig_w_spin = QDoubleSpinBox()
        self.fig_w_spin.setRange(4.0, 24.0)
        self.fig_w_spin.setValue(fig.get("width", 10.0))
        self.fig_w_spin.setSuffix(" in")
        form.addRow("Width:", self.fig_w_spin)

        self.fig_h_spin = QDoubleSpinBox()
        self.fig_h_spin.setRange(3.0, 16.0)
        self.fig_h_spin.setValue(fig.get("height", 6.0))
        self.fig_h_spin.setSuffix(" in")
        form.addRow("Height:", self.fig_h_spin)

        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 24)
        self.font_spin.setValue(fig.get("font_size", 12))
        form.addRow("Font Size:", self.font_spin)

        self.palette_combo = QComboBox()
        self.palette_combo.addItems([
            "Classic", "Bold", "Earth", "Minimal", "Nordic", "Sunset",
            "Blues", "Greens", "Reds", "Grays", "Grayscale", "Vivid",
        ])
        self.palette_combo.setCurrentText(fig.get("palette", "Classic"))
        form.addRow("Color Palette:", self.palette_combo)

        self.chk_log_x = QCheckBox()
        self.chk_log_x.setChecked(fig.get("log_x", True))
        form.addRow("Log X axis:", self.chk_log_x)

        self.chk_log_y = QCheckBox()
        self.chk_log_y.setChecked(fig.get("log_y", False))
        form.addRow("Log Y axis:", self.chk_log_y)

        self.chk_grid = QCheckBox()
        self.chk_grid.setChecked(fig.get("grid", True))
        form.addRow("Show Grid:", self.chk_grid)

        fmt_row = QHBoxLayout()
        self.chk_png = QCheckBox("PNG")
        self.chk_png.setChecked(fig.get("save_png", True))
        fmt_row.addWidget(self.chk_png)
        self.chk_pdf = QCheckBox("PDF")
        self.chk_pdf.setChecked(fig.get("save_pdf", False))
        fmt_row.addWidget(self.chk_pdf)
        self.chk_svg = QCheckBox("SVG")
        self.chk_svg.setChecked(fig.get("save_svg", False))
        fmt_row.addWidget(self.chk_svg)
        fmt_row.addStretch()
        form.addRow("Save Formats:", fmt_row)

        return w

    def _build_peak_tab(self, d) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        pk = d.get("peak_detection", {})

        self.peak_preset_combo = QComboBox()
        self.peak_preset_combo.addItems([
            "default", "forward_modeling", "forward_modeling_sharp",
            "conservative", "custom",
        ])
        self.peak_preset_combo.setCurrentText(pk.get("preset", "default"))
        form.addRow("Preset:", self.peak_preset_combo)

        self.peak_method_combo = QComboBox()
        self.peak_method_combo.addItems(["max", "find_peaks", "manual"])
        self.peak_method_combo.setCurrentText(pk.get("method", "find_peaks"))
        form.addRow("Detection Method:", self.peak_method_combo)

        self.peak_select_combo = QComboBox()
        self.peak_select_combo.addItems([
            "leftmost", "max", "sharpest", "leftmost_sharpest",
        ])
        self.peak_select_combo.setCurrentText(pk.get("select", "leftmost"))
        form.addRow("Peak Selection:", self.peak_select_combo)

        return w

    # -------------------------------------------------------------- helpers

    def _open_dr_settings(self):
        dlg = DualResonanceSettingsDialog(
            self, ratio=self._dr_ratio, shift=self._dr_shift,
        )
        if dlg.exec():
            vals = dlg.get_values()
            self._dr_ratio = vals["separation_ratio_threshold"]
            self._dr_shift = vals["separation_shift_threshold"]

    def get_config(self) -> dict:
        """Return the full batch config dict."""
        return {
            "hv_forward": {
                "fmin": self.fmin_spin.value(),
                "fmax": self.fmax_spin.value(),
                "nf": self.nf_spin.value(),
            },
            "engine_name": self.engine_combo.currentText(),
            "generate_report": self.chk_report.isChecked(),
            "dual_resonance": {
                "enable": self.chk_dual_resonance.isChecked(),
                "separation_ratio_threshold": self._dr_ratio,
                "separation_shift_threshold": self._dr_shift,
            },
            "figure": {
                "dpi": self.dpi_spin.value(),
                "width": self.fig_w_spin.value(),
                "height": self.fig_h_spin.value(),
                "font_size": self.font_spin.value(),
                "palette": self.palette_combo.currentText(),
                "log_x": self.chk_log_x.isChecked(),
                "log_y": self.chk_log_y.isChecked(),
                "grid": self.chk_grid.isChecked(),
                "save_png": self.chk_png.isChecked(),
                "save_pdf": self.chk_pdf.isChecked(),
                "save_svg": self.chk_svg.isChecked(),
            },
            "peak_detection": {
                "preset": self.peak_preset_combo.currentText(),
                "method": self.peak_method_combo.currentText(),
                "select": self.peak_select_combo.currentText(),
            },
        }
