"""Config Panel — Frequency, engine, peak detection, dual-resonance, plot settings.

Replaces the old settings_page.py with CollapsibleGroupBox sections.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QPushButton,
    QFormLayout, QColorDialog,
)

from ..widgets.collapsible_group import CollapsibleGroupBox
from ..widgets.style_constants import (
    OUTER_MARGINS, SECONDARY_LABEL, BUTTON_PRIMARY, BUTTON_SUCCESS,
    GEAR_BUTTON, EMOJI,
)

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]
ENGINE_DESCRIPTIONS = {
    "diffuse_field": "Full diffuse wavefield H/V (HVf.exe)",
    "sh_wave": "SH-wave transfer function (Python)",
    "ellipticity": "Rayleigh wave ellipticity (gpell.exe)",
}


class ConfigPanel(QWidget):
    """Left-panel tab for all analysis configuration."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._build_ui()
        self._apply_defaults()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(*OUTER_MARGINS)

        # ── Frequency Parameters ────────────────────────────────
        freq = CollapsibleGroupBox(f"{EMOJI['frequency']} Frequency Parameters")
        form = QFormLayout()
        self._fmin = QDoubleSpinBox()
        self._fmin.setRange(0.01, 10); self._fmin.setValue(0.2); self._fmin.setDecimals(2)
        self._fmax = QDoubleSpinBox()
        self._fmax.setRange(1, 100); self._fmax.setValue(20.0); self._fmax.setDecimals(1)
        self._nf = QSpinBox()
        self._nf.setRange(10, 2000); self._nf.setValue(71)
        form.addRow("Freq Min (Hz):", self._fmin)
        form.addRow("Freq Max (Hz):", self._fmax)
        form.addRow("Points:", self._nf)
        freq.setContentLayout(form)
        lay.addWidget(freq)

        # ── Engine Selection ────────────────────────────────────
        eng = CollapsibleGroupBox(f"{EMOJI['engine']} Engine Selection")
        eng_lay = QVBoxLayout()
        eng_row = QHBoxLayout()
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(ENGINES)
        self._engine_combo.currentTextChanged.connect(self._on_engine_changed)
        eng_row.addWidget(self._engine_combo)
        gear = QPushButton(EMOJI['settings'])
        gear.setFixedWidth(28); gear.setStyleSheet(GEAR_BUTTON)
        gear.clicked.connect(self._on_engine_settings)
        eng_row.addWidget(gear)
        eng_lay.addLayout(eng_row)
        self._engine_desc = QLabel(ENGINE_DESCRIPTIONS.get("diffuse_field", ""))
        self._engine_desc.setStyleSheet(SECONDARY_LABEL)
        self._engine_desc.setWordWrap(True)
        eng_lay.addWidget(self._engine_desc)
        eng.setContentLayout(eng_lay)
        lay.addWidget(eng)

        # ── Peak Detection ──────────────────────────────────────
        peak = CollapsibleGroupBox(f"{EMOJI['peak']} Peak Detection")
        peak_form = QFormLayout()
        self._peak_preset = QComboBox()
        self._peak_preset.addItems(["default", "sensitive", "strict", "custom"])
        peak_form.addRow("Preset:", self._peak_preset)
        self._peak_method = QComboBox()
        self._peak_method.addItems(["find_peaks", "argmax", "prominence"])
        peak_form.addRow("Method:", self._peak_method)
        self._peak_select = QComboBox()
        self._peak_select.addItems(["leftmost", "highest", "custom"])
        peak_form.addRow("Selection:", self._peak_select)
        self._min_prom = QDoubleSpinBox()
        self._min_prom.setRange(0.0, 50.0); self._min_prom.setValue(0.5)
        self._min_prom.setDecimals(2); self._min_prom.setSingleStep(0.1)
        peak_form.addRow("Min Prominence:", self._min_prom)
        peak.setContentLayout(peak_form)
        lay.addWidget(peak)

        # ── Dual-Resonance ──────────────────────────────────────
        dr = CollapsibleGroupBox(f"{EMOJI['dual']} Dual-Resonance")
        dr_form = QFormLayout()
        self._dr_enable = QCheckBox("Enable dual-resonance analysis")
        dr_form.addRow(self._dr_enable)
        self._dr_ratio = QDoubleSpinBox()
        self._dr_ratio.setRange(1.0, 5.0); self._dr_ratio.setValue(1.2)
        self._dr_ratio.setDecimals(2); self._dr_ratio.setSingleStep(0.1)
        dr_form.addRow("Separation Ratio:", self._dr_ratio)
        self._dr_shift = QDoubleSpinBox()
        self._dr_shift.setRange(0.0, 2.0); self._dr_shift.setValue(0.3)
        self._dr_shift.setDecimals(2); self._dr_shift.setSingleStep(0.05)
        dr_form.addRow("Shift Threshold:", self._dr_shift)
        dr.setContentLayout(dr_form)
        lay.addWidget(dr)

        # ── Plot Settings ───────────────────────────────────────
        plot = CollapsibleGroupBox(f"{EMOJI['plot']} Plot Settings")
        plot_form = QFormLayout()
        self._dpi = QSpinBox()
        self._dpi.setRange(72, 600); self._dpi.setValue(200)
        plot_form.addRow("DPI:", self._dpi)
        self._x_scale = QComboBox()
        self._x_scale.addItems(["log", "linear"])
        plot_form.addRow("X Scale:", self._x_scale)
        self._y_scale = QComboBox()
        self._y_scale.addItems(["log", "linear"])
        plot_form.addRow("Y Scale:", self._y_scale)
        self._font_size = QSpinBox()
        self._font_size.setRange(6, 24); self._font_size.setValue(10)
        plot_form.addRow("Font Size:", self._font_size)
        self._linewidth = QDoubleSpinBox()
        self._linewidth.setRange(0.5, 6.0); self._linewidth.setValue(1.5)
        self._linewidth.setDecimals(1)
        plot_form.addRow("Line Width:", self._linewidth)
        plot.setContentLayout(plot_form)
        lay.addWidget(plot)

        # ── Options ─────────────────────────────────────────────
        opts = CollapsibleGroupBox(f"{EMOJI['config']} Options")
        opts_lay = QVBoxLayout()
        self._chk_report = QCheckBox("Generate comprehensive report")
        self._chk_report.setChecked(True)
        opts_lay.addWidget(self._chk_report)
        self._chk_interactive = QCheckBox("Interactive peak selection")
        opts_lay.addWidget(self._chk_interactive)
        opts.setContentLayout(opts_lay)
        lay.addWidget(opts)

        # ── Save Button ─────────────────────────────────────────
        save_btn = QPushButton(f"{EMOJI['save']} Save Settings")
        save_btn.setStyleSheet(BUTTON_SUCCESS)
        save_btn.clicked.connect(self._save_to_main)
        lay.addWidget(save_btn)

        lay.addStretch()
        scroll.setWidget(inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def get_config(self):
        """Return current panel state as config dict."""
        return {
            "hv_forward": {
                "fmin": self._fmin.value(),
                "fmax": self._fmax.value(),
                "nf": self._nf.value(),
            },
            "engine_name": self._engine_combo.currentText(),
            "peak_detection": {
                "preset": self._peak_preset.currentText(),
                "method": self._peak_method.currentText(),
                "select": self._peak_select.currentText(),
                "min_prominence": self._min_prom.value(),
            },
            "dual_resonance": {
                "enable": self._dr_enable.isChecked(),
                "separation_ratio_threshold": self._dr_ratio.value(),
                "separation_shift_threshold": self._dr_shift.value(),
            },
            "plot": {
                "dpi": self._dpi.value(),
                "x_axis_scale": self._x_scale.currentText(),
                "y_axis_scale": self._y_scale.currentText(),
                "font_size": self._font_size.value(),
                "linewidth": self._linewidth.value(),
            },
            "generate_report": self._chk_report.isChecked(),
            "interactive_mode": self._chk_interactive.isChecked(),
        }

    def get_engine_name(self):
        return self._engine_combo.currentText()

    # ══════════════════════════════════════════════════════════════
    #  INTERNALS
    # ══════════════════════════════════════════════════════════════
    def _apply_defaults(self):
        if self._mw:
            cfg = self._mw.config
            hv = cfg.get("hv_forward", {})
            if "fmin" in hv: self._fmin.setValue(hv["fmin"])
            if "fmax" in hv: self._fmax.setValue(hv["fmax"])
            if "nf" in hv: self._nf.setValue(hv["nf"])
            eng = cfg.get("engine", {})
            name = eng.get("name", "diffuse_field") if isinstance(eng, dict) else "diffuse_field"
            idx = self._engine_combo.findText(name)
            if idx >= 0:
                self._engine_combo.setCurrentIndex(idx)
            dr = cfg.get("dual_resonance", {})
            self._dr_enable.setChecked(dr.get("enable", False))
            if "separation_ratio_threshold" in dr:
                self._dr_ratio.setValue(dr["separation_ratio_threshold"])
            if "separation_shift_threshold" in dr:
                self._dr_shift.setValue(dr["separation_shift_threshold"])

    def _on_engine_changed(self, name):
        self._engine_desc.setText(ENGINE_DESCRIPTIONS.get(name, ""))
        if self._mw:
            self._mw._on_engine_changed(name)

    def _on_engine_settings(self):
        if self._mw:
            self._mw._on_engine_settings()

    def _save_to_main(self):
        if self._mw:
            self._mw.update_config(self.get_config())
            self._mw.log("Settings saved")
            self._mw.set_status("Settings saved")
