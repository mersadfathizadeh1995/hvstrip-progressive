"""Settings Page — 7 configuration cards with scrollable layout.

Cards:
  1. Engine Selection + Configure button
  2. HVf Executable Path
  3. Default Frequency Range
  4. Plot Settings
  5. Peak Detection
  6. Dual-Resonance
  7. Config File (load/save/reset)
"""
import os
import yaml

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QGroupBox, QFormLayout, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QFileDialog,
    QMessageBox,
)

from ..widgets.style_constants import (
    OUTER_MARGINS, SECONDARY_LABEL, BUTTON_SUCCESS, EMOJI,
)

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]
PEAK_PRESETS = ["default", "forward_modeling", "forward_modeling_sharp", "conservative", "custom"]
PEAK_METHODS = ["find_peaks", "max", "manual"]
PEAK_SELECTS = ["leftmost", "max", "sharpest", "leftmost_sharpest"]


class SettingsPage(QWidget):
    """Global settings page with 7 configuration card groups."""

    settingsSaved = pyqtSignal(dict)

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(*OUTER_MARGINS)

        hdr = QLabel(f"<b>{EMOJI['settings']} Global Settings</b>")
        hdr.setStyleSheet("font-size: 14px; padding: 4px;")
        outer.addWidget(hdr)
        desc = QLabel("Configure application-wide defaults. Changes apply to all pages.")
        desc.setStyleSheet(SECONDARY_LABEL)
        outer.addWidget(desc)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sw = QWidget()
        layout = QVBoxLayout(sw)

        # 1 — Engine Selection
        eng_grp = QGroupBox(f"{EMOJI['engine']} Engine Selection")
        eng_form = QFormLayout(eng_grp)
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(ENGINES)
        eng_form.addRow("Default Engine:", self._engine_combo)
        self._btn_configure = QPushButton("Configure Engines...")
        self._btn_configure.clicked.connect(self._open_engine_settings)
        eng_form.addRow(self._btn_configure)
        layout.addWidget(eng_grp)

        # 2 — HVf Path
        hvf_grp = QGroupBox(f"{EMOJI['file']} HVf Executable")
        hvf_layout = QHBoxLayout(hvf_grp)
        self._hvf_path = QLineEdit()
        self._hvf_path.setPlaceholderText("Auto-detect from bin/exe_Win/HVf.exe")
        hvf_layout.addWidget(self._hvf_path)
        btn_hvf = QPushButton("Browse...")
        btn_hvf.clicked.connect(self._browse_hvf)
        hvf_layout.addWidget(btn_hvf)
        layout.addWidget(hvf_grp)

        # 3 — Frequency Range
        freq_grp = QGroupBox(f"{EMOJI['frequency']} Default Frequency Range")
        freq_form = QFormLayout(freq_grp)
        self._fmin = QDoubleSpinBox(); self._fmin.setRange(0.01, 10); self._fmin.setValue(0.2)
        self._fmax = QDoubleSpinBox(); self._fmax.setRange(1, 100); self._fmax.setValue(20.0)
        self._nf = QSpinBox(); self._nf.setRange(10, 2000); self._nf.setValue(71)
        freq_form.addRow("Freq Min (Hz):", self._fmin)
        freq_form.addRow("Freq Max (Hz):", self._fmax)
        freq_form.addRow("Points:", self._nf)
        layout.addWidget(freq_grp)

        # 4 — Plot Settings
        plot_grp = QGroupBox(f"{EMOJI['chart']} Plot Settings")
        plot_form = QFormLayout(plot_grp)
        self._dpi = QSpinBox(); self._dpi.setRange(72, 600); self._dpi.setValue(150)
        self._x_scale = QComboBox(); self._x_scale.addItems(["log", "linear"])
        self._y_scale = QComboBox(); self._y_scale.addItems(["linear", "log"])
        self._font_size = QSpinBox(); self._font_size.setRange(6, 24); self._font_size.setValue(12)
        plot_form.addRow("DPI:", self._dpi)
        plot_form.addRow("X Axis Scale:", self._x_scale)
        plot_form.addRow("Y Axis Scale:", self._y_scale)
        plot_form.addRow("Font Size:", self._font_size)
        layout.addWidget(plot_grp)

        # 5 — Peak Detection
        peak_grp = QGroupBox(f"{EMOJI['peak']} Peak Detection")
        peak_form = QFormLayout(peak_grp)
        self._peak_preset = QComboBox(); self._peak_preset.addItems(PEAK_PRESETS)
        self._peak_method = QComboBox(); self._peak_method.addItems(PEAK_METHODS)
        self._peak_select = QComboBox(); self._peak_select.addItems(PEAK_SELECTS)
        peak_form.addRow("Preset:", self._peak_preset)
        peak_form.addRow("Method:", self._peak_method)
        peak_form.addRow("Selection Strategy:", self._peak_select)
        layout.addWidget(peak_grp)

        # 6 — Dual-Resonance
        dr_grp = QGroupBox(f"{EMOJI['dual']} Dual-Resonance")
        dr_form = QFormLayout(dr_grp)
        self._dr_ratio = QDoubleSpinBox(); self._dr_ratio.setRange(1.0, 5.0); self._dr_ratio.setValue(1.2); self._dr_ratio.setSingleStep(0.1)
        self._dr_shift = QDoubleSpinBox(); self._dr_shift.setRange(0.01, 5.0); self._dr_shift.setValue(0.3); self._dr_shift.setSingleStep(0.05)
        dr_form.addRow("Separation Ratio (f1/f0):", self._dr_ratio)
        dr_form.addRow("Min Freq Shift (Hz):", self._dr_shift)
        layout.addWidget(dr_grp)

        # 7 — Config File
        cfg_grp = QGroupBox(f"{EMOJI['save']} Configuration File")
        cfg_layout = QVBoxLayout(cfg_grp)
        btn_row = QHBoxLayout()
        btn_load = QPushButton("Load Config...")
        btn_load.clicked.connect(self._load_config_file)
        btn_save = QPushButton("Save Config As...")
        btn_save.clicked.connect(self._save_config_file)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_save)
        cfg_layout.addLayout(btn_row)
        layout.addWidget(cfg_grp)

        layout.addStretch()
        scroll.setWidget(sw)
        outer.addWidget(scroll)

        # Bottom buttons
        bottom = QHBoxLayout()
        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_apply = QPushButton(f"{EMOJI['save']} Save Settings")
        btn_apply.setStyleSheet(BUTTON_SUCCESS)
        btn_apply.clicked.connect(self._save_settings)
        bottom.addWidget(btn_reset)
        bottom.addStretch()
        bottom.addWidget(btn_apply)
        outer.addLayout(bottom)

    # ═══════════════════════════════════════════════════════════
    #  CONFIG <-> UI
    # ═══════════════════════════════════════════════════════════
    def _collect_config(self):
        return {
            "engine": {"name": self._engine_combo.currentText()},
            "hv_forward": {
                "exe_path": self._hvf_path.text().strip(),
                "fmin": self._fmin.value(),
                "fmax": self._fmax.value(),
                "nf": self._nf.value(),
            },
            "plot": {
                "dpi": self._dpi.value(),
                "x_axis_scale": self._x_scale.currentText(),
                "y_axis_scale": self._y_scale.currentText(),
                "font_size": self._font_size.value(),
            },
            "peak_detection": {
                "preset": self._peak_preset.currentText(),
                "method": self._peak_method.currentText(),
                "select": self._peak_select.currentText(),
            },
            "dual_resonance": {
                "separation_ratio_threshold": self._dr_ratio.value(),
                "separation_shift_threshold": self._dr_shift.value(),
            },
        }

    def apply_config(self, cfg):
        """Apply settings from config dict to UI."""
        engine = cfg.get("engine", {})
        name = engine.get("name", "diffuse_field") if isinstance(engine, dict) else str(engine)
        idx = self._engine_combo.findText(name)
        if idx >= 0: self._engine_combo.setCurrentIndex(idx)

        hv = cfg.get("hv_forward", {})
        if "exe_path" in hv: self._hvf_path.setText(str(hv["exe_path"]))
        if "fmin" in hv: self._fmin.setValue(hv["fmin"])
        if "fmax" in hv: self._fmax.setValue(hv["fmax"])
        if "nf" in hv: self._nf.setValue(hv["nf"])

        plot = cfg.get("plot", {})
        if "dpi" in plot: self._dpi.setValue(plot["dpi"])
        if "x_axis_scale" in plot:
            idx = self._x_scale.findText(plot["x_axis_scale"])
            if idx >= 0: self._x_scale.setCurrentIndex(idx)
        if "y_axis_scale" in plot:
            idx = self._y_scale.findText(plot["y_axis_scale"])
            if idx >= 0: self._y_scale.setCurrentIndex(idx)
        if "font_size" in plot: self._font_size.setValue(plot["font_size"])

        peak = cfg.get("peak_detection", {})
        for key, combo in [("preset", self._peak_preset), ("method", self._peak_method), ("select", self._peak_select)]:
            if key in peak:
                idx = combo.findText(peak[key])
                if idx >= 0: combo.setCurrentIndex(idx)

        dr = cfg.get("dual_resonance", {})
        if "separation_ratio_threshold" in dr: self._dr_ratio.setValue(dr["separation_ratio_threshold"])
        if "separation_shift_threshold" in dr: self._dr_shift.setValue(dr["separation_shift_threshold"])

    # ═══════════════════════════════════════════════════════════
    #  ACTIONS
    # ═══════════════════════════════════════════════════════════
    def _save_settings(self):
        cfg = self._collect_config()
        if self._main_window:
            self._main_window.update_config(cfg)
            # Propagate to other pages
            for attr in ["_home_page", "_forward_page", "_figures_page"]:
                page = getattr(self._main_window, attr, None)
                if page and hasattr(page, "apply_config"):
                    try:
                        page.apply_config(cfg)
                    except Exception:
                        pass
        self.settingsSaved.emit(cfg)
        if self._main_window:
            self._main_window.set_status("Settings saved")

    def _reset_defaults(self):
        from ..strip_window import _get_default_config
        self.apply_config(_get_default_config())
        if self._main_window:
            self._main_window.set_status("Settings reset to defaults")

    def _open_engine_settings(self):
        from ..dialogs.engine_settings_dialog import EngineSettingsDialog
        current = {}
        if self._main_window:
            current = self._main_window.get_engine_settings()
        dlg = EngineSettingsDialog(current, parent=self)
        if dlg.exec_() == EngineSettingsDialog.Accepted:
            if self._main_window:
                self._main_window.update_config({"engine_settings": dlg.get_config()})

    def _browse_hvf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select HVf Executable", "",
                                              "Executables (*.exe);;All (*)")
        if path:
            self._hvf_path.setText(path)

    def _load_config_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "",
                                              "YAML (*.yaml *.yml);;All (*)")
        if not path:
            return
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                self.apply_config(cfg)
                QMessageBox.information(self, "Loaded", f"Configuration loaded from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def _save_config_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "",
                                              "YAML (*.yaml);;All (*)")
        if not path:
            return
        try:
            cfg = self._collect_config()
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            QMessageBox.information(self, "Saved", f"Configuration saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
