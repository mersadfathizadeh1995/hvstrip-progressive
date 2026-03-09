"""Strip Single Panel — settings and run for single profile HV stripping.

Data loading is handled by the Data Input canvas tab.
This panel contains: engine, frequency, peak detection, dual-resonance,
output dir, report/interactive options, Run button.
"""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QFileDialog, QScrollArea, QProgressBar, QCheckBox,
)

from ..widgets.style_constants import (
    BUTTON_SUCCESS, GEAR_BUTTON, SECONDARY_LABEL, EMOJI,
)
from ..widgets.collapsible_group import CollapsibleGroupBox

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]

PEAK_PRESETS = ["default", "sesame", "custom"]
PEAK_METHODS = ["find_peaks", "argmax"]
PEAK_SELECTIONS = ["leftmost", "highest", "global_max"]


class StripSinglePanel(QWidget):
    """Left-panel content for HV Strip → Single sub-tab."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # ── Settings ───────────────────────────────────────────
        settings_grp = CollapsibleGroupBox(f"{EMOJI['settings']} Settings")
        settings_lay = QVBoxLayout()

        # Engine
        eng_row = QHBoxLayout()
        eng_row.addWidget(QLabel("Engine:"))
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(ENGINES)
        eng_row.addWidget(self._engine_combo, 1)
        self._gear_btn = QPushButton(EMOJI["settings"])
        self._gear_btn.setFixedSize(26, 26)
        self._gear_btn.setStyleSheet(GEAR_BUTTON)
        self._gear_btn.clicked.connect(self._open_engine_settings)
        eng_row.addWidget(self._gear_btn)
        settings_lay.addLayout(eng_row)

        # Frequency
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Fmin:"))
        self._fmin = QDoubleSpinBox()
        self._fmin.setRange(0.01, 10.0)
        self._fmin.setValue(0.2)
        self._fmin.setSingleStep(0.1)
        self._fmin.setDecimals(2)
        freq_row.addWidget(self._fmin)
        freq_row.addWidget(QLabel("Fmax:"))
        self._fmax = QDoubleSpinBox()
        self._fmax.setRange(1.0, 100.0)
        self._fmax.setValue(20.0)
        freq_row.addWidget(self._fmax)
        freq_row.addWidget(QLabel("Pts:"))
        self._nf = QSpinBox()
        self._nf.setRange(50, 2000)
        self._nf.setValue(500)
        freq_row.addWidget(self._nf)
        settings_lay.addLayout(freq_row)

        settings_grp.setContentLayout(settings_lay)
        lay.addWidget(settings_grp)

        # ── Peak Detection ─────────────────────────────────────
        peak_grp = CollapsibleGroupBox(f"{EMOJI['peak']} Peak Detection")
        peak_lay = QVBoxLayout()

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._peak_preset = QComboBox()
        self._peak_preset.addItems(PEAK_PRESETS)
        preset_row.addWidget(self._peak_preset, 1)
        peak_lay.addLayout(preset_row)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._peak_method = QComboBox()
        self._peak_method.addItems(PEAK_METHODS)
        method_row.addWidget(self._peak_method, 1)
        method_row.addWidget(QLabel("Select:"))
        self._peak_select = QComboBox()
        self._peak_select.addItems(PEAK_SELECTIONS)
        method_row.addWidget(self._peak_select, 1)
        peak_lay.addLayout(method_row)

        prom_row = QHBoxLayout()
        prom_row.addWidget(QLabel("Min Prominence:"))
        self._min_prom = QDoubleSpinBox()
        self._min_prom.setRange(0.0, 10.0)
        self._min_prom.setValue(0.1)
        self._min_prom.setSingleStep(0.05)
        self._min_prom.setDecimals(3)
        prom_row.addWidget(self._min_prom)
        prom_row.addStretch()
        peak_lay.addLayout(prom_row)

        peak_grp.setContentLayout(peak_lay)
        lay.addWidget(peak_grp)

        # ── Dual-Resonance ─────────────────────────────────────
        dr_grp = CollapsibleGroupBox(
            f"{EMOJI['dual']} Dual-Resonance", collapsed=True)
        dr_lay = QVBoxLayout()

        self._chk_dr = QCheckBox("Enable dual-resonance analysis")
        dr_lay.addWidget(self._chk_dr)

        dr_params = QHBoxLayout()
        dr_params.addWidget(QLabel("Ratio thresh:"))
        self._dr_ratio = QDoubleSpinBox()
        self._dr_ratio.setRange(0.5, 5.0)
        self._dr_ratio.setValue(1.2)
        self._dr_ratio.setSingleStep(0.1)
        dr_params.addWidget(self._dr_ratio)
        dr_params.addWidget(QLabel("Shift thresh:"))
        self._dr_shift = QDoubleSpinBox()
        self._dr_shift.setRange(0.0, 2.0)
        self._dr_shift.setValue(0.3)
        self._dr_shift.setSingleStep(0.05)
        dr_params.addWidget(self._dr_shift)
        dr_lay.addLayout(dr_params)

        dr_grp.setContentLayout(dr_lay)
        lay.addWidget(dr_grp)

        # ── Output ─────────────────────────────────────────────
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("Output directory (required)")
        out_row.addWidget(self._out_dir, 1)
        btn_bro = QPushButton("...")
        btn_bro.setFixedWidth(30)
        btn_bro.clicked.connect(self._browse_output)
        out_row.addWidget(btn_bro)
        lay.addLayout(out_row)

        # ── Options ────────────────────────────────────────────
        self._chk_report = QCheckBox("Generate comprehensive report")
        self._chk_report.setChecked(True)
        lay.addWidget(self._chk_report)

        self._chk_interactive = QCheckBox("Interactive peak selection")
        lay.addWidget(self._chk_interactive)

        # ── Run ────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self._btn_run = QPushButton(f"{EMOJI['run']} Run Stripping")
        self._btn_run.setStyleSheet(BUTTON_SUCCESS)
        self._btn_run.clicked.connect(self._run_stripping)
        run_row.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        run_row.addWidget(self._btn_cancel)
        lay.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._result_label = QLabel("")
        self._result_label.setStyleSheet(SECONDARY_LABEL)
        self._result_label.setWordWrap(True)
        lay.addWidget(self._result_label)

        lay.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._apply_config()

    def _apply_config(self):
        if not self._mw:
            return
        cfg = self._mw.config
        engine = self._mw.get_engine_name()
        idx = self._engine_combo.findText(engine)
        if idx >= 0:
            self._engine_combo.setCurrentIndex(idx)
        fwd = cfg.get("hv_forward", {})
        self._fmin.setValue(fwd.get("fmin", 0.2))
        self._fmax.setValue(fwd.get("fmax", 20.0))
        self._nf.setValue(fwd.get("nf", 71))
        pd = cfg.get("peak_detection", {})
        idx = self._peak_preset.findText(pd.get("preset", "default"))
        if idx >= 0:
            self._peak_preset.setCurrentIndex(idx)
        dr = cfg.get("dual_resonance", {})
        self._chk_dr.setChecked(dr.get("enable", False))
        self._dr_ratio.setValue(dr.get("separation_ratio_threshold", 1.2))
        self._dr_shift.setValue(dr.get("separation_shift_threshold", 0.3))

    def _get_data_input(self):
        """Get the DataInputView from the canvas."""
        if self._mw:
            from ..strip_window import MODE_STRIP_SINGLE
            return self._mw.get_data_input(MODE_STRIP_SINGLE)
        return None

    # ── Slots ──────────────────────────────────────────────────
    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self._out_dir.setText(d)

    def _open_engine_settings(self):
        if self._mw:
            self._mw._on_engine_settings()

    def _run_stripping(self):
        """Run the complete stripping workflow."""
        di = self._get_data_input()
        profile = di.get_profile() if di else None
        if not profile:
            self._result_label.setText(
                "Load a profile in the Data Input tab first.")
            self._result_label.setStyleSheet("color: orange; font-size: 11px;")
            return

        out_dir = self._out_dir.text().strip()
        if not out_dir:
            self._result_label.setText("Set an output directory.")
            self._result_label.setStyleSheet("color: orange; font-size: 11px;")
            return

        # Write temp model file
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w")
        tmp.write(profile.to_hvf_format())
        tmp.close()

        config = self._build_workflow_config()
        engine_name = self._engine_combo.currentText()
        config["engine"] = {"name": engine_name}

        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._result_label.setText("Running stripping workflow...")

        try:
            from ..workers.workflow_worker import WorkflowWorker
            self._worker = WorkflowWorker(
                tmp.name, out_dir, config)
            self._worker.finished_signal.connect(
                lambda res: self._on_strip_done(res, tmp.name))
            self._worker.error.connect(
                lambda err: self._on_strip_error(err, tmp.name))
            if hasattr(self._worker, 'progress'):
                self._worker.progress.connect(self._on_progress)
            self._worker.start()
        except ImportError:
            self._result_label.setText("WorkflowWorker not available")
            self._btn_run.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._progress.setVisible(False)
            os.unlink(tmp.name)

    def _on_strip_done(self, result, tmp_path):
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)

        n_steps = len(result.get("step_results", []))
        self._result_label.setText(f"Stripping completed: {n_steps} steps")
        self._result_label.setStyleSheet("color: green; font-size: 11px;")

        if self._mw:
            self._mw.set_result(result)
            strip_dir = result.get("strip_directory")
            if strip_dir:
                self._mw.update_overlay(strip_dir)
            self._mw.update_strip_results(result)
            self._mw.log(f"Stripping done: {n_steps} steps")

            if self._chk_interactive.isChecked():
                self._mw._on_open_peak_picker()

    def _on_strip_error(self, err_msg, tmp_path):
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        self._result_label.setText(f"Error: {err_msg}")
        self._result_label.setStyleSheet("color: red; font-size: 11px;")
        if self._mw:
            self._mw.log(f"Strip error: {err_msg}")

    def _on_progress(self, msg):
        self._result_label.setText(msg)

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        self._result_label.setText("Cancelled")

    def _build_workflow_config(self):
        engine_name = self._engine_combo.currentText()
        cfg = {
            "engine_name": engine_name,
            "generate_report": self._chk_report.isChecked(),
            "interactive_mode": self._chk_interactive.isChecked(),
            "peak_detection": {
                "preset": self._peak_preset.currentText(),
                "method": self._peak_method.currentText(),
                "select": self._peak_select.currentText(),
                "min_prominence": self._min_prom.value(),
            },
            "dual_resonance": {
                "enable": self._chk_dr.isChecked(),
                "separation_ratio_threshold": self._dr_ratio.value(),
                "separation_shift_threshold": self._dr_shift.value(),
            },
            "hv_forward": {
                "fmin": self._fmin.value(),
                "fmax": self._fmax.value(),
                "nf": self._nf.value(),
            },
        }
        if self._mw:
            es = self._mw.get_engine_settings().get(engine_name, {})
            cfg["hv_forward"].update(es)
            cfg["hv_forward"]["fmin"] = self._fmin.value()
            cfg["hv_forward"]["fmax"] = self._fmax.value()
            cfg["hv_forward"]["nf"] = self._nf.value()
        return cfg

    # ── Public API ─────────────────────────────────────────────
    def get_engine_name(self):
        return self._engine_combo.currentText()
