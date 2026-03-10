"""Strip Single Panel — settings and run for single profile HV stripping.

Data loading is handled by the Data Input canvas tab.
Peak detection and dual-resonance settings are now in the HV Strip Wizard.
This panel contains: engine, frequency, output dir, report option, Run button.
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


class StripSinglePanel(QWidget):
    """Left-panel content for HV Strip → Single sub-tab."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._worker = None
        self._auto_peak_cfg = None
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

        # Dual-resonance toggle (simple on/off, details in wizard)
        self._chk_dr = QCheckBox("Enable dual-resonance analysis")
        settings_lay.addWidget(self._chk_dr)

        settings_grp.setContentLayout(settings_lay)
        lay.addWidget(settings_grp)

        # ── Peak Detection ─────────────────────────────────────
        auto_grp = CollapsibleGroupBox(
            f"{EMOJI['peak']} Peak Detection", collapsed=True)
        auto_lay = QVBoxLayout()
        auto_lay.setSpacing(2)

        auto_row = QHBoxLayout()
        self._chk_auto = QCheckBox("Auto-detect peaks")
        self._chk_auto.setChecked(True)
        self._chk_auto.setToolTip(
            "When checked, uses configurable peak detection\n"
            "(prominence + range) instead of simple max.\n"
            "Also auto-detects secondary peaks.")
        auto_row.addWidget(self._chk_auto)
        self._auto_gear = QPushButton(EMOJI["settings"])
        self._auto_gear.setFixedSize(26, 26)
        self._auto_gear.setStyleSheet(GEAR_BUTTON)
        self._auto_gear.setToolTip("Auto Peak Detection Settings")
        self._auto_gear.clicked.connect(self._open_auto_peak_settings)
        auto_row.addWidget(self._auto_gear)
        auto_row.addStretch()
        auto_lay.addLayout(auto_row)

        self._peak_info_label = QLabel("")
        self._peak_info_label.setStyleSheet(SECONDARY_LABEL)
        self._peak_info_label.setWordWrap(True)
        auto_lay.addWidget(self._peak_info_label)

        auto_grp.setContentLayout(auto_lay)
        lay.addWidget(auto_grp)

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
        dr = cfg.get("dual_resonance", {})
        self._chk_dr.setChecked(dr.get("enable", False))

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
            self._mw.update_strip_results(result)
            self._mw.log(f"Stripping done: {n_steps} steps")

            # Pass auto peak config to wizard
            wiz = self._mw.get_strip_wizard()
            if wiz:
                auto_cfg = self.get_auto_peak_config()
                if auto_cfg:
                    wiz.set_auto_peak_config(auto_cfg)

            # Switch to wizard tab (index 1)
            canvas = self._mw._canvas_stacks.get(self._mw._active_mode)
            if canvas and canvas.count() >= 2:
                canvas.setCurrentIndex(1)

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

    def _open_auto_peak_settings(self):
        """Open the Auto Peak Detection settings dialog."""
        try:
            from ..dialogs.auto_peak_settings_dialog import AutoPeakSettingsDialog
            dlg = AutoPeakSettingsDialog(parent=self)
            if self._auto_peak_cfg:
                dlg._load_config(self._auto_peak_cfg)
            if dlg.exec_() == AutoPeakSettingsDialog.Accepted:
                self._auto_peak_cfg = dlg.get_config()
                self._peak_info_label.setText(
                    f"Prominence: {self._auto_peak_cfg.get('min_prominence', 0.3)}, "
                    f"Secondary: {self._auto_peak_cfg.get('n_secondary', 1)}")
        except ImportError:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Settings",
                                "AutoPeakSettingsDialog not available")

    def _build_workflow_config(self):
        engine_name = self._engine_combo.currentText()
        cfg = {
            "engine_name": engine_name,
            "generate_report": self._chk_report.isChecked(),
            "interactive_mode": False,  # Wizard handles interactivity now
            "dual_resonance": {
                "enable": self._chk_dr.isChecked(),
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

    def get_auto_peak_config(self):
        """Return auto peak config if enabled, else None."""
        if self._chk_auto.isChecked():
            return self._auto_peak_cfg
        return None

    def is_report_enabled(self):
        """Whether the user wants a comprehensive report."""
        return self._chk_report.isChecked()
