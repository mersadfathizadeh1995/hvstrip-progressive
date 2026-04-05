"""Strip Batch Panel — settings and run for batch HV stripping.

Data loading (file list) is handled by the Batch Input canvas tab.
This panel contains: engine, frequency, peak detection, dual-resonance,
output dir, report option, Run Batch button.
"""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QFileDialog, QScrollArea, QProgressBar, QCheckBox,
)

from ..widgets.style_constants import (
    BUTTON_SUCCESS, BUTTON_DANGER, GEAR_BUTTON,
    SECONDARY_LABEL, EMOJI,
)
from ..widgets.collapsible_group import CollapsibleGroupBox

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]
PEAK_PRESETS = ["default", "sesame", "custom"]
PEAK_METHODS = ["find_peaks", "argmax"]
PEAK_SELECTIONS = ["leftmost", "highest", "global_max"]


class StripBatchPanel(QWidget):
    """Left-panel content for HV Strip → Batch sub-tab."""

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
        peak_grp = CollapsibleGroupBox(
            f"{EMOJI['peak']} Peak Detection", collapsed=True)
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

        peak_grp.setContentLayout(peak_lay)
        lay.addWidget(peak_grp)

        # ── Dual-Resonance ─────────────────────────────────────
        dr_grp = CollapsibleGroupBox(
            f"{EMOJI['dual']} Dual-Resonance", collapsed=True)
        dr_lay = QVBoxLayout()
        self._chk_dr = QCheckBox("Enable dual-resonance per profile")
        dr_lay.addWidget(self._chk_dr)
        dr_params = QHBoxLayout()
        dr_params.addWidget(QLabel("Ratio:"))
        self._dr_ratio = QDoubleSpinBox()
        self._dr_ratio.setRange(0.5, 5.0)
        self._dr_ratio.setValue(1.2)
        dr_params.addWidget(self._dr_ratio)
        dr_params.addWidget(QLabel("Shift:"))
        self._dr_shift = QDoubleSpinBox()
        self._dr_shift.setRange(0.0, 2.0)
        self._dr_shift.setValue(0.3)
        dr_params.addWidget(self._dr_shift)
        dr_lay.addLayout(dr_params)
        dr_grp.setContentLayout(dr_lay)
        lay.addWidget(dr_grp)

        # ── Output ─────────────────────────────────────────────
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("Batch output root directory")
        out_row.addWidget(self._out_dir, 1)
        btn_bro = QPushButton("...")
        btn_bro.setFixedWidth(30)
        btn_bro.clicked.connect(self._browse_output)
        out_row.addWidget(btn_bro)
        lay.addLayout(out_row)

        # ── Options ────────────────────────────────────────────
        self._chk_report = QCheckBox("Generate report per profile")
        self._chk_report.setChecked(True)
        lay.addWidget(self._chk_report)

        # ── Run ────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self._btn_run = QPushButton(f"{EMOJI['run']} Run Batch")
        self._btn_run.setStyleSheet(BUTTON_SUCCESS)
        self._btn_run.clicked.connect(self._run_batch)
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

    def _get_data_input(self):
        """Get the BatchInputView from the canvas."""
        if self._mw:
            from ..strip_window import MODE_STRIP_BATCH
            return self._mw.get_data_input(MODE_STRIP_BATCH)
        return None

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Batch Output Root")
        if d:
            self._out_dir.setText(d)

    def _open_engine_settings(self):
        if self._mw:
            self._mw._on_engine_settings()

    def _run_batch(self):
        """Run batch stripping on all files from Data Input tab."""
        di = self._get_data_input()
        file_list = di.get_files() if di else []
        if not file_list:
            self._result_label.setText(
                "Add files in the Data Input tab first.")
            return
        out_dir = self._out_dir.text().strip()
        if not out_dir:
            self._result_label.setText("Set output directory.")
            return

        config = self._build_config()
        engine_name = self._engine_combo.currentText()

        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(file_list))
        self._progress.setValue(0)
        self._result_label.setText("Starting batch...")

        try:
            from ..workers.batch_worker import BatchWorker
            self._worker = BatchWorker(
                file_list, out_dir, config, engine_name)
            self._worker.finished_signal.connect(self._on_batch_done)
            self._worker.error.connect(self._on_batch_error)
            if hasattr(self._worker, 'progress'):
                self._worker.progress.connect(self._on_batch_progress)
            self._worker.start()
        except ImportError:
            self._result_label.setText("BatchWorker not available")
            self._btn_run.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            self._progress.setVisible(False)

    def _on_batch_done(self, results):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)

        success = sum(1 for r in results if r.get("success", False))
        fail = len(results) - success
        self._result_label.setText(
            f"Batch done: {success} success, {fail} failed")
        self._result_label.setStyleSheet("color: green; font-size: 11px;")

        if self._mw:
            self._mw.log(f"Batch done: {success}/{len(results)} succeeded")

    def _on_batch_error(self, err_msg):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        self._result_label.setText(f"Error: {err_msg}")
        self._result_label.setStyleSheet("color: red; font-size: 11px;")

    def _on_batch_progress(self, msg, current, total):
        self._progress.setValue(current)
        self._result_label.setText(msg)

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        self._result_label.setText("Cancelled")

    def _build_config(self):
        engine_name = self._engine_combo.currentText()
        cfg = {
            "engine_name": engine_name,
            "generate_report": self._chk_report.isChecked(),
            "peak_detection": {
                "preset": self._peak_preset.currentText(),
                "method": self._peak_method.currentText(),
                "select": self._peak_select.currentText(),
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
    def set_batch_folder(self, folder):
        di = self._get_data_input()
        if di and hasattr(di, 'set_batch_folder'):
            di.set_batch_folder(folder)

    def get_engine_name(self):
        return self._engine_combo.currentText()
