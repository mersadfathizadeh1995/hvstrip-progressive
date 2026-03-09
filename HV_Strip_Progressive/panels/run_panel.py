"""Run Panel — Forward modeling, single stripping, batch processing controls.

Replaces the run/compute sections from the old home_page and forward_modeling_page.
"""
import os
from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QProgressBar, QTextEdit, QCheckBox, QListWidget,
)

from ..widgets.collapsible_group import CollapsibleGroupBox
from ..widgets.style_constants import (
    OUTER_MARGINS, SECONDARY_LABEL, MONOSPACE_PREVIEW,
    BUTTON_PRIMARY, BUTTON_SUCCESS, BUTTON_DANGER, EMOJI,
)
from ..workers.forward_worker import ForwardWorker
from ..workers.workflow_worker import WorkflowWorker
from ..workers.batch_worker import BatchWorker


class RunPanel(QWidget):
    """Left-panel tab for running analyses."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._fwd_worker = None
        self._wf_worker = None
        self._batch_worker = None
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(*OUTER_MARGINS)

        # ── Forward Modeling ────────────────────────────────────
        fwd = CollapsibleGroupBox(f"{EMOJI['forward']} Forward Modeling")
        fwd_lay = QVBoxLayout()
        self._btn_fwd = QPushButton(f"{EMOJI['run']} Compute HV Curve")
        self._btn_fwd.setStyleSheet(BUTTON_PRIMARY)
        self._btn_fwd.clicked.connect(self.run_forward)
        fwd_lay.addWidget(self._btn_fwd)
        self._fwd_result = QLabel("")
        self._fwd_result.setStyleSheet(SECONDARY_LABEL)
        self._fwd_result.setWordWrap(True)
        fwd_lay.addWidget(self._fwd_result)
        fwd.setContentLayout(fwd_lay)
        lay.addWidget(fwd)

        # ── Single Stripping ────────────────────────────────────
        strip = CollapsibleGroupBox(f"{EMOJI['layer']} Single Strip Workflow")
        strip_lay = QVBoxLayout()

        orow = QHBoxLayout()
        orow.addWidget(QLabel("Output:"))
        self._strip_output = QLineEdit()
        self._strip_output.setPlaceholderText("Output directory")
        orow.addWidget(self._strip_output)
        btn_o = QPushButton("...")
        btn_o.setFixedWidth(30)
        btn_o.clicked.connect(lambda: self._browse_dir(self._strip_output))
        orow.addWidget(btn_o)
        strip_lay.addLayout(orow)

        self._chk_interactive = QCheckBox("Interactive peak selection")
        strip_lay.addWidget(self._chk_interactive)

        btn_row = QHBoxLayout()
        self._btn_strip = QPushButton(f"{EMOJI['run']} Run Stripping")
        self._btn_strip.setStyleSheet(BUTTON_SUCCESS)
        self._btn_strip.clicked.connect(self.run_stripping)
        btn_row.addWidget(self._btn_strip)
        self._btn_strip_cancel = QPushButton("Cancel")
        self._btn_strip_cancel.setEnabled(False)
        self._btn_strip_cancel.clicked.connect(self._cancel_strip)
        btn_row.addWidget(self._btn_strip_cancel)
        strip_lay.addLayout(btn_row)

        self._strip_progress = QProgressBar()
        self._strip_progress.setVisible(False)
        strip_lay.addWidget(self._strip_progress)
        strip.setContentLayout(strip_lay)
        lay.addWidget(strip)

        # ── Batch Processing ────────────────────────────────────
        batch = CollapsibleGroupBox(f"{EMOJI['batch']} Batch Processing")
        batch_lay = QVBoxLayout()

        file_row = QHBoxLayout()
        btn_add = QPushButton("Add Files...")
        btn_add.clicked.connect(self._batch_add_files)
        btn_dir = QPushButton("Add Dir...")
        btn_dir.clicked.connect(self._batch_add_dir)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self._batch_list.clear())
        file_row.addWidget(btn_add)
        file_row.addWidget(btn_dir)
        file_row.addWidget(btn_clear)
        batch_lay.addLayout(file_row)

        self._batch_list = QListWidget()
        self._batch_list.setMaximumHeight(120)
        batch_lay.addWidget(self._batch_list)

        bout_row = QHBoxLayout()
        bout_row.addWidget(QLabel("Output:"))
        self._batch_output = QLineEdit()
        bout_row.addWidget(self._batch_output)
        btn_bo = QPushButton("...")
        btn_bo.setFixedWidth(30)
        btn_bo.clicked.connect(lambda: self._browse_dir(self._batch_output))
        bout_row.addWidget(btn_bo)
        batch_lay.addLayout(bout_row)

        self._btn_batch = QPushButton(f"{EMOJI['run']} Run Batch")
        self._btn_batch.setStyleSheet(BUTTON_SUCCESS)
        self._btn_batch.clicked.connect(self.run_batch)
        batch_lay.addWidget(self._btn_batch)

        self._batch_progress = QProgressBar()
        self._batch_progress.setVisible(False)
        batch_lay.addWidget(self._batch_progress)
        batch.setContentLayout(batch_lay)
        lay.addWidget(batch)

        # ── Results Log ─────────────────────────────────────────
        log = CollapsibleGroupBox(f"{EMOJI['info']} Results Log")
        log_lay = QVBoxLayout()
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(150)
        self._log_text.setStyleSheet(MONOSPACE_PREVIEW)
        log_lay.addWidget(self._log_text)
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.clicked.connect(self._log_text.clear)
        log_lay.addWidget(btn_clear_log)
        log.setContentLayout(log_lay)
        lay.addWidget(log)

        lay.addStretch()
        scroll.setWidget(inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    # ══════════════════════════════════════════════════════════════
    #  FORWARD MODELING
    # ══════════════════════════════════════════════════════════════
    def run_forward(self):
        if not self._mw:
            return
        model_path = None
        if hasattr(self._mw, '_input_panel') and hasattr(self._mw._input_panel, 'get_model_path'):
            model_path = self._mw._input_panel.get_model_path()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "Error", "Please load a valid model file first.")
            return

        engine_name = self._mw.get_engine_name()
        cfg = self._mw.get_engine_settings().get(engine_name, {}).copy()
        if hasattr(self._mw, '_config_panel') and hasattr(self._mw._config_panel, 'get_config'):
            panel_cfg = self._mw._config_panel.get_config()
            hv = panel_cfg.get("hv_forward", {})
            cfg.update({"fmin": hv.get("fmin", 0.2),
                        "fmax": hv.get("fmax", 20.0),
                        "nf": hv.get("nf", 71)})

        self._btn_fwd.setEnabled(False)
        self._btn_fwd.setText("Computing...")
        self._fwd_result.setText("Running forward model...")
        self._mw.log(f"Forward: {engine_name} on {Path(model_path).name}")

        self._fwd_worker = ForwardWorker(model_path, cfg, engine_name, parent=self)
        self._fwd_worker.finished_signal.connect(self._on_fwd_done)
        self._fwd_worker.error.connect(self._on_fwd_error)
        self._fwd_worker.start()

    def _on_fwd_done(self, result):
        freqs, amps = result
        self._btn_fwd.setEnabled(True)
        self._btn_fwd.setText(f"{EMOJI['run']} Compute HV Curve")

        import numpy as np
        idx = np.argmax(amps)
        f0 = freqs[idx]
        a0 = amps[idx]
        self._fwd_result.setText(
            f"<b>f0 = {f0:.4f} Hz</b> (amp = {a0:.3f}), {len(freqs)} pts")
        self._log(f"Forward done: f0={f0:.4f} Hz, {len(freqs)} pts")

        if self._mw:
            profile = None
            if hasattr(self._mw, '_input_panel'):
                profile = self._mw._input_panel.get_profile()
            self._mw.update_hv_curve(freqs, amps, profile)
            self._mw.set_status(f"Forward model: f0 = {f0:.3f} Hz")

    def _on_fwd_error(self, msg):
        self._btn_fwd.setEnabled(True)
        self._btn_fwd.setText(f"{EMOJI['run']} Compute HV Curve")
        self._fwd_result.setText(f"<span style='color:red;'>Error: {msg}</span>")
        self._log(f"Forward ERROR: {msg}")
        QMessageBox.critical(self, "Forward Error", msg)

    # ══════════════════════════════════════════════════════════════
    #  STRIPPING WORKFLOW
    # ══════════════════════════════════════════════════════════════
    def run_stripping(self):
        if not self._mw:
            return
        model_path = None
        if hasattr(self._mw, '_input_panel'):
            model_path = self._mw._input_panel.get_model_path()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "Error", "Please load a valid model file first.")
            return

        output_dir = self._strip_output.text().strip()
        if not output_dir:
            if hasattr(self._mw, '_input_panel'):
                output_dir = self._mw._input_panel.get_output_dir()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return
        os.makedirs(output_dir, exist_ok=True)

        # Build config from config panel
        config = {}
        if hasattr(self._mw, '_config_panel'):
            config = self._mw._config_panel.get_config()

        engine_name = config.get("engine_name", self._mw.get_engine_name())
        es = self._mw.get_engine_settings().get(engine_name, {})
        config.update(es)

        self._btn_strip.setEnabled(False)
        self._btn_strip_cancel.setEnabled(True)
        self._strip_progress.setVisible(True)
        self._strip_progress.setRange(0, 0)
        self._log("Starting stripping workflow...")

        self._wf_worker = WorkflowWorker(
            model_path, output_dir, config, parent=self)
        self._wf_worker.progress.connect(lambda msg: self._log(msg))
        self._wf_worker.finished_signal.connect(self._on_strip_done)
        self._wf_worker.error.connect(self._on_strip_error)
        self._wf_worker.start()

    def _on_strip_done(self, result):
        self._btn_strip.setEnabled(True)
        self._btn_strip_cancel.setEnabled(False)
        self._strip_progress.setVisible(False)
        self._log("Stripping workflow completed successfully!")

        if self._mw:
            self._mw.set_result(result)
            strip_dir = result.get("strip_directory")
            if strip_dir:
                self._mw.update_overlay(str(strip_dir))
            self._mw.update_strip_results(result)
            self._mw.set_status("Stripping completed")

            # Open interactive picker if requested
            if self._chk_interactive.isChecked():
                self._mw._on_open_peak_picker()

    def _on_strip_error(self, msg):
        self._btn_strip.setEnabled(True)
        self._btn_strip_cancel.setEnabled(False)
        self._strip_progress.setVisible(False)
        self._log(f"Stripping ERROR: {msg}")
        QMessageBox.critical(self, "Stripping Error", msg)

    def _cancel_strip(self):
        if self._wf_worker and self._wf_worker.isRunning():
            self._wf_worker.terminate()
            self._log("Stripping cancelled.")
        self._btn_strip.setEnabled(True)
        self._btn_strip_cancel.setEnabled(False)
        self._strip_progress.setVisible(False)

    # ══════════════════════════════════════════════════════════════
    #  BATCH PROCESSING
    # ══════════════════════════════════════════════════════════════
    def set_batch_folder(self, folder):
        """Add all .txt files from folder to batch list."""
        for f in sorted(Path(folder).glob("*.txt")):
            self._batch_list.addItem(str(f))
        self._batch_output.setText(str(Path(folder) / "batch_results"))

    def run_batch(self):
        n = self._batch_list.count()
        if n == 0:
            QMessageBox.warning(self, "Error", "No profiles loaded.")
            return
        output_dir = self._batch_output.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Select output directory.")
            return
        os.makedirs(output_dir, exist_ok=True)

        config = {}
        if hasattr(self._mw, '_config_panel'):
            config = self._mw._config_panel.get_config()

        engine_name = config.get("engine_name", self._mw.get_engine_name())
        es = self._mw.get_engine_settings().get(engine_name, {})
        config.update(es)

        file_paths = [self._batch_list.item(i).text() for i in range(n)]

        self._btn_batch.setEnabled(False)
        self._batch_progress.setVisible(True)
        self._batch_progress.setRange(0, n)
        self._log(f"Starting batch: {n} profiles...")

        self._batch_worker = BatchWorker(
            file_paths, output_dir, config, parent=self)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished_signal.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

    def _on_batch_progress(self, msg, current, total):
        self._log(msg)
        self._batch_progress.setValue(current + 1)

    def _on_batch_done(self, results):
        self._btn_batch.setEnabled(True)
        self._batch_progress.setVisible(False)
        ok = sum(1 for r in results if r.get("status") == "success")
        err = sum(1 for r in results if r.get("status") == "error")
        self._log(f"Batch completed: {ok} success, {err} errors")
        if self._mw:
            self._mw.set_status(f"Batch: {ok}/{len(results)} succeeded")

    def _on_batch_error(self, msg):
        self._btn_batch.setEnabled(True)
        self._batch_progress.setVisible(False)
        self._log(f"Batch ERROR: {msg}")
        QMessageBox.critical(self, "Batch Error", msg)

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════
    def _batch_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Profile Files", "",
            "Model Files (*.txt);;All (*)")
        for p in paths:
            self._batch_list.addItem(p)

    def _batch_add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            for f in sorted(Path(d).glob("*.txt")):
                self._batch_list.addItem(str(f))

    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            edit.setText(d)

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.append(f"[{ts}] {msg}")
        if self._mw:
            self._mw.log(msg)
