"""Home Page — Single-profile stripping + Batch processing.

Faithfully replicates the original PySide6 HomePage structure:
  Tab 1: Single Profile (input source, config, options, run/cancel, progress)
  Tab 2: Batch Processing (file list, config, options, run, progress)
"""
import os
import tempfile
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QFormLayout, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QTextEdit,
    QRadioButton, QButtonGroup, QFileDialog, QMessageBox,
    QListWidget, QProgressBar, QScrollArea,
)

from ..workers.workflow_worker import WorkflowWorker
from ..workers.batch_worker import BatchWorker
from ..widgets.style_constants import (
    OUTER_MARGINS, SECONDARY_LABEL, MONOSPACE_PREVIEW,
    BUTTON_PRIMARY, BUTTON_SUCCESS, BUTTON_DANGER, GEAR_BUTTON, EMOJI,
)

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]
ENGINE_DESCRIPTIONS = {
    "diffuse_field": "Full diffuse wavefield H/V ratio (HVf.exe required)",
    "sh_wave": "SH-wave transfer function (pure Python, no external tools)",
    "ellipticity": "Rayleigh wave ellipticity (gpell.exe required)",
}


class HomePage(QWidget):
    """Home page with Single Profile and Batch Processing tabs."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._worker = None
        self._batch_worker = None
        self._engine_settings_config = {}
        self._dr_ratio = 1.2
        self._dr_shift = 0.3

        self._build_ui()

    # ═══════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*OUTER_MARGINS)

        hdr = QLabel(f"<b>{EMOJI['home']} HVSR Progressive Layer Stripping Analysis</b>")
        hdr.setStyleSheet("font-size: 14px; padding: 4px;")
        layout.addWidget(hdr)
        desc = QLabel("Run complete analysis workflow on single or multiple soil profiles")
        desc.setStyleSheet(SECONDARY_LABEL)
        layout.addWidget(desc)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_single_tab(), "Single Profile")
        self._tabs.addTab(self._build_batch_tab(), "Batch Processing")
        layout.addWidget(self._tabs)

    # ── Single Profile Tab ──────────────────────────────────────
    def _build_single_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)

        # Input Source
        src_grp = QGroupBox(f"{EMOJI['file']} Input Source")
        src_layout = QVBoxLayout(src_grp)
        self._src_group = QButtonGroup()
        self._rb_hvf = QRadioButton("HVf File")
        self._rb_hvf.setChecked(True)
        self._rb_dinver = QRadioButton("Dinver Output")
        self._src_group.addButton(self._rb_hvf)
        self._src_group.addButton(self._rb_dinver)
        src_row = QHBoxLayout()
        src_row.addWidget(self._rb_hvf)
        src_row.addWidget(self._rb_dinver)
        src_row.addStretch()
        src_layout.addLayout(src_row)

        # HVf file input
        self._hvf_widget = QWidget()
        hvf_layout = QHBoxLayout(self._hvf_widget)
        hvf_layout.setContentsMargins(0, 0, 0, 0)
        hvf_layout.addWidget(QLabel("Model File:"))
        self._model_edit = QLineEdit()
        hvf_layout.addWidget(self._model_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_model)
        hvf_layout.addWidget(btn)
        src_layout.addWidget(self._hvf_widget)

        # Dinver input
        self._dinver_widget = QWidget()
        self._dinver_widget.setVisible(False)
        din_layout = QFormLayout(self._dinver_widget)
        din_layout.setContentsMargins(0, 0, 0, 0)
        self._din_vs = QLineEdit()
        btn_vs = QPushButton("..."); btn_vs.setFixedWidth(30)
        btn_vs.clicked.connect(lambda: self._browse_dinver_file("vs"))
        r1 = QHBoxLayout(); r1.addWidget(self._din_vs); r1.addWidget(btn_vs)
        din_layout.addRow("Vs File:", r1)
        self._din_vp = QLineEdit(); self._din_vp.setPlaceholderText("Optional")
        btn_vp = QPushButton("..."); btn_vp.setFixedWidth(30)
        btn_vp.clicked.connect(lambda: self._browse_dinver_file("vp"))
        r2 = QHBoxLayout(); r2.addWidget(self._din_vp); r2.addWidget(btn_vp)
        din_layout.addRow("Vp File:", r2)
        self._din_rho = QLineEdit(); self._din_rho.setPlaceholderText("Optional")
        btn_rho = QPushButton("..."); btn_rho.setFixedWidth(30)
        btn_rho.clicked.connect(lambda: self._browse_dinver_file("rho"))
        r3 = QHBoxLayout(); r3.addWidget(self._din_rho); r3.addWidget(btn_rho)
        din_layout.addRow("Density File:", r3)
        src_layout.addWidget(self._dinver_widget)

        self._src_group.buttonClicked.connect(self._on_source_changed)
        layout.addWidget(src_grp)

        # Output Directory
        out_grp = QGroupBox(f"{EMOJI['folder']} Output Directory")
        out_layout = QHBoxLayout(out_grp)
        self._output_edit = QLineEdit()
        out_layout.addWidget(self._output_edit)
        btn_out = QPushButton("Browse...")
        btn_out.clicked.connect(self._browse_output)
        out_layout.addWidget(btn_out)
        layout.addWidget(out_grp)

        # Configuration
        cfg_grp = QGroupBox(f"{EMOJI['config']} Configuration")
        cfg_form = QFormLayout(cfg_grp)
        self._fmin = QDoubleSpinBox(); self._fmin.setRange(0.01, 10); self._fmin.setValue(0.2)
        self._fmax = QDoubleSpinBox(); self._fmax.setRange(1, 100); self._fmax.setValue(20.0)
        self._nf = QSpinBox(); self._nf.setRange(10, 2000); self._nf.setValue(71)
        cfg_form.addRow("Freq Min (Hz):", self._fmin)
        cfg_form.addRow("Freq Max (Hz):", self._fmax)
        cfg_form.addRow("Points:", self._nf)

        engine_row = QHBoxLayout()
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(ENGINES)
        self._engine_combo.currentTextChanged.connect(self._on_engine_changed)
        engine_row.addWidget(self._engine_combo)
        self._gear_btn = QPushButton("⚙")
        self._gear_btn.setFixedWidth(30)
        self._gear_btn.setStyleSheet(GEAR_BUTTON)
        self._gear_btn.setToolTip("Configure engine settings")
        self._gear_btn.clicked.connect(self._open_engine_settings)
        engine_row.addWidget(self._gear_btn)
        cfg_form.addRow("Engine:", engine_row)
        self._engine_desc = QLabel(ENGINE_DESCRIPTIONS.get("diffuse_field", ""))
        self._engine_desc.setStyleSheet(SECONDARY_LABEL)
        self._engine_desc.setWordWrap(True)
        cfg_form.addRow("", self._engine_desc)
        layout.addWidget(cfg_grp)

        # Options
        opt_grp = QGroupBox(f"{EMOJI['config']} Options")
        opt_layout = QVBoxLayout(opt_grp)
        self._chk_report = QCheckBox("Generate comprehensive report")
        self._chk_report.setChecked(True)
        self._chk_report.setToolTip("Generate analysis report with overlay plots, peak evolution, and summary")
        opt_layout.addWidget(self._chk_report)

        self._chk_interactive = QCheckBox("Interactive peak selection")
        self._chk_interactive.setToolTip("Manually select peaks by clicking on each HVSR curve after processing")
        opt_layout.addWidget(self._chk_interactive)

        dr_row = QHBoxLayout()
        self._chk_dual = QCheckBox("Run dual-resonance analysis")
        dr_row.addWidget(self._chk_dual)
        self._dr_gear = QPushButton("⚙")
        self._dr_gear.setFixedWidth(30)
        self._dr_gear.setStyleSheet(GEAR_BUTTON)
        self._dr_gear.clicked.connect(self._open_dr_settings)
        dr_row.addWidget(self._dr_gear)
        dr_row.addStretch()
        opt_layout.addLayout(dr_row)
        layout.addWidget(opt_grp)

        # Control
        ctrl = QHBoxLayout()
        self._btn_run = QPushButton(f"{EMOJI['run']} Run Analysis")
        self._btn_run.setStyleSheet(BUTTON_PRIMARY)
        self._btn_run.clicked.connect(self._run_single)
        ctrl.addWidget(self._btn_run)
        self._btn_cancel = QPushButton(f"{EMOJI['stop']} Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel)
        ctrl.addWidget(self._btn_cancel)
        layout.addLayout(ctrl)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._output_text = QTextEdit()
        self._output_text.setReadOnly(True)
        self._output_text.setMaximumHeight(200)
        self._output_text.setStyleSheet(MONOSPACE_PREVIEW)
        layout.addWidget(self._output_text)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    # ── Batch Processing Tab ────────────────────────────────────
    def _build_batch_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)

        # Input files
        inp_grp = QGroupBox(f"{EMOJI['file']} Input Profiles")
        inp_layout = QVBoxLayout(inp_grp)
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Files...")
        btn_add.clicked.connect(self._batch_add_files)
        btn_dir = QPushButton("Add Directory...")
        btn_dir.clicked.connect(self._batch_add_dir)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._batch_clear)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_dir)
        btn_row.addWidget(btn_clear)
        inp_layout.addLayout(btn_row)
        self._batch_list = QListWidget()
        self._batch_list.setMinimumHeight(150)
        inp_layout.addWidget(self._batch_list)
        layout.addWidget(inp_grp)

        # Batch output directory
        bout_grp = QGroupBox(f"{EMOJI['folder']} Output Directory")
        bout_layout = QHBoxLayout(bout_grp)
        self._batch_output_edit = QLineEdit()
        bout_layout.addWidget(self._batch_output_edit)
        btn_bout = QPushButton("Browse...")
        btn_bout.clicked.connect(lambda: self._browse_dir(self._batch_output_edit))
        bout_layout.addWidget(btn_bout)
        layout.addWidget(bout_grp)

        # Batch configuration
        bcfg_grp = QGroupBox(f"{EMOJI['config']} Configuration")
        bcfg_form = QFormLayout(bcfg_grp)
        self._batch_fmin = QDoubleSpinBox(); self._batch_fmin.setRange(0.01, 10); self._batch_fmin.setValue(0.2)
        self._batch_fmax = QDoubleSpinBox(); self._batch_fmax.setRange(1, 100); self._batch_fmax.setValue(20.0)
        self._batch_nf = QSpinBox(); self._batch_nf.setRange(10, 2000); self._batch_nf.setValue(71)
        bcfg_form.addRow("Freq Min (Hz):", self._batch_fmin)
        bcfg_form.addRow("Freq Max (Hz):", self._batch_fmax)
        bcfg_form.addRow("Points:", self._batch_nf)

        bengine_row = QHBoxLayout()
        self._batch_engine = QComboBox()
        self._batch_engine.addItems(ENGINES)
        bengine_row.addWidget(self._batch_engine)
        bgear = QPushButton("⚙")
        bgear.setFixedWidth(30)
        bgear.setStyleSheet(GEAR_BUTTON)
        bgear.clicked.connect(self._open_engine_settings)
        bengine_row.addWidget(bgear)
        bcfg_form.addRow("Engine:", bengine_row)
        layout.addWidget(bcfg_grp)

        # Batch options
        bopt_grp = QGroupBox(f"{EMOJI['config']} Options")
        bopt_layout = QVBoxLayout(bopt_grp)
        self._batch_chk_report = QCheckBox("Generate comprehensive report for each profile")
        self._batch_chk_report.setChecked(True)
        bopt_layout.addWidget(self._batch_chk_report)
        bdr_row = QHBoxLayout()
        self._batch_chk_dual = QCheckBox("Run dual-resonance analysis")
        bdr_row.addWidget(self._batch_chk_dual)
        bdr_gear = QPushButton("⚙")
        bdr_gear.setFixedWidth(30)
        bdr_gear.setStyleSheet(GEAR_BUTTON)
        bdr_gear.clicked.connect(self._open_dr_settings)
        bdr_row.addWidget(bdr_gear)
        bdr_row.addStretch()
        bopt_layout.addLayout(bdr_row)
        layout.addWidget(bopt_grp)

        # Run batch
        self._btn_batch_run = QPushButton(f"{EMOJI['run']} Run Batch Analysis")
        self._btn_batch_run.setStyleSheet(BUTTON_SUCCESS)
        self._btn_batch_run.clicked.connect(self._run_batch)
        layout.addWidget(self._btn_batch_run)

        # Batch progress
        self._batch_progress = QProgressBar()
        self._batch_progress.setVisible(False)
        layout.addWidget(self._batch_progress)

        self._batch_output_text = QTextEdit()
        self._batch_output_text.setReadOnly(True)
        self._batch_output_text.setMaximumHeight(200)
        self._batch_output_text.setStyleSheet(MONOSPACE_PREVIEW)
        layout.addWidget(self._batch_output_text)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    # ═══════════════════════════════════════════════════════════
    #  SINGLE PROFILE WORKFLOW
    # ═══════════════════════════════════════════════════════════
    def _run_single(self):
        # Get model path
        model_path = None
        if self._rb_hvf.isChecked():
            model_path = self._model_edit.text().strip()
            if not model_path or not os.path.isfile(model_path):
                QMessageBox.warning(self, "Error", "Please select a valid model file.")
                return
        else:
            vs_path = self._din_vs.text().strip()
            if not vs_path:
                QMessageBox.warning(self, "Error", "Please select a Vs file.")
                return
            try:
                from core.soil_profile import SoilProfile
                vp = self._din_vp.text().strip() or None
                rho = self._din_rho.text().strip() or None
                prof = SoilProfile.from_dinver_files(vs_path, vp, rho)
                tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
                tmp.write(prof.to_hvf_format())
                tmp.close()
                model_path = tmp.name
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load Dinver files: {e}")
                return

        output_dir = self._output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return
        os.makedirs(output_dir, exist_ok=True)

        # Build config
        engine_name = self._engine_combo.currentText()
        config = {
            "hv_forward": {
                "fmin": self._fmin.value(),
                "fmax": self._fmax.value(),
                "nf": self._nf.value(),
            },
            "engine_name": engine_name,
            "generate_report": self._chk_report.isChecked(),
            "interactive_mode": self._chk_interactive.isChecked(),
        }
        config.update(self._engine_settings_config.get(engine_name, {}))

        if self._chk_dual.isChecked():
            config["dual_resonance"] = {
                "enabled": True,
                "separation_ratio_threshold": self._dr_ratio,
                "separation_shift_threshold": self._dr_shift,
            }

        # Start worker
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._output_text.clear()
        self._output_text.append("Starting analysis...")

        self._worker = WorkflowWorker(model_path, output_dir, config, parent=self)
        self._worker.progress.connect(lambda msg: self._output_text.append(msg))
        self._worker.finished_signal.connect(self._on_single_finished)
        self._worker.error.connect(self._on_single_error)
        self._worker.start()

    def _on_single_finished(self, result):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        self._output_text.append("\n✓ Analysis completed successfully!")

        if self._chk_interactive.isChecked():
            self._output_text.append("Opening interactive peak picker...")
            try:
                from ..dialogs.interactive_peak_picker import InteractivePeakPickerDialog
                dlg = InteractivePeakPickerDialog(result, parent=self)
                dlg.exec_()
            except Exception as e:
                self._output_text.append(f"Interactive picker error: {e}")

        if self._chk_report.isChecked():
            self._output_text.append("Opening figure wizard...")
            try:
                from ..dialogs.figure_wizard_dialog import FigureWizardDialog
                dlg = FigureWizardDialog(result, parent=self)
                dlg.exec_()
            except Exception as e:
                self._output_text.append(f"Figure wizard error: {e}")

        if self._main_window:
            self._main_window.set_status("Analysis completed")

    def _on_single_error(self, msg):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)
        self._output_text.append(f"\n✗ Error: {msg}")
        QMessageBox.critical(self, "Analysis Error", msg)

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._output_text.append("Cancelled.")
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setVisible(False)

    # ═══════════════════════════════════════════════════════════
    #  BATCH WORKFLOW
    # ═══════════════════════════════════════════════════════════
    def _run_batch(self):
        n = self._batch_list.count()
        if n == 0:
            QMessageBox.warning(self, "Error", "No profiles loaded. Add files first.")
            return
        output_dir = self._batch_output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return

        # Open BatchSettingsDialog for confirmation
        from ..dialogs.batch_settings_dialog import BatchSettingsDialog
        batch_cfg = {
            "frequency": {
                "fmin": self._batch_fmin.value(),
                "fmax": self._batch_fmax.value(),
                "nf": self._batch_nf.value(),
            }
        }
        dlg = BatchSettingsDialog(batch_cfg, parent=self)
        if dlg.exec_() != BatchSettingsDialog.Accepted:
            return

        confirmed_cfg = dlg.get_config()
        file_paths = [self._batch_list.item(i).text() for i in range(n)]
        engine_name = confirmed_cfg["frequency"]["engine"]

        config = {
            "hv_forward": confirmed_cfg["frequency"],
            "engine_name": engine_name,
            "generate_report": confirmed_cfg["options"]["generate_report"],
        }
        if confirmed_cfg["options"]["dual_resonance"]:
            config["dual_resonance"] = {
                "enabled": True,
                "separation_ratio_threshold": self._dr_ratio,
                "separation_shift_threshold": self._dr_shift,
            }
        config.update(self._engine_settings_config.get(engine_name, {}))

        # Start batch worker
        self._btn_batch_run.setEnabled(False)
        self._batch_progress.setVisible(True)
        self._batch_progress.setRange(0, n)
        self._batch_output_text.clear()

        self._batch_worker = BatchWorker(file_paths, output_dir, config, parent=self)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.finished_signal.connect(self._on_batch_finished)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

    def _on_batch_progress(self, msg, current, total):
        self._batch_output_text.append(msg)
        self._batch_progress.setValue(current + 1)

    def _on_batch_finished(self, results):
        self._btn_batch_run.setEnabled(True)
        self._batch_progress.setVisible(False)
        success = sum(1 for r in results if r["status"] == "success")
        errors = sum(1 for r in results if r["status"] == "error")
        self._batch_output_text.append(f"\n✓ Batch completed: {success} success, {errors} errors")

        if self._main_window:
            self._main_window.set_status(f"Batch: {success}/{len(results)} succeeded")

    def _on_batch_error(self, msg):
        self._btn_batch_run.setEnabled(True)
        self._batch_progress.setVisible(False)
        self._batch_output_text.append(f"\n✗ Batch error: {msg}")
        QMessageBox.critical(self, "Batch Error", msg)

    # ═══════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════
    def _on_source_changed(self):
        is_hvf = self._rb_hvf.isChecked()
        self._hvf_widget.setVisible(is_hvf)
        self._dinver_widget.setVisible(not is_hvf)

    def _on_engine_changed(self, engine_name):
        desc = ENGINE_DESCRIPTIONS.get(engine_name, "")
        if hasattr(self, '_engine_desc'):
            self._engine_desc.setText(desc)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.txt);;All (*)")
        if path:
            self._model_edit.setText(path)

    def _browse_dinver_file(self, ftype):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {ftype.upper()} File", "", "Text Files (*.txt);;All (*)")
        if not path:
            return
        target = {"vs": self._din_vs, "vp": self._din_vp, "rho": self._din_rho}[ftype]
        target.setText(path)
        if ftype == "vs":
            base = path.replace("_vs.txt", "").replace("_Vs.txt", "")
            for suffix, edit in [("_vp.txt", self._din_vp), ("_rho.txt", self._din_rho)]:
                cand = base + suffix
                if os.path.isfile(cand) and not edit.text():
                    edit.setText(cand)

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_edit.setText(d)

    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            edit.setText(d)

    def _batch_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Profile Files", "", "Model Files (*.txt);;All (*)")
        for p in paths:
            self._batch_list.addItem(p)

    def _batch_add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            for f in sorted(Path(d).glob("*.txt")):
                self._batch_list.addItem(str(f))

    def _batch_clear(self):
        self._batch_list.clear()

    def _open_engine_settings(self):
        from ..dialogs.engine_settings_dialog import EngineSettingsDialog
        dlg = EngineSettingsDialog(self._engine_settings_config, parent=self)
        if dlg.exec_() == EngineSettingsDialog.Accepted:
            self._engine_settings_config = dlg.get_config()
            if self._main_window:
                self._main_window.update_config({"engine_settings": self._engine_settings_config})

    def _open_dr_settings(self):
        from ..dialogs.dual_resonance_settings_dialog import DualResonanceSettingsDialog
        dlg = DualResonanceSettingsDialog(self._dr_ratio, self._dr_shift, parent=self)
        if dlg.exec_() == DualResonanceSettingsDialog.Accepted:
            vals = dlg.get_values()
            self._dr_ratio = vals["separation_ratio_threshold"]
            self._dr_shift = vals["separation_shift_threshold"]

    def apply_config(self, cfg):
        """Apply settings from main window config."""
        hv = cfg.get("hv_forward", {})
        if "fmin" in hv: self._fmin.setValue(hv["fmin"]); self._batch_fmin.setValue(hv["fmin"])
        if "fmax" in hv: self._fmax.setValue(hv["fmax"]); self._batch_fmax.setValue(hv["fmax"])
        if "nf" in hv: self._nf.setValue(hv["nf"]); self._batch_nf.setValue(hv["nf"])

        engine = cfg.get("engine", {})
        engine_name = engine.get("name", "diffuse_field") if isinstance(engine, dict) else str(engine)
        for combo in [self._engine_combo, self._batch_engine]:
            idx = combo.findText(engine_name)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        dr = cfg.get("dual_resonance", {})
        if "separation_ratio_threshold" in dr: self._dr_ratio = dr["separation_ratio_threshold"]
        if "separation_shift_threshold" in dr: self._dr_shift = dr["separation_shift_threshold"]

        es = cfg.get("engine_settings", {})
        if es:
            self._engine_settings_config = dict(es)
