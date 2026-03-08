"""
Home page - Consolidated workflow and batch processing.
"""

import tempfile
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QLineEdit, QTextEdit, QSplitter,
    QTabWidget, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QProgressBar,
    QListWidget, QListWidgetItem, QMessageBox, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, Signal, QThread

import numpy as np

from ...core.batch_workflow import run_complete_workflow
from ...core.soil_profile import SoilProfile
from ..dialogs.interactive_peak_picker import InteractivePeakPickerDialog
from ..dialogs.dual_resonance_settings_dialog import DualResonanceSettingsDialog
from ..dialogs.batch_settings_dialog import BatchSettingsDialog
from ..dialogs.figure_wizard_dialog import FigureWizardDialog
from ..dialogs.engine_settings_dialog import EngineSettingsDialog


class WorkflowWorker(QThread):
    """Worker thread for running workflow."""
    progress = Signal(str)
    finished_signal = Signal(dict)
    error = Signal(str)

    def __init__(self, model_path, output_dir, config):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config

    def run(self):
        try:
            self.progress.emit("Starting workflow...")
            results = run_complete_workflow(
                self.model_path, 
                self.output_dir, 
                self.config
            )
            self.finished_signal.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class BatchWorker(QThread):
    """Worker thread for batch processing."""
    progress = Signal(str, int, int)
    finished_signal = Signal(list)
    error = Signal(str)

    def __init__(self, profiles, output_dir, config):
        super().__init__()
        self.profiles = profiles
        self.output_dir = output_dir
        self.config = config

    def run(self):
        results = []
        total = len(self.profiles)
        for i, profile_path in enumerate(self.profiles):
            try:
                self.progress.emit(f"Processing {Path(profile_path).name}...", i + 1, total)
                profile_output = Path(self.output_dir) / Path(profile_path).stem
                profile_output.mkdir(parents=True, exist_ok=True)
                result = run_complete_workflow(
                    profile_path,
                    str(profile_output),
                    self.config
                )
                results.append({"path": profile_path, "success": True, "result": result})
            except Exception as e:
                results.append({"path": profile_path, "success": False, "error": str(e)})
        self.finished_signal.emit(results)


class HomePage(QWidget):
    """
    Home page with consolidated workflow and batch processing.
    
    Provides two modes:
    - Single Profile: Run complete workflow on one model
    - Batch Processing: Process multiple profiles at once
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HomePage")
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("HVSR Progressive Layer Stripping Analysis")
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        subtitle = QLabel("Run complete analysis workflow on single or multiple soil profiles")
        subtitle.setStyleSheet("font-size: 12px; color: gray; margin-bottom: 20px;")
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_single_tab(), "Single Profile")
        self.tabs.addTab(self._create_batch_tab(), "Batch Processing")
        layout.addWidget(self.tabs)

        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        output_layout.addWidget(self.progress_bar)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(200)
        output_layout.addWidget(self.output_text)
        
        layout.addWidget(output_group)

    def _create_single_tab(self):
        """Create single profile workflow tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout(input_group)

        source_layout = QHBoxLayout()
        self.source_group = QButtonGroup(self)
        self.radio_file = QRadioButton("HVf File")
        self.radio_file.setChecked(True)
        self.radio_dinver = QRadioButton("Dinver Output")
        self.source_group.addButton(self.radio_file, 0)
        self.source_group.addButton(self.radio_dinver, 1)
        source_layout.addWidget(self.radio_file)
        source_layout.addWidget(self.radio_dinver)
        source_layout.addStretch()
        input_layout.addLayout(source_layout)
        self.source_group.buttonClicked.connect(self._on_source_changed)

        self.file_widget = QWidget()
        file_layout = QVBoxLayout(self.file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model File:"))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Select velocity model file (.txt)")
        model_layout.addWidget(self.model_edit)
        btn_model = QPushButton("Browse...")
        btn_model.clicked.connect(self._browse_model)
        model_layout.addWidget(btn_model)
        file_layout.addLayout(model_layout)
        input_layout.addWidget(self.file_widget)

        self.dinver_widget = QWidget()
        self.dinver_widget.setVisible(False)
        dinver_layout = QVBoxLayout(self.dinver_widget)
        dinver_layout.setContentsMargins(0, 0, 0, 0)
        
        vs_row = QHBoxLayout()
        vs_row.addWidget(QLabel("Vs File:"))
        self.home_vs_edit = QLineEdit()
        self.home_vs_edit.setPlaceholderText("Required: *_vs.txt")
        vs_row.addWidget(self.home_vs_edit)
        btn_vs = QPushButton("...")
        btn_vs.setMaximumWidth(30)
        btn_vs.clicked.connect(lambda: self._browse_dinver("vs"))
        vs_row.addWidget(btn_vs)
        dinver_layout.addLayout(vs_row)
        
        vp_row = QHBoxLayout()
        vp_row.addWidget(QLabel("Vp File:"))
        self.home_vp_edit = QLineEdit()
        self.home_vp_edit.setPlaceholderText("Optional: *_vp.txt")
        vp_row.addWidget(self.home_vp_edit)
        btn_vp = QPushButton("...")
        btn_vp.setMaximumWidth(30)
        btn_vp.clicked.connect(lambda: self._browse_dinver("vp"))
        vp_row.addWidget(btn_vp)
        dinver_layout.addLayout(vp_row)
        
        rho_row = QHBoxLayout()
        rho_row.addWidget(QLabel("Density:"))
        self.home_rho_edit = QLineEdit()
        self.home_rho_edit.setPlaceholderText("Optional: *_rho.txt")
        rho_row.addWidget(self.home_rho_edit)
        btn_rho = QPushButton("...")
        btn_rho.setMaximumWidth(30)
        btn_rho.clicked.connect(lambda: self._browse_dinver("rho"))
        rho_row.addWidget(btn_rho)
        dinver_layout.addLayout(rho_row)
        
        input_layout.addWidget(self.dinver_widget)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output directory")
        output_layout.addWidget(self.output_edit)
        btn_output = QPushButton("Browse...")
        btn_output.clicked.connect(self._browse_output)
        output_layout.addWidget(btn_output)
        input_layout.addLayout(output_layout)

        layout.addWidget(input_group)

        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout(config_group)

        config_layout.addWidget(QLabel("Freq Min:"))
        self.fmin_spin = QDoubleSpinBox()
        self.fmin_spin.setRange(0.1, 10.0)
        self.fmin_spin.setValue(0.5)
        self.fmin_spin.setSingleStep(0.1)
        config_layout.addWidget(self.fmin_spin)

        config_layout.addWidget(QLabel("Freq Max:"))
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(1.0, 50.0)
        self.fmax_spin.setValue(20.0)
        self.fmax_spin.setSingleStep(1.0)
        config_layout.addWidget(self.fmax_spin)

        config_layout.addWidget(QLabel("Num Points:"))
        self.nf_spin = QSpinBox()
        self.nf_spin.setRange(100, 2000)
        self.nf_spin.setValue(500)
        config_layout.addWidget(self.nf_spin)

        config_layout.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["diffuse_field", "sh_wave", "ellipticity"])
        config_layout.addWidget(self.engine_combo)

        self._engine_settings_config = {}
        btn_engine_settings = QPushButton("\u2699")
        btn_engine_settings.setMaximumWidth(30)
        btn_engine_settings.setToolTip("Engine settings")
        btn_engine_settings.clicked.connect(self._open_engine_settings)
        config_layout.addWidget(btn_engine_settings)

        config_layout.addStretch()
        layout.addWidget(config_group)

        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.chk_generate_report = QCheckBox("Generate comprehensive report")
        self.chk_generate_report.setChecked(True)
        self.chk_generate_report.setToolTip(
            "Generate analysis report with overlay plots, peak evolution, and summary"
        )
        options_layout.addWidget(self.chk_generate_report)
        
        self.chk_interactive_peaks = QCheckBox("Interactive peak selection")
        self.chk_interactive_peaks.setChecked(False)
        self.chk_interactive_peaks.setToolTip(
            "Manually select peaks by clicking on each HVSR curve after processing"
        )
        options_layout.addWidget(self.chk_interactive_peaks)

        dr_row = QHBoxLayout()
        self.chk_dual_resonance = QCheckBox("Run dual-resonance analysis")
        self.chk_dual_resonance.setChecked(False)
        self.chk_dual_resonance.setToolTip(
            "Extract deep (f0) and shallow (f1) resonance frequencies "
            "and generate separation figure"
        )
        dr_row.addWidget(self.chk_dual_resonance)
        self.btn_dr_settings = QPushButton("\u2699")
        self.btn_dr_settings.setMaximumWidth(30)
        self.btn_dr_settings.setToolTip("Dual-resonance settings")
        self.btn_dr_settings.clicked.connect(self._open_dr_settings)
        dr_row.addWidget(self.btn_dr_settings)
        dr_row.addStretch()
        options_layout.addLayout(dr_row)

        layout.addWidget(options_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_run_single = QPushButton("Run Analysis")
        self.btn_run_single.setStyleSheet(
            "background-color: #0078d4; color: white; padding: 10px 30px; font-size: 14px;"
        )
        self.btn_run_single.clicked.connect(self._run_single)
        btn_layout.addWidget(self.btn_run_single)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        btn_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(btn_layout)
        layout.addStretch()

        return widget

    def _create_batch_tab(self):
        """Create batch processing tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        input_group = QGroupBox("Input Profiles")
        input_layout = QVBoxLayout(input_group)

        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add Files...")
        btn_add.clicked.connect(self._add_batch_files)
        btn_layout.addWidget(btn_add)
        
        btn_add_dir = QPushButton("Add Directory...")
        btn_add_dir.clicked.connect(self._add_batch_directory)
        btn_layout.addWidget(btn_add_dir)
        
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_batch_files)
        btn_layout.addWidget(btn_clear)
        
        btn_layout.addStretch()
        input_layout.addLayout(btn_layout)

        self.batch_list = QListWidget()
        self.batch_list.setMinimumHeight(150)
        input_layout.addWidget(self.batch_list)

        layout.addWidget(input_group)

        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)
        output_layout.addWidget(QLabel("Output Directory:"))
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setPlaceholderText("Select output directory for batch results")
        output_layout.addWidget(self.batch_output_edit)
        btn_batch_output = QPushButton("Browse...")
        btn_batch_output.clicked.connect(self._browse_batch_output)
        output_layout.addWidget(btn_batch_output)
        layout.addWidget(output_group)

        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout(config_group)

        config_layout.addWidget(QLabel("Freq Min:"))
        self.batch_fmin = QDoubleSpinBox()
        self.batch_fmin.setRange(0.1, 10.0)
        self.batch_fmin.setValue(0.5)
        config_layout.addWidget(self.batch_fmin)

        config_layout.addWidget(QLabel("Freq Max:"))
        self.batch_fmax = QDoubleSpinBox()
        self.batch_fmax.setRange(1.0, 50.0)
        self.batch_fmax.setValue(20.0)
        config_layout.addWidget(self.batch_fmax)

        config_layout.addWidget(QLabel("Num Points:"))
        self.batch_nf = QSpinBox()
        self.batch_nf.setRange(100, 2000)
        self.batch_nf.setValue(500)
        config_layout.addWidget(self.batch_nf)

        config_layout.addWidget(QLabel("Engine:"))
        self.batch_engine_combo = QComboBox()
        self.batch_engine_combo.addItems(["diffuse_field", "sh_wave", "ellipticity"])
        config_layout.addWidget(self.batch_engine_combo)

        btn_batch_engine_settings = QPushButton("\u2699")
        btn_batch_engine_settings.setMaximumWidth(30)
        btn_batch_engine_settings.setToolTip("Engine settings")
        btn_batch_engine_settings.clicked.connect(self._open_batch_engine_settings)
        config_layout.addWidget(btn_batch_engine_settings)

        config_layout.addStretch()
        layout.addWidget(config_group)

        # Options group for batch
        batch_options_group = QGroupBox("Options")
        batch_options_layout = QVBoxLayout(batch_options_group)
        
        self.batch_chk_generate_report = QCheckBox("Generate comprehensive report for each profile")
        self.batch_chk_generate_report.setChecked(True)
        self.batch_chk_generate_report.setToolTip(
            "Generate analysis report with overlay plots, peak evolution, and summary"
        )
        batch_options_layout.addWidget(self.batch_chk_generate_report)

        batch_dr_row = QHBoxLayout()
        self.batch_chk_dual_resonance = QCheckBox("Run dual-resonance analysis")
        self.batch_chk_dual_resonance.setChecked(False)
        self.batch_chk_dual_resonance.setToolTip(
            "Extract f0/f1 resonance frequencies for each profile"
        )
        batch_dr_row.addWidget(self.batch_chk_dual_resonance)
        self.batch_btn_dr_settings = QPushButton("\u2699")
        self.batch_btn_dr_settings.setMaximumWidth(30)
        self.batch_btn_dr_settings.setToolTip("Dual-resonance settings")
        self.batch_btn_dr_settings.clicked.connect(self._open_dr_settings)
        batch_dr_row.addWidget(self.batch_btn_dr_settings)
        batch_dr_row.addStretch()
        batch_options_layout.addLayout(batch_dr_row)

        layout.addWidget(batch_options_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_run_batch = QPushButton("Run Batch Analysis")
        self.btn_run_batch.setStyleSheet(
            "background-color: #107c10; color: white; padding: 10px 30px; font-size: 14px;"
        )
        self.btn_run_batch.clicked.connect(self._run_batch)
        btn_layout.addWidget(self.btn_run_batch)
        
        layout.addLayout(btn_layout)
        layout.addStretch()

        return widget

    # ------------------------------------------------------------------
    # Dual-resonance helpers
    # ------------------------------------------------------------------

    _dr_ratio = 1.2
    _dr_shift = 0.3

    def _open_dr_settings(self):
        """Open dual-resonance threshold settings popup."""
        dlg = DualResonanceSettingsDialog(
            self, ratio=self._dr_ratio, shift=self._dr_shift,
        )
        if dlg.exec():
            vals = dlg.get_values()
            self._dr_ratio = vals["separation_ratio_threshold"]
            self._dr_shift = vals["separation_shift_threshold"]

    def _get_dr_config(self, enabled: bool) -> dict:
        """Build dual_resonance config dict."""
        return {
            "enable": enabled,
            "separation_ratio_threshold": self._dr_ratio,
            "separation_shift_threshold": self._dr_shift,
        }

    def _on_source_changed(self):
        """Handle source type change."""
        is_dinver = self.radio_dinver.isChecked()
        self.file_widget.setVisible(not is_dinver)
        self.dinver_widget.setVisible(is_dinver)

    def _browse_dinver(self, file_type: str):
        """Browse for Dinver file."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {file_type.upper()} File", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            if file_type == "vs":
                self.home_vs_edit.setText(path)
                base = path.replace("_vs.txt", "").replace("_Vs.txt", "")
                for suffix, edit in [("_vp.txt", self.home_vp_edit), ("_rho.txt", self.home_rho_edit)]:
                    candidate = base + suffix
                    if Path(candidate).exists() and not edit.text():
                        edit.setText(candidate)
            elif file_type == "vp":
                self.home_vp_edit.setText(path)
            elif file_type == "rho":
                self.home_rho_edit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            self.model_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_edit.setText(path)

    def _browse_batch_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Batch Output Directory")
        if path:
            self.batch_output_edit.setText(path)

    def _add_batch_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Model Files", "", "Text Files (*.txt);;All Files (*)"
        )
        for path in paths:
            if not self._is_in_batch_list(path):
                self.batch_list.addItem(path)

    def _add_batch_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory with Model Files")
        if path:
            for file in Path(path).glob("*.txt"):
                if not self._is_in_batch_list(str(file)):
                    self.batch_list.addItem(str(file))

    def _is_in_batch_list(self, path):
        for i in range(self.batch_list.count()):
            if self.batch_list.item(i).text() == path:
                return True
        return False

    def _clear_batch_files(self):
        self.batch_list.clear()

    def _open_engine_settings(self):
        """Open engine settings dialog (single-run section)."""
        dlg = EngineSettingsDialog(
            self,
            config=self._engine_settings_config,
            active_engine=self.engine_combo.currentText(),
        )
        if dlg.exec():
            self._engine_settings_config = dlg.get_config()

    def _open_batch_engine_settings(self):
        """Open engine settings dialog (batch section)."""
        dlg = EngineSettingsDialog(
            self,
            config=self._engine_settings_config,
            active_engine=self.batch_engine_combo.currentText(),
        )
        if dlg.exec():
            self._engine_settings_config = dlg.get_config()

    def _run_single(self):
        output_dir = self.output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return

        self._temp_model_path = None

        if self.radio_dinver.isChecked():
            vs_file = self.home_vs_edit.text().strip()
            if not vs_file or not Path(vs_file).exists():
                QMessageBox.warning(self, "Error", "Please select a valid Vs file.")
                return
            vp_file = self.home_vp_edit.text().strip() or None
            rho_file = self.home_rho_edit.text().strip() or None
            
            try:
                profile = SoilProfile.from_dinver_files(vs_file, vp_file, rho_file)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(profile.to_hvf_format())
                    model_path = f.name
                self._temp_model_path = model_path
                self.output_text.append(f"Loaded Dinver profile: {len(profile.layers)} layers")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load Dinver files: {e}")
                return
        else:
            model_path = self.model_edit.text().strip()
            if not model_path or not Path(model_path).exists():
                QMessageBox.warning(self, "Error", "Please select a valid model file.")
                return

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Store for interactive peak selection
        self._interactive_peaks = self.chk_interactive_peaks.isChecked()
        self._current_output_dir = output_dir
        self._generate_report = self.chk_generate_report.isChecked()
        
        config = {
            "hv_forward": {
                "fmin": self.fmin_spin.value(),
                "fmax": self.fmax_spin.value(),
                "nf": self.nf_spin.value(),
            },
            "engine_name": self.engine_combo.currentText(),
            "generate_report": self._generate_report,
            "interactive_mode": self._interactive_peaks,
            "dual_resonance": self._get_dr_config(
                self.chk_dual_resonance.isChecked()
            ),
        }

        self.output_text.clear()
        self.output_text.append("Starting analysis...")
        self.btn_run_single.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.worker = WorkflowWorker(model_path, output_dir, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_single_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _run_batch(self):
        if self.batch_list.count() == 0:
            QMessageBox.warning(self, "Error", "Please add at least one model file.")
            return

        output_dir = self.batch_output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select an output directory.")
            return

        # Collect current UI values as defaults for the settings dialog
        defaults = {
            "fmin": self.batch_fmin.value(),
            "fmax": self.batch_fmax.value(),
            "nf": self.batch_nf.value(),
            "engine": self.batch_engine_combo.currentText(),
            "generate_report": self.batch_chk_generate_report.isChecked(),
            "dual_resonance": self._get_dr_config(
                self.batch_chk_dual_resonance.isChecked()
            ),
        }

        dlg = BatchSettingsDialog(self, defaults=defaults)
        if not dlg.exec():
            return

        config = dlg.get_config()

        # Sync main-tab controls back from dialog choices
        self.batch_fmin.setValue(config["hv_forward"]["fmin"])
        self.batch_fmax.setValue(config["hv_forward"]["fmax"])
        self.batch_nf.setValue(config["hv_forward"]["nf"])
        self.batch_engine_combo.setCurrentText(config["engine_name"])
        self.batch_chk_generate_report.setChecked(config["generate_report"])
        self.batch_chk_dual_resonance.setChecked(
            config["dual_resonance"]["enable"]
        )
        self._dr_ratio = config["dual_resonance"]["separation_ratio_threshold"]
        self._dr_shift = config["dual_resonance"]["separation_shift_threshold"]

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        profiles = [self.batch_list.item(i).text() for i in range(self.batch_list.count())]

        self.output_text.clear()
        self.output_text.append(f"Starting batch processing of {len(profiles)} profiles...")
        self.btn_run_batch.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(profiles))
        self.progress_bar.setValue(0)

        self.worker = BatchWorker(profiles, output_dir, config)
        self.worker.progress.connect(self._on_batch_progress)
        self.worker.finished_signal.connect(self._on_batch_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.output_text.append("Cancelled.")
            self._reset_ui()

    def _on_progress(self, message):
        self.output_text.append(message)

    def _on_batch_progress(self, message, current, total):
        self.output_text.append(message)
        self.progress_bar.setValue(current)

    def _on_single_finished(self, results):
        self.output_text.append("\n--- Analysis Complete ---")
        
        # Get step count from results
        step_results = results.get('step_results', {})
        num_steps = len(step_results) if step_results else results.get('summary', {}).get('total_steps', 'N/A')
        self.output_text.append(f"Steps processed: {num_steps}")
        self.output_text.append(f"Output directory: {results.get('output_directory', self._current_output_dir)}")
        
        # Check if interactive peak selection is enabled
        if getattr(self, '_interactive_peaks', False) and step_results:
            self.output_text.append("\nOpening interactive peak selector...")
            self._open_interactive_peak_picker(results)
        else:
            self._reset_ui()

        # Offer Figure Wizard for reviewing / exporting figures
        strip_dir = results.get('strip_directory', '')
        output_dir = results.get('output_directory', self._current_output_dir)
        if strip_dir and Path(strip_dir).exists():
            has_dr = results.get('dual_resonance') is not None
            reply = QMessageBox.question(
                self, "Review Figures",
                "Open the Figure Wizard to review and export figures?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                wizard = FigureWizardDialog(
                    strip_dir=str(strip_dir),
                    output_dir=str(output_dir),
                    has_dual_resonance=has_dr,
                    parent=self,
                )
                wizard.exec()
                self.output_text.append("Figure wizard closed.")

    def _on_batch_finished(self, results):
        success = sum(1 for r in results if r["success"])
        failed = len(results) - success
        self.output_text.append(f"\n--- Batch Complete ---")
        self.output_text.append(f"Successful: {success}")
        self.output_text.append(f"Failed: {failed}")
        
        if failed > 0:
            self.output_text.append("\nFailed profiles:")
            for r in results:
                if not r["success"]:
                    self.output_text.append(f"  - {Path(r['path']).name}: {r['error']}")
        
        self._reset_ui()

        # Offer to review figures for successful profiles
        ok_results = [r for r in results if r["success"]]
        if ok_results:
            reply = QMessageBox.question(
                self, "Review Batch Figures",
                f"Open Figure Wizard for {len(ok_results)} successful profile(s)?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._open_batch_figure_browser(ok_results)

    def _open_batch_figure_browser(self, ok_results):
        """Let user pick a batch profile and open the wizard for it."""
        from PySide6.QtWidgets import QInputDialog

        names = []
        for r in ok_results:
            res = r.get("result", {})
            strip_dir = res.get("strip_directory", "")
            label = Path(r["path"]).stem
            if strip_dir and Path(strip_dir).exists():
                names.append((label, strip_dir, res.get("output_directory", "")))

        if not names:
            QMessageBox.information(
                self, "No Figures", "No strip directories found."
            )
            return

        labels = [n[0] for n in names]
        chosen, ok = QInputDialog.getItem(
            self, "Select Profile",
            "Choose a profile to review figures:",
            labels, 0, False,
        )
        if not ok:
            return

        idx = labels.index(chosen)
        _, strip_dir, output_dir = names[idx]
        wizard = FigureWizardDialog(
            strip_dir=str(strip_dir),
            output_dir=str(output_dir or strip_dir),
            has_dual_resonance=False,
            parent=self,
        )
        wizard.exec()
        self.output_text.append(f"Figure wizard closed for: {chosen}")

    def _on_error(self, error):
        self.output_text.append(f"\nERROR: {error}")
        self._reset_ui()

    def _reset_ui(self):
        self.btn_run_single.setEnabled(True)
        self.btn_run_batch.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def _open_interactive_peak_picker(self, results):
        """Open interactive peak picker dialog after workflow completion."""
        try:
            # Prepare step data for the picker
            step_data_list = []
            step_results = results.get('step_results', {})
            
            for step_name, step_info in step_results.items():
                # Load HV curve data
                hv_csv = step_info.get('hv_csv')
                if hv_csv and Path(hv_csv).exists():
                    try:
                        data = np.loadtxt(hv_csv, delimiter=',', skiprows=1)
                        freqs = data[:, 0]
                        amps = data[:, 1]
                        
                        step_data_list.append({
                            'name': step_name,
                            'folder': Path(hv_csv).parent,
                            'freqs': freqs,
                            'amps': amps,
                            'model_file': step_info.get('model_file')
                        })
                    except Exception as e:
                        self.output_text.append(f"Warning: Could not load {step_name}: {e}")
            
            if not step_data_list:
                self.output_text.append("No HV curves found for interactive selection.")
                self._reset_ui()
                return
            
            # Sort by step number
            step_data_list.sort(key=lambda x: int(x['name'].split('_')[0].replace('Step', '')))
            
            # Open the picker dialog
            dialog = InteractivePeakPickerDialog(step_data_list, self)
            dialog.peaks_selected.connect(self._on_peaks_selected)
            
            if dialog.exec():
                # User finished selection
                selected_peaks = dialog.get_selected_peaks()
                self.output_text.append(f"\nManually selected peaks for {len(selected_peaks)} steps.")
                self._save_manual_peaks(results, selected_peaks)
            else:
                self.output_text.append("\nInteractive peak selection cancelled.")
            
        except Exception as e:
            self.output_text.append(f"\nError opening peak picker: {e}")
        
        self._reset_ui()
    
    def _on_peaks_selected(self, selected_peaks):
        """Handle peaks selected from interactive dialog."""
        self.output_text.append(f"Peaks selected: {len(selected_peaks)} steps")
    
    def _save_manual_peaks(self, results, selected_peaks):
        """Save manually selected peaks to summary files."""
        import csv
        
        step_results = results.get('step_results', {})
        output_dir = Path(results.get('output_directory', self._current_output_dir))
        
        for step_name, peak_info in selected_peaks.items():
            peak_freq, peak_amp, peak_idx = peak_info
            
            # Update the step summary CSV if it exists
            if step_name in step_results:
                step_folder = step_results[step_name].get('hv_csv')
                if step_folder:
                    step_folder = Path(step_folder).parent
                    summary_csv = step_folder / 'step_summary.csv'
                    
                    # Read existing summary and update peak info
                    if summary_csv.exists():
                        try:
                            rows = []
                            with open(summary_csv, 'r', newline='') as f:
                                reader = csv.DictReader(f)
                                fieldnames = reader.fieldnames
                                for row in reader:
                                    row['Peak_Frequency_Hz'] = f"{peak_freq:.4f}"
                                    row['Peak_Amplitude'] = f"{peak_amp:.2f}"
                                    row['Manual_Selection'] = 'True'
                                    rows.append(row)
                            
                            # Add Manual_Selection field if not present
                            if 'Manual_Selection' not in fieldnames:
                                fieldnames = list(fieldnames) + ['Manual_Selection']
                            
                            with open(summary_csv, 'w', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(rows)
                        except Exception as e:
                            self.output_text.append(f"Warning: Could not update {step_name} summary: {e}")
        
        # Save a combined manual peaks summary
        manual_peaks_csv = output_dir / 'manual_peaks_summary.csv'
        try:
            with open(manual_peaks_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Step', 'Peak_Frequency_Hz', 'Peak_Amplitude', 'Peak_Index'])
                for step_name, (freq, amp, idx) in sorted(selected_peaks.items()):
                    writer.writerow([step_name, f"{freq:.4f}", f"{amp:.2f}", idx])
            self.output_text.append(f"Manual peaks saved to: {manual_peaks_csv}")
        except Exception as e:
            self.output_text.append(f"Warning: Could not save manual peaks summary: {e}")
        
        # Now run post-processing with selected peaks
        self._run_post_processing_with_peaks(results, selected_peaks)
    
    def _run_post_processing_with_peaks(self, results, selected_peaks):
        """Run post-processing and report generation with manually selected peaks."""
        self.output_text.append("\nRunning post-processing with selected peaks...")
        
        try:
            from ...core.hv_postprocess import process
        except ImportError:
            from hvstrip_progressive.core.hv_postprocess import process
        
        step_results = results.get('step_results', {})
        output_dir = Path(results.get('output_directory', self._current_output_dir))
        strip_dir = results.get('strip_directory')
        
        successful_post = 0
        
        for step_name, step_info in step_results.items():
            hv_csv = step_info.get('hv_csv')
            model_file = step_info.get('model_file')
            
            if not hv_csv or not Path(hv_csv).exists():
                continue
            if not model_file or not Path(model_file).exists():
                continue
            
            step_folder = Path(hv_csv).parent
            
            # Create config with manual peak if available
            postprocess_config = {
                "peak_detection": {"preset": "default"},
                "hv_plot": {"x_axis_scale": "log", "y_axis_scale": "log"},
                "vs_plot": {"show": True},
                "output": {"save_separate": True, "save_combined": True}
            }
            
            # Override with manual peak selection
            if step_name in selected_peaks:
                peak_freq, peak_amp, peak_idx = selected_peaks[step_name]
                postprocess_config["peak_detection"] = {
                    "method": "manual",
                    "manual_frequency": peak_freq
                }
            
            try:
                post_results = process(
                    str(hv_csv),
                    str(model_file),
                    str(step_folder),
                    postprocess_config
                )
                successful_post += 1
                
                # Update results
                if step_name in step_results:
                    step_results[step_name].update(post_results)
                    
            except Exception as e:
                self.output_text.append(f"Warning: Post-processing {step_name} failed: {e}")
        
        self.output_text.append(f"Post-processing completed: {successful_post}/{len(step_results)} steps")
        
        # Generate report if enabled
        if getattr(self, '_generate_report', True):
            self._generate_final_report(results, selected_peaks)
    
    def _generate_final_report(self, results, selected_peaks):
        """Generate comprehensive report after manual peak selection."""
        self.output_text.append("\nGenerating comprehensive report...")
        
        try:
            from ...core.report_generator import ProgressiveStrippingReporter
        except ImportError:
            from hvstrip_progressive.core.report_generator import ProgressiveStrippingReporter
        
        output_dir = Path(results.get('output_directory', self._current_output_dir))
        strip_dir = results.get('strip_directory')
        
        if not strip_dir:
            self.output_text.append("Warning: Cannot generate report - strip directory not found")
            return
        
        try:
            reporter = ProgressiveStrippingReporter(
                str(strip_dir),
                str(output_dir / "reports")
            )
            report_files = reporter.generate_comprehensive_report()
            self.output_text.append(f"Report generated: {len(report_files)} files")
            self.output_text.append(f"Report location: {output_dir / 'reports'}")
        except Exception as e:
            self.output_text.append(f"Warning: Report generation failed: {e}")
