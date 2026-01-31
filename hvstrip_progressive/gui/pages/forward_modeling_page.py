"""
HV Forward Modeling page with integrated Profile Editor.
"""

import tempfile
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QLineEdit, QTextEdit, QSplitter,
    QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Signal, QThread

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.layer_table_widget import LayerTableWidget
from ..widgets.profile_preview_widget import ProfilePreviewWidget
from ...core.soil_profile import SoilProfile, Layer
from ...core.velocity_utils import VelocityConverter
from ...core.hv_forward import compute_hv_curve


class ForwardWorker(QThread):
    """Worker thread for HV forward modeling."""
    finished_signal = Signal(tuple)
    error = Signal(str)

    def __init__(self, model_path, config):
        super().__init__()
        self.model_path = model_path
        self.config = config

    def run(self):
        try:
            freqs, amps = compute_hv_curve(self.model_path, self.config)
            self.finished_signal.emit((freqs, amps))
        except Exception as e:
            self.error.emit(str(e))


class ForwardModelingPage(QWidget):
    """
    HV Forward Modeling with integrated Profile Editor.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ForwardModelingPage")
        self._current_profile = None
        self._temp_model_path = None
        self._last_freqs = None
        self._last_amps = None
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("HV Forward Modeling")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(header)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        input_tabs = QTabWidget()
        input_tabs.addTab(self._create_file_tab(), "From File")
        input_tabs.addTab(self._create_dinver_tab(), "From Dinver")
        input_tabs.addTab(self._create_editor_tab(), "Profile Editor")
        left_layout.addWidget(input_tabs)
        self.input_tabs = input_tabs

        config_group = QGroupBox("Frequency Parameters")
        config_layout = QHBoxLayout(config_group)

        config_layout.addWidget(QLabel("Min:"))
        self.fmin_spin = QDoubleSpinBox()
        self.fmin_spin.setRange(0.01, 10.0)
        self.fmin_spin.setValue(0.5)
        self.fmin_spin.setSingleStep(0.1)
        self.fmin_spin.setSuffix(" Hz")
        config_layout.addWidget(self.fmin_spin)

        config_layout.addWidget(QLabel("Max:"))
        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(1.0, 100.0)
        self.fmax_spin.setValue(20.0)
        self.fmax_spin.setSingleStep(1.0)
        self.fmax_spin.setSuffix(" Hz")
        config_layout.addWidget(self.fmax_spin)

        config_layout.addWidget(QLabel("Points:"))
        self.nf_spin = QSpinBox()
        self.nf_spin.setRange(50, 2000)
        self.nf_spin.setValue(500)
        config_layout.addWidget(self.nf_spin)

        config_layout.addStretch()
        left_layout.addWidget(config_group)

        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Compute HV Curve")
        self.btn_run.setStyleSheet(
            "background-color: #0078d4; color: white; padding: 8px 16px; font-size: 13px;"
        )
        self.btn_run.clicked.connect(self._run_forward)
        btn_layout.addWidget(self.btn_run)

        self.btn_save = QPushButton("Save Results...")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_results)
        btn_layout.addWidget(self.btn_save)

        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(120)
        results_layout.addWidget(self.results_text)
        left_layout.addWidget(results_group)

        main_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.hv_plot = MatplotlibWidget(figsize=(8, 6))
        right_layout.addWidget(self.hv_plot)

        plot_options = QHBoxLayout()
        self.log_x_check = QCheckBox("Log X")
        self.log_x_check.stateChanged.connect(self._update_plot)
        plot_options.addWidget(self.log_x_check)

        self.log_y_check = QCheckBox("Log Y")
        self.log_y_check.stateChanged.connect(self._update_plot)
        plot_options.addWidget(self.log_y_check)

        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self._update_plot)
        plot_options.addWidget(self.grid_check)

        plot_options.addStretch()
        right_layout.addLayout(plot_options)

        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([500, 500])

        layout.addWidget(main_splitter)
        self._draw_empty_plot()

    def _create_file_tab(self):
        """Create the From File tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Model File:"))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Select HVf format model file (.txt)")
        file_row.addWidget(self.model_edit)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_model)
        file_row.addWidget(btn_browse)
        layout.addLayout(file_row)

        self.file_preview = ProfilePreviewWidget()
        layout.addWidget(self.file_preview)

        return widget

    def _create_dinver_tab(self):
        """Create the From Dinver tab for importing Dinver output files."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_label = QLabel(
            "Import velocity model from Dinver output files.\n"
            "Requires Vs file; Vp and density files are optional."
        )
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)

        vs_row = QHBoxLayout()
        vs_row.addWidget(QLabel("Vs File:"))
        self.dinver_vs_edit = QLineEdit()
        self.dinver_vs_edit.setPlaceholderText("Required: *_vs.txt")
        vs_row.addWidget(self.dinver_vs_edit)
        btn_vs = QPushButton("...")
        btn_vs.setMaximumWidth(30)
        btn_vs.clicked.connect(lambda: self._browse_dinver_file("vs"))
        vs_row.addWidget(btn_vs)
        layout.addLayout(vs_row)

        vp_row = QHBoxLayout()
        vp_row.addWidget(QLabel("Vp File:"))
        self.dinver_vp_edit = QLineEdit()
        self.dinver_vp_edit.setPlaceholderText("Optional: *_vp.txt")
        vp_row.addWidget(self.dinver_vp_edit)
        btn_vp = QPushButton("...")
        btn_vp.setMaximumWidth(30)
        btn_vp.clicked.connect(lambda: self._browse_dinver_file("vp"))
        vp_row.addWidget(btn_vp)
        layout.addLayout(vp_row)

        rho_row = QHBoxLayout()
        rho_row.addWidget(QLabel("Density File:"))
        self.dinver_rho_edit = QLineEdit()
        self.dinver_rho_edit.setPlaceholderText("Optional: *_rho.txt")
        rho_row.addWidget(self.dinver_rho_edit)
        btn_rho = QPushButton("...")
        btn_rho.setMaximumWidth(30)
        btn_rho.clicked.connect(lambda: self._browse_dinver_file("rho"))
        rho_row.addWidget(btn_rho)
        layout.addLayout(rho_row)

        btn_load = QPushButton("Load Dinver Profile")
        btn_load.clicked.connect(self._load_dinver_profile)
        layout.addWidget(btn_load)

        self.dinver_preview = ProfilePreviewWidget()
        layout.addWidget(self.dinver_preview)

        self.dinver_status = QLabel("")
        layout.addWidget(self.dinver_status)

        return widget

    def _browse_dinver_file(self, file_type: str):
        """Browse for Dinver file."""
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {file_type.upper()} File", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            if file_type == "vs":
                self.dinver_vs_edit.setText(path)
                base = path.replace("_vs.txt", "").replace("_Vs.txt", "")
                for suffix, edit in [("_vp.txt", self.dinver_vp_edit), ("_rho.txt", self.dinver_rho_edit)]:
                    candidate = base + suffix
                    if Path(candidate).exists() and not edit.text():
                        edit.setText(candidate)
            elif file_type == "vp":
                self.dinver_vp_edit.setText(path)
            elif file_type == "rho":
                self.dinver_rho_edit.setText(path)

    def _load_dinver_profile(self):
        """Load profile from Dinver files."""
        vs_file = self.dinver_vs_edit.text().strip()
        if not vs_file:
            QMessageBox.warning(self, "Error", "Please select a Vs file.")
            return

        vp_file = self.dinver_vp_edit.text().strip() or None
        rho_file = self.dinver_rho_edit.text().strip() or None

        try:
            profile = SoilProfile.from_dinver_files(vs_file, vp_file, rho_file)
            self._dinver_profile = profile
            self.dinver_preview.set_profile(profile)
            self.dinver_status.setText(
                f"Loaded: {len(profile.layers)} layers, "
                f"depth: {profile.get_total_thickness():.1f} m"
            )
            self.dinver_status.setStyleSheet("color: green;")
        except Exception as e:
            self._dinver_profile = None
            self.dinver_status.setText(f"Error: {e}")
            self.dinver_status.setStyleSheet("color: red;")

    def _create_editor_tab(self):
        """Create the Profile Editor tab with full functionality."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        editor_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        file_ops = QHBoxLayout()
        btn_new = QPushButton("New")
        btn_new.clicked.connect(self._new_profile)
        file_ops.addWidget(btn_new)

        btn_open = QPushButton("Open...")
        btn_open.clicked.connect(self._open_profile)
        file_ops.addWidget(btn_open)

        btn_save_profile = QPushButton("Save...")
        btn_save_profile.clicked.connect(self._save_profile)
        file_ops.addWidget(btn_save_profile)

        file_ops.addStretch()
        left_layout.addLayout(file_ops)

        self.layer_table = LayerTableWidget()
        self.layer_table.profile_changed.connect(self._on_profile_changed)
        left_layout.addWidget(self.layer_table)

        ref_group = QGroupBox("Poisson's Ratio Reference")
        ref_layout = QVBoxLayout(ref_group)
        self.ref_table = QTableWidget()
        self.ref_table.setColumnCount(4)
        self.ref_table.setHorizontalHeaderLabels(["Material", "Min", "Max", "Typical"])
        self.ref_table.horizontalHeader().setStretchLastSection(True)
        self.ref_table.setMaximumHeight(150)
        self._populate_reference_table()
        ref_layout.addWidget(self.ref_table)
        left_layout.addWidget(ref_group)

        editor_splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.profile_preview = ProfilePreviewWidget()
        right_layout.addWidget(self.profile_preview)

        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_group)
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(80)
        validation_layout.addWidget(self.validation_text)
        right_layout.addWidget(validation_group)

        editor_splitter.addWidget(right_widget)
        editor_splitter.setSizes([400, 250])

        layout.addWidget(editor_splitter)

        self._create_default_profile()

        return widget

    def _populate_reference_table(self):
        """Populate the reference table with typical nu values."""
        data = VelocityConverter.get_typical_values_table()
        self.ref_table.setRowCount(len(data))
        for row, (material, nu_min, nu_max, nu_typical) in enumerate(data):
            self.ref_table.setItem(row, 0, QTableWidgetItem(material))
            self.ref_table.setItem(row, 1, QTableWidgetItem(f"{nu_min:.2f}"))
            self.ref_table.setItem(row, 2, QTableWidgetItem(f"{nu_max:.2f}"))
            self.ref_table.setItem(row, 3, QTableWidgetItem(f"{nu_typical:.2f}"))
        self.ref_table.resizeColumnsToContents()

    def _create_default_profile(self):
        """Create a default profile."""
        profile = SoilProfile(name="New Profile")
        profile.add_layer(Layer(thickness=5.0, vs=150, density=1700))
        profile.add_layer(Layer(thickness=10.0, vs=300, density=1900))
        profile.add_layer(Layer(thickness=0, vs=600, density=2100, is_halfspace=True))
        self.layer_table.set_profile(profile)
        self.profile_preview.set_profile(profile)
        self._current_profile = profile
        self._validate_profile()

    def _new_profile(self):
        """Create a new empty profile."""
        self._create_default_profile()

    def _open_profile(self):
        """Open a profile file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Profile", "",
            "All Supported (*.txt *.csv);;Text Files (*.txt);;CSV Files (*.csv)"
        )
        if path:
            try:
                if path.lower().endswith('.csv'):
                    profile = SoilProfile.from_csv_file(path)
                else:
                    profile = SoilProfile.from_txt_file(path)
                self.layer_table.set_profile(profile)
                self.profile_preview.set_profile(profile)
                self._current_profile = profile
                self._validate_profile()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def _save_profile(self):
        """Save profile to file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", "", "HVf Format (*.txt);;CSV Format (*.csv)"
        )
        if path:
            try:
                profile = self.layer_table.get_profile()
                if path.lower().endswith('.csv'):
                    profile.save_csv(path)
                else:
                    profile.save_hvf(path)
                QMessageBox.information(self, "Saved", f"Profile saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _on_profile_changed(self):
        """Handle profile changes from editor."""
        self._current_profile = self.layer_table.get_profile()
        self.profile_preview.set_profile(self._current_profile)
        self._validate_profile()

    def _validate_profile(self):
        """Validate and show status."""
        profile = self.layer_table.get_profile()
        is_valid, errors = profile.validate()
        if is_valid:
            self.validation_text.setStyleSheet("color: green;")
            self.validation_text.setText(
                f"Valid profile\n"
                f"Layers: {len(profile.layers)}, "
                f"Depth: {profile.get_total_thickness():.1f} m"
            )
        else:
            self.validation_text.setStyleSheet("color: red;")
            self.validation_text.setText("Errors:\n" + "\n".join(errors))

    def _browse_model(self):
        """Browse for model file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            self.model_edit.setText(path)
            try:
                profile = SoilProfile.from_txt_file(path)
                self.file_preview.set_profile(profile)
            except Exception as e:
                self.results_text.setText(f"Error loading preview: {e}")

    def _run_forward(self):
        """Run forward modeling."""
        tab_index = self.input_tabs.currentIndex()
        
        if tab_index == 0:  # From File
            model_path = self.model_edit.text().strip()
            if not model_path or not Path(model_path).exists():
                QMessageBox.warning(self, "Error", "Please select a valid model file.")
                return
        elif tab_index == 1:  # From Dinver
            if not hasattr(self, '_dinver_profile') or self._dinver_profile is None:
                QMessageBox.warning(self, "Error", "Please load a Dinver profile first.")
                return
            profile = self._dinver_profile
            is_valid, errors = profile.validate()
            if not is_valid:
                QMessageBox.warning(self, "Error", "Invalid profile:\n" + "\n".join(errors))
                return
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(profile.to_hvf_format())
                model_path = f.name
            self._temp_model_path = model_path
        else:  # Profile Editor
            profile = self.layer_table.get_profile()
            is_valid, errors = profile.validate()
            if not is_valid:
                QMessageBox.warning(self, "Error", "Invalid profile:\n" + "\n".join(errors))
                return

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(profile.to_hvf_format())
                model_path = f.name
            self._temp_model_path = model_path

        config = {
            "fmin": self.fmin_spin.value(),
            "fmax": self.fmax_spin.value(),
            "nf": self.nf_spin.value(),
        }

        self.btn_run.setEnabled(False)
        self.results_text.setText("Computing HV curve...")

        self.worker = ForwardWorker(model_path, config)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, result):
        freqs, amps = result
        self._last_freqs = freqs
        self._last_amps = amps

        peak_idx = amps.index(max(amps))
        peak_freq = freqs[peak_idx]
        peak_amp = amps[peak_idx]

        self.results_text.setText(
            f"Peak Frequency: {peak_freq:.3f} Hz\n"
            f"Peak Amplitude: {peak_amp:.2f}\n"
            f"Range: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz"
        )

        self._update_plot()
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)

        if self._temp_model_path:
            try:
                Path(self._temp_model_path).unlink()
            except:
                pass
            self._temp_model_path = None

    def _on_error(self, error):
        self.results_text.setText(f"Error: {error}")
        self.btn_run.setEnabled(True)

        if self._temp_model_path:
            try:
                Path(self._temp_model_path).unlink()
            except:
                pass
            self._temp_model_path = None

    def _update_plot(self):
        """Update the HV curve plot."""
        if self._last_freqs is None or self._last_amps is None:
            return

        fig = self.hv_plot.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)

        ax.plot(self._last_freqs, self._last_amps, 'b-', linewidth=1.5, label='H/V')

        peak_idx = self._last_amps.index(max(self._last_amps))
        ax.scatter(
            [self._last_freqs[peak_idx]], [self._last_amps[peak_idx]],
            color='red', s=100, zorder=5, label=f'Peak: {self._last_freqs[peak_idx]:.2f} Hz'
        )

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('H/V Amplitude', fontsize=11)
        ax.set_title('HV Spectral Ratio', fontsize=12)

        if self.log_x_check.isChecked():
            ax.set_xscale('log')
        if self.log_y_check.isChecked():
            ax.set_yscale('log')
        if self.grid_check.isChecked():
            ax.grid(True, alpha=0.3)

        ax.legend(loc='upper right')
        ax.set_xlim(self._last_freqs[0], self._last_freqs[-1])

        self.hv_plot.refresh()

    def _draw_empty_plot(self):
        """Draw empty placeholder plot."""
        fig = self.hv_plot.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, "Click 'Compute HV Curve' to generate results",
            ha='center', va='center', fontsize=12, color='gray'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.hv_plot.refresh()

    def _save_results(self):
        """Save HV curve results to CSV."""
        if self._last_freqs is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "hv_curve.csv", "CSV Files (*.csv)"
        )
        if path:
            with open(path, 'w') as f:
                f.write("frequency,amplitude\n")
                for freq, amp in zip(self._last_freqs, self._last_amps):
                    f.write(f"{freq},{amp}\n")
            self.results_text.append(f"\nSaved to: {path}")
