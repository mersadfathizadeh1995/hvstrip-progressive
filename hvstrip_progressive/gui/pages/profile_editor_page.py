"""
Soil Profile Editor page for creating and editing velocity models.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QLineEdit, QTextEdit, QSplitter,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Signal

from ..widgets.layer_table_widget import LayerTableWidget
from ..widgets.profile_preview_widget import ProfilePreviewWidget
from ...core.soil_profile import SoilProfile, Layer
from ...core.velocity_utils import VelocityConverter


class ProfileEditorPage(QWidget):
    """
    Page for creating and editing soil profiles.
    
    Features:
    - Manual layer entry via table
    - Import from CSV, TXT, or HVf files
    - Export to various formats
    - Live Vs profile preview
    - Vp auto-calculation with nu suggestions
    """
    
    profile_ready = Signal(SoilProfile)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ProfileEditorPage")
        self._current_file = None
        self._setup_ui()
        self._create_default_profile()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        title = QLabel("Soil Profile Editor")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        file_group = QGroupBox("File Operations")
        file_layout = QHBoxLayout(file_group)
        
        self.btn_new = QPushButton("New Profile")
        self.btn_new.clicked.connect(self._new_profile)
        file_layout.addWidget(self.btn_new)
        
        self.btn_open = QPushButton("Open File...")
        self.btn_open.clicked.connect(self._open_file)
        file_layout.addWidget(self.btn_open)
        
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save_file)
        file_layout.addWidget(self.btn_save)
        
        self.btn_save_as = QPushButton("Save As...")
        self.btn_save_as.clicked.connect(self._save_file_as)
        file_layout.addWidget(self.btn_save_as)
        
        file_layout.addStretch()
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.file_label)
        
        main_layout.addWidget(file_group)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        table_group = QGroupBox("Layer Properties")
        table_layout = QVBoxLayout(table_group)
        
        self.layer_table = LayerTableWidget()
        self.layer_table.profile_changed.connect(self._on_profile_changed)
        table_layout.addWidget(self.layer_table)
        
        left_layout.addWidget(table_group)
        
        ref_group = QGroupBox("Poisson's Ratio Reference")
        ref_layout = QVBoxLayout(ref_group)
        
        self.ref_table = QTableWidget()
        self.ref_table.setColumnCount(4)
        self.ref_table.setHorizontalHeaderLabels(["Material", "Nu Min", "Nu Max", "Typical"])
        self.ref_table.horizontalHeader().setStretchLastSection(True)
        self.ref_table.setMaximumHeight(200)
        self._populate_reference_table()
        
        ref_layout.addWidget(self.ref_table)
        left_layout.addWidget(ref_group)
        
        splitter.addWidget(left_widget)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        preview_group = QGroupBox("Profile Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.profile_preview = ProfilePreviewWidget()
        preview_layout.addWidget(self.profile_preview)
        
        right_layout.addWidget(preview_group)
        
        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_group)
        
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(100)
        self.validation_text.setStyleSheet("font-family: monospace;")
        validation_layout.addWidget(self.validation_text)
        
        right_layout.addWidget(validation_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter, 1)
        
        action_layout = QHBoxLayout()
        
        self.btn_export_hvf = QPushButton("Export to HVf Format")
        self.btn_export_hvf.clicked.connect(self._export_hvf)
        action_layout.addWidget(self.btn_export_hvf)
        
        self.btn_export_csv = QPushButton("Export to CSV")
        self.btn_export_csv.clicked.connect(self._export_csv)
        action_layout.addWidget(self.btn_export_csv)
        
        action_layout.addStretch()
        
        self.btn_use_profile = QPushButton("Use This Profile for Forward Modeling")
        self.btn_use_profile.setStyleSheet(
            "background-color: #0078d4; color: white; padding: 8px 16px;"
        )
        self.btn_use_profile.clicked.connect(self._use_profile)
        action_layout.addWidget(self.btn_use_profile)
        
        main_layout.addLayout(action_layout)
    
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
        """Create a default profile with example layers."""
        profile = SoilProfile(name="New Profile")
        
        profile.add_layer(Layer(thickness=5.0, vs=150, density=1700))
        profile.add_layer(Layer(thickness=10.0, vs=250, density=1850))
        profile.add_layer(Layer(thickness=15.0, vs=400, density=2000))
        profile.add_layer(Layer(thickness=0, vs=800, density=2200, is_halfspace=True))
        
        self.layer_table.set_profile(profile)
        self.profile_preview.set_profile(profile)
        self._validate_profile()
    
    def _on_profile_changed(self):
        """Handle profile changes from the layer table."""
        profile = self.layer_table.get_profile()
        self.profile_preview.set_profile(profile)
        self._validate_profile()
    
    def _validate_profile(self):
        """Validate the current profile and show results."""
        profile = self.layer_table.get_profile()
        is_valid, errors = profile.validate()
        
        if is_valid:
            self.validation_text.setStyleSheet(
                "font-family: monospace; color: green;"
            )
            self.validation_text.setText(
                "Profile is valid.\n"
                f"Total layers: {len(profile.layers)}\n"
                f"Total thickness: {profile.get_total_thickness():.1f} m"
            )
            self.btn_use_profile.setEnabled(True)
        else:
            self.validation_text.setStyleSheet(
                "font-family: monospace; color: red;"
            )
            self.validation_text.setText("Validation errors:\n" + "\n".join(errors))
            self.btn_use_profile.setEnabled(False)
    
    def _new_profile(self):
        """Create a new empty profile."""
        reply = QMessageBox.question(
            self, "New Profile",
            "Create a new profile? Unsaved changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._current_file = None
            self.file_label.setText("No file loaded")
            self._create_default_profile()
    
    def _open_file(self):
        """Open a profile file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Profile",
            "",
            "All Supported (*.txt *.csv *.hvf);;Text Files (*.txt);;CSV Files (*.csv);;HVf Files (*.hvf)"
        )
        
        if not file_path:
            return
        
        try:
            path = Path(file_path)
            
            if path.suffix.lower() == '.csv':
                profile = SoilProfile.from_csv_file(file_path)
            else:
                profile = SoilProfile.from_txt_file(file_path)
            
            self._current_file = file_path
            self.file_label.setText(path.name)
            self.layer_table.set_profile(profile)
            self.profile_preview.set_profile(profile)
            self._validate_profile()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load file:\n{str(e)}"
            )
    
    def _save_file(self):
        """Save to current file or prompt for new file."""
        if self._current_file:
            self._save_to_file(self._current_file)
        else:
            self._save_file_as()
    
    def _save_file_as(self):
        """Save profile to a new file."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Profile",
            "",
            "HVf Format (*.txt);;CSV Format (*.csv)"
        )
        
        if file_path:
            self._save_to_file(file_path)
            self._current_file = file_path
            self.file_label.setText(Path(file_path).name)
    
    def _save_to_file(self, file_path: str):
        """Save profile to specified file."""
        try:
            profile = self.layer_table.get_profile()
            path = Path(file_path)
            
            if path.suffix.lower() == '.csv':
                profile.save_csv(file_path)
            else:
                profile.save_hvf(file_path)
            
            QMessageBox.information(
                self, "Saved",
                f"Profile saved to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to save file:\n{str(e)}"
            )
    
    def _export_hvf(self):
        """Export profile to HVf format."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export HVf",
            "",
            "HVf Format (*.txt)"
        )
        
        if file_path:
            try:
                profile = self.layer_table.get_profile()
                profile.save_hvf(file_path)
                QMessageBox.information(
                    self, "Exported",
                    f"Profile exported to HVf format:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to export:\n{str(e)}"
                )
    
    def _export_csv(self):
        """Export profile to CSV format."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            "",
            "CSV Format (*.csv)"
        )
        
        if file_path:
            try:
                profile = self.layer_table.get_profile()
                profile.save_csv(file_path, include_computed=True)
                QMessageBox.information(
                    self, "Exported",
                    f"Profile exported to CSV:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error",
                    f"Failed to export:\n{str(e)}"
                )
    
    def _use_profile(self):
        """Emit the current profile for use in forward modeling."""
        profile = self.layer_table.get_profile()
        is_valid, _ = profile.validate()
        
        if is_valid:
            self.profile_ready.emit(profile)
            QMessageBox.information(
                self, "Profile Ready",
                "Profile is ready for forward modeling.\n"
                "Switch to the Forward Modeling tab to run the analysis."
            )
        else:
            QMessageBox.warning(
                self, "Invalid Profile",
                "Please fix validation errors before using this profile."
            )
    
    def get_profile(self) -> SoilProfile:
        """Get the current profile."""
        return self.layer_table.get_profile()
