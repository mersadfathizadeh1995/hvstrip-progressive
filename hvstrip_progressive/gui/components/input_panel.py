"""
Input Panel Component
Handles file/directory selection for model files, HVf.exe, and output directory
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget,
    BodyLabel,
    PushButton,
    LineEdit,
    FluentIcon,
    InfoBar,
    InfoBarPosition
)


class InputPanel(CardWidget):
    """Panel for input file and directory selection"""

    # Signals
    model_file_changed = Signal(str)
    exe_path_changed = Signal(str)
    output_dir_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.model_file_path = ""
        self.exe_path = ""
        self.output_dir_path = ""

        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Model file selection
        model_layout = QHBoxLayout()
        self.model_label = BodyLabel("Velocity Model File:")
        self.model_label.setFixedWidth(150)
        self.model_edit = LineEdit()
        self.model_edit.setPlaceholderText("Select HVf format model file (.txt)")
        self.model_edit.setReadOnly(True)
        self.model_browse_btn = PushButton("Browse", self, FluentIcon.FOLDER)
        self.model_browse_btn.clicked.connect(self.browse_model_file)

        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_edit, 1)
        model_layout.addWidget(self.model_browse_btn)
        layout.addLayout(model_layout)

        # HVf.exe path selection
        exe_layout = QHBoxLayout()
        self.exe_label = BodyLabel("HVf.exe Path:")
        self.exe_label.setFixedWidth(150)
        self.exe_edit = LineEdit()
        self.exe_edit.setPlaceholderText("Select HVf.exe executable")
        self.exe_edit.setReadOnly(True)
        self.exe_browse_btn = PushButton("Browse", self, FluentIcon.FOLDER)
        self.exe_browse_btn.clicked.connect(self.browse_exe_path)

        exe_layout.addWidget(self.exe_label)
        exe_layout.addWidget(self.exe_edit, 1)
        exe_layout.addWidget(self.exe_browse_btn)
        layout.addLayout(exe_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_label = BodyLabel("Output Directory:")
        self.output_label.setFixedWidth(150)
        self.output_edit = LineEdit()
        self.output_edit.setPlaceholderText("Select output directory")
        self.output_edit.setReadOnly(True)
        self.output_browse_btn = PushButton("Browse", self, FluentIcon.FOLDER)
        self.output_browse_btn.clicked.connect(self.browse_output_dir)

        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(self.output_browse_btn)
        layout.addLayout(output_layout)

    def browse_model_file(self):
        """Open file browser for model file selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Velocity Model File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            self.model_file_path = file_path
            self.model_edit.setText(file_path)
            self.model_file_changed.emit(file_path)

            # Validate file
            if self.validate_model_file(file_path):
                InfoBar.success(
                    "Success",
                    "Model file loaded successfully",
                    duration=2000,
                    position=InfoBarPosition.TOP_RIGHT,
                    parent=self.window()
                )

    def browse_exe_path(self):
        """Open file browser for HVf.exe selection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HVf.exe",
            "",
            "Executable Files (*.exe);;All Files (*)"
        )

        if file_path:
            self.exe_path = file_path
            self.exe_edit.setText(file_path)
            self.exe_path_changed.emit(file_path)

            # Check if file exists
            if Path(file_path).exists():
                InfoBar.success(
                    "Success",
                    "HVf.exe path set successfully",
                    duration=2000,
                    position=InfoBarPosition.TOP_RIGHT,
                    parent=self.window()
                )

    def browse_output_dir(self):
        """Open directory browser for output directory selection"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )

        if dir_path:
            self.output_dir_path = dir_path
            self.output_edit.setText(dir_path)
            self.output_dir_changed.emit(dir_path)

            InfoBar.success(
                "Success",
                "Output directory set successfully",
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )

    def validate_model_file(self, file_path):
        """Basic validation of model file format"""
        try:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                if len(lines) < 2:
                    InfoBar.warning(
                        "Warning",
                        "Model file appears to be empty or invalid",
                        duration=3000,
                        position=InfoBarPosition.TOP_RIGHT,
                        parent=self.window()
                    )
                    return False
                # Check first line is a number
                try:
                    int(lines[0])
                    return True
                except ValueError:
                    InfoBar.warning(
                        "Warning",
                        "First line should be the number of layers",
                        duration=3000,
                        position=InfoBarPosition.TOP_RIGHT,
                        parent=self.window()
                    )
                    return False
        except Exception as e:
            InfoBar.error(
                "Error",
                f"Failed to read model file: {str(e)}",
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
            return False

    def get_inputs(self):
        """Get all input values"""
        return {
            'model_file': self.model_file_path,
            'exe_path': self.exe_path,
            'output_dir': self.output_dir_path
        }

    def set_inputs(self, inputs):
        """Set input values programmatically"""
        if 'model_file' in inputs and inputs['model_file']:
            self.model_file_path = inputs['model_file']
            self.model_edit.setText(inputs['model_file'])

        if 'exe_path' in inputs and inputs['exe_path']:
            self.exe_path = inputs['exe_path']
            self.exe_edit.setText(inputs['exe_path'])

        if 'output_dir' in inputs and inputs['output_dir']:
            self.output_dir_path = inputs['output_dir']
            self.output_edit.setText(inputs['output_dir'])
