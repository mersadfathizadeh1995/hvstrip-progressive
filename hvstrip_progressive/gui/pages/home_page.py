"""
Home Page - Main Workflow Execution
Provides interface for running single or batch workflows
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QTextEdit
)
from qfluentwidgets import (
    ScrollArea,
    PrimaryPushButton,
    PushButton,
    ProgressBar,
    BodyLabel,
    TitleLabel,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    MessageBox,
    ComboBox
)

from ..components.input_panel import InputPanel
from ..workers.workflow_worker import WorkflowWorker
from ..utils.config_manager import ConfigManager


class HomePage(ScrollArea):
    """Home page for workflow execution"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.workflow_worker = None
        self.config_manager = ConfigManager()

        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        # Create scroll widget
        self.scroll_widget = QWidget()
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)
        self.setObjectName("homePage")

        # Main layout
        layout = QVBoxLayout(self.scroll_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Page title
        title = TitleLabel("HVSTRIP-Progressive Workflow")
        layout.addWidget(title)

        # Description
        desc = BodyLabel(
            "Progressive layer stripping analysis of HVSR data. "
            "Select your input files and configure processing mode below."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Input panel
        self.input_panel = InputPanel()
        layout.addWidget(self.input_panel)

        # Processing mode selection
        mode_layout = QHBoxLayout()
        mode_label = BodyLabel("Processing Mode:")
        mode_label.setFixedWidth(150)
        self.mode_combo = ComboBox()
        self.mode_combo.addItems(["Complete Workflow", "Strip Only", "Forward Only", "Postprocess Only"])
        self.mode_combo.setCurrentText("Complete Workflow")

        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Action buttons
        button_layout = QHBoxLayout()

        self.run_button = PrimaryPushButton("Run Workflow", self, FluentIcon.PLAY)
        self.run_button.setFixedWidth(200)
        self.run_button.clicked.connect(self.run_workflow)

        self.stop_button = PushButton("Stop", self, FluentIcon.CANCEL)
        self.stop_button.setFixedWidth(100)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_workflow)

        self.load_config_button = PushButton("Load Config", self, FluentIcon.FOLDER)
        self.load_config_button.setFixedWidth(150)
        self.load_config_button.clicked.connect(self.load_configuration)

        self.save_config_button = PushButton("Save Config", self, FluentIcon.SAVE)
        self.save_config_button.setFixedWidth(150)
        self.save_config_button.clicked.connect(self.save_configuration)

        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        button_layout.addWidget(self.load_config_button)
        button_layout.addWidget(self.save_config_button)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = ProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = BodyLabel("Ready")
        layout.addWidget(self.status_label)

        # Log output
        log_label = BodyLabel("Workflow Log:")
        log_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        layout.addWidget(log_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        self.log_output.setPlaceholderText("Workflow output will appear here...")
        layout.addWidget(self.log_output)

        layout.addStretch()

    def run_workflow(self):
        """Execute the workflow"""
        # Validate inputs
        inputs = self.input_panel.get_inputs()

        if not inputs['model_file']:
            InfoBar.error(
                "Error",
                "Please select a velocity model file",
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
            return

        if not inputs['exe_path']:
            InfoBar.error(
                "Error",
                "Please select HVf.exe path",
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
            return

        if not inputs['output_dir']:
            InfoBar.error(
                "Error",
                "Please select an output directory",
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
            return

        # Check if files exist
        if not Path(inputs['model_file']).exists():
            InfoBar.error(
                "Error",
                "Model file does not exist",
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
            return

        if not Path(inputs['exe_path']).exists():
            InfoBar.error(
                "Error",
                "HVf.exe does not exist",
                duration=3000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
            return

        # Create output directory if it doesn't exist
        Path(inputs['output_dir']).mkdir(parents=True, exist_ok=True)

        # Get configuration from main window
        config = self.get_workflow_config()

        # Clear log
        self.log_output.clear()
        self.append_log("Starting workflow...")

        # Create and start worker
        self.workflow_worker = WorkflowWorker(
            model_file=inputs['model_file'],
            exe_path=inputs['exe_path'],
            output_dir=inputs['output_dir'],
            mode=self.mode_combo.currentText(),
            config=config
        )

        # Connect signals
        self.workflow_worker.progress_updated.connect(self.on_progress_updated)
        self.workflow_worker.log_message.connect(self.append_log)
        self.workflow_worker.workflow_completed.connect(self.on_workflow_completed)
        self.workflow_worker.workflow_failed.connect(self.on_workflow_failed)

        # Update UI
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running...")

        # Start worker
        self.workflow_worker.start()

    def stop_workflow(self):
        """Stop the running workflow"""
        if self.workflow_worker and self.workflow_worker.isRunning():
            self.workflow_worker.stop()
            self.append_log("Stopping workflow...")

    @Slot(int, str)
    def on_progress_updated(self, value, message):
        """Handle progress updates"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    @Slot(str)
    def append_log(self, message):
        """Append message to log output"""
        self.log_output.append(message)

    @Slot(str)
    def on_workflow_completed(self, output_dir):
        """Handle workflow completion"""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Workflow completed successfully!")

        InfoBar.success(
            "Success",
            f"Workflow completed! Results saved to {output_dir}",
            duration=5000,
            position=InfoBarPosition.TOP_RIGHT,
            parent=self.window()
        )

        self.append_log(f"\n=== Workflow Completed ===")
        self.append_log(f"Results saved to: {output_dir}")

    @Slot(str)
    def on_workflow_failed(self, error_message):
        """Handle workflow failure"""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Workflow failed")

        InfoBar.error(
            "Error",
            f"Workflow failed: {error_message}",
            duration=5000,
            position=InfoBarPosition.TOP_RIGHT,
            parent=self.window()
        )

        self.append_log(f"\n=== Workflow Failed ===")
        self.append_log(f"Error: {error_message}")

    def get_workflow_config(self):
        """Get workflow configuration from settings page"""
        # Access settings page from main window
        main_window = self.window()
        if hasattr(main_window, 'settings_page'):
            return main_window.settings_page.get_all_settings()
        return {}

    def load_configuration(self):
        """Load configuration from file"""
        config = self.config_manager.load_config_dialog(self.window())
        if config:
            # Set input panel values
            if 'inputs' in config:
                self.input_panel.set_inputs(config['inputs'])

            # Set mode
            if 'mode' in config:
                index = self.mode_combo.findText(config['mode'])
                if index >= 0:
                    self.mode_combo.setCurrentIndex(index)

            # Set settings in settings page
            main_window = self.window()
            if hasattr(main_window, 'settings_page') and 'settings' in config:
                main_window.settings_page.set_all_settings(config['settings'])

            InfoBar.success(
                "Success",
                "Configuration loaded successfully",
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )

    def save_configuration(self):
        """Save current configuration to file"""
        config = {
            'inputs': self.input_panel.get_inputs(),
            'mode': self.mode_combo.currentText(),
            'settings': self.get_workflow_config()
        }

        if self.config_manager.save_config_dialog(config, self.window()):
            InfoBar.success(
                "Success",
                "Configuration saved successfully",
                duration=2000,
                position=InfoBarPosition.TOP_RIGHT,
                parent=self.window()
            )
