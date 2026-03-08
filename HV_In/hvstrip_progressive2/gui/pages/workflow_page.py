"""
Complete Workflow Page

Runs the complete progressive layer stripping workflow:
1. Layer stripping
2. HV forward modeling
3. Post-processing
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, SpinBox, DoubleSpinBox,
    ProgressBar, TextEdit, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    StrongBodyLabel, TransparentToolButton
, ScrollArea)

# Import from parent package
from ...core.batch_workflow import run_complete_workflow


class WorkflowWorker(QThread):
    """Worker thread for running the complete workflow."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, model_path, output_dir, config):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config

    def run(self):
        """Run the workflow."""
        try:
            results = run_complete_workflow(
                self.model_path,
                self.output_dir,
                self.config
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class WorkflowPage(QWidget):
    """Complete workflow page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workflowPage")
        self.worker = None
        self.initUI()

    def initUI(self):
        """Initialize user interface."""
        # Main layout
        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        scrollArea = ScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Content widget
        contentWidget = QWidget()
        layout = QVBoxLayout(contentWidget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        titleLabel = TitleLabel("Complete Workflow", self)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Run the complete progressive layer stripping analysis workflow.\n"
            "This includes layer stripping, HV forward modeling, and post-processing.",
            self
        )
        descLabel.setWordWrap(True)
        layout.addWidget(descLabel)

        # Input Configuration Card
        inputCard = self.createInputCard()
        layout.addWidget(inputCard)

        # Frequency Configuration Card
        freqCard = self.createFrequencyCard()
        layout.addWidget(freqCard)

        # Progress Card
        progressCard = self.createProgressCard()
        layout.addWidget(progressCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        self.runButton = PrimaryPushButton(FIF.PLAY, "Run Complete Workflow", self)
        self.runButton.clicked.connect(self.runWorkflow)
        buttonLayout.addWidget(self.runButton)

        self.cancelButton = PushButton(FIF.CLOSE, "Cancel", self)
        self.cancelButton.setEnabled(False)
        self.cancelButton.clicked.connect(self.cancelWorkflow)
        buttonLayout.addWidget(self.cancelButton)

        layout.addLayout(buttonLayout)

        # Set scroll area content
        scrollArea.setWidget(contentWidget)
        mainLayout.addWidget(scrollArea)
        layout.addStretch()

    def createInputCard(self):
        """Create input configuration card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Input Configuration", card)
        cardLayout.addWidget(cardTitle)

        # Model file input
        modelLayout = QHBoxLayout()
        modelLabel = BodyLabel("Model File:", card)
        modelLabel.setFixedWidth(150)
        modelLayout.addWidget(modelLabel)

        self.modelEdit = LineEdit(card)
        self.modelEdit.setPlaceholderText("Select velocity model file...")
        modelLayout.addWidget(self.modelEdit)

        modelButton = TransparentToolButton(FIF.FOLDER, card)
        modelButton.clicked.connect(self.selectModelFile)
        modelLayout.addWidget(modelButton)

        cardLayout.addLayout(modelLayout)

        # Output directory input
        outputLayout = QHBoxLayout()
        outputLabel = BodyLabel("Output Directory:", card)
        outputLabel.setFixedWidth(150)
        outputLayout.addWidget(outputLabel)

        self.outputEdit = LineEdit(card)
        self.outputEdit.setPlaceholderText("Select output directory...")
        outputLayout.addWidget(self.outputEdit)

        outputButton = TransparentToolButton(FIF.FOLDER, card)
        outputButton.clicked.connect(self.selectOutputDir)
        outputLayout.addWidget(outputButton)

        cardLayout.addLayout(outputLayout)

        # HVf executable path
        exeLayout = QHBoxLayout()
        exeLabel = BodyLabel("HVf Executable:", card)
        exeLabel.setFixedWidth(150)
        exeLayout.addWidget(exeLabel)

        self.exeEdit = LineEdit(card)
        self.exeEdit.setPlaceholderText("Auto-detected (leave blank for auto-detect)")
        exeLayout.addWidget(self.exeEdit)

        exeButton = TransparentToolButton(FIF.FOLDER, card)
        exeButton.clicked.connect(self.selectExeFile)
        exeLayout.addWidget(exeButton)

        cardLayout.addLayout(exeLayout)

        return card

    def createFrequencyCard(self):
        """Create frequency configuration card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Frequency Configuration", card)
        cardLayout.addWidget(cardTitle)

        # Frequency grid layout
        gridLayout = QHBoxLayout()

        # Minimum frequency
        fminLayout = QVBoxLayout()
        fminLabel = BodyLabel("Min Frequency (Hz):", card)
        fminLayout.addWidget(fminLabel)
        self.fminSpin = DoubleSpinBox(card)
        self.fminSpin.setRange(0.01, 100.0)
        self.fminSpin.setValue(0.2)
        self.fminSpin.setDecimals(2)
        fminLayout.addWidget(self.fminSpin)
        gridLayout.addLayout(fminLayout)

        # Maximum frequency
        fmaxLayout = QVBoxLayout()
        fmaxLabel = BodyLabel("Max Frequency (Hz):", card)
        fmaxLayout.addWidget(fmaxLabel)
        self.fmaxSpin = DoubleSpinBox(card)
        self.fmaxSpin.setRange(0.1, 200.0)
        self.fmaxSpin.setValue(20.0)
        self.fmaxSpin.setDecimals(1)
        fmaxLayout.addWidget(self.fmaxSpin)
        gridLayout.addLayout(fmaxLayout)

        # Number of frequency points
        nfLayout = QVBoxLayout()
        nfLabel = BodyLabel("Frequency Points:", card)
        nfLayout.addWidget(nfLabel)
        self.nfSpin = SpinBox(card)
        self.nfSpin.setRange(10, 500)
        self.nfSpin.setValue(71)
        nfLayout.addWidget(self.nfSpin)
        gridLayout.addLayout(nfLayout)

        cardLayout.addLayout(gridLayout)

        return card

    def createProgressCard(self):
        """Create progress display card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Progress", card)
        cardLayout.addWidget(cardTitle)

        # Progress bar
        self.progressBar = ProgressBar(card)
        self.progressBar.setValue(0)
        cardLayout.addWidget(self.progressBar)

        # Status label
        self.statusLabel = BodyLabel("Ready to start workflow...", card)
        cardLayout.addWidget(self.statusLabel)

        # Output text
        self.outputText = TextEdit(card)
        self.outputText.setReadOnly(True)
        self.outputText.setFixedHeight(200)
        cardLayout.addWidget(self.outputText)

        return card

    def selectModelFile(self):
        """Select model file."""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Velocity Model File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file:
            self.modelEdit.setText(file)

    def selectOutputDir(self):
        """Select output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        if directory:
            self.outputEdit.setText(directory)

    def selectExeFile(self):
        """Select HVf executable file."""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select HVf Executable",
            "",
            "Executable Files (*.exe HVf HVf_Serial);;All Files (*)"
        )
        if file:
            self.exeEdit.setText(file)

    def runWorkflow(self):
        """Run the complete workflow."""
        # Validate inputs
        model_path = self.modelEdit.text()
        output_dir = self.outputEdit.text()

        if not model_path:
            InfoBar.error(
                "Error",
                "Please select a model file",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        if not output_dir:
            InfoBar.error(
                "Error",
                "Please select an output directory",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        if not Path(model_path).exists():
            InfoBar.error(
                "Error",
                "Model file does not exist",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        # Build configuration
        config = {
            "hv_forward": {
                "fmin": self.fminSpin.value(),
                "fmax": self.fmaxSpin.value(),
                "nf": self.nfSpin.value()
            }
        }

        # Add exe path if specified
        exe_path = self.exeEdit.text()
        if exe_path:
            config["hv_forward"]["exe_path"] = exe_path

        # Disable run button, enable cancel
        self.runButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        self.progressBar.setValue(0)
        self.statusLabel.setText("Starting workflow...")
        self.outputText.clear()

        # Create and start worker
        self.worker = WorkflowWorker(model_path, output_dir, config)
        self.worker.progress.connect(self.updateProgress)
        self.worker.finished.connect(self.workflowFinished)
        self.worker.error.connect(self.workflowError)
        self.worker.start()

    def cancelWorkflow(self):
        """Cancel the running workflow."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.statusLabel.setText("Workflow cancelled")
            self.runButton.setEnabled(True)
            self.cancelButton.setEnabled(False)
            InfoBar.warning(
                "Cancelled",
                "Workflow was cancelled by user",
                parent=self,
                position=InfoBarPosition.TOP
            )

    def updateProgress(self, message):
        """Update progress display."""
        self.outputText.append(message)
        self.statusLabel.setText(message)

    def workflowFinished(self, results):
        """Handle workflow completion."""
        self.progressBar.setValue(100)
        self.statusLabel.setText("Workflow completed successfully!")

        # Display summary
        summary = results.get("summary", {})
        self.outputText.append("\n" + "="*60)
        self.outputText.append("WORKFLOW COMPLETED SUCCESSFULLY!")
        self.outputText.append("="*60)
        self.outputText.append(f"Total steps processed: {summary.get('total_steps', 0)}")
        self.outputText.append(f"Successful HV computations: {summary.get('successful_hv', 0)}")
        self.outputText.append(f"Successful post-processing: {summary.get('successful_post', 0)}")
        self.outputText.append(f"Completion rate: {summary.get('completion_rate', 0):.1f}%")
        self.outputText.append(f"\nAll outputs saved in: {results.get('output_directory', '')}")

        # Re-enable buttons
        self.runButton.setEnabled(True)
        self.cancelButton.setEnabled(False)

        # Show success message
        InfoBar.success(
            "Success",
            f"Workflow completed successfully! Processed {summary.get('total_steps', 0)} steps.",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )

    def workflowError(self, error_msg):
        """Handle workflow error."""
        self.statusLabel.setText("Workflow failed!")
        self.outputText.append(f"\n❌ ERROR: {error_msg}")

        # Re-enable buttons
        self.runButton.setEnabled(True)
        self.cancelButton.setEnabled(False)

        # Show error message
        InfoBar.error(
            "Error",
            f"Workflow failed: {error_msg}",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )
