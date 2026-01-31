"""
Batch Processing Page

Process multiple velocity models in batch mode with complete workflow.
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, SpinBox, DoubleSpinBox, CheckBox,
    TextEdit, ProgressBar, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    TransparentToolButton
, ScrollArea)

# Import from parent package
from ...core.batch_workflow import run_complete_workflow


class BatchWorker(QThread):
    """Worker thread for batch processing."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, profiles_dir, output_dir, file_pattern, config):
        super().__init__()
        self.profiles_dir = profiles_dir
        self.output_dir = output_dir
        self.file_pattern = file_pattern
        self.config = config

    def run(self):
        """Run batch processing."""
        try:
            profiles_dir = Path(self.profiles_dir)
            output_dir = Path(self.output_dir)

            # Find all matching profiles
            profile_files = list(profiles_dir.glob(self.file_pattern))

            if not profile_files:
                self.error.emit(f"No files matching pattern '{self.file_pattern}' found in {profiles_dir}")
                return

            self.progress.emit(f"Found {len(profile_files)} profiles to process")

            all_results = []
            successful = 0
            failed = 0

            for i, profile_file in enumerate(profile_files, 1):
                self.progress.emit(f"\nProcessing {i}/{len(profile_files)}: {profile_file.name}")

                try:
                    # Create output directory for this profile
                    profile_output = output_dir / profile_file.stem

                    # Run complete workflow
                    result = run_complete_workflow(
                        str(profile_file),
                        str(profile_output),
                        self.config
                    )

                    all_results.append({
                        'profile': profile_file.name,
                        'status': 'success',
                        'result': result
                    })
                    successful += 1
                    self.progress.emit(f"  Success: {profile_file.name}")

                except Exception as e:
                    all_results.append({
                        'profile': profile_file.name,
                        'status': 'failed',
                        'error': str(e)
                    })
                    failed += 1
                    self.progress.emit(f"  Failed: {str(e)}")

            # Compile summary
            summary = {
                'total_profiles': len(profile_files),
                'successful': successful,
                'failed': failed,
                'results': all_results,
                'output_directory': str(output_dir)
            }

            self.finished.emit(summary)

        except Exception as e:
            self.error.emit(str(e))


class BatchPage(QWidget):
    """Batch processing page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("batchPage")
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
        titleLabel = TitleLabel("Batch Processing", self)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Process multiple velocity models in batch mode.\n"
            "Runs complete workflow (stripping, HV forward, post-processing) for all profiles.",
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

        # Options Card
        optionsCard = self.createOptionsCard()
        layout.addWidget(optionsCard)

        # Progress Card
        progressCard = self.createProgressCard()
        layout.addWidget(progressCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        self.runButton = PrimaryPushButton(FIF.IOT, "Start Batch Processing", self)
        self.runButton.clicked.connect(self.runBatch)
        buttonLayout.addWidget(self.runButton)

        self.cancelButton = PushButton(FIF.CLOSE, "Cancel", self)
        self.cancelButton.setEnabled(False)
        self.cancelButton.clicked.connect(self.cancelBatch)
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

        # Profiles directory
        profilesLayout = QHBoxLayout()
        profilesLabel = BodyLabel("Profiles Directory:", card)
        profilesLabel.setFixedWidth(150)
        profilesLayout.addWidget(profilesLabel)

        self.profilesEdit = LineEdit(card)
        self.profilesEdit.setPlaceholderText("Select directory containing velocity models...")
        profilesLayout.addWidget(self.profilesEdit)

        profilesButton = TransparentToolButton(FIF.FOLDER, card)
        profilesButton.clicked.connect(self.selectProfilesDir)
        profilesLayout.addWidget(profilesButton)

        cardLayout.addLayout(profilesLayout)

        # File pattern
        patternLayout = QHBoxLayout()
        patternLabel = BodyLabel("File Pattern:", card)
        patternLabel.setFixedWidth(150)
        patternLayout.addWidget(patternLabel)

        self.patternEdit = LineEdit(card)
        self.patternEdit.setText("*.txt")
        self.patternEdit.setPlaceholderText("File pattern (e.g., *.txt, profile_*.txt)")
        patternLayout.addWidget(self.patternEdit)

        cardLayout.addLayout(patternLayout)

        # Output directory
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

        # HVf executable
        exeLayout = QHBoxLayout()
        exeLabel = BodyLabel("HVf Executable:", card)
        exeLabel.setFixedWidth(150)
        exeLayout.addWidget(exeLabel)

        self.exeEdit = LineEdit(card)
        self.exeEdit.setPlaceholderText("Optional: HVf executable path (auto-detected)")
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

    def createOptionsCard(self):
        """Create options card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Options", card)
        cardLayout.addWidget(cardTitle)

        # Generate report checkbox
        self.reportCheck = CheckBox("Generate comprehensive report after batch processing", card)
        self.reportCheck.setChecked(True)
        cardLayout.addWidget(self.reportCheck)

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
        self.statusLabel = BodyLabel("Ready to start batch processing...", card)
        cardLayout.addWidget(self.statusLabel)

        # Output text
        self.outputText = TextEdit(card)
        self.outputText.setReadOnly(True)
        self.outputText.setFixedHeight(250)
        cardLayout.addWidget(self.outputText)

        return card

    def selectProfilesDir(self):
        """Select profiles directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Profiles Directory"
        )
        if directory:
            self.profilesEdit.setText(directory)

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

    def runBatch(self):
        """Run batch processing."""
        profiles_dir = self.profilesEdit.text()
        output_dir = self.outputEdit.text()
        file_pattern = self.patternEdit.text()

        # Validate inputs
        if not profiles_dir or not output_dir:
            InfoBar.error(
                "Error",
                "Please select profiles directory and output directory",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        if not Path(profiles_dir).exists():
            InfoBar.error(
                "Error",
                "Profiles directory does not exist",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        if not file_pattern:
            file_pattern = "*.txt"

        # Build configuration
        config = {
            "hv_forward": {
                "fmin": self.fminSpin.value(),
                "fmax": self.fmaxSpin.value(),
                "nf": self.nfSpin.value()
            },
            "generate_report": self.reportCheck.isChecked()
        }

        # Add exe path if specified
        exe_path = self.exeEdit.text()
        if exe_path:
            config["hv_forward"]["exe_path"] = exe_path

        # Disable run button, enable cancel
        self.runButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        self.progressBar.setValue(0)
        self.statusLabel.setText("Starting batch processing...")
        self.outputText.clear()

        # Create and start worker
        self.worker = BatchWorker(profiles_dir, output_dir, file_pattern, config)
        self.worker.progress.connect(self.updateProgress)
        self.worker.finished.connect(self.batchFinished)
        self.worker.error.connect(self.batchError)
        self.worker.start()

    def cancelBatch(self):
        """Cancel batch processing."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.statusLabel.setText("Batch processing cancelled")
            self.runButton.setEnabled(True)
            self.cancelButton.setEnabled(False)
            InfoBar.warning(
                "Cancelled",
                "Batch processing was cancelled by user",
                parent=self,
                position=InfoBarPosition.TOP
            )

    def updateProgress(self, message):
        """Update progress display."""
        self.outputText.append(message)
        self.statusLabel.setText(message)

    def batchFinished(self, summary):
        """Handle batch processing completion."""
        self.progressBar.setValue(100)
        self.statusLabel.setText("Batch processing completed!")

        # Display summary
        self.outputText.append("\n" + "="*60)
        self.outputText.append("BATCH PROCESSING COMPLETED!")
        self.outputText.append("="*60)
        self.outputText.append(f"Total profiles: {summary.get('total_profiles', 0)}")
        self.outputText.append(f"Successful: {summary.get('successful', 0)}")
        self.outputText.append(f"Failed: {summary.get('failed', 0)}")

        if summary.get('failed', 0) > 0:
            self.outputText.append("\nFailed profiles:")
            for result in summary.get('results', []):
                if result['status'] == 'failed':
                    self.outputText.append(f"  - {result['profile']}: {result.get('error', 'Unknown error')}")

        self.outputText.append(f"\nAll outputs saved in: {summary.get('output_directory', '')}")

        # Re-enable buttons
        self.runButton.setEnabled(True)
        self.cancelButton.setEnabled(False)

        # Show success message
        InfoBar.success(
            "Success",
            f"Processed {summary.get('successful', 0)}/{summary.get('total_profiles', 0)} profiles successfully!",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )

    def batchError(self, error_msg):
        """Handle batch processing error."""
        self.statusLabel.setText("Batch processing failed!")
        self.outputText.append(f"\nERROR: {error_msg}")

        # Re-enable buttons
        self.runButton.setEnabled(True)
        self.cancelButton.setEnabled(False)

        # Show error message
        InfoBar.error(
            "Error",
            f"Batch processing failed: {error_msg}",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )
