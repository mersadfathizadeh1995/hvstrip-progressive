"""
HV Forward Modeling Page

Compute HV curve for a single velocity model using HVf executable.
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, SpinBox, DoubleSpinBox,
    TextEdit, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    TransparentToolButton
, ScrollArea)

# Import from parent package
from ...core.hv_forward import compute_hv_curve


class ForwardWorker(QThread):
    """Worker thread for HV forward computation."""

    finished = Signal(list, list)
    error = Signal(str)

    def __init__(self, model_path, config):
        super().__init__()
        self.model_path = model_path
        self.config = config

    def run(self):
        """Run HV forward computation."""
        try:
            freqs, amps = compute_hv_curve(self.model_path, self.config)
            self.finished.emit(freqs, amps)
        except Exception as e:
            self.error.emit(str(e))


class ForwardPage(QWidget):
    """HV Forward modeling page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("forwardPage")
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
        titleLabel = TitleLabel("HV Forward Modeling", self)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Compute theoretical HVSR curve from a velocity model using HVf executable.\n"
            "This tool runs forward modeling to predict the HVSR response.",
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

        # Output Card
        outputCard = self.createOutputCard()
        layout.addWidget(outputCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        self.runButton = PrimaryPushButton(FIF.PLAY, "Compute HV Curve", self)
        self.runButton.clicked.connect(self.runForward)
        buttonLayout.addWidget(self.runButton)

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

        # Model file
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

        # HVf executable path
        exeLayout = QHBoxLayout()
        exeLabel = BodyLabel("HVf Executable:", card)
        exeLabel.setFixedWidth(150)
        exeLayout.addWidget(exeLabel)

        self.exeEdit = LineEdit(card)
        self.exeEdit.setPlaceholderText("Path to HVf executable (optional, auto-detected)")
        exeLayout.addWidget(self.exeEdit)

        exeButton = TransparentToolButton(FIF.FOLDER, card)
        exeButton.clicked.connect(self.selectExeFile)
        exeLayout.addWidget(exeButton)

        cardLayout.addLayout(exeLayout)

        # Output CSV file (optional)
        outputLayout = QHBoxLayout()
        outputLabel = BodyLabel("Output CSV:", card)
        outputLabel.setFixedWidth(150)
        outputLayout.addWidget(outputLabel)

        self.outputEdit = LineEdit(card)
        self.outputEdit.setPlaceholderText("Optional: save results to CSV file...")
        outputLayout.addWidget(self.outputEdit)

        outputButton = TransparentToolButton(FIF.SAVE, card)
        outputButton.clicked.connect(self.selectOutputFile)
        outputLayout.addWidget(outputButton)

        cardLayout.addLayout(outputLayout)

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

    def createOutputCard(self):
        """Create output display card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Results", card)
        cardLayout.addWidget(cardTitle)

        # Output text
        self.outputText = TextEdit(card)
        self.outputText.setReadOnly(True)
        self.outputText.setFixedHeight(300)
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

    def selectOutputFile(self):
        """Select output CSV file."""
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Save HV Curve CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file:
            self.outputEdit.setText(file)

    def runForward(self):
        """Run HV forward modeling."""
        model_path = self.modelEdit.text()

        if not model_path:
            InfoBar.error(
                "Error",
                "Please select a model file",
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
            "fmin": self.fminSpin.value(),
            "fmax": self.fmaxSpin.value(),
            "nf": self.nfSpin.value()
        }

        # Add exe path if specified
        exe_path = self.exeEdit.text()
        if exe_path:
            config["exe_path"] = exe_path

        # Disable run button
        self.runButton.setEnabled(False)
        self.outputText.clear()
        self.outputText.append("Starting HV forward computation...")
        self.outputText.append(f"Model: {model_path}")
        self.outputText.append(f"Frequency range: {config['fmin']:.2f} - {config['fmax']:.1f} Hz")
        self.outputText.append(f"Number of points: {config['nf']}")
        self.outputText.append("")

        # Create and start worker
        self.worker = ForwardWorker(model_path, config)
        self.worker.finished.connect(self.forwardFinished)
        self.worker.error.connect(self.forwardError)
        self.worker.start()

    def forwardFinished(self, freqs, amps):
        """Handle forward computation completion."""
        import numpy as np

        # Find peak
        peak_idx = np.argmax(amps)
        peak_freq = freqs[peak_idx]
        peak_amp = amps[peak_idx]

        self.outputText.append("HV computation completed successfully!")
        self.outputText.append("")
        self.outputText.append(f"Peak Frequency: {peak_freq:.3f} Hz")
        self.outputText.append(f"Peak Amplitude: {peak_amp:.2f}")
        self.outputText.append(f"Total points: {len(freqs)}")

        # Save to CSV if output path specified
        output_path = self.outputEdit.text()
        if output_path:
            try:
                import csv
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frequency_Hz', 'HV_Amplitude'])
                    for freq, amp in zip(freqs, amps):
                        writer.writerow([f'{freq:.6f}', f'{amp:.6f}'])

                self.outputText.append(f"\nResults saved to: {output_path}")
            except Exception as e:
                self.outputText.append(f"\nWarning: Could not save CSV: {e}")

        # Re-enable button
        self.runButton.setEnabled(True)

        # Show success message
        InfoBar.success(
            "Success",
            f"HV curve computed! Peak at {peak_freq:.3f} Hz",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=3000
        )

    def forwardError(self, error_msg):
        """Handle forward computation error."""
        self.outputText.append(f"\nERROR: {error_msg}")

        # Re-enable button
        self.runButton.setEnabled(True)

        # Show error message
        InfoBar.error(
            "Error",
            f"HV computation failed: {error_msg}",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )
