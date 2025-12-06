"""
Post-Processing Page

Generate publication-ready plots and analysis from HV curve and velocity model.
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, SpinBox, ComboBox, CheckBox,
    TextEdit, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    TransparentToolButton
)

# Import from parent package
from ...core.hv_postprocess import process


class PostprocessWorker(QThread):
    """Worker thread for post-processing."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, hv_csv_path, model_path, output_dir, plot_config):
        super().__init__()
        self.hv_csv_path = hv_csv_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.plot_config = plot_config

    def run(self):
        """Run post-processing."""
        try:
            outputs = process(
                self.hv_csv_path,
                self.model_path,
                self.output_dir,
                self.plot_config
            )
            self.finished.emit(outputs)
        except Exception as e:
            self.error.emit(str(e))


class PostprocessPage(QWidget):
    """Post-processing page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("postprocessPage")
        self.worker = None
        self.initUI()

    def initUI(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        titleLabel = TitleLabel("Post-Processing & Visualization", self)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Generate publication-ready plots and analysis from HV curve data.\n"
            "Creates HV curve plots, velocity profile plots, and summary statistics.",
            self
        )
        descLabel.setWordWrap(True)
        layout.addWidget(descLabel)

        # Input Configuration Card
        inputCard = self.createInputCard()
        layout.addWidget(inputCard)

        # Plot Settings Card
        plotCard = self.createPlotSettingsCard()
        layout.addWidget(plotCard)

        # Output Card
        outputCard = self.createOutputCard()
        layout.addWidget(outputCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        self.runButton = PrimaryPushButton(FIF.PENCIL_INK, "Generate Plots", self)
        self.runButton.clicked.connect(self.runPostprocess)
        buttonLayout.addWidget(self.runButton)

        layout.addLayout(buttonLayout)
        layout.addStretch()

    def createInputCard(self):
        """Create input configuration card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Input Files", card)
        cardLayout.addWidget(cardTitle)

        # HV CSV file
        hvLayout = QHBoxLayout()
        hvLabel = BodyLabel("HV CSV File:", card)
        hvLabel.setFixedWidth(150)
        hvLayout.addWidget(hvLabel)

        self.hvEdit = LineEdit(card)
        self.hvEdit.setPlaceholderText("Select HV curve CSV file...")
        hvLayout.addWidget(self.hvEdit)

        hvButton = TransparentToolButton(FIF.FOLDER, card)
        hvButton.clicked.connect(self.selectHvFile)
        hvLayout.addWidget(hvButton)

        cardLayout.addLayout(hvLayout)

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

        return card

    def createPlotSettingsCard(self):
        """Create plot settings card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Plot Settings", card)
        cardLayout.addWidget(cardTitle)

        # Scale settings
        scaleLayout = QHBoxLayout()

        # X-axis scale
        xscaleLayout = QVBoxLayout()
        xscaleLabel = BodyLabel("X-axis Scale:", card)
        xscaleLayout.addWidget(xscaleLabel)
        self.xscaleCombo = ComboBox(card)
        self.xscaleCombo.addItems(["log", "linear"])
        self.xscaleCombo.setCurrentIndex(0)
        xscaleLayout.addWidget(self.xscaleCombo)
        scaleLayout.addLayout(xscaleLayout)

        # Y-axis scale
        yscaleLayout = QVBoxLayout()
        yscaleLabel = BodyLabel("Y-axis Scale:", card)
        yscaleLayout.addWidget(yscaleLabel)
        self.yscaleCombo = ComboBox(card)
        self.yscaleCombo.addItems(["linear", "log"])
        self.yscaleCombo.setCurrentIndex(0)
        yscaleLayout.addWidget(self.yscaleCombo)
        scaleLayout.addLayout(yscaleLayout)

        # DPI
        dpiLayout = QVBoxLayout()
        dpiLabel = BodyLabel("DPI:", card)
        dpiLayout.addWidget(dpiLabel)
        self.dpiSpin = SpinBox(card)
        self.dpiSpin.setRange(72, 600)
        self.dpiSpin.setValue(150)
        dpiLayout.addWidget(self.dpiSpin)
        scaleLayout.addLayout(dpiLayout)

        cardLayout.addLayout(scaleLayout)

        # Smoothing checkbox
        self.smoothingCheck = CheckBox("Enable smoothing (Savitzky-Golay filter)", card)
        self.smoothingCheck.setChecked(True)
        cardLayout.addWidget(self.smoothingCheck)

        # Show bands checkbox
        self.bandsCheck = CheckBox("Show frequency bands", card)
        self.bandsCheck.setChecked(True)
        cardLayout.addWidget(self.bandsCheck)

        return card

    def createOutputCard(self):
        """Create output display card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Output", card)
        cardLayout.addWidget(cardTitle)

        # Output text
        self.outputText = TextEdit(card)
        self.outputText.setReadOnly(True)
        self.outputText.setFixedHeight(300)
        cardLayout.addWidget(self.outputText)

        return card

    def selectHvFile(self):
        """Select HV CSV file."""
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select HV CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file:
            self.hvEdit.setText(file)

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

    def runPostprocess(self):
        """Run post-processing."""
        hv_path = self.hvEdit.text()
        model_path = self.modelEdit.text()
        output_dir = self.outputEdit.text()

        # Validate inputs
        if not hv_path or not model_path or not output_dir:
            InfoBar.error(
                "Error",
                "Please select all required files and output directory",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        if not Path(hv_path).exists():
            InfoBar.error(
                "Error",
                "HV CSV file does not exist",
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

        # Build plot configuration
        plot_config = {
            "hv_plot": {
                "x_axis_scale": self.xscaleCombo.currentText(),
                "y_axis_scale": self.yscaleCombo.currentText(),
                "dpi": self.dpiSpin.value(),
                "smoothing": {
                    "enable": self.smoothingCheck.isChecked()
                },
                "show_bands": self.bandsCheck.isChecked()
            },
            "vs_plot": {
                "dpi": self.dpiSpin.value()
            }
        }

        # Disable run button
        self.runButton.setEnabled(False)
        self.outputText.clear()
        self.outputText.append("Starting post-processing...")
        self.outputText.append(f"HV file: {hv_path}")
        self.outputText.append(f"Model file: {model_path}")
        self.outputText.append(f"Output directory: {output_dir}")
        self.outputText.append("")

        # Create and start worker
        self.worker = PostprocessWorker(hv_path, model_path, output_dir, plot_config)
        self.worker.finished.connect(self.postprocessFinished)
        self.worker.error.connect(self.postprocessError)
        self.worker.start()

    def postprocessFinished(self, outputs):
        """Handle post-processing completion."""
        self.outputText.append("Post-processing completed successfully!")
        self.outputText.append("")
        self.outputText.append(f"Peak Frequency: {outputs.get('peak_frequency', 0):.3f} Hz")
        self.outputText.append(f"Peak Amplitude: {outputs.get('peak_amplitude', 0):.2f}")
        self.outputText.append("")
        self.outputText.append("Generated files:")

        # Display generated files
        for key, value in outputs.items():
            if isinstance(value, Path):
                self.outputText.append(f"  {value.name}")

        # Re-enable button
        self.runButton.setEnabled(True)

        # Show success message
        InfoBar.success(
            "Success",
            "Plots generated successfully!",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=3000
        )

    def postprocessError(self, error_msg):
        """Handle post-processing error."""
        self.outputText.append(f"\nERROR: {error_msg}")

        # Re-enable button
        self.runButton.setEnabled(True)

        # Show error message
        InfoBar.error(
            "Error",
            f"Post-processing failed: {error_msg}",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )
