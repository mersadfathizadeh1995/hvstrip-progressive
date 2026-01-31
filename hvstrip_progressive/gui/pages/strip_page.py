"""
Layer Stripping Page

Progressive layer removal functionality.
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, TextEdit, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    TransparentToolButton
, ScrollArea)

# Import from parent package
from ...core.stripper import write_peel_sequence


class StripWorker(QThread):
    """Worker thread for layer stripping."""

    finished = Signal(Path)
    error = Signal(str)

    def __init__(self, model_path, output_dir):
        super().__init__()
        self.model_path = model_path
        self.output_dir = output_dir

    def run(self):
        try:
            strip_dir = write_peel_sequence(self.model_path, self.output_dir)
            self.finished.emit(strip_dir)
        except Exception as e:
            self.error.emit(str(e))


class StripPage(QWidget):
    """Layer stripping page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stripPage")
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
        titleLabel = TitleLabel("Layer Stripping", self)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Progressively remove the deepest finite layer and promote it to half-space.\n"
            "Creates a sequence of peeled models for analysis.",
            self
        )
        descLabel.setWordWrap(True)
        layout.addWidget(descLabel)

        # Input Card
        inputCard = self.createInputCard()
        layout.addWidget(inputCard)

        # Output Card
        outputCard = self.createOutputCard()
        layout.addWidget(outputCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        self.runButton = PrimaryPushButton(FIF.CUT, "Strip Layers", self)
        self.runButton.clicked.connect(self.runStripping)
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

    def createOutputCard(self):
        """Create output display card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        cardTitle = SubtitleLabel("Output", card)
        cardLayout.addWidget(cardTitle)

        self.outputText = TextEdit(card)
        self.outputText.setReadOnly(True)
        self.outputText.setFixedHeight(300)
        cardLayout.addWidget(self.outputText)

        return card

    def selectModelFile(self):
        """Select model file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Velocity Model File", "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file:
            self.modelEdit.setText(file)

    def selectOutputDir(self):
        """Select output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if directory:
            self.outputEdit.setText(directory)

    def runStripping(self):
        """Run layer stripping."""
        model_path = self.modelEdit.text()
        output_dir = self.outputEdit.text()

        if not model_path or not output_dir:
            InfoBar.error("Error", "Please select model file and output directory",
                         parent=self, position=InfoBarPosition.TOP)
            return

        if not Path(model_path).exists():
            InfoBar.error("Error", "Model file does not exist",
                         parent=self, position=InfoBarPosition.TOP)
            return

        self.runButton.setEnabled(False)
        self.outputText.clear()
        self.outputText.append("Starting layer stripping...")

        self.worker = StripWorker(model_path, output_dir)
        self.worker.finished.connect(self.strippingFinished)
        self.worker.error.connect(self.strippingError)
        self.worker.start()

    def strippingFinished(self, strip_dir):
        """Handle stripping completion."""
        step_folders = list(strip_dir.glob("Step*"))

        self.outputText.append("\n✅ Layer stripping completed successfully!")
        self.outputText.append(f"📂 Strip directory: {strip_dir}")
        self.outputText.append(f"📊 Generated {len(step_folders)} stripped models")
        self.outputText.append("\nGenerated steps:")

        for folder in sorted(step_folders):
            self.outputText.append(f"  • {folder.name}")

        self.runButton.setEnabled(True)
        InfoBar.success("Success", f"Generated {len(step_folders)} stripped models",
                       parent=self, position=InfoBarPosition.TOP, duration=3000)

    def strippingError(self, error_msg):
        """Handle stripping error."""
        self.outputText.append(f"\n❌ ERROR: {error_msg}")
        self.runButton.setEnabled(True)
        InfoBar.error("Error", f"Layer stripping failed: {error_msg}",
                     parent=self, position=InfoBarPosition.TOP, duration=5000)
