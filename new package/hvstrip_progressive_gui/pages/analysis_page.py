"""
Advanced Analysis Page

Perform advanced statistical analysis on progressive stripping results.
"""

import sys
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, CheckBox, TextEdit, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    TransparentToolButton
)

# Add parent directory to path to import hvstrip_progressive
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hvstrip_progressive.core.advanced_analysis import analyze_strip_directory


class AnalysisWorker(QThread):
    """Worker thread for advanced analysis."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, strip_dir, output_dir, options):
        super().__init__()
        self.strip_dir = strip_dir
        self.output_dir = output_dir
        self.options = options

    def run(self):
        """Run advanced analysis."""
        try:
            # Run analysis
            results = analyze_strip_directory(
                Path(self.strip_dir),
                Path(self.output_dir) if self.output_dir else None
            )

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class AnalysisPage(QWidget):
    """Advanced analysis page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisPage")
        self.worker = None
        self.initUI()

    def initUI(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        titleLabel = TitleLabel("Advanced Analysis", self)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Perform advanced statistical analysis on progressive layer stripping results.\n"
            "Includes controlling interface detection, layer contribution analysis, and statistics.",
            self
        )
        descLabel.setWordWrap(True)
        layout.addWidget(descLabel)

        # Input Configuration Card
        inputCard = self.createInputCard()
        layout.addWidget(inputCard)

        # Analysis Options Card
        optionsCard = self.createOptionsCard()
        layout.addWidget(optionsCard)

        # Output Card
        outputCard = self.createOutputCard()
        layout.addWidget(outputCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        self.runButton = PrimaryPushButton(FIF.BOOK_SHELF, "Run Analysis", self)
        self.runButton.clicked.connect(self.runAnalysis)
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
        cardTitle = SubtitleLabel("Input Configuration", card)
        cardLayout.addWidget(cardTitle)

        # Strip directory
        stripLayout = QHBoxLayout()
        stripLabel = BodyLabel("Strip Directory:", card)
        stripLabel.setFixedWidth(150)
        stripLayout.addWidget(stripLabel)

        self.stripEdit = LineEdit(card)
        self.stripEdit.setPlaceholderText("Select strip directory (contains StepX folders)...")
        stripLayout.addWidget(self.stripEdit)

        stripButton = TransparentToolButton(FIF.FOLDER, card)
        stripButton.clicked.connect(self.selectStripDir)
        stripLayout.addWidget(stripButton)

        cardLayout.addLayout(stripLayout)

        # Output directory
        outputLayout = QHBoxLayout()
        outputLabel = BodyLabel("Output Directory:", card)
        outputLabel.setFixedWidth(150)
        outputLayout.addWidget(outputLabel)

        self.outputEdit = LineEdit(card)
        self.outputEdit.setPlaceholderText("Optional: output directory for analysis results")
        outputLayout.addWidget(self.outputEdit)

        outputButton = TransparentToolButton(FIF.FOLDER, card)
        outputButton.clicked.connect(self.selectOutputDir)
        outputLayout.addWidget(outputButton)

        cardLayout.addLayout(outputLayout)

        return card

    def createOptionsCard(self):
        """Create analysis options card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Analysis Options", card)
        cardLayout.addWidget(cardTitle)

        # Statistics checkbox
        self.statsCheck = CheckBox("Compute statistical summary", card)
        self.statsCheck.setChecked(True)
        cardLayout.addWidget(self.statsCheck)

        # Controlling interfaces checkbox
        self.interfacesCheck = CheckBox("Detect controlling interfaces", card)
        self.interfacesCheck.setChecked(True)
        cardLayout.addWidget(self.interfacesCheck)

        # Layer contributions checkbox
        self.contributionsCheck = CheckBox("Analyze layer contributions", card)
        self.contributionsCheck.setChecked(True)
        cardLayout.addWidget(self.contributionsCheck)

        # Export data checkbox
        self.exportCheck = CheckBox("Export analysis data to CSV", card)
        self.exportCheck.setChecked(True)
        cardLayout.addWidget(self.exportCheck)

        # Info label
        infoLabel = BodyLabel(
            "Analysis will include:\n"
            "  • Peak frequency and amplitude statistics\n"
            "  • Step-wise change analysis\n"
            "  • Interface significance scoring\n"
            "  • Layer-by-layer contribution metrics\n"
            "  • Spectral energy change quantification",
            card
        )
        infoLabel.setWordWrap(True)
        cardLayout.addWidget(infoLabel)

        return card

    def createOutputCard(self):
        """Create output display card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Analysis Results", card)
        cardLayout.addWidget(cardTitle)

        # Output text
        self.outputText = TextEdit(card)
        self.outputText.setReadOnly(True)
        self.outputText.setFixedHeight(350)
        cardLayout.addWidget(self.outputText)

        return card

    def selectStripDir(self):
        """Select strip directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Strip Directory"
        )
        if directory:
            self.stripEdit.setText(directory)

    def selectOutputDir(self):
        """Select output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        if directory:
            self.outputEdit.setText(directory)

    def runAnalysis(self):
        """Run advanced analysis."""
        strip_dir = self.stripEdit.text()
        output_dir = self.outputEdit.text()

        # Validate inputs
        if not strip_dir:
            InfoBar.error(
                "Error",
                "Please select a strip directory",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        if not Path(strip_dir).exists():
            InfoBar.error(
                "Error",
                "Strip directory does not exist",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        # Build options
        options = {
            'statistics': self.statsCheck.isChecked(),
            'controlling_interfaces': self.interfacesCheck.isChecked(),
            'layer_contributions': self.contributionsCheck.isChecked(),
            'export_data': self.exportCheck.isChecked()
        }

        # Disable run button
        self.runButton.setEnabled(False)
        self.outputText.clear()
        self.outputText.append("Starting advanced analysis...")
        self.outputText.append(f"Strip directory: {strip_dir}")
        if output_dir:
            self.outputText.append(f"Output directory: {output_dir}")
        self.outputText.append("")

        # Create and start worker
        self.worker = AnalysisWorker(strip_dir, output_dir, options)
        self.worker.progress.connect(self.updateProgress)
        self.worker.finished.connect(self.analysisFinished)
        self.worker.error.connect(self.analysisError)
        self.worker.start()

    def updateProgress(self, message):
        """Update progress display."""
        self.outputText.append(message)

    def analysisFinished(self, results):
        """Handle analysis completion."""
        self.outputText.append("")
        self.outputText.append("="*60)
        self.outputText.append("ADVANCED ANALYSIS COMPLETED!")
        self.outputText.append("="*60)
        self.outputText.append("")

        # Display statistics
        stats = results.get('statistics', {})
        if stats:
            self.outputText.append("STATISTICAL SUMMARY:")
            self.outputText.append(f"  Number of steps: {stats.get('n_steps', 0)}")
            self.outputText.append(f"  Peak frequency range: {stats.get('peak_freq_range', (0, 0))[0]:.2f} - {stats.get('peak_freq_range', (0, 0))[1]:.2f} Hz")
            self.outputText.append(f"  Peak frequency mean: {stats.get('peak_freq_mean', 0):.2f} Hz")
            self.outputText.append(f"  Peak frequency std: {stats.get('peak_freq_std', 0):.2f} Hz")
            self.outputText.append(f"  Total frequency change: {stats.get('peak_freq_change_total', 0):.2f} Hz")
            self.outputText.append("")

        # Display controlling interfaces
        controlling = results.get('controlling_interfaces', [])
        if controlling:
            significant = [c for c in controlling if c.get('is_controlling', False)]
            self.outputText.append(f"CONTROLLING INTERFACES: {len(significant)} detected")
            for ci in significant[:3]:  # Show top 3
                self.outputText.append(f"  Step {ci['step']} -> {ci['step']+1}:")
                self.outputText.append(f"    Significance score: {ci['significance_score']:.2f}")
                self.outputText.append(f"    Frequency change: {ci['freq_change']:.3f} Hz ({ci['freq_change_pct']:.1f}%)")
            self.outputText.append("")

        # Display layer contributions summary
        contributions = results.get('layer_contributions')
        if contributions is not None and not contributions.empty:
            self.outputText.append(f"LAYER CONTRIBUTIONS: {len(contributions)} transitions analyzed")
            self.outputText.append(f"  Max frequency shift: {contributions['freq_shift_hz'].abs().max():.3f} Hz")
            self.outputText.append(f"  Max amplitude change: {contributions['amp_change'].abs().max():.3f}")
            self.outputText.append("")

        # Display output files
        output_dir = self.outputEdit.text()
        if output_dir:
            self.outputText.append("Generated files:")
            output_path = Path(output_dir)
            if output_path.exists():
                for file in output_path.glob("*"):
                    if file.is_file():
                        self.outputText.append(f"  {file.name}")

        # Re-enable button
        self.runButton.setEnabled(True)

        # Show success message
        InfoBar.success(
            "Success",
            f"Analysis completed! Analyzed {stats.get('n_steps', 0)} stripping steps.",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )

    def analysisError(self, error_msg):
        """Handle analysis error."""
        self.outputText.append("")
        self.outputText.append(f"ERROR: {error_msg}")

        # Re-enable button
        self.runButton.setEnabled(True)

        # Show error message
        InfoBar.error(
            "Error",
            f"Analysis failed: {error_msg}",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )
