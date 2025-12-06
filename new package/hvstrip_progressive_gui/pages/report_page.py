"""
Report Generation Page

Generate comprehensive analysis reports from progressive stripping results.
"""

import sys
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, TextEdit, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    TransparentToolButton
)

# Add parent directory to path to import hvstrip_progressive
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hvstrip_progressive.core.report_generator import ProgressiveStrippingReporter


class ReportWorker(QThread):
    """Worker thread for report generation."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, strip_dir, output_dir):
        super().__init__()
        self.strip_dir = strip_dir
        self.output_dir = output_dir

    def run(self):
        """Run report generation."""
        try:
            # Create reporter
            reporter = ProgressiveStrippingReporter(self.strip_dir, self.output_dir)

            # Generate comprehensive report
            report_files = reporter.generate_comprehensive_report()

            self.finished.emit(report_files)
        except Exception as e:
            self.error.emit(str(e))


class ReportPage(QWidget):
    """Report generation page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("reportPage")
        self.worker = None
        self.initUI()

    def initUI(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        titleLabel = TitleLabel("Report Generation", self)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Generate comprehensive analysis reports from progressive layer stripping results.\n"
            "Creates publication-ready figures, statistical summaries, and detailed analysis.",
            self
        )
        descLabel.setWordWrap(True)
        layout.addWidget(descLabel)

        # Input Configuration Card
        inputCard = self.createInputCard()
        layout.addWidget(inputCard)

        # Output Card
        outputCard = self.createOutputCard()
        layout.addWidget(outputCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        self.runButton = PrimaryPushButton(FIF.DOCUMENT, "Generate Report", self)
        self.runButton.clicked.connect(self.runReportGeneration)
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
        self.outputEdit.setPlaceholderText("Optional: output directory (default: strip_dir/reports)")
        outputLayout.addWidget(self.outputEdit)

        outputButton = TransparentToolButton(FIF.FOLDER, card)
        outputButton.clicked.connect(self.selectOutputDir)
        outputLayout.addWidget(outputButton)

        cardLayout.addLayout(outputLayout)

        # Info label
        infoLabel = BodyLabel(
            "The report will include:\n"
            "  • HV curves overlay plot\n"
            "  • Peak evolution analysis\n"
            "  • Interface analysis\n"
            "  • Waterfall plot\n"
            "  • Publication-ready figure\n"
            "  • Statistical summary CSV\n"
            "  • Text report",
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
        cardTitle = SubtitleLabel("Output", card)
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

    def runReportGeneration(self):
        """Run report generation."""
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

        # Use default output directory if not specified
        if not output_dir:
            output_dir = str(Path(strip_dir).parent / 'reports')

        # Disable run button
        self.runButton.setEnabled(False)
        self.outputText.clear()
        self.outputText.append("Starting report generation...")
        self.outputText.append(f"Strip directory: {strip_dir}")
        self.outputText.append(f"Output directory: {output_dir}")
        self.outputText.append("")

        # Create and start worker
        self.worker = ReportWorker(strip_dir, output_dir)
        self.worker.progress.connect(self.updateProgress)
        self.worker.finished.connect(self.reportFinished)
        self.worker.error.connect(self.reportError)
        self.worker.start()

    def updateProgress(self, message):
        """Update progress display."""
        self.outputText.append(message)

    def reportFinished(self, report_files):
        """Handle report generation completion."""
        self.outputText.append("")
        self.outputText.append("="*60)
        self.outputText.append("REPORT GENERATION COMPLETED SUCCESSFULLY!")
        self.outputText.append("="*60)
        self.outputText.append("")
        self.outputText.append(f"Generated {len(report_files)} report files:")
        self.outputText.append("")

        # Display generated files
        for key, file_path in report_files.items():
            if isinstance(file_path, Path):
                self.outputText.append(f"  {file_path.name}")

        # Re-enable button
        self.runButton.setEnabled(True)

        # Show success message
        InfoBar.success(
            "Success",
            f"Report generated successfully! Created {len(report_files)} files.",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )

    def reportError(self, error_msg):
        """Handle report generation error."""
        self.outputText.append("")
        self.outputText.append(f"ERROR: {error_msg}")

        # Re-enable button
        self.runButton.setEnabled(True)

        # Show error message
        InfoBar.error(
            "Error",
            f"Report generation failed: {error_msg}",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )
