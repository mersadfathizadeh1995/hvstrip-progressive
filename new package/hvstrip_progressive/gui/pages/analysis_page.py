"""
Advanced Analysis Page

Perform advanced statistical analysis on progressive stripping results.
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QSizePolicy
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, CheckBox, TextEdit, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    TransparentToolButton, ScrollArea
)

# Import from parent package
from ...core.advanced_analysis import analyze_strip_directory
from ..widgets.plot_widget import MatplotlibWidget


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
        self.latest_results = None
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
        titleLabel = TitleLabel("Advanced Analysis", contentWidget)
        layout.addWidget(titleLabel)

        # Description
        descLabel = BodyLabel(
            "Perform advanced statistical analysis on progressive layer stripping results.\n"
            "Includes controlling interface detection, layer contribution analysis, and statistics.",
            contentWidget
        )
        descLabel.setWordWrap(True)
        layout.addWidget(descLabel)

        # Input Configuration Card
        inputCard = self.createInputCard(contentWidget)
        layout.addWidget(inputCard)

        # Analysis Options Card
        optionsCard = self.createOptionsCard(contentWidget)
        layout.addWidget(optionsCard)

        # Visualization Card
        vizCard = self.createVisualizationCard(contentWidget)
        layout.addWidget(vizCard)

        # Output Card
        outputCard = self.createOutputCard(contentWidget)
        layout.addWidget(outputCard)

        # Control Buttons
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()

        self.runButton = PrimaryPushButton(FIF.BOOK_SHELF, "Run Analysis", contentWidget)
        self.runButton.clicked.connect(self.runAnalysis)
        self.runButton.setFixedHeight(36)
        buttonLayout.addWidget(self.runButton)

        layout.addLayout(buttonLayout)
        layout.addStretch()

        # Set scroll area content
        scrollArea.setWidget(contentWidget)
        mainLayout.addWidget(scrollArea)

    def createInputCard(self, parent):
        """Create input configuration card."""
        card = CardWidget(parent)
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
        self.stripEdit.setMinimumWidth(400)
        stripLayout.addWidget(self.stripEdit, 1)

        stripButton = TransparentToolButton(FIF.FOLDER, card)
        stripButton.clicked.connect(self.selectStripDir)
        stripButton.setFixedSize(36, 36)
        stripLayout.addWidget(stripButton)

        cardLayout.addLayout(stripLayout)

        # Output directory
        outputLayout = QHBoxLayout()
        outputLabel = BodyLabel("Output Directory:", card)
        outputLabel.setFixedWidth(150)
        outputLayout.addWidget(outputLabel)

        self.outputEdit = LineEdit(card)
        self.outputEdit.setPlaceholderText("Optional: output directory for analysis results")
        self.outputEdit.setMinimumWidth(400)
        outputLayout.addWidget(self.outputEdit, 1)

        outputButton = TransparentToolButton(FIF.FOLDER, card)
        outputButton.clicked.connect(self.selectOutputDir)
        outputButton.setFixedSize(36, 36)
        outputLayout.addWidget(outputButton)

        cardLayout.addLayout(outputLayout)

        return card

    def createOptionsCard(self, parent):
        """Create analysis options card."""
        card = CardWidget(parent)
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

    def createVisualizationCard(self, parent):
        """Create visualization card with plots."""
        card = CardWidget(parent)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        titleLayout = QHBoxLayout()
        cardTitle = SubtitleLabel("Visualization", card)
        titleLayout.addWidget(cardTitle)
        titleLayout.addStretch()

        self.plotButton = PushButton(FIF.PIE_SINGLE, "Generate Plots", card)
        self.plotButton.clicked.connect(self.generatePlots)
        self.plotButton.setEnabled(False)
        self.plotButton.setFixedHeight(32)
        titleLayout.addWidget(self.plotButton)

        self.exportPlotButton = PushButton(FIF.SAVE, "Export Plot", card)
        self.exportPlotButton.clicked.connect(self.exportPlot)
        self.exportPlotButton.setEnabled(False)
        self.exportPlotButton.setFixedHeight(32)
        titleLayout.addWidget(self.exportPlotButton)

        cardLayout.addLayout(titleLayout)

        # Matplotlib widget
        self.plotWidget = MatplotlibWidget(card, figsize=(10, 6))
        cardLayout.addWidget(self.plotWidget)

        return card

    def createOutputCard(self, parent):
        """Create output display card."""
        card = CardWidget(parent)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        cardTitle = SubtitleLabel("Analysis Results", card)
        cardLayout.addWidget(cardTitle)

        # Output text
        self.outputText = TextEdit(card)
        self.outputText.setReadOnly(True)
        self.outputText.setMinimumHeight(300)
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
        self.plotButton.setEnabled(False)
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
        self.latest_results = results

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
            peak_range = stats.get('peak_freq_range', (0, 0))
            self.outputText.append(f"  Peak frequency range: {peak_range[0]:.2f} - {peak_range[1]:.2f} Hz")
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

        # Re-enable buttons
        self.runButton.setEnabled(True)
        self.plotButton.setEnabled(True)

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

    def generatePlots(self):
        """Generate visualization plots from analysis results."""
        if not self.latest_results:
            InfoBar.warning(
                "No Data",
                "Run analysis first to generate plots",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        try:
            import numpy as np
            fig = self.plotWidget.get_figure()
            fig.clear()

            stats = self.latest_results.get('statistics', {})
            controlling = self.latest_results.get('controlling_interfaces', [])

            # Create 2x2 subplot
            axes = fig.subplots(2, 2)

            # Plot 1: Peak frequency evolution
            if 'peak_freqs' in stats:
                peak_freqs = stats['peak_freqs']
                steps = range(1, len(peak_freqs) + 1)
                axes[0, 0].plot(steps, peak_freqs, 'o-', linewidth=2, markersize=8)
                axes[0, 0].set_xlabel('Step Number')
                axes[0, 0].set_ylabel('Peak Frequency (Hz)')
                axes[0, 0].set_title('Peak Frequency Evolution')
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Peak amplitude evolution
            if 'peak_amps' in stats:
                peak_amps = stats['peak_amps']
                steps = range(1, len(peak_amps) + 1)
                axes[0, 1].plot(steps, peak_amps, 's-', color='orange', linewidth=2, markersize=8)
                axes[0, 1].set_xlabel('Step Number')
                axes[0, 1].set_ylabel('Peak Amplitude')
                axes[0, 1].set_title('Peak Amplitude Evolution')
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Frequency changes
            if controlling:
                steps = [c['step'] for c in controlling]
                freq_changes = [c['freq_change'] for c in controlling]
                colors = ['red' if c.get('is_controlling', False) else 'blue' for c in controlling]
                axes[1, 0].bar(steps, freq_changes, color=colors, alpha=0.7)
                axes[1, 0].set_xlabel('Step Transition')
                axes[1, 0].set_ylabel('Frequency Change (Hz)')
                axes[1, 0].set_title('Step-wise Frequency Changes')
                axes[1, 0].grid(True, alpha=0.3, axis='y')

            # Plot 4: Significance scores
            if controlling:
                steps = [c['step'] for c in controlling]
                scores = [c['significance_score'] for c in controlling]
                axes[1, 1].bar(steps, scores, color='green', alpha=0.7)
                axes[1, 1].set_xlabel('Step Transition')
                axes[1, 1].set_ylabel('Significance Score')
                axes[1, 1].set_title('Interface Significance')
                axes[1, 1].grid(True, alpha=0.3, axis='y')

            fig.tight_layout()
            self.plotWidget.refresh()
            self.exportPlotButton.setEnabled(True)

            InfoBar.success(
                "Plots Generated",
                "Visualization plots created successfully",
                parent=self,
                position=InfoBarPosition.TOP
            )

        except Exception as e:
            InfoBar.error(
                "Plot Error",
                f"Failed to generate plots: {str(e)}",
                parent=self,
                position=InfoBarPosition.TOP
            )

    def exportPlot(self):
        """Export the current plot to file."""
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "analysis_plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )

        if fileName:
            try:
                self.plotWidget.get_figure().savefig(fileName, dpi=300, bbox_inches='tight')
                InfoBar.success(
                    "Exported",
                    f"Plot saved to {Path(fileName).name}",
                    parent=self,
                    position=InfoBarPosition.TOP
                )
            except Exception as e:
                InfoBar.error(
                    "Export Error",
                    f"Failed to save plot: {str(e)}",
                    parent=self,
                    position=InfoBarPosition.TOP
                )
