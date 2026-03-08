"""
Research Workflow Page

Complete research workflow for Two Resonance Separation analysis.
Processes multiple profiles and generates statistical summaries and figures.
"""

from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    PushButton, LineEdit, SpinBox, DoubleSpinBox, CheckBox,
    TextEdit, ProgressBar, FluentIcon as FIF,
    InfoBar, InfoBarPosition, PrimaryPushButton,
    TransparentToolButton, ScrollArea, ProgressRing
)

import sys
import os

# Add parent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ...core.research_workflow import run_batch_analysis, ProfileResult, BatchStatistics
from ...visualization.statistics_plots import generate_all_statistics_plots
from ...visualization.special_plots import generate_resonance_separation_figure


class ResearchWorker(QThread):
    """Worker thread for research workflow processing."""

    progress = Signal(int, int, str)  # current, total, message
    log = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, profiles_dir, output_dir, config, n_examples):
        super().__init__()
        self.profiles_dir = profiles_dir
        self.output_dir = output_dir
        self.config = config
        self.n_examples = n_examples

    def run(self):
        """Run research workflow."""
        try:
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.log.emit("=" * 60)
            self.log.emit("Starting Two Resonance Research Workflow")
            self.log.emit("=" * 60)

            # Step 1: Run batch analysis
            self.log.emit("\n[1/3] Running batch analysis...")
            
            def progress_callback(current, total, name):
                self.progress.emit(current, total, name)
                self.log.emit(f"  Processing {current}/{total}: {name}")
            
            batch_result = run_batch_analysis(
                profiles_dir=self.profiles_dir,
                output_dir=str(output_path / "individual_results"),
                workflow_config=self.config,
                progress_callback=progress_callback
            )

            if not batch_result['success']:
                self.error.emit("Batch analysis failed")
                return

            # Step 2: Generate statistical plots
            self.log.emit("\n[2/3] Generating statistical plots...")
            stats_output = output_path / "statistics"
            stats_output.mkdir(exist_ok=True)
            
            plot_paths = generate_all_statistics_plots(
                results_csv=batch_result['results_file'],
                stats_json=batch_result['stats_file'],
                output_dir=str(stats_output)
            )
            self.log.emit(f"  Generated {len(plot_paths)} plots")

            # Step 3: Generate example figures
            self.log.emit(f"\n[3/3] Generating {self.n_examples} example figures...")
            examples_output = output_path / "example_figures"
            examples_output.mkdir(exist_ok=True)
            
            successful_results = [r for r in batch_result['results'] if r.success]
            successful_results.sort(key=lambda x: x.freq_ratio, reverse=True)
            
            example_paths = []
            for i, result in enumerate(successful_results[:self.n_examples]):
                strip_dir = Path(batch_result['output_dir']) / result.profile_name / "strip"
                if strip_dir.exists():
                    output_fig = examples_output / f"example_{i+1}_{result.profile_name}.png"
                    try:
                        generate_resonance_separation_figure(str(strip_dir), str(output_fig))
                        example_paths.append(str(output_fig))
                        self.log.emit(f"  Example {i+1}: {result.profile_name} (ratio={result.freq_ratio:.2f})")
                    except Exception as e:
                        self.log.emit(f"  Failed for {result.profile_name}: {e}")

            # Copy main files to output root
            import shutil
            shutil.copy(batch_result['results_file'], output_path / "batch_results.csv")
            shutil.copy(batch_result['stats_file'], output_path / "batch_statistics.json")

            self.log.emit("\n" + "=" * 60)
            self.log.emit("RESEARCH WORKFLOW COMPLETE!")
            self.log.emit("=" * 60)

            self.finished.emit({
                'success': True,
                'batch_result': batch_result,
                'plot_paths': plot_paths,
                'example_paths': example_paths,
                'output_dir': str(output_path)
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class ResearchPage(QWidget):
    """Research workflow page for Two Resonance analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("researchPage")
        self.worker = None
        self.initUI()

    def initUI(self):
        """Initialize user interface."""
        mainLayout = QVBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0)

        scrollArea = ScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        contentWidget = QWidget()
        layout = QVBoxLayout(contentWidget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        titleLabel = TitleLabel("Two Resonance Research Workflow", self)
        layout.addWidget(titleLabel)

        descLabel = BodyLabel(
            "Complete research workflow for analyzing multiple soil profiles.\n"
            "Generates statistical summaries and publication-ready figures for your paper.",
            self
        )
        descLabel.setWordWrap(True)
        layout.addWidget(descLabel)

        # Input Configuration Card
        inputCard = self.createInputCard()
        layout.addWidget(inputCard)

        # Analysis Parameters Card
        paramsCard = self.createParametersCard()
        layout.addWidget(paramsCard)

        # Progress Card
        progressCard = self.createProgressCard()
        layout.addWidget(progressCard)

        # Results Card
        resultsCard = self.createResultsCard()
        layout.addWidget(resultsCard)

        # Run Button
        self.runButton = PrimaryPushButton(FIF.PLAY, "Run Research Workflow", self)
        self.runButton.setFixedHeight(40)
        self.runButton.clicked.connect(self.runWorkflow)
        layout.addWidget(self.runButton)

        layout.addStretch()
        scrollArea.setWidget(contentWidget)
        mainLayout.addWidget(scrollArea)

    def createInputCard(self):
        """Create input configuration card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        # Title
        titleLabel = SubtitleLabel("Input Configuration", card)
        cardLayout.addWidget(titleLabel)

        # Profiles Directory
        profilesLayout = QHBoxLayout()
        profilesLabel = BodyLabel("Profiles Directory:", card)
        profilesLabel.setFixedWidth(150)
        self.profilesDirEdit = LineEdit(card)
        self.profilesDirEdit.setPlaceholderText("Select directory containing .txt profile files")
        profilesBrowse = TransparentToolButton(FIF.FOLDER, card)
        profilesBrowse.clicked.connect(self.browseProfilesDir)
        profilesLayout.addWidget(profilesLabel)
        profilesLayout.addWidget(self.profilesDirEdit)
        profilesLayout.addWidget(profilesBrowse)
        cardLayout.addLayout(profilesLayout)

        # Output Directory
        outputLayout = QHBoxLayout()
        outputLabel = BodyLabel("Output Directory:", card)
        outputLabel.setFixedWidth(150)
        self.outputDirEdit = LineEdit(card)
        self.outputDirEdit.setPlaceholderText("Select output directory for results")
        outputBrowse = TransparentToolButton(FIF.FOLDER, card)
        outputBrowse.clicked.connect(self.browseOutputDir)
        outputLayout.addWidget(outputLabel)
        outputLayout.addWidget(self.outputDirEdit)
        outputLayout.addWidget(outputBrowse)
        cardLayout.addLayout(outputLayout)

        return card

    def createParametersCard(self):
        """Create analysis parameters card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        titleLabel = SubtitleLabel("Analysis Parameters", card)
        cardLayout.addWidget(titleLabel)

        # Frequency range
        freqLayout = QHBoxLayout()
        
        fminLabel = BodyLabel("Min Freq (Hz):", card)
        self.fminSpin = DoubleSpinBox(card)
        self.fminSpin.setRange(0.01, 10.0)
        self.fminSpin.setValue(0.1)
        self.fminSpin.setSingleStep(0.1)
        
        fmaxLabel = BodyLabel("Max Freq (Hz):", card)
        self.fmaxSpin = DoubleSpinBox(card)
        self.fmaxSpin.setRange(1.0, 100.0)
        self.fmaxSpin.setValue(30.0)
        self.fmaxSpin.setSingleStep(5.0)
        
        nfLabel = BodyLabel("Num Points:", card)
        self.nfSpin = SpinBox(card)
        self.nfSpin.setRange(50, 500)
        self.nfSpin.setValue(100)
        
        freqLayout.addWidget(fminLabel)
        freqLayout.addWidget(self.fminSpin)
        freqLayout.addSpacing(20)
        freqLayout.addWidget(fmaxLabel)
        freqLayout.addWidget(self.fmaxSpin)
        freqLayout.addSpacing(20)
        freqLayout.addWidget(nfLabel)
        freqLayout.addWidget(self.nfSpin)
        freqLayout.addStretch()
        cardLayout.addLayout(freqLayout)

        # Number of examples
        examplesLayout = QHBoxLayout()
        examplesLabel = BodyLabel("Example Figures:", card)
        self.examplesSpin = SpinBox(card)
        self.examplesSpin.setRange(1, 10)
        self.examplesSpin.setValue(3)
        examplesLayout.addWidget(examplesLabel)
        examplesLayout.addWidget(self.examplesSpin)
        examplesLayout.addStretch()
        cardLayout.addLayout(examplesLayout)

        return card

    def createProgressCard(self):
        """Create progress display card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        titleLabel = SubtitleLabel("Progress", card)
        cardLayout.addWidget(titleLabel)

        # Progress bar
        progressLayout = QHBoxLayout()
        self.progressBar = ProgressBar(card)
        self.progressBar.setValue(0)
        self.progressLabel = BodyLabel("Ready", card)
        progressLayout.addWidget(self.progressBar, 1)
        progressLayout.addWidget(self.progressLabel)
        cardLayout.addLayout(progressLayout)

        # Log output
        self.logText = TextEdit(card)
        self.logText.setReadOnly(True)
        self.logText.setMinimumHeight(200)
        self.logText.setMaximumHeight(300)
        cardLayout.addWidget(self.logText)

        return card

    def createResultsCard(self):
        """Create results display card."""
        card = CardWidget(self)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(20, 20, 20, 20)
        cardLayout.setSpacing(15)

        titleLabel = SubtitleLabel("Results Summary", card)
        cardLayout.addWidget(titleLabel)

        # Results table
        self.resultsTable = QTableWidget(card)
        self.resultsTable.setColumnCount(4)
        self.resultsTable.setHorizontalHeaderLabels(["Metric", "Value", "Unit", "Description"])
        self.resultsTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.resultsTable.setMinimumHeight(200)
        cardLayout.addWidget(self.resultsTable)

        # Open output folder button
        self.openFolderButton = PushButton(FIF.FOLDER, "Open Output Folder", card)
        self.openFolderButton.setEnabled(False)
        self.openFolderButton.clicked.connect(self.openOutputFolder)
        cardLayout.addWidget(self.openFolderButton)

        return card

    def browseProfilesDir(self):
        """Browse for profiles directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Profiles Directory",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.profilesDirEdit.setText(directory)

    def browseOutputDir(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.outputDirEdit.setText(directory)

    def runWorkflow(self):
        """Run the research workflow."""
        profiles_dir = self.profilesDirEdit.text().strip()
        output_dir = self.outputDirEdit.text().strip()

        if not profiles_dir or not Path(profiles_dir).exists():
            InfoBar.error(
                title="Invalid Input",
                content="Please select a valid profiles directory",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT
            )
            return

        if not output_dir:
            InfoBar.error(
                title="Invalid Input",
                content="Please select an output directory",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT
            )
            return

        # Prepare configuration
        config = {
            'hv_forward': {
                'fmin': self.fminSpin.value(),
                'fmax': self.fmaxSpin.value(),
                'nf': self.nfSpin.value()
            }
        }

        # Disable UI
        self.runButton.setEnabled(False)
        self.logText.clear()
        self.progressBar.setValue(0)
        self.progressLabel.setText("Starting...")

        # Create and start worker
        self.worker = ResearchWorker(
            profiles_dir, output_dir, config, self.examplesSpin.value()
        )
        self.worker.progress.connect(self.onProgress)
        self.worker.log.connect(self.onLog)
        self.worker.finished.connect(self.onFinished)
        self.worker.error.connect(self.onError)
        self.worker.start()

    def onProgress(self, current, total, message):
        """Handle progress update."""
        if total > 0:
            self.progressBar.setValue(int(current / total * 100))
        self.progressLabel.setText(f"{current}/{total}: {message}")

    def onLog(self, message):
        """Handle log message."""
        self.logText.append(message)

    def onFinished(self, result):
        """Handle workflow completion."""
        self.runButton.setEnabled(True)
        self.progressBar.setValue(100)
        self.progressLabel.setText("Complete!")
        self.output_dir = result.get('output_dir', '')
        self.openFolderButton.setEnabled(True)

        # Update results table
        if 'batch_result' in result and result['batch_result'].get('statistics'):
            stats = result['batch_result']['statistics']
            self.updateResultsTable(stats)

        InfoBar.success(
            title="Workflow Complete",
            content=f"Results saved to: {result.get('output_dir', 'N/A')}",
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=5000
        )

    def onError(self, error_msg):
        """Handle workflow error."""
        self.runButton.setEnabled(True)
        self.progressLabel.setText("Error!")
        self.logText.append(f"\n[ERROR] {error_msg}")

        InfoBar.error(
            title="Workflow Error",
            content=error_msg[:100],
            parent=self,
            position=InfoBarPosition.TOP_RIGHT
        )

    def updateResultsTable(self, stats):
        """Update results table with statistics."""
        data = [
            ("Profiles Analyzed", str(stats.n_profiles), "", "Total number of profiles"),
            ("Success Rate", f"{stats.success_rate:.1f}", "%", "Processing success rate"),
            ("f0 (Deep)", f"{stats.f0_mean:.2f} +/- {stats.f0_std:.2f}", "Hz", "Deep resonance frequency"),
            ("f1 (Shallow)", f"{stats.f1_mean:.2f} +/- {stats.f1_std:.2f}", "Hz", "Shallow resonance frequency"),
            ("Ratio f1/f0", f"{stats.freq_ratio_mean:.2f} +/- {stats.freq_ratio_std:.2f}", "", "Frequency ratio"),
            ("Separation Success", f"{stats.separation_success_rate:.1f}", "%", "Clear separation achieved"),
        ]

        self.resultsTable.setRowCount(len(data))
        for row, (metric, value, unit, desc) in enumerate(data):
            self.resultsTable.setItem(row, 0, QTableWidgetItem(metric))
            self.resultsTable.setItem(row, 1, QTableWidgetItem(value))
            self.resultsTable.setItem(row, 2, QTableWidgetItem(unit))
            self.resultsTable.setItem(row, 3, QTableWidgetItem(desc))

    def openOutputFolder(self):
        """Open output folder in file explorer."""
        if hasattr(self, 'output_dir') and self.output_dir:
            import subprocess
            import platform
            if platform.system() == 'Windows':
                subprocess.run(['explorer', self.output_dir])
            elif platform.system() == 'Darwin':
                subprocess.run(['open', self.output_dir])
            else:
                subprocess.run(['xdg-open', self.output_dir])
