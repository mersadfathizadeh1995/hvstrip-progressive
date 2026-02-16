"""
Parallel Batch Analysis GUI Page
================================

GUI interface for running parallel batch analysis on multiple soil profiles.

Two tabs:
1. Batch Processing - Run HVSR stripping on profiles (parallel)
2. Post-Processing - Generate figures and statistics from results

Author: Mersad Fathizadeh
Date: December 2025
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing as mp
import pandas as pd
import numpy as np

from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QGroupBox, QFileDialog,
    QComboBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFrame, QTabWidget
)
from qfluentwidgets import (
    CardWidget, PrimaryPushButton, PushButton, 
    LineEdit, SpinBox, DoubleSpinBox, ComboBox,
    ProgressBar, TextEdit, CheckBox, InfoBar, InfoBarPosition,
    FluentIcon as FIF, ToolButton
)


class WorkerSignals(QObject):
    """Signals for the worker thread."""
    progress = Signal(int, int, str)  # current, total, message
    result = Signal(dict)  # single result
    finished = Signal(list)  # all results
    error = Signal(str)
    log = Signal(str)


class ParallelAnalysisWorker(QThread):
    """Worker thread for parallel batch analysis."""
    
    signals = WorkerSignals()
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True
        
    def run(self):
        """Run the parallel analysis."""
        try:
            from hvstrip_progressive.core.batch_workflow import run_complete_workflow
            
            input_dir = Path(self.config['input_dir'])
            output_dir = Path(self.config['output_dir'])
            n_workers = self.config['n_workers']
            hvf_path = self.config.get('hvf_path', '')
            fmin = self.config.get('fmin', 0.1)
            fmax = self.config.get('fmax', 30.0)
            
            # Collect all profile files
            self.signals.log.emit("Collecting profile files...")
            
            profile_files = []
            
            # Check if input has scenario subfolders
            scenarios = ['classic_basin', 'shallow_basin', 'very_deep_basin', 
                        'high_contrast', 'variable_structure']
            
            has_scenarios = any((input_dir / s / 'txt').exists() for s in scenarios)
            
            if has_scenarios:
                for scenario in scenarios:
                    scenario_txt = input_dir / scenario / 'txt'
                    if scenario_txt.exists():
                        for f in sorted(scenario_txt.glob('*.txt')):
                            profile_files.append((f, scenario))
            else:
                # Flat structure - look for txt folder or direct .txt files
                txt_dir = input_dir / 'txt' if (input_dir / 'txt').exists() else input_dir
                for f in sorted(txt_dir.glob('*.txt')):
                    profile_files.append((f, 'default'))
            
            total = len(profile_files)
            self.signals.log.emit(f"Found {total} profiles to process")
            self.signals.progress.emit(0, total, f"Starting with {n_workers} workers...")
            
            if total == 0:
                self.signals.error.emit("No profile files found!")
                return
            
            # Process profiles
            results = []
            completed = 0
            successful = 0
            
            # Create workflow config
            workflow_config = {
                "hv_forward": {
                    "fmin": fmin,
                    "fmax": fmax,
                    "nf": 100
                }
            }
            
            if hvf_path and Path(hvf_path).exists():
                workflow_config["hvf_path"] = hvf_path
            
            # Process in batches to allow cancellation
            batch_size = n_workers * 2
            
            for batch_start in range(0, total, batch_size):
                if self._is_cancelled:
                    self.signals.log.emit("Analysis cancelled by user")
                    break
                
                batch_end = min(batch_start + batch_size, total)
                batch = profile_files[batch_start:batch_end]
                
                # Process batch using ThreadPoolExecutor (safer for GUI)
                from concurrent.futures import ThreadPoolExecutor
                
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_to_profile = {}
                    
                    for profile_path, scenario in batch:
                        profile_name = profile_path.stem
                        
                        if has_scenarios:
                            profile_output = output_dir / scenario / profile_name
                        else:
                            profile_output = output_dir / profile_name
                        
                        future = executor.submit(
                            self._process_single,
                            str(profile_path),
                            str(profile_output),
                            workflow_config,
                            profile_name,
                            scenario
                        )
                        future_to_profile[future] = (profile_name, scenario)
                    
                    for future in as_completed(future_to_profile):
                        if self._is_cancelled:
                            break
                            
                        profile_name, scenario = future_to_profile[future]
                        
                        try:
                            result = future.result(timeout=300)  # 5 min timeout
                            results.append(result)
                            
                            if result.get('success', False):
                                successful += 1
                            
                            completed += 1
                            
                            self.signals.progress.emit(
                                completed, total,
                                f"Processed {profile_name} ({scenario})"
                            )
                            self.signals.result.emit(result)
                            
                        except Exception as e:
                            completed += 1
                            error_result = {
                                'profile_name': profile_name,
                                'scenario': scenario,
                                'success': False,
                                'error': str(e)[:100]
                            }
                            results.append(error_result)
                            self.signals.result.emit(error_result)
                            self.signals.log.emit(f"Error: {profile_name}: {e}")
            
            # Save results
            self._save_results(results, output_dir)
            
            self.signals.log.emit(f"\nCompleted: {completed}/{total}")
            self.signals.log.emit(f"Successful: {successful}")
            self.signals.log.emit(f"Failed: {completed - successful}")
            
            self.signals.finished.emit(results)
            
        except Exception as e:
            self.signals.error.emit(str(e))
    
    def _process_single(self, profile_path: str, output_dir: str, 
                        config: dict, name: str, scenario: str) -> dict:
        """Process a single profile."""
        start_time = time.time()
        
        try:
            from hvstrip_progressive.core.batch_workflow import run_complete_workflow
            
            result = run_complete_workflow(profile_path, output_dir, config)
            
            elapsed = time.time() - start_time
            
            # Extract frequencies
            f0, f1 = 0.0, 0.0
            if result.get('success', False):
                strip_dir = Path(result.get('strip_directory', ''))
                
                # Get f0 from Step_0
                step0_summary = strip_dir / "Step_0" / "step_summary.csv"
                if step0_summary.exists():
                    import pandas as pd
                    df = pd.read_csv(step0_summary)
                    if not df.empty:
                        f0 = float(df.iloc[0].get('Peak_Frequency_Hz', 0))
                
                # Get f1 from last step
                for step_name in ['Step_2', 'Step_1']:
                    step_summary = strip_dir / step_name / "step_summary.csv"
                    if step_summary.exists():
                        import pandas as pd
                        df = pd.read_csv(step_summary)
                        if not df.empty:
                            f1 = float(df.iloc[0].get('Peak_Frequency_Hz', 0))
                            break
            
            return {
                'profile_name': name,
                'scenario': scenario,
                'success': result.get('success', False),
                'f0_hz': f0,
                'f1_hz': f1,
                'time_seconds': elapsed,
                'error': ''
            }
            
        except Exception as e:
            return {
                'profile_name': name,
                'scenario': scenario,
                'success': False,
                'f0_hz': 0,
                'f1_hz': 0,
                'time_seconds': time.time() - start_time,
                'error': str(e)[:100]
            }
    
    def _save_results(self, results: list, output_dir: Path):
        """Save results to CSV."""
        import pandas as pd
        
        data = []
        for r in results:
            f0 = r.get('f0_hz', 0)
            f1 = r.get('f1_hz', 0)
            data.append({
                'profile_name': r.get('profile_name', ''),
                'scenario': r.get('scenario', ''),
                'success': r.get('success', False),
                'f0_hz': f0,
                'f1_hz': f1,
                'freq_ratio': f1 / f0 if f0 > 0 else 0,
                'time_seconds': r.get('time_seconds', 0),
                'error': r.get('error', '')
            })
        
        df = pd.DataFrame(data)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'batch_results.csv', index=False)


class PostProcessingWorker(QThread):
    """Worker thread for post-processing (figures and statistics)."""
    
    signals = WorkerSignals()
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self._is_cancelled = False
        
    def cancel(self):
        self._is_cancelled = True
        
    def run(self):
        """Run the post-processing."""
        try:
            from hvstrip_progressive.core.complete_batch_workflow import (
                extract_frequencies_from_profile,
                generate_resonance_separation_figure,
                generate_study_statistics,
                generate_publication_figures
            )
            
            results_dir = Path(self.config['results_dir'])
            n_examples = self.config.get('n_examples', 10)
            
            self.signals.log.emit("=" * 50)
            self.signals.log.emit("POST-PROCESSING WORKFLOW")
            self.signals.log.emit("=" * 50)
            
            # Find profile directories
            self.signals.log.emit("\nPhase 1: Scanning for processed profiles...")
            
            profile_dirs = []
            for item in results_dir.rglob("*"):
                if item.is_dir() and (item / "strip").exists():
                    profile_dirs.append(item)
            
            if not profile_dirs:
                self.signals.error.emit("No processed profiles found!")
                return
            
            total = len(profile_dirs)
            self.signals.log.emit(f"Found {total} processed profiles")
            self.signals.progress.emit(0, total, "Extracting frequencies...")
            
            # Extract frequencies from all profiles
            results = []
            for i, profile_dir in enumerate(sorted(profile_dirs)):
                if self._is_cancelled:
                    return
                    
                f0, f1, metadata = extract_frequencies_from_profile(profile_dir)
                
                results.append({
                    'profile_name': profile_dir.name,
                    'output_dir': str(profile_dir),
                    'success': True,
                    'f0_hz': f0,
                    'f1_hz': f1,
                    'freq_ratio': f1 / f0 if f0 > 0 else 0,
                    'f0_amplitude': metadata.get('f0_amplitude', 0),
                    'f1_amplitude': metadata.get('f1_amplitude', 0),
                })
                
                if (i + 1) % 50 == 0:
                    self.signals.progress.emit(i + 1, total, f"Extracted {i+1}/{total}")
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_dir / "batch_results_complete.csv", index=False)
            
            valid = results_df[(results_df['f0_hz'] > 0) & (results_df['f1_hz'] > 0)]
            self.signals.log.emit(f"\nValid frequencies: {len(valid)}/{len(results_df)}")
            
            if len(valid) > 0:
                self.signals.log.emit(f"  f0: {valid['f0_hz'].min():.3f} - {valid['f0_hz'].max():.3f} Hz")
                self.signals.log.emit(f"  f1: {valid['f1_hz'].min():.3f} - {valid['f1_hz'].max():.3f} Hz")
            
            # Generate example figures
            self.signals.log.emit(f"\nPhase 2: Generating {n_examples} example figures...")
            self.signals.progress.emit(0, n_examples, "Creating example figures...")
            
            examples_dir = results_dir / "example_figures"
            examples_dir.mkdir(exist_ok=True)
            
            if len(valid) > 0:
                sample_indices = np.linspace(0, len(valid)-1, 
                                             min(n_examples, len(valid)), 
                                             dtype=int)
                sample_profiles = valid.iloc[sample_indices]
                
                for i, (_, row) in enumerate(sample_profiles.iterrows()):
                    if self._is_cancelled:
                        return
                        
                    profile_output = Path(row['output_dir'])
                    fig_path = examples_dir / f"example_{i+1}_{row['profile_name']}.png"
                    
                    success = generate_resonance_separation_figure(profile_output, fig_path)
                    self.signals.progress.emit(i + 1, n_examples, f"Created {fig_path.name}")
                    self.signals.log.emit(f"  [{i+1}] {fig_path.name}: {'OK' if success else 'FAILED'}")
            
            # Generate statistics
            self.signals.log.emit("\nPhase 3: Computing statistics...")
            self.signals.progress.emit(0, 1, "Computing statistics...")
            
            stats_dir = results_dir / "statistics"
            stats = generate_study_statistics(results_df, stats_dir)
            
            if stats:
                self.signals.log.emit(f"  f0: {stats['f0_mean']:.3f} +/- {stats['f0_std']:.3f} Hz")
                self.signals.log.emit(f"  f1: {stats['f1_mean']:.3f} +/- {stats['f1_std']:.3f} Hz")
                self.signals.log.emit(f"  Ratio: {stats['ratio_mean']:.2f} +/- {stats['ratio_std']:.2f}")
            
            # Generate publication figures
            self.signals.log.emit("\nPhase 4: Generating publication figures...")
            self.signals.progress.emit(0, 4, "Creating publication figures...")
            
            figures = generate_publication_figures(results_df, results_dir)
            
            for fig in figures:
                self.signals.log.emit(f"  Created: {Path(fig).name}")
            
            self.signals.log.emit("\n" + "=" * 50)
            self.signals.log.emit("POST-PROCESSING COMPLETE")
            self.signals.log.emit("=" * 50)
            self.signals.log.emit(f"\nOutput: {results_dir}")
            self.signals.log.emit("  - batch_results_complete.csv")
            self.signals.log.emit("  - statistics/")
            self.signals.log.emit("  - example_figures/")
            self.signals.log.emit("  - figures/")
            
            self.signals.finished.emit([{'success': True, 'stats': stats}])
            
        except Exception as e:
            import traceback
            self.signals.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class ParallelBatchPage(QWidget):
    """GUI page for parallel batch analysis with two tabs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("parallelBatchPage")
        self.worker = None
        self.postprocess_worker = None
        self.results = []
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface with tabs."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Parallel Batch Analysis")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("QTabWidget::pane { border: 1px solid #ccc; }")
        
        # Tab 1: Batch Processing
        batch_tab = self._create_batch_tab()
        self.tab_widget.addTab(batch_tab, "1. Batch Processing")
        
        # Tab 2: Post-Processing (Figures & Statistics)
        postprocess_tab = self._create_postprocess_tab()
        self.tab_widget.addTab(postprocess_tab, "2. Figures & Statistics")
        
        layout.addWidget(self.tab_widget, 1)
    
    # =========================================================================
    # TAB 1: BATCH PROCESSING
    # =========================================================================
    
    def _create_batch_tab(self) -> QWidget:
        """Create the batch processing tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Configuration
        left_panel = self._create_config_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 600])
        layout.addWidget(splitter, 1)
        
        # Bottom - Progress and controls
        bottom_panel = self._create_control_panel()
        layout.addWidget(bottom_panel)
        
        return tab
        
    def _create_config_panel(self) -> QWidget:
        """Create the configuration panel."""
        panel = CardWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        
        # Input Directory
        input_group = QGroupBox("Input Directory")
        input_layout = QHBoxLayout(input_group)
        self.input_edit = LineEdit()
        self.input_edit.setPlaceholderText("Select profiles directory...")
        input_browse = ToolButton(FIF.FOLDER)
        input_browse.clicked.connect(self._browse_input)
        input_layout.addWidget(self.input_edit, 1)
        input_layout.addWidget(input_browse)
        layout.addWidget(input_group)
        
        # Output Directory
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout(output_group)
        self.output_edit = LineEdit()
        self.output_edit.setPlaceholderText("Select output directory...")
        output_browse = ToolButton(FIF.FOLDER)
        output_browse.clicked.connect(self._browse_output)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(output_browse)
        layout.addWidget(output_group)
        
        # HVf.exe Path
        hvf_group = QGroupBox("HVf.exe Path (Optional)")
        hvf_layout = QHBoxLayout(hvf_group)
        self.hvf_edit = LineEdit()
        self.hvf_edit.setPlaceholderText("Path to HVf.exe (leave empty for default)")
        hvf_browse = ToolButton(FIF.APPLICATION)
        hvf_browse.clicked.connect(self._browse_hvf)
        hvf_layout.addWidget(self.hvf_edit, 1)
        hvf_layout.addWidget(hvf_browse)
        layout.addWidget(hvf_group)
        
        # Worker Settings
        worker_group = QGroupBox("Parallel Processing")
        worker_layout = QGridLayout(worker_group)
        
        # Number of workers
        worker_layout.addWidget(QLabel("Workers:"), 0, 0)
        self.workers_spin = SpinBox()
        self.workers_spin.setRange(1, mp.cpu_count() * 2)
        self.workers_spin.setValue(min(10, mp.cpu_count()))
        self.workers_spin.setToolTip(f"CPU cores available: {mp.cpu_count()}")
        worker_layout.addWidget(self.workers_spin, 0, 1)
        
        cpu_label = QLabel(f"(CPU: {mp.cpu_count()} cores)")
        cpu_label.setStyleSheet("color: gray;")
        worker_layout.addWidget(cpu_label, 0, 2)
        
        layout.addWidget(worker_group)
        
        # Frequency Settings
        freq_group = QGroupBox("Frequency Range")
        freq_layout = QGridLayout(freq_group)
        
        freq_layout.addWidget(QLabel("f_min (Hz):"), 0, 0)
        self.fmin_spin = DoubleSpinBox()
        self.fmin_spin.setRange(0.01, 10.0)
        self.fmin_spin.setValue(0.1)
        self.fmin_spin.setDecimals(2)
        freq_layout.addWidget(self.fmin_spin, 0, 1)
        
        freq_layout.addWidget(QLabel("f_max (Hz):"), 1, 0)
        self.fmax_spin = DoubleSpinBox()
        self.fmax_spin.setRange(1.0, 100.0)
        self.fmax_spin.setValue(30.0)
        self.fmax_spin.setDecimals(1)
        freq_layout.addWidget(self.fmax_spin, 1, 1)
        
        layout.addWidget(freq_group)
        
        # Profile Info
        self.info_label = QLabel("No profiles loaded")
        self.info_label.setStyleSheet("color: gray; padding: 10px;")
        layout.addWidget(self.info_label)
        
        # Scan button
        scan_btn = PushButton("Scan Profiles")
        scan_btn.setIcon(FIF.SEARCH)
        scan_btn.clicked.connect(self._scan_profiles)
        layout.addWidget(scan_btn)
        
        layout.addStretch()
        
        return panel
    
    def _create_results_panel(self) -> QWidget:
        """Create the results panel."""
        panel = CardWidget()
        layout = QVBoxLayout(panel)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Profile", "Scenario", "Status", "f₀ (Hz)", "f₁ (Hz)", "Time (s)"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        layout.addWidget(self.results_table, 1)
        
        # Log output
        log_label = QLabel("Log Output:")
        layout.addWidget(log_label)
        
        self.log_text = TextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        return panel
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel."""
        panel = CardWidget()
        layout = QHBoxLayout(panel)
        
        # Progress bar
        self.progress_bar = ProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar, 1)
        
        # Progress label
        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(200)
        layout.addWidget(self.progress_label)
        
        # Buttons
        self.start_btn = PrimaryPushButton("Start Analysis")
        self.start_btn.setIcon(FIF.PLAY)
        self.start_btn.clicked.connect(self._start_analysis)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = PushButton("Stop")
        self.stop_btn.setIcon(FIF.PAUSE)
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        return panel
    
    def _browse_input(self):
        """Browse for input directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Profiles Directory",
            str(Path.home())
        )
        if path:
            self.input_edit.setText(path)
            self._scan_profiles()
    
    def _browse_output(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            str(Path.home())
        )
        if path:
            self.output_edit.setText(path)
    
    def _browse_hvf(self):
        """Browse for HVf.exe."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HVf.exe",
            str(Path.home()),
            "Executable (*.exe)"
        )
        if path:
            self.hvf_edit.setText(path)
    
    def _scan_profiles(self):
        """Scan the input directory for profiles."""
        input_path = self.input_edit.text()
        if not input_path:
            return
        
        input_dir = Path(input_path)
        if not input_dir.exists():
            self.info_label.setText("Directory not found!")
            self.info_label.setStyleSheet("color: red;")
            return
        
        # Count profiles
        scenarios = ['classic_basin', 'shallow_basin', 'very_deep_basin',
                    'high_contrast', 'variable_structure']
        
        counts = {}
        total = 0
        
        # Check for scenario structure
        for scenario in scenarios:
            scenario_txt = input_dir / scenario / 'txt'
            if scenario_txt.exists():
                count = len(list(scenario_txt.glob('*.txt')))
                if count > 0:
                    counts[scenario] = count
                    total += count
        
        # Check flat structure
        if total == 0:
            txt_dir = input_dir / 'txt' if (input_dir / 'txt').exists() else input_dir
            count = len(list(txt_dir.glob('*.txt')))
            if count > 0:
                counts['default'] = count
                total = count
        
        # Update info
        if total > 0:
            info_parts = [f"<b>Total: {total} profiles</b>"]
            for scenario, count in counts.items():
                info_parts.append(f"  • {scenario}: {count}")
            
            self.info_label.setText("<br>".join(info_parts))
            self.info_label.setStyleSheet("color: green;")
        else:
            self.info_label.setText("No .txt profile files found!")
            self.info_label.setStyleSheet("color: red;")
    
    def _start_analysis(self):
        """Start the parallel analysis."""
        input_path = self.input_edit.text()
        output_path = self.output_edit.text()
        
        if not input_path:
            InfoBar.error(
                title="Error",
                content="Please select an input directory",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        if not output_path:
            # Default output path
            output_path = str(Path(input_path).parent / "batch_results")
            self.output_edit.setText(output_path)
        
        # Prepare config
        config = {
            'input_dir': input_path,
            'output_dir': output_path,
            'n_workers': self.workers_spin.value(),
            'hvf_path': self.hvf_edit.text(),
            'fmin': self.fmin_spin.value(),
            'fmax': self.fmax_spin.value()
        }
        
        # Clear previous results
        self.results_table.setRowCount(0)
        self.log_text.clear()
        self.results = []
        
        # Create and start worker
        self.worker = ParallelAnalysisWorker(config)
        self.worker.signals = WorkerSignals()
        self.worker.signals.progress.connect(self._on_progress)
        self.worker.signals.result.connect(self._on_result)
        self.worker.signals.finished.connect(self._on_finished)
        self.worker.signals.error.connect(self._on_error)
        self.worker.signals.log.connect(self._on_log)
        
        self.worker.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_label.setText("Starting...")
        
        self._log("Analysis started...")
    
    def _stop_analysis(self):
        """Stop the analysis."""
        if self.worker:
            self.worker.cancel()
            self._log("Stopping analysis...")
    
    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress update."""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
        self.progress_label.setText(f"{current}/{total} - {message}")
    
    def _on_result(self, result: dict):
        """Handle single result."""
        self.results.append(result)
        
        # Add to table
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(result.get('profile_name', '')))
        self.results_table.setItem(row, 1, QTableWidgetItem(result.get('scenario', '')))
        
        status = "✓" if result.get('success', False) else "✗"
        status_item = QTableWidgetItem(status)
        status_item.setForeground(Qt.green if result.get('success') else Qt.red)
        self.results_table.setItem(row, 2, status_item)
        
        f0 = result.get('f0_hz', 0)
        f1 = result.get('f1_hz', 0)
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{f0:.2f}" if f0 > 0 else "-"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{f1:.2f}" if f1 > 0 else "-"))
        self.results_table.setItem(row, 5, QTableWidgetItem(f"{result.get('time_seconds', 0):.1f}"))
        
        # Scroll to bottom
        self.results_table.scrollToBottom()
    
    def _on_finished(self, results: list):
        """Handle analysis completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)
        
        self.progress_bar.setValue(100)
        self.progress_label.setText(f"Complete: {successful}/{total} successful")
        
        self._log(f"\n{'='*50}")
        self._log(f"Analysis complete!")
        self._log(f"Total: {total}, Successful: {successful}, Failed: {total - successful}")
        self._log(f"Results saved to: {self.output_edit.text()}")
        
        InfoBar.success(
            title="Complete",
            content=f"Processed {total} profiles ({successful} successful)",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )
    
    def _on_error(self, error: str):
        """Handle error."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self._log(f"ERROR: {error}")
        
        InfoBar.error(
            title="Error",
            content=error[:100],
            parent=self,
            position=InfoBarPosition.TOP
        )
    
    def _on_log(self, message: str):
        """Handle log message."""
        self._log(message)
    
    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)
    
    # =========================================================================
    # TAB 2: POST-PROCESSING (FIGURES & STATISTICS)
    # =========================================================================
    
    def _create_postprocess_tab(self) -> QWidget:
        """Create the post-processing tab for figures and statistics."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Description
        desc = QLabel(
            "<b>Generate Figures & Statistics</b><br>"
            "Use this tab to generate publication figures and statistics from "
            "already processed batch results. Point to a directory containing "
            "processed profiles (with 'strip' subfolders)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; background: #f0f8ff; border-radius: 5px;")
        layout.addWidget(desc)
        
        # Main content
        content = QSplitter(Qt.Horizontal)
        
        # Left - Configuration
        left = CardWidget()
        left_layout = QVBoxLayout(left)
        
        # Results Directory
        dir_group = QGroupBox("Results Directory")
        dir_layout = QHBoxLayout(dir_group)
        self.pp_results_edit = LineEdit()
        self.pp_results_edit.setPlaceholderText("Select directory with processed profiles...")
        pp_browse = ToolButton(FIF.FOLDER)
        pp_browse.clicked.connect(self._pp_browse_results)
        dir_layout.addWidget(self.pp_results_edit, 1)
        dir_layout.addWidget(pp_browse)
        left_layout.addWidget(dir_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QGridLayout(options_group)
        
        options_layout.addWidget(QLabel("Example figures:"), 0, 0)
        self.pp_examples_spin = SpinBox()
        self.pp_examples_spin.setRange(1, 50)
        self.pp_examples_spin.setValue(10)
        options_layout.addWidget(self.pp_examples_spin, 0, 1)
        
        left_layout.addWidget(options_group)
        
        # Scan info
        self.pp_info_label = QLabel("No directory selected")
        self.pp_info_label.setStyleSheet("color: gray; padding: 10px;")
        left_layout.addWidget(self.pp_info_label)
        
        # Scan button
        scan_btn = PushButton("Scan Directory")
        scan_btn.setIcon(FIF.SEARCH)
        scan_btn.clicked.connect(self._pp_scan_results)
        left_layout.addWidget(scan_btn)
        
        left_layout.addStretch()
        content.addWidget(left)
        
        # Right - Log
        right = CardWidget()
        right_layout = QVBoxLayout(right)
        
        right_layout.addWidget(QLabel("<b>Processing Log:</b>"))
        self.pp_log_text = TextEdit()
        self.pp_log_text.setReadOnly(True)
        right_layout.addWidget(self.pp_log_text)
        
        content.addWidget(right)
        content.setSizes([350, 650])
        
        layout.addWidget(content, 1)
        
        # Bottom controls
        bottom = CardWidget()
        bottom_layout = QHBoxLayout(bottom)
        
        self.pp_progress_bar = ProgressBar()
        self.pp_progress_bar.setRange(0, 100)
        self.pp_progress_bar.setValue(0)
        bottom_layout.addWidget(self.pp_progress_bar, 1)
        
        self.pp_progress_label = QLabel("Ready")
        self.pp_progress_label.setMinimumWidth(200)
        bottom_layout.addWidget(self.pp_progress_label)
        
        self.pp_start_btn = PrimaryPushButton("Generate Figures & Statistics")
        self.pp_start_btn.setIcon(FIF.PHOTO)
        self.pp_start_btn.clicked.connect(self._pp_start)
        bottom_layout.addWidget(self.pp_start_btn)
        
        self.pp_stop_btn = PushButton("Stop")
        self.pp_stop_btn.setIcon(FIF.PAUSE)
        self.pp_stop_btn.clicked.connect(self._pp_stop)
        self.pp_stop_btn.setEnabled(False)
        bottom_layout.addWidget(self.pp_stop_btn)
        
        # Open output folder button
        self.pp_open_btn = PushButton("Open Output")
        self.pp_open_btn.setIcon(FIF.FOLDER)
        self.pp_open_btn.clicked.connect(self._pp_open_output)
        self.pp_open_btn.setEnabled(False)
        bottom_layout.addWidget(self.pp_open_btn)
        
        layout.addWidget(bottom)
        
        return tab
    
    def _pp_browse_results(self):
        """Browse for results directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Results Directory",
            str(Path.home())
        )
        if path:
            self.pp_results_edit.setText(path)
            self._pp_scan_results()
    
    def _pp_scan_results(self):
        """Scan for processed profiles in the results directory."""
        results_path = self.pp_results_edit.text()
        if not results_path:
            return
        
        results_dir = Path(results_path)
        if not results_dir.exists():
            self.pp_info_label.setText("Directory not found!")
            self.pp_info_label.setStyleSheet("color: red;")
            return
        
        # Count processed profiles (directories with 'strip' subfolder)
        profile_count = 0
        for item in results_dir.rglob("*"):
            if item.is_dir() and (item / "strip").exists():
                profile_count += 1
        
        if profile_count > 0:
            self.pp_info_label.setText(
                f"<b style='color: green;'>Found {profile_count} processed profiles</b><br>"
                f"Ready to generate figures and statistics."
            )
            self.pp_info_label.setStyleSheet("")
        else:
            self.pp_info_label.setText(
                "No processed profiles found!<br>"
                "Make sure the directory contains profile folders with 'strip' subfolders."
            )
            self.pp_info_label.setStyleSheet("color: red;")
    
    def _pp_start(self):
        """Start the post-processing."""
        results_path = self.pp_results_edit.text()
        
        if not results_path:
            InfoBar.error(
                title="Error",
                content="Please select a results directory",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        # Prepare config
        config = {
            'results_dir': results_path,
            'n_examples': self.pp_examples_spin.value()
        }
        
        # Clear log
        self.pp_log_text.clear()
        
        # Create and start worker
        self.postprocess_worker = PostProcessingWorker(config)
        self.postprocess_worker.signals = WorkerSignals()
        self.postprocess_worker.signals.progress.connect(self._pp_on_progress)
        self.postprocess_worker.signals.finished.connect(self._pp_on_finished)
        self.postprocess_worker.signals.error.connect(self._pp_on_error)
        self.postprocess_worker.signals.log.connect(self._pp_on_log)
        
        self.postprocess_worker.start()
        
        # Update UI
        self.pp_start_btn.setEnabled(False)
        self.pp_stop_btn.setEnabled(True)
        self.pp_open_btn.setEnabled(False)
        self.pp_progress_label.setText("Starting...")
    
    def _pp_stop(self):
        """Stop the post-processing."""
        if self.postprocess_worker:
            self.postprocess_worker.cancel()
            self._pp_on_log("Stopping...")
    
    def _pp_on_progress(self, current: int, total: int, message: str):
        """Handle progress update."""
        if total > 0:
            percent = int(current / total * 100)
            self.pp_progress_bar.setValue(percent)
        self.pp_progress_label.setText(f"{current}/{total} - {message}")
    
    def _pp_on_finished(self, results: list):
        """Handle completion."""
        self.pp_start_btn.setEnabled(True)
        self.pp_stop_btn.setEnabled(False)
        self.pp_open_btn.setEnabled(True)
        
        self.pp_progress_bar.setValue(100)
        self.pp_progress_label.setText("Complete!")
        
        InfoBar.success(
            title="Complete",
            content="Figures and statistics generated successfully!",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=5000
        )
    
    def _pp_on_error(self, error: str):
        """Handle error."""
        self.pp_start_btn.setEnabled(True)
        self.pp_stop_btn.setEnabled(False)
        
        self._pp_on_log(f"ERROR: {error}")
        
        InfoBar.error(
            title="Error",
            content=error[:100],
            parent=self,
            position=InfoBarPosition.TOP
        )
    
    def _pp_on_log(self, message: str):
        """Handle log message."""
        self.pp_log_text.append(message)
    
    def _pp_open_output(self):
        """Open the output directory."""
        import subprocess
        results_path = self.pp_results_edit.text()
        if results_path and Path(results_path).exists():
            subprocess.Popen(f'explorer "{results_path}"')
