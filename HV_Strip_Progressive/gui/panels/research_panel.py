"""Research Panel — left-panel for the Research comparison study workflow.

Provides configuration editors for all ComparisonStudyConfig sections
and pipeline execution buttons with progress tracking.
"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QCheckBox,
    QFileDialog, QScrollArea, QProgressBar, QMessageBox, QGroupBox,
)

from ..widgets.style_constants import (
    BUTTON_PRIMARY, BUTTON_SUCCESS, BUTTON_DANGER, GEAR_BUTTON,
    SECONDARY_LABEL, EMOJI,
)
from ..widgets.collapsible_group import CollapsibleGroupBox

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]
SCENARIOS = [
    "gradual_increase", "sharp_contrast", "velocity_inversion",
    "shallow_bedrock", "thick_soft_deposit", "thick_stiff_layer",
]
FIGURE_FORMATS = ["png", "pdf", "svg"]


class ResearchPanel(QWidget):
    """Left-panel content for the Research tab."""

    # Emitted when a pipeline phase completes — (phase_name, result_dict)
    phase_complete = pyqtSignal(str, dict)

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._worker = None
        self._runner = None
        self._build_ui()

    # ══════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # ── Study Info ──────────────────────────────────────────
        info_grp = CollapsibleGroupBox(
            f"{EMOJI['study']} Study Info", collapsed=False)
        info_lay = QVBoxLayout()
        info_lay.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Study Name:"))
        self._study_name = QLineEdit("HVSR Forward Modeling Comparison")
        row.addWidget(self._study_name)
        info_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output Dir:"))
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Select output directory...")
        row.addWidget(self._output_dir)
        btn = QPushButton("...")
        btn.setFixedWidth(30)
        btn.clicked.connect(self._browse_output_dir)
        row.addWidget(btn)
        info_lay.addLayout(row)

        info_grp.setContentLayout(info_lay)
        lay.addWidget(info_grp)

        # ── Profile Generation ──────────────────────────────────
        prof_grp = CollapsibleGroupBox(
            f"{EMOJI['profile']} Profile Generation", collapsed=True)
        prof_lay = QVBoxLayout()
        prof_lay.setSpacing(4)

        prof_lay.addWidget(QLabel("Geological Scenarios:"))
        self._scenario_checks = {}
        for sc in SCENARIOS:
            cb = QCheckBox(sc.replace("_", " ").title())
            cb.setChecked(True)
            self._scenario_checks[sc] = cb
            prof_lay.addWidget(cb)

        row = QHBoxLayout()
        row.addWidget(QLabel("Per scenario:"))
        self._n_per_scenario = QSpinBox()
        self._n_per_scenario.setRange(1, 100)
        self._n_per_scenario.setValue(15)
        row.addWidget(self._n_per_scenario)
        row.addWidget(QLabel("Random:"))
        self._n_random = QSpinBox()
        self._n_random.setRange(0, 200)
        self._n_random.setValue(30)
        row.addWidget(self._n_random)
        prof_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Depth:"))
        self._min_depth = QDoubleSpinBox()
        self._min_depth.setRange(5, 500)
        self._min_depth.setValue(20)
        self._min_depth.setSuffix(" m")
        row.addWidget(self._min_depth)
        row.addWidget(QLabel("–"))
        self._max_depth = QDoubleSpinBox()
        self._max_depth.setRange(10, 1000)
        self._max_depth.setValue(100)
        self._max_depth.setSuffix(" m")
        row.addWidget(self._max_depth)
        prof_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Vs:"))
        self._min_vs = QDoubleSpinBox()
        self._min_vs.setRange(50, 1000)
        self._min_vs.setValue(100)
        self._min_vs.setSuffix(" m/s")
        row.addWidget(self._min_vs)
        row.addWidget(QLabel("–"))
        self._max_vs = QDoubleSpinBox()
        self._max_vs.setRange(100, 5000)
        self._max_vs.setValue(1500)
        self._max_vs.setSuffix(" m/s")
        row.addWidget(self._max_vs)
        prof_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Layers:"))
        self._min_layers = QSpinBox()
        self._min_layers.setRange(2, 20)
        self._min_layers.setValue(3)
        row.addWidget(self._min_layers)
        row.addWidget(QLabel("–"))
        self._max_layers = QSpinBox()
        self._max_layers.setRange(3, 50)
        self._max_layers.setValue(10)
        row.addWidget(self._max_layers)
        prof_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("SoilGen Path:"))
        self._soilgen_path = QLineEdit()
        self._soilgen_path.setPlaceholderText("Path to SoilGen package...")
        row.addWidget(self._soilgen_path)
        btn = QPushButton("...")
        btn.setFixedWidth(30)
        btn.clicked.connect(self._browse_soilgen)
        row.addWidget(btn)
        prof_lay.addLayout(row)

        prof_grp.setContentLayout(prof_lay)
        lay.addWidget(prof_grp)

        # ── Engine Settings ─────────────────────────────────────
        eng_grp = CollapsibleGroupBox(
            f"{EMOJI['engine']} Engines", collapsed=True)
        eng_lay = QVBoxLayout()
        eng_lay.setSpacing(4)

        eng_lay.addWidget(QLabel("Active Engines:"))
        self._engine_checks = {}
        for eng in ENGINES:
            cb = QCheckBox(eng.replace("_", " ").title())
            cb.setChecked(True)
            self._engine_checks[eng] = cb
            eng_lay.addWidget(cb)

        row = QHBoxLayout()
        row.addWidget(QLabel("f min:"))
        self._freq_min = QDoubleSpinBox()
        self._freq_min.setRange(0.01, 10)
        self._freq_min.setValue(0.1)
        self._freq_min.setDecimals(2)
        self._freq_min.setSuffix(" Hz")
        row.addWidget(self._freq_min)
        row.addWidget(QLabel("f max:"))
        self._freq_max = QDoubleSpinBox()
        self._freq_max.setRange(1, 100)
        self._freq_max.setValue(50)
        self._freq_max.setSuffix(" Hz")
        row.addWidget(self._freq_max)
        eng_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("N freq:"))
        self._n_freq = QSpinBox()
        self._n_freq.setRange(50, 2000)
        self._n_freq.setValue(500)
        row.addWidget(self._n_freq)
        eng_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min prominence:"))
        self._min_prominence = QDoubleSpinBox()
        self._min_prominence.setRange(0.01, 5)
        self._min_prominence.setValue(0.3)
        self._min_prominence.setDecimals(2)
        row.addWidget(self._min_prominence)
        row.addWidget(QLabel("Min amplitude:"))
        self._min_amplitude = QDoubleSpinBox()
        self._min_amplitude.setRange(0.1, 20)
        self._min_amplitude.setValue(1.5)
        self._min_amplitude.setDecimals(1)
        row.addWidget(self._min_amplitude)
        eng_lay.addLayout(row)

        eng_grp.setContentLayout(eng_lay)
        lay.addWidget(eng_grp)

        # ── Metrics Settings ────────────────────────────────────
        met_grp = CollapsibleGroupBox(
            f"{EMOJI['metrics']} Metrics", collapsed=True)
        met_lay = QVBoxLayout()
        met_lay.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("f0 tolerance:"))
        self._f0_tolerance = QDoubleSpinBox()
        self._f0_tolerance.setRange(0.01, 1)
        self._f0_tolerance.setValue(0.1)
        self._f0_tolerance.setDecimals(2)
        self._f0_tolerance.setSuffix(" Hz")
        row.addWidget(self._f0_tolerance)
        met_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Agreement threshold:"))
        self._agreement_threshold = QDoubleSpinBox()
        self._agreement_threshold.setRange(0.01, 0.5)
        self._agreement_threshold.setValue(0.15)
        self._agreement_threshold.setDecimals(2)
        row.addWidget(self._agreement_threshold)
        met_lay.addLayout(row)

        self._normalize_curves = QCheckBox("Normalize curves before comparison")
        self._normalize_curves.setChecked(True)
        met_lay.addWidget(self._normalize_curves)

        met_grp.setContentLayout(met_lay)
        lay.addWidget(met_grp)

        # ── Output Settings ─────────────────────────────────────
        out_grp = CollapsibleGroupBox(
            f"{EMOJI['export']} Output Formats", collapsed=True)
        out_lay = QVBoxLayout()
        out_lay.setSpacing(4)

        self._save_csv = QCheckBox("CSV"); self._save_csv.setChecked(True)
        self._save_json = QCheckBox("JSON"); self._save_json.setChecked(True)
        self._save_latex = QCheckBox("LaTeX"); self._save_latex.setChecked(True)
        self._save_figures = QCheckBox("Figures"); self._save_figures.setChecked(True)
        for cb in [self._save_csv, self._save_json, self._save_latex, self._save_figures]:
            out_lay.addWidget(cb)

        row = QHBoxLayout()
        row.addWidget(QLabel("Figure format:"))
        self._fig_format = QComboBox()
        self._fig_format.addItems(FIGURE_FORMATS)
        row.addWidget(self._fig_format)
        row.addWidget(QLabel("DPI:"))
        self._fig_dpi = QSpinBox()
        self._fig_dpi.setRange(72, 600)
        self._fig_dpi.setValue(300)
        row.addWidget(self._fig_dpi)
        out_lay.addLayout(row)

        out_grp.setContentLayout(out_lay)
        lay.addWidget(out_grp)

        # ── Pipeline Controls ───────────────────────────────────
        pipe_grp = CollapsibleGroupBox(
            f"{EMOJI['run']} Pipeline", collapsed=False)
        pipe_lay = QVBoxLayout()
        pipe_lay.setSpacing(4)

        self._btn_gen_profiles = QPushButton(
            f"{EMOJI['generate']} Generate Profiles")
        self._btn_gen_profiles.setStyleSheet(BUTTON_PRIMARY)
        self._btn_gen_profiles.clicked.connect(
            lambda: self._run_phase("profiles"))
        pipe_lay.addWidget(self._btn_gen_profiles)

        self._btn_load_profiles = QPushButton(
            f"{EMOJI['folder']} Load Profiles...")
        self._btn_load_profiles.clicked.connect(self._load_profiles)
        pipe_lay.addWidget(self._btn_load_profiles)

        self._btn_run_comparison = QPushButton(
            f"{EMOJI['forward']} Run Comparison")
        self._btn_run_comparison.setStyleSheet(BUTTON_PRIMARY)
        self._btn_run_comparison.clicked.connect(
            lambda: self._run_phase("comparison"))
        pipe_lay.addWidget(self._btn_run_comparison)

        self._btn_compute_metrics = QPushButton(
            f"{EMOJI['metrics']} Compute Metrics")
        self._btn_compute_metrics.setStyleSheet(BUTTON_PRIMARY)
        self._btn_compute_metrics.clicked.connect(
            lambda: self._run_phase("metrics"))
        pipe_lay.addWidget(self._btn_compute_metrics)

        self._btn_field_validation = QPushButton(
            f"{EMOJI['field']} Field Validation")
        self._btn_field_validation.clicked.connect(
            lambda: self._run_phase("field_validation"))
        pipe_lay.addWidget(self._btn_field_validation)

        self._btn_gen_report = QPushButton(
            f"{EMOJI['report']} Generate Report")
        self._btn_gen_report.setStyleSheet(BUTTON_PRIMARY)
        self._btn_gen_report.clicked.connect(
            lambda: self._run_phase("report"))
        pipe_lay.addWidget(self._btn_gen_report)

        pipe_lay.addWidget(self._separator())

        self._btn_full_study = QPushButton(
            f"{EMOJI['run']} Run Full Study")
        self._btn_full_study.setStyleSheet(BUTTON_SUCCESS)
        self._btn_full_study.clicked.connect(
            lambda: self._run_phase("full"))
        pipe_lay.addWidget(self._btn_full_study)

        self._btn_cancel = QPushButton(f"{EMOJI['stop']} Cancel")
        self._btn_cancel.setStyleSheet(BUTTON_DANGER)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_run)
        pipe_lay.addWidget(self._btn_cancel)

        pipe_grp.setContentLayout(pipe_lay)
        lay.addWidget(pipe_grp)

        # ── Progress ────────────────────────────────────────────
        prog_grp = CollapsibleGroupBox(
            f"{EMOJI['info']} Progress", collapsed=False)
        prog_lay = QVBoxLayout()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        prog_lay.addWidget(self._progress_bar)

        self._progress_label = QLabel("Ready")
        self._progress_label.setStyleSheet(SECONDARY_LABEL)
        self._progress_label.setWordWrap(True)
        prog_lay.addWidget(self._progress_label)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        prog_lay.addWidget(self._result_label)

        prog_grp.setContentLayout(prog_lay)
        lay.addWidget(prog_grp)

        lay.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ══════════════════════════════════════════════════════════════
    #  CONFIG <-> WIDGETS
    # ══════════════════════════════════════════════════════════════

    def _build_config(self):
        """Build ComparisonStudyConfig from current widget values."""
        from ...research.config import ComparisonStudyConfig

        cfg = ComparisonStudyConfig()
        cfg.study_name = self._study_name.text()

        # Profiles
        cfg.profiles.scenarios = [
            sc for sc, cb in self._scenario_checks.items() if cb.isChecked()
        ]
        cfg.profiles.n_per_scenario = self._n_per_scenario.value()
        cfg.profiles.n_random = self._n_random.value()
        cfg.profiles.min_depth = self._min_depth.value()
        cfg.profiles.max_depth = self._max_depth.value()
        cfg.profiles.min_vs = self._min_vs.value()
        cfg.profiles.max_vs = self._max_vs.value()
        cfg.profiles.min_layers = self._min_layers.value()
        cfg.profiles.max_layers = self._max_layers.value()
        soilgen = self._soilgen_path.text().strip()
        if soilgen:
            cfg.profiles.soilgen_path = soilgen

        # Engines
        cfg.engines.engines = [
            eng for eng, cb in self._engine_checks.items() if cb.isChecked()
        ]
        cfg.engines.fmin = self._freq_min.value()
        cfg.engines.fmax = self._freq_max.value()
        cfg.engines.n_frequencies = self._n_freq.value()
        cfg.engines.min_prominence = self._min_prominence.value()
        cfg.engines.min_amplitude = self._min_amplitude.value()

        # Metrics
        cfg.metrics.f0_tolerance = self._f0_tolerance.value()
        cfg.metrics.agreement_threshold = self._agreement_threshold.value()
        cfg.metrics.normalize_curves = self._normalize_curves.isChecked()

        # Output
        cfg.output.save_csv = self._save_csv.isChecked()
        cfg.output.save_json = self._save_json.isChecked()
        cfg.output.save_latex = self._save_latex.isChecked()
        cfg.output.save_figures = self._save_figures.isChecked()
        out_dir = self._output_dir.text().strip()
        if out_dir:
            cfg.output.output_dir = out_dir

        # Visualization
        cfg.visualization.figure_format = self._fig_format.currentText()
        cfg.visualization.dpi = self._fig_dpi.value()

        return cfg

    def _get_runner(self):
        """Lazy-create and return the ComparisonStudyRunner."""
        if self._runner is None:
            from ...research.runner import ComparisonStudyRunner
            self._runner = ComparisonStudyRunner()
        # Always push latest config
        cfg = self._build_config()
        self._runner._config = cfg
        return self._runner

    # ══════════════════════════════════════════════════════════════
    #  PIPELINE EXECUTION
    # ══════════════════════════════════════════════════════════════

    def _run_phase(self, phase):
        """Launch the specified pipeline phase in a worker thread."""
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(
                self, "Busy", "A pipeline phase is already running.")
            return

        from ..workers.research_worker import ResearchWorker

        runner = self._get_runner()
        self._worker = ResearchWorker(runner, phase, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.phase_complete.connect(self._on_phase_complete)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._set_running(True)
        self._progress_label.setText(f"Running: {phase}...")
        self._progress_bar.setValue(0)
        self._result_label.setText("")
        self._worker.start()

    def _load_profiles(self):
        """Load a pre-generated profile suite from a directory."""
        d = QFileDialog.getExistingDirectory(
            self, "Select Profile Suite Directory")
        if not d:
            return
        try:
            runner = self._get_runner()
            result = runner.load_profiles(d)
            n = result.get("n_profiles", 0)
            self._progress_label.setText(f"Loaded {n} profiles from {d}")
            self._result_label.setText(f"Profiles ready: {n}")
            if self._mw:
                self._mw.update_research_results("profile_loading", result)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _cancel_run(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._progress_label.setText("Cancelling...")

    # ══════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ══════════════════════════════════════════════════════════════

    def _on_progress(self, current, total, message):
        if total > 0:
            self._progress_bar.setValue(int(100 * current / total))
        self._progress_label.setText(message)

    def _on_phase_complete(self, phase_name, result):
        self._result_label.setText(
            f"{EMOJI['ok']} {phase_name} complete")
        if self._mw:
            self._mw.update_research_results(phase_name, result)
        self.phase_complete.emit(phase_name, result)

    def _on_finished(self, results):
        self._set_running(False)
        self._progress_bar.setValue(100)
        elapsed = results.get("total_elapsed_seconds", 0)
        self._progress_label.setText(
            f"Study complete in {elapsed:.1f}s")

    def _on_error(self, msg):
        self._set_running(False)
        self._progress_label.setText(f"{EMOJI['error']} Error")
        self._result_label.setText(msg)
        QMessageBox.critical(self, "Pipeline Error", msg)

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════

    def _set_running(self, running):
        """Enable/disable UI during pipeline execution."""
        for btn in [self._btn_gen_profiles, self._btn_load_profiles,
                     self._btn_run_comparison, self._btn_compute_metrics,
                     self._btn_field_validation, self._btn_gen_report,
                     self._btn_full_study]:
            btn.setEnabled(not running)
        self._btn_cancel.setEnabled(running)

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_dir.setText(d)

    def _browse_soilgen(self):
        d = QFileDialog.getExistingDirectory(self, "Select SoilGen Directory")
        if d:
            self._soilgen_path.setText(d)

    @staticmethod
    def _separator():
        from PyQt5.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line
