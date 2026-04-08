"""HV Strip Progressive — Main Window (Restructured GUI v3).

Tabbed left panel with two top-level tabs (Forward Model / HV Strip),
each containing sub-tabs (Single/Multiple and Single/Batch).
Context-sensitive center canvas, collapsible right-side log dock.
No toolbar — each sub-tab has its own Run button and settings inline.
"""
import os
import platform
import yaml
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QStatusBar, QLabel, QAction, QMenuBar,
    QComboBox, QPushButton, QFileDialog, QMessageBox,
    QScrollArea, QDockWidget, QStackedWidget,
)
from PyQt5.QtGui import QFont

from .widgets.style_constants import (
    OUTER_MARGINS, LEFT_PANEL_MIN, LEFT_PANEL_MAX,
    BUTTON_PRIMARY, GEAR_BUTTON, SECONDARY_LABEL, EMOJI,
)

_SETTINGS_DIR = Path.home() / ".hvstrip"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.yaml"

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]


def _local_gpell_path() -> str:
    """Return gpell path from local_config, falling back to a generic default."""
    try:
        from ..local_config import GPELL_PATH
        return GPELL_PATH
    except Exception:
        return r"C:\Geopsy.org\bin\gpell.exe"


def _local_git_bash_path() -> str:
    """Return git-bash path from local_config, falling back to a generic default."""
    try:
        from ..local_config import GIT_BASH_PATH
        return GIT_BASH_PATH
    except Exception:
        return r"C:\Program Files\Git\git-bash.exe"


def _default_hvf_exe() -> str:
    """Resolve HVf executable inside core/engines/diffuse_wave_field/."""
    dwf = Path(__file__).resolve().parent / "core" / "engines" / "diffuse_wave_field"
    system = platform.system()
    if system == "Windows":
        p = dwf / "exe_Win" / "HVf.exe"
    else:
        p = dwf / "exe_Linux" / "HVf"
        if not p.exists():
            p = dwf / "exe_Linux" / "HVf_Serial"
    return str(p)


def _get_default_config():
    """Return default config matching core DEFAULT_WORKFLOW_CONFIG."""
    return {
        "engine": {"name": "diffuse_field"},
        "hv_forward": {
            "exe_path": _default_hvf_exe(),
            "fmin": 0.2,
            "fmax": 20.0,
            "nf": 71,
            "nmr": 10,
            "nml": 10,
            "nks": 10,
            "adaptive": {
                "enable": True,
                "max_passes": 2,
                "edge_margin_frac": 0.05,
                "fmax_expand_factor": 2.0,
                "fmin_shrink_factor": 0.5,
                "fmax_limit": 60.0,
                "fmin_limit": 0.05,
            },
        },
        "hv_postprocess": {
            "peak_detection": {"preset": "default"},
            "hv_plot": {
                "x_axis_scale": "log",
                "y_axis_scale": "log",
                "y_compression": 1.5,
                "smoothing": {"enable": True, "window_length": 9, "poly_order": 3},
                "show_bands": True,
                "freq_window_left": 0.3,
                "freq_window_right": 3.0,
                "figure_width": 12,
                "figure_height": 6,
                "dpi": 200,
            },
            "vs_plot": {
                "show": True,
                "annotate_deepest": True,
                "annotate_max_vs": True,
                "annotate_f0": True,
                "figure_width": 6,
                "figure_height": 8,
                "dpi": 200,
            },
            "output": {
                "save_separate": True,
                "save_combined": True,
                "hv_filename": "hv_curve.png",
                "vs_filename": "vs_profile.png",
                "combined_filename": "combined_figure.png",
                "summary_filename": "step_summary.csv",
            },
        },
        "engine_name": "diffuse_field",
        "plot": {
            "dpi": 200,
            "x_axis_scale": "log",
            "y_axis_scale": "log",
        },
        "peak_detection": {
            "preset": "default",
            "method": "find_peaks",
            "select": "leftmost",
        },
        "dual_resonance": {
            "enable": False,
            "separation_ratio_threshold": 1.2,
            "separation_shift_threshold": 0.3,
        },
        "generate_report": True,
        "interactive_mode": False,
        "engine_settings": {
            "diffuse_field": {
                "exe_path": _default_hvf_exe(),
                "fmin": 0.2,
                "fmax": 20.0,
                "nf": 71,
                "nmr": 10,
                "nml": 10,
                "nks": 10,
            },
            "ellipticity": {
                "gpell_path": _local_gpell_path(),
                "git_bash_path": _local_git_bash_path(),
                "fmin": 0.5,
                "fmax": 20.0,
                "n_samples": 500,
                "n_modes": 1,
                "sampling": "log",
                "absolute": True,
                "peak_refinement": False,
                "love_alpha": 0.0,
                "auto_q": False,
                "q_formula": "default",
                "clip_factor": 50.0,
                "timeout": 30,
            },
            "sh_wave": {
                "fmin": 0.1,
                "fmax": 30.0,
                "n_samples": 512,
                "sampling": "log",
                "Dsoil": None,
                "Drock": 0.5,
                "d_tf": 0,
                "darendeli_curvetype": 1,
                "gamma_max": 23.0,
                "f0_search_fmin": None,
                "f0_search_fmax": None,
                "clip_tf": 0.0,
            },
        },
    }


# Sub-tab identifiers
MODE_FWD_SINGLE = "forward_single"
MODE_FWD_MULTI = "forward_multi"
MODE_STRIP_SINGLE = "strip_single"
MODE_STRIP_BATCH = "strip_batch"
MODE_RESEARCH = "research"


class HVStripWindow(QMainWindow):
    """Main window with tabbed left panel, context canvas, log dock."""

    closed = pyqtSignal()

    def __init__(self, parent=None, project_context=None):
        super().__init__(parent)
        self._project_context = project_context
        self.setWindowTitle("HVSR Progressive Layer Stripping Analysis")
        self.resize(1500, 950)
        self.setMinimumSize(1100, 750)

        # State
        self._config = _get_default_config()
        self._last_result = None
        self._last_strip_dir = None
        self._loaded_profile_path = None
        self._active_mode = MODE_FWD_SINGLE

        # Canvas stacks per mode
        self._canvas_stacks = {}  # mode -> QTabWidget
        self._panels = {}         # mode -> panel widget

        self._load_settings()
        self._build_menu_bar()
        self._build_ui()
        self._build_log_dock()
        self._build_summary_dock()
        self._build_status_bar()
        self._apply_saved_config()

        if project_context:
            self._apply_project_context()

    # ══════════════════════════════════════════════════════════════
    #  PROJECT INTEGRATION
    # ══════════════════════════════════════════════════════════════

    def _apply_project_context(self):
        """Apply project context — set output dir, update title, restore state."""
        ctx = self._project_context
        if not ctx:
            return
        project = ctx.get('project')
        profile_id = ctx.get('profile_id', '')
        if not project:
            return

        self.setWindowTitle(
            f"HV Strip — {project.name} — {profile_id}")
        profile_dir = project.ensure_module_dir('hv_strip', profile_id)
        self._last_strip_dir = str(profile_dir)

        # Restore saved state
        try:
            from hvsr_pro.packages.project_manager.module_state.hvstrip_state_io import (
                has_hvstrip_state, load_hvstrip_state,
            )
            if has_hvstrip_state(profile_dir):
                state = load_hvstrip_state(profile_dir)

                # Restore config (deep-merge onto defaults)
                saved_cfg = state.get("config", {})
                if saved_cfg:
                    self._deep_merge_config(self._config, saved_cfg)
                    self._apply_saved_config()

                # Restore extra metadata
                extra = state.get("extra", {})
                if extra.get("loaded_profile_path"):
                    self._loaded_profile_path = extra["loaded_profile_path"]
                if extra.get("active_mode"):
                    try:
                        self._switch_mode(extra["active_mode"])
                    except Exception:
                        pass

                # Restore results
                results = state.get("results")
                if results is not None:
                    self._last_result = results
                    self.update_strip_results(results)

                self.statusBar().showMessage(
                    f"Restored strip state: {profile_id}", 5000)
        except Exception as e:
            self.statusBar().showMessage(
                f"Could not restore state: {e}", 5000)

    def _deep_merge_config(self, base, override):
        """Recursively merge *override* dict into *base* dict in-place."""
        for key, val in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                self._deep_merge_config(base[key], val)
            else:
                base[key] = val

    def _save_project_state(self):
        """Persist state into the project folder on close."""
        ctx = self._project_context
        if not ctx:
            return
        try:
            from hvsr_pro.packages.project_manager.module_state.hvstrip_state_io import (
                save_hvstrip_state,
            )
            project = ctx['project']
            profile_id = ctx.get('profile_id', 'profile_001')
            profile_dir = project.ensure_module_dir('hv_strip', profile_id)

            save_hvstrip_state(
                profile_dir,
                config=self._config,
                results=self._last_result,
                extra={
                    "loaded_profile_path": self._loaded_profile_path,
                    "active_mode": self._active_mode,
                    "last_strip_dir": self._last_strip_dir,
                },
            )
            project.log_activity('hv_strip', f'State saved for {profile_id}')
            project.save()
        except Exception:
            pass  # best-effort

    # ══════════════════════════════════════════════════════════════
    #  MENU BAR  (no toolbar)
    # ══════════════════════════════════════════════════════════════
    def _build_menu_bar(self):
        mb = self.menuBar()

        # ── File ────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")
        file_menu.addAction("Load Profile...", self._on_load_profile, "Ctrl+O")
        file_menu.addAction("Load Dinver Output...", self._on_load_dinver)
        file_menu.addAction("Load Directory...", self._on_load_batch_folder)
        file_menu.addSeparator()
        file_menu.addAction("Settings...", self._on_open_settings, "Ctrl+,")
        file_menu.addSeparator()
        file_menu.addAction("Export Results...", self._on_export, "Ctrl+E")
        file_menu.addSeparator()
        file_menu.addAction("Close", self.close, "Ctrl+W")

        # ── View ────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")
        self._act_log = QAction("Show Log Panel", self, checkable=True, checked=True)
        self._act_log.setShortcut("Ctrl+L")
        self._act_log.toggled.connect(self._toggle_log_dock)
        view_menu.addAction(self._act_log)

        self._act_summary = QAction("Show Summary Panel", self, checkable=True, checked=True)
        self._act_summary.setShortcut("Ctrl+Shift+S")
        self._act_summary.toggled.connect(self._toggle_summary_dock)
        view_menu.addAction(self._act_summary)

        self._act_strip_results = QAction(
            "Show Strip Results", self, checkable=True, checked=True)
        self._act_strip_results.setShortcut("Ctrl+Shift+R")
        self._act_strip_results.toggled.connect(self._toggle_strip_results_dock)
        view_menu.addAction(self._act_strip_results)

        self._act_research_results = QAction(
            "Show Research Results", self, checkable=True, checked=True)
        self._act_research_results.setShortcut("Ctrl+Shift+C")
        self._act_research_results.toggled.connect(self._toggle_research_dock)
        view_menu.addAction(self._act_research_results)

        view_menu.addAction("Reset Layout", self._reset_layout)

        # ── Tools ───────────────────────────────────────────────
        tools_menu = mb.addMenu("&Tools")
        tools_menu.addAction("Figure Studio...", self._on_open_figure_studio)
        tools_menu.addAction("Interactive Peak Picker...", self._on_open_peak_picker)
        tools_menu.addAction("Profile Loader...", self._on_open_profile_loader)
        tools_menu.addSeparator()
        tools_menu.addAction("Engine Settings...", self._on_engine_settings)

        # ── Help ────────────────────────────────────────────────
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("About...", self._on_about)

    # ══════════════════════════════════════════════════════════════
    #  CENTRAL UI — Left panel (top-level tabs) + Center canvas
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(*OUTER_MARGINS)
        root.setSpacing(4)

        self.splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.splitter)

        # ── Left Panel (scrollable) ────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(LEFT_PANEL_MIN)
        left_scroll.setMaximumWidth(LEFT_PANEL_MAX)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # ── Top-level tab widget (Forward Model | HV Strip) ───
        self._main_tabs = QTabWidget()
        self._main_tabs.setTabPosition(QTabWidget.North)
        self._main_tabs.setDocumentMode(True)

        # ── Forward Model tab (contains Single/Multiple sub-tabs) ──
        self._fwd_tabs = QTabWidget()
        self._fwd_tabs.setTabPosition(QTabWidget.North)
        self._fwd_tabs.setDocumentMode(True)
        self._fwd_tabs.currentChanged.connect(
            lambda idx: self._on_subtab_changed("forward", idx))

        self._fwd_single_panel = self._create_panel(MODE_FWD_SINGLE,
                                                      "Forward Single")
        self._fwd_tabs.addTab(self._fwd_single_panel, "Single")

        self._fwd_multi_panel = self._create_panel(MODE_FWD_MULTI,
                                                     "Forward Multiple")
        self._fwd_tabs.addTab(self._fwd_multi_panel, "Multiple")

        self._main_tabs.addTab(self._fwd_tabs, "Forward Model")

        # ── HV Strip tab (contains Single/Batch sub-tabs) ─────
        self._strip_tabs = QTabWidget()
        self._strip_tabs.setTabPosition(QTabWidget.North)
        self._strip_tabs.setDocumentMode(True)
        self._strip_tabs.currentChanged.connect(
            lambda idx: self._on_subtab_changed("strip", idx))

        self._strip_single_panel = self._create_panel(MODE_STRIP_SINGLE,
                                                        "Strip Single")
        self._strip_tabs.addTab(self._strip_single_panel, "Single")

        self._strip_batch_panel = self._create_panel(MODE_STRIP_BATCH,
                                                       "Strip Batch")
        self._strip_tabs.addTab(self._strip_batch_panel, "Batch")

        self._main_tabs.addTab(self._strip_tabs, "HV Strip")

        # ── Research tab (single panel, no sub-tabs) ───────────
        self._research_panel = self._create_panel(MODE_RESEARCH,
                                                    "Research")
        self._main_tabs.addTab(self._research_panel, "Research")

        # Switch canvas when the top-level tab changes
        self._main_tabs.currentChanged.connect(self._on_main_tab_changed)

        left_layout.addWidget(self._main_tabs)
        left_scroll.setWidget(left_widget)
        self.splitter.addWidget(left_scroll)

        # ── Center: Canvas Stack ───────────────────────────────
        self._canvas_stack = QStackedWidget()

        for mode in [MODE_FWD_SINGLE, MODE_FWD_MULTI,
                     MODE_STRIP_SINGLE, MODE_STRIP_BATCH,
                     MODE_RESEARCH]:
            canvas = self._create_canvas(mode)
            self._canvas_stacks[mode] = canvas
            self._canvas_stack.addWidget(canvas)

        self.splitter.addWidget(self._canvas_stack)

        # Connect wizard signals (after all canvases are created)
        self._connect_wizard_signals()

        # ── Right: Dock toggle strip ──────────────────────────
        self._dock_strip = QWidget()
        self._dock_strip.setFixedWidth(22)
        ds_lay = QVBoxLayout(self._dock_strip)
        ds_lay.setContentsMargins(0, 0, 0, 0)
        ds_lay.setSpacing(2)
        ds_lay.addStretch()

        self._btn_toggle_dock = QPushButton("◀")
        self._btn_toggle_dock.setFixedSize(20, 60)
        self._btn_toggle_dock.setToolTip("Toggle right panels")
        self._btn_toggle_dock.setStyleSheet(
            "QPushButton { font-size: 10px; border: 1px solid #ccc; "
            "border-radius: 3px; background: #f5f5f5; }"
            "QPushButton:hover { background: #e0e0e0; }")
        self._btn_toggle_dock.clicked.connect(self._toggle_right_docks)
        ds_lay.addWidget(self._btn_toggle_dock)

        ds_lay.addStretch()
        self.splitter.addWidget(self._dock_strip)

        # Splitter sizing: left fixed, center stretches, dock strip fixed
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([420, 1060, 22])

        self.setCentralWidget(central)

        # Start with Forward Single active
        self._switch_mode(MODE_FWD_SINGLE)

    def _connect_wizard_signals(self):
        """Connect StripWizardView.wizard_finished to result handling."""
        wiz = self._find_view_in_mode(MODE_STRIP_SINGLE, "StripWizardView")
        if wiz and hasattr(wiz, 'wizard_finished'):
            wiz.wizard_finished.connect(self._on_wizard_finished)

    def _find_view_in_mode(self, mode, class_name):
        """Find a specific view widget in a mode's canvas stack."""
        canvas = self._canvas_stacks.get(mode)
        if canvas:
            for i in range(canvas.count()):
                w = canvas.widget(i)
                if type(w).__name__ == class_name:
                    return w
        return None

    def _on_wizard_finished(self, peak_data):
        """Handle wizard Finish: persist peaks, update dock, figure studio, report."""
        # Write modified peaks back to disk summary CSVs
        self._persist_peak_data(peak_data)

        # Update in-memory step_results
        result = getattr(self, '_last_result', None) or {}
        sr = result.get("step_results", {})
        for step_name, pdata in peak_data.items():
            f0 = pdata.get("f0")
            if f0 and step_name in sr:
                sr[step_name]["peak_frequency"] = f0[0]
                sr[step_name]["peak_amplitude"] = f0[1]

        # Update strip summary dock with peak data
        dock = self.get_strip_summary_dock()
        if dock and hasattr(dock, 'set_results'):
            dock.set_results(result, peak_data=peak_data)
            dock.setVisible(True)
            dock.raise_()

        # Re-initialize figure studio reporter to pick up updated CSV files
        fs = self.get_figure_studio()
        if fs and hasattr(fs, '_init_reporter'):
            fs._init_reporter()
            # Push wizard peaks for dual-resonance figure
            if hasattr(fs, 'set_wizard_peaks'):
                # Build peak_overrides: step_name → (freq, amp)
                overrides = {}
                for step_name, pdata in peak_data.items():
                    f0 = pdata.get("f0")
                    if f0 and len(f0) >= 2:
                        overrides[step_name] = (f0[0], f0[1])
                fs.set_wizard_peaks(overrides)
            elif hasattr(fs, '_draw_current'):
                fs._draw_current()

        # Regenerate report if panel checkbox is checked
        panel = getattr(self, '_strip_single_panel', None)
        if panel and hasattr(panel, 'is_report_enabled'):
            if panel.is_report_enabled():
                self._regenerate_strip_report()

        self.log("Wizard finished — peak data persisted and report updated")

    def _persist_peak_data(self, peak_data):
        """Write wizard-selected peaks back to step summary CSVs on disk.

        The report generator reads ``Peak_Frequency_Hz`` and ``Peak_Amplitude``
        from these summary files, so updating them ensures the regenerated
        report reflects the user's peak selections.
        Also persists Vs30/VsAvg/bedrock data to ``vs_results.json``.
        """
        import csv as _csv
        result = getattr(self, '_last_result', None)
        if not result:
            return
        strip_dir = result.get("strip_directory")
        if not strip_dir:
            return
        strip_path = Path(strip_dir)
        
        vs_data = {}
        
        for step_name, pdata in peak_data.items():
            f0 = pdata.get("f0")
            if not f0:
                continue
            step_folder = strip_path / step_name
            if not step_folder.is_dir():
                continue
            # Find existing summary CSV(s)
            summary_files = list(step_folder.glob('*summary*.csv'))
            if summary_files:
                for sf in summary_files:
                    self._update_summary_csv(sf, f0[0], f0[1])
            else:
                # Create a minimal summary CSV so the reporter picks it up
                sf = step_folder / 'step_summary.csv'
                self._create_minimal_summary_csv(sf, step_name, f0[0], f0[1])
            
            # Collect Vs data for persistence
            vs30 = pdata.get("vs30")
            vsavg = pdata.get("vsavg")
            bedrock_depth = pdata.get("bedrock_depth")
            if vs30 or vsavg or bedrock_depth:
                vs_data[step_name] = {
                    "vs30": vs30,
                    "vsavg": vsavg,
                    "bedrock_depth": bedrock_depth
                }
        
        if vs_data:
            self._persist_vs_data(strip_path, vs_data)
        
        self.log(f"Peak data written to {len(peak_data)} step folders")

    def _persist_vs_data(self, strip_path: Path, vs_data: dict):
        """Write vs_results.json alongside step folders for report regeneration."""
        import json
        vs_path = strip_path / 'vs_results.json'
        try:
            with open(vs_path, 'w') as f:
                json.dump(vs_data, f, indent=2, default=str)
            self.log(f"Vs data saved to {vs_path.name}")
        except Exception as e:
            self.log(f"Failed to save vs_data: {e}")

    @staticmethod
    def _update_summary_csv(csv_path, peak_freq, peak_amp):
        """Update Peak_Frequency_Hz and Peak_Amplitude in an existing CSV."""
        import csv as _csv
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = _csv.reader(f)
                rows = list(reader)
            if len(rows) < 2:
                return
            header = rows[0]
            # Find column indices
            freq_col = None
            amp_col = None
            for i, h in enumerate(header):
                if h.strip() == 'Peak_Frequency_Hz':
                    freq_col = i
                elif h.strip() == 'Peak_Amplitude':
                    amp_col = i
            if freq_col is not None and amp_col is not None:
                rows[1][freq_col] = f'{peak_freq:.6f}'
                rows[1][amp_col] = f'{peak_amp:.6f}'
                with open(csv_path, 'w', newline='') as f:
                    writer = _csv.writer(f)
                    writer.writerows(rows)
        except Exception:
            pass

    @staticmethod
    def _create_minimal_summary_csv(csv_path, step_name, peak_freq, peak_amp):
        """Create a minimal summary CSV for the report generator to read."""
        import csv as _csv
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = _csv.writer(f)
                writer.writerow(['Step', 'N_Finite_Layers',
                                 'Peak_Frequency_Hz', 'Peak_Amplitude'])
                parts = step_name.split('_')
                step_num = parts[0].replace('Step', '')
                n_layers = parts[1].split('-')[0] if len(parts) > 1 else '?'
                writer.writerow([f'Step{step_num}', n_layers,
                                 f'{peak_freq:.6f}', f'{peak_amp:.6f}'])
        except Exception:
            pass

    def _regenerate_strip_report(self):
        """Regenerate comprehensive report (reads updated CSVs + vs_results.json from disk)."""
        result = getattr(self, '_last_result', None)
        if not result:
            return
        strip_dir = result.get("strip_directory")
        if not strip_dir:
            return
        try:
            from ..core.report_generator import ProgressiveStrippingReporter
            reporter = ProgressiveStrippingReporter(
                strip_dir, output_dir=str(Path(strip_dir).parent))
            reporter.generate_comprehensive_report()
            self.log("Report regenerated with updated peaks and Vs data")
        except Exception as e:
            self.log(f"Report regen error: {e}")

    # ── Panel creation (placeholder for now, real panels in phase 2+) ──
    def _create_panel(self, mode, label):
        """Create a left-panel widget for the given mode.

        Tries to import the real panel class; falls back to a placeholder.
        """
        panel_map = {
            MODE_FWD_SINGLE: (".panels.forward_single_panel", "ForwardSinglePanel"),
            MODE_FWD_MULTI: (".panels.forward_multi_panel", "ForwardMultiPanel"),
            MODE_STRIP_SINGLE: (".panels.strip_single_panel", "StripSinglePanel"),
            MODE_STRIP_BATCH: (".panels.strip_batch_panel", "StripBatchPanel"),
            MODE_RESEARCH: (".panels.research_panel", "ResearchPanel"),
        }
        mod_path, cls_name = panel_map.get(mode, (None, None))
        if mod_path:
            try:
                import importlib
                mod = importlib.import_module(mod_path, package=__package__)
                cls = getattr(mod, cls_name)
                panel = cls(main_window=self)
                self._panels[mode] = panel
                return panel
            except Exception as e:
                print(f"[HVStrip] {cls_name} not ready: {e}")

        # Placeholder
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lbl = QLabel(f"{label}\n\n(Panel will be built in next phase)")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #888; font-size: 13px;")
        lay.addWidget(lbl)
        self._panels[mode] = w
        return w

    # ── Canvas creation (placeholder tabs per mode) ────────────
    def _create_canvas(self, mode):
        """Create a QTabWidget with the correct canvas tabs for the mode."""
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMinimumHeight(500)

        tab_defs = {
            MODE_FWD_SINGLE: [
                ("data_input_single", f"{EMOJI['file']} Data Input"),
                ("hv_curve", f"{EMOJI['forward']} HV Curve"),
                ("vs_profile", f"{EMOJI['profile']} Vs Profile"),
            ],
            MODE_FWD_MULTI: [
                ("data_input_multi", f"{EMOJI['file']} Data Input"),
                ("profile_wizard", f"{EMOJI['forward']} Profile Wizard"),
                ("all_profiles", f"{EMOJI['chart']} All Profiles"),
            ],
            MODE_STRIP_SINGLE: [
                ("data_input_single", f"{EMOJI['file']} Data Input"),
                ("strip_wizard", f"{EMOJI['forward']} HV Strip Wizard"),
                ("figure_studio_view", f"{EMOJI['figures']} Figure Studio"),
            ],
            MODE_STRIP_BATCH: [
                ("data_input_batch", f"{EMOJI['file']} Data Input"),
                ("progress", f"{EMOJI['batch']} Progress"),
                ("profile_results", f"{EMOJI['report']} Profile Results"),
                ("figure_studio", f"{EMOJI['figures']} Figure Studio"),
                ("summary", f"{EMOJI['chart']} Summary"),
            ],
            MODE_RESEARCH: [
                ("research_config", f"{EMOJI['settings']} Configuration"),
                ("research_profiles", f"{EMOJI['profile']} Profiles"),
                ("research_comparison", f"{EMOJI['forward']} Comparison"),
                ("research_figures", f"{EMOJI['figures']} Figures"),
                ("research_report", f"{EMOJI['report']} Report"),
            ],
        }

        for tab_id, tab_label in tab_defs.get(mode, []):
            view = self._create_canvas_view(mode, tab_id)
            tabs.addTab(view, tab_label)

        return tabs

    def _create_canvas_view(self, mode, tab_id):
        """Create a single canvas view widget. Tries real views first."""
        # Data input views
        if tab_id == "data_input_single":
            try:
                from .views.data_input_view import DataInputView
                return DataInputView(main_window=self)
            except Exception as e:
                print(f"[HVStrip] DataInputView: {e}")
                return self._placeholder("Data Input")
        if tab_id == "data_input_multi":
            try:
                from .views.multi_input_view import MultiInputView
                return MultiInputView(main_window=self)
            except Exception as e:
                print(f"[HVStrip] MultiInputView: {e}")
                return self._placeholder("Multi Input")
        if tab_id == "data_input_batch":
            try:
                from .views.batch_input_view import BatchInputView
                return BatchInputView(main_window=self)
            except Exception as e:
                print(f"[HVStrip] BatchInputView: {e}")
                return self._placeholder("Batch Input")

        # Map tab_id to existing view classes
        view_map = {
            "hv_curve": (".views.hv_curve_view", "HVCurveView"),
            "vs_profile": (".views.vs_profile_view", "VsProfileView"),
            "hv_overlay": (".views.hv_overlay_view", "HVOverlayView"),
            "strip_results": (".views.strip_results_view", "StripResultsView"),
            "summary_table": (".views.summary_table_view", "SummaryTableView"),
            "profile_wizard": (".views.profile_wizard_view", "ProfileWizardView"),
            "all_profiles": (".views.all_profiles_view", "AllProfilesView"),
            "strip_wizard": (".views.strip_wizard_view", "StripWizardView"),
            "figure_studio_view": (".views.figure_studio_view", "FigureStudioView"),
            # Research views
            "research_config": (".views.research_config_view", "ResearchConfigView"),
            "research_profiles": (".views.research_profiles_view", "ResearchProfilesView"),
            "research_comparison": (".views.research_comparison_view", "ResearchComparisonView"),
            "research_figures": (".views.research_figures_view", "ResearchFiguresView"),
            "research_report": (".views.research_report_view", "ResearchReportView"),
        }
        # Some tab_ids re-use existing views
        if tab_id in ("current_profile",):
            view_map[tab_id] = view_map.get("hv_curve", (None, None))
        if tab_id in ("all_overlay", "median_hv"):
            view_map[tab_id] = view_map.get("hv_overlay", (None, None))

        mod_path, cls_name = view_map.get(tab_id, (None, None))
        if mod_path and cls_name:
            try:
                import importlib
                mod = importlib.import_module(mod_path, package=__package__)
                cls = getattr(mod, cls_name)
                return cls(main_window=self)
            except Exception as e:
                print(f"[HVStrip] Canvas {tab_id}: {e}")

        # Placeholder
        return self._placeholder(f"{tab_id.replace('_', ' ').title()}")

    # ══════════════════════════════════════════════════════════════
    #  LOG DOCK (right side, collapsible, hidden by default)
    # ══════════════════════════════════════════════════════════════
    def _build_log_dock(self):
        self._log_dock = QDockWidget("Analysis Log", self)
        self._log_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self._log_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)

        try:
            from .views.log_view import LogView
            self._log_view = LogView(main_window=self)
        except Exception as e:
            self._log_view = self._placeholder("Log")
            print(f"[HVStrip] Log view error: {e}")

        self._log_dock.setWidget(self._log_view)
        self.addDockWidget(Qt.RightDockWidgetArea, self._log_dock)
        self._log_dock.setVisible(True)  # visible by default

        # Sync dock visibility with menu action
        self._log_dock.visibilityChanged.connect(self._on_log_dock_vis)

    def _on_log_dock_vis(self, visible):
        if hasattr(self, '_act_log'):
            self._act_log.blockSignals(True)
            self._act_log.setChecked(visible)
            self._act_log.blockSignals(False)

    def _toggle_log_dock(self, show):
        self._log_dock.setVisible(show)

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY DOCK (right side, tabified with log dock)
    # ══════════════════════════════════════════════════════════════
    def _build_summary_dock(self):
        try:
            from .widgets.summary_dock import SummaryDockWidget
            self._summary_dock = SummaryDockWidget(self)
        except Exception as e:
            print(f"[HVStrip] SummaryDock error: {e}")
            self._summary_dock = QDockWidget("Summary", self)
            self._summary_dock.setWidget(self._placeholder("Summary"))

        self.addDockWidget(Qt.RightDockWidgetArea, self._summary_dock)
        # Tabify with log dock so user can switch between them
        self.tabifyDockWidget(self._log_dock, self._summary_dock)
        self._summary_dock.setVisible(True)  # visible by default

        self._summary_dock.visibilityChanged.connect(self._on_summary_dock_vis)

        # Strip summary dock (for MODE_STRIP_SINGLE)
        try:
            from .widgets.strip_summary_dock import StripSummaryDock
            self._strip_summary_dock = StripSummaryDock(self)
        except Exception as e:
            print(f"[HVStrip] StripSummaryDock error: {e}")
            self._strip_summary_dock = QDockWidget("Strip Results", self)
            self._strip_summary_dock.setWidget(
                self._placeholder("Strip Results"))

        self.addDockWidget(Qt.RightDockWidgetArea, self._strip_summary_dock)
        self.tabifyDockWidget(self._summary_dock, self._strip_summary_dock)
        self._strip_summary_dock.setVisible(True)  # visible by default

        self._strip_summary_dock.visibilityChanged.connect(
            self._on_strip_results_dock_vis)

        # Research results dock
        try:
            from .widgets.research_results_dock import ResearchResultsDock
            self._research_dock = ResearchResultsDock(self)
        except Exception as e:
            print(f"[HVStrip] ResearchResultsDock error: {e}")
            self._research_dock = QDockWidget("Research Results", self)
            self._research_dock.setWidget(
                self._placeholder("Research Results"))

        self._research_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self._research_dock)
        self.tabifyDockWidget(self._strip_summary_dock, self._research_dock)
        self._research_dock.setVisible(False)  # hidden until Research mode

        self._research_dock.visibilityChanged.connect(
            self._on_research_dock_vis)

    def _on_strip_results_dock_vis(self, visible):
        if hasattr(self, '_act_strip_results'):
            self._act_strip_results.blockSignals(True)
            self._act_strip_results.setChecked(visible)
            self._act_strip_results.blockSignals(False)

    def _toggle_strip_results_dock(self, show):
        if hasattr(self, '_strip_summary_dock'):
            self._strip_summary_dock.setVisible(show)

    def _on_summary_dock_vis(self, visible):
        if hasattr(self, '_act_summary'):
            self._act_summary.blockSignals(True)
            self._act_summary.setChecked(visible)
            self._act_summary.blockSignals(False)

    def _toggle_summary_dock(self, show):
        self._summary_dock.setVisible(show)

    def _on_research_dock_vis(self, visible):
        if hasattr(self, '_act_research_results'):
            self._act_research_results.blockSignals(True)
            self._act_research_results.setChecked(visible)
            self._act_research_results.blockSignals(False)

    def _toggle_research_dock(self, show):
        if hasattr(self, '_research_dock'):
            self._research_dock.setVisible(show)

    def _toggle_right_docks(self):
        """Toggle all right docks visible/hidden."""
        log_vis = self._log_dock.isVisible()
        sum_vis = self._summary_dock.isVisible()
        strip_vis = getattr(self, '_strip_summary_dock', None) and \
            self._strip_summary_dock.isVisible()
        research_vis = getattr(self, '_research_dock', None) and \
            self._research_dock.isVisible()
        if log_vis or sum_vis or strip_vis or research_vis:
            self._log_dock.setVisible(False)
            self._summary_dock.setVisible(False)
            if hasattr(self, '_strip_summary_dock'):
                self._strip_summary_dock.setVisible(False)
            if hasattr(self, '_research_dock'):
                self._research_dock.setVisible(False)
            self._btn_toggle_dock.setText("◀")
            self._btn_toggle_dock.setToolTip("Expand right panels")
        else:
            self._log_dock.setVisible(True)
            self._log_dock.raise_()
            self._btn_toggle_dock.setText("▶")
            self._btn_toggle_dock.setToolTip("Collapse right panels")

    def _raise_log_dock(self):
        """Show and raise the Log dock."""
        self._log_dock.setVisible(True)
        self._log_dock.raise_()
        self._btn_toggle_dock.setText("▶")

    def _raise_summary_dock(self):
        """Show and raise the Summary dock."""
        self._summary_dock.setVisible(True)
        self._summary_dock.raise_()
        self._btn_toggle_dock.setText("▶")

    # ══════════════════════════════════════════════════════════════
    #  STATUS BAR
    # ══════════════════════════════════════════════════════════════
    def _build_status_bar(self):
        sb = QStatusBar()
        self._status_msg = QLabel("Ready")
        self._status_engine = QLabel("Engine: diffuse_field")
        self._status_profile = QLabel("Profile: —")
        self._status_result = QLabel("")
        self._status_mode = QLabel(f"Mode: Forward Single")

        sb.addWidget(self._status_msg, 1)
        sb.addPermanentWidget(self._status_mode)
        sb.addPermanentWidget(self._status_profile)
        sb.addPermanentWidget(self._status_engine)
        sb.addPermanentWidget(self._status_result)
        self.setStatusBar(sb)
        sb.showMessage("Ready — select a mode and load a soil profile", 5000)

    # ══════════════════════════════════════════════════════════════
    #  MODE SWITCHING
    # ══════════════════════════════════════════════════════════════
    def _on_main_tab_changed(self, idx):
        """Handle top-level tab change (Forward Model ↔ HV Strip ↔ Research)."""
        if not hasattr(self, '_canvas_stack'):
            return
        if idx == 0:  # Forward Model
            sub_idx = self._fwd_tabs.currentIndex()
            mode = MODE_FWD_SINGLE if sub_idx == 0 else MODE_FWD_MULTI
        elif idx == 1:  # HV Strip
            sub_idx = self._strip_tabs.currentIndex()
            mode = MODE_STRIP_SINGLE if sub_idx == 0 else MODE_STRIP_BATCH
        else:  # Research
            mode = MODE_RESEARCH
        self._switch_mode(mode)

    def _on_subtab_changed(self, section, idx):
        """Handle sub-tab change in Forward or Strip section."""
        if not hasattr(self, '_canvas_stack'):
            return  # Guard: called during init before canvas is built
        if section == "forward":
            mode = MODE_FWD_SINGLE if idx == 0 else MODE_FWD_MULTI
        else:
            mode = MODE_STRIP_SINGLE if idx == 0 else MODE_STRIP_BATCH
        self._switch_mode(mode)

    def _switch_mode(self, mode):
        """Switch the center canvas to match the active left-panel mode."""
        self._active_mode = mode
        # Find the stacked index for this mode
        modes = [MODE_FWD_SINGLE, MODE_FWD_MULTI,
                 MODE_STRIP_SINGLE, MODE_STRIP_BATCH,
                 MODE_RESEARCH]
        if mode in modes:
            self._canvas_stack.setCurrentIndex(modes.index(mode))

        # Show/hide strip results dock based on mode
        if hasattr(self, '_strip_summary_dock'):
            is_strip = (mode == MODE_STRIP_SINGLE)
            self._strip_summary_dock.setVisible(is_strip)
            if is_strip:
                self._strip_summary_dock.raise_()

        # In strip single mode, show strip results instead of summary
        if hasattr(self, '_summary_dock'):
            if mode == MODE_STRIP_SINGLE:
                self._summary_dock.setVisible(False)
            elif hasattr(self, '_act_summary') and self._act_summary.isChecked():
                self._summary_dock.setVisible(True)

        # Show/hide research results dock
        if hasattr(self, '_research_dock'):
            is_research = (mode == MODE_RESEARCH)
            self._research_dock.setVisible(
                is_research and self._act_research_results.isChecked())
            if is_research:
                self._research_dock.raise_()

        # Update status bar
        labels = {
            MODE_FWD_SINGLE: "Forward Single",
            MODE_FWD_MULTI: "Forward Multiple",
            MODE_STRIP_SINGLE: "Strip Single",
            MODE_STRIP_BATCH: "Strip Batch",
            MODE_RESEARCH: "Research",
        }
        if hasattr(self, '_status_mode'):
            self._status_mode.setText(f"Mode: {labels.get(mode, mode)}")

    # ══════════════════════════════════════════════════════════════
    #  MENU ACTIONS
    # ══════════════════════════════════════════════════════════════
    def _on_load_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Soil Profile", "",
            "All Supported (*.txt *.csv *.xlsx);;Text Files (*.txt);;"
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)")
        if path:
            self._loaded_profile_path = path
            self._status_profile.setText(f"Profile: {Path(path).name}")
            # Delegate to canvas data input view
            di = self.get_data_input()
            if di and hasattr(di, 'load_profile'):
                di.load_profile(path)
            self.log(f"Loaded profile: {path}")

    def _on_load_dinver(self):
        panel = self._panels.get(self._active_mode)
        if panel and hasattr(panel, 'load_dinver'):
            panel.load_dinver()
        else:
            vs_path, _ = QFileDialog.getOpenFileName(
                self, "Load Dinver Vs File", "",
                "Dinver Vs (*_vs.txt);;All Files (*)")
            if vs_path:
                self._loaded_profile_path = vs_path
                self._status_profile.setText(
                    f"Profile: {Path(vs_path).name}")
                self.log(f"Loaded Dinver: {vs_path}")

    def _on_load_batch_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            panel = self._panels.get(self._active_mode)
            if panel and hasattr(panel, 'set_batch_folder'):
                panel.set_batch_folder(d)
            self.log(f"Folder: {d}")

    def _on_export(self):
        if self._last_strip_dir:
            d = QFileDialog.getExistingDirectory(self, "Export Results To")
            if d:
                self.log(f"Exporting to: {d}")
                self._on_open_figure_studio()
        else:
            QMessageBox.information(
                self, "Export", "No results to export. Run an analysis first.")

    def _on_open_settings(self):
        """Open the global Settings window."""
        try:
            from .dialogs.settings_window import SettingsWindow
            dlg = SettingsWindow(self._config, parent=self)
            if dlg.exec_() == SettingsWindow.Accepted:
                new_cfg = dlg.get_config()
                self.update_config(new_cfg)
                self.log("Settings updated")
        except ImportError:
            # Settings window not yet built — fallback to engine settings
            self._on_engine_settings()
        except Exception as e:
            QMessageBox.warning(self, "Settings Error", str(e))

    def _on_open_figure_studio(self):
        if not self._last_strip_dir or not os.path.isdir(self._last_strip_dir):
            QMessageBox.information(
                self, "Figure Studio",
                "No stripping results available.\n"
                "Run a stripping workflow first.")
            return
        try:
            from .dialogs.figure_studio import FigureStudioWindow
            dlg = FigureStudioWindow(
                self._last_strip_dir,
                has_dual_resonance=self._config.get(
                    "dual_resonance", {}).get("enable", False),
                parent=self)
            dlg.show()
        except Exception as e:
            QMessageBox.warning(self, "Figure Studio Error", str(e))

    def _on_open_peak_picker(self):
        if not self._last_result:
            QMessageBox.information(
                self, "Peak Picker",
                "No results to pick peaks from.\nRun an analysis first.")
            return
        try:
            from .dialogs.peak_picker_dialog import PeakPickerDialog
            dlg = PeakPickerDialog(self._last_result, parent=self)
            dlg.exec_()
        except Exception as e:
            QMessageBox.warning(self, "Peak Picker Error", str(e))

    def _on_open_profile_loader(self):
        try:
            from .dialogs.profile_loader_dialog import ProfileLoaderDialog
            dlg = ProfileLoaderDialog(parent=self)
            if dlg.exec_() == ProfileLoaderDialog.Accepted:
                data = dlg.get_data()
                if data:
                    self._loaded_profile_path = data.get("path")
                    if self._loaded_profile_path:
                        self._status_profile.setText(
                            f"Profile: {Path(self._loaded_profile_path).name}")
                    panel = self._panels.get(self._active_mode)
                    if panel and hasattr(panel, 'set_profile_data'):
                        panel.set_profile_data(data)
        except Exception as e:
            QMessageBox.warning(self, "Profile Loader Error", str(e))

    def _on_engine_settings(self):
        try:
            from .dialogs.engine_settings_dialog import EngineSettingsDialog
            current = self._config.get("engine_settings", {})
            dlg = EngineSettingsDialog(current, parent=self)
            if dlg.exec_() == EngineSettingsDialog.Accepted:
                self.update_config({"engine_settings": dlg.get_config()})
                self.log("Engine settings updated")
        except Exception as e:
            QMessageBox.warning(self, "Engine Settings Error", str(e))

    def _on_about(self):
        QMessageBox.about(
            self, "About HV Strip Progressive",
            "<b>HVSR Progressive Layer Stripping Analysis</b><br>"
            "Version 3.0<br><br>"
            "Two-section GUI with Forward Model and HV Strip modes.<br>"
            "Supports diffuse wavefield, SH-wave transfer function, "
            "and Rayleigh wave ellipticity engines.")

    def _reset_layout(self):
        self.splitter.setSizes([420, 1080])
        self._log_dock.setVisible(True)
        self._act_log.setChecked(True)
        if hasattr(self, '_summary_dock'):
            self._summary_dock.setVisible(True)
        if hasattr(self, '_act_summary'):
            self._act_summary.setChecked(True)
        if hasattr(self, '_strip_summary_dock'):
            self._strip_summary_dock.setVisible(
                self._active_mode == MODE_STRIP_SINGLE)
        if hasattr(self, '_act_strip_results'):
            self._act_strip_results.setChecked(
                self._active_mode == MODE_STRIP_SINGLE)
        if hasattr(self, '_research_dock'):
            self._research_dock.setVisible(
                self._active_mode == MODE_RESEARCH)
        if hasattr(self, '_act_research_results'):
            self._act_research_results.setChecked(
                self._active_mode == MODE_RESEARCH)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API  (used by panels, views, dialogs)
    # ══════════════════════════════════════════════════════════════
    @property
    def config(self):
        return self._config

    @property
    def active_mode(self):
        return self._active_mode

    def get_engine_name(self):
        engine = self._config.get("engine", {})
        if isinstance(engine, dict):
            return engine.get("name", "diffuse_field")
        return str(engine) if engine else "diffuse_field"

    def get_engine_settings(self):
        return self._config.get("engine_settings", {})

    def update_config(self, cfg):
        """Deep-merge cfg into current config and persist."""
        self._deep_merge(self._config, cfg)
        self._save_settings()

    def set_status(self, msg, timeout=5000):
        self.statusBar().showMessage(msg, timeout)
        self._status_msg.setText(msg)

    def set_result(self, result_dict):
        """Store last workflow result for use by Figure Studio / Peak Picker."""
        self._last_result = result_dict
        strip_dir = result_dict.get("strip_directory")
        if strip_dir:
            self._last_strip_dir = str(strip_dir)
        self._status_result.setText("Result: ready")

    def log(self, msg):
        """Append message to the Log dock."""
        if hasattr(self, '_log_view') and hasattr(self._log_view, 'append'):
            self._log_view.append(msg)

    def get_active_canvas(self):
        """Return the QTabWidget for the currently active mode."""
        return self._canvas_stacks.get(self._active_mode)

    def get_canvas_view(self, mode, tab_index):
        """Return a specific canvas view widget by mode and tab index."""
        canvas = self._canvas_stacks.get(mode)
        if canvas and 0 <= tab_index < canvas.count():
            return canvas.widget(tab_index)
        return None

    def get_panel(self, mode=None):
        """Return the left-panel widget for the given mode."""
        mode = mode or self._active_mode
        return self._panels.get(mode)

    def get_data_input(self, mode=None):
        """Return the Data Input view (tab index 0) for the given mode."""
        mode = mode or self._active_mode
        return self.get_canvas_view(mode, 0)

    def get_summary_dock(self):
        """Return the summary dock widget."""
        return getattr(self, '_summary_dock', None)

    def get_strip_summary_dock(self):
        """Return the strip summary dock widget."""
        return getattr(self, '_strip_summary_dock', None)

    def get_strip_wizard(self):
        """Return the StripWizardView if active."""
        canvas = self._canvas_stacks.get(self._active_mode)
        if canvas:
            for i in range(canvas.count()):
                w = canvas.widget(i)
                if type(w).__name__ == "StripWizardView":
                    return w
        return None

    def get_figure_studio(self):
        """Return the FigureStudioView if active."""
        canvas = self._canvas_stacks.get(self._active_mode)
        if canvas:
            for i in range(canvas.count()):
                w = canvas.widget(i)
                if type(w).__name__ == "FigureStudioView":
                    return w
        return None

    def update_hv_curve(self, freqs, amps, profile=None):
        """Push forward-model result to the active HV Curve view."""
        canvas = self._canvas_stacks.get(self._active_mode)
        if canvas:
            for i in range(canvas.count()):
                w = canvas.widget(i)
                if hasattr(w, 'set_data'):
                    w.set_data(freqs, amps, profile)
                    break

    def update_vs_profile(self, profile):
        """Push profile to Vs Profile view in the active canvas."""
        canvas = self._canvas_stacks.get(self._active_mode)
        if canvas:
            for i in range(canvas.count()):
                w = canvas.widget(i)
                if hasattr(w, 'set_profile'):
                    w.set_profile(profile)
                    break

    def update_overlay(self, strip_dir):
        """Push strip directory to overlay view."""
        canvas = self._canvas_stacks.get(self._active_mode)
        if canvas:
            for i in range(canvas.count()):
                w = canvas.widget(i)
                if hasattr(w, 'load_strip_dir'):
                    w.load_strip_dir(strip_dir)
                    break

    def update_strip_results(self, result_dict):
        """Push results to the strip summary dock + wizard + figure studio."""
        # Update strip summary dock
        dock = self.get_strip_summary_dock()
        if dock and hasattr(dock, 'set_results'):
            dock.set_results(result_dict)
            dock.setVisible(True)
            dock.raise_()

        # Update wizard with step data
        wiz = self.get_strip_wizard()
        if wiz and hasattr(wiz, 'set_strip_results'):
            wiz.set_strip_results(result_dict)

        # Update figure studio with strip dir
        strip_dir = result_dict.get("strip_directory")
        if strip_dir:
            fs = self.get_figure_studio()
            if fs and hasattr(fs, 'set_strip_data'):
                has_dr = self._config.get(
                    "dual_resonance", {}).get("enable", False)
                fs.set_strip_data(strip_dir,
                                  has_dual_resonance=has_dr)

        # Fallback: still push to any view that accepts set_results
        canvas = self._canvas_stacks.get(self._active_mode)
        if canvas:
            for i in range(canvas.count()):
                w = canvas.widget(i)
                if hasattr(w, 'set_results') and w is not wiz:
                    w.set_results(result_dict)
                    break

    def get_research_dock(self):
        """Return the research results dock widget."""
        return getattr(self, '_research_dock', None)

    def update_research_results(self, phase_name, result_dict):
        """Push research study results to the dock and canvas views.

        Called by ResearchPanel/ResearchWorker after each pipeline phase.
        """
        # Update research dock
        dock = self.get_research_dock()
        if dock and hasattr(dock, 'update_phase'):
            dock.update_phase(phase_name, result_dict)
            if self._active_mode == MODE_RESEARCH:
                dock.setVisible(True)
                dock.raise_()

        # Propagate to canvas views that accept results
        canvas = self._canvas_stacks.get(MODE_RESEARCH)
        if canvas:
            for i in range(canvas.count()):
                w = canvas.widget(i)
                method = f"on_{phase_name}_complete"
                if hasattr(w, method):
                    getattr(w, method)(result_dict)
                elif hasattr(w, 'set_results'):
                    w.set_results(result_dict)

        self.log(f"Research {phase_name}: done")

    # ══════════════════════════════════════════════════════════════
    #  SETTINGS PERSISTENCE
    # ══════════════════════════════════════════════════════════════
    def _load_settings(self):
        if _SETTINGS_FILE.exists():
            try:
                with open(_SETTINGS_FILE, "r") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    self._deep_merge(self._config, data)
                    eng = self._config.get("engine", "diffuse_field")
                    if isinstance(eng, dict):
                        self._config["engine"] = eng
                    elif isinstance(eng, str):
                        self._config["engine"] = {"name": eng}
            except Exception:
                pass

    def _save_settings(self):
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(_SETTINGS_FILE, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except Exception:
            pass

    def _apply_saved_config(self):
        engine_name = self.get_engine_name()
        if hasattr(self, '_status_engine'):
            self._status_engine.setText(f"Engine: {engine_name}")

    @staticmethod
    def _deep_merge(base, update):
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                HVStripWindow._deep_merge(base[k], v)
            else:
                base[k] = v

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════
    @staticmethod
    def _placeholder(text):
        w = QWidget()
        layout = QVBoxLayout(w)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: gray; font-size: 16px;")
        layout.addWidget(lbl)
        return w

    def closeEvent(self, event):
        self._save_settings()
        self._save_project_state()
        self.closed.emit()
        super().closeEvent(event)
