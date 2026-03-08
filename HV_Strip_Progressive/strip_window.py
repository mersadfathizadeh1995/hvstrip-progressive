"""
HV Strip Progressive — Main Window

QMainWindow that hosts the entire HV Strip Progressive GUI.
Layout follows the bedrock_mapping architectural pattern:
  Left:  control tabs (Profile, Engine, Strip, Batch)
  Center: view tabs (Forward, Strip Results, Figures)
  Right:  dock widget (Results + Settings)
"""

import os
import sys
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QTabWidget,
    QDockWidget, QStatusBar, QLabel, QMessageBox, QFileDialog,
)

from .state import HVStripState
from .widgets.profile_editor import ProfileEditor
from .widgets.engine_panel import EnginePanel
from .widgets.strip_panel import StripPanel
from .widgets.batch_panel import BatchPanel
from .widgets.forward_view import ForwardView
from .widgets.strip_view import StripView
from .widgets.figure_gallery import FigureGallery
from .widgets.results_panel import ResultsPanel
from .widgets.settings_panel import SettingsPanel


class HVStripWindow(QMainWindow):
    """Top-level window for the HV Strip Progressive package."""

    _SETTINGS_DIR = os.path.join(os.path.expanduser('~'), '.hvstrip')
    _SETTINGS_FILE = os.path.join(_SETTINGS_DIR, 'settings.yaml')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('HV Strip Progressive')
        self.resize(1400, 900)
        self.setMinimumSize(1000, 650)

        self.state = HVStripState(self)
        self.state.status_message.connect(self._show_status)
        self._worker = None

        self._load_settings()
        self._build_menu_bar()
        self._build_ui()
        self._build_status_bar()
        self._connect_signals()

        self.statusBar().showMessage(
            'Ready — load a soil profile or model file to begin', 5000)

    # ── Menu bar ─────────────────────────────────────────────────────

    def _build_menu_bar(self):
        mb = self.menuBar()

        file_menu = mb.addMenu('&File')
        open_act = file_menu.addAction('&Open Model…')
        open_act.setShortcut('Ctrl+O')
        open_act.triggered.connect(self._on_open_model)
        file_menu.addSeparator()
        export_act = file_menu.addAction('&Export Results…')
        export_act.setShortcut('Ctrl+E')
        export_act.triggered.connect(self._on_export_results)
        file_menu.addSeparator()
        close_act = file_menu.addAction('&Close')
        close_act.setShortcut('Ctrl+W')
        close_act.triggered.connect(self.close)

        view_menu = mb.addMenu('&View')
        self._dock_toggle = None  # set after dock created

        tools_menu = mb.addMenu('&Tools')
        tools_menu.addAction('⚙ Engine &Settings…').triggered.connect(self._on_engine_settings)
        tools_menu.addAction('📊 &Figure Wizard…').triggered.connect(self._on_figure_wizard)
        tools_menu.addAction('👁 &Output Viewer…').triggered.connect(self._on_output_viewer)
        tools_menu.addAction('🔀 &Dual-Resonance…').triggered.connect(self._on_dual_res)
        tools_menu.addSeparator()
        tools_menu.addAction('💾 &Save Settings').triggered.connect(self._save_settings)

    # ── Central layout ───────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter)

        # Left: control tabs
        self.control_tabs = QTabWidget()
        self.control_tabs.setMinimumWidth(320)
        self.control_tabs.setMaximumWidth(550)

        self.profile_editor = ProfileEditor(self.state)
        self.engine_panel = EnginePanel(self.state)
        self.strip_panel = StripPanel(self.state)
        self.batch_panel = BatchPanel(self.state)

        self.control_tabs.addTab(self.profile_editor, '📁 Profile')
        self.control_tabs.addTab(self.engine_panel, '⚙ Engine')
        self.control_tabs.addTab(self.strip_panel, '🔬 Strip')
        self.control_tabs.addTab(self.batch_panel, '📦 Batch')
        self.splitter.addWidget(self.control_tabs)

        # Center: view tabs
        self.view_tabs = QTabWidget()
        self.forward_view = ForwardView(self.state)
        self.strip_view = StripView(self.state)
        self.figure_gallery = FigureGallery(self.state)

        self.view_tabs.addTab(self.forward_view, '📊 Forward')
        self.view_tabs.addTab(self.strip_view, '🔬 Strip Results')
        self.view_tabs.addTab(self.figure_gallery, '📈 Figures')
        self.splitter.addWidget(self.view_tabs)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([400, 1000])

        # Right dock
        dock = QDockWidget('Results && Settings', self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.right_tabs = QTabWidget()
        self.results_panel = ResultsPanel(self.state)
        self.settings_panel = SettingsPanel(self.state)
        self.right_tabs.addTab(self.results_panel, '📋 Results')
        self.right_tabs.addTab(self.settings_panel, '🔧 Settings')
        dock.setWidget(self.right_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # Add dock toggle to View menu
        view_menu = self.menuBar().actions()[1].menu()
        if view_menu:
            view_menu.addAction(dock.toggleViewAction())

    # ── Status bar ───────────────────────────────────────────────────

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.engine_label = QLabel('Engine: diffuse_field')
        self.profile_label = QLabel('Profile: —')
        self.f0_label = QLabel('f0: —')
        for lbl in (self.engine_label, self.profile_label, self.f0_label):
            sb.addPermanentWidget(lbl)

    # ── Signal connections ───────────────────────────────────────────

    def _connect_signals(self):
        s = self.state

        # Status bar updates
        s.engine_changed.connect(
            lambda name: self.engine_label.setText(f'Engine: {name}'))
        s.profile_changed.connect(self._update_profile_label)
        s.forward_result_ready.connect(self._update_f0_label)

        # Profile editor emits profile_changed; push to state
        self.profile_editor.profile_changed.connect(
            lambda: s.set_profile(self.profile_editor.get_profile())
            if hasattr(self.profile_editor, 'get_profile') else None)

        # Engine panel already pushes to state internally via _on_engine_changed / _push_freq
        # Wire its settings button to our dialog
        self.engine_panel.engine_settings_requested.connect(self._on_engine_settings)

        # Forward view peak picks already connected to state via state.forward_result_ready
        self.forward_view.peak_selected.connect(
            lambda f, a, i: s.set_forward_f0((f, a, i)))

        # Strip panel
        self.strip_panel.run_requested.connect(self._run_strip)
        self.strip_panel.cancel_requested.connect(self._cancel_worker)
        self.strip_panel.dual_resonance_settings_requested.connect(self._on_dual_res)

        # Batch panel
        self.batch_panel.run_batch_requested.connect(self._run_batch)
        self.batch_panel.batch_settings_requested.connect(self._on_batch_settings)

        # Figure gallery
        self.figure_gallery.figure_wizard_requested.connect(self._on_figure_wizard)

    # ── Workers ──────────────────────────────────────────────────────

    def _run_forward(self):
        if self.state.active_profile is None:
            QMessageBox.warning(self, 'No Profile', 'Load a soil profile first.')
            return
        from .workers.forward_worker import ForwardWorker
        self._worker = ForwardWorker(
            self.state.active_profile,
            self.state.engine_name,
            self.state.engine_configs.get(self.state.engine_name, {}),
            self.state.freq_config,
            parent=self,
        )
        self._worker.progress.connect(self._show_status)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, 'Error', e))
        self._worker.finished.connect(self._on_forward_done)
        self._worker.start()

    def _on_forward_done(self, result):
        if result is not None:
            self.state.set_forward_result(result)
        self._worker = None

    def _run_strip(self):
        if self.state.active_profile is None:
            QMessageBox.warning(self, 'No Profile', 'Load a soil profile first.')
            return

        # Check if interactive mode
        if self.strip_panel.cb_interactive.isChecked():
            self._run_strip_interactive()
            return

        from .workers.strip_worker import StripWorker
        self._worker = StripWorker(
            self.state.active_profile,
            self.state.engine_name,
            self.state.engine_configs.get(self.state.engine_name, {}),
            self.state.freq_config,
            self.state.peak_config,
            self.state.dual_resonance_config if self.strip_panel.cb_dual.isChecked() else {},
            self.strip_panel.cb_report.isChecked(),
            self.strip_panel.out_edit.text(),
            parent=self,
        )
        self._worker.progress.connect(
            lambda pct, msg: self.state.batch_progress.emit(pct, msg))
        self._worker.error.connect(lambda e: QMessageBox.critical(self, 'Error', e))
        self._worker.finished.connect(self._on_strip_done)
        self._worker.start()

    def _run_strip_interactive(self):
        """Run strip then open interactive peak picker."""
        self._run_strip_auto_then_interactive = True
        from .workers.strip_worker import StripWorker
        self._worker = StripWorker(
            self.state.active_profile,
            self.state.engine_name,
            self.state.engine_configs.get(self.state.engine_name, {}),
            self.state.freq_config,
            self.state.peak_config,
            parent=self,
        )
        self._worker.progress.connect(
            lambda pct, msg: self.state.batch_progress.emit(pct, msg))
        self._worker.error.connect(lambda e: QMessageBox.critical(self, 'Error', e))
        self._worker.finished.connect(self._on_strip_done_interactive)
        self._worker.start()

    def _on_strip_done(self, results):
        self.state.set_strip_results(results)
        self._worker = None

    def _on_strip_done_interactive(self, results):
        self._worker = None
        steps = results.get('steps', [])
        if not steps:
            QMessageBox.warning(self, 'No Steps', 'Stripping produced no results.')
            return
        from .dialogs.interactive_peak_picker import InteractivePeakPicker
        dlg = InteractivePeakPicker(steps, self.state.peak_config, parent=self)
        if dlg.exec_() == dlg.Accepted:
            for i, step in enumerate(steps):
                picked = dlg.get_results().get(i)
                if picked:
                    step['f0'] = picked.get('f0')
                    step['f0_amp'] = picked.get('f0_amp')
            self.state.set_strip_results(results)

    def _run_batch(self):
        files = self.batch_panel.get_file_paths()
        if not files:
            QMessageBox.warning(self, 'No Files', 'Add profile files first.')
            return
        from .workers.batch_worker import BatchWorker
        self._worker = BatchWorker(
            files,
            self.state.engine_name,
            self.state.engine_configs.get(self.state.engine_name, {}),
            self.state.freq_config,
            self.state.peak_config,
            {},
            self.batch_panel.out_edit.text(),
            parent=self,
        )
        self._worker.progress.connect(self.batch_panel._on_progress)
        self._worker.error.connect(lambda e: QMessageBox.critical(self, 'Error', e))
        self._worker.finished.connect(self._on_batch_done)
        self._worker.start()

    def _on_batch_done(self, results):
        n_ok = sum(1 for r in results if r.get('success'))
        self._show_status(f'Batch complete: {n_ok}/{len(results)} succeeded')
        self._worker = None

    def _cancel_worker(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._show_status('Cancelling…')

    # ── Menu actions ─────────────────────────────────────────────────

    def _on_open_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Model File', '',
            'Model files (*.txt *.csv *.xlsx);;All files (*)')
        if not path:
            return
        try:
            pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            hvstrip_root = os.path.join(pkg_root, 'hvstrip_progressive')
            if hvstrip_root not in sys.path:
                sys.path.insert(0, hvstrip_root)
            from core.soil_profile import SoilProfile
            profile = SoilProfile.from_auto(path)
            self.state.set_profile(profile)
            self._show_status(f'Loaded: {os.path.basename(path)}')
            self.profile_editor.load_profile(profile)
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))

    def _on_export_results(self):
        path = QFileDialog.getExistingDirectory(self, 'Export Results To')
        if not path:
            return
        # Delegate to forward_view's save logic
        self.forward_view._on_save_to_dir(path)
        self._show_status(f'Results exported to {path}')

    def _on_batch_settings(self):
        from .dialogs.batch_settings_dialog import BatchSettingsDialog
        dlg = BatchSettingsDialog(parent=self)
        if dlg.exec_() == dlg.Accepted:
            cfg = dlg.get_config()
            self.state.freq_config.update(cfg.get('frequency', {}))
            self.state.peak_config.update(cfg.get('peak', {}))
            self.state.settings_changed.emit()

    def _on_engine_settings(self):
        from .dialogs.engine_settings_dialog import EngineSettingsDialog
        dlg = EngineSettingsDialog(self.state.engine_configs, parent=self)
        if dlg.exec_() == dlg.Accepted:
            self.state.engine_configs = dlg.get_configs()
            self.state.settings_changed.emit()

    def _on_figure_wizard(self):
        steps = self.state.strip_steps
        if not steps:
            QMessageBox.information(self, 'No Data', 'Run a strip analysis first.')
            return
        from .dialogs.figure_wizard_dialog import FigureWizardDialog
        dlg = FigureWizardDialog(steps, self.state.figure_configs, parent=self)
        dlg.exec_()

    def _on_output_viewer(self):
        from .dialogs.output_viewer_dialog import OutputViewerDialog
        dlg = OutputViewerDialog(parent=self)
        dlg.exec_()

    def _on_dual_res(self):
        from .dialogs.dual_resonance_dialog import DualResonanceDialog
        dlg = DualResonanceDialog(self.state.dual_resonance_config, parent=self)
        if dlg.exec_() == dlg.Accepted:
            self.state.dual_resonance_config = dlg.get_config()

    # ── Settings persistence ─────────────────────────────────────────

    def _load_settings(self):
        if os.path.isfile(self._SETTINGS_FILE):
            try:
                with open(self._SETTINGS_FILE, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                # Engine name (may be str or dict with 'name' key)
                eng = cfg.get('engine')
                if isinstance(eng, str):
                    self.state.engine_name = eng
                elif isinstance(eng, dict) and 'name' in eng:
                    self.state.engine_name = eng['name']
                if 'engine_configs' in cfg and isinstance(cfg['engine_configs'], dict):
                    self.state.engine_configs.update(cfg['engine_configs'])
                if 'freq_config' in cfg and isinstance(cfg['freq_config'], dict):
                    self.state.freq_config = cfg['freq_config']
                if 'peak_config' in cfg and isinstance(cfg['peak_config'], dict):
                    self.state.peak_config = cfg['peak_config']
                if 'plot_config' in cfg and isinstance(cfg['plot_config'], dict):
                    self.state.plot_config = cfg['plot_config']
                if 'dual_resonance' in cfg and isinstance(cfg['dual_resonance'], dict):
                    self.state.dual_resonance_config = cfg['dual_resonance']
            except Exception:
                pass

    def _save_settings(self):
        try:
            os.makedirs(self._SETTINGS_DIR, exist_ok=True)
            cfg = {
                'engine': self.state.engine_name,
                'engine_configs': self.state.engine_configs,
                'freq_config': self.state.freq_config,
                'peak_config': self.state.peak_config,
                'plot_config': self.state.plot_config,
                'dual_resonance': self.state.dual_resonance_config,
            }
            with open(self._SETTINGS_FILE, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
            self._show_status(f'Settings saved to {self._SETTINGS_FILE}')
        except Exception as e:
            QMessageBox.warning(self, 'Save Error', str(e))

    # ── Helpers ──────────────────────────────────────────────────────

    def _show_status(self, msg: str):
        self.statusBar().showMessage(msg, 5000)

    def _update_profile_label(self):
        p = self.state.active_profile
        if p is not None:
            n = len(p.layers) if hasattr(p, 'layers') else '?'
            name = getattr(p, 'name', '') or 'unnamed'
            self.profile_label.setText(f'Profile: {name} ({n} layers)')
        else:
            self.profile_label.setText('Profile: —')

    def _update_f0_label(self):
        f0 = self.state.forward_f0
        if f0 is not None:
            self.f0_label.setText(f'f0: {f0[0]:.2f} Hz')
        else:
            self.f0_label.setText('f0: —')

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)
