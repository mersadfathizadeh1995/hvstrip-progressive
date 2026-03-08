"""
Centralized state management for HVStripWindow.

All mutable application state lives here so widgets can share data
without circular references.  The state object emits Qt signals when
key data changes so widgets can react accordingly.
"""

import os
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

# ── Lazy imports for core types (avoid import errors at startup) ─────
try:
    from .core.soil_profile import SoilProfile
    from .core.engines.base import EngineResult
except ImportError:
    SoilProfile = None
    EngineResult = None


class HVStripState(QObject):
    """Centralized reactive state for the HV Strip Progressive application.

    Emits signals when key data changes so all connected widgets
    stay synchronised without polling.
    """

    # ── Signals ──────────────────────────────────────────────────────
    profile_changed = pyqtSignal()             # active profile loaded / modified
    profiles_changed = pyqtSignal()            # batch profile list changed
    engine_changed = pyqtSignal(str)           # engine selection changed
    freq_config_changed = pyqtSignal()         # fmin / fmax / nf changed
    forward_result_ready = pyqtSignal()        # single forward model done
    strip_result_ready = pyqtSignal()          # full strip workflow done
    batch_progress = pyqtSignal(int, str)      # (percent, message)
    batch_done = pyqtSignal()                  # batch processing finished
    peak_changed = pyqtSignal()                # peak selection modified
    settings_changed = pyqtSignal()            # any setting changed
    status_message = pyqtSignal(str)           # for status bar

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Active profile ───────────────────────────────────────────
        self.active_profile: Optional['SoilProfile'] = None
        self.active_profile_path: Optional[str] = None

        # ── Batch profiles ───────────────────────────────────────────
        self.profiles: List['SoilProfile'] = []
        self.profile_paths: List[str] = []

        # ── Engine configuration ─────────────────────────────────────
        self.engine_name: str = 'diffuse_field'
        self.engine_configs: Dict[str, Dict[str, Any]] = {
            'diffuse_field': {
                'exe_path': 'HVf.exe',
                'nmr': 10, 'nml': 10, 'nks': 10,
            },
            'sh_wave': {
                'sampling': 'log',
                'soil_damping': 0.0,   # 0 = auto Darendeli
                'rock_damping': 1.0,
                'reference_depth': 0,
                'darendeli_curve': 1,
                'gamma_max': 25.0,
                'clip_tf': 0,
            },
            'ellipticity': {
                'gpell_path': '', 'git_bash_path': '',
                'n_modes': 1, 'sampling': 'log',
                'alpha': 0.0, 'auto_q': True,
                'q_formula': 'default', 'clip_factor': 0,
                'absolute': False, 'peak_refined': False,
            },
        }

        # ── Frequency configuration ──────────────────────────────────
        self.fmin: float = 0.2
        self.fmax: float = 20.0
        self.nf: int = 71

        # ── Adaptive scanning ────────────────────────────────────────
        self.adaptive_enabled: bool = True
        self.adaptive_max_passes: int = 2
        self.adaptive_edge_margin: float = 0.05

        # ── Peak detection ───────────────────────────────────────────
        self.peak_preset: str = 'default'
        self.peak_method: str = 'find_peaks'
        self.peak_selection: str = 'leftmost'
        self.peak_prominence: float = 0.2
        self.peak_distance: int = 3
        self.peak_freq_min: Optional[float] = 0.5
        self.peak_freq_max: Optional[float] = None

        # ── Dual-resonance ───────────────────────────────────────────
        self.dual_resonance_enabled: bool = False
        self.separation_ratio_threshold: float = 1.2
        self.separation_shift_threshold: float = 0.3

        # ── Forward model results ────────────────────────────────────
        self.forward_freqs: Optional[np.ndarray] = None
        self.forward_amps: Optional[np.ndarray] = None
        self.forward_f0: Optional[Tuple[float, float, int]] = None  # (freq, amp, idx)
        self.forward_secondary: List[Tuple[float, float, int]] = []

        # ── Strip workflow results ───────────────────────────────────
        self.strip_steps: List[Dict[str, Any]] = []
        # Each step: {name, profile, freqs, amps, f0, layers_removed, ...}
        self.strip_dual_result: Optional[Dict[str, Any]] = None
        self.strip_vs30: Optional[float] = None

        # ── Plot configuration ───────────────────────────────────────
        self.plot_dpi: int = 300
        self.plot_x_scale: str = 'log'
        self.plot_y_scale: str = 'linear'
        self.plot_grid: bool = True
        self.plot_palette: str = 'tab10'
        self.plot_line_alpha: float = 0.8
        self.plot_line_width: float = 2.0

        # ── Figure generation configs ────────────────────────────────
        self.figure_configs: Dict[str, Dict[str, Any]] = {
            'hv_overlay': {'log_x': True, 'grid': True, 'cmap': 'tab10',
                           'linewidth': 2.0, 'alpha': 0.8, 'show_peaks': True,
                           'marker_size': 8, 'xlim_min': 0, 'xlim_max': 0},
            'peak_evolution': {'grid': True, 'show_fill': True,
                               'marker_size': 8, 'linewidth': 2.0},
            'interface_analysis': {'grid': True, 'marker_size': 8,
                                   'linewidth': 2.0, 'annot_font': 10},
            'waterfall': {'log_x': True, 'grid': True, 'cmap': 'tab10',
                          'linewidth': 2.0, 'alpha': 0.8,
                          'offset_factor': 1.5, 'normalize': False},
            'publication': {'grid': True, 'cmap': 'tab10',
                            'linewidth': 2.0, 'alpha': 0.8, 'table_font': 8},
            'dual_resonance': {'grid': True, 'linewidth': 2.5,
                               'f0_offset': (0, 0), 'f1_offset': (0, 0),
                               'show_stripped': True, 'hs_ratio': 0.25},
        }

        # ── Output configuration ─────────────────────────────────────
        self.output_dir: Optional[str] = None
        self.save_formats: List[str] = ['png', 'pdf']
        self.figure_width: float = 10.0
        self.figure_height: float = 6.0
        self.font_size: int = 12

        # ── Workflow options ─────────────────────────────────────────
        self.generate_report: bool = True
        self.interactive_peaks: bool = False
        self.show_vs_profile: bool = True
        self.halfspace_depth_pct: float = 25.0

    # ── Convenience setters that emit signals ────────────────────────

    def set_profile(self, profile, path: Optional[str] = None):
        """Set active soil profile and emit signal."""
        self.active_profile = profile
        self.active_profile_path = path
        self.profile_changed.emit()

    def set_engine(self, name: str):
        """Set active engine and emit signal."""
        if name != self.engine_name:
            self.engine_name = name
            self.engine_changed.emit(name)

    def set_freq_config(self, fmin: float, fmax: float, nf: int):
        """Set frequency configuration and emit signal."""
        self.fmin, self.fmax, self.nf = fmin, fmax, nf
        self.freq_config_changed.emit()

    def set_forward_result(self, result):
        """Store forward model result and emit signal.
        
        Accepts either an EngineResult object or (freqs, amps) tuple.
        """
        if hasattr(result, 'frequencies'):
            self.forward_freqs = result.frequencies
            self.forward_amps = result.amplitudes
        else:
            self.forward_freqs = result[0] if isinstance(result, tuple) else result
            self.forward_amps = result[1] if isinstance(result, tuple) else None
        self.forward_result_ready.emit()

    def set_forward_f0(self, f0_tuple):
        """Set the selected f0 peak and emit signal."""
        self.forward_f0 = f0_tuple
        self.peak_changed.emit()

    def set_strip_results(self, results):
        """Store strip workflow results and emit signal.
        
        Accepts a dict with 'steps', 'dual_resonance', etc.
        """
        if isinstance(results, dict):
            self.strip_steps = results.get('steps', [])
            self.strip_dual_result = results.get('dual_resonance')
            ctrl = results.get('controlling_interface')
            if ctrl:
                self.strip_vs30 = ctrl.get('vs30')
        else:
            self.strip_steps = results
        self.strip_result_ready.emit()

    def set_peak_config(self, config: dict):
        """Update peak detection configuration from a dict."""
        if 'preset' in config: self.peak_preset = config['preset']
        if 'method' in config: self.peak_method = config['method']
        if 'selection' in config: self.peak_selection = config['selection']
        if 'min_prominence' in config: self.peak_prominence = config['min_prominence']
        if 'min_distance' in config: self.peak_distance = config['min_distance']
        self.settings_changed.emit()

    def set_plot_config(self, config: dict):
        """Update plot configuration from a dict."""
        if 'dpi' in config: self.plot_dpi = config['dpi']
        if 'palette' in config: self.plot_palette = config['palette']
        if 'x_scale' in config: self.plot_x_scale = config['x_scale']
        if 'y_scale' in config: self.plot_y_scale = config['y_scale']
        if 'grid' in config: self.plot_grid = config['grid']
        if 'line_alpha' in config: self.plot_line_alpha = config['line_alpha']
        if 'line_width' in config: self.plot_line_width = config['line_width']
        self.settings_changed.emit()

    @property
    def freq_config(self) -> Dict[str, Any]:
        return {'fmin': self.fmin, 'fmax': self.fmax, 'nf': self.nf}

    @freq_config.setter
    def freq_config(self, value: dict):
        if 'fmin' in value: self.fmin = value['fmin']
        if 'fmax' in value: self.fmax = value['fmax']
        if 'nf' in value: self.nf = value['nf']

    @property
    def peak_config(self) -> Dict[str, Any]:
        return self.get_peak_config()

    @peak_config.setter
    def peak_config(self, value: dict):
        self.set_peak_config(value)

    @property
    def plot_config(self) -> Dict[str, Any]:
        return {
            'dpi': self.plot_dpi, 'palette': self.plot_palette,
            'x_scale': self.plot_x_scale, 'y_scale': self.plot_y_scale,
            'grid': self.plot_grid, 'line_alpha': self.plot_line_alpha,
            'line_width': self.plot_line_width,
        }

    @plot_config.setter
    def plot_config(self, value: dict):
        self.set_plot_config(value)

    @property
    def dual_resonance_config(self) -> Dict[str, Any]:
        return {
            'enabled': self.dual_resonance_enabled,
            'separation_ratio': self.separation_ratio_threshold,
            'min_shift': self.separation_shift_threshold,
        }

    @dual_resonance_config.setter
    def dual_resonance_config(self, value: dict):
        if 'enabled' in value: self.dual_resonance_enabled = value['enabled']
        if 'separation_ratio' in value: self.separation_ratio_threshold = value['separation_ratio']
        if 'min_shift' in value: self.separation_shift_threshold = value['min_shift']

    def clear_results(self):
        """Reset all computed results."""
        self.forward_freqs = None
        self.forward_amps = None
        self.forward_f0 = None
        self.forward_secondary = []
        self.strip_steps = []
        self.strip_dual_result = None
        self.strip_vs30 = None

    def get_engine_config(self) -> Dict[str, Any]:
        """Return config dict for the currently selected engine."""
        cfg = dict(self.engine_configs.get(self.engine_name, {}))
        cfg['fmin'] = self.fmin
        cfg['fmax'] = self.fmax
        cfg['nf'] = self.nf
        return cfg

    def get_peak_config(self) -> Dict[str, Any]:
        """Return peak detection config dict."""
        return {
            'preset': self.peak_preset,
            'method': self.peak_method,
            'select': self.peak_selection,
            'find_peaks_params': {
                'prominence': self.peak_prominence,
                'distance': self.peak_distance,
            },
            'freq_min': self.peak_freq_min,
            'freq_max': self.peak_freq_max,
        }
