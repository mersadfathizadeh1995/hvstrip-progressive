"""
Per-figure settings panels for the Figure Wizard.

Each panel provides controls specific to one figure type and
a get_config() method returning a dict of current settings.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QCheckBox, QDoubleSpinBox,
    QSpinBox, QComboBox, QLabel, QGroupBox,
)

_CMAPS = ['tab10', 'tab20', 'Set1', 'viridis', 'plasma', 'inferno',
          'magma', 'coolwarm', 'RdYlBu', 'Spectral']


class _BasePanel(QWidget):
    """Base class for figure settings panels."""

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(4, 4, 4, 4)

    def _add_check(self, label, key, default):
        cb = QCheckBox(label)
        cb.setChecked(default)
        self._form.addRow(cb)
        setattr(self, f'_{key}', cb)
        return cb

    def _add_spin(self, label, key, lo, hi, default, decimals=0, suffix=''):
        if decimals > 0:
            s = QDoubleSpinBox()
            s.setDecimals(decimals)
        else:
            s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(default)
        if suffix:
            s.setSuffix(suffix)
        self._form.addRow(label, s)
        setattr(self, f'_{key}', s)
        return s

    def _add_combo(self, label, key, items, default):
        c = QComboBox()
        c.addItems(items)
        c.setCurrentText(str(default))
        self._form.addRow(label, c)
        setattr(self, f'_{key}', c)
        return c

    def get_config(self) -> dict:
        raise NotImplementedError


class HVOverlayPanel(_BasePanel):
    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self._add_check('Log X axis', 'log_x', cfg.get('log_x', True))
        self._add_check('Grid', 'grid', cfg.get('grid', True))
        self._add_combo('Colormap:', 'cmap', _CMAPS, cfg.get('cmap', 'tab10'))
        self._add_spin('Line width:', 'linewidth', 0.5, 6, cfg.get('linewidth', 2), 1)
        self._add_spin('Alpha:', 'alpha', 0.1, 1.0, cfg.get('alpha', 0.8), 2)
        self._add_check('Show peaks', 'show_peaks', cfg.get('show_peaks', True))
        self._add_spin('Marker size:', 'marker_size', 4, 20, cfg.get('marker_size', 8))
        self._add_spin('X min (Hz):', 'xlim_min', 0, 50, cfg.get('xlim_min', 0), 1)
        self._add_spin('X max (Hz):', 'xlim_max', 0, 100, cfg.get('xlim_max', 0), 1)

    def get_config(self):
        return {
            'log_x': self._log_x.isChecked(), 'grid': self._grid.isChecked(),
            'cmap': self._cmap.currentText(), 'linewidth': self._linewidth.value(),
            'alpha': self._alpha.value(), 'show_peaks': self._show_peaks.isChecked(),
            'marker_size': self._marker_size.value(),
            'xlim_min': self._xlim_min.value(), 'xlim_max': self._xlim_max.value(),
        }


class PeakEvolutionPanel(_BasePanel):
    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self._add_check('Grid', 'grid', cfg.get('grid', True))
        self._add_check('Show fill under bars', 'show_fill', cfg.get('show_fill', True))
        self._add_spin('Marker size:', 'marker_size', 4, 20, cfg.get('marker_size', 8))
        self._add_spin('Line width:', 'linewidth', 0.5, 6, cfg.get('linewidth', 2), 1)

    def get_config(self):
        return {
            'grid': self._grid.isChecked(), 'show_fill': self._show_fill.isChecked(),
            'marker_size': self._marker_size.value(), 'linewidth': self._linewidth.value(),
        }


class InterfaceAnalysisPanel(_BasePanel):
    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self._add_check('Grid', 'grid', cfg.get('grid', True))
        self._add_spin('Marker size:', 'marker_size', 4, 20, cfg.get('marker_size', 8))
        self._add_spin('Line width:', 'linewidth', 0.5, 6, cfg.get('linewidth', 2), 1)
        self._add_spin('Annotation font:', 'annot_font', 6, 18, cfg.get('annot_font', 10))

    def get_config(self):
        return {
            'grid': self._grid.isChecked(), 'marker_size': self._marker_size.value(),
            'linewidth': self._linewidth.value(), 'annot_font': self._annot_font.value(),
        }


class WaterfallPanel(_BasePanel):
    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self._add_check('Log X axis', 'log_x', cfg.get('log_x', True))
        self._add_check('Grid', 'grid', cfg.get('grid', True))
        self._add_combo('Colormap:', 'cmap', _CMAPS, cfg.get('cmap', 'tab10'))
        self._add_spin('Line width:', 'linewidth', 0.5, 6, cfg.get('linewidth', 2), 1)
        self._add_spin('Alpha:', 'alpha', 0.1, 1.0, cfg.get('alpha', 0.8), 2)
        self._add_spin('Offset factor:', 'offset_factor', 0.5, 5, cfg.get('offset_factor', 1.5), 1)
        self._add_check('Normalize curves', 'normalize', cfg.get('normalize', False))

    def get_config(self):
        return {
            'log_x': self._log_x.isChecked(), 'grid': self._grid.isChecked(),
            'cmap': self._cmap.currentText(), 'linewidth': self._linewidth.value(),
            'alpha': self._alpha.value(), 'offset_factor': self._offset_factor.value(),
            'normalize': self._normalize.isChecked(),
        }


class PublicationPanel(_BasePanel):
    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self._add_check('Grid', 'grid', cfg.get('grid', True))
        self._add_combo('Colormap:', 'cmap', _CMAPS, cfg.get('cmap', 'tab10'))
        self._add_spin('Line width:', 'linewidth', 0.5, 6, cfg.get('linewidth', 2), 1)
        self._add_spin('Alpha:', 'alpha', 0.1, 1.0, cfg.get('alpha', 0.8), 2)
        self._add_spin('Table font:', 'table_font', 6, 16, cfg.get('table_font', 8))

    def get_config(self):
        return {
            'grid': self._grid.isChecked(), 'cmap': self._cmap.currentText(),
            'linewidth': self._linewidth.value(), 'alpha': self._alpha.value(),
            'table_font': self._table_font.value(),
        }


class DualResonancePanel(_BasePanel):
    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        self._add_check('Grid', 'grid', cfg.get('grid', True))
        self._add_spin('Line width:', 'linewidth', 0.5, 6, cfg.get('linewidth', 2.5), 1)

        self._form.addRow(QLabel('f₀ Annotation Offset:'))
        self._add_spin('Δx (Hz):', 'f0_dx', -10, 10, 0, 1)
        self._add_spin('Δy (amp):', 'f0_dy', -50, 50, 0, 1)

        self._form.addRow(QLabel('f₁ Annotation Offset:'))
        self._add_spin('Δx (Hz):', 'f1_dx', -10, 10, 0, 1)
        self._add_spin('Δy (amp):', 'f1_dy', -50, 50, 0, 1)

        self._add_check('Show stripped curve', 'show_stripped', cfg.get('show_stripped', True))
        self._add_spin('HS depth %:', 'hs_ratio', 0.1, 1.0, cfg.get('hs_ratio', 0.25), 2)

    def get_config(self):
        return {
            'grid': self._grid.isChecked(), 'linewidth': self._linewidth.value(),
            'f0_offset': (self._f0_dx.value(), self._f0_dy.value()),
            'f1_offset': (self._f1_dx.value(), self._f1_dy.value()),
            'show_stripped': self._show_stripped.isChecked(),
            'hs_ratio': self._hs_ratio.value(),
        }


# ── Factory function ─────────────────────────────────────────────────

_PANEL_MAP = {
    'hv_overlay': HVOverlayPanel,
    'peak_evolution': PeakEvolutionPanel,
    'interface_analysis': InterfaceAnalysisPanel,
    'waterfall': WaterfallPanel,
    'publication': PublicationPanel,
    'dual_resonance': DualResonancePanel,
}


def create_panels(fig_key: str, cfg: dict) -> _BasePanel:
    """Create the appropriate settings panel for a figure type."""
    cls = _PANEL_MAP.get(fig_key, _BasePanel)
    return cls(cfg)
