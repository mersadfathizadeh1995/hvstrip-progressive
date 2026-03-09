"""
Settings dock panel — peak detection and plot style controls.

Provides peak detection presets / method / selection combos and
plot style settings (DPI, palette, log scales, grid) in the
right dock.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLabel,
)

from ..widgets.collapsible_group import CollapsibleGroup

_PRESETS = ['default', 'forward_modeling', 'forward_modeling_sharp', 'conservative', 'custom']
_METHODS = ['find_peaks', 'max', 'manual']
_SELECTIONS = ['leftmost', 'sharpest', 'leftmost_sharpest', 'max']
_PALETTES = [
    'tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Paired',
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'coolwarm', 'RdYlBu', 'Spectral', 'Dark2', 'Accent',
]


class SettingsPanel(QWidget):
    """Peak detection + plot style settings for the right dock."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Peak detection ───────────────────────────────────────────
        peak_grp = CollapsibleGroup('Peak Detection')
        pform = QFormLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(_PRESETS)
        self.preset_combo.setCurrentText(state.peak_preset)
        pform.addRow('Preset:', self.preset_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(_METHODS)
        self.method_combo.setCurrentText(state.peak_method)
        pform.addRow('Method:', self.method_combo)

        self.select_combo = QComboBox()
        self.select_combo.addItems(_SELECTIONS)
        self.select_combo.setCurrentText(state.peak_selection)
        pform.addRow('Selection:', self.select_combo)

        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.01, 5.0)
        self.prominence_spin.setValue(state.peak_prominence)
        self.prominence_spin.setDecimals(2)
        pform.addRow('Prominence:', self.prominence_spin)

        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 50)
        self.distance_spin.setValue(state.peak_distance)
        pform.addRow('Distance:', self.distance_spin)

        self.preset_desc = QLabel('')
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet('color: gray; font-size: 10px;')
        pform.addRow(self.preset_desc)

        peak_grp.add_layout(pform)
        layout.addWidget(peak_grp)

        # ── Plot style ───────────────────────────────────────────────
        plot_grp = CollapsibleGroup('Plot Style')
        sform = QFormLayout()

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(state.plot_dpi)
        sform.addRow('DPI:', self.dpi_spin)

        self.palette_combo = QComboBox()
        self.palette_combo.addItems(_PALETTES)
        self.palette_combo.setCurrentText(state.plot_palette)
        sform.addRow('Palette:', self.palette_combo)

        self.cb_logx = QCheckBox('Log X axis')
        self.cb_logx.setChecked(state.plot_x_scale == 'log')
        sform.addRow(self.cb_logx)

        self.cb_logy = QCheckBox('Log Y axis')
        self.cb_logy.setChecked(state.plot_y_scale == 'log')
        sform.addRow(self.cb_logy)

        self.cb_grid = QCheckBox('Show Grid')
        self.cb_grid.setChecked(state.plot_grid)
        sform.addRow(self.cb_grid)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 1.0)
        self.alpha_spin.setValue(state.plot_line_alpha)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setSingleStep(0.05)
        sform.addRow('Line Alpha:', self.alpha_spin)

        self.lw_spin = QDoubleSpinBox()
        self.lw_spin.setRange(0.5, 6.0)
        self.lw_spin.setValue(state.plot_line_width)
        self.lw_spin.setDecimals(1)
        sform.addRow('Line Width:', self.lw_spin)

        plot_grp.add_layout(sform)
        layout.addWidget(plot_grp)

        layout.addStretch()

        # ── Connections ──────────────────────────────────────────────
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.method_combo.currentTextChanged.connect(self._push_peak)
        self.select_combo.currentTextChanged.connect(self._push_peak)
        self.prominence_spin.valueChanged.connect(self._push_peak)
        self.distance_spin.valueChanged.connect(self._push_peak)

        self.dpi_spin.valueChanged.connect(self._push_plot)
        self.palette_combo.currentTextChanged.connect(self._push_plot)
        self.cb_logx.toggled.connect(self._push_plot)
        self.cb_logy.toggled.connect(self._push_plot)
        self.cb_grid.toggled.connect(self._push_plot)
        self.alpha_spin.valueChanged.connect(self._push_plot)
        self.lw_spin.valueChanged.connect(self._push_plot)

        self._on_preset_changed(state.peak_preset)

    def _on_preset_changed(self, preset: str):
        """Update method/selection combos based on preset."""
        presets_map = {
            'default': ('find_peaks', 'leftmost', 0.2, 3),
            'forward_modeling': ('find_peaks', 'leftmost', 0.15, 3),
            'forward_modeling_sharp': ('find_peaks', 'sharpest', 0.3, 5),
            'conservative': ('find_peaks', 'leftmost_sharpest', 0.4, 5),
        }
        if preset in presets_map:
            method, sel, prom, dist = presets_map[preset]
            self.method_combo.setCurrentText(method)
            self.select_combo.setCurrentText(sel)
            self.prominence_spin.setValue(prom)
            self.distance_spin.setValue(dist)
            self.method_combo.setEnabled(False)
            self.select_combo.setEnabled(False)
            self.preset_desc.setText(f'Preset: {method}, {sel}, prominence={prom}')
        else:
            self.method_combo.setEnabled(True)
            self.select_combo.setEnabled(True)
            self.preset_desc.setText('Custom: configure freely')
        self._push_peak()

    def _push_peak(self, *_):
        self.state.peak_preset = self.preset_combo.currentText()
        self.state.peak_method = self.method_combo.currentText()
        self.state.peak_selection = self.select_combo.currentText()
        self.state.peak_prominence = self.prominence_spin.value()
        self.state.peak_distance = self.distance_spin.value()
        self.state.settings_changed.emit()

    def _push_plot(self, *_):
        self.state.plot_dpi = self.dpi_spin.value()
        self.state.plot_palette = self.palette_combo.currentText()
        self.state.plot_x_scale = 'log' if self.cb_logx.isChecked() else 'linear'
        self.state.plot_y_scale = 'log' if self.cb_logy.isChecked() else 'linear'
        self.state.plot_grid = self.cb_grid.isChecked()
        self.state.plot_line_alpha = self.alpha_spin.value()
        self.state.plot_line_width = self.lw_spin.value()
        self.state.settings_changed.emit()
