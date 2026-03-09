"""
Batch settings dialog — configuration for batch HV Strip processing.

4 tabs: Frequency, Options, Figure Defaults, Peak Detection.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QFormLayout, QDialogButtonBox,
    QWidget, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLabel,
)


class BatchSettingsDialog(QDialog):
    """Modal dialog for configuring batch processing parameters."""

    def __init__(self, config: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Batch Settings')
        self.resize(450, 420)
        self.setMinimumWidth(400)

        cfg = config or {}
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # ── Tab 1: Frequency ─────────────────────────────────────────
        freq_tab = QWidget()
        fl = QFormLayout(freq_tab)
        freq = cfg.get('frequency', {})

        self.fmin = QDoubleSpinBox()
        self.fmin.setRange(0.01, 50); self.fmin.setValue(freq.get('fmin', 0.3))
        self.fmin.setDecimals(2)
        fl.addRow('f_min (Hz):', self.fmin)

        self.fmax = QDoubleSpinBox()
        self.fmax.setRange(1, 200); self.fmax.setValue(freq.get('fmax', 40))
        self.fmax.setDecimals(1)
        fl.addRow('f_max (Hz):', self.fmax)

        self.nf = QSpinBox()
        self.nf.setRange(50, 2000); self.nf.setValue(freq.get('nf', 300))
        fl.addRow('Num. Freq. Points:', self.nf)

        self.adaptive = QCheckBox('Adaptive frequency range')
        self.adaptive.setChecked(freq.get('adaptive', True))
        self.adaptive.setToolTip('Auto-expand range if peak near boundary')
        fl.addRow(self.adaptive)

        tabs.addTab(freq_tab, 'Frequency')

        # ── Tab 2: Options ───────────────────────────────────────────
        opt_tab = QWidget()
        ol = QFormLayout(opt_tab)
        opts = cfg.get('options', {})

        self.gen_report = QCheckBox('Generate report figures per profile')
        self.gen_report.setChecked(opts.get('generate_report', True))
        ol.addRow(self.gen_report)

        self.interactive = QCheckBox('Interactive peak picking')
        self.interactive.setChecked(opts.get('interactive', False))
        ol.addRow(self.interactive)

        self.dual_res = QCheckBox('Dual-resonance extraction')
        self.dual_res.setChecked(opts.get('dual_resonance', False))
        ol.addRow(self.dual_res)

        self.sep_ratio = QDoubleSpinBox()
        self.sep_ratio.setRange(1.0, 10.0); self.sep_ratio.setValue(opts.get('separation_ratio', 2.0))
        self.sep_ratio.setDecimals(1)
        ol.addRow('Separation ratio:', self.sep_ratio)

        self.min_shift = QDoubleSpinBox()
        self.min_shift.setRange(0.01, 5.0); self.min_shift.setValue(opts.get('min_shift', 0.2))
        self.min_shift.setDecimals(2)
        ol.addRow('Min. freq. shift (Hz):', self.min_shift)

        tabs.addTab(opt_tab, 'Options')

        # ── Tab 3: Figure Defaults ───────────────────────────────────
        fig_tab = QWidget()
        fil = QFormLayout(fig_tab)
        figs = cfg.get('figures', {})

        self.fig_dpi = QSpinBox()
        self.fig_dpi.setRange(72, 600); self.fig_dpi.setValue(figs.get('dpi', 300))
        fil.addRow('DPI:', self.fig_dpi)

        self.fig_font = QSpinBox()
        self.fig_font.setRange(6, 24); self.fig_font.setValue(figs.get('font_size', 12))
        fil.addRow('Font size:', self.fig_font)

        self.fig_overlay = QCheckBox('Generate overlay figure')
        self.fig_overlay.setChecked(figs.get('overlay', True))
        fil.addRow(self.fig_overlay)

        self.fig_waterfall = QCheckBox('Generate waterfall figure')
        self.fig_waterfall.setChecked(figs.get('waterfall', True))
        fil.addRow(self.fig_waterfall)

        self.fig_pub = QCheckBox('Generate publication figure')
        self.fig_pub.setChecked(figs.get('publication', True))
        fil.addRow(self.fig_pub)

        self.fig_peak_evo = QCheckBox('Generate peak evolution figure')
        self.fig_peak_evo.setChecked(figs.get('peak_evolution', True))
        fil.addRow(self.fig_peak_evo)

        tabs.addTab(fig_tab, 'Figure Defaults')

        # ── Tab 4: Peak Detection ────────────────────────────────────
        pk_tab = QWidget()
        pl = QFormLayout(pk_tab)
        peak = cfg.get('peak', {})

        self.pk_preset = QComboBox()
        self.pk_preset.addItems(['default', 'forward_modeling', 'forward_modeling_sharp', 'conservative'])
        self.pk_preset.setCurrentText(peak.get('preset', 'forward_modeling'))
        pl.addRow('Preset:', self.pk_preset)

        self.pk_method = QComboBox()
        self.pk_method.addItems(['max', 'find_peaks', 'manual'])
        self.pk_method.setCurrentText(peak.get('method', 'find_peaks'))
        pl.addRow('Method:', self.pk_method)

        self.pk_selection = QComboBox()
        self.pk_selection.addItems(['leftmost', 'sharpest', 'leftmost_sharpest', 'max'])
        self.pk_selection.setCurrentText(peak.get('selection', 'leftmost'))
        pl.addRow('Selection:', self.pk_selection)

        self.pk_prominence = QDoubleSpinBox()
        self.pk_prominence.setRange(0.0, 5.0); self.pk_prominence.setValue(peak.get('min_prominence', 0.3))
        self.pk_prominence.setDecimals(2)
        pl.addRow('Min. prominence:', self.pk_prominence)

        self.pk_distance = QSpinBox()
        self.pk_distance.setRange(1, 100); self.pk_distance.setValue(peak.get('min_distance', 5))
        pl.addRow('Min. distance:', self.pk_distance)

        self.pk_clarity = QDoubleSpinBox()
        self.pk_clarity.setRange(0.0, 5.0); self.pk_clarity.setValue(peak.get('clarity_threshold', 1.5))
        self.pk_clarity.setDecimals(1)
        pl.addRow('Clarity ratio:', self.pk_clarity)

        tabs.addTab(pk_tab, 'Peak Detection')

        layout.addWidget(tabs)

        # ── Buttons ──────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._restore_defaults)
        layout.addWidget(btns)

    def _restore_defaults(self):
        self.fmin.setValue(0.3); self.fmax.setValue(40); self.nf.setValue(300)
        self.adaptive.setChecked(True)
        self.gen_report.setChecked(True); self.interactive.setChecked(False)
        self.dual_res.setChecked(False)
        self.sep_ratio.setValue(2.0); self.min_shift.setValue(0.2)
        self.fig_dpi.setValue(300); self.fig_font.setValue(12)
        self.fig_overlay.setChecked(True); self.fig_waterfall.setChecked(True)
        self.fig_pub.setChecked(True); self.fig_peak_evo.setChecked(True)
        self.pk_preset.setCurrentText('forward_modeling')
        self.pk_method.setCurrentText('find_peaks')
        self.pk_selection.setCurrentText('leftmost')
        self.pk_prominence.setValue(0.3); self.pk_distance.setValue(5)
        self.pk_clarity.setValue(1.5)

    def get_config(self) -> dict:
        return {
            'frequency': {
                'fmin': self.fmin.value(), 'fmax': self.fmax.value(),
                'nf': self.nf.value(), 'adaptive': self.adaptive.isChecked(),
            },
            'options': {
                'generate_report': self.gen_report.isChecked(),
                'interactive': self.interactive.isChecked(),
                'dual_resonance': self.dual_res.isChecked(),
                'separation_ratio': self.sep_ratio.value(),
                'min_shift': self.min_shift.value(),
            },
            'figures': {
                'dpi': self.fig_dpi.value(), 'font_size': self.fig_font.value(),
                'overlay': self.fig_overlay.isChecked(),
                'waterfall': self.fig_waterfall.isChecked(),
                'publication': self.fig_pub.isChecked(),
                'peak_evolution': self.fig_peak_evo.isChecked(),
            },
            'peak': {
                'preset': self.pk_preset.currentText(),
                'method': self.pk_method.currentText(),
                'selection': self.pk_selection.currentText(),
                'min_prominence': self.pk_prominence.value(),
                'min_distance': self.pk_distance.value(),
                'clarity_threshold': self.pk_clarity.value(),
            },
        }
