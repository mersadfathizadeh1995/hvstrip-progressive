"""
Multi-profile picker dialog — interactive peak selection across multiple profiles.

Computes forward models for each profile, displays combined overlay,
lets the user pick f0/secondary peaks, computes median curve.
"""

import os
import csv
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QWidget, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QFormLayout, QGroupBox, QMessageBox, QFileDialog,
)

from ..widgets.plot_canvas import PlotCanvas
from ..widgets.collapsible_group import CollapsibleGroup

_PALETTES = [
    'tab10', 'tab20', 'Set1', 'Set2', 'Paired', 'Dark2',
    'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'Spectral',
    'RdYlBu', 'RdBu', 'PiYG', 'BrBG',
]
_SHAPES = ['*', 'D', 'o', 's', '^', 'v', '<', '>', 'p', 'h']
_COLORS = ['red', 'black', 'blue', 'green', 'orange', 'purple', 'brown', 'gray']


class MultiProfileDialog(QDialog):
    """Modal dialog for multi-profile forward model + peak picking."""

    def __init__(self, profiles: list, freq_config: dict,
                 output_dir: str = '', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Multi-Profile Analysis')
        self.resize(1400, 800)
        self.setMinimumSize(1100, 700)

        self._profiles = profiles  # list of (name, SoilProfile)
        self._freq_config = freq_config
        self._output_dir = output_dir
        self._results = []  # list of {name, freqs, amps, f0, secondary, computed}
        self._current = 0
        self._pick_mode = 'f0'
        self._median = None

        # Init results list
        for name, prof in profiles:
            self._results.append({
                'name': name, 'profile': prof,
                'freqs': None, 'amps': None,
                'f0': None, 'secondary': [], 'computed': False,
            })

        layout = QHBoxLayout(self)

        # ── Left panel ───────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(220)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        ll.addWidget(QLabel('Profiles:'))
        self.prof_list = QListWidget()
        for name, _ in profiles:
            self.prof_list.addItem(name)
        self.prof_list.addItem('── Median HV ──')
        self.prof_list.currentRowChanged.connect(self._on_profile_clicked)
        ll.addWidget(self.prof_list)

        # Plot settings
        settings = CollapsibleGroup('Plot Settings', collapsed=True)
        sform = QFormLayout()

        self.palette_combo = QComboBox()
        self.palette_combo.addItems(_PALETTES)
        sform.addRow('Palette:', self.palette_combo)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0); self.alpha_spin.setValue(0.45)
        self.alpha_spin.setDecimals(2)
        sform.addRow('Indiv. α:', self.alpha_spin)

        self.lw_spin = QDoubleSpinBox()
        self.lw_spin.setRange(0.3, 5.0); self.lw_spin.setValue(1.0)
        sform.addRow('Indiv. LW:', self.lw_spin)

        self.med_lw = QDoubleSpinBox()
        self.med_lw.setRange(0.5, 8.0); self.med_lw.setValue(3.0)
        sform.addRow('Median LW:', self.med_lw)

        # Marker combos
        for prefix, default_color, default_shape in [('f0', 'red', '*'), ('sec', 'black', 'D')]:
            color_combo = QComboBox(); color_combo.addItems(_COLORS)
            color_combo.setCurrentText(default_color)
            sform.addRow(f'{prefix} Color:', color_combo)
            shape_combo = QComboBox(); shape_combo.addItems(_SHAPES)
            shape_combo.setCurrentText(default_shape)
            sform.addRow(f'{prefix} Shape:', shape_combo)
            setattr(self, f'{prefix}_color', color_combo)
            setattr(self, f'{prefix}_shape', shape_combo)

        settings.add_layout(sform)
        ll.addWidget(settings)

        layout.addWidget(left)

        # ── Right panel ──────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel()
        self.info_label.setStyleSheet('font-weight: bold;')
        rl.addWidget(self.info_label)

        self.canvas = PlotCanvas(figsize=(14, 5), dpi=100)
        rl.addWidget(self.canvas, 1)

        # Pick mode
        pick_row = QHBoxLayout()
        self.f0_btn = QPushButton('Select f0')
        self.f0_btn.setCheckable(True); self.f0_btn.setChecked(True)
        self.f0_btn.setStyleSheet('QPushButton:checked { background-color: #27ae60; color: white; }')
        self.f0_btn.toggled.connect(lambda c: self._set_pick('f0') if c else None)
        pick_row.addWidget(self.f0_btn)

        self.sec_btn = QPushButton('Select Secondary')
        self.sec_btn.setCheckable(True)
        self.sec_btn.setStyleSheet('QPushButton:checked { background-color: #e67e22; color: white; }')
        self.sec_btn.toggled.connect(lambda c: self._set_pick('secondary') if c else None)
        pick_row.addWidget(self.sec_btn)

        clear_btn = QPushButton('Clear Secondary')
        clear_btn.clicked.connect(self._clear_secondary)
        pick_row.addWidget(clear_btn)
        pick_row.addStretch()
        rl.addLayout(pick_row)

        self.sel_label = QLabel('')
        self.sel_label.setStyleSheet('color: gray;')
        rl.addWidget(self.sel_label)

        # Navigation
        nav = QHBoxLayout()
        self.prev_btn = QPushButton('◀ Previous'); self.prev_btn.clicked.connect(self._on_prev)
        self.next_btn = QPushButton('Next ▶'); self.next_btn.clicked.connect(self._on_next)
        nav.addWidget(self.prev_btn); nav.addWidget(self.next_btn)

        auto_btn = QPushButton('Auto-detect'); auto_btn.clicked.connect(self._on_auto)
        skip_btn = QPushButton('Skip'); skip_btn.clicked.connect(self._on_skip)
        auto_all_btn = QPushButton('Auto All Remaining'); auto_all_btn.clicked.connect(self._on_auto_all)
        nav.addWidget(auto_btn); nav.addWidget(skip_btn); nav.addWidget(auto_all_btn)

        nav.addStretch()
        report_btn = QPushButton('Generate Report')
        report_btn.clicked.connect(self._on_report)
        nav.addWidget(report_btn)

        finish_btn = QPushButton('Finish')
        finish_btn.setStyleSheet('background-color: #27ae60; color: white; font-weight: bold;')
        finish_btn.clicked.connect(self.accept)
        nav.addWidget(finish_btn)

        cancel_btn = QPushButton('Cancel'); cancel_btn.clicked.connect(self.reject)
        nav.addWidget(cancel_btn)
        rl.addLayout(nav)

        layout.addWidget(right, 1)

        # Click connection
        self.canvas.canvas.mpl_connect('button_press_event', self._on_click)
        self.palette_combo.currentTextChanged.connect(lambda: self._replot())
        self.alpha_spin.valueChanged.connect(lambda: self._replot())
        self.lw_spin.valueChanged.connect(lambda: self._replot())

        self.prof_list.setCurrentRow(0)

    # ── Pick mode ────────────────────────────────────────────────────

    def _set_pick(self, mode):
        self._pick_mode = mode
        self.f0_btn.setChecked(mode == 'f0')
        self.sec_btn.setChecked(mode == 'secondary')

    def _clear_secondary(self):
        r = self._results[self._current]
        r['secondary'] = []
        self._replot()

    # ── Navigation ───────────────────────────────────────────────────

    def _on_profile_clicked(self, row):
        self._current = min(row, len(self._results) - 1)
        self._replot()

    def _on_prev(self):
        if self._current > 0:
            self._current -= 1
            self.prof_list.setCurrentRow(self._current)

    def _on_next(self):
        if self._current < len(self._results):
            self._current += 1
            self.prof_list.setCurrentRow(min(self._current, self.prof_list.count() - 1))

    # ── Plotting ─────────────────────────────────────────────────────

    def _replot(self):
        import matplotlib.pyplot as plt
        self.canvas.clear()
        ax = self.canvas.add_subplot(111)
        cmap = plt.cm.get_cmap(self.palette_combo.currentText())
        alpha = self.alpha_spin.value()
        lw = self.lw_spin.value()

        n = len(self._results)
        picked = sum(1 for r in self._results if r['f0'] is not None)
        self.info_label.setText(
            f'Profile {self._current + 1} of {n + 1}: '
            f'{self._results[self._current]["name"] if self._current < n else "Median"} '
            f'| Peaks selected: {picked}/{n}')

        for i, r in enumerate(self._results):
            if r['freqs'] is None:
                continue
            is_cur = (i == self._current)
            ax.plot(r['freqs'], r['amps'],
                    color=cmap(i % 10),
                    alpha=1.0 if is_cur else alpha,
                    lw=2.0 if is_cur else lw,
                    label=r['name'] if is_cur else None)
            if r['f0']:
                ax.plot(r['f0'][0], r['f0'][1],
                        marker=self.f0_shape.currentText(),
                        color=self.f0_color.currentText(),
                        ms=12 if is_cur else 8,
                        alpha=1.0 if is_cur else 0.5)
            for sf, sa, _ in r['secondary']:
                ax.plot(sf, sa,
                        marker=self.sec_shape.currentText(),
                        color=self.sec_color.currentText(),
                        ms=10 if is_cur else 6,
                        alpha=1.0 if is_cur else 0.4)

        # Median
        if self._median is not None:
            med_f, med_a = self._median
            ax.plot(med_f, med_a, 'k-', lw=self.med_lw.value(), label='Median')

        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('H/V Amplitude')
        if any(r['freqs'] is not None for r in self._results):
            ax.legend(fontsize=7, loc='upper right')
        self.canvas.tight_layout()
        self.canvas.draw()

        self._update_sel()
        self._update_list_marks()

    def _update_list_marks(self):
        for i in range(len(self._results)):
            name = self._results[i]['name']
            mark = '✓ ' if self._results[i]['f0'] else ''
            self.prof_list.item(i).setText(f'{mark}{name}')

    def _update_sel(self):
        if self._current >= len(self._results):
            self.sel_label.setText('Median curve')
            return
        r = self._results[self._current]
        if r['f0']:
            self.sel_label.setText(f'f0 = {r["f0"][0]:.2f} Hz, A = {r["f0"][1]:.2f}')
        else:
            self.sel_label.setText('Click to select')

    # ── Click picking ────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes is None or self._current >= len(self._results):
            return
        r = self._results[self._current]
        if r['freqs'] is None:
            return
        log_f = np.log10(np.maximum(r['freqs'], 1e-10))
        log_c = np.log10(max(event.xdata, 1e-10))
        idx = int(np.argmin(np.abs(log_f - log_c)))
        freq, amp = float(r['freqs'][idx]), float(r['amps'][idx])

        if self._pick_mode == 'f0':
            r['f0'] = (freq, amp, idx)
        else:
            r['secondary'].append((freq, amp, idx))
        self._replot()

    def _on_auto(self):
        if self._current >= len(self._results):
            return
        r = self._results[self._current]
        if r['amps'] is not None:
            idx = int(np.argmax(r['amps']))
            r['f0'] = (float(r['freqs'][idx]), float(r['amps'][idx]), idx)
        self._replot()

    def _on_skip(self):
        self._on_auto()
        self._on_next()

    def _on_auto_all(self):
        for r in self._results:
            if r['f0'] is None and r['amps'] is not None:
                idx = int(np.argmax(r['amps']))
                r['f0'] = (float(r['freqs'][idx]), float(r['amps'][idx]), idx)
        self._compute_median()
        self._replot()

    # ── Median ───────────────────────────────────────────────────────

    def _compute_median(self):
        valid = [r for r in self._results if r['freqs'] is not None]
        if len(valid) < 2:
            return
        fmin = max(r['freqs'][0] for r in valid)
        fmax = min(r['freqs'][-1] for r in valid)
        common_f = np.geomspace(fmin, fmax, 200)
        interp_amps = []
        for r in valid:
            interp_amps.append(np.interp(common_f, r['freqs'], r['amps']))
        self._median = (common_f, np.median(interp_amps, axis=0))

    # ── Report ───────────────────────────────────────────────────────

    def _on_report(self):
        if not self._output_dir:
            self._output_dir = QFileDialog.getExistingDirectory(self, 'Output Directory')
        if not self._output_dir:
            return
        self._compute_median()
        self._save_outputs(self._output_dir)
        QMessageBox.information(self, 'Report', f'Results saved to {self._output_dir}')

    def _save_outputs(self, base: str):
        os.makedirs(base, exist_ok=True)
        for r in self._results:
            if r['freqs'] is None:
                continue
            d = os.path.join(base, r['name'])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'hv_curve.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frequency_hz', 'amplitude'])
                for fr, am in zip(r['freqs'], r['amps']):
                    w.writerow([f'{fr:.6f}', f'{am:.6f}'])
            with open(os.path.join(d, 'peak_info.txt'), 'w') as f:
                if r['f0']:
                    f.write(f'f0: {r["f0"][0]:.4f} Hz  amplitude: {r["f0"][1]:.4f}\n')
                for i, (sf, sa, si) in enumerate(r['secondary']):
                    f.write(f'secondary_{i+1}: {sf:.4f} Hz  amplitude: {sa:.4f}\n')

        # Combined summary
        with open(os.path.join(base, 'combined_summary.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['profile', 'f0_hz', 'f0_amp'])
            for r in self._results:
                f0 = r.get('f0')
                w.writerow([r['name'], f'{f0[0]:.4f}' if f0 else '', f'{f0[1]:.4f}' if f0 else ''])

        # Median
        if self._median:
            with open(os.path.join(base, 'median_hv_curve.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frequency_hz', 'amplitude'])
                for fr, am in zip(*self._median):
                    w.writerow([f'{fr:.6f}', f'{am:.6f}'])

        # Combined figure
        for fmt in ('png', 'pdf'):
            self.canvas.figure.savefig(
                os.path.join(base, f'combined_hv_curves.{fmt}'),
                dpi=300, bbox_inches='tight')
