"""
Forward modeling view — HV curve canvas with peak picking.

Displays the computed HV curve (and optionally Vs profile) from a single
forward model.  Supports interactive f0/secondary peak selection by
clicking on the curve, log-scale toggles, grid, and save-results.
"""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QToolBar, QCheckBox,
    QDoubleSpinBox, QPushButton, QLabel, QFileDialog, QMessageBox,
)

from ..widgets.plot_canvas import PlotCanvas


class ForwardView(QWidget):
    """HV forward-model curve view with interactive peak picking."""

    peak_selected = pyqtSignal(float, float, int)  # freq, amp, index

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._pick_mode = 'f0'  # 'f0' or 'secondary'
        self._click_enabled = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # ── Canvas ───────────────────────────────────────────────────
        self.canvas = PlotCanvas(figsize=(10, 5), dpi=100)
        layout.addWidget(self.canvas, 1)

        # ── Plot toolbar ─────────────────────────────────────────────
        tb = QHBoxLayout()

        self.cb_logx = QCheckBox('Log X')
        self.cb_logx.setChecked(True)
        self.cb_logy = QCheckBox('Log Y')
        self.cb_grid = QCheckBox('Grid')
        self.cb_grid.setChecked(True)
        self.cb_vs = QCheckBox('Show Vs')
        self.cb_vs.setChecked(True)

        tb.addWidget(self.cb_logx)
        tb.addWidget(self.cb_logy)
        tb.addWidget(self.cb_grid)
        tb.addWidget(self.cb_vs)

        tb.addWidget(QLabel('HS %:'))
        self.hs_spin = QDoubleSpinBox()
        self.hs_spin.setRange(10, 100)
        self.hs_spin.setValue(25)
        self.hs_spin.setSuffix('%')
        tb.addWidget(self.hs_spin)

        tb.addStretch()

        self.f0_btn = QPushButton('🔘 Select f0')
        self.f0_btn.setCheckable(True)
        self.f0_btn.setChecked(True)
        self.f0_btn.setStyleSheet('QPushButton:checked { background-color: #27ae60; color: white; }')
        tb.addWidget(self.f0_btn)

        self.sec_btn = QPushButton('🔘 Secondary')
        self.sec_btn.setCheckable(True)
        self.sec_btn.setStyleSheet('QPushButton:checked { background-color: #e67e22; color: white; }')
        tb.addWidget(self.sec_btn)

        self.clear_sec_btn = QPushButton('Clear Secondary')
        tb.addWidget(self.clear_sec_btn)

        layout.addLayout(tb)

        # ── Selection info ───────────────────────────────────────────
        self.info_label = QLabel('Click on curve to select peak')
        self.info_label.setStyleSheet('color: gray; font-size: 10px;')
        layout.addWidget(self.info_label)

        # ── Save button ──────────────────────────────────────────────
        save_row = QHBoxLayout()
        save_row.addStretch()
        self.save_btn = QPushButton('Save Results…')
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self.save_btn)
        layout.addLayout(save_row)

        # ── Connections ──────────────────────────────────────────────
        self.f0_btn.toggled.connect(self._on_f0_toggle)
        self.sec_btn.toggled.connect(self._on_sec_toggle)
        self.clear_sec_btn.clicked.connect(self._on_clear_secondary)
        self.cb_logx.toggled.connect(lambda: self._replot())
        self.cb_logy.toggled.connect(lambda: self._replot())
        self.cb_grid.toggled.connect(lambda: self._replot())
        self.cb_vs.toggled.connect(lambda: self._replot())

        self.canvas.canvas.mpl_connect('button_press_event', self._on_click)
        self.state.forward_result_ready.connect(self._replot)

    # ── Plotting ─────────────────────────────────────────────────────

    def _replot(self):
        """Redraw the HV curve (and Vs profile if toggled)."""
        freqs = self.state.forward_freqs
        amps = self.state.forward_amps
        if freqs is None or amps is None:
            return

        self.canvas.clear()
        show_vs = self.cb_vs.isChecked() and self.state.active_profile is not None

        if show_vs:
            gs = self.canvas.figure.add_gridspec(1, 2, width_ratios=[4, 1])
            ax_hv = self.canvas.figure.add_subplot(gs[0])
            ax_vs = self.canvas.figure.add_subplot(gs[1])
        else:
            ax_hv = self.canvas.add_subplot(111)
            ax_vs = None

        # HV curve
        ax_hv.plot(freqs, amps, 'b-', lw=1.5, label='H/V')

        # Peak markers
        f0 = self.state.forward_f0
        if f0:
            ax_hv.plot(f0[0], f0[1], 'r*', ms=12, zorder=5, label=f'f0 = {f0[0]:.2f} Hz')
            ax_hv.axvline(f0[0], color='r', ls='--', alpha=0.5)

        for sf, sa, si in self.state.forward_secondary:
            ax_hv.plot(sf, sa, 'k*', ms=10, zorder=5)
            ax_hv.axvline(sf, color='k', ls=':', alpha=0.4)

        if self.cb_logx.isChecked():
            ax_hv.set_xscale('log')
        if self.cb_logy.isChecked():
            ax_hv.set_yscale('log')
        ax_hv.grid(self.cb_grid.isChecked(), alpha=0.3)
        ax_hv.set_xlabel('Frequency (Hz)')
        ax_hv.set_ylabel('H/V Amplitude')
        ax_hv.legend(fontsize=8)

        # Vs profile
        if ax_vs and self.state.active_profile:
            self._draw_vs_profile(ax_vs)

        self.canvas.tight_layout()
        self.canvas.draw()
        self.save_btn.setEnabled(True)
        self._update_info()

    def _draw_vs_profile(self, ax):
        """Draw Vs vs depth step-function."""
        profile = self.state.active_profile
        depths, vs_vals = [0.0], []
        for layer in profile.layers:
            vs_vals.extend([layer.vs, layer.vs])
            depths.append(depths[-1])
            if layer.is_halfspace:
                total = sum(l.thickness for l in profile.layers if not l.is_halfspace)
                depths.append(depths[-1] + total * self.hs_spin.value() / 100.0)
            else:
                depths.append(depths[-1] + layer.thickness)
        depths = depths[:len(vs_vals)]
        ax.step(vs_vals, depths, where='post', color='teal', lw=1.5)
        ax.invert_yaxis()
        ax.set_xlabel('Vs (m/s)', fontsize=8)
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # ── Interactive picking ──────────────────────────────────────────

    def _on_click(self, event):
        if not self._click_enabled or event.inaxes is None:
            return
        freqs = self.state.forward_freqs
        amps = self.state.forward_amps
        if freqs is None:
            return

        # Find nearest point (log-distance for semilog)
        xclick = event.xdata
        log_freqs = np.log10(np.maximum(freqs, 1e-10))
        log_click = np.log10(max(xclick, 1e-10))
        dists = np.abs(log_freqs - log_click)
        idx = int(np.argmin(dists))

        freq, amp = float(freqs[idx]), float(amps[idx])

        if self._pick_mode == 'f0':
            self.state.forward_f0 = (freq, amp, idx)
        else:
            self.state.forward_secondary.append((freq, amp, idx))

        self.state.peak_changed.emit()
        self._replot()

    def _on_f0_toggle(self, checked):
        if checked:
            self.sec_btn.setChecked(False)
            self._pick_mode = 'f0'

    def _on_sec_toggle(self, checked):
        if checked:
            self.f0_btn.setChecked(False)
            self._pick_mode = 'secondary'

    def _on_clear_secondary(self):
        self.state.forward_secondary.clear()
        self.state.peak_changed.emit()
        self._replot()

    def _update_info(self):
        parts = []
        f0 = self.state.forward_f0
        if f0:
            parts.append(f'f0 = {f0[0]:.2f} Hz (A = {f0[1]:.2f})')
        sec = self.state.forward_secondary
        if sec:
            parts.append(f'{len(sec)} secondary peak(s)')
        self.info_label.setText(' | '.join(parts) if parts else 'Click on curve to select peak')

    # ── Save ─────────────────────────────────────────────────────────

    def _on_save(self):
        path = QFileDialog.getExistingDirectory(self, 'Save Results To')
        if not path:
            return
        import os, csv
        freqs = self.state.forward_freqs
        amps = self.state.forward_amps

        # CSV
        csv_path = os.path.join(path, 'hv_curve.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frequency_hz', 'amplitude'])
            for fr, am in zip(freqs, amps):
                w.writerow([f'{fr:.6f}', f'{am:.6f}'])

        # Peak info
        peak_path = os.path.join(path, 'peak_info.txt')
        with open(peak_path, 'w') as f:
            f0 = self.state.forward_f0
            if f0:
                f.write(f'f0: {f0[0]:.4f} Hz  amplitude: {f0[1]:.4f}  index: {f0[2]}\n')
            for i, (sf, sa, si) in enumerate(self.state.forward_secondary):
                f.write(f'secondary_{i+1}: {sf:.4f} Hz  amplitude: {sa:.4f}  index: {si}\n')

        # Figure
        for fmt in ('png', 'pdf'):
            fig_path = os.path.join(path, f'hv_forward_curve.{fmt}')
            self.canvas.figure.savefig(fig_path, dpi=300, bbox_inches='tight')

        self.state.status_message.emit(f'Results saved to {path}')

    def _on_save_to_dir(self, path: str):
        """Save results to an explicit directory (called from main window)."""
        if self.state.forward_freqs is None:
            return
        import os, csv
        freqs = self.state.forward_freqs
        amps = self.state.forward_amps
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'hv_curve.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frequency_hz', 'amplitude'])
            for fr, am in zip(freqs, amps):
                w.writerow([f'{fr:.6f}', f'{am:.6f}'])
        with open(os.path.join(path, 'peak_info.txt'), 'w') as f:
            f0 = self.state.forward_f0
            if f0:
                f.write(f'f0: {f0[0]:.4f} Hz  amplitude: {f0[1]:.4f}\n')
            for i, (sf, sa, si) in enumerate(self.state.forward_secondary):
                f.write(f'secondary_{i+1}: {sf:.4f} Hz  amplitude: {sa:.4f}\n')
        for fmt in ('png', 'pdf'):
            self.canvas.figure.savefig(
                os.path.join(path, f'hv_forward_curve.{fmt}'),
                dpi=300, bbox_inches='tight')
