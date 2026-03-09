"""
Strip results view — overlay of HV curves from all stripping steps.

Displays all computed steps from the progressive layer-stripping workflow
with step navigation, peak annotations, and per-step detail.
"""

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QCheckBox,
)

from ..widgets.plot_canvas import PlotCanvas


class StripView(QWidget):
    """Overlay view of all stripping steps with navigation."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._current_step = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # ── Canvas ───────────────────────────────────────────────────
        self.canvas = PlotCanvas(figsize=(10, 5), dpi=100)
        layout.addWidget(self.canvas, 1)

        # ── Info bar ─────────────────────────────────────────────────
        self.info_label = QLabel('No strip results yet.')
        self.info_label.setStyleSheet('font-weight: bold;')
        layout.addWidget(self.info_label)

        # ── Controls ─────────────────────────────────────────────────
        ctrl = QHBoxLayout()

        self.prev_btn = QPushButton('◀ Previous')
        self.prev_btn.clicked.connect(self._on_prev)
        ctrl.addWidget(self.prev_btn)

        self.step_combo = QComboBox()
        self.step_combo.currentIndexChanged.connect(self._on_step_changed)
        ctrl.addWidget(self.step_combo, 1)

        self.next_btn = QPushButton('Next ▶')
        self.next_btn.clicked.connect(self._on_next)
        ctrl.addWidget(self.next_btn)

        ctrl.addStretch()

        self.cb_overlay = QCheckBox('Show all steps')
        self.cb_overlay.setChecked(True)
        self.cb_overlay.toggled.connect(self._replot)
        ctrl.addWidget(self.cb_overlay)

        self.cb_peaks = QCheckBox('Show peaks')
        self.cb_peaks.setChecked(True)
        self.cb_peaks.toggled.connect(self._replot)
        ctrl.addWidget(self.cb_peaks)

        layout.addLayout(ctrl)

        # ── Step detail ──────────────────────────────────────────────
        self.detail_label = QLabel('')
        self.detail_label.setStyleSheet('color: gray; font-size: 10px;')
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

        # ── Connections ──────────────────────────────────────────────
        self.state.strip_result_ready.connect(self._on_results_ready)

    # ── Data handling ────────────────────────────────────────────────

    def _on_results_ready(self):
        """Called when strip workflow completes."""
        steps = self.state.strip_steps
        self.step_combo.blockSignals(True)
        self.step_combo.clear()
        for i, step in enumerate(steps):
            name = step.get('name', f'Step {i}')
            self.step_combo.addItem(name)
        self.step_combo.blockSignals(False)
        self._current_step = 0
        if steps:
            self.step_combo.setCurrentIndex(0)
        self._replot()

    # ── Navigation ───────────────────────────────────────────────────

    def _on_prev(self):
        if self._current_step > 0:
            self._current_step -= 1
            self.step_combo.setCurrentIndex(self._current_step)

    def _on_next(self):
        if self._current_step < len(self.state.strip_steps) - 1:
            self._current_step += 1
            self.step_combo.setCurrentIndex(self._current_step)

    def _on_step_changed(self, idx):
        if idx >= 0:
            self._current_step = idx
            self._replot()

    # ── Plotting ─────────────────────────────────────────────────────

    def _replot(self):
        steps = self.state.strip_steps
        if not steps:
            self.info_label.setText('No strip results yet.')
            return

        n = len(steps)
        self.info_label.setText(
            f'Step {self._current_step + 1} of {n}: '
            f'{steps[self._current_step].get("name", "")}')
        self.prev_btn.setEnabled(self._current_step > 0)
        self.next_btn.setEnabled(self._current_step < n - 1)

        self.canvas.clear()
        ax = self.canvas.add_subplot(111)

        cmap = __import__('matplotlib.pyplot', fromlist=['cm']).cm.get_cmap('tab10')

        for i, step in enumerate(steps):
            freqs = step.get('freqs')
            amps = step.get('amps')
            if freqs is None or amps is None:
                continue

            is_current = (i == self._current_step)
            show = self.cb_overlay.isChecked() or is_current
            if not show:
                continue

            alpha = 1.0 if is_current else 0.35
            lw = 2.0 if is_current else 1.0
            color = cmap(i % 10)
            label = step.get('name', f'Step {i}')
            ax.plot(freqs, amps, color=color, alpha=alpha, lw=lw, label=label)

            if self.cb_peaks.isChecked():
                f0 = step.get('f0')
                if f0:
                    marker_alpha = 1.0 if is_current else 0.4
                    ax.plot(f0[0], f0[1], '*', color=color,
                            ms=12 if is_current else 8,
                            alpha=marker_alpha, zorder=5)

        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('H/V Amplitude')
        ax.legend(fontsize=7, loc='upper right')

        self.canvas.tight_layout()
        self.canvas.draw()

        # Detail for current step
        step = steps[self._current_step]
        detail_parts = []
        f0 = step.get('f0')
        if f0:
            detail_parts.append(f'f0 = {f0[0]:.2f} Hz (A = {f0[1]:.2f})')
        n_layers = step.get('n_layers')
        if n_layers is not None:
            detail_parts.append(f'Layers: {n_layers}')
        removed = step.get('layers_removed')
        if removed:
            detail_parts.append(f'Removed: {removed}')
        self.detail_label.setText(' | '.join(detail_parts))
