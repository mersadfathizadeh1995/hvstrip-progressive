"""
Interactive peak picker dialog — step-by-step peak selection.

Allows the user to walk through each stripping step, click on the
HV curve to select f0, undo, auto-detect, skip, or finish.
"""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QSplitter, QWidget, QMessageBox,
)

from ..widgets.plot_canvas import PlotCanvas


class InteractivePeakPicker(QDialog):
    """Modal dialog for step-by-step interactive peak selection."""

    peaks_selected = pyqtSignal(dict)

    def __init__(self, steps: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Interactive Peak Picker')
        self.resize(1100, 700)
        self.setMinimumSize(900, 600)

        self._steps = steps
        self._current = 0
        self._picks = {}  # {step_name: (freq, amp, idx)}
        self._undo_stack = []
        self._click_enabled = True

        layout = QHBoxLayout(self)

        # ── Left: step list ──────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(200)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        ll.addWidget(QLabel('Steps:'))
        self.step_list = QListWidget()
        for i, step in enumerate(steps):
            name = step.get('name', f'Step {i}')
            self.step_list.addItem(name)
        self.step_list.currentRowChanged.connect(self._on_step_clicked)
        ll.addWidget(self.step_list)
        layout.addWidget(left)

        # ── Right: canvas + controls ─────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel()
        self.info_label.setStyleSheet('font-weight: bold;')
        rl.addWidget(self.info_label)

        self.canvas = PlotCanvas(figsize=(12, 5), dpi=100)
        rl.addWidget(self.canvas, 1)

        # Pick toggle
        ctrl1 = QHBoxLayout()
        self.pick_btn = QPushButton('Click Selection: ON')
        self.pick_btn.setCheckable(True)
        self.pick_btn.setChecked(True)
        self.pick_btn.setStyleSheet('QPushButton:checked { background-color: #27ae60; color: white; }')
        self.pick_btn.toggled.connect(self._toggle_pick)
        ctrl1.addWidget(self.pick_btn)

        self.vs_btn = QPushButton('Show Vs Profile')
        self.vs_btn.setCheckable(True)
        self.vs_btn.setChecked(True)
        self.vs_btn.setStyleSheet('QPushButton:checked { background-color: #2E86AB; color: white; }')
        self.vs_btn.toggled.connect(lambda: self._plot_current())
        ctrl1.addWidget(self.vs_btn)
        ctrl1.addStretch()
        rl.addLayout(ctrl1)

        self.sel_label = QLabel('Click on curve to select f0')
        self.sel_label.setStyleSheet('color: gray;')
        rl.addWidget(self.sel_label)

        # Navigation
        ctrl2 = QHBoxLayout()
        self.prev_btn = QPushButton('◀ Previous')
        self.prev_btn.clicked.connect(self._on_prev)
        ctrl2.addWidget(self.prev_btn)
        self.next_btn = QPushButton('Next ▶')
        self.next_btn.clicked.connect(self._on_next)
        ctrl2.addWidget(self.next_btn)

        ctrl2.addWidget(self._sep())

        undo_btn = QPushButton('Undo')
        undo_btn.clicked.connect(self._on_undo)
        ctrl2.addWidget(undo_btn)
        auto_btn = QPushButton('Auto-detect')
        auto_btn.clicked.connect(self._on_auto)
        ctrl2.addWidget(auto_btn)
        skip_btn = QPushButton('Skip')
        skip_btn.clicked.connect(self._on_skip)
        ctrl2.addWidget(skip_btn)

        ctrl2.addWidget(self._sep())

        finish_btn = QPushButton('Finish')
        finish_btn.setStyleSheet('background-color: #27ae60; color: white; font-weight: bold;')
        finish_btn.clicked.connect(self._on_finish)
        ctrl2.addWidget(finish_btn)
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        ctrl2.addWidget(cancel_btn)
        rl.addLayout(ctrl2)

        layout.addWidget(right, 1)

        # Matplotlib click
        self.canvas.canvas.mpl_connect('button_press_event', self._on_click)

        # Show first step
        self.step_list.setCurrentRow(0)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sep():
        lbl = QLabel('|')
        lbl.setStyleSheet('color: #ccc;')
        return lbl

    def _step_name(self, idx):
        return self._steps[idx].get('name', f'Step {idx}')

    def _update_list_marks(self):
        for i in range(self.step_list.count()):
            name = self._step_name(i)
            mark = '✓ ' if name in self._picks else ''
            self.step_list.item(i).setText(f'{mark}{name}')

    # ── Navigation ───────────────────────────────────────────────────

    def _on_step_clicked(self, row):
        if 0 <= row < len(self._steps):
            self._current = row
            self._plot_current()

    def _on_prev(self):
        if self._current > 0:
            self._current -= 1
            self.step_list.setCurrentRow(self._current)

    def _on_next(self):
        if self._current < len(self._steps) - 1:
            self._current += 1
            self.step_list.setCurrentRow(self._current)

    # ── Plotting ─────────────────────────────────────────────────────

    def _plot_current(self):
        step = self._steps[self._current]
        freqs = step.get('freqs')
        amps = step.get('amps')
        n = len(self._steps)
        picked = sum(1 for s in self._steps if self._step_name(self._steps.index(s)) in self._picks)

        self.info_label.setText(
            f'Step {self._current + 1} of {n}: {self._step_name(self._current)} '
            f'| Peaks selected: {picked}/{n}')
        self.prev_btn.setEnabled(self._current > 0)
        self.next_btn.setEnabled(self._current < n - 1)

        self.canvas.clear()
        show_vs = self.vs_btn.isChecked() and step.get('profile') is not None

        if show_vs:
            gs = self.canvas.figure.add_gridspec(1, 2, width_ratios=[4, 1])
            ax = self.canvas.figure.add_subplot(gs[0])
            ax_vs = self.canvas.figure.add_subplot(gs[1])
        else:
            ax = self.canvas.add_subplot(111)
            ax_vs = None

        if freqs is not None and amps is not None:
            ax.plot(freqs, amps, 'b-', lw=1.5, label='H/V')

            # Auto-detected peak (gray circle)
            auto_f0 = step.get('f0')
            if auto_f0:
                ax.plot(auto_f0[0], auto_f0[1], 'o', color='gray', ms=10,
                        alpha=0.5, label='Auto f0')

            # Manual pick (red star)
            name = self._step_name(self._current)
            if name in self._picks:
                pf, pa, pi = self._picks[name]
                ax.plot(pf, pa, 'r*', ms=14, zorder=5, label=f'f0 = {pf:.2f} Hz')
                ax.axvline(pf, color='r', ls='--', alpha=0.5)

        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('H/V Amplitude')
        ax.legend(fontsize=8)

        if ax_vs and step.get('profile'):
            self._draw_vs(ax_vs, step['profile'])

        self.canvas.tight_layout()
        self.canvas.draw()
        self._update_sel_label()

    def _draw_vs(self, ax, profile):
        depths, vs_vals = [0.0], []
        for layer in profile.layers:
            vs_vals.extend([layer.vs, layer.vs])
            depths.append(depths[-1])
            depths.append(depths[-1] + (layer.thickness if not layer.is_halfspace else 5))
        depths = depths[:len(vs_vals)]
        ax.step(vs_vals, depths, where='post', color='teal', lw=1.5)
        ax.invert_yaxis()
        ax.set_xlabel('Vs', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # ── Click picking ────────────────────────────────────────────────

    def _toggle_pick(self, on):
        self._click_enabled = on
        self.pick_btn.setText(f'Click Selection: {"ON" if on else "OFF"}')

    def _on_click(self, event):
        if not self._click_enabled or event.inaxes is None:
            return
        step = self._steps[self._current]
        freqs, amps = step.get('freqs'), step.get('amps')
        if freqs is None:
            return

        log_f = np.log10(np.maximum(freqs, 1e-10))
        log_c = np.log10(max(event.xdata, 1e-10))
        idx = int(np.argmin(np.abs(log_f - log_c)))
        name = self._step_name(self._current)

        prev = self._picks.get(name)
        self._undo_stack.append((name, prev))
        self._picks[name] = (float(freqs[idx]), float(amps[idx]), idx)

        self._update_list_marks()
        self._plot_current()

    def _on_undo(self):
        if not self._undo_stack:
            return
        name, prev = self._undo_stack.pop()
        if prev is None:
            self._picks.pop(name, None)
        else:
            self._picks[name] = prev
        self._update_list_marks()
        self._plot_current()

    def _on_auto(self):
        step = self._steps[self._current]
        amps = step.get('amps')
        freqs = step.get('freqs')
        if amps is None:
            return
        idx = int(np.argmax(amps))
        name = self._step_name(self._current)
        prev = self._picks.get(name)
        self._undo_stack.append((name, prev))
        self._picks[name] = (float(freqs[idx]), float(amps[idx]), idx)
        self._update_list_marks()
        self._plot_current()

    def _on_skip(self):
        self._on_auto()
        self._on_next()

    def _on_finish(self):
        unselected = [self._step_name(i) for i in range(len(self._steps))
                       if self._step_name(i) not in self._picks]
        if unselected:
            ret = QMessageBox.question(
                self, 'Unselected Steps',
                f'{len(unselected)} step(s) have no peak selected.\n'
                'Auto-detect for remaining?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ret == QMessageBox.Cancel:
                return
            if ret == QMessageBox.Yes:
                for name in unselected:
                    idx_s = next(i for i, s in enumerate(self._steps) if self._step_name(i) == name)
                    step = self._steps[idx_s]
                    amps = step.get('amps')
                    freqs = step.get('freqs')
                    if amps is not None:
                        idx = int(np.argmax(amps))
                        self._picks[name] = (float(freqs[idx]), float(amps[idx]), idx)

        self.peaks_selected.emit(self._picks)
        self.accept()

    def _update_sel_label(self):
        name = self._step_name(self._current)
        if name in self._picks:
            f, a, _ = self._picks[name]
            self.sel_label.setText(f'Selected: f0 = {f:.2f} Hz, A = {a:.2f}')
        else:
            self.sel_label.setText('Click on curve to select f0')

    def get_selected_peaks(self) -> dict:
        return dict(self._picks)
