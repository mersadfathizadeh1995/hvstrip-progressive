"""
Output viewer dialog — load and visualize previously saved results.

Provides folder selection, layer visibility tree, plot settings,
and figure export.
"""

import os
import csv
import glob
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QFileDialog, QComboBox, QDoubleSpinBox,
    QCheckBox, QFormLayout, QWidget, QGroupBox, QMessageBox,
)

from ..widgets.plot_canvas import PlotCanvas
from ..widgets.collapsible_group import CollapsibleGroup

_PALETTES = [
    'tab10', 'tab20', 'Set1', 'Set2', 'Paired', 'Dark2',
    'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'Spectral',
]


class OutputViewerDialog(QDialog):
    """Dialog for loading and visualizing saved HV Strip results."""

    def __init__(self, initial_dir: str = '', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Output Viewer')
        self.resize(1200, 700)
        self.setMinimumSize(900, 600)

        self._root_dir = initial_dir
        self._data = {}  # name → {freqs, amps, f0, secondary}

        layout = QHBoxLayout(self)

        # ── Left ─────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(260)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        browse_btn = QPushButton('Browse Folder...')
        browse_btn.clicked.connect(self._on_browse)
        ll.addWidget(browse_btn)

        self.path_label = QLabel(initial_dir or '(no folder selected)')
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet('color: gray;')
        ll.addWidget(self.path_label)

        ll.addWidget(QLabel('Profiles:'))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Name', 'Visible'])
        self.tree.setColumnWidth(0, 160)
        self.tree.itemChanged.connect(lambda: self._replot())
        ll.addWidget(self.tree, 1)

        show_all = QPushButton('Show All')
        show_all.clicked.connect(lambda: self._set_all_visible(True))
        hide_all = QPushButton('Hide All')
        hide_all.clicked.connect(lambda: self._set_all_visible(False))
        row = QHBoxLayout()
        row.addWidget(show_all); row.addWidget(hide_all)
        ll.addLayout(row)

        # Settings
        settings = CollapsibleGroup('Plot Settings', collapsed=True)
        form = QFormLayout()

        self.palette = QComboBox()
        self.palette.addItems(_PALETTES)
        form.addRow('Palette:', self.palette)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 1.0); self.alpha_spin.setValue(0.7)
        self.alpha_spin.setDecimals(2)
        form.addRow('Alpha:', self.alpha_spin)

        self.lw = QDoubleSpinBox()
        self.lw.setRange(0.3, 5.0); self.lw.setValue(1.5)
        form.addRow('Line width:', self.lw)

        self.show_peaks = QCheckBox('Show peaks')
        self.show_peaks.setChecked(True)
        form.addRow(self.show_peaks)

        self.log_x = QCheckBox('Log X')
        self.log_x.setChecked(True)
        form.addRow(self.log_x)

        settings.add_layout(form)
        ll.addWidget(settings)

        self.palette.currentTextChanged.connect(lambda: self._replot())
        self.alpha_spin.valueChanged.connect(lambda: self._replot())
        self.lw.valueChanged.connect(lambda: self._replot())
        self.show_peaks.toggled.connect(lambda: self._replot())
        self.log_x.toggled.connect(lambda: self._replot())

        layout.addWidget(left)

        # ── Right: canvas ────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.canvas = PlotCanvas(figsize=(12, 6), dpi=100)
        rl.addWidget(self.canvas, 1)

        btn_row = QHBoxLayout()
        save_btn = QPushButton('Save Figure')
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        rl.addLayout(btn_row)

        layout.addWidget(right, 1)

        if initial_dir:
            self._load_dir(initial_dir)

    def _on_browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if d:
            self._load_dir(d)

    def _load_dir(self, path: str):
        self._root_dir = path
        self.path_label.setText(path)
        self._data.clear()
        self.tree.clear()

        # Look for hv_curve.csv in subdirectories or root
        for entry in sorted(os.listdir(path)):
            sub = os.path.join(path, entry)
            csv_path = os.path.join(sub, 'hv_curve.csv') if os.path.isdir(sub) else None
            if csv_path and os.path.isfile(csv_path):
                self._load_profile(entry, sub)
            elif entry.endswith('.csv') and 'hv_curve' in entry.lower():
                self._load_single_csv(entry, os.path.join(path, entry))

        # Median
        med = os.path.join(path, 'median_hv_curve.csv')
        if os.path.isfile(med):
            self._load_single_csv('Median', med)

        self._replot()

    def _load_profile(self, name: str, folder: str):
        csv_path = os.path.join(folder, 'hv_curve.csv')
        freqs, amps = self._read_csv(csv_path)
        if freqs is None:
            return
        entry = {'freqs': freqs, 'amps': amps, 'f0': None, 'secondary': []}

        peak_path = os.path.join(folder, 'peak_info.txt')
        if os.path.isfile(peak_path):
            entry.update(self._read_peaks(peak_path))

        self._data[name] = entry
        item = QTreeWidgetItem([name, ''])
        item.setCheckState(1, Qt.Checked)
        self.tree.addTopLevelItem(item)

    def _load_single_csv(self, name: str, path: str):
        freqs, amps = self._read_csv(path)
        if freqs is None:
            return
        self._data[name] = {'freqs': freqs, 'amps': amps, 'f0': None, 'secondary': []}
        item = QTreeWidgetItem([name, ''])
        item.setCheckState(1, Qt.Checked)
        self.tree.addTopLevelItem(item)

    def _read_csv(self, path: str):
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                freqs, amps = [], []
                for row in reader:
                    if len(row) >= 2:
                        freqs.append(float(row[0]))
                        amps.append(float(row[1]))
            return np.array(freqs), np.array(amps)
        except Exception:
            return None, None

    def _read_peaks(self, path: str):
        info = {'f0': None, 'secondary': []}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('f0:'):
                        parts = line.split()
                        freq = float(parts[1])
                        amp = float(parts[4]) if len(parts) > 4 else 0
                        info['f0'] = (freq, amp)
                    elif line.startswith('secondary'):
                        parts = line.split()
                        freq = float(parts[1])
                        amp = float(parts[4]) if len(parts) > 4 else 0
                        info['secondary'].append((freq, amp))
        except Exception:
            pass
        return info

    def _set_all_visible(self, visible: bool):
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setCheckState(1, Qt.Checked if visible else Qt.Unchecked)
        self._replot()

    def _replot(self):
        import matplotlib.pyplot as plt
        self.canvas.clear()
        ax = self.canvas.add_subplot(111)
        cmap = plt.cm.get_cmap(self.palette.currentText())
        alpha = self.alpha_spin.value()
        lw = self.lw.value()

        visible = set()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.checkState(1) == Qt.Checked:
                visible.add(item.text(0))

        idx = 0
        for name in visible:
            d = self._data.get(name)
            if d is None or d['freqs'] is None:
                continue
            color = cmap(idx % 10)
            ax.plot(d['freqs'], d['amps'], color=color, alpha=alpha, lw=lw, label=name)

            if self.show_peaks.isChecked() and d['f0']:
                ax.plot(d['f0'][0], d['f0'][1], '*', color='red', ms=10)
            if self.show_peaks.isChecked():
                for sf, sa in d['secondary']:
                    ax.plot(sf, sa, 'D', color='black', ms=7)
            idx += 1

        if self.log_x.isChecked():
            ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('H/V Amplitude')
        if visible:
            ax.legend(fontsize=7, loc='upper right')
        self.canvas.tight_layout()
        self.canvas.draw()

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Figure', '',
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg)')
        if path:
            self.canvas.figure.savefig(path, dpi=300, bbox_inches='tight')
