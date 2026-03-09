"""
Results dock panel — step table, peak summary, Vs30 display.

Shows a tabular summary of the stripping workflow results and
key analysis outputs in the right dock.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QGroupBox, QFormLayout, QHeaderView, QAbstractItemView,
)

from ..widgets.collapsible_group import CollapsibleGroup


class ResultsPanel(QWidget):
    """Step table + peak summary for the right dock."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Step table ───────────────────────────────────────────────
        grp = CollapsibleGroup('Stripping Steps')
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Step', 'Layers', 'f0 (Hz)', 'Amplitude'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        grp.add_widget(self.table)
        layout.addWidget(grp)

        # ── Peak summary ─────────────────────────────────────────────
        peak_grp = CollapsibleGroup('Peak Summary')
        pform = QFormLayout()
        self.f0_label = QLabel('—')
        self.f0_label.setStyleSheet('font-weight: bold; color: #E63946;')
        pform.addRow('Primary (f0):', self.f0_label)
        self.f1_label = QLabel('—')
        pform.addRow('Shallow (f1):', self.f1_label)
        self.ratio_label = QLabel('—')
        pform.addRow('f1/f0 Ratio:', self.ratio_label)
        self.separation_label = QLabel('—')
        pform.addRow('Separated:', self.separation_label)
        peak_grp.add_layout(pform)
        layout.addWidget(peak_grp)

        # ── Vs30 / analysis summary ──────────────────────────────────
        analysis_grp = CollapsibleGroup('Analysis')
        aform = QFormLayout()
        self.vs30_label = QLabel('—')
        self.vs30_label.setStyleSheet('font-weight: bold;')
        aform.addRow('Vs30:', self.vs30_label)
        self.controlling_label = QLabel('—')
        aform.addRow('Controlling Interface:', self.controlling_label)
        self.max_shift_label = QLabel('—')
        aform.addRow('Max Freq Shift:', self.max_shift_label)
        analysis_grp.add_layout(aform)
        layout.addWidget(analysis_grp)

        layout.addStretch()

        # ── Connections ──────────────────────────────────────────────
        self.state.strip_result_ready.connect(self._update)
        self.state.forward_result_ready.connect(self._update_forward)

    def _update(self):
        """Populate from strip workflow results."""
        steps = self.state.strip_steps
        self.table.setRowCount(len(steps))
        for i, step in enumerate(steps):
            self.table.setItem(i, 0, QTableWidgetItem(step.get('name', f'Step {i}')))
            n_layers = step.get('n_layers', '?')
            self.table.setItem(i, 1, QTableWidgetItem(str(n_layers)))
            f0 = step.get('f0')
            if f0:
                self.table.setItem(i, 2, QTableWidgetItem(f'{f0[0]:.3f}'))
                self.table.setItem(i, 3, QTableWidgetItem(f'{f0[1]:.3f}'))
            else:
                self.table.setItem(i, 2, QTableWidgetItem('—'))
                self.table.setItem(i, 3, QTableWidgetItem('—'))

        # Dual-resonance
        dr = self.state.strip_dual_result
        if dr:
            f0 = dr.get('f0', '—')
            f1 = dr.get('f1', '—')
            self.f0_label.setText(f'{f0:.3f} Hz' if isinstance(f0, (int, float)) else str(f0))
            self.f1_label.setText(f'{f1:.3f} Hz' if isinstance(f1, (int, float)) else str(f1))
            ratio = dr.get('freq_ratio', None)
            self.ratio_label.setText(f'{ratio:.3f}' if ratio else '—')
            sep = dr.get('separated', None)
            self.separation_label.setText('✅ Yes' if sep else '❌ No' if sep is not None else '—')
        else:
            self.f0_label.setText('—')
            self.f1_label.setText('—')
            self.ratio_label.setText('—')
            self.separation_label.setText('—')

        # Vs30
        vs30 = self.state.strip_vs30
        self.vs30_label.setText(f'{vs30:.1f} m/s' if vs30 else '—')

        # Controlling interface
        if steps:
            max_shift = 0
            controlling = '—'
            for i, step in enumerate(steps):
                shift = step.get('freq_shift', 0)
                if abs(shift) > abs(max_shift):
                    max_shift = shift
                    controlling = step.get('name', f'Step {i}')
            self.controlling_label.setText(controlling)
            self.max_shift_label.setText(f'{max_shift:.3f} Hz')

    def _update_forward(self):
        """Update from a single forward model result."""
        f0 = self.state.forward_f0
        if f0:
            self.f0_label.setText(f'{f0[0]:.3f} Hz (A = {f0[1]:.3f})')
        vs30 = self.state.strip_vs30
        self.vs30_label.setText(f'{vs30:.1f} m/s' if vs30 else '—')
