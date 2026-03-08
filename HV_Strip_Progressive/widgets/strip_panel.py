"""
HV strip workflow control panel.

Provides input source selection (HVf / Dinver), workflow options
(report, interactive peaks, dual-resonance), output directory,
and Run / Cancel buttons.
"""

import os
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QRadioButton, QButtonGroup, QLabel, QLineEdit, QPushButton,
    QCheckBox, QFileDialog, QGroupBox,
)

from ..widgets.collapsible_group import CollapsibleGroup


class StripPanel(QWidget):
    """Controls for the progressive layer-stripping workflow."""

    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    dual_resonance_settings_requested = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Input source ─────────────────────────────────────────────
        src_grp = CollapsibleGroup('Input Source')
        slay = QVBoxLayout()

        self._src_group = QButtonGroup(self)
        self.rb_hvf = QRadioButton('HVf File')
        self.rb_dinver = QRadioButton('Dinver Output')
        self.rb_hvf.setChecked(True)
        self._src_group.addButton(self.rb_hvf, 0)
        self._src_group.addButton(self.rb_dinver, 1)
        slay.addWidget(self.rb_hvf)
        slay.addWidget(self.rb_dinver)

        # HVf file row
        hvf_row = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText('Model file path…')
        browse_btn = QPushButton('Browse…')
        browse_btn.clicked.connect(self._browse_model)
        hvf_row.addWidget(self.model_edit, 1)
        hvf_row.addWidget(browse_btn)
        slay.addLayout(hvf_row)

        # Dinver rows (hidden until selected)
        self.dinver_widget = QWidget()
        dform = QFormLayout(self.dinver_widget)
        dform.setContentsMargins(0, 0, 0, 0)
        self.vs_edit = QLineEdit()
        self.vp_edit = QLineEdit()
        self.rho_edit = QLineEdit()
        for label, edit, tip in [
            ('Vs File:', self.vs_edit, 'Required'),
            ('Vp File:', self.vp_edit, 'Optional'),
            ('Density:', self.rho_edit, 'Optional'),
        ]:
            edit.setPlaceholderText(tip)
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            btn = QPushButton('…')
            btn.setFixedWidth(30)
            btn.clicked.connect(lambda _, e=edit: self._browse_dinver(e))
            row.addWidget(btn)
            dform.addRow(label, row)
        self.dinver_widget.setVisible(False)
        slay.addWidget(self.dinver_widget)

        src_grp.add_layout(slay)
        layout.addWidget(src_grp)

        self._src_group.buttonToggled.connect(self._on_source_toggled)

        # ── Workflow options ──────────────────────────────────────────
        opt_grp = CollapsibleGroup('Options')
        olay = QVBoxLayout()

        self.cb_report = QCheckBox('Generate comprehensive report')
        self.cb_report.setChecked(state.generate_report)
        olay.addWidget(self.cb_report)

        self.cb_interactive = QCheckBox('Interactive peak selection')
        self.cb_interactive.setChecked(state.interactive_peaks)
        olay.addWidget(self.cb_interactive)

        dr_row = QHBoxLayout()
        self.cb_dual = QCheckBox('Run dual-resonance analysis')
        self.cb_dual.setChecked(state.dual_resonance_enabled)
        dr_row.addWidget(self.cb_dual)
        dr_btn = QPushButton('⚙')
        dr_btn.setFixedWidth(28)
        dr_btn.setToolTip('Dual-resonance settings')
        dr_btn.clicked.connect(self.dual_resonance_settings_requested.emit)
        dr_row.addWidget(dr_btn)
        olay.addLayout(dr_row)

        opt_grp.add_layout(olay)
        layout.addWidget(opt_grp)

        # ── Output directory ──────────────────────────────────────────
        out_grp = CollapsibleGroup('Output')
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText('Output directory…')
        out_browse = QPushButton('Browse…')
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_browse)
        out_grp.add_layout(out_row)
        layout.addWidget(out_grp)

        # ── Action buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton('▶ Run Analysis')
        self.run_btn.setStyleSheet('background-color: #2E86AB; color: white; font-weight: bold; padding: 6px 16px;')
        self.run_btn.clicked.connect(self.run_requested.emit)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

    # ── Slots ────────────────────────────────────────────────────────

    def _on_source_toggled(self, btn, checked):
        if not checked:
            return
        is_dinver = self._src_group.checkedId() == 1
        self.dinver_widget.setVisible(is_dinver)
        self.model_edit.setVisible(not is_dinver)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Model', '',
            'Model files (*.txt *.csv *.xlsx);;All files (*)')
        if path:
            self.model_edit.setText(path)

    def _browse_dinver(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Dinver File', '',
            'Text files (*.txt);;All files (*)')
        if path:
            edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Output Directory')
        if path:
            self.out_edit.setText(path)
            self.state.output_dir = path

    def get_config(self) -> dict:
        """Return current panel configuration."""
        return {
            'source': 'hvf' if self.rb_hvf.isChecked() else 'dinver',
            'model_path': self.model_edit.text(),
            'vs_path': self.vs_edit.text(),
            'vp_path': self.vp_edit.text(),
            'rho_path': self.rho_edit.text(),
            'generate_report': self.cb_report.isChecked(),
            'interactive': self.cb_interactive.isChecked(),
            'dual_resonance': self.cb_dual.isChecked(),
            'output_dir': self.out_edit.text(),
        }
