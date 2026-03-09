"""
Engine configuration panel.

Provides engine selection combo, frequency range spinboxes,
and a button to open the per-engine settings dialog.
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox,
)

from ..widgets.collapsible_group import CollapsibleGroup

_ENGINE_DESCRIPTIONS = {
    'diffuse_field': 'Diffuse-field wavefield (HVf.exe) — default, most accurate for deep sites.',
    'sh_wave': 'SH-wave transfer matrix (Kramer 1996) — pure Python, no external binary.',
    'ellipticity': 'Rayleigh-wave ellipticity (gpell.exe) — surface-wave dominant sites.',
}


class EnginePanel(QWidget):
    """Engine selection and frequency configuration panel."""

    engine_settings_requested = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Engine selection ─────────────────────────────────────────
        grp = CollapsibleGroup('Forward-Modeling Engine')
        form = QFormLayout()

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(['diffuse_field', 'sh_wave', 'ellipticity'])
        self.engine_combo.setCurrentText(state.engine_name)
        form.addRow('Engine:', self.engine_combo)

        self.desc_label = QLabel(_ENGINE_DESCRIPTIONS.get(state.engine_name, ''))
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet('color: gray; font-size: 10px;')
        form.addRow(self.desc_label)

        settings_btn = QPushButton('⚙ Engine Settings…')
        settings_btn.clicked.connect(self.engine_settings_requested.emit)
        form.addRow(settings_btn)

        grp.add_layout(form)
        layout.addWidget(grp)

        # ── Frequency configuration ──────────────────────────────────
        freq_grp = CollapsibleGroup('Frequency Range')
        fform = QFormLayout()

        self.fmin_spin = QDoubleSpinBox()
        self.fmin_spin.setRange(0.01, 10.0)
        self.fmin_spin.setValue(state.fmin)
        self.fmin_spin.setDecimals(2)
        self.fmin_spin.setSuffix(' Hz')
        fform.addRow('Min Frequency:', self.fmin_spin)

        self.fmax_spin = QDoubleSpinBox()
        self.fmax_spin.setRange(1.0, 100.0)
        self.fmax_spin.setValue(state.fmax)
        self.fmax_spin.setDecimals(1)
        self.fmax_spin.setSuffix(' Hz')
        fform.addRow('Max Frequency:', self.fmax_spin)

        self.nf_spin = QSpinBox()
        self.nf_spin.setRange(10, 2000)
        self.nf_spin.setValue(state.nf)
        self.nf_spin.setSuffix(' pts')
        fform.addRow('Frequency Points:', self.nf_spin)

        freq_grp.add_layout(fform)
        layout.addWidget(freq_grp)

        layout.addStretch()

        # ── Connections ──────────────────────────────────────────────
        self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
        self.fmin_spin.valueChanged.connect(self._push_freq)
        self.fmax_spin.valueChanged.connect(self._push_freq)
        self.nf_spin.valueChanged.connect(self._push_freq)

    def _on_engine_changed(self, name: str):
        self.desc_label.setText(_ENGINE_DESCRIPTIONS.get(name, ''))
        self.state.set_engine(name)

    def _push_freq(self):
        self.state.set_freq_config(
            self.fmin_spin.value(),
            self.fmax_spin.value(),
            self.nf_spin.value(),
        )
