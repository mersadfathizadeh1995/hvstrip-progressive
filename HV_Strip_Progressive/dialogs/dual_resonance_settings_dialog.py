"""Dual-Resonance Settings Dialog."""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox, QPushButton, QHBoxLayout, QLabel,
)


class DualResonanceSettingsDialog(QDialog):
    """Configure f0/f1 separation thresholds."""

    def __init__(self, ratio=1.2, shift=0.3, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dual-Resonance Settings")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(1.0, 5.0)
        self.ratio_spin.setValue(ratio)
        self.ratio_spin.setSingleStep(0.1)
        self.ratio_spin.setToolTip("f1/f0 must exceed this ratio for separation success")
        form.addRow("Separation Ratio Threshold (f1/f0):", self.ratio_spin)

        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(0.01, 5.0)
        self.shift_spin.setValue(shift)
        self.shift_spin.setSingleStep(0.05)
        self.shift_spin.setToolTip("Minimum step-to-step frequency shift (Hz)")
        form.addRow("Min Frequency Shift (Hz):", self.shift_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def get_values(self):
        return {
            "separation_ratio_threshold": self.ratio_spin.value(),
            "separation_shift_threshold": self.shift_spin.value(),
        }
