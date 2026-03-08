"""
Dual-Resonance Settings Dialog.

Small popup for configuring dual-resonance analysis thresholds.
"""

from PySide6.QtWidgets import (
    QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout,
)


class DualResonanceSettingsDialog(QDialog):
    """Popup dialog for dual-resonance threshold configuration."""

    def __init__(self, parent=None, ratio=1.2, shift=0.3):
        super().__init__(parent)
        self.setWindowTitle("Dual-Resonance Settings")
        self.setFixedWidth(360)

        layout = QVBoxLayout(self)

        desc = QLabel(
            "Configure thresholds for the dual-resonance\n"
            "f0 / f1 separation analysis."
        )
        desc.setStyleSheet("color: gray; font-style: italic; margin-bottom: 8px;")
        layout.addWidget(desc)

        form = QFormLayout()

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(1.0, 5.0)
        self.ratio_spin.setSingleStep(0.1)
        self.ratio_spin.setDecimals(2)
        self.ratio_spin.setValue(ratio)
        self.ratio_spin.setToolTip("f1/f0 must exceed this ratio for separation success")
        form.addRow("Separation ratio threshold:", self.ratio_spin)

        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(0.01, 5.0)
        self.shift_spin.setSingleStep(0.05)
        self.shift_spin.setDecimals(2)
        self.shift_spin.setValue(shift)
        self.shift_spin.setSuffix(" Hz")
        self.shift_spin.setToolTip("Minimum step-to-step frequency shift (Hz)")
        form.addRow("Min frequency shift:", self.shift_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def get_values(self) -> dict:
        """Return current threshold values."""
        return {
            "separation_ratio_threshold": self.ratio_spin.value(),
            "separation_shift_threshold": self.shift_spin.value(),
        }
