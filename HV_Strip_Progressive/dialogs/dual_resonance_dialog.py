"""
Dual-resonance dialog — configure thresholds for dual-resonance extraction.

Provides separation ratio and minimum frequency shift thresholds.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDoubleSpinBox, QLabel,
    QDialogButtonBox, QGroupBox,
)


class DualResonanceDialog(QDialog):
    """Modal dialog for dual-resonance threshold configuration."""

    def __init__(self, config: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Dual-Resonance Settings')
        self.setFixedSize(380, 260)

        cfg = config or {}
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            'Dual-resonance extraction identifies deep (f₀) and '
            'shallow (f₁) resonance frequencies from adjacent '
            'stripping steps.'
        ))

        group = QGroupBox('Thresholds')
        form = QFormLayout()

        self.separation = QDoubleSpinBox()
        self.separation.setRange(1.0, 20.0)
        self.separation.setValue(cfg.get('separation_ratio', 2.0))
        self.separation.setDecimals(1)
        self.separation.setToolTip(
            'Min ratio f₁/f₀ (or f₀/f₁) for dual-resonance to be declared successful')
        form.addRow('Separation ratio (f₁/f₀):', self.separation)

        self.min_shift = QDoubleSpinBox()
        self.min_shift.setRange(0.01, 10.0)
        self.min_shift.setValue(cfg.get('min_shift', 0.2))
        self.min_shift.setDecimals(2)
        self.min_shift.setSuffix(' Hz')
        self.min_shift.setToolTip(
            'Min absolute frequency shift between steps to be considered significant')
        form.addRow('Min. frequency shift:', self.min_shift)

        self.min_amp = QDoubleSpinBox()
        self.min_amp.setRange(0.0, 20.0)
        self.min_amp.setValue(cfg.get('min_amplitude', 1.5))
        self.min_amp.setDecimals(1)
        self.min_amp.setToolTip('Min amplitude at peak for resonance to be valid')
        form.addRow('Min. peak amplitude:', self.min_amp)

        group.setLayout(form)
        layout.addWidget(group)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_config(self) -> dict:
        return {
            'separation_ratio': self.separation.value(),
            'min_shift': self.min_shift.value(),
            'min_amplitude': self.min_amp.value(),
        }
