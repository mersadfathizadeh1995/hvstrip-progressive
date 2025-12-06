"""
Frequency Settings Panel Component
Handles frequency range and adaptive scanning settings
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
from qfluentwidgets import (
    CardWidget,
    BodyLabel,
    SpinBox,
    DoubleSpinBox,
    SwitchButton,
    ExpandLayout,
    ExpandGroupSettingCard,
    FluentIcon
)


class FrequencyPanel(CardWidget):
    """Panel for frequency settings"""

    # Signals
    settings_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = BodyLabel("Frequency Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Basic frequency settings
        basic_layout = QGridLayout()
        basic_layout.setSpacing(12)

        # Minimum frequency
        fmin_label = BodyLabel("Minimum Frequency (Hz):")
        self.fmin_spinbox = DoubleSpinBox()
        self.fmin_spinbox.setRange(0.01, 10.0)
        self.fmin_spinbox.setValue(0.2)
        self.fmin_spinbox.setSingleStep(0.1)
        self.fmin_spinbox.setDecimals(2)
        self.fmin_spinbox.valueChanged.connect(self.on_settings_changed)

        basic_layout.addWidget(fmin_label, 0, 0)
        basic_layout.addWidget(self.fmin_spinbox, 0, 1)

        # Maximum frequency
        fmax_label = BodyLabel("Maximum Frequency (Hz):")
        self.fmax_spinbox = DoubleSpinBox()
        self.fmax_spinbox.setRange(1.0, 100.0)
        self.fmax_spinbox.setValue(20.0)
        self.fmax_spinbox.setSingleStep(1.0)
        self.fmax_spinbox.setDecimals(1)
        self.fmax_spinbox.valueChanged.connect(self.on_settings_changed)

        basic_layout.addWidget(fmax_label, 1, 0)
        basic_layout.addWidget(self.fmax_spinbox, 1, 1)

        # Number of frequency points
        nf_label = BodyLabel("Number of Points:")
        self.nf_spinbox = SpinBox()
        self.nf_spinbox.setRange(10, 500)
        self.nf_spinbox.setValue(71)
        self.nf_spinbox.setSingleStep(10)
        self.nf_spinbox.valueChanged.connect(self.on_settings_changed)

        basic_layout.addWidget(nf_label, 2, 0)
        basic_layout.addWidget(self.nf_spinbox, 2, 1)

        layout.addLayout(basic_layout)

        # Advanced settings (expandable)
        self.advanced_card = ExpandGroupSettingCard(
            FluentIcon.CHEVRON_DOWN,
            "Advanced Frequency Settings",
            "Adaptive scanning and advanced parameters"
        )

        # Adaptive scanning
        adaptive_widget = QWidget()
        adaptive_layout = QGridLayout(adaptive_widget)
        adaptive_layout.setContentsMargins(0, 10, 0, 10)

        adaptive_label = BodyLabel("Enable Adaptive Scanning:")
        self.adaptive_switch = SwitchButton()
        self.adaptive_switch.setChecked(True)
        self.adaptive_switch.checkedChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(adaptive_label, 0, 0)
        adaptive_layout.addWidget(self.adaptive_switch, 0, 1)

        # Max passes
        passes_label = BodyLabel("Max Adaptive Passes:")
        self.passes_spinbox = SpinBox()
        self.passes_spinbox.setRange(1, 5)
        self.passes_spinbox.setValue(2)
        self.passes_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(passes_label, 1, 0)
        adaptive_layout.addWidget(self.passes_spinbox, 1, 1)

        # Edge margin
        margin_label = BodyLabel("Edge Margin Fraction:")
        self.margin_spinbox = DoubleSpinBox()
        self.margin_spinbox.setRange(0.01, 0.2)
        self.margin_spinbox.setValue(0.05)
        self.margin_spinbox.setSingleStep(0.01)
        self.margin_spinbox.setDecimals(2)
        self.margin_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(margin_label, 2, 0)
        adaptive_layout.addWidget(self.margin_spinbox, 2, 1)

        # Expand factors
        expand_label = BodyLabel("Fmax Expand Factor:")
        self.expand_spinbox = DoubleSpinBox()
        self.expand_spinbox.setRange(1.5, 5.0)
        self.expand_spinbox.setValue(2.0)
        self.expand_spinbox.setSingleStep(0.5)
        self.expand_spinbox.setDecimals(1)
        self.expand_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(expand_label, 3, 0)
        adaptive_layout.addWidget(self.expand_spinbox, 3, 1)

        shrink_label = BodyLabel("Fmin Shrink Factor:")
        self.shrink_spinbox = DoubleSpinBox()
        self.shrink_spinbox.setRange(0.1, 0.9)
        self.shrink_spinbox.setValue(0.5)
        self.shrink_spinbox.setSingleStep(0.1)
        self.shrink_spinbox.setDecimals(1)
        self.shrink_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(shrink_label, 4, 0)
        adaptive_layout.addWidget(self.shrink_spinbox, 4, 1)

        # Frequency limits
        fmax_limit_label = BodyLabel("Fmax Limit (Hz):")
        self.fmax_limit_spinbox = DoubleSpinBox()
        self.fmax_limit_spinbox.setRange(10.0, 200.0)
        self.fmax_limit_spinbox.setValue(60.0)
        self.fmax_limit_spinbox.setSingleStep(10.0)
        self.fmax_limit_spinbox.setDecimals(1)
        self.fmax_limit_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(fmax_limit_label, 5, 0)
        adaptive_layout.addWidget(self.fmax_limit_spinbox, 5, 1)

        fmin_limit_label = BodyLabel("Fmin Limit (Hz):")
        self.fmin_limit_spinbox = DoubleSpinBox()
        self.fmin_limit_spinbox.setRange(0.01, 1.0)
        self.fmin_limit_spinbox.setValue(0.05)
        self.fmin_limit_spinbox.setSingleStep(0.01)
        self.fmin_limit_spinbox.setDecimals(2)
        self.fmin_limit_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(fmin_limit_label, 6, 0)
        adaptive_layout.addWidget(self.fmin_limit_spinbox, 6, 1)

        # HVf parameters
        nmr_label = BodyLabel("nmr Parameter:")
        self.nmr_spinbox = SpinBox()
        self.nmr_spinbox.setRange(1, 50)
        self.nmr_spinbox.setValue(10)
        self.nmr_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(nmr_label, 7, 0)
        adaptive_layout.addWidget(self.nmr_spinbox, 7, 1)

        nml_label = BodyLabel("nml Parameter:")
        self.nml_spinbox = SpinBox()
        self.nml_spinbox.setRange(1, 50)
        self.nml_spinbox.setValue(10)
        self.nml_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(nml_label, 8, 0)
        adaptive_layout.addWidget(self.nml_spinbox, 8, 1)

        nks_label = BodyLabel("nks Parameter:")
        self.nks_spinbox = SpinBox()
        self.nks_spinbox.setRange(1, 50)
        self.nks_spinbox.setValue(10)
        self.nks_spinbox.valueChanged.connect(self.on_settings_changed)

        adaptive_layout.addWidget(nks_label, 9, 0)
        adaptive_layout.addWidget(self.nks_spinbox, 9, 1)

        self.advanced_card.viewLayout.addWidget(adaptive_widget)
        layout.addWidget(self.advanced_card)

        layout.addStretch()

    def on_settings_changed(self):
        """Emit signal when settings change"""
        self.settings_changed.emit(self.get_settings())

    def get_settings(self):
        """Get all frequency settings"""
        return {
            'fmin': self.fmin_spinbox.value(),
            'fmax': self.fmax_spinbox.value(),
            'nf': self.nf_spinbox.value(),
            'nmr': self.nmr_spinbox.value(),
            'nml': self.nml_spinbox.value(),
            'nks': self.nks_spinbox.value(),
            'adaptive_scanning': {
                'enable': self.adaptive_switch.isChecked(),
                'max_passes': self.passes_spinbox.value(),
                'edge_margin_frac': self.margin_spinbox.value(),
                'fmax_expand_factor': self.expand_spinbox.value(),
                'fmin_shrink_factor': self.shrink_spinbox.value(),
                'fmax_limit': self.fmax_limit_spinbox.value(),
                'fmin_limit': self.fmin_limit_spinbox.value()
            }
        }

    def set_settings(self, settings):
        """Set frequency settings programmatically"""
        if 'fmin' in settings:
            self.fmin_spinbox.setValue(settings['fmin'])
        if 'fmax' in settings:
            self.fmax_spinbox.setValue(settings['fmax'])
        if 'nf' in settings:
            self.nf_spinbox.setValue(settings['nf'])
        if 'nmr' in settings:
            self.nmr_spinbox.setValue(settings['nmr'])
        if 'nml' in settings:
            self.nml_spinbox.setValue(settings['nml'])
        if 'nks' in settings:
            self.nks_spinbox.setValue(settings['nks'])

        if 'adaptive_scanning' in settings:
            adaptive = settings['adaptive_scanning']
            if 'enable' in adaptive:
                self.adaptive_switch.setChecked(adaptive['enable'])
            if 'max_passes' in adaptive:
                self.passes_spinbox.setValue(adaptive['max_passes'])
            if 'edge_margin_frac' in adaptive:
                self.margin_spinbox.setValue(adaptive['edge_margin_frac'])
            if 'fmax_expand_factor' in adaptive:
                self.expand_spinbox.setValue(adaptive['fmax_expand_factor'])
            if 'fmin_shrink_factor' in adaptive:
                self.shrink_spinbox.setValue(adaptive['fmin_shrink_factor'])
            if 'fmax_limit' in adaptive:
                self.fmax_limit_spinbox.setValue(adaptive['fmax_limit'])
            if 'fmin_limit' in adaptive:
                self.fmin_limit_spinbox.setValue(adaptive['fmin_limit'])
