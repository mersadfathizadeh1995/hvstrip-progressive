"""
Peak Detection Panel Component
Handles peak detection method and parameters
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
from qfluentwidgets import (
    CardWidget,
    BodyLabel,
    ComboBox,
    SpinBox,
    DoubleSpinBox,
    FluentIcon
)


class PeakDetectionPanel(CardWidget):
    """Panel for peak detection settings"""

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
        title = BodyLabel("Peak Detection Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Settings grid
        grid_layout = QGridLayout()
        grid_layout.setSpacing(12)

        # Detection method
        method_label = BodyLabel("Detection Method:")
        self.method_combo = ComboBox()
        self.method_combo.addItems(["Global Maximum", "Find Peaks", "Manual"])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        self.method_combo.currentTextChanged.connect(self.on_settings_changed)

        grid_layout.addWidget(method_label, 0, 0)
        grid_layout.addWidget(self.method_combo, 0, 1)

        # Peak selection (for find_peaks method)
        selection_label = BodyLabel("Peak Selection:")
        self.selection_combo = ComboBox()
        self.selection_combo.addItems(["Fundamental (Leftmost)", "Maximum Amplitude"])
        self.selection_combo.currentTextChanged.connect(self.on_settings_changed)

        grid_layout.addWidget(selection_label, 1, 0)
        grid_layout.addWidget(self.selection_combo, 1, 1)

        # Prominence
        prominence_label = BodyLabel("Minimum Prominence:")
        self.prominence_spinbox = DoubleSpinBox()
        self.prominence_spinbox.setRange(0.01, 2.0)
        self.prominence_spinbox.setValue(0.2)
        self.prominence_spinbox.setSingleStep(0.05)
        self.prominence_spinbox.setDecimals(2)
        self.prominence_spinbox.valueChanged.connect(self.on_settings_changed)

        grid_layout.addWidget(prominence_label, 2, 0)
        grid_layout.addWidget(self.prominence_spinbox, 2, 1)

        # Distance
        distance_label = BodyLabel("Minimum Distance:")
        self.distance_spinbox = SpinBox()
        self.distance_spinbox.setRange(1, 20)
        self.distance_spinbox.setValue(3)
        self.distance_spinbox.valueChanged.connect(self.on_settings_changed)

        grid_layout.addWidget(distance_label, 3, 0)
        grid_layout.addWidget(self.distance_spinbox, 3, 1)

        # Frequency constraints
        freq_min_label = BodyLabel("Min Frequency (Hz):")
        self.freq_min_spinbox = DoubleSpinBox()
        self.freq_min_spinbox.setRange(0.0, 20.0)
        self.freq_min_spinbox.setValue(0.5)
        self.freq_min_spinbox.setSingleStep(0.1)
        self.freq_min_spinbox.setDecimals(2)
        self.freq_min_spinbox.valueChanged.connect(self.on_settings_changed)

        grid_layout.addWidget(freq_min_label, 4, 0)
        grid_layout.addWidget(self.freq_min_spinbox, 4, 1)

        freq_max_label = BodyLabel("Max Frequency (Hz):")
        self.freq_max_spinbox = DoubleSpinBox()
        self.freq_max_spinbox.setRange(0.0, 100.0)
        self.freq_max_spinbox.setValue(0.0)  # 0 means no limit
        self.freq_max_spinbox.setSingleStep(1.0)
        self.freq_max_spinbox.setDecimals(1)
        self.freq_max_spinbox.setSpecialValueText("No Limit")
        self.freq_max_spinbox.valueChanged.connect(self.on_settings_changed)

        grid_layout.addWidget(freq_max_label, 5, 0)
        grid_layout.addWidget(self.freq_max_spinbox, 5, 1)

        # Minimum relative height
        min_height_label = BodyLabel("Min Relative Height:")
        self.min_height_spinbox = DoubleSpinBox()
        self.min_height_spinbox.setRange(0.0, 1.0)
        self.min_height_spinbox.setValue(0.25)
        self.min_height_spinbox.setSingleStep(0.05)
        self.min_height_spinbox.setDecimals(2)
        self.min_height_spinbox.valueChanged.connect(self.on_settings_changed)

        grid_layout.addWidget(min_height_label, 6, 0)
        grid_layout.addWidget(self.min_height_spinbox, 6, 1)

        # Exclude first N points
        exclude_label = BodyLabel("Exclude First N Points:")
        self.exclude_spinbox = SpinBox()
        self.exclude_spinbox.setRange(0, 10)
        self.exclude_spinbox.setValue(1)
        self.exclude_spinbox.valueChanged.connect(self.on_settings_changed)

        grid_layout.addWidget(exclude_label, 7, 0)
        grid_layout.addWidget(self.exclude_spinbox, 7, 1)

        layout.addLayout(grid_layout)
        layout.addStretch()

        # Initialize enabled/disabled state
        self.on_method_changed(self.method_combo.currentText())

    def on_method_changed(self, method):
        """Handle detection method change"""
        # Enable/disable controls based on method
        is_find_peaks = method == "Find Peaks"

        self.selection_combo.setEnabled(is_find_peaks)
        self.prominence_spinbox.setEnabled(is_find_peaks)
        self.distance_spinbox.setEnabled(is_find_peaks)
        self.freq_min_spinbox.setEnabled(is_find_peaks)
        self.freq_max_spinbox.setEnabled(is_find_peaks)
        self.min_height_spinbox.setEnabled(is_find_peaks)
        self.exclude_spinbox.setEnabled(is_find_peaks)

    def on_settings_changed(self):
        """Emit signal when settings change"""
        self.settings_changed.emit(self.get_settings())

    def get_settings(self):
        """Get all peak detection settings"""
        method_map = {
            "Global Maximum": "max",
            "Find Peaks": "find_peaks",
            "Manual": "manual"
        }

        selection_map = {
            "Fundamental (Leftmost)": "leftmost",
            "Maximum Amplitude": "max"
        }

        settings = {
            'method': method_map.get(self.method_combo.currentText(), "max"),
            'selection': selection_map.get(self.selection_combo.currentText(), "leftmost"),
            'prominence': self.prominence_spinbox.value(),
            'distance': self.distance_spinbox.value(),
            'freq_min': self.freq_min_spinbox.value(),
            'freq_max': self.freq_max_spinbox.value() if self.freq_max_spinbox.value() > 0 else None,
            'min_rel_height': self.min_height_spinbox.value(),
            'exclude_first_n': self.exclude_spinbox.value()
        }

        return settings

    def set_settings(self, settings):
        """Set peak detection settings programmatically"""
        method_reverse_map = {
            "max": "Global Maximum",
            "find_peaks": "Find Peaks",
            "manual": "Manual"
        }

        selection_reverse_map = {
            "leftmost": "Fundamental (Leftmost)",
            "max": "Maximum Amplitude"
        }

        if 'method' in settings:
            method_text = method_reverse_map.get(settings['method'], "Global Maximum")
            index = self.method_combo.findText(method_text)
            if index >= 0:
                self.method_combo.setCurrentIndex(index)

        if 'selection' in settings:
            selection_text = selection_reverse_map.get(settings['selection'], "Fundamental (Leftmost)")
            index = self.selection_combo.findText(selection_text)
            if index >= 0:
                self.selection_combo.setCurrentIndex(index)

        if 'prominence' in settings:
            self.prominence_spinbox.setValue(settings['prominence'])
        if 'distance' in settings:
            self.distance_spinbox.setValue(settings['distance'])
        if 'freq_min' in settings:
            self.freq_min_spinbox.setValue(settings['freq_min'])
        if 'freq_max' in settings:
            self.freq_max_spinbox.setValue(settings['freq_max'] if settings['freq_max'] else 0.0)
        if 'min_rel_height' in settings:
            self.min_height_spinbox.setValue(settings['min_rel_height'])
        if 'exclude_first_n' in settings:
            self.exclude_spinbox.setValue(settings['exclude_first_n'])
