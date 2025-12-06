"""
Visualization Panel Component
Handles plot settings and visualization options
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
from qfluentwidgets import (
    CardWidget,
    BodyLabel,
    ComboBox,
    SpinBox,
    DoubleSpinBox,
    SwitchButton,
    CheckBox,
    ExpandGroupSettingCard,
    FluentIcon
)


class VisualizationPanel(CardWidget):
    """Panel for visualization settings"""

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
        title = BodyLabel("Visualization Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # HV Curve Settings
        hv_card = ExpandGroupSettingCard(
            FluentIcon.PHOTO,
            "HV Curve Plot Settings",
            "Configure HV curve visualization"
        )

        hv_widget = QWidget()
        hv_layout = QGridLayout(hv_widget)
        hv_layout.setContentsMargins(0, 10, 0, 10)

        # Axis scales
        x_scale_label = BodyLabel("X-axis Scale:")
        self.x_scale_combo = ComboBox()
        self.x_scale_combo.addItems(["Logarithmic", "Linear"])
        self.x_scale_combo.currentTextChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(x_scale_label, 0, 0)
        hv_layout.addWidget(self.x_scale_combo, 0, 1)

        y_scale_label = BodyLabel("Y-axis Scale:")
        self.y_scale_combo = ComboBox()
        self.y_scale_combo.addItems(["Logarithmic", "Linear"])
        self.y_scale_combo.currentTextChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(y_scale_label, 1, 0)
        hv_layout.addWidget(self.y_scale_combo, 1, 1)

        # Y compression
        y_comp_label = BodyLabel("Y-axis Compression:")
        self.y_comp_spinbox = DoubleSpinBox()
        self.y_comp_spinbox.setRange(1.0, 3.0)
        self.y_comp_spinbox.setValue(1.5)
        self.y_comp_spinbox.setSingleStep(0.1)
        self.y_comp_spinbox.setDecimals(1)
        self.y_comp_spinbox.valueChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(y_comp_label, 2, 0)
        hv_layout.addWidget(self.y_comp_spinbox, 2, 1)

        # Smoothing
        smooth_label = BodyLabel("Enable Smoothing:")
        self.smooth_switch = SwitchButton()
        self.smooth_switch.setChecked(True)
        self.smooth_switch.checkedChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(smooth_label, 3, 0)
        hv_layout.addWidget(self.smooth_switch, 3, 1)

        # Smoothing parameters
        window_label = BodyLabel("Smoothing Window:")
        self.window_spinbox = SpinBox()
        self.window_spinbox.setRange(3, 21)
        self.window_spinbox.setValue(9)
        self.window_spinbox.setSingleStep(2)  # Must be odd
        self.window_spinbox.valueChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(window_label, 4, 0)
        hv_layout.addWidget(self.window_spinbox, 4, 1)

        poly_label = BodyLabel("Polynomial Order:")
        self.poly_spinbox = SpinBox()
        self.poly_spinbox.setRange(1, 5)
        self.poly_spinbox.setValue(3)
        self.poly_spinbox.valueChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(poly_label, 5, 0)
        hv_layout.addWidget(self.poly_spinbox, 5, 1)

        # Show frequency bands
        bands_label = BodyLabel("Show Frequency Bands:")
        self.bands_switch = SwitchButton()
        self.bands_switch.setChecked(True)
        self.bands_switch.checkedChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(bands_label, 6, 0)
        hv_layout.addWidget(self.bands_switch, 6, 1)

        # Band widths
        left_band_label = BodyLabel("Left Band Multiplier:")
        self.left_band_spinbox = DoubleSpinBox()
        self.left_band_spinbox.setRange(0.1, 1.0)
        self.left_band_spinbox.setValue(0.3)
        self.left_band_spinbox.setSingleStep(0.1)
        self.left_band_spinbox.setDecimals(1)
        self.left_band_spinbox.valueChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(left_band_label, 7, 0)
        hv_layout.addWidget(self.left_band_spinbox, 7, 1)

        right_band_label = BodyLabel("Right Band Multiplier:")
        self.right_band_spinbox = DoubleSpinBox()
        self.right_band_spinbox.setRange(1.0, 5.0)
        self.right_band_spinbox.setValue(3.0)
        self.right_band_spinbox.setSingleStep(0.5)
        self.right_band_spinbox.setDecimals(1)
        self.right_band_spinbox.valueChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(right_band_label, 8, 0)
        hv_layout.addWidget(self.right_band_spinbox, 8, 1)

        # Figure size
        width_label = BodyLabel("Figure Width (inches):")
        self.width_spinbox = SpinBox()
        self.width_spinbox.setRange(6, 20)
        self.width_spinbox.setValue(12)
        self.width_spinbox.valueChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(width_label, 9, 0)
        hv_layout.addWidget(self.width_spinbox, 9, 1)

        height_label = BodyLabel("Figure Height (inches):")
        self.height_spinbox = SpinBox()
        self.height_spinbox.setRange(4, 15)
        self.height_spinbox.setValue(6)
        self.height_spinbox.valueChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(height_label, 10, 0)
        hv_layout.addWidget(self.height_spinbox, 10, 1)

        # DPI
        dpi_label = BodyLabel("DPI (Resolution):")
        self.dpi_spinbox = SpinBox()
        self.dpi_spinbox.setRange(72, 600)
        self.dpi_spinbox.setValue(200)
        self.dpi_spinbox.setSingleStep(50)
        self.dpi_spinbox.valueChanged.connect(self.on_settings_changed)

        hv_layout.addWidget(dpi_label, 11, 0)
        hv_layout.addWidget(self.dpi_spinbox, 11, 1)

        hv_card.viewLayout.addWidget(hv_widget)
        layout.addWidget(hv_card)

        # Vs Profile Settings
        vs_card = ExpandGroupSettingCard(
            FluentIcon.CHART,
            "Vs Profile Plot Settings",
            "Configure velocity profile visualization"
        )

        vs_widget = QWidget()
        vs_layout = QVBoxLayout(vs_widget)
        vs_layout.setContentsMargins(0, 10, 0, 10)

        # Show Vs profile
        self.show_vs_checkbox = CheckBox("Show Vs Profile")
        self.show_vs_checkbox.setChecked(True)
        self.show_vs_checkbox.stateChanged.connect(self.on_settings_changed)
        vs_layout.addWidget(self.show_vs_checkbox)

        # Annotations
        self.annotate_deepest_checkbox = CheckBox("Annotate Deepest Interface")
        self.annotate_deepest_checkbox.setChecked(True)
        self.annotate_deepest_checkbox.stateChanged.connect(self.on_settings_changed)
        vs_layout.addWidget(self.annotate_deepest_checkbox)

        self.annotate_max_vs_checkbox = CheckBox("Annotate Maximum Vs")
        self.annotate_max_vs_checkbox.setChecked(True)
        self.annotate_max_vs_checkbox.stateChanged.connect(self.on_settings_changed)
        vs_layout.addWidget(self.annotate_max_vs_checkbox)

        self.annotate_f0_checkbox = CheckBox("Annotate Fundamental Frequency")
        self.annotate_f0_checkbox.setChecked(True)
        self.annotate_f0_checkbox.stateChanged.connect(self.on_settings_changed)
        vs_layout.addWidget(self.annotate_f0_checkbox)

        vs_card.viewLayout.addWidget(vs_widget)
        layout.addWidget(vs_card)

        # Output Options
        output_card = ExpandGroupSettingCard(
            FluentIcon.SAVE,
            "Output Options",
            "Configure saved files"
        )

        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 10, 0, 10)

        self.save_separate_checkbox = CheckBox("Save Individual Plots")
        self.save_separate_checkbox.setChecked(True)
        self.save_separate_checkbox.stateChanged.connect(self.on_settings_changed)
        output_layout.addWidget(self.save_separate_checkbox)

        self.save_combined_checkbox = CheckBox("Save Combined Figure")
        self.save_combined_checkbox.setChecked(True)
        self.save_combined_checkbox.stateChanged.connect(self.on_settings_changed)
        output_layout.addWidget(self.save_combined_checkbox)

        output_card.viewLayout.addWidget(output_widget)
        layout.addWidget(output_card)

        layout.addStretch()

    def on_settings_changed(self):
        """Emit signal when settings change"""
        self.settings_changed.emit(self.get_settings())

    def get_settings(self):
        """Get all visualization settings"""
        scale_map = {
            "Logarithmic": "log",
            "Linear": "linear"
        }

        return {
            'hv_curve': {
                'x_axis_scale': scale_map.get(self.x_scale_combo.currentText(), "log"),
                'y_axis_scale': scale_map.get(self.y_scale_combo.currentText(), "log"),
                'y_compression': self.y_comp_spinbox.value(),
                'smoothing': {
                    'enable': self.smooth_switch.isChecked(),
                    'window_length': self.window_spinbox.value(),
                    'poly_order': self.poly_spinbox.value()
                },
                'show_bands': self.bands_switch.isChecked(),
                'freq_window_left': self.left_band_spinbox.value(),
                'freq_window_right': self.right_band_spinbox.value(),
                'figure_width': self.width_spinbox.value(),
                'figure_height': self.height_spinbox.value(),
                'dpi': self.dpi_spinbox.value()
            },
            'vs_profile': {
                'show': self.show_vs_checkbox.isChecked(),
                'annotate_deepest': self.annotate_deepest_checkbox.isChecked(),
                'annotate_max_vs': self.annotate_max_vs_checkbox.isChecked(),
                'annotate_f0': self.annotate_f0_checkbox.isChecked()
            },
            'output': {
                'save_separate': self.save_separate_checkbox.isChecked(),
                'save_combined': self.save_combined_checkbox.isChecked()
            }
        }

    def set_settings(self, settings):
        """Set visualization settings programmatically"""
        scale_reverse_map = {
            "log": "Logarithmic",
            "linear": "Linear"
        }

        if 'hv_curve' in settings:
            hv = settings['hv_curve']

            if 'x_axis_scale' in hv:
                scale_text = scale_reverse_map.get(hv['x_axis_scale'], "Logarithmic")
                index = self.x_scale_combo.findText(scale_text)
                if index >= 0:
                    self.x_scale_combo.setCurrentIndex(index)

            if 'y_axis_scale' in hv:
                scale_text = scale_reverse_map.get(hv['y_axis_scale'], "Logarithmic")
                index = self.y_scale_combo.findText(scale_text)
                if index >= 0:
                    self.y_scale_combo.setCurrentIndex(index)

            if 'y_compression' in hv:
                self.y_comp_spinbox.setValue(hv['y_compression'])

            if 'smoothing' in hv:
                if 'enable' in hv['smoothing']:
                    self.smooth_switch.setChecked(hv['smoothing']['enable'])
                if 'window_length' in hv['smoothing']:
                    self.window_spinbox.setValue(hv['smoothing']['window_length'])
                if 'poly_order' in hv['smoothing']:
                    self.poly_spinbox.setValue(hv['smoothing']['poly_order'])

            if 'show_bands' in hv:
                self.bands_switch.setChecked(hv['show_bands'])
            if 'freq_window_left' in hv:
                self.left_band_spinbox.setValue(hv['freq_window_left'])
            if 'freq_window_right' in hv:
                self.right_band_spinbox.setValue(hv['freq_window_right'])
            if 'figure_width' in hv:
                self.width_spinbox.setValue(hv['figure_width'])
            if 'figure_height' in hv:
                self.height_spinbox.setValue(hv['figure_height'])
            if 'dpi' in hv:
                self.dpi_spinbox.setValue(hv['dpi'])

        if 'vs_profile' in settings:
            vs = settings['vs_profile']
            if 'show' in vs:
                self.show_vs_checkbox.setChecked(vs['show'])
            if 'annotate_deepest' in vs:
                self.annotate_deepest_checkbox.setChecked(vs['annotate_deepest'])
            if 'annotate_max_vs' in vs:
                self.annotate_max_vs_checkbox.setChecked(vs['annotate_max_vs'])
            if 'annotate_f0' in vs:
                self.annotate_f0_checkbox.setChecked(vs['annotate_f0'])

        if 'output' in settings:
            output = settings['output']
            if 'save_separate' in output:
                self.save_separate_checkbox.setChecked(output['save_separate'])
            if 'save_combined' in output:
                self.save_combined_checkbox.setChecked(output['save_combined'])
