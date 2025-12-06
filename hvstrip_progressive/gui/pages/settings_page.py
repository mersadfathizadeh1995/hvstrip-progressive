"""
Settings Page - Advanced Configuration
Provides interface for all advanced settings
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import (
    ScrollArea,
    TitleLabel,
    BodyLabel,
    PrimaryPushButton,
    FluentIcon,
    InfoBar,
    InfoBarPosition
)

from ..components.frequency_panel import FrequencyPanel
from ..components.peak_detection_panel import PeakDetectionPanel
from ..components.visualization_panel import VisualizationPanel
from ..components.report_panel import ReportPanel


class SettingsPage(ScrollArea):
    """Settings page for advanced configuration"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        # Create scroll widget
        self.scroll_widget = QWidget()
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)
        self.setObjectName("settingsPage")

        # Main layout
        layout = QVBoxLayout(self.scroll_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Page title
        title = TitleLabel("Advanced Settings")
        layout.addWidget(title)

        # Description
        desc = BodyLabel(
            "Configure advanced parameters for frequency analysis, peak detection, "
            "visualization, and report generation. Changes are applied immediately."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Frequency settings panel
        self.frequency_panel = FrequencyPanel()
        layout.addWidget(self.frequency_panel)

        # Peak detection panel
        self.peak_detection_panel = PeakDetectionPanel()
        layout.addWidget(self.peak_detection_panel)

        # Visualization panel
        self.visualization_panel = VisualizationPanel()
        layout.addWidget(self.visualization_panel)

        # Report generation panel
        self.report_panel = ReportPanel()
        layout.addWidget(self.report_panel)

        # Reset to defaults button
        reset_layout = QVBoxLayout()
        self.reset_button = PrimaryPushButton("Reset to Defaults", self, FluentIcon.SYNC)
        self.reset_button.setFixedWidth(200)
        self.reset_button.clicked.connect(self.reset_to_defaults)
        reset_layout.addWidget(self.reset_button)
        layout.addLayout(reset_layout)

        layout.addStretch()

    def reset_to_defaults(self):
        """Reset all settings to default values"""
        # Reset frequency settings
        self.frequency_panel.set_settings({
            'fmin': 0.2,
            'fmax': 20.0,
            'nf': 71,
            'nmr': 10,
            'nml': 10,
            'nks': 10,
            'adaptive_scanning': {
                'enable': True,
                'max_passes': 2,
                'edge_margin_frac': 0.05,
                'fmax_expand_factor': 2.0,
                'fmin_shrink_factor': 0.5,
                'fmax_limit': 60.0,
                'fmin_limit': 0.05
            }
        })

        # Reset peak detection settings
        self.peak_detection_panel.set_settings({
            'method': 'find_peaks',
            'selection': 'leftmost',
            'prominence': 0.2,
            'distance': 3,
            'freq_min': 0.5,
            'freq_max': None,
            'min_rel_height': 0.25,
            'exclude_first_n': 1
        })

        # Reset visualization settings
        self.visualization_panel.set_settings({
            'hv_curve': {
                'x_axis_scale': 'log',
                'y_axis_scale': 'log',
                'y_compression': 1.5,
                'smoothing': {
                    'enable': True,
                    'window_length': 9,
                    'poly_order': 3
                },
                'show_bands': True,
                'freq_window_left': 0.3,
                'freq_window_right': 3.0,
                'figure_width': 12,
                'figure_height': 6,
                'dpi': 200
            },
            'vs_profile': {
                'show': True,
                'annotate_deepest': True,
                'annotate_max_vs': True,
                'annotate_f0': True
            },
            'output': {
                'save_separate': True,
                'save_combined': True
            }
        })

        # Reset report settings
        self.report_panel.set_settings({
            'generate_reports': {
                'summary_csv': True,
                'metadata_json': True,
                'text_report': True
            },
            'generate_visualizations': {
                'hv_overlay': True,
                'peak_evolution': True,
                'interface_analysis': True,
                'waterfall': True,
                'publication_figure': True
            },
            'publication_format': 'png'
        })

        InfoBar.success(
            "Success",
            "All settings reset to default values",
            duration=2000,
            position=InfoBarPosition.TOP_RIGHT,
            parent=self.window()
        )

    def get_all_settings(self):
        """Get all settings from all panels"""
        return {
            'frequency': self.frequency_panel.get_settings(),
            'peak_detection': self.peak_detection_panel.get_settings(),
            'visualization': self.visualization_panel.get_settings(),
            'reports': self.report_panel.get_settings()
        }

    def set_all_settings(self, settings):
        """Set all settings from a configuration dictionary"""
        if 'frequency' in settings:
            self.frequency_panel.set_settings(settings['frequency'])

        if 'peak_detection' in settings:
            self.peak_detection_panel.set_settings(settings['peak_detection'])

        if 'visualization' in settings:
            self.visualization_panel.set_settings(settings['visualization'])

        if 'reports' in settings:
            self.report_panel.set_settings(settings['reports'])
