"""
Report Generation Panel Component
Handles report generation options and output formats
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
from qfluentwidgets import (
    CardWidget,
    BodyLabel,
    CheckBox,
    ComboBox,
    FluentIcon
)


class ReportPanel(CardWidget):
    """Panel for report generation settings"""

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
        title = BodyLabel("Report Generation Options")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Description
        desc = BodyLabel(
            "Select which reports and visualizations to generate after workflow completion."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Report types checkboxes
        reports_layout = QVBoxLayout()
        reports_layout.setSpacing(8)

        self.summary_csv_checkbox = CheckBox("Progressive Stripping Summary (CSV)")
        self.summary_csv_checkbox.setChecked(True)
        self.summary_csv_checkbox.stateChanged.connect(self.on_settings_changed)
        reports_layout.addWidget(self.summary_csv_checkbox)

        self.metadata_checkbox = CheckBox("Analysis Metadata (JSON)")
        self.metadata_checkbox.setChecked(True)
        self.metadata_checkbox.stateChanged.connect(self.on_settings_changed)
        reports_layout.addWidget(self.metadata_checkbox)

        self.text_report_checkbox = CheckBox("Text Report (TXT)")
        self.text_report_checkbox.setChecked(True)
        self.text_report_checkbox.stateChanged.connect(self.on_settings_changed)
        reports_layout.addWidget(self.text_report_checkbox)

        layout.addLayout(reports_layout)

        # Visualization types
        viz_title = BodyLabel("Visualizations")
        viz_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(viz_title)

        viz_layout = QVBoxLayout()
        viz_layout.setSpacing(8)

        self.hv_overlay_checkbox = CheckBox("HV Curves Overlay")
        self.hv_overlay_checkbox.setChecked(True)
        self.hv_overlay_checkbox.setToolTip("All HV curves overlaid on one plot")
        self.hv_overlay_checkbox.stateChanged.connect(self.on_settings_changed)
        viz_layout.addWidget(self.hv_overlay_checkbox)

        self.peak_evolution_checkbox = CheckBox("Peak Evolution Analysis")
        self.peak_evolution_checkbox.setChecked(True)
        self.peak_evolution_checkbox.setToolTip("3-panel peak frequency/amplitude evolution")
        self.peak_evolution_checkbox.stateChanged.connect(self.on_settings_changed)
        viz_layout.addWidget(self.peak_evolution_checkbox)

        self.interface_analysis_checkbox = CheckBox("Interface Analysis")
        self.interface_analysis_checkbox.setChecked(True)
        self.interface_analysis_checkbox.setToolTip("Impedance contrast vs depth")
        self.interface_analysis_checkbox.stateChanged.connect(self.on_settings_changed)
        viz_layout.addWidget(self.interface_analysis_checkbox)

        self.waterfall_checkbox = CheckBox("Waterfall Plot")
        self.waterfall_checkbox.setChecked(True)
        self.waterfall_checkbox.setToolTip("3D-style waterfall view of all curves")
        self.waterfall_checkbox.stateChanged.connect(self.on_settings_changed)
        viz_layout.addWidget(self.waterfall_checkbox)

        self.publication_checkbox = CheckBox("Publication Figure")
        self.publication_checkbox.setChecked(True)
        self.publication_checkbox.setToolTip("4-panel publication-ready figure")
        self.publication_checkbox.stateChanged.connect(self.on_settings_changed)
        viz_layout.addWidget(self.publication_checkbox)

        layout.addLayout(viz_layout)

        # Output format
        format_layout = QHBoxLayout()
        format_label = BodyLabel("Publication Format:")
        self.format_combo = ComboBox()
        self.format_combo.addItems(["PNG", "PDF", "Both"])
        self.format_combo.setCurrentText("PNG")
        self.format_combo.currentTextChanged.connect(self.on_settings_changed)

        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()

        layout.addLayout(format_layout)

        layout.addStretch()

    def on_settings_changed(self):
        """Emit signal when settings change"""
        self.settings_changed.emit(self.get_settings())

    def get_settings(self):
        """Get all report generation settings"""
        return {
            'generate_reports': {
                'summary_csv': self.summary_csv_checkbox.isChecked(),
                'metadata_json': self.metadata_checkbox.isChecked(),
                'text_report': self.text_report_checkbox.isChecked(),
            },
            'generate_visualizations': {
                'hv_overlay': self.hv_overlay_checkbox.isChecked(),
                'peak_evolution': self.peak_evolution_checkbox.isChecked(),
                'interface_analysis': self.interface_analysis_checkbox.isChecked(),
                'waterfall': self.waterfall_checkbox.isChecked(),
                'publication_figure': self.publication_checkbox.isChecked(),
            },
            'publication_format': self.format_combo.currentText().lower()
        }

    def set_settings(self, settings):
        """Set report generation settings programmatically"""
        if 'generate_reports' in settings:
            reports = settings['generate_reports']
            if 'summary_csv' in reports:
                self.summary_csv_checkbox.setChecked(reports['summary_csv'])
            if 'metadata_json' in reports:
                self.metadata_checkbox.setChecked(reports['metadata_json'])
            if 'text_report' in reports:
                self.text_report_checkbox.setChecked(reports['text_report'])

        if 'generate_visualizations' in settings:
            viz = settings['generate_visualizations']
            if 'hv_overlay' in viz:
                self.hv_overlay_checkbox.setChecked(viz['hv_overlay'])
            if 'peak_evolution' in viz:
                self.peak_evolution_checkbox.setChecked(viz['peak_evolution'])
            if 'interface_analysis' in viz:
                self.interface_analysis_checkbox.setChecked(viz['interface_analysis'])
            if 'waterfall' in viz:
                self.waterfall_checkbox.setChecked(viz['waterfall'])
            if 'publication_figure' in viz:
                self.publication_checkbox.setChecked(viz['publication_figure'])

        if 'publication_format' in settings:
            fmt = settings['publication_format'].upper()
            index = self.format_combo.findText(fmt)
            if index >= 0:
                self.format_combo.setCurrentIndex(index)
