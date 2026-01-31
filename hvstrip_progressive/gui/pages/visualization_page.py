"""
Visualization Studio - Consolidated post-processing and figure generation.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFileDialog, QLineEdit, QTextEdit, QSplitter,
    QTabWidget, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QScrollArea, QGridLayout, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread

from ..widgets.plot_widget import MatplotlibWidget


class VisualizationPage(QWidget):
    """
    Visualization Studio for creating and customizing figures.
    
    Features:
    - Load HV curve data and model files
    - Interactive plot customization
    - Export publication-quality figures
    - Batch figure generation
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VisualizationPage")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("Visualization Studio")
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        data_group = QGroupBox("Data Sources")
        data_layout = QVBoxLayout(data_group)

        hv_layout = QHBoxLayout()
        hv_layout.addWidget(QLabel("HV Curve CSV:"))
        self.hv_edit = QLineEdit()
        self.hv_edit.setPlaceholderText("Select HV curve data file")
        hv_layout.addWidget(self.hv_edit)
        btn_hv = QPushButton("...")
        btn_hv.setMaximumWidth(30)
        btn_hv.clicked.connect(self._browse_hv)
        hv_layout.addWidget(btn_hv)
        data_layout.addLayout(hv_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model File:"))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Select velocity model file")
        model_layout.addWidget(self.model_edit)
        btn_model = QPushButton("...")
        btn_model.setMaximumWidth(30)
        btn_model.clicked.connect(self._browse_model)
        model_layout.addWidget(btn_model)
        data_layout.addLayout(model_layout)

        results_layout = QHBoxLayout()
        results_layout.addWidget(QLabel("Results Dir:"))
        self.results_edit = QLineEdit()
        self.results_edit.setPlaceholderText("Select analysis results directory")
        results_layout.addWidget(self.results_edit)
        btn_results = QPushButton("...")
        btn_results.setMaximumWidth(30)
        btn_results.clicked.connect(self._browse_results)
        results_layout.addWidget(btn_results)
        data_layout.addLayout(results_layout)

        btn_load = QPushButton("Load Data")
        btn_load.clicked.connect(self._load_data)
        data_layout.addWidget(btn_load)

        left_layout.addWidget(data_group)

        style_group = QGroupBox("Figure Style")
        style_layout = QGridLayout(style_group)

        style_layout.addWidget(QLabel("DPI:"), 0, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(150)
        style_layout.addWidget(self.dpi_spin, 0, 1)

        style_layout.addWidget(QLabel("Figure Width:"), 1, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(4, 20)
        self.width_spin.setValue(8)
        style_layout.addWidget(self.width_spin, 1, 1)

        style_layout.addWidget(QLabel("Figure Height:"), 2, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(3, 15)
        self.height_spin.setValue(6)
        style_layout.addWidget(self.height_spin, 2, 1)

        style_layout.addWidget(QLabel("Color Scheme:"), 3, 0)
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Default", "Viridis", "Plasma", "Grayscale"])
        style_layout.addWidget(self.color_combo, 3, 1)

        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        style_layout.addWidget(self.grid_check, 4, 0, 1, 2)

        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(True)
        style_layout.addWidget(self.legend_check, 5, 0, 1, 2)

        left_layout.addWidget(style_group)

        axis_group = QGroupBox("Axis Settings")
        axis_layout = QGridLayout(axis_group)

        axis_layout.addWidget(QLabel("X Scale:"), 0, 0)
        self.xscale_combo = QComboBox()
        self.xscale_combo.addItems(["Linear", "Log"])
        axis_layout.addWidget(self.xscale_combo, 0, 1)

        axis_layout.addWidget(QLabel("Y Scale:"), 1, 0)
        self.yscale_combo = QComboBox()
        self.yscale_combo.addItems(["Linear", "Log"])
        axis_layout.addWidget(self.yscale_combo, 1, 1)

        axis_layout.addWidget(QLabel("Freq Min:"), 2, 0)
        self.freq_min = QDoubleSpinBox()
        self.freq_min.setRange(0.01, 10)
        self.freq_min.setValue(0.1)
        axis_layout.addWidget(self.freq_min, 2, 1)

        axis_layout.addWidget(QLabel("Freq Max:"), 3, 0)
        self.freq_max = QDoubleSpinBox()
        self.freq_max.setRange(1, 100)
        self.freq_max.setValue(20)
        axis_layout.addWidget(self.freq_max, 3, 1)

        left_layout.addWidget(axis_group)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "PDF", "SVG", "EPS"])
        format_layout.addWidget(self.format_combo)
        export_layout.addLayout(format_layout)

        btn_export = QPushButton("Export Current Figure")
        btn_export.clicked.connect(self._export_figure)
        export_layout.addWidget(btn_export)

        btn_export_all = QPushButton("Export All Figures")
        btn_export_all.clicked.connect(self._export_all)
        export_layout.addWidget(btn_export_all)

        left_layout.addWidget(export_group)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_tabs = QTabWidget()

        self.hv_plot = MatplotlibWidget(figsize=(8, 6))
        self.plot_tabs.addTab(self.hv_plot, "HV Curve")

        self.vs_plot = MatplotlibWidget(figsize=(6, 8))
        self.plot_tabs.addTab(self.vs_plot, "Vs Profile")

        self.overlay_plot = MatplotlibWidget(figsize=(10, 6))
        self.plot_tabs.addTab(self.overlay_plot, "HV Overlay")

        self.peak_plot = MatplotlibWidget(figsize=(8, 6))
        self.plot_tabs.addTab(self.peak_plot, "Peak Evolution")

        right_layout.addWidget(self.plot_tabs)

        btn_layout = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Plot")
        btn_refresh.clicked.connect(self._refresh_current_plot)
        btn_layout.addWidget(btn_refresh)
        
        btn_layout.addStretch()
        right_layout.addLayout(btn_layout)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 650])

        layout.addWidget(splitter)

    def _browse_hv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select HV Curve File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.hv_edit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            self.model_edit.setText(path)

    def _browse_results(self):
        path = QFileDialog.getExistingDirectory(self, "Select Results Directory")
        if path:
            self.results_edit.setText(path)

    def _load_data(self):
        self._draw_placeholder(self.hv_plot, "HV Curve\n(Load data to display)")
        self._draw_placeholder(self.vs_plot, "Vs Profile\n(Load data to display)")
        self._draw_placeholder(self.overlay_plot, "HV Overlay\n(Load results directory)")
        self._draw_placeholder(self.peak_plot, "Peak Evolution\n(Load results directory)")

    def _draw_placeholder(self, plot_widget, text):
        fig = plot_widget.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=14, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        plot_widget.refresh()

    def _refresh_current_plot(self):
        current_tab = self.plot_tabs.currentIndex()
        self._load_data()

    def _export_figure(self):
        format_ext = self.format_combo.currentText().lower()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", f"figure.{format_ext}",
            f"{self.format_combo.currentText()} Files (*.{format_ext})"
        )
        if path:
            current_widget = self.plot_tabs.currentWidget()
            if hasattr(current_widget, 'get_figure'):
                fig = current_widget.get_figure()
                fig.savefig(path, dpi=self.dpi_spin.value(), bbox_inches='tight')

    def _export_all(self):
        path = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if path:
            format_ext = self.format_combo.currentText().lower()
            dpi = self.dpi_spin.value()
            
            exports = [
                (self.hv_plot, "hv_curve"),
                (self.vs_plot, "vs_profile"),
                (self.overlay_plot, "hv_overlay"),
                (self.peak_plot, "peak_evolution"),
            ]
            
            for widget, name in exports:
                if hasattr(widget, 'get_figure'):
                    fig = widget.get_figure()
                    filepath = Path(path) / f"{name}.{format_ext}"
                    fig.savefig(str(filepath), dpi=dpi, bbox_inches='tight')
