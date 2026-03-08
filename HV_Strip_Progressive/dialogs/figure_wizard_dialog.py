"""
Figure wizard dialog — review and export analysis figures.

Left: figure list + common settings + per-figure settings (stacked).
Right: preview canvas with Apply / Export This / Export All / Close.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QStackedWidget, QPushButton, QLabel, QSpinBox, QFormLayout,
    QWidget, QFileDialog, QMessageBox,
)

from ..widgets.plot_canvas import PlotCanvas
from ..widgets.collapsible_group import CollapsibleGroup

_FIGURE_KEYS = [
    ('hv_overlay', 'HV Curves Overlay'),
    ('peak_evolution', 'Peak Evolution'),
    ('interface_analysis', 'Interface Analysis'),
    ('waterfall', 'Waterfall Plot'),
    ('publication', 'Publication Figure (2×2)'),
    ('dual_resonance', 'Dual-Resonance Separation'),
]


class FigureWizardDialog(QDialog):
    """Modal dialog for reviewing and exporting analysis figures."""

    def __init__(self, steps: list, figure_configs: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Figure Wizard')
        self.resize(1600, 1000)
        self.setMinimumSize(1400, 900)

        self._steps = steps
        self._configs = figure_configs

        layout = QHBoxLayout(self)

        # ── Left panel ───────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(280)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        ll.addWidget(QLabel('Figures:'))
        self.fig_list = QListWidget()
        for key, label in _FIGURE_KEYS:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            self.fig_list.addItem(item)
        self.fig_list.currentItemChanged.connect(self._on_figure_changed)
        ll.addWidget(self.fig_list)

        # Common settings
        common = CollapsibleGroup('Common Settings')
        cform = QFormLayout()
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600); self.dpi_spin.setValue(300)
        cform.addRow('DPI:', self.dpi_spin)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 24); self.font_spin.setValue(12)
        cform.addRow('Font:', self.font_spin)
        common.add_layout(cform)
        ll.addWidget(common)

        # Per-figure settings (stacked)
        self.settings_stack = QStackedWidget()
        self._panels = {}
        from .figure_settings_panels import create_panels
        for key, label in _FIGURE_KEYS:
            panel = create_panels(key, self._configs.get(key, {}))
            self.settings_stack.addWidget(panel)
            self._panels[key] = panel
        ll.addWidget(self.settings_stack)

        layout.addWidget(left)

        # ── Right: canvas + buttons ──────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.canvas = PlotCanvas(figsize=(14, 8), dpi=100)
        rl.addWidget(self.canvas, 1)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton('Apply')
        apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(apply_btn)
        export_btn = QPushButton('Export This')
        export_btn.clicked.connect(self._on_export_this)
        btn_row.addWidget(export_btn)
        export_all_btn = QPushButton('Export All')
        export_all_btn.setStyleSheet('background-color: #2E86AB; color: white; font-weight: bold;')
        export_all_btn.clicked.connect(self._on_export_all)
        btn_row.addWidget(export_all_btn)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        rl.addLayout(btn_row)

        layout.addWidget(right, 1)

        # Select first figure
        self.fig_list.setCurrentRow(0)

    def _on_figure_changed(self, current, previous):
        if current is None:
            return
        key = current.data(Qt.UserRole)
        idx = [k for k, _ in _FIGURE_KEYS].index(key)
        self.settings_stack.setCurrentIndex(idx)
        self._render(key)

    def _on_apply(self):
        item = self.fig_list.currentItem()
        if item:
            self._render(item.data(Qt.UserRole))

    def _render(self, fig_key: str):
        """Render figure using core visualization or fallback."""
        self.canvas.clear()
        panel = self._panels.get(fig_key)
        cfg = panel.get_config() if panel else {}

        # Use the same fallback rendering as FigureGallery
        from ..widgets.figure_gallery import FigureGallery
        gallery = FigureGallery.__new__(FigureGallery)
        gallery.canvas = self.canvas
        gallery.state = type('S', (), {'strip_steps': self._steps, 'figure_configs': {fig_key: cfg}})()
        gallery._render_from_steps(fig_key, self._steps, cfg)
        self.canvas.draw()

    def _on_export_this(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Figure', '',
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)')
        if path:
            self.canvas.figure.savefig(path, dpi=self.dpi_spin.value(), bbox_inches='tight')

    def _on_export_all(self):
        import os
        path = QFileDialog.getExistingDirectory(self, 'Export All Figures')
        if not path:
            return
        for key, label in _FIGURE_KEYS:
            self._render(key)
            for fmt in ('png', 'pdf'):
                self.canvas.figure.savefig(
                    os.path.join(path, f'{key}.{fmt}'),
                    dpi=self.dpi_spin.value(), bbox_inches='tight')
        QMessageBox.information(self, 'Export', f'All figures exported to {path}')
