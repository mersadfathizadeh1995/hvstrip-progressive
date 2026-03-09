"""
Figure gallery view — browse and export analysis figures.

Provides a list of available figure types with per-figure settings,
preview canvas, and export (individual / all) functionality.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QStackedWidget, QFileDialog, QMessageBox,
    QSplitter, QSpinBox, QFormLayout,
)

from ..widgets.plot_canvas import PlotCanvas
from ..widgets.collapsible_group import CollapsibleGroup

# Available figure types
_FIGURE_TYPES = [
    ('hv_overlay', 'HV Curves Overlay'),
    ('peak_evolution', 'Peak Evolution'),
    ('interface_analysis', 'Interface Analysis'),
    ('waterfall', 'Waterfall Plot'),
    ('publication', 'Publication Figure (2×2)'),
    ('dual_resonance', 'Dual-Resonance Separation'),
]


class FigureGallery(QWidget):
    """Figure browser with per-figure settings and preview."""

    figure_wizard_requested = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # ── Left: figure list + common settings ──────────────────────
        left = QWidget()
        left.setMaximumWidth(280)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)

        left_lay.addWidget(QLabel('Available Figures:'))
        self.fig_list = QListWidget()
        for key, label in _FIGURE_TYPES:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            self.fig_list.addItem(item)
        self.fig_list.currentItemChanged.connect(self._on_figure_selected)
        left_lay.addWidget(self.fig_list)

        # Common settings
        common = CollapsibleGroup('Common Settings')
        cform = QFormLayout()
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        cform.addRow('DPI:', self.dpi_spin)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 24)
        self.font_spin.setValue(12)
        cform.addRow('Font Size:', self.font_spin)
        common.add_layout(cform)
        left_lay.addWidget(common)

        # Buttons
        btn_row = QVBoxLayout()
        refresh_btn = QPushButton('🔄 Refresh')
        refresh_btn.clicked.connect(self._on_refresh)
        btn_row.addWidget(refresh_btn)
        export_btn = QPushButton('💾 Export This')
        export_btn.clicked.connect(self._on_export_current)
        btn_row.addWidget(export_btn)
        export_all_btn = QPushButton('📦 Export All')
        export_all_btn.clicked.connect(self._on_export_all)
        btn_row.addWidget(export_all_btn)
        wizard_btn = QPushButton('🎨 Figure Wizard…')
        wizard_btn.clicked.connect(self.figure_wizard_requested.emit)
        btn_row.addWidget(wizard_btn)
        left_lay.addLayout(btn_row)

        layout.addWidget(left)

        # ── Right: preview canvas ────────────────────────────────────
        self.canvas = PlotCanvas(figsize=(10, 7), dpi=100)
        layout.addWidget(self.canvas, 1)

    # ── Figure rendering ─────────────────────────────────────────────

    def _on_figure_selected(self, current, previous):
        if current is None:
            return
        self._render_figure(current.data(Qt.UserRole))

    def _on_refresh(self):
        item = self.fig_list.currentItem()
        if item:
            self._render_figure(item.data(Qt.UserRole))

    def _render_figure(self, fig_key: str):
        """Render the selected figure type on the canvas."""
        steps = self.state.strip_steps
        if not steps:
            self.canvas.clear()
            ax = self.canvas.add_subplot(111)
            ax.text(0.5, 0.5, 'No strip results available.\nRun a stripping workflow first.',
                    ha='center', va='center', fontsize=12, color='gray',
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw()
            return

        self.canvas.clear()
        cfg = self.state.figure_configs.get(fig_key, {})

        try:
            from ..visualization import figure_generators as fg
            fig = fg.generate_figure(fig_key, steps, cfg,
                                     dpi=self.dpi_spin.value(),
                                     font_size=self.font_spin.value())
            # Copy generated figure onto our canvas
            self.canvas.figure.clear()
            # Re-render using core visualization
            self._render_from_steps(fig_key, steps, cfg)
        except (ImportError, AttributeError):
            self._render_from_steps(fig_key, steps, cfg)

        self.canvas.draw()

    def _render_from_steps(self, fig_key: str, steps: list, cfg: dict):
        """Fallback: render figures directly using matplotlib."""
        import matplotlib.pyplot as plt

        if fig_key == 'hv_overlay':
            ax = self.canvas.add_subplot(111)
            cmap = plt.cm.get_cmap(cfg.get('cmap', 'tab10'))
            for i, step in enumerate(steps):
                freqs, amps = step.get('freqs'), step.get('amps')
                if freqs is not None:
                    ax.plot(freqs, amps, color=cmap(i % 10),
                            lw=cfg.get('linewidth', 2),
                            alpha=cfg.get('alpha', 0.8),
                            label=step.get('name', f'Step {i}'))
                    if cfg.get('show_peaks', True):
                        f0 = step.get('f0')
                        if f0:
                            ax.plot(f0[0], f0[1], '*', color=cmap(i % 10),
                                    ms=cfg.get('marker_size', 8))
            ax.set_xscale('log')
            ax.grid(cfg.get('grid', True), alpha=0.3)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('H/V Amplitude')
            ax.legend(fontsize=7)
            ax.set_title('HV Curves Overlay')

        elif fig_key == 'peak_evolution':
            ax = self.canvas.add_subplot(111)
            f0s = [(i, step['f0'][0]) for i, step in enumerate(steps) if step.get('f0')]
            if f0s:
                xs, ys = zip(*f0s)
                ax.plot(xs, ys, 'o-', color='#E63946',
                        ms=cfg.get('marker_size', 8),
                        lw=cfg.get('linewidth', 2))
                if cfg.get('show_fill', True):
                    ax.fill_between(xs, 0, ys, alpha=0.15, color='#E63946')
            ax.set_xlabel('Step')
            ax.set_ylabel('f0 (Hz)')
            ax.grid(cfg.get('grid', True), alpha=0.3)
            ax.set_title('Peak Evolution')

        elif fig_key == 'waterfall':
            ax = self.canvas.add_subplot(111)
            offset = cfg.get('offset_factor', 1.5)
            cmap = plt.cm.get_cmap(cfg.get('cmap', 'tab10'))
            for i, step in enumerate(steps):
                freqs, amps = step.get('freqs'), step.get('amps')
                if freqs is not None:
                    shifted = amps + i * offset
                    ax.plot(freqs, shifted, color=cmap(i % 10),
                            lw=cfg.get('linewidth', 2),
                            alpha=cfg.get('alpha', 0.8))
                    ax.fill_between(freqs, i * offset, shifted,
                                    color=cmap(i % 10), alpha=0.1)
            ax.set_xscale('log')
            ax.grid(cfg.get('grid', True), alpha=0.3)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('H/V + Offset')
            ax.set_title('Waterfall Plot')

        else:
            ax = self.canvas.add_subplot(111)
            ax.text(0.5, 0.5, f'{fig_key}\n(Full rendering available via Figure Wizard)',
                    ha='center', va='center', fontsize=11, color='gray',
                    transform=ax.transAxes)
            ax.set_axis_off()

        self.canvas.tight_layout()

    # ── Export ────────────────────────────────────────────────────────

    def _on_export_current(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Figure', '',
            'PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;EPS (*.eps)')
        if path:
            self.canvas.figure.savefig(path, dpi=self.dpi_spin.value(),
                                       bbox_inches='tight')
            self.state.status_message.emit(f'Figure saved: {path}')

    def _on_export_all(self):
        path = QFileDialog.getExistingDirectory(self, 'Export All Figures To')
        if not path:
            return
        import os
        for key, label in _FIGURE_TYPES:
            self._render_figure(key)
            for fmt in ('png', 'pdf'):
                fpath = os.path.join(path, f'{key}.{fmt}')
                self.canvas.figure.savefig(fpath, dpi=self.dpi_spin.value(),
                                           bbox_inches='tight')
        self.state.status_message.emit(f'All figures exported to {path}')
