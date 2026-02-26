"""
Interactive Figure Wizard Dialog.

After stripping workflow completes, lets users review and tweak every
generated figure before final save.  Each figure type gets its own
dedicated settings panel (via QStackedWidget); changes re-render live
on a matplotlib canvas.
"""

from pathlib import Path
from typing import Dict, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from .figure_settings_panels import PANEL_REGISTRY


# Map of figure key -> display name
_FIGURE_TYPES = [
    ("hv_overlay", "HV Curves Overlay"),
    ("peak_evolution", "Peak Evolution"),
    ("interface_analysis", "Interface Analysis"),
    ("waterfall", "Waterfall Plot"),
    ("publication", "Publication Figure (2\u00d72)"),
    ("dual_resonance", "Dual-Resonance Separation"),
]


class FigureWizardDialog(QDialog):
    """Interactive figure review wizard for stripping results."""

    def __init__(
        self,
        strip_dir: str,
        output_dir: str,
        has_dual_resonance: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Figure Wizard \u2014 Review & Export")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        self._strip_dir = strip_dir
        self._output_dir = output_dir
        self._has_dr = has_dual_resonance
        self._reporter = None
        self._current_key = ""
        self._panels: Dict[str, object] = {}

        self._init_reporter()
        self._setup_ui()
        self._select_first()

    # ------------------------------------------------------------------ init

    def _init_reporter(self):
        try:
            from ...core.report_generator import ProgressiveStrippingReporter
            self._reporter = ProgressiveStrippingReporter(
                self._strip_dir, self._output_dir,
            )
        except Exception as exc:
            import traceback
            print(f"[FigureWizard] Reporter init failed: {exc}")
            traceback.print_exc()
            self._reporter = None

    # ------------------------------------------------------------------ UI

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: figure list + common + per-figure settings ---
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(5, 5, 5, 5)

        lbl = QLabel("Figures:")
        lbl.setStyleSheet("font-weight: bold;")
        left_lay.addWidget(lbl)

        self.fig_list = QListWidget()
        self.fig_list.setMaximumWidth(260)
        self._active_keys: list[str] = []
        for key, name in _FIGURE_TYPES:
            if key == "dual_resonance" and not self._has_dr:
                continue
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, key)
            self.fig_list.addItem(item)
            self._active_keys.append(key)
        self.fig_list.currentRowChanged.connect(self._on_fig_selected)
        left_lay.addWidget(self.fig_list)

        # --- Common settings ---
        common_grp = QGroupBox("Common")
        c_lay = QVBoxLayout(common_grp)
        row_dpi = QHBoxLayout()
        row_dpi.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        row_dpi.addWidget(self.dpi_spin)
        row_dpi.addStretch()
        c_lay.addLayout(row_dpi)

        row_font = QHBoxLayout()
        row_font.addWidget(QLabel("Font:"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 24)
        self.font_spin.setValue(12)
        row_font.addWidget(self.font_spin)
        row_font.addStretch()
        c_lay.addLayout(row_font)
        left_lay.addWidget(common_grp)

        # --- Per-figure settings (stacked) ---
        settings_lbl = QLabel("Figure Settings:")
        settings_lbl.setStyleSheet("font-weight: bold; margin-top: 4px;")
        left_lay.addWidget(settings_lbl)

        self.settings_stack = QStackedWidget()
        for key in self._active_keys:
            panel_cls = PANEL_REGISTRY.get(key)
            if panel_cls:
                panel = panel_cls()
                self._panels[key] = panel
                self.settings_stack.addWidget(panel)
            else:
                placeholder = QLabel(f"No settings for {key}")
                self._panels[key] = placeholder
                self.settings_stack.addWidget(placeholder)
        left_lay.addWidget(self.settings_stack, 1)

        splitter.addWidget(left)

        # --- Right: canvas ---
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(5, 5, 5, 5)

        self.figure = Figure(figsize=(14, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_lay.addWidget(self.toolbar)
        right_lay.addWidget(self.canvas, 1)

        # --- Buttons ---
        btn_row = QHBoxLayout()

        btn_apply = QPushButton("Apply")
        btn_apply.setToolTip("Re-render with current settings")
        btn_apply.clicked.connect(self._apply)
        btn_row.addWidget(btn_apply)

        btn_export = QPushButton("Export This")
        btn_export.clicked.connect(self._export_current)
        btn_row.addWidget(btn_export)

        btn_export_all = QPushButton("Export All")
        btn_export_all.setStyleSheet(
            "background-color: #0078d4; color: white; "
            "padding: 6px 16px; font-weight: bold;"
        )
        btn_export_all.clicked.connect(self._export_all)
        btn_row.addWidget(btn_export_all)

        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        right_lay.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([280, 720])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------ draw

    def _select_first(self):
        if self.fig_list.count() > 0:
            self.fig_list.setCurrentRow(0)

    def _on_fig_selected(self, row: int):
        if row < 0:
            return
        item = self.fig_list.item(row)
        self._current_key = item.data(Qt.UserRole)
        idx = self._active_keys.index(self._current_key)
        self.settings_stack.setCurrentIndex(idx)
        self._draw_current()

    def _apply(self, *_args):
        self._draw_current()

    def _get_kwargs_for(self, key: str) -> dict:
        kw = {"font_size": self.font_spin.value()}
        panel = self._panels.get(key)
        if panel and hasattr(panel, "get_kwargs"):
            kw.update(panel.get_kwargs())
        return kw

    def _draw_current(self):
        key = self._current_key
        if not key:
            return
        kw = self._get_kwargs_for(key)
        ok = self._draw_figure(key, kw)

        if not ok:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"No data for: {key}",
                    ha="center", va="center", fontsize=14,
                    transform=ax.transAxes)
        self.canvas.draw()

    def _draw_figure(self, key: str, kw: dict) -> bool:
        if key == "dual_resonance":
            from ...visualization.resonance_plots import draw_resonance_separation
            return draw_resonance_separation(self._strip_dir, self.figure, **kw)
        if self._reporter is None:
            return False
        draw_fn = {
            "hv_overlay": self._reporter.draw_hv_overlay_on_figure,
            "peak_evolution": self._reporter.draw_peak_evolution_on_figure,
            "interface_analysis": self._reporter.draw_interface_analysis_on_figure,
            "waterfall": self._reporter.draw_waterfall_on_figure,
            "publication": self._reporter.draw_publication_on_figure,
        }.get(key)
        if draw_fn:
            return draw_fn(self.figure, **kw)
        return False

    # ------------------------------------------------------------------ export

    def _export_current(self):
        if not self._current_key:
            return
        dpi = self.dpi_spin.value()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure",
            f"{self._current_key}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All (*)",
        )
        if path:
            self.figure.savefig(path, dpi=dpi, bbox_inches='tight')

    def _export_all(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Export Directory",
        )
        if not folder:
            return
        out = Path(folder)
        dpi = self.dpi_spin.value()
        saved = 0

        for key in self._active_keys:
            kw = self._get_kwargs_for(key)
            ok = self._draw_figure(key, kw)
            if ok:
                self.figure.savefig(
                    str(out / f"{key}.png"), dpi=dpi, bbox_inches='tight',
                )
                saved += 1

        self._draw_current()
        QMessageBox.information(
            self, "Exported",
            f"Saved {saved} figures to:\n{folder}",
        )
