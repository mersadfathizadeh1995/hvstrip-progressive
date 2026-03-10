"""Figure Studio — Separate window for reviewing and exporting publication figures.

Ported from the old PySide6 FigureWizardDialog to PyQt5, enhanced with
bedrock-mapping-style layout. Provides live preview of 6 figure types
with per-figure settings panels.
"""
import os
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
    QStackedWidget, QFileDialog, QMessageBox, QFormLayout, QComboBox,
    QCheckBox, QStatusBar,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.style_constants import (
    BUTTON_PRIMARY, BUTTON_SUCCESS, SECONDARY_LABEL, EMOJI,
)

# Figure types available
_FIGURE_TYPES = [
    ("hv_overlay", "HV Curves Overlay"),
    ("peak_evolution", "Peak Evolution"),
    ("interface_analysis", "Interface Analysis"),
    ("waterfall", "Waterfall Plot"),
    ("publication", "Publication Figure (2x2)"),
    ("dual_resonance", "Dual-Resonance Separation"),
]


# ══════════════════════════════════════════════════════════════════
#  Settings Panels  (one per figure type)
# ══════════════════════════════════════════════════════════════════

class _BasePanel(QWidget):
    def _spin_int(self, lo, hi, val, step=1):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
        return s

    def _spin_float(self, lo, hi, val, step=0.1, decimals=2):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val)
        s.setSingleStep(step); s.setDecimals(decimals)
        return s

    def _combo(self, items, current=0):
        c = QComboBox(); c.addItems(items)
        if isinstance(current, int):
            c.setCurrentIndex(current)
        else:
            idx = c.findText(current)
            if idx >= 0: c.setCurrentIndex(idx)
        return c

    def _row(self, label, widget, layout):
        layout.addRow(label, widget)

    def get_kwargs(self):
        return {}


class HVOverlayPanel(_BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.log_x = QCheckBox(); self.log_x.setChecked(True)
        self.grid = QCheckBox(); self.grid.setChecked(True)
        self.cmap = self._combo(
            ["Blues", "BuPu", "GnBu", "PuBu", "YlGnBu",
             "cividis", "viridis", "plasma", "inferno", "tab10"])
        self.lw = self._spin_float(0.5, 6.0, 1.5)
        self.alpha = self._spin_float(0.1, 1.0, 0.85, 0.05)
        self.show_peaks = QCheckBox(); self.show_peaks.setChecked(True)
        self.marker_size = self._spin_int(4, 20, 8)
        self.show_annotations = QCheckBox(); self.show_annotations.setChecked(True)
        self.annotation_size = self._spin_int(5, 18, 8)
        self.xlim_min = self._spin_float(0, 50, 0, 0.1)
        self.xlim_max = self._spin_float(0, 100, 0, 1)
        for lbl, w in [("Log X:", self.log_x), ("Grid:", self.grid),
                        ("Colormap:", self.cmap), ("Line Width:", self.lw),
                        ("Alpha:", self.alpha), ("Show Peaks:", self.show_peaks),
                        ("Marker Size:", self.marker_size),
                        ("Peak Labels:", self.show_annotations),
                        ("Label Size:", self.annotation_size),
                        ("X-axis Min:", self.xlim_min), ("X-axis Max:", self.xlim_max)]:
            form.addRow(lbl, w)

    def get_kwargs(self):
        kw = {
            "log_x": self.log_x.isChecked(), "grid": self.grid.isChecked(),
            "cmap": self.cmap.currentText(), "linewidth": self.lw.value(),
            "alpha": self.alpha.value(), "show_peaks": self.show_peaks.isChecked(),
            "marker_size": self.marker_size.value(),
            "show_annotations": self.show_annotations.isChecked(),
            "annotation_size": self.annotation_size.value(),
        }
        if self.xlim_min.value() > 0: kw["xlim_min"] = self.xlim_min.value()
        if self.xlim_max.value() > 0: kw["xlim_max"] = self.xlim_max.value()
        return kw


class PeakEvolutionPanel(_BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.grid = QCheckBox(); self.grid.setChecked(True)
        self.show_fill = QCheckBox(); self.show_fill.setChecked(True)
        self.marker_size = self._spin_int(4, 20, 8)
        self.lw = self._spin_float(0.5, 6.0, 1.5)
        self.show_annotations = QCheckBox(); self.show_annotations.setChecked(True)
        self.annotation_size = self._spin_int(5, 18, 8)
        for lbl, w in [("Grid:", self.grid), ("Show Fill:", self.show_fill),
                        ("Marker Size:", self.marker_size), ("Line Width:", self.lw),
                        ("Peak Labels:", self.show_annotations),
                        ("Label Size:", self.annotation_size)]:
            form.addRow(lbl, w)

    def get_kwargs(self):
        return {"grid": self.grid.isChecked(), "show_fill": self.show_fill.isChecked(),
                "marker_size": self.marker_size.value(), "linewidth": self.lw.value(),
                "show_annotations": self.show_annotations.isChecked(),
                "annotation_size": self.annotation_size.value()}


class InterfaceAnalysisPanel(_BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.grid = QCheckBox(); self.grid.setChecked(True)
        self.marker_size = self._spin_int(4, 20, 8)
        self.lw = self._spin_float(0.5, 6.0, 1.5)
        self.annot_font = self._spin_int(6, 18, 8)
        for lbl, w in [("Grid:", self.grid), ("Marker Size:", self.marker_size),
                        ("Line Width:", self.lw), ("Annotation Font:", self.annot_font)]:
            form.addRow(lbl, w)

    def get_kwargs(self):
        return {"grid": self.grid.isChecked(), "marker_size": self.marker_size.value(),
                "linewidth": self.lw.value(), "annot_font": self.annot_font.value()}


class WaterfallPanel(_BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.log_x = QCheckBox(); self.log_x.setChecked(True)
        self.grid = QCheckBox(); self.grid.setChecked(True)
        self.cmap = self._combo(
            ["Blues", "BuPu", "GnBu", "PuBu", "YlGnBu",
             "cividis", "viridis", "plasma", "inferno", "tab10"])
        self.lw = self._spin_float(0.5, 6.0, 1.5)
        self.alpha = self._spin_float(0.1, 1.0, 0.85)
        self.offset = self._spin_float(0.5, 5.0, 1.0, 0.1)
        self.normalize = QCheckBox(); self.normalize.setChecked(True)
        self.show_annotations = QCheckBox(); self.show_annotations.setChecked(True)
        self.annotation_size = self._spin_int(5, 18, 8)
        for lbl, w in [("Log X:", self.log_x), ("Grid:", self.grid),
                        ("Colormap:", self.cmap), ("Line Width:", self.lw),
                        ("Alpha:", self.alpha), ("Offset Factor:", self.offset),
                        ("Normalize:", self.normalize),
                        ("Peak Labels:", self.show_annotations),
                        ("Label Size:", self.annotation_size)]:
            form.addRow(lbl, w)

    def get_kwargs(self):
        return {
            "log_x": self.log_x.isChecked(), "grid": self.grid.isChecked(),
            "cmap": self.cmap.currentText(), "linewidth": self.lw.value(),
            "alpha": self.alpha.value(), "offset_factor": self.offset.value(),
            "normalize": self.normalize.isChecked(),
            "show_annotations": self.show_annotations.isChecked(),
            "annotation_size": self.annotation_size.value(),
        }


class PublicationPanel(_BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.grid = QCheckBox(); self.grid.setChecked(True)
        self.cmap = self._combo(
            ["Blues", "BuPu", "GnBu", "PuBu", "YlGnBu",
             "cividis", "viridis", "plasma", "tab10"])
        self.lw = self._spin_float(0.5, 6.0, 1.5)
        self.alpha = self._spin_float(0.1, 1.0, 0.85)
        self.table_font = self._spin_int(6, 16, 8)
        self.show_annotations = QCheckBox(); self.show_annotations.setChecked(True)
        self.annotation_size = self._spin_int(5, 18, 8)
        for lbl, w in [("Grid:", self.grid), ("Colormap:", self.cmap),
                        ("Line Width:", self.lw), ("Alpha:", self.alpha),
                        ("Table Font:", self.table_font),
                        ("Peak Labels:", self.show_annotations),
                        ("Label Size:", self.annotation_size)]:
            form.addRow(lbl, w)

    def get_kwargs(self):
        return {"grid": self.grid.isChecked(), "cmap": self.cmap.currentText(),
                "linewidth": self.lw.value(), "alpha": self.alpha.value(),
                "table_font": self.table_font.value(),
                "show_annotations": self.show_annotations.isChecked(),
                "annotation_size": self.annotation_size.value()}


class DualResonancePanel(_BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        self.grid = QCheckBox(); self.grid.setChecked(True)
        self.lw = self._spin_float(0.5, 6.0, 1.5)
        self.f0_dx = self._spin_float(-5, 5, 0.0, 0.1)
        self.f0_dy = self._spin_float(-10, 10, 0.0, 0.5)
        self.f1_dx = self._spin_float(-5, 5, 0.0, 0.1)
        self.f1_dy = self._spin_float(-10, 10, 0.0, 0.5)
        self.show_stripped = QCheckBox(); self.show_stripped.setChecked(True)
        self.hs_ratio = self._spin_float(0.1, 1.0, 0.25, 0.05)
        for lbl, w in [("Grid:", self.grid), ("Line Width:", self.lw),
                        ("f0 Offset X:", self.f0_dx), ("f0 Offset Y:", self.f0_dy),
                        ("f1 Offset X:", self.f1_dx), ("f1 Offset Y:", self.f1_dy),
                        ("Show Stripped:", self.show_stripped),
                        ("HS Depth %:", self.hs_ratio)]:
            form.addRow(lbl, w)

    def get_kwargs(self):
        return {
            "grid": self.grid.isChecked(), "linewidth": self.lw.value(),
            "f0_offset": (self.f0_dx.value(), self.f0_dy.value()),
            "f1_offset": (self.f1_dx.value(), self.f1_dy.value()),
            "show_stripped": self.show_stripped.isChecked(),
            "hs_ratio": self.hs_ratio.value(),
        }


_PANEL_CLASSES = {
    "hv_overlay": HVOverlayPanel,
    "peak_evolution": PeakEvolutionPanel,
    "interface_analysis": InterfaceAnalysisPanel,
    "waterfall": WaterfallPanel,
    "publication": PublicationPanel,
    "dual_resonance": DualResonancePanel,
}


# ══════════════════════════════════════════════════════════════════
#  Figure Studio Window
# ══════════════════════════════════════════════════════════════════

class FigureStudioWindow(QMainWindow):
    """Separate window for reviewing and exporting publication figures."""

    def __init__(self, strip_dir, output_dir=None, has_dual_resonance=False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Studio — Progressive Stripping")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        self._strip_dir = strip_dir
        self._output_dir = output_dir or str(Path(strip_dir).parent)
        self._has_dr = has_dual_resonance
        self._reporter = None
        self._current_key = None
        self._panels = {}
        self._active_keys = []

        self._init_reporter()
        self._build_ui()
        self._select_first()

    def _init_reporter(self):
        try:
            from ..core.report_generator import ProgressiveStrippingReporter
            self._reporter = ProgressiveStrippingReporter(
                self._strip_dir, output_dir=self._output_dir)
        except Exception as e:
            print(f"[FigureStudio] Reporter init error: {e}")

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: Figure list + settings ───────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(2, 2, 2, 2)

        left_lay.addWidget(QLabel("<b>Figure Type</b>"))
        self._fig_list = QListWidget()
        self._fig_list.setMaximumHeight(180)
        for key, label in _FIGURE_TYPES:
            if key == "dual_resonance" and not self._has_dr:
                continue
            self._fig_list.addItem(label)
            self._active_keys.append(key)
        self._fig_list.currentRowChanged.connect(self._on_fig_selected)
        left_lay.addWidget(self._fig_list)

        # Common settings
        left_lay.addWidget(QLabel("<b>Common Settings</b>"))
        common = QFormLayout()
        self._dpi = QSpinBox()
        self._dpi.setRange(72, 600); self._dpi.setValue(200)
        common.addRow("DPI:", self._dpi)
        self._font_size = QSpinBox()
        self._font_size.setRange(6, 24); self._font_size.setValue(10)
        common.addRow("Font Size:", self._font_size)
        left_lay.addLayout(common)

        # Per-figure settings stack
        left_lay.addWidget(QLabel("<b>Figure Settings</b>"))
        self._settings_stack = QStackedWidget()
        for key in self._active_keys:
            cls = _PANEL_CLASSES.get(key, _BasePanel)
            panel = cls()
            self._panels[key] = panel
            self._settings_stack.addWidget(panel)
        left_lay.addWidget(self._settings_stack)
        left_lay.addStretch()

        left.setMinimumWidth(260)
        left.setMaximumWidth(360)
        splitter.addWidget(left)

        # ── Right: Canvas + buttons ────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(2, 2, 2, 2)

        self._canvas = MatplotlibWidget(figsize=(12, 8))
        right_lay.addWidget(self._canvas)

        btn_row = QHBoxLayout()
        btn_apply = QPushButton(f"{EMOJI['run']} Apply")
        btn_apply.setStyleSheet(BUTTON_PRIMARY)
        btn_apply.clicked.connect(self._apply)
        btn_row.addWidget(btn_apply)

        btn_export = QPushButton(f"{EMOJI['export']} Export This")
        btn_export.clicked.connect(self._export_current)
        btn_row.addWidget(btn_export)

        btn_all = QPushButton(f"{EMOJI['export']} Export All")
        btn_all.setStyleSheet(BUTTON_SUCCESS)
        btn_all.clicked.connect(self._export_all)
        btn_row.addWidget(btn_all)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_close)
        right_lay.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        root.addWidget(splitter)
        self.setCentralWidget(central)

        sb = QStatusBar()
        self._status = QLabel("Ready")
        sb.addWidget(self._status, 1)
        self.setStatusBar(sb)

    def _select_first(self):
        if self._fig_list.count() > 0:
            self._fig_list.setCurrentRow(0)

    def _on_fig_selected(self, row):
        if 0 <= row < len(self._active_keys):
            self._current_key = self._active_keys[row]
            self._settings_stack.setCurrentIndex(row)
            self._draw_current()

    def _apply(self):
        self._draw_current()

    def _draw_current(self):
        if not self._current_key or not self._reporter:
            return
        kw = self._get_kwargs()
        fig = self._canvas.figure
        fig.clear()

        try:
            self._draw_figure(self._current_key, kw)
            self._status.setText(f"Rendered: {self._current_key}")
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                    color="red", transform=ax.transAxes, fontsize=12)
            self._status.setText(f"Error: {e}")

        self._canvas.refresh()

    def _get_kwargs(self):
        kw = {}
        if self._current_key and self._current_key in self._panels:
            kw = self._panels[self._current_key].get_kwargs()
        kw["font_size"] = self._font_size.value()
        kw["dpi"] = self._dpi.value()
        return kw

    def _draw_figure(self, key, kw):
        fig = self._canvas.figure
        r = self._reporter

        dispatch = {
            "hv_overlay": r.draw_hv_overlay_on_figure,
            "peak_evolution": r.draw_peak_evolution_on_figure,
            "interface_analysis": r.draw_interface_analysis_on_figure,
            "waterfall": r.draw_waterfall_on_figure,
            "publication": r.draw_publication_on_figure,
        }

        if key in dispatch:
            dispatch[key](fig, **kw)
        elif key == "dual_resonance":
            try:
                from ..visualization.resonance_plots import (
                    draw_resonance_separation)
                draw_resonance_separation(self._strip_dir, fig, **kw)
            except Exception as e:
                raise RuntimeError(f"Dual-resonance plot failed: {e}") from e

    def _export_current(self):
        if not self._current_key:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure",
            f"{self._current_key}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All (*)")
        if path:
            dpi = self._dpi.value()
            self._canvas.figure.savefig(path, dpi=dpi, bbox_inches="tight")
            self._status.setText(f"Exported: {path}")

    def _export_all(self):
        d = QFileDialog.getExistingDirectory(self, "Export All Figures To")
        if not d:
            return
        dpi = self._dpi.value()
        for key in self._active_keys:
            kw = {}
            if key in self._panels:
                kw = self._panels[key].get_kwargs()
            kw["font_size"] = self._font_size.value()
            kw["dpi"] = dpi
            fig = self._canvas.figure
            fig.clear()
            try:
                self._draw_figure(key, kw)
                for ext in ["png", "pdf"]:
                    fig.savefig(
                        os.path.join(d, f"{key}.{ext}"),
                        dpi=dpi, bbox_inches="tight")
            except Exception:
                pass
        self._status.setText(f"All figures exported to {d}")
        QMessageBox.information(self, "Export Complete",
                                f"All figures exported to:\n{d}")
