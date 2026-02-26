"""
Per-figure settings panels for the Figure Wizard.

Each panel is a QWidget subclass exposing ``get_kwargs() -> dict`` that
returns keyword arguments understood by the corresponding
``draw_*_on_figure`` method.
"""

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Base panel with common helpers
# ---------------------------------------------------------------------------

class _BaseSettingsPanel(QWidget):
    """Shared helpers for settings panels."""

    def _row(self, label_text: str, widget, layout: QVBoxLayout):
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        row.addWidget(widget)
        row.addStretch()
        layout.addLayout(row)
        return widget

    def _spin_int(self, lo, hi, val, step=1):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        return s

    def _spin_float(self, lo, hi, val, step=0.1, decimals=2):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        return s

    def _combo(self, items, current=0):
        c = QComboBox()
        c.addItems(items)
        c.setCurrentIndex(current)
        return c

    def get_kwargs(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HV Overlay
# ---------------------------------------------------------------------------

class HVOverlaySettingsPanel(_BaseSettingsPanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.chk_log_x = QCheckBox("Log X axis")
        self.chk_log_x.setChecked(True)
        lay.addWidget(self.chk_log_x)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        lay.addWidget(self.chk_grid)

        self.cmap = self._combo([
            "cividis", "viridis", "plasma", "inferno", "magma",
            "coolwarm", "tab10", "tab20",
        ])
        self._row("Colormap:", self.cmap, lay)

        self.lw = self._spin_float(0.5, 6.0, 2.0, 0.5)
        self._row("Line width:", self.lw, lay)

        self.alpha = self._spin_float(0.1, 1.0, 0.8, 0.05)
        self._row("Alpha:", self.alpha, lay)

        self.chk_peaks = QCheckBox("Show peaks")
        self.chk_peaks.setChecked(True)
        lay.addWidget(self.chk_peaks)

        self.marker_size = self._spin_int(4, 20, 8)
        self._row("Peak marker size:", self.marker_size, lay)

        self.fmin = self._spin_float(0.01, 50.0, 0.0, 0.1)
        self.fmin.setSpecialValueText("auto")
        self._row("X min (Hz):", self.fmin, lay)

        self.fmax = self._spin_float(0.01, 100.0, 0.0, 1.0)
        self.fmax.setSpecialValueText("auto")
        self._row("X max (Hz):", self.fmax, lay)

        lay.addStretch()

    def get_kwargs(self) -> dict:
        kw = {
            "log_x": self.chk_log_x.isChecked(),
            "grid": self.chk_grid.isChecked(),
            "cmap": self.cmap.currentText(),
            "linewidth": self.lw.value(),
            "alpha": self.alpha.value(),
            "show_peaks": self.chk_peaks.isChecked(),
            "marker_size": self.marker_size.value(),
        }
        if self.fmin.value() > 0:
            kw["xlim_min"] = self.fmin.value()
        if self.fmax.value() > 0:
            kw["xlim_max"] = self.fmax.value()
        return kw


# ---------------------------------------------------------------------------
# Peak Evolution
# ---------------------------------------------------------------------------

class PeakEvolutionSettingsPanel(_BaseSettingsPanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        lay.addWidget(self.chk_grid)

        self.chk_fill = QCheckBox("Show fill under bars")
        self.chk_fill.setChecked(True)
        lay.addWidget(self.chk_fill)

        self.marker_size = self._spin_int(4, 20, 8)
        self._row("Marker size:", self.marker_size, lay)

        self.lw = self._spin_float(0.5, 6.0, 2.0, 0.5)
        self._row("Line width:", self.lw, lay)

        lay.addStretch()

    def get_kwargs(self) -> dict:
        return {
            "grid": self.chk_grid.isChecked(),
            "show_fill": self.chk_fill.isChecked(),
            "marker_size": self.marker_size.value(),
            "linewidth": self.lw.value(),
        }


# ---------------------------------------------------------------------------
# Interface Analysis
# ---------------------------------------------------------------------------

class InterfaceAnalysisSettingsPanel(_BaseSettingsPanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        lay.addWidget(self.chk_grid)

        self.marker_size = self._spin_int(4, 20, 8)
        self._row("Marker size:", self.marker_size, lay)

        self.lw = self._spin_float(0.5, 6.0, 2.0, 0.5)
        self._row("Line width:", self.lw, lay)

        self.annot_font = self._spin_int(6, 18, 10)
        self._row("Annotation font:", self.annot_font, lay)

        lay.addStretch()

    def get_kwargs(self) -> dict:
        return {
            "grid": self.chk_grid.isChecked(),
            "marker_size": self.marker_size.value(),
            "linewidth": self.lw.value(),
            "annot_font": self.annot_font.value(),
        }


# ---------------------------------------------------------------------------
# Waterfall
# ---------------------------------------------------------------------------

class WaterfallSettingsPanel(_BaseSettingsPanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.chk_log_x = QCheckBox("Log X axis")
        self.chk_log_x.setChecked(True)
        lay.addWidget(self.chk_log_x)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        lay.addWidget(self.chk_grid)

        self.cmap = self._combo([
            "cividis", "viridis", "plasma", "inferno", "magma",
            "coolwarm", "tab10", "tab20",
        ])
        self._row("Colormap:", self.cmap, lay)

        self.lw = self._spin_float(0.5, 6.0, 2.0, 0.5)
        self._row("Line width:", self.lw, lay)

        self.alpha = self._spin_float(0.1, 1.0, 0.8, 0.05)
        self._row("Alpha:", self.alpha, lay)

        self.offset = self._spin_float(0.5, 5.0, 1.5, 0.25)
        self._row("Offset factor:", self.offset, lay)

        self.chk_normalize = QCheckBox("Normalize curves")
        self.chk_normalize.setChecked(False)
        lay.addWidget(self.chk_normalize)

        lay.addStretch()

    def get_kwargs(self) -> dict:
        return {
            "log_x": self.chk_log_x.isChecked(),
            "grid": self.chk_grid.isChecked(),
            "cmap": self.cmap.currentText(),
            "linewidth": self.lw.value(),
            "alpha": self.alpha.value(),
            "offset_factor": self.offset.value(),
            "normalize": self.chk_normalize.isChecked(),
        }


# ---------------------------------------------------------------------------
# Publication (2×2)
# ---------------------------------------------------------------------------

class PublicationSettingsPanel(_BaseSettingsPanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        lay.addWidget(self.chk_grid)

        self.cmap = self._combo([
            "cividis", "viridis", "plasma", "inferno", "magma",
            "coolwarm", "tab10", "tab20",
        ])
        self._row("Colormap:", self.cmap, lay)

        self.lw = self._spin_float(0.5, 6.0, 2.0, 0.5)
        self._row("Line width:", self.lw, lay)

        self.alpha = self._spin_float(0.1, 1.0, 0.8, 0.05)
        self._row("Alpha:", self.alpha, lay)

        self.table_font = self._spin_int(6, 16, 8)
        self._row("Table font:", self.table_font, lay)

        lay.addStretch()

    def get_kwargs(self) -> dict:
        return {
            "grid": self.chk_grid.isChecked(),
            "cmap": self.cmap.currentText(),
            "linewidth": self.lw.value(),
            "alpha": self.alpha.value(),
            "table_font": self.table_font.value(),
        }


# ---------------------------------------------------------------------------
# Dual-Resonance
# ---------------------------------------------------------------------------

class DualResonanceSettingsPanel(_BaseSettingsPanel):

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        lay.addWidget(self.chk_grid)

        self.lw = self._spin_float(0.5, 6.0, 2.5, 0.5)
        self._row("Line width:", self.lw, lay)

        # --- Annotation offsets ---
        grp = QGroupBox("f₀ Annotation Offset")
        g_lay = QVBoxLayout(grp)
        self.f0_dx = self._spin_float(-10.0, 10.0, 0.0, 0.1)
        self._row("Δx (Hz):", self.f0_dx, g_lay)
        self.f0_dy = self._spin_float(-50.0, 50.0, 0.0, 1.0)
        self._row("Δy (amp):", self.f0_dy, g_lay)
        lay.addWidget(grp)

        grp2 = QGroupBox("f₁ Annotation Offset")
        g2_lay = QVBoxLayout(grp2)
        self.f1_dx = self._spin_float(-10.0, 10.0, 0.0, 0.1)
        self._row("Δx (Hz):", self.f1_dx, g2_lay)
        self.f1_dy = self._spin_float(-50.0, 50.0, 0.0, 1.0)
        self._row("Δy (amp):", self.f1_dy, g2_lay)
        lay.addWidget(grp2)

        self.chk_show_stripped = QCheckBox("Show stripped curve")
        self.chk_show_stripped.setChecked(True)
        lay.addWidget(self.chk_show_stripped)

        self.hs_ratio = self._spin_float(0.10, 1.0, 0.25, 0.05)
        self._row("HS depth %:", self.hs_ratio, lay)

        lay.addStretch()

    def get_kwargs(self) -> dict:
        return {
            "grid": self.chk_grid.isChecked(),
            "linewidth": self.lw.value(),
            "f0_offset": (self.f0_dx.value(), self.f0_dy.value()),
            "f1_offset": (self.f1_dx.value(), self.f1_dy.value()),
            "show_stripped": self.chk_show_stripped.isChecked(),
            "hs_ratio": self.hs_ratio.value(),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PANEL_REGISTRY: dict[str, type] = {
    "hv_overlay": HVOverlaySettingsPanel,
    "peak_evolution": PeakEvolutionSettingsPanel,
    "interface_analysis": InterfaceAnalysisSettingsPanel,
    "waterfall": WaterfallSettingsPanel,
    "publication": PublicationSettingsPanel,
    "dual_resonance": DualResonanceSettingsPanel,
}
