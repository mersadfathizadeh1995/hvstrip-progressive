"""HV Curve View — Interactive HV forward-model display with peak picking.

Right-panel canvas tab showing the computed H/V spectral ratio curve
with collapsible plot settings, interactive peak selection, and save.
"""
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit,
    QFileDialog, QSizePolicy,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.collapsible_group import CollapsibleGroupBox
from ..widgets.style_constants import (
    EMOJI, BUTTON_SUCCESS, GEAR_BUTTON, SECONDARY_LABEL,
)

# ── Constants ──────────────────────────────────────────────────
CURVE_COLORS = {
    "Royal Blue": "royalblue",
    "Steel Blue": "steelblue",
    "Teal": "teal",
    "Dark Green": "darkgreen",
    "Crimson": "crimson",
    "Black": "black",
    "Navy": "navy",
}

SEC_COLORS = ["green", "purple", "orange", "brown", "teal"]

LEGEND_POSITIONS = [
    "upper right", "upper left", "lower right", "lower left",
    "center right", "center left", "best",
]

MARKER_SHAPES = {
    "★ Star": "*",
    "● Circle": "o",
    "◆ Diamond": "D",
    "▲ Triangle": "^",
    "■ Square": "s",
}

FIGURE_SIZES = {
    "Standard (10×7)": (10, 7),
    "Large (14×10)": (14, 10),
    "Publication (12×8)": (12, 8),
    "Wide (16×6)": (16, 6),
}


class HVCurveView(QWidget):
    """Canvas view for a single HV forward-model curve."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._freqs = None
        self._amps = None
        self._profile = None
        self._f0 = None          # (freq, amp, idx)
        self._secondary = []     # [(freq, amp, idx), ...]
        self._pick_f0 = False
        self._pick_sec = False
        self._build_ui()

    # ══════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        # 1. Plot canvas
        self._plot = MatplotlibWidget(figsize=(14, 5))
        self._plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._plot.canvas.mpl_connect("button_press_event", self._on_click)
        lay.addWidget(self._plot, 1)

        # 2. Collapsible Plot Settings
        self._build_settings_section(lay)

        # 3. Peak selection row (outside collapsible)
        self._build_peak_row(lay)

        # 4. Save row
        self._build_save_row(lay)

    # ── Settings section ──────────────────────────────────────
    def _build_settings_section(self, parent_lay):
        grp = CollapsibleGroupBox(
            f"{EMOJI['settings']} Plot Settings", collapsed=True)
        settings_lay = QVBoxLayout()
        settings_lay.setSpacing(2)
        settings_lay.setContentsMargins(2, 2, 2, 2)

        self._build_display_sub(settings_lay)
        self._build_curve_sub(settings_lay)
        self._build_peaks_sub(settings_lay)
        self._build_axes_sub(settings_lay)
        self._build_labels_sub(settings_lay)
        self._build_legend_sub(settings_lay)
        self._build_export_sub(settings_lay)

        grp.setContentLayout(settings_lay)
        parent_lay.addWidget(grp)

    def _build_display_sub(self, parent):
        """Display toggles — always visible row inside settings."""
        row = QHBoxLayout()
        row.setSpacing(6)
        self._log_x = QCheckBox("Log X"); self._log_x.setChecked(True)
        self._log_y = QCheckBox("Log Y")
        self._grid = QCheckBox("Grid"); self._grid.setChecked(True)
        self._show_vs = QCheckBox("Show Vs"); self._show_vs.setChecked(True)
        for w in [self._log_x, self._log_y, self._grid, self._show_vs]:
            w.stateChanged.connect(lambda _: self._redraw())
            row.addWidget(w)
        row.addStretch()
        parent.addLayout(row)

    def _build_curve_sub(self, parent):
        grp = CollapsibleGroupBox("Curve Style", collapsed=True)
        lay = QHBoxLayout()
        lay.setSpacing(4); lay.setContentsMargins(2, 2, 2, 2)

        lay.addWidget(QLabel("Color:"))
        self._line_color = QComboBox()
        self._line_color.addItems(list(CURVE_COLORS.keys()))
        self._line_color.setMaximumWidth(100)
        self._line_color.currentIndexChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._line_color)

        lay.addWidget(QLabel("LW:"))
        self._line_width = QDoubleSpinBox()
        self._line_width.setRange(0.5, 5.0); self._line_width.setValue(1.5)
        self._line_width.setSingleStep(0.5); self._line_width.setMaximumWidth(55)
        self._line_width.valueChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._line_width)

        lay.addWidget(QLabel("HS:"))
        self._hs_pct = QDoubleSpinBox()
        self._hs_pct.setRange(10, 100); self._hs_pct.setValue(25)
        self._hs_pct.setSuffix(" %"); self._hs_pct.setMaximumWidth(65)
        self._hs_pct.valueChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._hs_pct)

        lay.addStretch()
        grp.setContentLayout(lay)
        parent.addWidget(grp)

    def _build_peaks_sub(self, parent):
        grp = CollapsibleGroupBox("Peak Markers", collapsed=True)
        lay = QHBoxLayout()
        lay.setSpacing(4); lay.setContentsMargins(2, 2, 2, 2)

        lay.addWidget(QLabel("f0 Shape:"))
        self._f0_marker = QComboBox()
        self._f0_marker.addItems(list(MARKER_SHAPES.keys()))
        self._f0_marker.setMaximumWidth(80)
        self._f0_marker.currentIndexChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._f0_marker)

        lay.addWidget(QLabel("Size:"))
        self._f0_size = QSpinBox()
        self._f0_size.setRange(4, 30); self._f0_size.setValue(14)
        self._f0_size.setMaximumWidth(50)
        self._f0_size.valueChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._f0_size)

        lay.addWidget(QLabel("Sec:"))
        self._sec_marker = QComboBox()
        self._sec_marker.addItems(list(MARKER_SHAPES.keys()))
        self._sec_marker.setCurrentIndex(1)
        self._sec_marker.setMaximumWidth(80)
        self._sec_marker.currentIndexChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._sec_marker)

        lay.addWidget(QLabel("Ann:"))
        self._ann_font = QSpinBox()
        self._ann_font.setRange(6, 20); self._ann_font.setValue(8)
        self._ann_font.setMaximumWidth(50)
        self._ann_font.valueChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._ann_font)

        lay.addStretch()
        grp.setContentLayout(lay)
        parent.addWidget(grp)

    def _build_axes_sub(self, parent):
        grp = CollapsibleGroupBox("Axes & Grid", collapsed=True)
        lay = QHBoxLayout()
        lay.setSpacing(4); lay.setContentsMargins(2, 2, 2, 2)

        lay.addWidget(QLabel("Grid α:"))
        self._grid_alpha = QDoubleSpinBox()
        self._grid_alpha.setRange(0.05, 1.0); self._grid_alpha.setValue(0.3)
        self._grid_alpha.setSingleStep(0.05); self._grid_alpha.setMaximumWidth(55)
        self._grid_alpha.valueChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._grid_alpha)

        self._chk_minor_grid = QCheckBox("Minor Grid")
        self._chk_minor_grid.setChecked(True)
        self._chk_minor_grid.stateChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._chk_minor_grid)

        lay.addStretch()
        grp.setContentLayout(lay)
        parent.addWidget(grp)

    def _build_labels_sub(self, parent):
        grp = CollapsibleGroupBox("Labels", collapsed=True)
        lay = QVBoxLayout()
        lay.setSpacing(2); lay.setContentsMargins(2, 2, 2, 2)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Title:"))
        self._title_edit = QLineEdit("HV Forward Model")
        self._title_edit.editingFinished.connect(self._redraw)
        r1.addWidget(self._title_edit, 1)
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("X:"))
        self._xlabel_edit = QLineEdit("Frequency (Hz)")
        r2.addWidget(self._xlabel_edit, 1)
        r2.addWidget(QLabel("Y:"))
        self._ylabel_edit = QLineEdit("H/V Ratio")
        r2.addWidget(self._ylabel_edit, 1)
        for w in [self._xlabel_edit, self._ylabel_edit]:
            w.editingFinished.connect(self._redraw)
        lay.addLayout(r2)

        grp.setContentLayout(lay)
        parent.addWidget(grp)

    def _build_legend_sub(self, parent):
        grp = CollapsibleGroupBox("Legend", collapsed=True)
        lay = QHBoxLayout()
        lay.setSpacing(4); lay.setContentsMargins(2, 2, 2, 2)

        self._show_legend = QCheckBox("Show")
        self._show_legend.setChecked(True)
        self._show_legend.stateChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._show_legend)

        lay.addWidget(QLabel("Pos:"))
        self._legend_loc = QComboBox()
        self._legend_loc.addItems(LEGEND_POSITIONS)
        self._legend_loc.setMaximumWidth(100)
        self._legend_loc.currentIndexChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._legend_loc)

        lay.addWidget(QLabel("Font:"))
        self._legend_font = QSpinBox()
        self._legend_font.setRange(5, 16); self._legend_font.setValue(8)
        self._legend_font.setMaximumWidth(50)
        self._legend_font.valueChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._legend_font)

        lay.addWidget(QLabel("α:"))
        self._legend_alpha = QDoubleSpinBox()
        self._legend_alpha.setRange(0.0, 1.0); self._legend_alpha.setValue(0.85)
        self._legend_alpha.setSingleStep(0.05); self._legend_alpha.setMaximumWidth(55)
        self._legend_alpha.valueChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._legend_alpha)

        self._legend_frame = QCheckBox("Frame")
        self._legend_frame.setChecked(True)
        self._legend_frame.stateChanged.connect(lambda _: self._redraw())
        lay.addWidget(self._legend_frame)

        lay.addStretch()
        grp.setContentLayout(lay)
        parent.addWidget(grp)

    def _build_export_sub(self, parent):
        grp = CollapsibleGroupBox("Export", collapsed=True)
        lay = QHBoxLayout()
        lay.setSpacing(4); lay.setContentsMargins(2, 2, 2, 2)

        lay.addWidget(QLabel("DPI:"))
        self._dpi = QSpinBox()
        self._dpi.setRange(72, 600); self._dpi.setValue(150)
        self._dpi.setMaximumWidth(65)
        lay.addWidget(self._dpi)

        lay.addWidget(QLabel("Fmt:"))
        self._export_fmt = QComboBox()
        self._export_fmt.addItems(["PNG", "PDF", "SVG"])
        self._export_fmt.setMaximumWidth(65)
        lay.addWidget(self._export_fmt)

        lay.addWidget(QLabel("Size:"))
        self._fig_size = QComboBox()
        self._fig_size.addItems(list(FIGURE_SIZES.keys()))
        self._fig_size.setMaximumWidth(140)
        lay.addWidget(self._fig_size)

        lay.addStretch()
        grp.setContentLayout(lay)
        parent.addWidget(grp)

    # ── Peak row (outside collapsible) ────────────────────────
    def _build_peak_row(self, parent_lay):
        peak = QHBoxLayout()
        peak.setSpacing(4)

        self._btn_f0 = QPushButton(f"{EMOJI['peak']} Select f0")
        self._btn_f0.setCheckable(True)
        self._btn_f0.setStyleSheet(
            "QPushButton:checked { background-color: #4CAF50; color: white; "
            "font-weight: bold; border-radius: 3px; padding: 3px 8px; }")
        self._btn_f0.toggled.connect(self._toggle_f0)
        peak.addWidget(self._btn_f0)

        self._btn_sec = QPushButton("🔶 Select Secondary")
        self._btn_sec.setCheckable(True)
        self._btn_sec.setStyleSheet(
            "QPushButton:checked { background-color: #FF9800; color: white; "
            "font-weight: bold; border-radius: 3px; padding: 3px 8px; }")
        self._btn_sec.toggled.connect(self._toggle_sec)
        peak.addWidget(self._btn_sec)

        self._btn_undo_sec = QPushButton("↩ Undo Sec.")
        self._btn_undo_sec.setToolTip("Remove last secondary peak")
        self._btn_undo_sec.clicked.connect(self._undo_sec)
        peak.addWidget(self._btn_undo_sec)

        btn_clear = QPushButton("✕ Clear Peaks")
        btn_clear.clicked.connect(self._clear_all_peaks)
        peak.addWidget(btn_clear)

        self._sel_label = QLabel("")
        self._sel_label.setStyleSheet("font-size: 11px; color: #333;")
        peak.addWidget(self._sel_label, 1)
        parent_lay.addLayout(peak)

    # ── Save row ──────────────────────────────────────────────
    def _build_save_row(self, parent_lay):
        row = QHBoxLayout()
        row.setSpacing(4)

        self._btn_save_csv = QPushButton(f"{EMOJI['save']} Save CSV")
        self._btn_save_csv.setEnabled(False)
        self._btn_save_csv.clicked.connect(self._save_csv)
        row.addWidget(self._btn_save_csv)

        self._btn_save_all = QPushButton(f"{EMOJI['save']} Save All")
        self._btn_save_all.setStyleSheet(BUTTON_SUCCESS)
        self._btn_save_all.setEnabled(False)
        self._btn_save_all.setToolTip(
            "Save HV figure, Vs figure, peak info, CSV, Vs30")
        self._btn_save_all.clicked.connect(self._save_all)
        row.addWidget(self._btn_save_all)

        self._btn_gear = QPushButton(EMOJI["settings"])
        self._btn_gear.setFixedSize(28, 28)
        self._btn_gear.setStyleSheet(GEAR_BUTTON)
        self._btn_gear.setToolTip(
            "Choose which figures to save & output directory")
        self._btn_gear.clicked.connect(self._open_save_options)
        row.addWidget(self._btn_gear)

        row.addStretch()

        self._save_label = QLabel("")
        self._save_label.setStyleSheet(SECONDARY_LABEL)
        row.addWidget(self._save_label)
        parent_lay.addLayout(row)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def set_data(self, freqs, amps, profile=None):
        self._freqs = np.asarray(freqs)
        self._amps = np.asarray(amps)
        self._profile = profile
        idx = int(np.argmax(self._amps))
        self._f0 = (self._freqs[idx], self._amps[idx], idx)
        self._secondary = []
        self._btn_save_csv.setEnabled(True)
        self._btn_save_all.setEnabled(True)
        self._redraw()
        self._update_label()

    def get_output_dir(self):
        """Get the output directory from the panel."""
        if self._mw:
            from ..strip_window import MODE_FWD_SINGLE
            panel = self._mw.get_panel(MODE_FWD_SINGLE)
            if panel and hasattr(panel, '_out_dir'):
                d = panel._out_dir.text().strip()
                if d:
                    return Path(d)
        return None

    # ══════════════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════════════
    def _redraw(self):
        if self._freqs is None:
            return
        fig = self._plot.figure
        fig.clear()

        show_vs = (self._show_vs.isChecked() and self._profile is not None)
        if show_vs:
            import matplotlib.gridspec as gs
            spec = gs.GridSpec(
                1, 2, width_ratios=[4, 1], wspace=0.05, figure=fig)
            ax = fig.add_subplot(spec[0])
            ax_vs = fig.add_subplot(spec[1])
        else:
            ax = fig.add_subplot(111)
            ax_vs = None

        color = CURVE_COLORS.get(
            self._line_color.currentText(), "royalblue")
        lw = self._line_width.value()
        ax.plot(self._freqs, self._amps, color=color, lw=lw, label="H/V")

        f0_shape = MARKER_SHAPES.get(
            self._f0_marker.currentText(), "*")
        f0_sz = self._f0_size.value()
        sec_shape = MARKER_SHAPES.get(
            self._sec_marker.currentText(), "o")
        ann_fs = self._ann_font.value()

        # Primary peak
        if self._f0:
            f, a, _ = self._f0
            ax.plot(f, a, f0_shape, color="red", ms=f0_sz, zorder=5,
                    markeredgecolor="darkred", markeredgewidth=0.8)
            ax.axvline(f, color="red", ls="--", alpha=0.5, lw=0.8)
            ax.annotate(
                f"f0 = {f:.3f} Hz\nA = {a:.2f}",
                xy=(f, a), xytext=(10, 10),
                textcoords="offset points", fontsize=ann_fs, color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="red", alpha=0.85, lw=0.5),
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        # Secondary peaks
        for i, (sf, sa, _) in enumerate(self._secondary):
            sc = SEC_COLORS[i % len(SEC_COLORS)]
            ax.plot(sf, sa, sec_shape, color=sc, ms=f0_sz - 3, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.axvline(sf, color=sc, ls=":", alpha=0.5, lw=0.8)
            y_off = -16 if i % 2 == 0 else 12
            ax.annotate(
                f"Sec.{i+1}: {sf:.3f} Hz ({sa:.2f})",
                xy=(sf, sa), xytext=(8, y_off),
                textcoords="offset points", fontsize=max(ann_fs - 1, 6),
                color=sc, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=sc, alpha=0.85, lw=0.5))

        if self._log_x.isChecked():
            ax.set_xscale("log")
        if self._log_y.isChecked():
            ax.set_yscale("log")
        ax.set_xlabel(self._xlabel_edit.text())
        ax.set_ylabel(self._ylabel_edit.text())
        ax.set_title(
            self._title_edit.text(), fontsize=12, fontweight="bold")
        if self._grid.isChecked():
            which = "both" if self._chk_minor_grid.isChecked() else "major"
            ax.grid(True, alpha=self._grid_alpha.value(), which=which)

        if self._show_legend.isChecked():
            ax.legend(
                fontsize=self._legend_font.value(),
                loc=self._legend_loc.currentText(),
                framealpha=self._legend_alpha.value(),
                frameon=self._legend_frame.isChecked())

        if show_vs and ax_vs:
            self._draw_vs(ax_vs)

        self._plot.refresh()

    def _draw_vs(self, ax):
        prof = self._profile
        if not prof:
            return
        depths, vs = [], []
        z = 0.0
        finite = [L for L in prof.layers if not L.is_halfspace]
        hs = [L for L in prof.layers if L.is_halfspace]
        for L in finite:
            depths.append(z); vs.append(L.vs)
            z += L.thickness
            depths.append(z); vs.append(L.vs)
        total = z
        if hs:
            hd = total * (self._hs_pct.value() / 100.0)
            depths.append(z); vs.append(hs[0].vs)
            z += hd
            depths.append(z); vs.append(hs[0].vs)

        ax.plot(vs, depths, color="teal", lw=1.5)
        ax.axhline(total, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=7)
        ax.set_title(f"{len(finite)}L", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.yaxis.tick_right()
        ax.grid(True, alpha=0.2)

        try:
            from ...core.vs_average import vs_average_from_profile
            result = vs_average_from_profile(prof, target_depth=30.0)
            ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.5)
            ax.annotate(f"Vs30={result.vs_avg:.0f}",
                        xy=(max(vs) * 0.5, 30.0), xytext=(0, -8),
                        textcoords="offset points", fontsize=6,
                        color="blue", fontweight="bold")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    #  PEAK PICKING
    # ══════════════════════════════════════════════════════════════
    def _toggle_f0(self, on):
        self._pick_f0 = on
        if on:
            self._btn_sec.setChecked(False)

    def _toggle_sec(self, on):
        self._pick_sec = on
        if on:
            self._btn_f0.setChecked(False)

    def _undo_sec(self):
        if self._secondary:
            self._secondary.pop()
            self._redraw()
            self._update_label()

    def _clear_all_peaks(self):
        self._f0 = None
        self._secondary = []
        self._redraw()
        self._update_label()

    def _on_click(self, event):
        if event.inaxes is None or self._freqs is None:
            return
        if not (self._pick_f0 or self._pick_sec):
            return
        if self._plot.toolbar.mode:
            return

        cx = event.xdata
        amp_interp = float(np.interp(cx, self._freqs, self._amps))
        idx = int(np.argmin(np.abs(self._freqs - cx)))
        f = float(cx)
        a = amp_interp

        if self._pick_f0:
            self._f0 = (f, a, idx)
            self._btn_f0.setChecked(False)
        elif self._pick_sec:
            self._secondary.append((f, a, idx))

        self._redraw()
        self._update_label()

    def _update_label(self):
        parts = []
        if self._f0:
            f, a, _ = self._f0
            parts.append(f"f0 = {f:.3f} Hz ({a:.2f})")
        for i, (sf, sa, _) in enumerate(self._secondary):
            parts.append(f"Sec.{i+1} = {sf:.3f} Hz ({sa:.2f})")
        self._sel_label.setText("  |  ".join(parts))

    # ══════════════════════════════════════════════════════════════
    #  SAVE
    # ══════════════════════════════════════════════════════════════
    def _get_save_dir(self, ask=True):
        d = self.get_output_dir()
        if d:
            return d
        if ask:
            p = QFileDialog.getExistingDirectory(self, "Save Results To")
            if p:
                return Path(p)
        return None

    def _save_csv(self):
        if self._freqs is None:
            return
        out = self._get_save_dir()
        if not out:
            return
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "hv_curve.csv",
                   np.column_stack([self._freqs, self._amps]),
                   delimiter=",", header="frequency,amplitude", comments="")
        self._save_label.setText(f"CSV → {out.name}/")
        if self._mw:
            self._mw.log(f"HV curve CSV saved to {out}")

    def _save_all(self):
        """Save all outputs: HV figure, Vs figure, CSV, peak info, Vs30."""
        if self._freqs is None:
            return
        out = self._get_save_dir()
        if not out:
            return
        out.mkdir(parents=True, exist_ok=True)
        dpi = self._dpi.value()
        fmt = self._export_fmt.currentText().lower()
        fig_key = self._fig_size.currentText()
        figsize = FIGURE_SIZES.get(fig_key, (12, 8))
        saved = []

        # 1. CSV
        np.savetxt(out / "hv_curve.csv",
                   np.column_stack([self._freqs, self._amps]),
                   delimiter=",", header="frequency,amplitude", comments="")
        saved.append("CSV")

        # 2. Peak info
        if self._f0:
            with open(out / "peak_info.txt", "w") as fh:
                fh.write(f"f0_Frequency_Hz,{self._f0[0]:.6f}\n")
                fh.write(f"f0_Amplitude,{self._f0[1]:.6f}\n")
                fh.write(f"f0_Index,{self._f0[2]}\n")
                for j, s in enumerate(self._secondary):
                    fh.write(f"Secondary_{j+1}_Frequency_Hz,{s[0]:.6f}\n")
                    fh.write(f"Secondary_{j+1}_Amplitude,{s[1]:.6f}\n")
                    fh.write(f"Secondary_{j+1}_Index,{s[2]}\n")
            saved.append("peaks")

        # 3. HV figure
        self._save_hv_figure(out, figsize, dpi, fmt)
        saved.append("HV fig")

        # 4. Combined HV + Vs
        if self._profile:
            self._save_vs_outputs(out, dpi, fmt)
            saved.append("Vs")
            self._save_combined_figure(out, figsize, dpi, fmt)
            saved.append("combined")

        self._save_label.setText(f"Saved: {', '.join(saved)} → {out.name}/")
        if self._mw:
            self._mw.log(f"All results saved to {out}")

    def _save_hv_figure(self, out_dir, figsize, dpi, fmt):
        try:
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt

            color = CURVE_COLORS.get(
                self._line_color.currentText(), "royalblue")

            fig = Figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.plot(self._freqs, self._amps,
                    color=color, lw=2.0, label="H/V Ratio")
            ax.set_xscale("log")
            ax.set_xlabel(self._xlabel_edit.text(), fontsize=11)
            ax.set_ylabel(self._ylabel_edit.text(), fontsize=11)
            ax.set_title(
                self._title_edit.text(), fontsize=13, fontweight="bold")
            ax.grid(True, alpha=self._grid_alpha.value(), which="both")

            f0_shape = MARKER_SHAPES.get(
                self._f0_marker.currentText(), "*")
            if self._f0:
                f, a, _ = self._f0
                ax.plot(f, a, f0_shape, color="firebrick", ms=16,
                        zorder=10, markeredgecolor="darkred",
                        markeredgewidth=0.8,
                        label=f"f0 = {f:.2f} Hz")
                ax.axvline(f, color="firebrick", ls="--",
                           lw=0.9, alpha=0.4)
            for j, (sf, sa, _) in enumerate(self._secondary):
                c = SEC_COLORS[j % len(SEC_COLORS)]
                sec_shape = MARKER_SHAPES.get(
                    self._sec_marker.currentText(), "o")
                ax.plot(sf, sa, sec_shape, color=c, ms=13, zorder=9,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=f"Sec.{j+1} ({sf:.2f} Hz)")
                ax.axvline(sf, color=c, ls=":", lw=0.8, alpha=0.4)

            ax.legend(
                fontsize=self._legend_font.value(),
                loc=self._legend_loc.currentText(),
                framealpha=self._legend_alpha.value())
            fig.tight_layout()
            fig.savefig(out_dir / f"hv_forward_curve.{fmt}", dpi=dpi)
            if fmt != "pdf":
                fig.savefig(out_dir / "hv_forward_curve.pdf")
            plt.close(fig)
        except Exception as e:
            if self._mw:
                self._mw.log(f"HV figure save error: {e}")

    def _save_vs_outputs(self, out_dir, dpi, fmt):
        prof = self._profile
        if not prof:
            return
        try:
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt

            finite = [L for L in prof.layers if not L.is_halfspace]
            hs = [L for L in prof.layers if L.is_halfspace]

            # CSV
            with open(out_dir / "vs_profile.csv", "w") as fh:
                fh.write("depth_top_m,depth_bot_m,thickness_m,"
                         "vs_m_s,vp_m_s,density_kg_m3\n")
                z = 0.0
                for L in finite:
                    fh.write(
                        f"{z:.2f},{z + L.thickness:.2f},{L.thickness:.2f},"
                        f"{L.vs:.2f},{L.vp:.2f},{L.density:.2f}\n")
                    z += L.thickness
                if hs:
                    L = hs[0]
                    fh.write(f"{z:.2f},inf,inf,"
                             f"{L.vs:.2f},{L.vp:.2f},{L.density:.2f}\n")

            # Figure
            fig = Figure(figsize=(5, 7))
            ax = fig.add_subplot(111)
            depths, vs = [], []
            z = 0.0
            for L in finite:
                depths.append(z); vs.append(L.vs)
                z += L.thickness
                depths.append(z); vs.append(L.vs)
            total = z
            if hs:
                hd = total * 0.25
                depths.append(z); vs.append(hs[0].vs)
                z += hd
                depths.append(z); vs.append(hs[0].vs)

            ax.plot(vs, depths, color="teal", lw=1.8)
            ax.fill_betweenx(depths, 0, vs, alpha=0.1, color="teal")
            ax.invert_yaxis()
            ax.set_xlabel("Vs (m/s)", fontsize=10)
            ax.set_ylabel("Depth (m)", fontsize=10)
            ax.set_title("Vs Profile", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

            try:
                from ...core.vs_average import vs_average_from_profile
                res30 = vs_average_from_profile(prof, target_depth=30.0)
                ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.6)
                vs_max = max(vs) if vs else 500
                ax.annotate(
                    f"Vs30 = {res30.vs_avg:.0f} m/s",
                    xy=(vs_max * 0.5, 30.0),
                    xytext=(0, -10), textcoords="offset points",
                    fontsize=8, color="blue", fontweight="bold")
                with open(out_dir / "vs30_info.txt", "w") as fh:
                    fh.write(f"Vs30_m_per_s,{res30.vs_avg:.2f}\n")
                    fh.write(f"Target_Depth_m,{res30.target_depth:.1f}\n")
                    fh.write(f"Actual_Depth_m,{res30.actual_depth:.1f}\n")
                    fh.write(f"Extrapolated,{res30.extrapolated}\n")
            except Exception:
                pass

            fig.tight_layout()
            fig.savefig(out_dir / f"vs_profile.{fmt}", dpi=dpi)
            if fmt != "pdf":
                fig.savefig(out_dir / "vs_profile.pdf")
            plt.close(fig)
        except Exception as e:
            if self._mw:
                self._mw.log(f"Vs output save error: {e}")

    def _save_combined_figure(self, out_dir, figsize, dpi, fmt):
        """Save a combined HV + Vs side-by-side figure."""
        try:
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gs

            fig = Figure(figsize=(figsize[0], figsize[1]))
            spec = gs.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.15,
                               figure=fig)
            ax_hv = fig.add_subplot(spec[0])
            ax_vs = fig.add_subplot(spec[1])

            color = CURVE_COLORS.get(
                self._line_color.currentText(), "royalblue")
            ax_hv.plot(self._freqs, self._amps,
                       color=color, lw=2.0, label="H/V Ratio")
            ax_hv.set_xscale("log")
            ax_hv.set_xlabel(self._xlabel_edit.text(), fontsize=10)
            ax_hv.set_ylabel(self._ylabel_edit.text(), fontsize=10)
            ax_hv.set_title(
                self._title_edit.text(), fontsize=12, fontweight="bold")
            ax_hv.grid(True, alpha=0.3, which="both")

            if self._f0:
                f, a, _ = self._f0
                mk = MARKER_SHAPES.get(
                    self._f0_marker.currentText(), "*")
                ax_hv.plot(f, a, mk, color="firebrick", ms=14, zorder=10,
                           markeredgecolor="darkred", markeredgewidth=0.8,
                           label=f"f0 = {f:.2f} Hz")
                ax_hv.axvline(f, color="firebrick", ls="--",
                              lw=0.9, alpha=0.4)
            for j, (sf, sa, _) in enumerate(self._secondary):
                c = SEC_COLORS[j % len(SEC_COLORS)]
                sm = MARKER_SHAPES.get(
                    self._sec_marker.currentText(), "o")
                ax_hv.plot(sf, sa, sm, color=c, ms=11, zorder=9,
                           markeredgecolor="black", markeredgewidth=0.5,
                           label=f"Sec.{j+1} ({sf:.2f} Hz)")
            ax_hv.legend(fontsize=8, framealpha=0.9)

            self._draw_vs(ax_vs)

            fig.suptitle("Forward Model Results", fontsize=14,
                         fontweight="bold", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(out_dir / f"hv_combined.{fmt}", dpi=dpi)
            if fmt != "pdf":
                fig.savefig(out_dir / "hv_combined.pdf")
            plt.close(fig)
        except Exception as e:
            if self._mw:
                self._mw.log(f"Combined figure save error: {e}")

    def _open_save_options(self):
        """Open a dialog to choose what to save and where."""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QGroupBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Save Options")
        dlg.setMinimumWidth(350)
        lay = QVBoxLayout(dlg)

        # Output directory
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:"))
        dir_edit = QLineEdit()
        od = self.get_output_dir()
        if od:
            dir_edit.setText(str(od))
        dir_row.addWidget(dir_edit, 1)
        btn_br = QPushButton("...")
        btn_br.setFixedWidth(30)
        btn_br.clicked.connect(
            lambda: dir_edit.setText(
                QFileDialog.getExistingDirectory(dlg, "Output") or
                dir_edit.text()))
        dir_row.addWidget(btn_br)
        lay.addLayout(dir_row)

        # Figures group
        fig_grp = QGroupBox("Figures to Save")
        fig_lay = QVBoxLayout()
        chk_hv = QCheckBox("HV Forward Curve (publication)")
        chk_hv.setChecked(True)
        chk_vs = QCheckBox("Vs Profile Figure")
        chk_vs.setChecked(True)
        chk_combined = QCheckBox("Combined HV + Vs (side by side)")
        chk_combined.setChecked(True)
        for w in [chk_hv, chk_vs, chk_combined]:
            fig_lay.addWidget(w)
        fig_grp.setLayout(fig_lay)
        lay.addWidget(fig_grp)

        # Data group
        data_grp = QGroupBox("Data to Save")
        data_lay = QVBoxLayout()
        chk_csv = QCheckBox("HV Curve CSV")
        chk_csv.setChecked(True)
        chk_peaks = QCheckBox("Peak Info (TXT)")
        chk_peaks.setChecked(True)
        chk_vs_csv = QCheckBox("Vs Profile CSV")
        chk_vs_csv.setChecked(True)
        chk_vs30 = QCheckBox("Vs30 Info")
        chk_vs30.setChecked(True)
        for w in [chk_csv, chk_peaks, chk_vs_csv, chk_vs30]:
            data_lay.addWidget(w)
        data_grp.setLayout(data_lay)
        lay.addWidget(data_grp)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        d = dir_edit.text().strip()
        if not d:
            return
        out = Path(d)
        out.mkdir(parents=True, exist_ok=True)
        dpi = self._dpi.value()
        fmt = self._export_fmt.currentText().lower()
        fig_key = self._fig_size.currentText()
        figsize = FIGURE_SIZES.get(fig_key, (12, 8))
        saved = []

        if chk_csv.isChecked() and self._freqs is not None:
            np.savetxt(out / "hv_curve.csv",
                       np.column_stack([self._freqs, self._amps]),
                       delimiter=",", header="frequency,amplitude",
                       comments="")
            saved.append("CSV")

        if chk_peaks.isChecked() and self._f0:
            with open(out / "peak_info.txt", "w") as fh:
                fh.write(f"f0_Frequency_Hz,{self._f0[0]:.6f}\n")
                fh.write(f"f0_Amplitude,{self._f0[1]:.6f}\n")
                for j, s in enumerate(self._secondary):
                    fh.write(
                        f"Secondary_{j+1}_Frequency_Hz,{s[0]:.6f}\n")
                    fh.write(
                        f"Secondary_{j+1}_Amplitude,{s[1]:.6f}\n")
            saved.append("peaks")

        if chk_hv.isChecked() and self._freqs is not None:
            self._save_hv_figure(out, figsize, dpi, fmt)
            saved.append("HV fig")

        if self._profile:
            if chk_vs.isChecked() or chk_vs_csv.isChecked() or chk_vs30.isChecked():
                self._save_vs_outputs(out, dpi, fmt)
                saved.append("Vs")
            if chk_combined.isChecked():
                self._save_combined_figure(out, figsize, dpi, fmt)
                saved.append("combined")

        self._save_label.setText(f"Saved: {', '.join(saved)} → {out.name}/")
        if self._mw:
            self._mw.log(f"Custom save to {out}: {', '.join(saved)}")
