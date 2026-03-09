"""All Profiles View — enhanced HV overlay with median, Vs panel, plot settings.

Canvas tab for Forward Multiple mode.  Shows all computed HV curves overlaid
with configurable plot settings, optional median + ±1σ bands, peak annotations,
and a togglable Vs profile panel on the right side.

v7: Full collapsible settings panel, fixed palettes, singular matrix fix,
    legend-based figure export matching old package quality, median peak picking.
"""
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox,
    QSizePolicy, QFileDialog, QLineEdit,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.collapsible_group import CollapsibleGroupBox
from ..widgets.style_constants import BUTTON_PRIMARY, BUTTON_SUCCESS, EMOJI

# Valid palettes: custom + valid matplotlib colormaps
PALETTES = [
    "Classic", "Bold", "Earth", "Nordic", "Sunset",
    "Green", "Blue", "Orange", "Red", "Purple",
    "tab10", "tab20", "Set1", "Set2", "Set3",
    "Pastel1", "Paired", "Dark2", "Accent",
]

_BUILTIN_COLORS = {
    "classic": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"],
    "bold": ["#e6194b", "#3cb44b", "#ffe119", "#4363d8",
             "#f58231", "#911eb4", "#42d4f4", "#f032e6"],
    "earth": ["#8B4513", "#228B22", "#DAA520", "#CD853F",
              "#2E8B57", "#D2691E", "#6B8E23", "#A0522D"],
    "nordic": ["#2E4057", "#048A81", "#54C6EB", "#8EE3EF",
               "#F25C54", "#F4845F", "#F7B267", "#7D82B8"],
    "sunset": ["#FF6B6B", "#FFA07A", "#FFD700", "#FF8C00",
               "#FF4500", "#DC143C", "#FF69B4", "#FF1493"],
    "green": ["#2ca02c", "#228B22", "#006400", "#32CD32",
              "#66CDAA", "#3CB371", "#00FA9A", "#90EE90"],
    "blue": ["#1f77b4", "#4169E1", "#000080", "#4682B4",
             "#6495ED", "#00BFFF", "#87CEEB", "#1E90FF"],
    "orange": ["#ff7f0e", "#FF8C00", "#FFA500", "#FF6347",
               "#E9967A", "#FFD700", "#F4A460", "#FF4500"],
    "red": ["#d62728", "#DC143C", "#B22222", "#FF0000",
            "#CD5C5C", "#FF6B6B", "#8B0000", "#E74C3C"],
    "purple": ["#9467bd", "#800080", "#8B008B", "#9932CC",
               "#BA55D3", "#DDA0DD", "#7B68EE", "#6A0DAD"],
}

MARKER_SHAPES = {"★": "*", "◆": "D", "●": "o", "▲": "^",
                 "▼": "v", "■": "s", "+": "P", "✕": "X"}

Y_LIMIT_METHODS = ["Auto", "95th Percentile", "Mean + 3σ", "Mean + 2×IQR"]

FIGURE_SIZES = {
    "Standard (10×7)": (10, 7),
    "Large (14×10)": (14, 10),
    "Publication (12×8)": (12, 8),
    "Wide (16×6)": (16, 6),
}


class AllProfilesView(QWidget):
    """Canvas view showing all HV profiles overlaid with plot controls."""

    results_loaded = pyqtSignal()

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._results = []
        self._peak_data = {}
        self._median_peaks = {"f0": None, "secondary": []}
        self._picking_mode = None  # None, "f0", or "secondary"
        self._drag_start = None          # (freq, amp) during drag
        self._drag_temp_marker = None    # temporary Line2D marker
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(2, 2, 2, 2)
        main.setSpacing(2)

        # ── Main split: HV canvas + Vs canvas ────────────────
        self._splitter = QSplitter(Qt.Horizontal)

        self._hv_plot = MatplotlibWidget(figsize=(12, 6))
        self._hv_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._hv_plot.canvas.mpl_connect("button_press_event", self._on_press)
        self._hv_plot.canvas.mpl_connect("button_release_event", self._on_release)
        self._splitter.addWidget(self._hv_plot)

        self._vs_panel = QWidget()
        vs_lay = QVBoxLayout(self._vs_panel)
        vs_lay.setContentsMargins(2, 2, 2, 2)
        self._vs_plot = MatplotlibWidget(figsize=(4, 6))
        self._vs_plot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        vs_lay.addWidget(self._vs_plot, 1)

        vs_opts = QHBoxLayout()
        self._chk_vs_median = QCheckBox("Median Vs")
        self._chk_vs_median.toggled.connect(self._redraw_vs)
        vs_opts.addWidget(self._chk_vs_median)
        self._chk_vs30 = QCheckBox("Vs30")
        self._chk_vs30.setChecked(True)
        self._chk_vs30.toggled.connect(self._redraw_vs)
        vs_opts.addWidget(self._chk_vs30)
        vs_lay.addLayout(vs_opts)

        self._vs_panel.setVisible(False)
        self._splitter.addWidget(self._vs_panel)
        self._splitter.setSizes([800, 250])
        main.addWidget(self._splitter, 1)

        # ── Collapsible Settings Panel ────────────────────────
        _SS = ("QGroupBox{font-size:9px; font-weight:bold; margin-top:2px; "
               "padding-top:10px; border:1px solid #ccc; border-radius:3px;}"
               "QGroupBox::title{subcontrol-origin:margin; left:6px;}")

        self._settings_group = CollapsibleGroupBox(
            f"{EMOJI.get('settings', '⚙')} Plot Settings", collapsed=True)
        settings_lay = QVBoxLayout()
        settings_lay.setSpacing(2)
        settings_lay.setContentsMargins(2, 2, 2, 2)

        # -- Toggles (always visible, not collapsible) --
        row_toggle = QHBoxLayout()
        row_toggle.setSpacing(6)
        self._chk_median = QCheckBox("Median")
        self._chk_median.setChecked(True)
        self._chk_median.toggled.connect(self._redraw)
        row_toggle.addWidget(self._chk_median)

        self._chk_sigma = QCheckBox("±1σ")
        self._chk_sigma.setChecked(True)
        self._chk_sigma.toggled.connect(self._redraw)
        row_toggle.addWidget(self._chk_sigma)

        self._chk_primary = QCheckBox("f0 Peaks")
        self._chk_primary.setChecked(True)
        self._chk_primary.toggled.connect(self._redraw)
        row_toggle.addWidget(self._chk_primary)

        self._chk_secondary = QCheckBox("Sec. Peaks")
        self._chk_secondary.setChecked(True)
        self._chk_secondary.toggled.connect(self._redraw)
        row_toggle.addWidget(self._chk_secondary)

        self._chk_annotations = QCheckBox("Annotations")
        self._chk_annotations.setChecked(True)
        self._chk_annotations.toggled.connect(self._redraw)
        row_toggle.addWidget(self._chk_annotations)

        self._chk_vs = QCheckBox("Vs Panel")
        self._chk_vs.toggled.connect(self._toggle_vs)
        row_toggle.addWidget(self._chk_vs)

        row_toggle.addStretch()
        settings_lay.addLayout(row_toggle)

        # -- Curves (collapsible sub-group) --
        curves_grp = CollapsibleGroupBox("Curves", collapsed=True)
        cg_lay = QHBoxLayout()
        cg_lay.setSpacing(4)
        cg_lay.setContentsMargins(2, 2, 2, 2)

        cg_lay.addWidget(QLabel("Palette:"))
        self._palette = QComboBox()
        self._palette.addItems(PALETTES)
        self._palette.setMaximumWidth(120)
        self._palette.currentIndexChanged.connect(self._redraw)
        cg_lay.addWidget(self._palette)

        cg_lay.addWidget(QLabel("α:"))
        self._alpha = QDoubleSpinBox()
        self._alpha.setRange(0.05, 1.0); self._alpha.setValue(0.5)
        self._alpha.setSingleStep(0.05); self._alpha.setMaximumWidth(60)
        self._alpha.valueChanged.connect(self._redraw)
        cg_lay.addWidget(self._alpha)

        cg_lay.addWidget(QLabel("LW:"))
        self._lw = QDoubleSpinBox()
        self._lw.setRange(0.3, 5.0); self._lw.setValue(1.2)
        self._lw.setMaximumWidth(60)
        self._lw.valueChanged.connect(self._redraw)
        cg_lay.addWidget(self._lw)

        cg_lay.addWidget(QLabel("Med LW:"))
        self._med_lw = QDoubleSpinBox()
        self._med_lw.setRange(0.5, 8.0); self._med_lw.setValue(3.0)
        self._med_lw.setMaximumWidth(60)
        self._med_lw.valueChanged.connect(self._redraw)
        cg_lay.addWidget(self._med_lw)

        cg_lay.addWidget(QLabel("±1σ α:"))
        self._sigma_alpha = QDoubleSpinBox()
        self._sigma_alpha.setRange(0.05, 0.6); self._sigma_alpha.setValue(0.15)
        self._sigma_alpha.setSingleStep(0.05); self._sigma_alpha.setMaximumWidth(60)
        self._sigma_alpha.valueChanged.connect(self._redraw)
        cg_lay.addWidget(self._sigma_alpha)

        cg_lay.addStretch()
        curves_grp.setContentLayout(cg_lay)
        settings_lay.addWidget(curves_grp)

        # -- Primary Peak (collapsible sub-group) --
        f0_grp = CollapsibleGroupBox("Primary Peak (f0)", collapsed=True)
        f0_lay = QHBoxLayout()
        f0_lay.setSpacing(4)
        f0_lay.setContentsMargins(2, 2, 2, 2)

        f0_lay.addWidget(QLabel("Shape:"))
        self._f0_marker = QComboBox()
        self._f0_marker.addItems(list(MARKER_SHAPES.keys()))
        self._f0_marker.setMaximumWidth(70)
        self._f0_marker.currentIndexChanged.connect(self._redraw)
        f0_lay.addWidget(self._f0_marker)

        f0_lay.addWidget(QLabel("Size:"))
        self._f0_size = QSpinBox()
        self._f0_size.setRange(4, 30); self._f0_size.setValue(14)
        self._f0_size.setMaximumWidth(55)
        self._f0_size.valueChanged.connect(self._redraw)
        f0_lay.addWidget(self._f0_size)

        f0_lay.addWidget(QLabel("Ann. Font:"))
        self._ann_font = QSpinBox()
        self._ann_font.setRange(6, 20); self._ann_font.setValue(8)
        self._ann_font.setMaximumWidth(55)
        self._ann_font.valueChanged.connect(self._redraw)
        f0_lay.addWidget(self._ann_font)

        f0_lay.addStretch()
        f0_grp.setContentLayout(f0_lay)
        settings_lay.addWidget(f0_grp)

        # -- Secondary Peak (collapsible sub-group) --
        sec_grp = CollapsibleGroupBox("Secondary Peaks", collapsed=True)
        sec_lay = QHBoxLayout()
        sec_lay.setSpacing(4)
        sec_lay.setContentsMargins(2, 2, 2, 2)

        sec_lay.addWidget(QLabel("Shape:"))
        self._sec_marker = QComboBox()
        self._sec_marker.addItems(list(MARKER_SHAPES.keys()))
        self._sec_marker.setCurrentIndex(1)  # ◆
        self._sec_marker.setMaximumWidth(70)
        self._sec_marker.currentIndexChanged.connect(self._redraw)
        sec_lay.addWidget(self._sec_marker)

        sec_lay.addWidget(QLabel("Size:"))
        self._sec_size = QSpinBox()
        self._sec_size.setRange(4, 30); self._sec_size.setValue(10)
        self._sec_size.setMaximumWidth(55)
        self._sec_size.valueChanged.connect(self._redraw)
        sec_lay.addWidget(self._sec_size)

        sec_lay.addStretch()
        sec_grp.setContentLayout(sec_lay)
        settings_lay.addWidget(sec_grp)

        # -- Axes & Grid (collapsible sub-group) --
        axes_grp = CollapsibleGroupBox("Axes && Grid", collapsed=True)
        ag_lay = QHBoxLayout()
        ag_lay.setSpacing(4)
        ag_lay.setContentsMargins(2, 2, 2, 2)

        self._chk_ylim_auto = QCheckBox("Y Auto")
        self._chk_ylim_auto.setChecked(True)
        self._chk_ylim_auto.toggled.connect(self._redraw)
        ag_lay.addWidget(self._chk_ylim_auto)

        ag_lay.addWidget(QLabel("Method:"))
        self._ylim_method = QComboBox()
        self._ylim_method.addItems(Y_LIMIT_METHODS)
        self._ylim_method.setMaximumWidth(120)
        self._ylim_method.currentIndexChanged.connect(self._redraw)
        ag_lay.addWidget(self._ylim_method)

        self._chk_grid = QCheckBox("Grid")
        self._chk_grid.setChecked(True)
        self._chk_grid.toggled.connect(self._redraw)
        ag_lay.addWidget(self._chk_grid)

        ag_lay.addWidget(QLabel("α:"))
        self._grid_alpha = QDoubleSpinBox()
        self._grid_alpha.setRange(0.1, 1.0); self._grid_alpha.setValue(0.3)
        self._grid_alpha.setSingleStep(0.1); self._grid_alpha.setMaximumWidth(60)
        self._grid_alpha.valueChanged.connect(self._redraw)
        ag_lay.addWidget(self._grid_alpha)

        ag_lay.addStretch()
        axes_grp.setContentLayout(ag_lay)
        settings_lay.addWidget(axes_grp)

        # -- Export (collapsible sub-group) --
        export_grp = CollapsibleGroupBox("Export", collapsed=True)
        eg_lay = QHBoxLayout()
        eg_lay.setSpacing(4)
        eg_lay.setContentsMargins(2, 2, 2, 2)

        eg_lay.addWidget(QLabel("DPI:"))
        self._dpi = QSpinBox()
        self._dpi.setRange(72, 600); self._dpi.setValue(300)
        self._dpi.setMaximumWidth(65)
        eg_lay.addWidget(self._dpi)

        eg_lay.addWidget(QLabel("Format:"))
        self._export_fmt = QComboBox()
        self._export_fmt.addItems(["PNG", "PDF", "SVG"])
        self._export_fmt.setMaximumWidth(70)
        eg_lay.addWidget(self._export_fmt)

        eg_lay.addWidget(QLabel("Size:"))
        self._fig_size = QComboBox()
        self._fig_size.addItems(list(FIGURE_SIZES.keys()))
        self._fig_size.setMaximumWidth(140)
        eg_lay.addWidget(self._fig_size)

        eg_lay.addStretch()
        export_grp.setContentLayout(eg_lay)
        settings_lay.addWidget(export_grp)

        # -- Labels (collapsible sub-group) --
        label_grp = CollapsibleGroupBox("Labels", collapsed=True)
        lg_lay = QVBoxLayout()
        lg_lay.setSpacing(2)
        lg_lay.setContentsMargins(2, 2, 2, 2)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self._title_edit = QLineEdit("All Profiles")
        self._title_edit.editingFinished.connect(self._redraw)
        title_row.addWidget(self._title_edit, 1)
        lg_lay.addLayout(title_row)

        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("X:"))
        self._xlabel_edit = QLineEdit("Frequency (Hz)")
        self._xlabel_edit.editingFinished.connect(self._redraw)
        axis_row.addWidget(self._xlabel_edit, 1)
        axis_row.addWidget(QLabel("Y:"))
        self._ylabel_edit = QLineEdit("H/V Amplitude Ratio")
        self._ylabel_edit.editingFinished.connect(self._redraw)
        axis_row.addWidget(self._ylabel_edit, 1)
        lg_lay.addLayout(axis_row)

        label_grp.setContentLayout(lg_lay)
        settings_lay.addWidget(label_grp)

        self._settings_group.setContentLayout(settings_lay)
        main.addWidget(self._settings_group)

        # ── Bottom bar: Peak picking + Save / Load ────────────
        bot = QHBoxLayout()
        bot.setSpacing(4)

        # -- f0 selection button --
        self._btn_pick_f0 = QPushButton("🎯 Select f0")
        self._btn_pick_f0.setCheckable(True)
        self._btn_pick_f0.toggled.connect(self._toggle_pick_f0)
        self._btn_pick_f0.setEnabled(False)
        self._btn_pick_f0.setToolTip("Click on median curve to set primary peak")
        bot.addWidget(self._btn_pick_f0)

        # -- Secondary peak button --
        self._btn_pick_sec = QPushButton("🔶 Select Secondary")
        self._btn_pick_sec.setCheckable(True)
        self._btn_pick_sec.toggled.connect(self._toggle_pick_sec)
        self._btn_pick_sec.setEnabled(False)
        self._btn_pick_sec.setToolTip("Click on median curve to add secondary peaks")
        bot.addWidget(self._btn_pick_sec)

        # -- Undo secondary --
        self._btn_undo_sec = QPushButton("↩ Undo Sec.")
        self._btn_undo_sec.clicked.connect(self._undo_last_secondary)
        self._btn_undo_sec.setToolTip("Remove last added secondary peak")
        bot.addWidget(self._btn_undo_sec)

        # -- Clear all peaks --
        btn_clear_med = QPushButton("✕ Clear Peaks")
        btn_clear_med.clicked.connect(self._clear_median_peaks)
        bot.addWidget(btn_clear_med)

        bot.addStretch()

        self._btn_save = QPushButton(f"{EMOJI.get('report', '💾')} Save Results")
        self._btn_save.setStyleSheet(BUTTON_SUCCESS)
        self._btn_save.clicked.connect(self._save_results)
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip("Save only All Profiles outputs")
        bot.addWidget(self._btn_save)

        self._btn_save_all = QPushButton("⚙ Save All")
        self._btn_save_all.clicked.connect(self._save_all_results)
        self._btn_save_all.setEnabled(False)
        self._btn_save_all.setToolTip("Full re-save: per-profile + all profiles")
        bot.addWidget(self._btn_save_all)

        self._btn_load = QPushButton(f"{EMOJI.get('file', '📂')} Load Results")
        self._btn_load.clicked.connect(self._load_results)
        bot.addWidget(self._btn_load)

        main.addLayout(bot)

    # ── Public API ─────────────────────────────────────────────

    def set_results(self, results, peak_data=None):
        """Set computed results and optional peak data, then redraw."""
        self._results = [r for r in results if r.computed]
        self._peak_data = peak_data or {}
        for r in self._results:
            if r.name not in self._peak_data:
                self._peak_data[r.name] = {
                    "f0": r.f0,
                    "secondary": list(r.secondary_peaks or []),
                }
        self._btn_save.setEnabled(bool(self._results))
        self._btn_save_all.setEnabled(bool(self._results))
        self._btn_pick_f0.setEnabled(bool(self._results))
        self._btn_pick_sec.setEnabled(bool(self._results))
        self._redraw()
        if self._chk_vs.isChecked():
            self._redraw_vs()

    def update_peak_data(self, peak_data):
        """Update peak selections (e.g. from wizard) and redraw."""
        self._peak_data.update(peak_data)
        self._redraw()

    # ── Median peak picking ────────────────────────────────────

    def _toggle_pick_f0(self, on):
        """Enable/disable primary peak picking on median curve."""
        if on:
            self._btn_pick_sec.setChecked(False)
        self._picking_mode = "f0" if on else None
        self._btn_pick_f0.setStyleSheet(
            "background-color: #FFB3B3;" if on else "")

    def _toggle_pick_sec(self, on):
        """Enable/disable secondary peak picking on median curve."""
        if on:
            self._btn_pick_f0.setChecked(False)
        self._picking_mode = "secondary" if on else None
        self._btn_pick_sec.setStyleSheet(
            "background-color: #FFDAB3;" if on else "")

    def _undo_last_secondary(self):
        """Remove the most recently added secondary peak."""
        if self._median_peaks["secondary"]:
            removed = self._median_peaks["secondary"].pop()
            if hasattr(self, '_median_ann_positions'):
                self._median_ann_positions.pop(f"{removed[0]:.6f}", None)
            self._redraw()

    def _clear_median_peaks(self):
        self._median_peaks = {"f0": None, "secondary": []}
        self._drag_start = None
        if self._drag_temp_marker is not None:
            try:
                self._drag_temp_marker.remove()
            except Exception:
                pass
            self._drag_temp_marker = None
        self._redraw()

    def _on_press(self, event):
        """Mouse press: snap to median curve, show temporary marker."""
        if self._picking_mode is None or event.inaxes is None:
            return
        if not self._results:
            return

        # Right-click: remove nearest median peak
        if event.button == 3:
            self._remove_nearest_median_peak(event.xdata)
            return

        if event.button != 1:
            return

        med_f, med_a, _ = self._compute_stats()
        if med_f is None:
            return

        cx = event.xdata
        if cx is None:
            return

        amp = float(np.interp(cx, med_f, med_a))
        idx = int(np.argmin(np.abs(med_f - cx)))
        self._drag_start = (float(med_f[idx]), float(med_a[idx]), idx)

        # Color depends on pick mode
        color = "red" if self._picking_mode == "f0" else "green"
        ax = self._hv_plot.figure.axes[0] if self._hv_plot.figure.axes else None
        if ax:
            self._drag_temp_marker, = ax.plot(
                self._drag_start[0], self._drag_start[1], "*",
                color=color, ms=14, markeredgecolor="black",
                markeredgewidth=0.8, zorder=20)
            self._hv_plot.canvas.draw_idle()

    def _on_release(self, event):
        """Mouse release: place annotation at release point with arrow to peak."""
        if self._drag_start is None or event.button != 1:
            return

        freq, amp, idx = self._drag_start
        self._drag_start = None

        # Remove temp marker
        if self._drag_temp_marker is not None:
            try:
                self._drag_temp_marker.remove()
            except Exception:
                pass
            self._drag_temp_marker = None

        peak = (freq, amp, idx)
        if self._picking_mode == "f0":
            self._median_peaks["f0"] = peak
        elif self._picking_mode == "secondary":
            self._median_peaks["secondary"].append(peak)

        # Store annotation position for use in _redraw
        if not hasattr(self, '_median_ann_positions'):
            self._median_ann_positions = {}
        if event.inaxes is not None and event.xdata is not None:
            self._median_ann_positions[f"{freq:.6f}"] = (event.xdata, event.ydata)

        self._redraw()

    def _remove_nearest_median_peak(self, xdata):
        """Remove the nearest median peak to the clicked x position."""
        if xdata is None:
            return
        all_peaks = []
        f0 = self._median_peaks.get("f0")
        if f0:
            all_peaks.append(("f0", 0, f0))
        for i, sp in enumerate(self._median_peaks.get("secondary", [])):
            all_peaks.append(("sec", i, sp))
        if not all_peaks:
            return
        dists = [abs(xdata - p[2][0]) for p in all_peaks]
        nearest = all_peaks[int(np.argmin(dists))]
        if nearest[0] == "f0":
            self._median_peaks["f0"] = None
        else:
            self._median_peaks["secondary"].pop(nearest[1])
        if hasattr(self, '_median_ann_positions'):
            key = f"{nearest[2][0]:.6f}"
            self._median_ann_positions.pop(key, None)
        self._redraw()

    # ── Drawing ────────────────────────────────────────────────

    def _redraw(self, *_args):
        if not self._results:
            return
        fig = self._hv_plot.figure
        fig.clear()
        ax = fig.add_subplot(111)

        n = len(self._results)
        colors = self._get_colors(n)
        alpha = self._alpha.value()
        lw = self._lw.value()
        f0_mk = MARKER_SHAPES.get(self._f0_marker.currentText(), "*")
        f0_ms = self._f0_size.value()
        sec_mk = MARKER_SHAPES.get(self._sec_marker.currentText(), "D")
        sec_ms = self._sec_size.value()
        ann_fs = self._ann_font.value()
        show_ann = self._chk_annotations.isChecked()

        for i, r in enumerate(self._results):
            c = colors[i % len(colors)]
            ax.plot(r.freqs, r.amps, color=c, lw=lw, alpha=alpha, label=r.name)

            pk = self._peak_data.get(r.name, {})
            f0 = pk.get("f0")
            if f0 and self._chk_primary.isChecked():
                ax.plot(f0[0], f0[1], f0_mk, color=c, ms=f0_ms, zorder=5,
                        markeredgecolor="black", markeredgewidth=0.3)
                if show_ann:
                    ax.annotate(f"{f0[0]:.3f}", xy=(f0[0], f0[1]),
                                xytext=(4, 6), textcoords="offset points",
                                fontsize=max(ann_fs - 2, 5), color=c,
                                fontweight="bold")

            if self._chk_secondary.isChecked():
                for s in pk.get("secondary", []):
                    ax.plot(s[0], s[1], sec_mk, color=c, ms=sec_ms,
                            zorder=5, alpha=0.7,
                            markeredgecolor="black", markeredgewidth=0.3)
                    if show_ann:
                        ax.annotate(f"{s[0]:.2f}", xy=(s[0], s[1]),
                                    xytext=(4, -8), textcoords="offset points",
                                    fontsize=max(ann_fs - 3, 4), color=c, alpha=0.8)

        # Median
        if self._chk_median.isChecked() and n >= 2:
            med_f, med_a, std = self._compute_stats()
            if med_f is not None:
                ax.plot(med_f, med_a, color="black", lw=self._med_lw.value(),
                        label="Median", zorder=10)

                # Median peaks
                mp = self._median_peaks
                f0m = mp.get("f0")
                if f0m is None:
                    # Auto: highest peak of median
                    idx = int(np.argmax(med_a))
                    f0m = (med_f[idx], med_a[idx], idx)

                ax.plot(f0m[0], f0m[1], "*", color="red", ms=14, zorder=11,
                        markeredgecolor="darkred", markeredgewidth=0.8)
                ax.axvline(f0m[0], color="red", ls="--", lw=0.8, alpha=0.4)
                if show_ann:
                    ann_key = f"{f0m[0]:.6f}"
                    ann_pos = getattr(self, '_median_ann_positions', {}).get(ann_key)
                    if ann_pos:
                        ax.annotate(
                            f"Median f0 = {f0m[0]:.3f} Hz",
                            xy=(f0m[0], f0m[1]), xycoords="data",
                            xytext=ann_pos, textcoords="data",
                            fontsize=ann_fs, color="red", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="red", alpha=0.9),
                            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
                    else:
                        ylim = ax.get_ylim()
                        y_range = ylim[1] - ylim[0]
                        y_off = -20 if f0m[1] > ylim[0] + 0.8 * y_range else 10
                        ax.annotate(
                            f"Median f0 = {f0m[0]:.3f} Hz",
                            xy=(f0m[0], f0m[1]), xytext=(10, y_off),
                            textcoords="offset points", fontsize=ann_fs,
                            color="red", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="red", alpha=0.9),
                            arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

                for j, sp in enumerate(mp.get("secondary", [])):
                    ax.plot(sp[0], sp[1], "*", color="green", ms=12, zorder=11,
                            markeredgecolor="darkgreen", markeredgewidth=0.6)
                    ax.axvline(sp[0], color="green", ls=":", lw=0.7, alpha=0.4)
                    if show_ann:
                        ann_key = f"{sp[0]:.6f}"
                        ann_pos = getattr(self, '_median_ann_positions', {}).get(ann_key)
                        if ann_pos:
                            ax.annotate(
                                f"Sec.{j+1} ({sp[0]:.2f} Hz)",
                                xy=(sp[0], sp[1]), xycoords="data",
                                xytext=ann_pos, textcoords="data",
                                fontsize=max(ann_fs - 1, 5), color="green",
                                arrowprops=dict(arrowstyle="->", color="green", lw=0.6),
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="green", alpha=0.8))
                        else:
                            ax.annotate(
                                f"Sec.{j+1} ({sp[0]:.2f} Hz)",
                                xy=(sp[0], sp[1]), xytext=(8, -14),
                                textcoords="offset points",
                                fontsize=max(ann_fs - 1, 5), color="green")

                if self._chk_sigma.isChecked() and std is not None:
                    ax.fill_between(med_f, med_a - std, med_a + std,
                                    alpha=self._sigma_alpha.value(),
                                    color="gray", label="±1σ")

        ax.set_xscale("log")
        ax.set_xlabel(self._xlabel_edit.text())
        ax.set_ylabel(self._ylabel_edit.text())
        title = self._title_edit.text() or f"All Profiles ({n})"
        ax.set_title(title, fontsize=12, fontweight="bold")

        if self._chk_grid.isChecked():
            ax.grid(True, alpha=self._grid_alpha.value(), which="both")

        # Y-limits
        if not self._chk_ylim_auto.isChecked():
            pass  # Manual limits (future spinbox)
        else:
            method = self._ylim_method.currentText()
            if method != "Auto":
                self._apply_smart_ylim(ax, method)

        if n <= 15:
            ax.legend(fontsize=6, loc="upper right", ncol=2,
                      framealpha=0.8)
        fig.tight_layout()
        self._hv_plot.refresh()

    def _apply_smart_ylim(self, ax, method):
        """Apply smart Y-limit methods."""
        all_amps = []
        for r in self._results:
            if r.amps is not None:
                all_amps.extend(r.amps.tolist())
        if not all_amps:
            return
        arr = np.array(all_amps)
        ymin = max(0, np.min(arr) * 0.9)
        if method == "95th Percentile":
            ymax = np.percentile(arr, 95) * 1.1
        elif method == "Mean + 3σ":
            ymax = np.mean(arr) + 3 * np.std(arr)
        elif method == "Mean + 2×IQR":
            q1, q3 = np.percentile(arr, [25, 75])
            ymax = np.mean(arr) + 2 * (q3 - q1)
        else:
            return
        ax.set_ylim(ymin, ymax)

    def _redraw_vs(self, *_args):
        """Draw all Vs profiles overlaid — robust singular matrix fix."""
        profiles = [r for r in self._results if r.profile]
        fig = self._vs_plot.figure

        # Guard: if the widget has zero size, skip (avoids singular matrix)
        size = fig.get_size_inches()
        if size[0] < 0.1 or size[1] < 0.1:
            return

        if not profiles:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Vs data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            self._safe_tight_layout(fig)
            self._vs_plot.refresh()
            return

        fig.clear()
        ax = fig.add_subplot(111)

        n = len(profiles)
        colors = self._get_colors(n)
        all_depths_list = []
        all_vs_list = []
        any_data = False

        for i, r in enumerate(profiles):
            depths, vs = [], []
            z = 0.0
            finite = [L for L in r.profile.layers if not L.is_halfspace]
            for L in finite:
                depths.append(z); vs.append(L.vs)
                z += L.thickness
                depths.append(z); vs.append(L.vs)
            if depths:
                ax.plot(vs, depths, color=colors[i % len(colors)],
                        lw=1.0, alpha=0.6, label=r.name)
                any_data = True
            all_depths_list.append(depths)
            all_vs_list.append(vs)

        if not any_data:
            ax.text(0.5, 0.5, "No Vs data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            self._safe_tight_layout(fig)
            self._vs_plot.refresh()
            return

        # Median Vs profile
        if self._chk_vs_median.isChecked() and len(profiles) >= 2:
            try:
                max_depth = max(max(d) for d in all_depths_list if d)
                common_d = np.linspace(0, max_depth, 200)
                interps = []
                for depths, vs in zip(all_depths_list, all_vs_list):
                    if len(depths) >= 2:
                        interps.append(np.interp(common_d, depths, vs))
                if len(interps) >= 2:
                    med_vs = np.median(np.array(interps), axis=0)
                    ax.plot(med_vs, common_d, color="black", lw=2.5,
                            label="Median Vs", zorder=10)
            except Exception:
                pass

        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=8)
        ax.set_ylabel("Depth (m)", fontsize=8)
        ax.set_title("Vs Profiles", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        # Force axis limits so transforms are resolved before axhline
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        fig.canvas.draw()

        # Vs30 reference line
        if self._chk_vs30.isChecked():
            try:
                ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.6)
                xlim = ax.get_xlim()
                ax.annotate("Vs30 (30 m)", xy=(xlim[0] + (xlim[1] - xlim[0]) * 0.05, 30.0),
                            fontsize=7, color="blue", fontweight="bold",
                            xytext=(0, -8), textcoords="offset points")
            except Exception:
                pass

        if n <= 10:
            ax.legend(fontsize=6, loc="lower right")
        self._safe_tight_layout(fig)
        self._vs_plot.refresh()

    def _toggle_vs(self, show):
        self._vs_panel.setVisible(show)
        if show and self._results:
            # Defer redraw so Qt can lay out the newly-visible widget first;
            # without this the figure has a 0×0 size → singular matrix.
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(50, self._redraw_vs)

    # ── Statistics ─────────────────────────────────────────────

    @staticmethod
    def _safe_tight_layout(fig):
        """Call tight_layout, swallowing singular-matrix errors."""
        try:
            fig.tight_layout()
        except Exception:
            pass

    def _compute_stats(self):
        computed = [r for r in self._results if r.freqs is not None]
        if len(computed) < 2:
            return None, None, None
        ref = computed[0].freqs
        interps = []
        for r in computed:
            interps.append(np.interp(ref, r.freqs, r.amps))
        arr = np.array(interps)
        return ref, np.median(arr, axis=0), np.std(arr, axis=0)

    # ── Palette ────────────────────────────────────────────────

    def _get_colors(self, n):
        import matplotlib.pyplot as plt
        name = self._palette.currentText().lower()
        if name in _BUILTIN_COLORS:
            base = _BUILTIN_COLORS[name]
            return [base[i % len(base)] for i in range(n)]
        try:
            cmap = plt.get_cmap(name, max(n, 2))
            return [cmap(i / max(n - 1, 1)) for i in range(n)]
        except ValueError:
            cmap = plt.get_cmap("tab10", max(n, 2))
            return [cmap(i / max(n - 1, 1)) for i in range(n)]

    # ── Save / Load ────────────────────────────────────────────

    def _save_results(self):
        """Save only All Profiles outputs (combined figures, median, tables)."""
        if not self._results:
            return
        folder = QFileDialog.getExistingDirectory(self, "Save All Profiles Output To")
        if not folder:
            return

        from pathlib import Path
        import matplotlib.pyplot as plt
        base = Path(folder)

        dpi = self._dpi.value()
        fmt = self._export_fmt.currentText().lower()
        fig_key = self._fig_size.currentText()
        figsize = FIGURE_SIZES.get(fig_key, (12, 8))
        computed = [r for r in self._results if r.computed]

        # Save to all_profile_output/ subfolder
        out_dir = base / "all_profile_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._save_combined(out_dir, computed, figsize, dpi, fmt)
        self._save_paper_figures(out_dir, computed, figsize, dpi, fmt)

        # Update peak_info.txt per profile ONLY if peaks were changed
        for r in computed:
            pk = self._peak_data.get(r.name, {})
            f0 = pk.get("f0") or r.f0
            if f0:
                prof_dir = base / r.name
                if prof_dir.exists():
                    with open(prof_dir / "peak_info.txt", "w") as f:
                        f.write(f"f0_Frequency_Hz,{f0[0]:.6f}\n")
                        f.write(f"f0_Amplitude,{f0[1]:.6f}\n")
                        f.write(f"f0_Index,{f0[2]}\n")
                        for j, s in enumerate(pk.get("secondary", [])):
                            f.write(f"Secondary_{j+1}_Frequency_Hz,{s[0]:.6f}\n")
                            f.write(f"Secondary_{j+1}_Amplitude,{s[1]:.6f}\n")
                            f.write(f"Secondary_{j+1}_Index,{s[2]}\n")

        if self._mw:
            self._mw.log(f"All Profiles output saved to {out_dir}")

    def _save_all_results(self):
        """Full re-save: per-profile figures + All Profiles outputs."""
        if not self._results:
            return
        folder = QFileDialog.getExistingDirectory(self, "Save All Results To")
        if not folder:
            return

        from pathlib import Path
        import matplotlib.pyplot as plt
        base = Path(folder)
        base.mkdir(parents=True, exist_ok=True)

        dpi = self._dpi.value()
        fmt = self._export_fmt.currentText().lower()
        fig_key = self._fig_size.currentText()
        figsize = FIGURE_SIZES.get(fig_key, (12, 8))
        computed = [r for r in self._results if r.computed]

        for r in computed:
            prof_dir = base / r.name
            prof_dir.mkdir(exist_ok=True)

            # hv_curve.csv
            if r.freqs is not None:
                with open(prof_dir / "hv_curve.csv", "w") as f:
                    f.write("frequency,amplitude\n")
                    for freq, amp in zip(r.freqs, r.amps):
                        f.write(f"{freq},{amp}\n")

            # peak_info.txt
            pk = self._peak_data.get(r.name, {})
            f0 = pk.get("f0") or r.f0
            if f0:
                with open(prof_dir / "peak_info.txt", "w") as f:
                    f.write(f"f0_Frequency_Hz,{f0[0]:.6f}\n")
                    f.write(f"f0_Amplitude,{f0[1]:.6f}\n")
                    f.write(f"f0_Index,{f0[2]}\n")
                    for j, s in enumerate(pk.get("secondary", [])):
                        f.write(f"Secondary_{j+1}_Frequency_Hz,{s[0]:.6f}\n")
                        f.write(f"Secondary_{j+1}_Amplitude,{s[1]:.6f}\n")
                        f.write(f"Secondary_{j+1}_Index,{s[2]}\n")

            # hv_forward_curve figure
            self._save_profile_figure(r, pk, prof_dir, figsize, dpi, fmt)

            # Vs profile figure
            if r.profile:
                try:
                    self._save_vs_figure(r, prof_dir, dpi)
                except Exception:
                    pass

            # Vs30/VsAvg info
            if r.profile:
                self._save_vs_info(r, prof_dir)

        # Combined outputs
        out_dir = base / "all_profile_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_combined(out_dir, computed, figsize, dpi, fmt)
        self._save_paper_figures(out_dir, computed, figsize, dpi, fmt)

        if self._mw:
            self._mw.log(f"Full results saved to {folder}")

    def _save_profile_figure(self, r, pk, prof_dir, figsize, dpi, fmt):
        """Save individual HV figure matching old package quality."""
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt

        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.plot(r.freqs, r.amps, color="royalblue", lw=2.0, label="H/V Ratio")
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("H/V Amplitude Ratio", fontsize=11)
        ax.set_title(f"HV Spectral Ratio — {r.name}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

        f0 = pk.get("f0") or r.f0
        sec = pk.get("secondary", [])

        if f0:
            ax.plot(f0[0], f0[1], "*", color="firebrick", ms=16, zorder=10,
                    markeredgecolor="darkred", markeredgewidth=0.8,
                    label=f"f0 = {f0[0]:.2f} Hz")
            ax.axvline(f0[0], color="firebrick", ls="--", lw=0.9, alpha=0.4)

        sec_colors = ["green", "purple", "orange", "brown", "teal"]
        for j, s in enumerate(sec):
            sc = sec_colors[j % len(sec_colors)]
            ax.plot(s[0], s[1], "*", color=sc, ms=13, zorder=9,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=f"Secondary ({s[0]:.2f} Hz, A={s[1]:.2f})")
            ax.axvline(s[0], color=sc, ls=":", lw=0.8, alpha=0.4)

        ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
                  edgecolor="gray")

        fig.tight_layout()
        fig.savefig(prof_dir / f"hv_forward_curve.{fmt}", dpi=dpi)
        if fmt != "pdf":
            fig.savefig(prof_dir / "hv_forward_curve.pdf", dpi=dpi)
        plt.close(fig)

    def _save_vs_figure(self, r, prof_dir, dpi=300):
        """Save individual Vs profile figure with Vs30/VsAvg annotations."""
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt

        fig = Figure(figsize=(5, 7))
        ax = fig.add_subplot(111)
        depths, vs = [], []
        z = 0.0
        finite = [L for L in r.profile.layers if not L.is_halfspace]
        hs = [L for L in r.profile.layers if L.is_halfspace]
        for L in finite:
            depths.append(z); vs.append(L.vs)
            z += L.thickness
            depths.append(z); vs.append(L.vs)
        if hs:
            depths.append(z); vs.append(hs[0].vs)
            depths.append(z + z * 0.25); vs.append(hs[0].vs)

        ax.plot(vs, depths, color="teal", lw=1.8)
        ax.fill_betweenx(depths, 0, vs, alpha=0.1, color="teal")
        ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=10)
        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.set_title(f"Vs Profile — {r.name}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        vs_max = max(vs) if vs else 500

        # Vs30
        try:
            from hvstrip_progressive.core.vs_average import vs_average_from_profile
            res30 = vs_average_from_profile(r.profile, target_depth=30.0)
            ax.axhline(30.0, color="blue", lw=0.8, ls="-.", alpha=0.6)
            ax.annotate(f"Vs30 = {res30.vs_avg:.0f} m/s",
                        xy=(vs_max * 0.5, 30.0),
                        xytext=(0, -10), textcoords="offset points",
                        fontsize=8, color="blue", fontweight="bold")
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(prof_dir / "vs_profile.png", dpi=dpi)
        fig.savefig(prof_dir / "vs_profile.pdf", dpi=dpi)
        plt.close(fig)

    def _save_vs_info(self, r, prof_dir):
        """Save Vs30 and VsAvg info files."""
        try:
            from hvstrip_progressive.core.vs_average import vs_average_from_profile
            res = vs_average_from_profile(r.profile, target_depth=30.0)
            with open(prof_dir / "vs30_info.txt", "w") as f:
                f.write(f"Vs30_m_per_s,{res.vs_avg:.2f}\n")
                f.write(f"Target_Depth_m,{res.target_depth:.1f}\n")
                f.write(f"Actual_Depth_m,{res.actual_depth:.1f}\n")
                f.write(f"Extrapolated,{res.extrapolated}\n")
        except Exception:
            pass

    def _save_combined(self, out_dir, computed, figsize, dpi, fmt):
        """Save combined/median outputs to out_dir (all_profile_output/)."""
        import matplotlib.pyplot as plt

        # Combined summary CSV with Vs30/VsAvg
        with open(out_dir / "combined_summary.csv", "w") as f:
            f.write("Profile,f0_Hz,f0_Amplitude,Secondary_Peaks,Vs30_m_s,VsAvg_m_s\n")
            for r in self._results:
                pk = self._peak_data.get(r.name, {})
                f0 = pk.get("f0") or r.f0
                vs30_str = ""
                vsavg_str = ""
                if r.profile:
                    try:
                        from hvstrip_progressive.core.vs_average import vs_average_from_profile
                        res30 = vs_average_from_profile(r.profile, target_depth=30.0)
                        vs30_str = f"{res30.vs_avg:.2f}"
                    except Exception:
                        pass
                if r.computed and f0:
                    sec = "; ".join(
                        f"{s[0]:.3f} Hz" for s in pk.get("secondary", []))
                    f.write(f'{r.name},{f0[0]:.6f},{f0[1]:.6f},"{sec}",{vs30_str},{vsavg_str}\n')
                else:
                    f.write(f"{r.name},,,,{vs30_str},{vsavg_str}\n")

            if len(computed) >= 2:
                med_f, med_a, _ = self._compute_stats()
                if med_f is not None:
                    idx = int(np.argmax(med_a))
                    f.write(f"Median,{med_f[idx]:.6f},{med_a[idx]:.6f},,,\n")

        # Median HV curve CSV
        if len(computed) >= 2:
            med_f, med_a, std = self._compute_stats()
            if med_f is not None:
                with open(out_dir / "median_hv_curve.csv", "w") as f:
                    f.write("frequency,median_amplitude,std\n")
                    for freq, amp, s in zip(med_f, med_a, std):
                        f.write(f"{freq},{amp},{s}\n")

                # Median peak info (user-selected or auto)
                mp = self._median_peaks
                f0m = mp.get("f0")
                if f0m is None:
                    idx = int(np.argmax(med_a))
                    f0m = (med_f[idx], med_a[idx], idx)
                with open(out_dir / "median_peak_info.txt", "w") as f:
                    f.write(f"Median_f0_Frequency_Hz,{f0m[0]:.6f}\n")
                    f.write(f"Median_f0_Amplitude,{f0m[1]:.6f}\n")
                    for j, sp in enumerate(mp.get("secondary", [])):
                        f.write(f"Median_Secondary_{j+1}_Frequency_Hz,{sp[0]:.6f}\n")
                        f.write(f"Median_Secondary_{j+1}_Amplitude,{sp[1]:.6f}\n")

        # Save current overlay figure at high quality
        try:
            self._hv_plot.figure.savefig(
                out_dir / f"combined_hv_curves.{fmt}", dpi=dpi)
            if fmt != "pdf":
                self._hv_plot.figure.savefig(
                    out_dir / "combined_hv_curves.pdf", dpi=dpi)
        except Exception:
            pass

        # Save Vs overlay if visible
        if self._chk_vs.isChecked():
            try:
                self._vs_plot.figure.savefig(
                    out_dir / f"vs_profiles_overlay.{fmt}", dpi=dpi)
                if fmt != "pdf":
                    self._vs_plot.figure.savefig(
                        out_dir / "vs_profiles_overlay.pdf", dpi=dpi)
            except Exception:
                pass

    def _save_paper_figures(self, out_dir, computed, figsize, dpi, fmt):
        """Generate publication-quality figures for journal papers."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        n = len(computed)
        colors = self._get_colors(n)

        # ── 1. Combined HV + Vs side-by-side ──────────────────
        try:
            fig = Figure(figsize=(16, 8))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            # Left panel: HV curves
            for i, r in enumerate(computed):
                c = colors[i % len(colors)]
                ax1.plot(r.freqs, r.amps, color=c, lw=0.8, alpha=0.5,
                         label=r.name)

            med_f, med_a, std = self._compute_stats()
            if med_f is not None:
                ax1.plot(med_f, med_a, color="black", lw=2.5,
                         label="Median", zorder=10)
                if std is not None:
                    ax1.fill_between(med_f, med_a - std, med_a + std,
                                     alpha=0.15, color="gray", label="±1σ")

                mp = self._median_peaks
                f0m = mp.get("f0")
                if f0m is None and med_a is not None:
                    idx = int(np.argmax(med_a))
                    f0m = (med_f[idx], med_a[idx], idx)
                if f0m:
                    ax1.plot(f0m[0], f0m[1], "*", color="red", ms=14,
                             zorder=11, markeredgecolor="darkred",
                             markeredgewidth=0.8,
                             label=f"f0 = {f0m[0]:.3f} Hz")
                    ax1.axvline(f0m[0], color="red", ls="--", lw=0.8,
                                alpha=0.4)
                for j, sp in enumerate(mp.get("secondary", [])):
                    ax1.plot(sp[0], sp[1], "*", color="green", ms=12,
                             zorder=11, markeredgecolor="darkgreen",
                             markeredgewidth=0.6,
                             label=f"Sec.{j+1} ({sp[0]:.2f} Hz)")

            ax1.set_xscale("log")
            ax1.set_xlabel("Frequency (Hz)", fontsize=11)
            ax1.set_ylabel("H/V Amplitude Ratio", fontsize=11)
            ax1.set_title("All Profiles — H/V Curves", fontsize=12,
                          fontweight="bold")
            ax1.grid(True, alpha=0.3, which="both")
            ax1.legend(fontsize=6, loc="upper right", ncol=2,
                       framealpha=0.9)

            # Right panel: Vs profiles
            for i, r in enumerate(computed):
                if not r.profile:
                    continue
                depths, vs = [], []
                z = 0.0
                finite = [L for L in r.profile.layers if not L.is_halfspace]
                for L in finite:
                    depths.append(z); vs.append(L.vs)
                    z += L.thickness
                    depths.append(z); vs.append(L.vs)
                if depths:
                    c = colors[i % len(colors)]
                    ax2.plot(vs, depths, color=c, lw=0.8, alpha=0.6,
                             label=r.name)

            ax2.invert_yaxis()
            ax2.set_xlabel("Vs (m/s)", fontsize=11)
            ax2.set_ylabel("Depth (m)", fontsize=11)
            ax2.set_title("Vs Profiles", fontsize=12, fontweight="bold")
            ax2.grid(True, alpha=0.3)

            # Vs30 line
            try:
                ax2.axhline(30.0, color="blue", lw=0.8, ls="-.",
                            alpha=0.6, label="Vs30 (30 m)")
            except Exception:
                pass

            ax2.legend(fontsize=6, loc="lower right", ncol=2,
                       framealpha=0.9)

            fig.tight_layout()
            fig.savefig(out_dir / f"publication_hv_vs_combined.{fmt}", dpi=dpi)
            if fmt != "pdf":
                fig.savefig(out_dir / "publication_hv_vs_combined.pdf", dpi=dpi)
            plt.close(fig)
        except Exception:
            pass

        # ── 2. Median-only clean figure ± σ ────────────────────
        try:
            if med_f is not None:
                fig = Figure(figsize=(12, 8))
                ax = fig.add_subplot(111)

                ax.plot(med_f, med_a, color="black", lw=2.5, label="Median H/V")
                if std is not None:
                    ax.fill_between(med_f, med_a - std, med_a + std,
                                     alpha=0.2, color="gray", label="±1σ")

                f0m = self._median_peaks.get("f0")
                if f0m is None:
                    idx = int(np.argmax(med_a))
                    f0m = (med_f[idx], med_a[idx], idx)
                if f0m:
                    ax.plot(f0m[0], f0m[1], "*", color="red", ms=16,
                            zorder=10, markeredgecolor="darkred",
                            markeredgewidth=0.8,
                            label=f"f0 = {f0m[0]:.3f} Hz (A = {f0m[1]:.2f})")
                    ax.axvline(f0m[0], color="red", ls="--", lw=0.9, alpha=0.4)

                for j, sp in enumerate(self._median_peaks.get("secondary", [])):
                    ax.plot(sp[0], sp[1], "*", color="green", ms=13,
                            zorder=9, markeredgecolor="black",
                            markeredgewidth=0.5,
                            label=f"Secondary ({sp[0]:.2f} Hz, A={sp[1]:.2f})")
                    ax.axvline(sp[0], color="green", ls=":", lw=0.8, alpha=0.4)

                ax.set_xscale("log")
                ax.set_xlabel("Frequency (Hz)", fontsize=12)
                ax.set_ylabel("H/V Amplitude Ratio", fontsize=12)
                ax.set_title("Median H/V Spectral Ratio", fontsize=14,
                             fontweight="bold")
                ax.grid(True, alpha=0.3, which="both")
                ax.legend(fontsize=10, loc="upper right", framealpha=0.9,
                          edgecolor="gray")

                fig.tight_layout()
                fig.savefig(out_dir / f"publication_median_hv.{fmt}", dpi=dpi)
                if fmt != "pdf":
                    fig.savefig(out_dir / "publication_median_hv.pdf", dpi=dpi)
                plt.close(fig)
        except Exception:
            pass

        # ── 3. Summary table (LaTeX-ready CSV) ────────────────
        try:
            with open(out_dir / "summary_table.csv", "w") as f:
                f.write("Profile,f0 (Hz),Amplitude,Vs30 (m/s),")
                f.write("Secondary Peaks\n")
                for r in computed:
                    pk = self._peak_data.get(r.name, {})
                    f0 = pk.get("f0") or r.f0
                    f0_str = f"{f0[0]:.3f}" if f0 else "—"
                    amp_str = f"{f0[1]:.2f}" if f0 else "—"
                    vs30_str = "—"
                    if r.profile:
                        try:
                            from hvstrip_progressive.core.vs_average import vs_average_from_profile
                            res30 = vs_average_from_profile(
                                r.profile, target_depth=30.0)
                            vs30_str = f"{res30.vs_avg:.0f}"
                        except Exception:
                            pass
                    sec_str = "; ".join(
                        f"{s[0]:.2f}" for s in pk.get("secondary", []))
                    f.write(f"{r.name},{f0_str},{amp_str},{vs30_str},{sec_str}\n")

                # Median row
                if med_f is not None:
                    f0m = self._median_peaks.get("f0")
                    if f0m is None:
                        idx = int(np.argmax(med_a))
                        f0m = (med_f[idx], med_a[idx], idx)
                    f.write(f"Median,{f0m[0]:.3f},{f0m[1]:.2f},—,\n")

            # LaTeX table
            with open(out_dir / "summary_table.tex", "w") as f:
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{HVSR Analysis Summary}\n")
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\hline\n")
                f.write("Profile & $f_0$ (Hz) & Amplitude & $V_{s30}$ (m/s) & ")
                f.write("Secondary Peaks \\\\\n")
                f.write("\\hline\n")
                for r in computed:
                    pk = self._peak_data.get(r.name, {})
                    f0 = pk.get("f0") or r.f0
                    f0_str = f"{f0[0]:.3f}" if f0 else "—"
                    amp_str = f"{f0[1]:.2f}" if f0 else "—"
                    vs30_str = "—"
                    if r.profile:
                        try:
                            from hvstrip_progressive.core.vs_average import vs_average_from_profile
                            res30 = vs_average_from_profile(
                                r.profile, target_depth=30.0)
                            vs30_str = f"{res30.vs_avg:.0f}"
                        except Exception:
                            pass
                    sec_str = "; ".join(
                        f"{s[0]:.2f}" for s in pk.get("secondary", []))
                    name = r.name.replace("_", "\\_")
                    f.write(f"{name} & {f0_str} & {amp_str} & ")
                    f.write(f"{vs30_str} & {sec_str} \\\\\n")
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
        except Exception:
            pass

        # ── 4. Vs comparison figure (standalone) ───────────────
        try:
            profiles_with_vs = [r for r in computed if r.profile]
            if profiles_with_vs:
                fig = Figure(figsize=(8, 10))
                ax = fig.add_subplot(111)
                all_d, all_v = [], []
                for i, r in enumerate(profiles_with_vs):
                    depths, vs = [], []
                    z = 0.0
                    finite = [L for L in r.profile.layers if not L.is_halfspace]
                    for L in finite:
                        depths.append(z); vs.append(L.vs)
                        z += L.thickness
                        depths.append(z); vs.append(L.vs)
                    if depths:
                        c = colors[i % len(colors)]
                        ax.plot(vs, depths, color=c, lw=1.0, alpha=0.6,
                                label=r.name)
                        all_d.append(depths)
                        all_v.append(vs)

                # Median Vs
                if len(all_d) >= 2:
                    max_d = max(max(d) for d in all_d)
                    common = np.linspace(0, max_d, 200)
                    interps = []
                    for d, v in zip(all_d, all_v):
                        if len(d) >= 2:
                            interps.append(np.interp(common, d, v))
                    if len(interps) >= 2:
                        med_vs = np.median(np.array(interps), axis=0)
                        ax.plot(med_vs, common, color="black", lw=2.5,
                                label="Median Vs", zorder=10)

                ax.invert_yaxis()
                ax.set_xlabel("Vs (m/s)", fontsize=12)
                ax.set_ylabel("Depth (m)", fontsize=12)
                ax.set_title("Vs Profile Comparison", fontsize=14,
                             fontweight="bold")
                ax.grid(True, alpha=0.3)

                try:
                    ax.axhline(30.0, color="blue", lw=0.8, ls="-.",
                               alpha=0.6, label="Vs30 (30 m)")
                except Exception:
                    pass

                ax.legend(fontsize=7, loc="lower right", ncol=2,
                          framealpha=0.9)
                fig.tight_layout()
                fig.savefig(out_dir / f"publication_vs_comparison.{fmt}",
                            dpi=dpi)
                if fmt != "pdf":
                    fig.savefig(out_dir / "publication_vs_comparison.pdf",
                                dpi=dpi)
                plt.close(fig)
        except Exception:
            pass

    def _load_results(self):
        """Load previously saved results folder."""
        folder = QFileDialog.getExistingDirectory(self, "Load Results Folder")
        if not folder:
            return

        from pathlib import Path
        from ..workers.multi_forward_worker import MultiForwardWorker

        base = Path(folder)
        skip_dirs = {"median_output", "all_profile_output"}
        results = []
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and sub.name not in skip_dirs:
                csv_f = sub / "hv_curve.csv"
                if csv_f.exists():
                    try:
                        pr = MultiForwardWorker._load_result_from_folder(
                            sub.name, str(sub))
                        results.append(pr)
                    except Exception as e:
                        if self._mw:
                            self._mw.log(f"Error loading {sub.name}: {e}")

        if results:
            self.set_results(results)
            self.results_loaded.emit()
            if self._mw:
                self._mw.log(f"Loaded {len(results)} profiles from {folder}")
