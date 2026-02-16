"""
Output Viewer Dialog.

Loads an existing multi-profile output folder and displays
a combined overlay figure with configurable settings and median toggle.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from .multi_profile_dialog import (
    FigureSettings, ProfileResult,
    get_palette_colors, _darken_color,
    _PALETTE_NAMES, _MARKER_COLORS, _MARKER_SHAPES,
)
from ...core.soil_profile import SoilProfile


# ------------------------------------------------------------------ loading

def load_profile_from_folder(folder: Path) -> Optional[ProfileResult]:
    """Load a ProfileResult from a saved profile subfolder."""
    csv_path = folder / "hv_curve.csv"
    peak_path = folder / "peak_info.txt"
    if not csv_path.exists():
        return None

    freqs, amps = [], []
    for line in csv_path.read_text().splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) == 2:
            freqs.append(float(parts[0]))
            amps.append(float(parts[1]))
    if not freqs:
        return None

    f0 = None
    secondary_peaks: List[Tuple[float, float, int]] = []

    if peak_path.exists():
        kv = {}
        for line in peak_path.read_text().splitlines():
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                kv[parts[0]] = parts[1]
        if "f0_Frequency_Hz" in kv:
            f0 = (
                float(kv["f0_Frequency_Hz"]),
                float(kv.get("f0_Amplitude", 0)),
                int(kv.get("f0_Index", 0)),
            )
        i = 1
        while f"Secondary_{i}_Frequency_Hz" in kv:
            secondary_peaks.append((
                float(kv[f"Secondary_{i}_Frequency_Hz"]),
                float(kv.get(f"Secondary_{i}_Amplitude", 0)),
                int(kv.get(f"Secondary_{i}_Index", 0)),
            ))
            i += 1

    dummy_profile = SoilProfile(name=folder.name)
    return ProfileResult(
        name=folder.name,
        profile=dummy_profile,
        freqs=np.array(freqs),
        amps=np.array(amps),
        f0=f0,
        secondary_peaks=secondary_peaks,
        computed=True,
    )


def load_median_from_folder(folder: Path) -> Optional[ProfileResult]:
    """Load median ProfileResult from the output root folder."""
    csv_path = folder / "median_hv_curve.csv"
    peak_path = folder / "median_peak_info.txt"
    if not csv_path.exists():
        return None

    freqs, amps = [], []
    header_col = "median_amplitude"
    for line in csv_path.read_text().splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) == 2:
            freqs.append(float(parts[0]))
            amps.append(float(parts[1]))
    if not freqs:
        return None

    f0 = None
    secondary_peaks: List[Tuple[float, float, int]] = []

    if peak_path.exists():
        kv = {}
        for line in peak_path.read_text().splitlines():
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                kv[parts[0]] = parts[1]
        if "Median_f0_Frequency_Hz" in kv:
            freq_val = float(kv["Median_f0_Frequency_Hz"])
            amp_val = float(kv.get("Median_f0_Amplitude", 0))
            idx = int(np.argmin(np.abs(np.array(freqs) - freq_val)))
            f0 = (freq_val, amp_val, idx)
        i = 1
        while f"Median_Secondary_{i}_Frequency_Hz" in kv:
            sf = float(kv[f"Median_Secondary_{i}_Frequency_Hz"])
            sa = float(kv.get(f"Median_Secondary_{i}_Amplitude", 0))
            si = int(np.argmin(np.abs(np.array(freqs) - sf)))
            secondary_peaks.append((sf, sa, si))
            i += 1

    dummy_profile = SoilProfile(name="Median HV")
    return ProfileResult(
        name="Median HV",
        profile=dummy_profile,
        freqs=np.array(freqs),
        amps=np.array(amps),
        f0=f0,
        secondary_peaks=secondary_peaks,
        computed=True,
    )


def load_output_folder(folder: Path) -> Tuple[List[ProfileResult], Optional[ProfileResult]]:
    """Load all profiles and median from an output folder.

    Returns
    -------
    results : list of ProfileResult
    median_result : ProfileResult or None
    """
    import re
    
    def natural_sort_key(path: Path) -> list:
        """Natural sort key for numeric ordering (Profile_1, Profile_2, ..., Profile_10)."""
        return [int(c) if c.isdigit() else c.lower() 
                for c in re.split(r'(\d+)', path.name)]
    
    results = []
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    for sub in sorted(subdirs, key=natural_sort_key):
        if (sub / "hv_curve.csv").exists():
            pr = load_profile_from_folder(sub)
            if pr is not None:
                results.append(pr)
    median = load_median_from_folder(folder)
    return results, median


# ------------------------------------------------------------------ dialog

class OutputViewerDialog(QDialog):
    """Dialog for viewing a loaded multi-profile output folder."""

    def __init__(
        self,
        results: List[ProfileResult],
        median_result: Optional[ProfileResult],
        fig_settings: FigureSettings,
        source_folder: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Output Viewer — Combined HV Curves")
        self.setMinimumSize(1200, 650)
        self._results = results
        self._median_result = median_result
        self._fig_settings = fig_settings
        self._source_folder = source_folder
        self._setup_ui()
        self._replot()

    # -------------------------------------------------------------- UI
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: settings ---
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(5, 5, 5, 5)

        lbl = QLabel(f"Profiles loaded: {len(self._results)}")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        left_lay.addWidget(lbl)

        if self._source_folder:
            src = QLabel(str(self._source_folder))
            src.setWordWrap(True)
            src.setStyleSheet("color: #555; font-size: 10px;")
            left_lay.addWidget(src)

        # ── Visibility Tree ──
        vis_grp = QGroupBox("Layer Visibility")
        vis_lay = QVBoxLayout(vis_grp)
        self.vis_tree = QTreeWidget()
        self.vis_tree.setHeaderHidden(True)
        self.vis_tree.setRootIsDecorated(True)
        self.vis_tree.setStyleSheet("QTreeWidget { font-size: 12px; }")

        # Profiles branch
        self._tree_profiles = QTreeWidgetItem(self.vis_tree, ["Profiles"])
        self._tree_profiles.setFlags(
            self._tree_profiles.flags() | Qt.ItemIsUserCheckable
        )
        self._tree_profiles.setCheckState(0, Qt.Checked)

        self._tree_prof_f0 = QTreeWidgetItem(self._tree_profiles, ["Primary Peaks (f0)"])
        self._tree_prof_f0.setFlags(self._tree_prof_f0.flags() | Qt.ItemIsUserCheckable)
        self._tree_prof_f0.setCheckState(0, Qt.Checked)

        self._tree_prof_sec = QTreeWidgetItem(self._tree_profiles, ["Secondary Peaks"])
        self._tree_prof_sec.setFlags(self._tree_prof_sec.flags() | Qt.ItemIsUserCheckable)
        self._tree_prof_sec.setCheckState(0, Qt.Checked)

        # Median branch
        has_med = self._median_result is not None
        self._tree_median = QTreeWidgetItem(self.vis_tree, ["Median Curve"])
        self._tree_median.setFlags(self._tree_median.flags() | Qt.ItemIsUserCheckable)
        self._tree_median.setCheckState(0, Qt.Checked if has_med else Qt.Unchecked)
        if not has_med:
            self._tree_median.setDisabled(True)

        self._tree_med_peaks = QTreeWidgetItem(self._tree_median, ["Median Peaks"])
        self._tree_med_peaks.setFlags(
            self._tree_med_peaks.flags() | Qt.ItemIsUserCheckable
        )
        self._tree_med_peaks.setCheckState(0, Qt.Checked if has_med else Qt.Unchecked)
        if not has_med:
            self._tree_med_peaks.setDisabled(True)

        self.vis_tree.expandAll()
        self.vis_tree.itemChanged.connect(self._on_tree_changed)
        vis_lay.addWidget(self.vis_tree)
        left_lay.addWidget(vis_grp)

        # --- Plot Settings ---
        s = self._fig_settings
        grp = QGroupBox("Plot Settings")
        g_lay = QVBoxLayout(grp)

        palette_row = QHBoxLayout()
        palette_row.addWidget(QLabel("Palette:"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(_PALETTE_NAMES)
        self.palette_combo.setCurrentText(s.color_palette)
        self.palette_combo.currentTextChanged.connect(self._on_settings_changed)
        palette_row.addWidget(self.palette_combo)
        g_lay.addLayout(palette_row)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Profile \u03b1:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(s.individual_alpha)
        self.alpha_spin.valueChanged.connect(self._on_settings_changed)
        alpha_row.addWidget(self.alpha_spin)
        g_lay.addLayout(alpha_row)

        ilw_row = QHBoxLayout()
        ilw_row.addWidget(QLabel("Profile LW:"))
        self.ilw_spin = QDoubleSpinBox()
        self.ilw_spin.setRange(0.3, 5.0)
        self.ilw_spin.setSingleStep(0.1)
        self.ilw_spin.setValue(s.individual_linewidth)
        self.ilw_spin.valueChanged.connect(self._on_settings_changed)
        ilw_row.addWidget(self.ilw_spin)
        g_lay.addLayout(ilw_row)

        mlw_row = QHBoxLayout()
        mlw_row.addWidget(QLabel("Median LW:"))
        self.mlw_spin = QDoubleSpinBox()
        self.mlw_spin.setRange(0.5, 8.0)
        self.mlw_spin.setSingleStep(0.5)
        self.mlw_spin.setValue(s.median_linewidth)
        self.mlw_spin.valueChanged.connect(self._on_settings_changed)
        mlw_row.addWidget(self.mlw_spin)
        g_lay.addLayout(mlw_row)

        peak_alpha_row = QHBoxLayout()
        peak_alpha_row.addWidget(QLabel("Peak \u03b1:"))
        self.peak_alpha_spin = QDoubleSpinBox()
        self.peak_alpha_spin.setRange(0.1, 1.0)
        self.peak_alpha_spin.setSingleStep(0.05)
        self.peak_alpha_spin.setValue(0.9)
        self.peak_alpha_spin.valueChanged.connect(self._on_settings_changed)
        peak_alpha_row.addWidget(self.peak_alpha_spin)
        g_lay.addLayout(peak_alpha_row)

        # ── Peak Markers ──
        sep = QLabel("── Peak Markers ──")
        sep.setStyleSheet("color: #555; font-size: 11px; margin-top: 4px;")
        g_lay.addWidget(sep)

        f0_row = QHBoxLayout()
        f0_row.addWidget(QLabel("f0:"))
        self.f0_color_combo = QComboBox()
        self.f0_color_combo.addItems(list(_MARKER_COLORS.keys()))
        self.f0_color_combo.setCurrentText(s.f0_marker_color)
        self.f0_color_combo.currentTextChanged.connect(self._on_settings_changed)
        f0_row.addWidget(self.f0_color_combo)
        self.f0_shape_combo = QComboBox()
        self.f0_shape_combo.addItems(list(_MARKER_SHAPES.keys()))
        self.f0_shape_combo.setCurrentText(s.f0_marker_shape)
        self.f0_shape_combo.currentTextChanged.connect(self._on_settings_changed)
        f0_row.addWidget(self.f0_shape_combo)
        g_lay.addLayout(f0_row)

        sec_row = QHBoxLayout()
        sec_row.addWidget(QLabel("Sec:"))
        self.sec_color_combo = QComboBox()
        self.sec_color_combo.addItems(list(_MARKER_COLORS.keys()))
        self.sec_color_combo.setCurrentText(s.secondary_marker_color)
        self.sec_color_combo.currentTextChanged.connect(self._on_settings_changed)
        sec_row.addWidget(self.sec_color_combo)
        self.sec_shape_combo = QComboBox()
        self.sec_shape_combo.addItems(list(_MARKER_SHAPES.keys()))
        self.sec_shape_combo.setCurrentText(s.secondary_marker_shape)
        self.sec_shape_combo.currentTextChanged.connect(self._on_settings_changed)
        sec_row.addWidget(self.sec_shape_combo)
        g_lay.addLayout(sec_row)

        left_lay.addWidget(grp)

        # Save button
        self.btn_save = QPushButton("Save Figure")
        self.btn_save.setStyleSheet(
            "padding: 8px 14px; background-color: #0078d4; "
            "color: white; font-weight: bold;"
        )
        self.btn_save.clicked.connect(self._save_figure)
        left_lay.addWidget(self.btn_save)

        left_lay.addStretch()

        self.btn_close = QPushButton("Close")
        self.btn_close.setStyleSheet("padding: 8px 14px;")
        self.btn_close.clicked.connect(self.close)
        left_lay.addWidget(self.btn_close)

        splitter.addWidget(left)

        # --- Right: figure ---
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(5, 5, 5, 5)

        self.figure = Figure(figsize=(14, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_lay.addWidget(self.toolbar)
        right_lay.addWidget(self.canvas)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

    def _on_tree_changed(self, item, column):
        """Propagate parent check state to children and replot."""
        # If a parent was toggled, sync children
        if item.childCount() > 0:
            state = item.checkState(column)
            self.vis_tree.blockSignals(True)
            for i in range(item.childCount()):
                child = item.child(i)
                if not child.isDisabled():
                    child.setCheckState(0, state)
            self.vis_tree.blockSignals(False)
        self._replot()

    # -------------------------------------------------------------- helpers
    def _is_checked(self, item: QTreeWidgetItem) -> bool:
        return item.checkState(0) == Qt.Checked

    # -------------------------------------------------------------- settings
    def _on_settings_changed(self):
        s = self._fig_settings
        s.color_palette = self.palette_combo.currentText()
        s.individual_alpha = self.alpha_spin.value()
        s.individual_linewidth = self.ilw_spin.value()
        s.median_linewidth = self.mlw_spin.value()
        s.f0_marker_color = self.f0_color_combo.currentText()
        s.f0_marker_shape = self.f0_shape_combo.currentText()
        s.secondary_marker_color = self.sec_color_combo.currentText()
        s.secondary_marker_shape = self.sec_shape_combo.currentText()
        self._replot()

    # -------------------------------------------------------------- plotting
    def _replot(self):
        self.figure.clear()
        s = self._fig_settings
        ax = self.figure.add_subplot(111)

        computed = [r for r in self._results if r.freqs is not None]
        if not computed:
            ax.text(0.5, 0.5, "No curves loaded", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        # Read tree visibility state
        show_profiles = self._is_checked(self._tree_profiles)
        show_prof_f0 = self._is_checked(self._tree_prof_f0)
        show_prof_sec = self._is_checked(self._tree_prof_sec)
        show_med = (
            self._is_checked(self._tree_median)
            and self._median_result is not None
            and self._median_result.freqs is not None
        )
        show_med_peaks = show_med and self._is_checked(self._tree_med_peaks)

        n = len(computed)
        colors = get_palette_colors(s.color_palette, n)
        alpha = s.individual_alpha
        lw = s.individual_linewidth
        peak_alpha = self.peak_alpha_spin.value()

        f0_c = _MARKER_COLORS.get(s.f0_marker_color, "#d62728")
        f0_m = _MARKER_SHAPES.get(s.f0_marker_shape, "*")
        sec_c = _MARKER_COLORS.get(s.secondary_marker_color, "#2ca02c")
        sec_m = _MARKER_SHAPES.get(s.secondary_marker_shape, "*")

        # Individual profiles
        if show_profiles:
            for i, r in enumerate(computed):
                c = colors[i]
                ax.plot(r.freqs, r.amps, linewidth=lw,
                        color=c, alpha=alpha, label=r.name)
                if show_prof_f0 and r.f0 is not None:
                    ax.scatter(
                        r.f0[0], r.f0[1], color=c, s=80, marker=f0_m,
                        edgecolors="black", linewidth=0.5, zorder=4,
                        alpha=peak_alpha,
                    )
                if show_prof_sec:
                    for sec_f, sec_a, _ in r.secondary_peaks:
                        ax.scatter(
                            sec_f, sec_a, color=c, s=60, marker=sec_m,
                            edgecolors="black", linewidth=0.5, zorder=4,
                            alpha=peak_alpha,
                        )

        # Median curve
        if show_med:
            mr = self._median_result
            ax.plot(mr.freqs, mr.amps, color="black",
                    linewidth=s.median_linewidth, label="Median HV", zorder=10)
            if show_med_peaks:
                if mr.f0 is not None:
                    f, a, _ = mr.f0
                    ax.scatter(
                        f, a, color=f0_c, s=s.f0_marker_size, marker=f0_m,
                        edgecolors=_darken_color(f0_c), linewidth=1.5,
                        zorder=11, alpha=peak_alpha,
                        label=f"Median f0 = {f:.2f} Hz (A={a:.2f})",
                    )
                    ax.axvline(x=f, color=f0_c, linestyle="--", alpha=0.5)
                for sec_f, sec_a, _ in mr.secondary_peaks:
                    ax.scatter(
                        sec_f, sec_a, color=sec_c,
                        s=s.secondary_marker_size, marker=sec_m,
                        edgecolors=_darken_color(sec_c), linewidth=1.5,
                        zorder=10, alpha=peak_alpha,
                        label=f"Median Sec. ({sec_f:.2f} Hz, A={sec_a:.2f})",
                    )
                    ax.axvline(x=sec_f, color=sec_c, linestyle=":", alpha=0.4)

        title = "Combined HV Curves — All Profiles"
        ax.set_xlabel("Frequency (Hz)", fontsize=s.font_size)
        ax.set_ylabel("H/V Amplitude Ratio", fontsize=s.font_size)
        ax.set_title(title, fontsize=s.title_size, fontweight="bold")
        if s.log_x:
            ax.set_xscale("log")
        if s.log_y:
            ax.set_yscale("log")
        if s.grid:
            ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=max(s.legend_size - 1, 6), ncol=2)
        if computed:
            ax.set_xlim(computed[0].freqs[0], computed[0].freqs[-1])

        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

    # -------------------------------------------------------------- save
    def _save_figure(self):
        from PySide6.QtWidgets import QFileDialog
        s = self._fig_settings
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)",
        )
        if path:
            self.figure.savefig(path, dpi=s.dpi, bbox_inches="tight")
