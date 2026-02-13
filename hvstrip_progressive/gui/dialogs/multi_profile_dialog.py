"""
Multi-Profile Forward Modeling Dialog.

Allows users to process multiple soil profiles sequentially,
select peaks interactively, and save per-profile + combined results.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from ...core.hv_forward import compute_hv_curve
from ...core.soil_profile import SoilProfile


@dataclass
class FigureSettings:
    """Configurable figure generation settings."""

    dpi: int = 300
    width: float = 10.0
    height: float = 5.0
    font_size: int = 12
    title_size: int = 13
    legend_size: int = 10
    log_x: bool = True
    log_y: bool = False
    grid: bool = True
    save_png: bool = True
    save_pdf: bool = True
    show_vs: bool = False
    # Combined plot settings
    color_palette: str = "Classic"  # palette name (see _PALETTES or matplotlib cmap)
    individual_alpha: float = 0.45
    individual_linewidth: float = 1.0
    median_linewidth: float = 3.0
    show_median: bool = True
    show_secondary_peaks: bool = True
    # Peak marker settings
    f0_marker_color: str = "Red"
    f0_marker_shape: str = "Star"
    f0_marker_size: int = 300
    secondary_marker_color: str = "Green"
    secondary_marker_shape: str = "Star"
    secondary_marker_size: int = 200


@dataclass
class ProfileResult:
    """Stores HV computation result and selected peaks for one profile."""

    name: str
    profile: SoilProfile
    freqs: Optional[np.ndarray] = None
    amps: Optional[np.ndarray] = None
    f0: Optional[Tuple[float, float, int]] = None
    secondary_peaks: List[Tuple[float, float, int]] = field(default_factory=list)
    computed: bool = False


_PALETTES = {
    # --- Multi-color ---
    "Classic": [
        "#1f77b4", "#000000", "#2ca02c", "#d62728",
        "#ff7f0e", "#9467bd", "#8c564b", "#7f7f7f",
        "#17becf", "#bcbd22",
    ],
    "Bold": [
        "#0033CC", "#000000", "#006600", "#CC0000",
        "#CC6600", "#660099", "#663300", "#444444",
        "#006666", "#666600",
    ],
    "Earth": [
        "#8B4513", "#006400", "#4682B4", "#8B0000",
        "#DAA520", "#2F4F4F", "#556B2F", "#800080",
        "#B8860B", "#708090",
    ],
    "Minimal": [
        "#2C3E50", "#7F8C8D", "#95A5A6", "#BDC3C7",
        "#34495E", "#5D6D7E", "#808B96", "#ABB2B9",
        "#D5D8DC", "#ECF0F1",
    ],
    "Nordic": [
        "#2E4057", "#048A81", "#54C6EB", "#8EE3EF",
        "#5C6B73", "#9DB4C0", "#253237", "#456268",
        "#C2DFE3", "#88B7B5",
    ],
    "Sunset": [
        "#F4D03F", "#F0B27A", "#E59866", "#D35400",
        "#C0392B", "#922B21", "#7B241C", "#EB984E",
        "#CD6155", "#AF601A",
    ],
    # --- Monochromatic ---
    "Blues": [
        "#08306b", "#08519c", "#2171b5", "#4292c6",
        "#6baed6", "#9ecae1", "#1a5276", "#2e86c1",
        "#5dade2", "#154360",
    ],
    "Greens": [
        "#00441b", "#006d2c", "#238b45", "#41ab5d",
        "#74c476", "#a1d99b", "#145a32", "#1e8449",
        "#27ae60", "#0e6251",
    ],
    "Reds": [
        "#67000d", "#a50f15", "#cb181d", "#ef3b2c",
        "#fb6a4a", "#fc9272", "#922b21", "#c0392b",
        "#e74c3c", "#7b241c",
    ],
    "Grays": [
        "#1a1a1a", "#333333", "#4d4d4d", "#666666",
        "#808080", "#999999", "#404040", "#5c5c5c",
        "#737373", "#252525",
    ],
    "Purples": [
        "#3f007d", "#54278f", "#6a51a3", "#807dba",
        "#9e9ac8", "#bcbddc", "#4a235a", "#6c3483",
        "#8e44ad", "#a569bd",
    ],
    "Oranges": [
        "#7f2704", "#a63603", "#d94801", "#f16913",
        "#fd8d3c", "#fdae6b", "#935116", "#ca6f1e",
        "#eb984e", "#af601a",
    ],
    "Teals": [
        "#004d40", "#00695c", "#00897b", "#00acc1",
        "#26c6da", "#4dd0e1", "#0e6655", "#148f77",
        "#1abc9c", "#45b39d",
    ],
    # --- Professional ---
    "Ocean": [
        "#1B4F72", "#21618C", "#2874A6", "#2E86C1",
        "#3498DB", "#5DADE2", "#85C1E9", "#154360",
        "#1A5276", "#1F618D",
    ],
    "Forest": [
        "#0B5345", "#145A32", "#196F3D", "#1E8449",
        "#239B56", "#27AE60", "#2ECC71", "#0E6251",
        "#117A65", "#148F77",
    ],
    "Grayscale": [
        "#000000", "#1c1c1c", "#383838", "#555555",
        "#717171", "#8d8d8d", "#aaaaaa", "#2e2e2e",
        "#4a4a4a", "#676767",
    ],
    "Pastel": [
        "#AED6F1", "#A9DFBF", "#F9E79F", "#F5B7B1",
        "#D7BDE2", "#FADBD8", "#D5F5E3", "#FDEBD0",
        "#D6EAF8", "#E8DAEF",
    ],
    "Vivid": [
        "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
        "#9B59B6", "#1ABC9C", "#E67E22", "#2980B9",
        "#27AE60", "#C0392B",
    ],
}

_PALETTE_NAMES = list(_PALETTES.keys()) + ["tab10", "tab20", "viridis", "plasma"]

# Marker shape options
_MARKER_SHAPES = {
    "Star": "*",
    "Diamond": "D",
    "Circle": "o",
    "Triangle ▲": "^",
    "Triangle ▼": "v",
    "Square": "s",
    "Pentagon": "p",
    "Hexagon": "h",
    "Plus": "P",
    "Cross": "X",
}

# Named color options for peak markers
_MARKER_COLORS = {
    "Red": "#d62728",
    "Dark Red": "#8B0000",
    "Green": "#2ca02c",
    "Dark Green": "#006400",
    "Blue": "#1f77b4",
    "Dark Blue": "#00008B",
    "Black": "#000000",
    "Orange": "#ff7f0e",
    "Purple": "#9467bd",
    "Magenta": "#e377c2",
    "Cyan": "#17becf",
    "Gold": "#DAA520",
    "Teal": "#008080",
    "Coral": "#FF6F61",
    "Navy": "#001f3f",
    "Lime": "#32CD32",
    "Crimson": "#DC143C",
    "Slate": "#708090",
    "Chocolate": "#D2691E",
    "White": "#FFFFFF",
}


def _darken_color(hex_color: str, factor: float = 0.6) -> str:
    """Darken a hex color by the given factor (0 = black, 1 = unchanged)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def get_palette_colors(palette_name: str, n: int) -> list:
    """Return a list of *n* colors from the named palette."""
    if palette_name in _PALETTES:
        colors = _PALETTES[palette_name]
        return [colors[i % len(colors)] for i in range(n)]
    cmap = plt.colormaps.get_cmap(palette_name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


class _ComputeWorker(QThread):
    """Worker thread for a single HV forward computation."""

    finished = Signal(int, object, object)  # index, freqs, amps
    error = Signal(int, str)

    def __init__(self, index: int, model_path: str, config: dict):
        super().__init__()
        self._index = index
        self._model_path = model_path
        self._config = config

    def run(self):
        try:
            freqs, amps = compute_hv_curve(self._model_path, self._config)
            self.finished.emit(self._index, freqs, amps)
        except Exception as e:
            self.error.emit(self._index, str(e))


class MultiProfilePickerDialog(QDialog):
    """Dialog for processing multiple profiles with sequential peak selection."""

    results_ready = Signal(list)  # List[ProfileResult]
    report_requested = Signal(list, object)  # results, median_result_or_None

    def __init__(
        self,
        profiles: List[Tuple[str, SoilProfile]],
        freq_config: dict,
        fig_settings: FigureSettings,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Multiple Profiles — HV Forward Modeling")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        self._freq_config = freq_config
        self._fig_settings = fig_settings
        self._results: List[ProfileResult] = [
            ProfileResult(name=name, profile=prof) for name, prof in profiles
        ]
        self._current_idx = 0
        self._pick_f0 = True
        self._pick_secondary = False
        self._workers: List[_ComputeWorker] = []  # prevent GC of running threads
        self._temp_paths: Dict[int, str] = {}  # idx -> temp file path
        self._n_profiles = len(self._results)  # count of real profiles
        self._median_idx = self._n_profiles  # index for the median step
        self._median_result: Optional[ProfileResult] = None  # filled when median step loads
        self._median_skipped = False

        self._setup_ui()
        self._compute_current()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: profile list ---
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(5, 5, 5, 5)

        lbl = QLabel("Profiles:")
        lbl.setStyleSheet("font-weight: bold;")
        left_lay.addWidget(lbl)

        self.profile_list = QListWidget()
        self.profile_list.setMaximumWidth(220)
        for r in self._results:
            self.profile_list.addItem(QListWidgetItem(r.name))
        # Median step as last entry
        median_item = QListWidgetItem("── Median HV ──")
        median_item.setForeground(Qt.darkMagenta)
        self.profile_list.addItem(median_item)
        self.profile_list.currentRowChanged.connect(self._on_list_select)
        left_lay.addWidget(self.profile_list)

        # --- Plot settings (collapsible) ---
        settings_grp = QGroupBox("Plot Settings")
        settings_grp.setCheckable(True)
        settings_grp.setChecked(False)
        s_lay = QVBoxLayout(settings_grp)

        palette_row = QHBoxLayout()
        palette_row.addWidget(QLabel("Palette:"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(_PALETTE_NAMES)
        self.palette_combo.setCurrentText(self._fig_settings.color_palette)
        self.palette_combo.currentTextChanged.connect(self._on_settings_changed)
        palette_row.addWidget(self.palette_combo)
        s_lay.addLayout(palette_row)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Indiv. α:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self._fig_settings.individual_alpha)
        self.alpha_spin.valueChanged.connect(self._on_settings_changed)
        alpha_row.addWidget(self.alpha_spin)
        s_lay.addLayout(alpha_row)

        ilw_row = QHBoxLayout()
        ilw_row.addWidget(QLabel("Indiv. LW:"))
        self.ilw_spin = QDoubleSpinBox()
        self.ilw_spin.setRange(0.3, 5.0)
        self.ilw_spin.setSingleStep(0.1)
        self.ilw_spin.setValue(self._fig_settings.individual_linewidth)
        self.ilw_spin.valueChanged.connect(self._on_settings_changed)
        ilw_row.addWidget(self.ilw_spin)
        s_lay.addLayout(ilw_row)

        mlw_row = QHBoxLayout()
        mlw_row.addWidget(QLabel("Median LW:"))
        self.mlw_spin = QDoubleSpinBox()
        self.mlw_spin.setRange(0.5, 8.0)
        self.mlw_spin.setSingleStep(0.5)
        self.mlw_spin.setValue(self._fig_settings.median_linewidth)
        self.mlw_spin.valueChanged.connect(self._on_settings_changed)
        mlw_row.addWidget(self.mlw_spin)
        s_lay.addLayout(mlw_row)

        # ── Peak Markers ──
        sep = QLabel("── Peak Markers ──")
        sep.setStyleSheet("color: #555; font-size: 11px; margin-top: 4px;")
        s_lay.addWidget(sep)

        f0_row = QHBoxLayout()
        f0_row.addWidget(QLabel("f0:"))
        self.f0_color_combo = QComboBox()
        self.f0_color_combo.addItems(list(_MARKER_COLORS.keys()))
        self.f0_color_combo.setCurrentText(self._fig_settings.f0_marker_color)
        self.f0_color_combo.currentTextChanged.connect(self._on_settings_changed)
        f0_row.addWidget(self.f0_color_combo)
        self.f0_shape_combo = QComboBox()
        self.f0_shape_combo.addItems(list(_MARKER_SHAPES.keys()))
        self.f0_shape_combo.setCurrentText(self._fig_settings.f0_marker_shape)
        self.f0_shape_combo.currentTextChanged.connect(self._on_settings_changed)
        f0_row.addWidget(self.f0_shape_combo)
        s_lay.addLayout(f0_row)

        sec_row = QHBoxLayout()
        sec_row.addWidget(QLabel("Sec:"))
        self.sec_color_combo = QComboBox()
        self.sec_color_combo.addItems(list(_MARKER_COLORS.keys()))
        self.sec_color_combo.setCurrentText(self._fig_settings.secondary_marker_color)
        self.sec_color_combo.currentTextChanged.connect(self._on_settings_changed)
        sec_row.addWidget(self.sec_color_combo)
        self.sec_shape_combo = QComboBox()
        self.sec_shape_combo.addItems(list(_MARKER_SHAPES.keys()))
        self.sec_shape_combo.setCurrentText(self._fig_settings.secondary_marker_shape)
        self.sec_shape_combo.currentTextChanged.connect(self._on_settings_changed)
        sec_row.addWidget(self.sec_shape_combo)
        s_lay.addLayout(sec_row)

        left_lay.addWidget(settings_grp)

        splitter.addWidget(left)

        # --- Right: plot + controls ---
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(5, 5, 5, 5)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 14px; padding: 5px;")
        right_lay.addWidget(self.info_label)

        self.figure = Figure(figsize=(14, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_lay.addWidget(self.toolbar)
        right_lay.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        # --- Pick-mode buttons ---
        pick_row = QHBoxLayout()

        self.btn_pick_f0 = QPushButton("Select f0")
        self.btn_pick_f0.setCheckable(True)
        self.btn_pick_f0.setChecked(True)
        self.btn_pick_f0.setStyleSheet(
            "QPushButton:checked { background-color: #107c10; color: white; "
            "font-weight: bold; padding: 8px 14px; }"
            "QPushButton { padding: 8px 14px; background-color: #666; color: white; }"
        )
        self.btn_pick_f0.clicked.connect(self._toggle_pick_f0)
        pick_row.addWidget(self.btn_pick_f0)

        self.btn_pick_sec = QPushButton("Select Secondary")
        self.btn_pick_sec.setCheckable(True)
        self.btn_pick_sec.setChecked(False)
        self.btn_pick_sec.setStyleSheet(
            "QPushButton:checked { background-color: #333; color: white; "
            "font-weight: bold; padding: 8px 14px; }"
            "QPushButton { padding: 8px 14px; background-color: #666; color: white; }"
        )
        self.btn_pick_sec.clicked.connect(self._toggle_pick_secondary)
        pick_row.addWidget(self.btn_pick_sec)

        self.btn_clear_sec = QPushButton("Clear Secondary")
        self.btn_clear_sec.setStyleSheet("padding: 8px 14px;")
        self.btn_clear_sec.clicked.connect(self._clear_secondary)
        pick_row.addWidget(self.btn_clear_sec)

        pick_row.addStretch()
        right_lay.addLayout(pick_row)

        self.selection_label = QLabel("Computing…")
        self.selection_label.setStyleSheet(
            "font-size: 12px; color: #0078d4; padding: 5px; "
            "background-color: #f0f0f0; border-radius: 3px;"
        )
        right_lay.addWidget(self.selection_label)

        # --- Navigation buttons ---
        nav_row = QHBoxLayout()

        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self._go_prev)
        nav_row.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self._go_next)
        nav_row.addWidget(self.btn_next)

        nav_row.addStretch()

        self.btn_auto = QPushButton("Auto-detect")
        self.btn_auto.setToolTip("Auto-detect f0 for current profile")
        self.btn_auto.clicked.connect(self._auto_detect_current)
        nav_row.addWidget(self.btn_auto)

        self.btn_skip = QPushButton("Skip (auto + next)")
        self.btn_skip.clicked.connect(self._skip_current)
        nav_row.addWidget(self.btn_skip)

        self.btn_auto_all = QPushButton("Auto All Remaining")
        self.btn_auto_all.setToolTip("Auto-detect peaks for all un-picked profiles")
        self.btn_auto_all.clicked.connect(self._auto_all_remaining)
        nav_row.addWidget(self.btn_auto_all)

        self.btn_skip_median = QPushButton("Skip Median")
        self.btn_skip_median.setToolTip("Finish without median curve")
        self.btn_skip_median.setStyleSheet("padding: 8px 14px;")
        self.btn_skip_median.clicked.connect(self._skip_median)
        self.btn_skip_median.setVisible(False)
        nav_row.addWidget(self.btn_skip_median)

        self.chk_include_median = QCheckBox("Include Median")
        self.chk_include_median.setChecked(True)
        self.chk_include_median.setVisible(False)
        self.chk_include_median.toggled.connect(self._on_median_toggle)
        nav_row.addWidget(self.chk_include_median)

        self.btn_generate_report = QPushButton("Generate Report")
        self.btn_generate_report.setStyleSheet(
            "padding: 8px 14px; background-color: #0078d4; color: white; font-weight: bold;"
        )
        self.btn_generate_report.setToolTip("Save all outputs without closing")
        self.btn_generate_report.clicked.connect(self._generate_report)
        self.btn_generate_report.setVisible(False)
        nav_row.addWidget(self.btn_generate_report)

        nav_row.addStretch()

        self.btn_finish = QPushButton("Finish")
        self.btn_finish.setStyleSheet(
            "background-color: #107c10; color: white; padding: 8px 20px;"
        )
        self.btn_finish.clicked.connect(self._finish)
        nav_row.addWidget(self.btn_finish)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        nav_row.addWidget(self.btn_cancel)

        right_lay.addLayout(nav_row)
        splitter.addWidget(right)
        splitter.setSizes([180, 820])
        layout.addWidget(splitter)

    # -------------------------------------------------------- toggle modes
    def _toggle_pick_f0(self):
        self._pick_f0 = self.btn_pick_f0.isChecked()
        if self._pick_f0:
            self._pick_secondary = False
            self.btn_pick_sec.setChecked(False)

    def _toggle_pick_secondary(self):
        self._pick_secondary = self.btn_pick_sec.isChecked()
        if self._pick_secondary:
            self._pick_f0 = False
            self.btn_pick_f0.setChecked(False)

    def _clear_secondary(self):
        r = self._active_result()
        if r is not None:
            r.secondary_peaks.clear()
            self._replot()
            self._update_selection_label()

    def _is_median_step(self) -> bool:
        return self._current_idx == self._median_idx

    def _active_result(self) -> Optional[ProfileResult]:
        """Return the ProfileResult for current step (profile or median)."""
        if self._is_median_step():
            return self._median_result
        if 0 <= self._current_idx < self._n_profiles:
            return self._results[self._current_idx]
        return None

    # -------------------------------------------------------- navigation
    def _on_list_select(self, row: int):
        if 0 <= row <= self._median_idx:
            self._navigate_to(row)

    def _go_prev(self):
        if self._current_idx > 0:
            self._navigate_to(self._current_idx - 1)

    def _go_next(self):
        if self._current_idx < self._median_idx:
            self._navigate_to(self._current_idx + 1)

    def _navigate_to(self, idx: int):
        self._current_idx = idx
        self.profile_list.blockSignals(True)
        self.profile_list.setCurrentRow(idx)
        self.profile_list.blockSignals(False)

        if self._is_median_step():
            self._compute_median()
        else:
            r = self._results[idx]
            if not r.computed:
                self._compute_current()
            else:
                self._replot()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        self.btn_prev.setEnabled(self._current_idx > 0)
        self.btn_next.setEnabled(self._current_idx < self._median_idx)
        is_med = self._is_median_step()
        self.btn_skip_median.setVisible(is_med)
        self.btn_auto_all.setVisible(not is_med)
        self.btn_skip.setVisible(not is_med)
        self.chk_include_median.setVisible(is_med)
        self.btn_generate_report.setVisible(is_med)

        n_done = sum(1 for r in self._results if r.f0 is not None)
        if is_med:
            self.info_label.setText(
                f"Median HV Curve  |  "
                f"All profiles: {n_done}/{self._n_profiles} peaks selected"
            )
        else:
            self.info_label.setText(
                f"Profile {self._current_idx + 1} of {self._n_profiles}: "
                f"{self._results[self._current_idx].name}  |  "
                f"Peaks selected: {n_done}/{self._n_profiles}"
            )

    # -------------------------------------------------------- computation
    def _compute_current(self):
        r = self._results[self._current_idx]
        if r.computed:
            self._replot()
            self._update_selection_label()
            self._update_nav_buttons()
            return

        self.selection_label.setText("Computing HV curve…")
        self.btn_next.setEnabled(False)
        self.btn_prev.setEnabled(False)

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="multi_fwd_"
        )
        tmp.write(r.profile.to_hvf_format())
        tmp.close()
        self._temp_paths[self._current_idx] = tmp.name

        worker = _ComputeWorker(
            self._current_idx, tmp.name, self._freq_config
        )
        worker.finished.connect(self._on_compute_done)
        worker.error.connect(self._on_compute_error)
        worker.finished.connect(lambda *_: self._cleanup_worker(worker))
        worker.error.connect(lambda *_: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _cleanup_worker(self, worker: _ComputeWorker):
        """Remove finished worker from the list."""
        try:
            self._workers.remove(worker)
        except ValueError:
            pass

    def _on_compute_done(self, idx: int, freqs, amps):
        r = self._results[idx]
        r.freqs = np.asarray(freqs)
        r.amps = np.asarray(amps)
        r.computed = True

        if r.f0 is None:
            peak_idx = int(np.argmax(r.amps))
            r.f0 = (float(r.freqs[peak_idx]), float(r.amps[peak_idx]), peak_idx)

        self._update_list_item(idx)
        if idx == self._current_idx:
            self._replot()
            self._update_selection_label()
            self._update_nav_buttons()

        # Clean up temp file
        tmp_path = self._temp_paths.pop(idx, None)
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _on_compute_error(self, idx: int, msg: str):
        self.selection_label.setText(f"Error: {msg}")
        self._update_nav_buttons()
        tmp_path = self._temp_paths.pop(idx, None)
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    # -------------------------------------------------------- median computation
    def _compute_median(self):
        """Build median HV from all computed profiles and display it."""
        computed = [r for r in self._results if r.computed and r.freqs is not None]
        if len(computed) < 2:
            self.selection_label.setText("Need at least 2 computed profiles for median.")
            return

        ref_freqs = computed[0].freqs
        all_amps = np.column_stack([
            np.interp(ref_freqs, r.freqs, r.amps) for r in computed
        ])
        median_amps = np.median(all_amps, axis=1)

        if self._median_result is None:
            self._median_result = ProfileResult(
                name="Median HV", profile=computed[0].profile,
            )
        self._median_result.freqs = ref_freqs
        self._median_result.amps = median_amps
        self._median_result.computed = True

        # Auto-detect f0 if not already picked manually
        if self._median_result.f0 is None:
            peak_idx = int(np.argmax(median_amps))
            self._median_result.f0 = (
                float(ref_freqs[peak_idx]), float(median_amps[peak_idx]), peak_idx
            )

        self._replot()
        self._update_selection_label()

    def _skip_median(self):
        """Finish without including median results."""
        self._median_skipped = True
        self._median_result = None
        self.results_ready.emit(self._results)
        self.accept()

    def _on_settings_changed(self):
        """Update fig_settings from controls and replot."""
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

    def _on_median_toggle(self, checked: bool):
        """Toggle median visibility on the live plot."""
        if self._is_median_step():
            self._replot()

    def _generate_report(self):
        """Emit report signal without closing the dialog."""
        include_median = self.chk_include_median.isChecked()
        median = self._median_result if include_median else None
        self.report_requested.emit(list(self._results), median)

    def save_current_plot(self, path: str, dpi: int = 300):
        """Save the current figure view to a file."""
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight")

    # -------------------------------------------------------- peak picking
    def _on_canvas_click(self, event):
        if not (self._pick_f0 or self._pick_secondary):
            return
        if event.inaxes is None or event.button != 1:
            return
        try:
            toolbar_mode = getattr(self.toolbar, "mode", "")
            if toolbar_mode:
                return
        except Exception:
            pass
        if event.xdata is None or event.ydata is None:
            return

        r = self._active_result()
        if r is None or r.freqs is None:
            return

        click_f = event.xdata
        if click_f < r.freqs.min() or click_f > r.freqs.max():
            return

        log_freqs = np.log10(r.freqs)
        log_click = np.log10(click_f)
        nearest_idx = int(np.argmin(np.abs(log_freqs - log_click)))
        sel_f = float(r.freqs[nearest_idx])
        sel_a = float(r.amps[nearest_idx])

        if self._pick_f0:
            r.f0 = (sel_f, sel_a, nearest_idx)
        elif self._pick_secondary:
            r.secondary_peaks.append((sel_f, sel_a, nearest_idx))

        if not self._is_median_step():
            self._update_list_item(self._current_idx)
        else:
            self._update_median_list_item()
        self._replot()
        self._update_selection_label()

    def _auto_detect_current(self):
        r = self._active_result()
        if r is None or r.freqs is None:
            return
        peak_idx = int(np.argmax(r.amps))
        r.f0 = (float(r.freqs[peak_idx]), float(r.amps[peak_idx]), peak_idx)
        if not self._is_median_step():
            self._update_list_item(self._current_idx)
        else:
            self._update_median_list_item()
        self._replot()
        self._update_selection_label()

    def _skip_current(self):
        self._auto_detect_current()
        self._go_next()

    def _auto_all_remaining(self):
        """Auto-detect f0 for all profiles that haven't been computed yet, then refresh."""
        for i, r in enumerate(self._results):
            if not r.computed:
                # Compute synchronously for remaining profiles
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, prefix="multi_fwd_"
                )
                tmp.write(r.profile.to_hvf_format())
                tmp.close()
                try:
                    freqs, amps = compute_hv_curve(tmp.name, self._freq_config)
                    r.freqs = np.asarray(freqs)
                    r.amps = np.asarray(amps)
                    r.computed = True
                except Exception:
                    pass
                finally:
                    Path(tmp.name).unlink(missing_ok=True)

            if r.computed and r.f0 is None:
                peak_idx = int(np.argmax(r.amps))
                r.f0 = (float(r.freqs[peak_idx]), float(r.amps[peak_idx]), peak_idx)

            self._update_list_item(i)

        self._replot()
        self._update_selection_label()
        self._update_nav_buttons()

    # -------------------------------------------------------- plotting
    def _replot(self):
        self.figure.clear()

        if self._is_median_step():
            self._replot_median()
            return

        r = self._results[self._current_idx]

        if r.freqs is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        s = self._fig_settings
        show_vs = s.show_vs and r.profile is not None

        if show_vs:
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.25)
            ax_hv = self.figure.add_subplot(gs[0])
            ax_vs = self.figure.add_subplot(gs[1])
            self._draw_vs(ax_vs, r.profile, s)
        else:
            ax_hv = self.figure.add_subplot(111)

        self._draw_hv(ax_hv, r, s)
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

    def _replot_median(self):
        """Draw the median step: all individual curves + optional median bold."""
        s = self._fig_settings
        ax = self.figure.add_subplot(111)

        computed = [r for r in self._results if r.computed and r.freqs is not None]
        if not computed:
            ax.text(0.5, 0.5, "No computed profiles", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        show_med = self.chk_include_median.isChecked()
        n = len(computed)
        colors = get_palette_colors(s.color_palette, n)

        # When median is hidden, boost individual visibility
        alpha = s.individual_alpha if show_med else min(s.individual_alpha + 0.4, 1.0)
        lw = s.individual_linewidth if show_med else s.individual_linewidth + 0.8

        # Individual profiles
        for i, r in enumerate(computed):
            c = colors[i]
            ax.plot(r.freqs, r.amps, linewidth=lw,
                    color=c, alpha=alpha, label=r.name)
            if r.f0 is not None:
                ax.scatter(r.f0[0], r.f0[1], color=c, s=80, marker="*",
                           edgecolors="black", linewidth=0.5, zorder=4,
                           alpha=min(alpha + 0.2, 1.0))

        # Median curve — only when checkbox is on
        mr = self._median_result
        f0_c = _MARKER_COLORS.get(s.f0_marker_color, "#d62728")
        f0_m = _MARKER_SHAPES.get(s.f0_marker_shape, "*")
        sec_c = _MARKER_COLORS.get(s.secondary_marker_color, "#2ca02c")
        sec_m = _MARKER_SHAPES.get(s.secondary_marker_shape, "*")

        if show_med and mr is not None and mr.freqs is not None:
            ax.plot(mr.freqs, mr.amps, color="black",
                    linewidth=s.median_linewidth, label="Median HV", zorder=10)

            if mr.f0 is not None:
                f, a, _ = mr.f0
                ax.scatter(f, a, color=f0_c, s=s.f0_marker_size, marker=f0_m,
                           edgecolors=_darken_color(f0_c), linewidth=1.5,
                           zorder=11,
                           label=f"Median f0 = {f:.2f} Hz (A={a:.2f})")
                ax.axvline(x=f, color=f0_c, linestyle="--", alpha=0.5)

            for sec_f, sec_a, _ in mr.secondary_peaks:
                ax.scatter(sec_f, sec_a, color=sec_c, s=s.secondary_marker_size,
                           marker=sec_m,
                           edgecolors=_darken_color(sec_c), linewidth=1.5,
                           zorder=10,
                           label=f"Median Sec. ({sec_f:.2f} Hz, A={sec_a:.2f})")
                ax.axvline(x=sec_f, color=sec_c, linestyle=":", alpha=0.4)

        title = "All Profiles" if not show_med else "Median HV Curve — All Profiles"
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

    @staticmethod
    def _draw_hv(ax, r: ProfileResult, s: FigureSettings):
        """Draw HV curve with peaks."""
        f0_c = _MARKER_COLORS.get(s.f0_marker_color, "#d62728")
        f0_m = _MARKER_SHAPES.get(s.f0_marker_shape, "*")
        sec_c = _MARKER_COLORS.get(s.secondary_marker_color, "#2ca02c")
        sec_m = _MARKER_SHAPES.get(s.secondary_marker_shape, "*")

        ax.plot(r.freqs, r.amps, "b-", linewidth=2, label="H/V Ratio")

        if r.f0 is not None:
            f, a, _ = r.f0
            ax.scatter(f, a, color=f0_c, s=s.f0_marker_size, marker=f0_m,
                       edgecolors=_darken_color(f0_c), linewidth=1.5,
                       label=f"f0 = {f:.2f} Hz", zorder=5)
            ax.axvline(x=f, color=f0_c, linestyle="--", alpha=0.4)

        for sec_f, sec_a, _ in r.secondary_peaks:
            ax.scatter(sec_f, sec_a, color=sec_c, s=s.secondary_marker_size,
                       marker=sec_m,
                       edgecolors=_darken_color(sec_c), linewidth=1.5,
                       label=f"Secondary ({sec_f:.2f} Hz, A={sec_a:.2f})", zorder=4)
            ax.axvline(x=sec_f, color=sec_c, linestyle=":", alpha=0.4)

        ax.set_xlabel("Frequency (Hz)", fontsize=s.font_size)
        ax.set_ylabel("H/V Amplitude Ratio", fontsize=s.font_size)
        ax.set_title(f"HV Spectral Ratio — {r.name}", fontsize=s.title_size, fontweight="bold")
        if s.log_x:
            ax.set_xscale("log")
        if s.log_y:
            ax.set_yscale("log")
        if s.grid:
            ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=s.legend_size)
        ax.set_xlim(r.freqs[0], r.freqs[-1])

    @staticmethod
    def _draw_vs(ax, profile: SoilProfile, s: FigureSettings):
        """Draw compact Vs profile."""
        depths = [0.0]
        vs_vals = []
        for layer in profile.layers:
            vs_vals.append(layer.vs)
            if layer.is_halfspace:
                depths.append(depths[-1] + max(100, depths[-1] * 0.3))
            else:
                depths.append(depths[-1] + layer.thickness)

        plot_d, plot_v = [], []
        for i in range(len(vs_vals)):
            plot_d.extend([depths[i], depths[i + 1]])
            plot_v.extend([vs_vals[i], vs_vals[i]])

        ax.fill_betweenx(plot_d, 0, plot_v, alpha=0.3, color="teal")
        ax.step(plot_v + [plot_v[-1]], [0] + plot_d, where="pre",
                color="teal", linewidth=1.5)

        finite_layers = [la for la in profile.layers if not la.is_halfspace]
        if finite_layers:
            total = sum(la.thickness for la in finite_layers)
            ax.axhline(y=total, color="red", linestyle="-", alpha=0.6, linewidth=1.5)

        ax.set_xlabel("Vs", fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=9)
        ax.set_title(f"{len(finite_layers)}L", fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)
        if plot_v:
            ax.set_xlim(0, max(plot_v) * 1.1)

    # -------------------------------------------------------- helpers
    def _update_list_item(self, idx: int):
        item = self.profile_list.item(idx)
        r = self._results[idx]
        prefix = "✓ " if r.f0 is not None else ""
        item.setText(f"{prefix}{r.name}")

    def _update_median_list_item(self):
        item = self.profile_list.item(self._median_idx)
        mr = self._median_result
        if mr is not None and mr.f0 is not None:
            item.setText(f"✓ ── Median HV ──")
        else:
            item.setText("── Median HV ──")

    def _update_selection_label(self):
        r = self._active_result()
        if r is None:
            self.selection_label.setText("No active profile")
            return
        parts = []
        if r.f0 is not None:
            prefix = "Median " if self._is_median_step() else ""
            parts.append(f"{prefix}f0 = {r.f0[0]:.3f} Hz (A = {r.f0[1]:.2f})")
        for i, (sf, sa, _) in enumerate(r.secondary_peaks):
            parts.append(f"Sec.{i + 1} = {sf:.3f} Hz")
        if parts:
            self.selection_label.setText("  |  ".join(parts))
            self.selection_label.setStyleSheet(
                "font-size: 12px; color: #107c10; padding: 5px; font-weight: bold; "
                "background-color: #e6ffe6; border-radius: 3px;"
            )
        else:
            self.selection_label.setText("Click on the curve to select f0")
            self.selection_label.setStyleSheet(
                "font-size: 12px; color: #0078d4; padding: 5px; "
                "background-color: #f0f0f0; border-radius: 3px;"
            )

    # -------------------------------------------------------- finish / save
    def _finish(self):
        unselected = [r.name for r in self._results if r.f0 is None]
        if unselected:
            reply = QMessageBox.question(
                self, "Unselected Profiles",
                f"{len(unselected)} profiles have no peak.\n"
                "Auto-detect for these?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                self._auto_all_remaining()

        self.results_ready.emit(self._results)
        self.accept()

    def get_results(self) -> List[ProfileResult]:
        return self._results

    def get_median_result(self) -> Optional[ProfileResult]:
        """Return the median result, or None if skipped."""
        if self._median_skipped:
            return None
        return self._median_result

    def _wait_for_workers(self):
        """Wait for all running worker threads to finish."""
        for w in list(self._workers):
            if w.isRunning():
                w.wait(5000)  # 5 second timeout per worker

    def closeEvent(self, event):
        """Ensure all workers are stopped before closing."""
        self._wait_for_workers()
        # Clean up any remaining temp files
        for tmp_path in self._temp_paths.values():
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_paths.clear()
        super().closeEvent(event)

    def reject(self):
        """Override reject to wait for workers before closing."""
        self._wait_for_workers()
        super().reject()

    def accept(self):
        """Override accept to wait for workers before closing."""
        self._wait_for_workers()
        super().accept()
