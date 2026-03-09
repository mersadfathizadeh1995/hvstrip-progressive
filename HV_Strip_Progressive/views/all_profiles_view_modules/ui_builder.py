"""UI construction helpers for the All Profiles View.

``build_ui(view)`` populates the host ``QWidget`` with all sub-widgets
(canvas splitter, collapsible settings panel, bottom toolbar).  The
function stores every widget as an attribute on *view* so the rest of
the code-base can reference them directly.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    FIGURE_SIZES,
    MARKER_SHAPES,
    PALETTES,
    Y_LIMIT_METHODS,
)


def build_ui(view) -> None:
    """Create the entire widget tree and attach it to *view*.

    *view* must be a ``QWidget`` subclass that also provides:

    * ``_on_press``, ``_on_release`` — mouse event handlers
    * ``_redraw``, ``_redraw_vs``, ``_toggle_vs`` — drawing callbacks
    * ``_toggle_pick_f0``, ``_toggle_pick_sec``
    * ``_undo_last_secondary``, ``_clear_median_peaks``
    * ``_save_results``, ``_save_all_results``, ``_load_results``

    And the following imports at module level:

    * ``MatplotlibWidget``, ``CollapsibleGroupBox``
    * Style constants: ``BUTTON_SUCCESS``, ``EMOJI``
    """
    from HV_Strip_Progressive.widgets import CollapsibleGroupBox, MatplotlibWidget
    from HV_Strip_Progressive.widgets.style_constants import BUTTON_SUCCESS, EMOJI

    main = QVBoxLayout(view)
    main.setContentsMargins(2, 2, 2, 2)
    main.setSpacing(2)

    # ── Main split: HV canvas + Vs canvas ────────────────────
    _build_canvas_splitter(view, main)

    # ── Collapsible Settings Panel ────────────────────────────
    _build_settings_panel(view, main, CollapsibleGroupBox)

    # ── Bottom bar: Peak picking + Save / Load ────────────────
    _build_bottom_bar(view, main, BUTTON_SUCCESS, EMOJI)


# ── Canvas splitter ────────────────────────────────────────────

def _build_canvas_splitter(view, parent_layout) -> None:
    from HV_Strip_Progressive.widgets import MatplotlibWidget

    view._splitter = QSplitter(Qt.Horizontal)

    view._hv_plot = MatplotlibWidget(figsize=(12, 6))
    view._hv_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    view._hv_plot.canvas.mpl_connect("button_press_event", view._on_press)
    view._hv_plot.canvas.mpl_connect("button_release_event", view._on_release)
    view._splitter.addWidget(view._hv_plot)

    view._vs_panel = QWidget()
    vs_lay = QVBoxLayout(view._vs_panel)
    vs_lay.setContentsMargins(2, 2, 2, 2)
    view._vs_plot = MatplotlibWidget(figsize=(4, 6))
    view._vs_plot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    vs_lay.addWidget(view._vs_plot, 1)

    vs_opts = QHBoxLayout()
    view._chk_vs_median = QCheckBox("Median Vs")
    view._chk_vs_median.toggled.connect(view._redraw_vs)
    vs_opts.addWidget(view._chk_vs_median)
    view._chk_vs30 = QCheckBox("Vs30")
    view._chk_vs30.setChecked(True)
    view._chk_vs30.toggled.connect(view._redraw_vs)
    vs_opts.addWidget(view._chk_vs30)
    vs_lay.addLayout(vs_opts)

    view._vs_panel.setVisible(False)
    view._splitter.addWidget(view._vs_panel)
    view._splitter.setSizes([800, 250])
    parent_layout.addWidget(view._splitter, 1)


# ── Settings panel ─────────────────────────────────────────────

def _build_settings_panel(view, parent_layout, CollapsibleGroupBox) -> None:
    from HV_Strip_Progressive.widgets.style_constants import EMOJI

    view._settings_group = CollapsibleGroupBox(
        f"{EMOJI.get('settings', '⚙')} Plot Settings", collapsed=True)
    settings_lay = QVBoxLayout()
    settings_lay.setSpacing(2)
    settings_lay.setContentsMargins(2, 2, 2, 2)

    # Toggles row
    _build_toggle_row(view, settings_lay)

    # Curves sub-group
    _build_curves_group(view, settings_lay, CollapsibleGroupBox)

    # Primary Peak sub-group
    _build_primary_group(view, settings_lay, CollapsibleGroupBox)

    # Secondary Peak sub-group
    _build_secondary_group(view, settings_lay, CollapsibleGroupBox)

    # Axes & Grid sub-group
    _build_axes_group(view, settings_lay, CollapsibleGroupBox)

    # Export sub-group
    _build_export_group(view, settings_lay, CollapsibleGroupBox)

    # Labels sub-group
    _build_labels_group(view, settings_lay, CollapsibleGroupBox)

    view._settings_group.setContentLayout(settings_lay)
    parent_layout.addWidget(view._settings_group)


def _build_toggle_row(view, parent_layout) -> None:
    row = QHBoxLayout()
    row.setSpacing(6)

    view._chk_median = QCheckBox("Median")
    view._chk_median.setChecked(True)
    view._chk_median.toggled.connect(view._redraw)
    row.addWidget(view._chk_median)

    view._chk_sigma = QCheckBox("±1σ")
    view._chk_sigma.setChecked(True)
    view._chk_sigma.toggled.connect(view._redraw)
    row.addWidget(view._chk_sigma)

    view._chk_primary = QCheckBox("f0 Peaks")
    view._chk_primary.setChecked(True)
    view._chk_primary.toggled.connect(view._redraw)
    row.addWidget(view._chk_primary)

    view._chk_secondary = QCheckBox("Sec. Peaks")
    view._chk_secondary.setChecked(True)
    view._chk_secondary.toggled.connect(view._redraw)
    row.addWidget(view._chk_secondary)

    view._chk_annotations = QCheckBox("Annotations")
    view._chk_annotations.setChecked(True)
    view._chk_annotations.toggled.connect(view._redraw)
    row.addWidget(view._chk_annotations)

    view._chk_vs = QCheckBox("Vs Panel")
    view._chk_vs.toggled.connect(view._toggle_vs)
    row.addWidget(view._chk_vs)

    row.addStretch()
    parent_layout.addLayout(row)


def _build_curves_group(view, parent_layout, CollapsibleGroupBox) -> None:
    grp = CollapsibleGroupBox("Curves", collapsed=True)
    lay = QHBoxLayout()
    lay.setSpacing(4)
    lay.setContentsMargins(2, 2, 2, 2)

    lay.addWidget(QLabel("Palette:"))
    view._palette = QComboBox()
    view._palette.addItems(PALETTES)
    view._palette.setMaximumWidth(120)
    view._palette.currentIndexChanged.connect(view._redraw)
    lay.addWidget(view._palette)

    lay.addWidget(QLabel("α:"))
    view._alpha = QDoubleSpinBox()
    view._alpha.setRange(0.05, 1.0)
    view._alpha.setValue(0.5)
    view._alpha.setSingleStep(0.05)
    view._alpha.setMaximumWidth(60)
    view._alpha.valueChanged.connect(view._redraw)
    lay.addWidget(view._alpha)

    lay.addWidget(QLabel("LW:"))
    view._lw = QDoubleSpinBox()
    view._lw.setRange(0.3, 5.0)
    view._lw.setValue(1.2)
    view._lw.setMaximumWidth(60)
    view._lw.valueChanged.connect(view._redraw)
    lay.addWidget(view._lw)

    lay.addWidget(QLabel("Med LW:"))
    view._med_lw = QDoubleSpinBox()
    view._med_lw.setRange(0.5, 8.0)
    view._med_lw.setValue(3.0)
    view._med_lw.setMaximumWidth(60)
    view._med_lw.valueChanged.connect(view._redraw)
    lay.addWidget(view._med_lw)

    lay.addWidget(QLabel("±1σ α:"))
    view._sigma_alpha = QDoubleSpinBox()
    view._sigma_alpha.setRange(0.05, 0.6)
    view._sigma_alpha.setValue(0.15)
    view._sigma_alpha.setSingleStep(0.05)
    view._sigma_alpha.setMaximumWidth(60)
    view._sigma_alpha.valueChanged.connect(view._redraw)
    lay.addWidget(view._sigma_alpha)

    lay.addStretch()
    grp.setContentLayout(lay)
    parent_layout.addWidget(grp)


def _build_primary_group(view, parent_layout, CollapsibleGroupBox) -> None:
    grp = CollapsibleGroupBox("Primary Peak (f0)", collapsed=True)
    lay = QHBoxLayout()
    lay.setSpacing(4)
    lay.setContentsMargins(2, 2, 2, 2)

    lay.addWidget(QLabel("Shape:"))
    view._f0_marker = QComboBox()
    view._f0_marker.addItems(list(MARKER_SHAPES.keys()))
    view._f0_marker.setMaximumWidth(70)
    view._f0_marker.currentIndexChanged.connect(view._redraw)
    lay.addWidget(view._f0_marker)

    lay.addWidget(QLabel("Size:"))
    view._f0_size = QSpinBox()
    view._f0_size.setRange(4, 30)
    view._f0_size.setValue(14)
    view._f0_size.setMaximumWidth(55)
    view._f0_size.valueChanged.connect(view._redraw)
    lay.addWidget(view._f0_size)

    lay.addWidget(QLabel("Ann. Font:"))
    view._ann_font = QSpinBox()
    view._ann_font.setRange(6, 20)
    view._ann_font.setValue(8)
    view._ann_font.setMaximumWidth(55)
    view._ann_font.valueChanged.connect(view._redraw)
    lay.addWidget(view._ann_font)

    lay.addStretch()
    grp.setContentLayout(lay)
    parent_layout.addWidget(grp)


def _build_secondary_group(view, parent_layout, CollapsibleGroupBox) -> None:
    grp = CollapsibleGroupBox("Secondary Peaks", collapsed=True)
    lay = QHBoxLayout()
    lay.setSpacing(4)
    lay.setContentsMargins(2, 2, 2, 2)

    lay.addWidget(QLabel("Shape:"))
    view._sec_marker = QComboBox()
    view._sec_marker.addItems(list(MARKER_SHAPES.keys()))
    view._sec_marker.setCurrentIndex(1)  # ◆
    view._sec_marker.setMaximumWidth(70)
    view._sec_marker.currentIndexChanged.connect(view._redraw)
    lay.addWidget(view._sec_marker)

    lay.addWidget(QLabel("Size:"))
    view._sec_size = QSpinBox()
    view._sec_size.setRange(4, 30)
    view._sec_size.setValue(10)
    view._sec_size.setMaximumWidth(55)
    view._sec_size.valueChanged.connect(view._redraw)
    lay.addWidget(view._sec_size)

    lay.addStretch()
    grp.setContentLayout(lay)
    parent_layout.addWidget(grp)


def _build_axes_group(view, parent_layout, CollapsibleGroupBox) -> None:
    grp = CollapsibleGroupBox("Axes && Grid", collapsed=True)
    lay = QHBoxLayout()
    lay.setSpacing(4)
    lay.setContentsMargins(2, 2, 2, 2)

    view._chk_ylim_auto = QCheckBox("Y Auto")
    view._chk_ylim_auto.setChecked(True)
    view._chk_ylim_auto.toggled.connect(view._redraw)
    lay.addWidget(view._chk_ylim_auto)

    lay.addWidget(QLabel("Method:"))
    view._ylim_method = QComboBox()
    view._ylim_method.addItems(Y_LIMIT_METHODS)
    view._ylim_method.setMaximumWidth(120)
    view._ylim_method.currentIndexChanged.connect(view._redraw)
    lay.addWidget(view._ylim_method)

    view._chk_grid = QCheckBox("Grid")
    view._chk_grid.setChecked(True)
    view._chk_grid.toggled.connect(view._redraw)
    lay.addWidget(view._chk_grid)

    lay.addWidget(QLabel("α:"))
    view._grid_alpha = QDoubleSpinBox()
    view._grid_alpha.setRange(0.1, 1.0)
    view._grid_alpha.setValue(0.3)
    view._grid_alpha.setSingleStep(0.1)
    view._grid_alpha.setMaximumWidth(60)
    view._grid_alpha.valueChanged.connect(view._redraw)
    lay.addWidget(view._grid_alpha)

    lay.addStretch()
    grp.setContentLayout(lay)
    parent_layout.addWidget(grp)


def _build_export_group(view, parent_layout, CollapsibleGroupBox) -> None:
    grp = CollapsibleGroupBox("Export", collapsed=True)
    lay = QHBoxLayout()
    lay.setSpacing(4)
    lay.setContentsMargins(2, 2, 2, 2)

    lay.addWidget(QLabel("DPI:"))
    view._dpi = QSpinBox()
    view._dpi.setRange(72, 600)
    view._dpi.setValue(300)
    view._dpi.setMaximumWidth(65)
    lay.addWidget(view._dpi)

    lay.addWidget(QLabel("Format:"))
    view._export_fmt = QComboBox()
    view._export_fmt.addItems(["PNG", "PDF", "SVG"])
    view._export_fmt.setMaximumWidth(70)
    lay.addWidget(view._export_fmt)

    lay.addWidget(QLabel("Size:"))
    view._fig_size = QComboBox()
    view._fig_size.addItems(list(FIGURE_SIZES.keys()))
    view._fig_size.setMaximumWidth(140)
    lay.addWidget(view._fig_size)

    lay.addStretch()
    grp.setContentLayout(lay)
    parent_layout.addWidget(grp)


def _build_labels_group(view, parent_layout, CollapsibleGroupBox) -> None:
    grp = CollapsibleGroupBox("Labels", collapsed=True)
    lay = QVBoxLayout()
    lay.setSpacing(2)
    lay.setContentsMargins(2, 2, 2, 2)

    title_row = QHBoxLayout()
    title_row.addWidget(QLabel("Title:"))
    view._title_edit = QLineEdit("All Profiles")
    view._title_edit.editingFinished.connect(view._redraw)
    title_row.addWidget(view._title_edit, 1)
    lay.addLayout(title_row)

    axis_row = QHBoxLayout()
    axis_row.addWidget(QLabel("X:"))
    view._xlabel_edit = QLineEdit("Frequency (Hz)")
    view._xlabel_edit.editingFinished.connect(view._redraw)
    axis_row.addWidget(view._xlabel_edit, 1)
    axis_row.addWidget(QLabel("Y:"))
    view._ylabel_edit = QLineEdit("H/V Amplitude Ratio")
    view._ylabel_edit.editingFinished.connect(view._redraw)
    axis_row.addWidget(view._ylabel_edit, 1)
    lay.addLayout(axis_row)

    grp.setContentLayout(lay)
    parent_layout.addWidget(grp)


# ── Bottom bar ─────────────────────────────────────────────────

def _build_bottom_bar(view, parent_layout, BUTTON_SUCCESS, EMOJI) -> None:
    bot = QHBoxLayout()
    bot.setSpacing(4)

    view._btn_pick_f0 = QPushButton("🎯 Select f0")
    view._btn_pick_f0.setCheckable(True)
    view._btn_pick_f0.toggled.connect(view._toggle_pick_f0)
    view._btn_pick_f0.setEnabled(False)
    view._btn_pick_f0.setToolTip("Click on median curve to set primary peak")
    bot.addWidget(view._btn_pick_f0)

    view._btn_pick_sec = QPushButton("🔶 Select Secondary")
    view._btn_pick_sec.setCheckable(True)
    view._btn_pick_sec.toggled.connect(view._toggle_pick_sec)
    view._btn_pick_sec.setEnabled(False)
    view._btn_pick_sec.setToolTip("Click on median curve to add secondary peaks")
    bot.addWidget(view._btn_pick_sec)

    view._btn_undo_sec = QPushButton("↩ Undo Sec.")
    view._btn_undo_sec.clicked.connect(view._undo_last_secondary)
    view._btn_undo_sec.setToolTip("Remove last added secondary peak")
    bot.addWidget(view._btn_undo_sec)

    btn_clear_med = QPushButton("✕ Clear Peaks")
    btn_clear_med.clicked.connect(view._clear_median_peaks)
    bot.addWidget(btn_clear_med)

    bot.addStretch()

    view._btn_save = QPushButton(f"{EMOJI.get('report', '💾')} Save Results")
    view._btn_save.setStyleSheet(BUTTON_SUCCESS)
    view._btn_save.clicked.connect(view._save_results)
    view._btn_save.setEnabled(False)
    view._btn_save.setToolTip("Save only All Profiles outputs")
    bot.addWidget(view._btn_save)

    view._btn_save_all = QPushButton("⚙ Save All")
    view._btn_save_all.clicked.connect(view._save_all_results)
    view._btn_save_all.setEnabled(False)
    view._btn_save_all.setToolTip("Full re-save: per-profile + all profiles")
    bot.addWidget(view._btn_save_all)

    view._btn_load = QPushButton(f"{EMOJI.get('file', '📂')} Load Results")
    view._btn_load.clicked.connect(view._load_results)
    bot.addWidget(view._btn_load)

    parent_layout.addLayout(bot)
