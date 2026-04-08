"""Figure Studio View — embedded canvas tab for reviewing & exporting figures.

Replaces the old floating FigureStudioWindow with an inline view that uses
collapsible settings (CollapsibleGroupBox) matching the rest of the UI.
Provides 6+ figure types with per-type settings, live preview, and
save/gear export pattern.
"""
import os
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QListWidget, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QFileDialog, QStackedWidget, QSizePolicy,
    QDialog, QDialogButtonBox, QGroupBox, QLineEdit,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.collapsible_group import CollapsibleGroupBox
from ..widgets.style_constants import (
    EMOJI, BUTTON_PRIMARY, BUTTON_SUCCESS, GEAR_BUTTON, SECONDARY_LABEL,
)

# ── Figure type definitions ──────────────────────────────────
FIGURE_TYPES = [
    ("hv_overlay", f"{EMOJI.get('chart', '📈')} HV Curves Overlay"),
    ("peak_evolution", f"{EMOJI.get('peak', '⭐')} Peak Evolution"),
    ("interface_analysis", "📐 Interface Analysis"),
    ("waterfall", "🌊 Waterfall Plot"),
    ("publication", "📰 Publication Figure (2×2)"),
    ("dual_resonance", f"{EMOJI.get('dual', '🔀')} Dual-Resonance"),
]

FIGURE_SIZES = {
    "Standard (10×7)": (10, 7),
    "Large (14×10)": (14, 10),
    "Publication (12×8)": (12, 8),
    "Wide (16×6)": (16, 6),
}


class FigureStudioView(QWidget):
    """Embedded figure studio for strip mode — canvas tab."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._strip_dir = None
        self._output_dir = None
        self._reporter = None
        self._has_dr = False
        self._current_key = None
        self._active_keys = []
        self._wizard_peak_data = {}   # step_name → (freq, amp)
        self._step_names = []          # ordered step folder names
        self._build_ui()

    # ══════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left sidebar: figure list + settings ─────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(4)

        left_lay.addWidget(QLabel(
            f"<b>{EMOJI.get('figures', '🎨')} Figure Type</b>"))
        self._fig_list = QListWidget()
        self._fig_list.setMaximumHeight(180)
        self._fig_list.currentRowChanged.connect(self._on_fig_selected)
        left_lay.addWidget(self._fig_list)

        # Collapsible common settings
        self._build_common_settings(left_lay)

        # Per-figure settings stack (inside a collapsible)
        self._build_per_figure_settings(left_lay)

        left_lay.addStretch()
        left.setMinimumWidth(250)
        left.setMaximumWidth(380)
        splitter.addWidget(left)

        # ── Right: Canvas + save row ─────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(2, 2, 2, 2)
        right_lay.setSpacing(2)

        self._canvas = MatplotlibWidget(figsize=(12, 8))
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_lay.addWidget(self._canvas, 1)

        # Apply button row
        apply_row = QHBoxLayout()
        btn_apply = QPushButton(f"{EMOJI.get('run', '▶')} Apply / Refresh")
        btn_apply.setStyleSheet(BUTTON_PRIMARY)
        btn_apply.clicked.connect(self._draw_current)
        apply_row.addWidget(btn_apply)
        apply_row.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(SECONDARY_LABEL)
        apply_row.addWidget(self._status_label)
        right_lay.addLayout(apply_row)

        # Save row
        self._build_save_row(right_lay)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        outer.addWidget(splitter)

    def _build_common_settings(self, parent_lay):
        grp = CollapsibleGroupBox(
            f"{EMOJI.get('settings', '⚙')} Common Settings", collapsed=True)
        lay = QVBoxLayout()
        lay.setSpacing(2)
        lay.setContentsMargins(4, 2, 4, 2)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("DPI:"))
        self._dpi = QSpinBox()
        self._dpi.setRange(72, 600); self._dpi.setValue(200)
        self._dpi.setMaximumWidth(65)
        r1.addWidget(self._dpi)

        r1.addWidget(QLabel("Font:"))
        self._font_size = QSpinBox()
        self._font_size.setRange(6, 24); self._font_size.setValue(10)
        self._font_size.setMaximumWidth(50)
        r1.addWidget(self._font_size)
        r1.addStretch()
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Fmt:"))
        self._export_fmt = QComboBox()
        self._export_fmt.addItems(["PNG", "PDF", "SVG"])
        self._export_fmt.setMaximumWidth(65)
        r2.addWidget(self._export_fmt)

        r2.addWidget(QLabel("Size:"))
        self._fig_size = QComboBox()
        self._fig_size.addItems(list(FIGURE_SIZES.keys()))
        self._fig_size.setMaximumWidth(140)
        r2.addWidget(self._fig_size)
        r2.addStretch()
        lay.addLayout(r2)

        grp.setContentLayout(lay)
        parent_lay.addWidget(grp)

    def _build_per_figure_settings(self, parent_lay):
        grp = CollapsibleGroupBox("Figure Settings", collapsed=True)
        lay = QVBoxLayout()
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)

        self._settings_stack = QStackedWidget()
        lay.addWidget(self._settings_stack)

        grp.setContentLayout(lay)
        self._settings_grp = grp
        parent_lay.addWidget(grp)

    def _build_save_row(self, parent_lay):
        row = QHBoxLayout()
        row.setSpacing(4)

        self._btn_export = QPushButton(
            f"{EMOJI.get('export', '📤')} Export This")
        self._btn_export.clicked.connect(self._export_current)
        row.addWidget(self._btn_export)

        self._btn_csv = QPushButton(
            f"{EMOJI.get('save', '💾')} Save CSV")
        self._btn_csv.setToolTip("Export summary results as CSV")
        self._btn_csv.clicked.connect(self._export_summary_csv)
        row.addWidget(self._btn_csv)

        self._btn_save_all = QPushButton(
            f"{EMOJI.get('save', '💾')} Export All")
        self._btn_save_all.setStyleSheet(BUTTON_SUCCESS)
        self._btn_save_all.clicked.connect(self._export_all)
        row.addWidget(self._btn_save_all)

        self._btn_gear = QPushButton(EMOJI.get("settings", "⚙"))
        self._btn_gear.setFixedSize(28, 28)
        self._btn_gear.setStyleSheet(GEAR_BUTTON)
        self._btn_gear.setToolTip(
            "Choose which figures & data to export")
        self._btn_gear.clicked.connect(self._open_export_options)
        row.addWidget(self._btn_gear)

        row.addStretch()

        self._save_label = QLabel("")
        self._save_label.setStyleSheet(SECONDARY_LABEL)
        row.addWidget(self._save_label)
        parent_lay.addLayout(row)

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════
    def set_strip_data(self, strip_dir, output_dir=None,
                       has_dual_resonance=False):
        """Load stripping output and initialize the reporter.

        Parameters
        ----------
        strip_dir : str or Path
            Directory containing Step* folders with hv_curve.csv files.
        output_dir : str or Path, optional
            Default output directory for exports.
        has_dual_resonance : bool
            Whether dual-resonance data is available.
        """
        self._strip_dir = str(strip_dir)
        self._output_dir = str(output_dir) if output_dir else str(
            Path(strip_dir).parent)
        self._has_dr = has_dual_resonance
        self._init_reporter()
        self._discover_steps()
        self._populate_figure_list()

        # Auto-select first and draw
        if self._fig_list.count() > 0:
            self._fig_list.setCurrentRow(0)

    def set_wizard_peaks(self, peak_data: dict):
        """Store wizard-selected peaks for dual-resonance figures.

        Parameters
        ----------
        peak_data : dict
            Step folder name → ``(freq_Hz, amplitude)`` tuples.
            e.g. ``{"Step0_4-layer": (1.05, 3.2), "Step1_3-layer": (2.1, 2.8)}``
        """
        self._wizard_peak_data = dict(peak_data) if peak_data else {}
        # Redraw if currently viewing dual_resonance
        if self._current_key == "dual_resonance":
            self._draw_current()

    # ══════════════════════════════════════════════════════════════
    #  INTERNALS
    # ══════════════════════════════════════════════════════════════
    def _init_reporter(self):
        try:
            from ...core.report_generator import ProgressiveStrippingReporter
            self._reporter = ProgressiveStrippingReporter(
                self._strip_dir, output_dir=self._output_dir)
        except Exception as e:
            print(f"[FigureStudio] Reporter init: {e}")
            self._reporter = None

    def _discover_steps(self):
        """Find all Step* folders for the step-pair selector."""
        if not self._strip_dir:
            self._step_names = []
            return
        sp = Path(self._strip_dir)
        step_dirs = sorted(sp.glob("Step*_*"), key=lambda p: p.name)
        self._step_names = [d.name for d in step_dirs]

    def _populate_step_combos(self, deep_combo, shallow_combo):
        """Fill step-pair combo boxes with discovered step names."""
        for combo in (deep_combo, shallow_combo):
            combo.clear()
        if not self._step_names:
            return
        for name in self._step_names:
            deep_combo.addItem(name)
            shallow_combo.addItem(name)
        # Defaults: deep = first step, shallow = second step
        deep_combo.setCurrentIndex(0)
        if len(self._step_names) > 1:
            shallow_combo.setCurrentIndex(1)

    def _populate_figure_list(self):
        self._fig_list.clear()
        self._active_keys = []

        # Remove old panels from stack
        while self._settings_stack.count() > 0:
            w = self._settings_stack.widget(0)
            self._settings_stack.removeWidget(w)
            w.deleteLater()

        for key, label in FIGURE_TYPES:
            if key == "dual_resonance" and not self._has_dr:
                continue
            self._fig_list.addItem(label)
            self._active_keys.append(key)
            # Create per-figure settings panel
            panel = self._create_settings_panel(key)
            self._settings_stack.addWidget(panel)

    def _create_settings_panel(self, key):
        """Create a settings panel widget for the given figure type."""
        from PyQt5.QtWidgets import QFormLayout

        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(4, 2, 4, 2)
        form.setSpacing(2)

        PALETTES = ["Blues", "BuPu", "GnBu", "PuBu", "YlGnBu",
                     "cividis", "viridis", "plasma", "inferno", "tab10"]

        def _spin_int(lo, hi, val, step=1):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step)
            return s

        def _spin_float(lo, hi, val, step=0.1, dec=2):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val)
            s.setSingleStep(step); s.setDecimals(dec)
            return s

        def _combo(items, idx=0):
            c = QComboBox(); c.addItems(items); c.setCurrentIndex(idx)
            return c

        def _add_annotation_controls(controls, form_layout, include_style=False):
            """Add show_annotations checkbox + annotation_size + offset + style."""
            controls["show_annotations"] = QCheckBox()
            controls["show_annotations"].setChecked(True)
            form_layout.addRow("Peak Labels:", controls["show_annotations"])
            controls["annotation_size"] = _spin_int(5, 18, 8)
            form_layout.addRow("Label Size:", controls["annotation_size"])
            controls["annotation_offset_x"] = _spin_int(-30, 30, 6)
            form_layout.addRow("Label Offset X:", controls["annotation_offset_x"])
            controls["annotation_offset_y"] = _spin_int(-30, 30, 14)
            form_layout.addRow("Label Offset Y:", controls["annotation_offset_y"])
            if include_style:
                COLORS = ["black", "gray", "red", "blue", "green",
                          "darkblue", "darkred", "orange"]
                BOX_COLORS = ["#FFFFCC", "white", "#E8F0FE", "#FFF0F0",
                              "#F0FFF0", "#F5F5DC", "lightyellow", "ivory"]
                controls["arrow_color"] = _combo(COLORS)
                form_layout.addRow("Arrow Color:", controls["arrow_color"])
                controls["arrow_width"] = _spin_float(0.3, 3.0, 0.8, 0.1)
                form_layout.addRow("Arrow Width:", controls["arrow_width"])
                controls["box_color"] = _combo(BOX_COLORS)
                form_layout.addRow("Box Color:", controls["box_color"])
                controls["text_color"] = _combo(COLORS)
                form_layout.addRow("Text Color:", controls["text_color"])

        # Store controls on the widget for retrieval
        w._controls = {}

        if key == "hv_overlay":
            w._controls["log_x"] = QCheckBox(); w._controls["log_x"].setChecked(True)
            w._controls["grid"] = QCheckBox(); w._controls["grid"].setChecked(True)
            w._controls["cmap"] = _combo(PALETTES)
            w._controls["linewidth"] = _spin_float(0.5, 6, 1.5)
            w._controls["alpha"] = _spin_float(0.1, 1.0, 0.85, 0.05)
            w._controls["show_peaks"] = QCheckBox()
            w._controls["show_peaks"].setChecked(True)
            w._controls["marker_size"] = _spin_int(4, 20, 8)
            for lbl, k in [("Log X:", "log_x"), ("Grid:", "grid"),
                           ("Colormap:", "cmap"), ("Line Width:", "linewidth"),
                           ("Alpha:", "alpha"), ("Show Peaks:", "show_peaks"),
                           ("Marker Size:", "marker_size")]:
                form.addRow(lbl, w._controls[k])
            _add_annotation_controls(w._controls, form, include_style=True)

        elif key == "peak_evolution":
            w._controls["grid"] = QCheckBox(); w._controls["grid"].setChecked(True)
            w._controls["show_fill"] = QCheckBox()
            w._controls["show_fill"].setChecked(True)
            w._controls["marker_size"] = _spin_int(4, 20, 8)
            w._controls["linewidth"] = _spin_float(0.5, 6, 1.5)
            for lbl, k in [("Grid:", "grid"), ("Show Fill:", "show_fill"),
                           ("Marker Size:", "marker_size"),
                           ("Line Width:", "linewidth")]:
                form.addRow(lbl, w._controls[k])
            _add_annotation_controls(w._controls, form)

        elif key == "interface_analysis":
            w._controls["grid"] = QCheckBox(); w._controls["grid"].setChecked(True)
            w._controls["marker_size"] = _spin_int(4, 20, 8)
            w._controls["linewidth"] = _spin_float(0.5, 6, 1.5)
            w._controls["annot_font"] = _spin_int(6, 18, 8)
            for lbl, k in [("Grid:", "grid"), ("Marker Size:", "marker_size"),
                           ("Line Width:", "linewidth"),
                           ("Annotation Font:", "annot_font")]:
                form.addRow(lbl, w._controls[k])

        elif key == "waterfall":
            w._controls["log_x"] = QCheckBox(); w._controls["log_x"].setChecked(True)
            w._controls["grid"] = QCheckBox(); w._controls["grid"].setChecked(True)
            w._controls["cmap"] = _combo(PALETTES)
            w._controls["linewidth"] = _spin_float(0.5, 6, 1.5)
            w._controls["alpha"] = _spin_float(0.1, 1.0, 0.85, 0.05)
            w._controls["offset_factor"] = _spin_float(0.5, 5.0, 1.0, 0.1)
            w._controls["normalize"] = QCheckBox()
            w._controls["normalize"].setChecked(True)
            for lbl, k in [("Log X:", "log_x"), ("Grid:", "grid"),
                           ("Colormap:", "cmap"), ("Line Width:", "linewidth"),
                           ("Alpha:", "alpha"), ("Offset Factor:", "offset_factor"),
                           ("Normalize:", "normalize")]:
                form.addRow(lbl, w._controls[k])
            _add_annotation_controls(w._controls, form)

        elif key == "publication":
            w._controls["grid"] = QCheckBox(); w._controls["grid"].setChecked(True)
            w._controls["cmap"] = _combo(PALETTES)
            w._controls["linewidth"] = _spin_float(0.5, 6, 1.5)
            w._controls["alpha"] = _spin_float(0.1, 1.0, 0.85, 0.05)
            w._controls["table_font"] = _spin_int(6, 16, 8)
            for lbl, k in [("Grid:", "grid"), ("Colormap:", "cmap"),
                           ("Line Width:", "linewidth"), ("Alpha:", "alpha"),
                           ("Table Font:", "table_font")]:
                form.addRow(lbl, w._controls[k])
            _add_annotation_controls(w._controls, form)

        elif key == "dual_resonance":
            # Step pair selectors
            w._controls["deep_step"] = QComboBox()
            w._controls["shallow_step"] = QComboBox()
            self._populate_step_combos(
                w._controls["deep_step"], w._controls["shallow_step"])
            form.addRow("Deep Step (f₀):", w._controls["deep_step"])
            form.addRow("Shallow Step (f₁):", w._controls["shallow_step"])

            w._controls["grid"] = QCheckBox(); w._controls["grid"].setChecked(True)
            w._controls["linewidth"] = _spin_float(0.5, 6, 1.5)
            w._controls["f0_dx"] = _spin_float(-5, 5, 0, 0.1)
            w._controls["f0_dy"] = _spin_float(-10, 10, 0, 0.5)
            w._controls["f1_dx"] = _spin_float(-5, 5, 0, 0.1)
            w._controls["f1_dy"] = _spin_float(-10, 10, 0, 0.5)
            w._controls["show_stripped"] = QCheckBox()
            w._controls["show_stripped"].setChecked(True)
            w._controls["hs_ratio"] = _spin_float(0.1, 1.0, 0.25, 0.05)
            for lbl, k in [("Grid:", "grid"), ("Line Width:", "linewidth"),
                           ("f0 Off X:", "f0_dx"), ("f0 Off Y:", "f0_dy"),
                           ("f1 Off X:", "f1_dx"), ("f1 Off Y:", "f1_dy"),
                           ("Show Stripped:", "show_stripped"),
                           ("HS Depth %:", "hs_ratio")]:
                form.addRow(lbl, w._controls[k])

        return w

    def _get_panel_kwargs(self, panel_widget):
        """Extract kwargs from a settings panel's controls."""
        kw = {}
        for name, ctrl in getattr(panel_widget, "_controls", {}).items():
            if isinstance(ctrl, QCheckBox):
                kw[name] = ctrl.isChecked()
            elif isinstance(ctrl, (QSpinBox, QDoubleSpinBox)):
                kw[name] = ctrl.value()
            elif isinstance(ctrl, QComboBox):
                kw[name] = ctrl.currentText()

        # Special handling for dual_resonance offsets
        if "f0_dx" in kw and "f0_dy" in kw:
            kw["f0_offset"] = (kw.pop("f0_dx"), kw.pop("f0_dy"))
        if "f1_dx" in kw and "f1_dy" in kw:
            kw["f1_offset"] = (kw.pop("f1_dx"), kw.pop("f1_dy"))

        return kw

    def _on_fig_selected(self, row):
        if 0 <= row < len(self._active_keys):
            self._current_key = self._active_keys[row]
            self._settings_stack.setCurrentIndex(row)
            self._draw_current()

    # ══════════════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════════════
    def _draw_current(self):
        if not self._current_key or not self._reporter:
            fig = self._canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Run stripping workflow first\n"
                    "to populate the Figure Studio",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, color="gray")
            self._canvas.refresh()
            return

        # Get settings
        idx = self._settings_stack.currentIndex()
        panel = self._settings_stack.widget(idx) if idx >= 0 else None
        kw = self._get_panel_kwargs(panel) if panel else {}
        kw["font_size"] = self._font_size.value()
        kw["dpi"] = self._dpi.value()

        fig = self._canvas.figure
        fig.clear()

        try:
            self._draw_figure(self._current_key, fig, kw)
            self._status_label.setText(f"✓ {self._current_key}")
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                    color="red", transform=ax.transAxes, fontsize=11)
            self._status_label.setText(f"Error: {e}")

        self._canvas.refresh()

    def _draw_figure(self, key, fig, kw):
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
                from ...visualization.resonance_plots import (
                    draw_resonance_separation)

                # Resolve step pair from combo boxes
                step_pair = self._resolve_step_pair(kw)

                draw_resonance_separation(
                    self._strip_dir, fig,
                    peak_overrides=self._wizard_peak_data or None,
                    step_pair=step_pair,
                    **kw,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Dual-resonance plot failed: {e}") from e

    def _resolve_step_pair(self, kw):
        """Convert step combo selections to a ``(deep_idx, shallow_idx)`` tuple."""
        deep_name = kw.pop("deep_step", None)
        shallow_name = kw.pop("shallow_step", None)
        if not deep_name or not shallow_name or not self._step_names:
            return None
        try:
            deep_idx = self._step_names.index(deep_name)
            shallow_idx = self._step_names.index(shallow_name)
            return (deep_idx, shallow_idx)
        except ValueError:
            return None

    # ══════════════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════════════
    def _export_current(self):
        if not self._current_key:
            return
        fmt = self._export_fmt.currentText().lower()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure",
            f"{self._current_key}.{fmt}",
            f"{fmt.upper()} (*.{fmt});;PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            dpi = self._dpi.value()
            self._canvas.figure.savefig(path, dpi=dpi, bbox_inches="tight")
            self._save_label.setText(f"Exported: {Path(path).name}")

    def _export_all(self):
        d = QFileDialog.getExistingDirectory(self, "Export All Figures To")
        if not d:
            return
        dpi = self._dpi.value()
        fmt = self._export_fmt.currentText().lower()
        saved = []
        for i, key in enumerate(self._active_keys):
            panel = self._settings_stack.widget(i)
            kw = self._get_panel_kwargs(panel) if panel else {}
            kw["font_size"] = self._font_size.value()
            kw["dpi"] = dpi

            fig = self._canvas.figure
            fig.clear()
            try:
                self._draw_figure(key, fig, kw)
                fig.savefig(os.path.join(d, f"{key}.{fmt}"),
                            dpi=dpi, bbox_inches="tight")
                if fmt != "pdf":
                    fig.savefig(os.path.join(d, f"{key}.pdf"),
                                bbox_inches="tight")
                saved.append(key)
            except Exception:
                pass

        self._save_label.setText(
            f"Exported {len(saved)}/{len(self._active_keys)} → {Path(d).name}/")

    def _export_summary_csv(self):
        """Quick-export summary results as CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary CSV", "strip_summary.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        try:
            self._write_summary_csv(path)
            self._save_label.setText(f"CSV saved: {Path(path).name}")
        except Exception as e:
            self._save_label.setText(f"CSV error: {e}")

    def _write_summary_csv(self, path):
        """Write step results summary to CSV."""
        if not self._reporter:
            return
        import csv
        steps_data = self._reporter.step_data
        if not steps_data:
            return
        vs_data = getattr(self._reporter, '_vs_data', {})
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Layers", "f0 (Hz)", "Amplitude",
                            "Vs30", "VsAvg", "Bedrock_Depth", "Status"])
            for sd in steps_data:
                name = sd.get("name", "")
                n_layers = sd.get("n_finite_layers", "?")
                hv = sd.get("hv_data", {})
                pf = hv.get("peak_frequency", "")
                pa = hv.get("peak_amplitude", "")
                vs_info = vs_data.get(name, {})
                vs30 = vs_info.get("vs30")
                vsavg = vs_info.get("vsavg")
                bd = vs_info.get("bedrock_depth")
                writer.writerow([name, n_layers,
                                f"{pf:.4f}" if pf else "",
                                f"{pa:.3f}" if pa else "",
                                f"{vs30:.0f}" if vs30 else "",
                                f"{vsavg:.0f}" if vsavg else "",
                                f"{bd:.2f}" if bd else "",
                                "OK"])

    def _write_peak_data_csv(self, path):
        """Write detailed peak data (primary + secondary) to CSV."""
        if not self._reporter:
            return
        import csv
        steps_data = self._reporter.step_data
        if not steps_data:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Peak_Type", "Frequency (Hz)",
                            "Amplitude", "Index"])
            for sd in steps_data:
                name = sd.get("name", "")
                hv = sd.get("hv_data", {})
                pf = hv.get("peak_frequency")
                pa = hv.get("peak_amplitude")
                if pf:
                    writer.writerow([name, "primary",
                                    f"{pf:.4f}", f"{pa:.3f}" if pa else "",
                                    ""])

    def _open_export_options(self):
        """Open dialog to choose which figures & data to export."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Options")
        dlg.setMinimumWidth(400)
        lay = QVBoxLayout(dlg)

        # Output directory
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:"))
        dir_edit = QLineEdit()
        if self._output_dir:
            dir_edit.setText(self._output_dir)
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
        fig_grp = QGroupBox("Figures to Export")
        fig_lay = QVBoxLayout()
        chk_map = {}
        for key, label in FIGURE_TYPES:
            if key == "dual_resonance" and not self._has_dr:
                continue
            chk = QCheckBox(label.replace(
                EMOJI.get('chart', '📈'), '').replace(
                EMOJI.get('peak', '⭐'), '').replace(
                EMOJI.get('dual', '🔀'), '').strip())
            chk.setChecked(True)
            fig_lay.addWidget(chk)
            chk_map[key] = chk
        fig_grp.setLayout(fig_lay)
        lay.addWidget(fig_grp)

        # Data export group
        data_grp = QGroupBox("Data Export")
        data_lay = QVBoxLayout()
        chk_summary_csv = QCheckBox("Summary CSV (step results, f0, Vs30)")
        chk_summary_csv.setChecked(True)
        data_lay.addWidget(chk_summary_csv)
        chk_peak_csv = QCheckBox("Peak Data CSV (primary + secondary peaks)")
        chk_peak_csv.setChecked(True)
        data_lay.addWidget(chk_peak_csv)
        chk_report_txt = QCheckBox("Full Report Text (comprehensive report)")
        chk_report_txt.setChecked(False)
        data_lay.addWidget(chk_report_txt)
        data_grp.setLayout(data_lay)
        lay.addWidget(data_grp)

        # Quality options
        qual_row = QHBoxLayout()
        chk_pdf = QCheckBox("Also export PDF")
        chk_pdf.setChecked(True)
        qual_row.addWidget(chk_pdf)
        chk_hires = QCheckBox("High-res (300 DPI)")
        chk_hires.setChecked(False)
        qual_row.addWidget(chk_hires)
        lay.addLayout(qual_row)

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

        dpi = 300 if chk_hires.isChecked() else self._dpi.value()
        fmt = self._export_fmt.currentText().lower()
        saved = []

        # Export figures
        for i, key in enumerate(self._active_keys):
            if key not in chk_map or not chk_map[key].isChecked():
                continue
            panel = self._settings_stack.widget(i)
            kw = self._get_panel_kwargs(panel) if panel else {}
            kw["font_size"] = self._font_size.value()
            kw["dpi"] = dpi

            fig = self._canvas.figure
            fig.clear()
            try:
                self._draw_figure(key, fig, kw)
                fig.savefig(str(out / f"{key}.{fmt}"),
                            dpi=dpi, bbox_inches="tight")
                if chk_pdf.isChecked() and fmt != "pdf":
                    fig.savefig(str(out / f"{key}.pdf"),
                                bbox_inches="tight")
                saved.append(key)
            except Exception:
                pass

        # Export data files
        data_saved = []
        if chk_summary_csv.isChecked():
            try:
                self._write_summary_csv(str(out / "strip_summary.csv"))
                data_saved.append("summary.csv")
            except Exception:
                pass
        if chk_peak_csv.isChecked():
            try:
                self._write_peak_data_csv(str(out / "peak_data.csv"))
                data_saved.append("peak_data.csv")
            except Exception:
                pass
        if chk_report_txt.isChecked() and self._reporter:
            try:
                report_path = out / "comprehensive_report.txt"
                self._reporter.generate_text_report(str(report_path))
                data_saved.append("report.txt")
            except Exception:
                pass

        total = len(saved) + len(data_saved)
        self._save_label.setText(
            f"Exported {len(saved)} figures + {len(data_saved)} data → {out.name}/")
        if self._mw:
            self._mw.log(
                f"Figure Studio: exported {total} items to {out}")
