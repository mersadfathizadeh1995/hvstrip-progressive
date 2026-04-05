"""Figures / Visualization Page — 5-tab plot studio.

Layout:
  Left panel — Data sources, Figure style, Axis settings, Export
  Right panel — 5 tabs: HV Curve / Vs Profile / HV Overlay / Peak Evolution / Dual-Resonance
"""
import os
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTabWidget,
    QGroupBox, QFormLayout, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QFileDialog,
    QScrollArea, QMessageBox, QColorDialog,
)

from ..widgets.plot_widget import MatplotlibWidget
from ..widgets.style_constants import (
    OUTER_MARGINS, SECONDARY_LABEL, BUTTON_PRIMARY, EMOJI,
)


class VisualizationPage(QWidget):
    """Figures page with 5 plot tabs and full style controls."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._hv_data = None
        self._model_data = None
        self._results_dir = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(*OUTER_MARGINS)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # LEFT PANEL
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(380)
        left_w = QWidget()
        left_layout = QVBoxLayout(left_w)

        # Data Sources
        ds_grp = QGroupBox(f"{EMOJI['file']} Data Sources")
        ds_form = QFormLayout(ds_grp)
        self._hv_csv_edit = QLineEdit(); self._hv_csv_edit.setPlaceholderText("HV curve CSV file")
        btn_hv = QPushButton("...")
        btn_hv.setFixedWidth(30)
        btn_hv.clicked.connect(lambda: self._browse_file(self._hv_csv_edit, "CSV Files (*.csv);;All (*)"))
        r1 = QHBoxLayout(); r1.addWidget(self._hv_csv_edit); r1.addWidget(btn_hv)
        ds_form.addRow("HV CSV:", r1)

        self._model_edit = QLineEdit(); self._model_edit.setPlaceholderText("Model/profile file")
        btn_model = QPushButton("...")
        btn_model.setFixedWidth(30)
        btn_model.clicked.connect(lambda: self._browse_file(self._model_edit, "Model Files (*.txt);;All (*)"))
        r2 = QHBoxLayout(); r2.addWidget(self._model_edit); r2.addWidget(btn_model)
        ds_form.addRow("Model File:", r2)

        self._results_edit = QLineEdit(); self._results_edit.setPlaceholderText("Results directory")
        btn_res = QPushButton("...")
        btn_res.setFixedWidth(30)
        btn_res.clicked.connect(self._browse_results_dir)
        r3 = QHBoxLayout(); r3.addWidget(self._results_edit); r3.addWidget(btn_res)
        ds_form.addRow("Results Dir:", r3)
        left_layout.addWidget(ds_grp)

        btn_load = QPushButton(f"{EMOJI['run']} Load Data")
        btn_load.setStyleSheet(BUTTON_PRIMARY)
        btn_load.clicked.connect(self._load_data)
        left_layout.addWidget(btn_load)

        # Figure Style
        style_grp = QGroupBox(f"{EMOJI['chart']} Figure Style")
        style_form = QFormLayout(style_grp)
        self._dpi = QSpinBox(); self._dpi.setRange(72, 600); self._dpi.setValue(150)
        self._fig_width = QDoubleSpinBox(); self._fig_width.setRange(4, 24); self._fig_width.setValue(10)
        self._fig_height = QDoubleSpinBox(); self._fig_height.setRange(3, 16); self._fig_height.setValue(6)
        self._font_size = QSpinBox(); self._font_size.setRange(6, 24); self._font_size.setValue(12)
        self._palette = QComboBox()
        self._palette.addItems([
            "tab10", "Set1", "Set2", "Dark2", "Pastel1", "viridis", "plasma",
            "inferno", "magma", "cividis", "coolwarm", "Spectral"])
        style_form.addRow("DPI:", self._dpi)
        style_form.addRow("Width (in):", self._fig_width)
        style_form.addRow("Height (in):", self._fig_height)
        style_form.addRow("Font Size:", self._font_size)
        style_form.addRow("Palette:", self._palette)

        self._chk_grid = QCheckBox("Show Grid"); self._chk_grid.setChecked(True)
        self._chk_legend = QCheckBox("Show Legend"); self._chk_legend.setChecked(True)
        style_form.addRow(self._chk_grid)
        style_form.addRow(self._chk_legend)

        self._line_width = QDoubleSpinBox(); self._line_width.setRange(0.5, 5); self._line_width.setValue(1.5)
        style_form.addRow("Line Width:", self._line_width)

        self._hv_color_btn = QPushButton("HV Color")
        self._hv_color_btn.setStyleSheet("background-color: #1f77b4; color: white;")
        self._hv_color = "#1f77b4"
        self._hv_color_btn.clicked.connect(lambda: self._pick_color("hv"))
        style_form.addRow("HV Color:", self._hv_color_btn)

        self._vs_color_btn = QPushButton("Vs Color")
        self._vs_color_btn.setStyleSheet("background-color: #ff7f0e; color: white;")
        self._vs_color = "#ff7f0e"
        self._vs_color_btn.clicked.connect(lambda: self._pick_color("vs"))
        style_form.addRow("Vs Color:", self._vs_color_btn)

        left_layout.addWidget(style_grp)

        # Axis Settings
        axis_grp = QGroupBox(f"{EMOJI['frequency']} Axis Settings")
        axis_form = QFormLayout(axis_grp)
        self._x_scale = QComboBox(); self._x_scale.addItems(["log", "linear"]); self._x_scale.setCurrentIndex(0)
        self._y_scale = QComboBox(); self._y_scale.addItems(["linear", "log"]); self._y_scale.setCurrentIndex(0)
        self._freq_min = QDoubleSpinBox(); self._freq_min.setRange(0.01, 10); self._freq_min.setValue(0.2)
        self._freq_max = QDoubleSpinBox(); self._freq_max.setRange(1, 100); self._freq_max.setValue(20.0)
        axis_form.addRow("X Scale:", self._x_scale)
        axis_form.addRow("Y Scale:", self._y_scale)
        axis_form.addRow("Freq Min:", self._freq_min)
        axis_form.addRow("Freq Max:", self._freq_max)
        left_layout.addWidget(axis_grp)

        # Export
        export_grp = QGroupBox(f"{EMOJI['save']} Export")
        export_layout = QVBoxLayout(export_grp)
        self._export_fmt = QComboBox()
        self._export_fmt.addItems(["PNG", "PDF", "SVG", "EPS"])
        export_layout.addWidget(self._export_fmt)
        btn_row = QHBoxLayout()
        btn_export = QPushButton("Export Current")
        btn_export.clicked.connect(self._export_current)
        btn_export_all = QPushButton("Export All Tabs")
        btn_export_all.clicked.connect(self._export_all)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_export_all)
        export_layout.addLayout(btn_row)
        left_layout.addWidget(export_grp)

        left_layout.addStretch()
        left_scroll.setWidget(left_w)
        splitter.addWidget(left_scroll)

        # RIGHT PANEL — 5 plot tabs
        right_w = QWidget()
        right_layout = QVBoxLayout(right_w)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._plot_tabs = QTabWidget()
        self._plots = {}
        tab_names = ["HV Curve", "Vs Profile", "HV Overlay", "Peak Evolution", "Dual-Resonance"]
        for name in tab_names:
            pw = MatplotlibWidget(figsize=(10, 6))
            self._plots[name] = pw
            self._plot_tabs.addTab(pw, name)
        right_layout.addWidget(self._plot_tabs)

        # Control row
        ctrl = QHBoxLayout()
        btn_refresh = QPushButton("Refresh All")
        btn_refresh.clicked.connect(self._refresh_all)
        btn_dual = QPushButton("Run Dual-Resonance")
        btn_dual.clicked.connect(self._run_dual_resonance)
        btn_export_dr = QPushButton("Export Dual-Resonance")
        btn_export_dr.clicked.connect(self._export_dual_resonance)
        ctrl.addWidget(btn_refresh)
        ctrl.addStretch()
        ctrl.addWidget(btn_dual)
        ctrl.addWidget(btn_export_dr)
        right_layout.addLayout(ctrl)

        splitter.addWidget(right_w)
        splitter.setSizes([350, 650])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ═══════════════════════════════════════════════════════════
    #  DATA LOADING
    # ═══════════════════════════════════════════════════════════
    def _load_data(self):
        hv_path = self._hv_csv_edit.text().strip()
        model_path = self._model_edit.text().strip()
        results_dir = self._results_edit.text().strip()

        loaded = []
        if hv_path and os.path.isfile(hv_path):
            try:
                data = np.loadtxt(hv_path, delimiter=",", skiprows=1)
                if data.shape[1] >= 2:
                    self._hv_data = {"freq": data[:, 0], "amp": data[:, 1]}
                    loaded.append("HV CSV")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load HV CSV: {e}")

        if model_path and os.path.isfile(model_path):
            try:
                from ...core.soil_profile import SoilProfile
                self._model_data = SoilProfile.from_auto(model_path)
                loaded.append("Model")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load model: {e}")

        if results_dir and os.path.isdir(results_dir):
            self._results_dir = results_dir
            loaded.append("Results dir")

        if loaded:
            self._refresh_all()
            if self._main_window:
                self._main_window.set_status(f"Loaded: {', '.join(loaded)}")
        else:
            QMessageBox.information(self, "No Data", "No data was loaded. Please check file paths.")

    # ═══════════════════════════════════════════════════════════
    #  PLOTTING
    # ═══════════════════════════════════════════════════════════
    def _get_style(self):
        return {
            "dpi": self._dpi.value(),
            "width": self._fig_width.value(),
            "height": self._fig_height.value(),
            "font_size": self._font_size.value(),
            "palette": self._palette.currentText(),
            "grid": self._chk_grid.isChecked(),
            "legend": self._chk_legend.isChecked(),
            "x_scale": self._x_scale.currentText(),
            "y_scale": self._y_scale.currentText(),
            "line_width": self._line_width.value(),
            "hv_color": self._hv_color,
            "vs_color": self._vs_color,
            "freq_min": self._freq_min.value(),
            "freq_max": self._freq_max.value(),
        }

    def _refresh_all(self):
        style = self._get_style()
        self._plot_hv_curve(style)
        self._plot_vs_profile(style)
        self._plot_hv_overlay(style)
        self._plot_peak_evolution(style)
        self._plot_dual_resonance(style)

    def _plot_hv_curve(self, style):
        pw = self._plots["HV Curve"]
        fig = pw.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        if self._hv_data is not None:
            ax.plot(self._hv_data["freq"], self._hv_data["amp"],
                    color=style["hv_color"], linewidth=style["line_width"], label="H/V")
        ax.set_xlabel("Frequency (Hz)", fontsize=style["font_size"])
        ax.set_ylabel("H/V Amplitude", fontsize=style["font_size"])
        ax.set_title("HVSR Curve", fontsize=style["font_size"] + 2)
        ax.set_xscale(style["x_scale"])
        ax.set_yscale(style["y_scale"])
        ax.set_xlim(style["freq_min"], style["freq_max"])
        if style["grid"]: ax.grid(True, alpha=0.3)
        if style["legend"]: ax.legend(fontsize=style["font_size"] - 2)
        fig.tight_layout()
        pw.refresh()

    def _plot_vs_profile(self, style):
        pw = self._plots["Vs Profile"]
        fig = pw.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        if self._model_data is not None:
            depths, vs_vals = [], []
            cum_depth = 0
            for layer in self._model_data.layers:
                depths.extend([cum_depth, cum_depth + layer.thickness])
                vs_vals.extend([layer.vs, layer.vs])
                cum_depth += layer.thickness
            ax.step(vs_vals, depths, where="post", color=style["vs_color"],
                    linewidth=style["line_width"], label="Vs")
            ax.invert_yaxis()
        ax.set_xlabel("Vs (m/s)", fontsize=style["font_size"])
        ax.set_ylabel("Depth (m)", fontsize=style["font_size"])
        ax.set_title("Vs Profile", fontsize=style["font_size"] + 2)
        if style["grid"]: ax.grid(True, alpha=0.3)
        if style["legend"]: ax.legend(fontsize=style["font_size"] - 2)
        fig.tight_layout()
        pw.refresh()

    def _plot_hv_overlay(self, style):
        pw = self._plots["HV Overlay"]
        fig = pw.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        if self._results_dir and os.path.isdir(self._results_dir):
            import glob
            csv_files = sorted(glob.glob(os.path.join(self._results_dir, "**/hv_curve*.csv"), recursive=True))
            import matplotlib.cm as cm
            cmap = cm.get_cmap(style["palette"])
            for i, csv_path in enumerate(csv_files):
                try:
                    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                    label = os.path.basename(os.path.dirname(csv_path))
                    color = cmap(i / max(1, len(csv_files) - 1))
                    ax.plot(data[:, 0], data[:, 1], color=color, linewidth=style["line_width"],
                            alpha=0.7, label=label)
                except Exception:
                    continue
        ax.set_xlabel("Frequency (Hz)", fontsize=style["font_size"])
        ax.set_ylabel("H/V Amplitude", fontsize=style["font_size"])
        ax.set_title("HV Overlay (All Steps)", fontsize=style["font_size"] + 2)
        ax.set_xscale(style["x_scale"])
        ax.set_xlim(style["freq_min"], style["freq_max"])
        if style["grid"]: ax.grid(True, alpha=0.3)
        if style["legend"]: ax.legend(fontsize=max(6, style["font_size"] - 4), ncol=2)
        fig.tight_layout()
        pw.refresh()

    def _plot_peak_evolution(self, style):
        pw = self._plots["Peak Evolution"]
        fig = pw.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        if self._results_dir and os.path.isdir(self._results_dir):
            import glob
            peak_files = sorted(glob.glob(os.path.join(self._results_dir, "**/peak_info.txt"), recursive=True))
            steps, freqs, amps = [], [], []
            for i, pf in enumerate(peak_files):
                try:
                    with open(pf) as f:
                        for line in f:
                            if "f0" in line.lower() and "=" in line:
                                val = float(line.split("=")[1].strip().split()[0])
                                steps.append(i)
                                freqs.append(val)
                            if "amplitude" in line.lower() and "=" in line:
                                val = float(line.split("=")[1].strip().split()[0])
                                amps.append(val)
                except Exception:
                    continue
            if freqs:
                ax.plot(steps, freqs, "o-", color=style["hv_color"], linewidth=style["line_width"],
                        markersize=6, label="f0 (Hz)")
                ax.set_ylabel("Frequency (Hz)", fontsize=style["font_size"])
                if amps and len(amps) == len(freqs):
                    ax2 = ax.twinx()
                    ax2.plot(steps, amps, "s--", color=style["vs_color"], linewidth=style["line_width"],
                             markersize=5, label="Amplitude")
                    ax2.set_ylabel("Amplitude", fontsize=style["font_size"])
        ax.set_xlabel("Stripping Step", fontsize=style["font_size"])
        ax.set_title("Peak Evolution", fontsize=style["font_size"] + 2)
        if style["grid"]: ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pw.refresh()

    def _plot_dual_resonance(self, style):
        pw = self._plots["Dual-Resonance"]
        fig = pw.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Load results and click 'Run Dual-Resonance'",
                ha="center", va="center", fontsize=12, color="gray",
                transform=ax.transAxes)
        ax.set_title("Dual-Resonance Analysis", fontsize=style.get("font_size", 12) + 2)
        fig.tight_layout()
        pw.refresh()

    def _run_dual_resonance(self):
        if not self._results_dir:
            QMessageBox.warning(self, "Error", "Please load a results directory first.")
            return
        try:
            from ...core.dual_resonance import extract_dual_resonance
            result = extract_dual_resonance(self._results_dir)
            style = self._get_style()
            pw = self._plots["Dual-Resonance"]
            fig = pw.get_figure()
            fig.clear()
            ax = fig.add_subplot(111)
            if result and "steps" in result:
                steps = result["steps"]
                f0s = result.get("f0_values", [])
                f1s = result.get("f1_values", [])
                if f0s:
                    ax.plot(steps, f0s, "o-", label="f0", color=style["hv_color"],
                            linewidth=style["line_width"])
                if f1s:
                    ax.plot(steps, f1s, "s--", label="f1", color=style["vs_color"],
                            linewidth=style["line_width"])
                ax.set_xlabel("Step", fontsize=style["font_size"])
                ax.set_ylabel("Frequency (Hz)", fontsize=style["font_size"])
                if style["legend"]: ax.legend()
            ax.set_title("Dual-Resonance Analysis", fontsize=style["font_size"] + 2)
            if style["grid"]: ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pw.refresh()
            self._plot_tabs.setCurrentWidget(pw)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Dual-resonance failed: {e}")

    def _export_dual_resonance(self):
        if not self._results_dir:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Dual-Resonance", "", "CSV (*.csv);;All (*)")
        if not path:
            return
        try:
            from ...core.dual_resonance import extract_dual_resonance
            result = extract_dual_resonance(self._results_dir)
            if result and "steps" in result:
                with open(path, "w") as f:
                    f.write("step,f0,f1\n")
                    for i, s in enumerate(result["steps"]):
                        f0 = result.get("f0_values", [None])[i] if i < len(result.get("f0_values", [])) else ""
                        f1 = result.get("f1_values", [None])[i] if i < len(result.get("f1_values", [])) else ""
                        f.write(f"{s},{f0},{f1}\n")
                QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")

    # ═══════════════════════════════════════════════════════════
    #  EXPORT
    # ═══════════════════════════════════════════════════════════
    def _export_current(self):
        idx = self._plot_tabs.currentIndex()
        name = self._plot_tabs.tabText(idx)
        pw = list(self._plots.values())[idx]
        fmt = self._export_fmt.currentText().lower()
        path, _ = QFileDialog.getSaveFileName(self, f"Export {name}", f"{name.replace(' ', '_')}.{fmt}",
                                              f"{fmt.upper()} (*.{fmt});;All (*)")
        if path:
            pw.get_figure().savefig(path, dpi=self._dpi.value(), bbox_inches="tight")
            if self._main_window:
                self._main_window.set_status(f"Exported: {path}")

    def _export_all(self):
        d = QFileDialog.getExistingDirectory(self, "Export All Figures To")
        if not d:
            return
        fmt = self._export_fmt.currentText().lower()
        for name, pw in self._plots.items():
            path = os.path.join(d, f"{name.replace(' ', '_')}.{fmt}")
            pw.get_figure().savefig(path, dpi=self._dpi.value(), bbox_inches="tight")
        if self._main_window:
            self._main_window.set_status(f"Exported all figures to {d}")

    # ═══════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════
    def _browse_file(self, edit, filt):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filt)
        if path:
            edit.setText(path)

    def _browse_results_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Results Directory")
        if d:
            self._results_edit.setText(d)

    def _pick_color(self, target):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            if target == "hv":
                self._hv_color = hex_color
                self._hv_color_btn.setStyleSheet(f"background-color: {hex_color}; color: white;")
            else:
                self._vs_color = hex_color
                self._vs_color_btn.setStyleSheet(f"background-color: {hex_color}; color: white;")

    def apply_config(self, cfg):
        """Apply settings from main window config."""
        plot = cfg.get("plot", {})
        if "dpi" in plot: self._dpi.setValue(plot["dpi"])
        if "x_axis_scale" in plot:
            idx = self._x_scale.findText(plot["x_axis_scale"])
            if idx >= 0: self._x_scale.setCurrentIndex(idx)
        if "y_axis_scale" in plot:
            idx = self._y_scale.findText(plot["y_axis_scale"])
            if idx >= 0: self._y_scale.setCurrentIndex(idx)

        hv = cfg.get("hv_forward", {})
        if "fmin" in hv: self._freq_min.setValue(hv["fmin"])
        if "fmax" in hv: self._freq_max.setValue(hv["fmax"])
