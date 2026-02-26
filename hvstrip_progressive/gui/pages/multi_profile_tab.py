"""
Multi-Profile Tab Widget — extracted from ForwardModelingPage.

Handles loading, processing, and saving multiple soil profiles
with sequential peak picking and combined overlay plotting.
"""

from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib.gridspec as gridspec
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..dialogs.multi_profile_dialog import (
    FigureSettings,
    MultiProfilePickerDialog,
    ProfileResult,
    _MARKER_COLORS,
    _MARKER_SHAPES,
    _PALETTE_NAMES,
    _darken_color,
    get_palette_colors,
)
from ..dialogs.output_viewer_dialog import OutputViewerDialog, load_output_folder
from ...core.soil_profile import SoilProfile


class MultiProfileTab(QWidget):
    """Self-contained widget for the Multiple Profiles tab.

    Parameters
    ----------
    get_freq_config : callable
        ``() -> dict`` returning ``{"fmin": …, "fmax": …, "nf": …}``.
    get_output_dir : callable
        ``() -> str`` returning the current output directory path.
    parent : QWidget, optional
    """

    status_message = Signal(str)

    def __init__(
        self,
        get_freq_config: Callable[[], Dict],
        get_output_dir: Callable[[], str],
        get_engine_name: Optional[Callable[[], str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._get_freq_config = get_freq_config
        self._get_output_dir = get_output_dir
        self._get_engine_name = get_engine_name or (lambda: "diffuse_field")
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "Load multiple profile files (.xlsx / .txt / .csv) and process "
            "them sequentially.\n"
            "Each file: Thickness | Vs | Vp | Density (last row = halfspace)."
        )
        info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.file_list)

        file_btns = QHBoxLayout()
        for label, slot in [
            ("Add Files…", self._add_files),
            ("Add Folder…", self._add_folder),
            ("Remove Selected", self._remove_selected),
            ("Clear All", self._clear_all),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            file_btns.addWidget(btn)
        file_btns.addStretch()
        layout.addLayout(file_btns)

        layout.addWidget(self._build_fig_settings_group())

        # Action buttons
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run All Profiles")
        self.btn_run.setStyleSheet(
            "background-color: #0078d4; color: white; "
            "padding: 8px 16px; font-size: 13px;"
        )
        self.btn_run.clicked.connect(self._run_profiles)
        btn_row.addWidget(self.btn_run)

        self.btn_load = QPushButton("Load Results Folder")
        self.btn_load.setStyleSheet(
            "background-color: #107c10; color: white; "
            "padding: 8px 16px; font-size: 13px;"
        )
        self.btn_load.setToolTip(
            "Open a previously saved output folder to view combined curves"
        )
        self.btn_load.clicked.connect(self._load_output)
        btn_row.addWidget(self.btn_load)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def _build_fig_settings_group(self) -> QGroupBox:
        fig_group = QGroupBox("Figure Settings")
        fig_lay = QVBoxLayout(fig_group)

        # Row 1: DPI / dimensions / font
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        row1.addWidget(self.dpi_spin)

        row1.addWidget(QLabel("Width:"))
        self.fig_w = QDoubleSpinBox()
        self.fig_w.setRange(4.0, 24.0)
        self.fig_w.setValue(10.0)
        self.fig_w.setSuffix(" in")
        row1.addWidget(self.fig_w)

        row1.addWidget(QLabel("Height:"))
        self.fig_h = QDoubleSpinBox()
        self.fig_h.setRange(3.0, 16.0)
        self.fig_h.setValue(5.0)
        self.fig_h.setSuffix(" in")
        row1.addWidget(self.fig_h)

        row1.addWidget(QLabel("Font:"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 24)
        self.font_spin.setValue(12)
        row1.addWidget(self.font_spin)
        row1.addStretch()
        fig_lay.addLayout(row1)

        # Row 2: toggles
        row2 = QHBoxLayout()
        self.logx = QCheckBox("Log X")
        self.logx.setChecked(True)
        row2.addWidget(self.logx)

        self.logy = QCheckBox("Log Y")
        row2.addWidget(self.logy)

        self.grid = QCheckBox("Grid")
        self.grid.setChecked(True)
        row2.addWidget(self.grid)

        self.save_png = QCheckBox("PNG")
        self.save_png.setChecked(True)
        row2.addWidget(self.save_png)

        self.save_pdf = QCheckBox("PDF")
        self.save_pdf.setChecked(True)
        row2.addWidget(self.save_pdf)

        self.show_vs = QCheckBox("Show Vs Profile")
        row2.addWidget(self.show_vs)
        row2.addStretch()
        fig_lay.addLayout(row2)

        # Combined plot settings sub-group
        comb = QGroupBox("Combined Plot Settings")
        comb_lay = QVBoxLayout(comb)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Color Palette:"))
        self.palette = QComboBox()
        self.palette.addItems(_PALETTE_NAMES)
        row3.addWidget(self.palette)

        row3.addWidget(QLabel("Line α:"))
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.1, 1.0)
        self.alpha.setValue(0.45)
        self.alpha.setSingleStep(0.05)
        row3.addWidget(self.alpha)

        row3.addWidget(QLabel("Line W:"))
        self.line_w = QDoubleSpinBox()
        self.line_w.setRange(0.3, 5.0)
        self.line_w.setValue(1.0)
        self.line_w.setSingleStep(0.1)
        row3.addWidget(self.line_w)

        row3.addWidget(QLabel("Median W:"))
        self.median_w = QDoubleSpinBox()
        self.median_w.setRange(1.0, 8.0)
        self.median_w.setValue(3.0)
        self.median_w.setSingleStep(0.5)
        row3.addWidget(self.median_w)
        row3.addStretch()
        comb_lay.addLayout(row3)

        row4 = QHBoxLayout()
        self.show_median = QCheckBox("Show Median Curve")
        self.show_median.setChecked(True)
        row4.addWidget(self.show_median)

        self.show_sec = QCheckBox("Show Secondary Peaks")
        self.show_sec.setChecked(True)
        row4.addWidget(self.show_sec)
        row4.addStretch()
        comb_lay.addLayout(row4)

        fig_lay.addWidget(comb)
        return fig_group

    # ------------------------------------------------------------------
    # File list management
    # ------------------------------------------------------------------

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Profile Files",
            "",
            "All Supported (*.xlsx *.txt *.csv);;"
            "Excel Files (*.xlsx);;Text Files (*.txt);;"
            "CSV Files (*.csv);;All Files (*)",
        )
        for p in paths:
            if not self._list_contains(p):
                item = QListWidgetItem(Path(p).name)
                item.setData(Qt.UserRole, p)
                self.file_list.addItem(item)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with Profile Files"
        )
        if folder:
            for p in sorted(
                f
                for ext in ("*.xlsx", "*.txt", "*.csv")
                for f in Path(folder).glob(ext)
            ):
                full = str(p)
                if not self._list_contains(full):
                    item = QListWidgetItem(p.name)
                    item.setData(Qt.UserRole, full)
                    self.file_list.addItem(item)

    def _remove_selected(self):
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

    def _clear_all(self):
        self.file_list.clear()

    def _list_contains(self, path: str) -> bool:
        return any(
            self.file_list.item(i).data(Qt.UserRole) == path
            for i in range(self.file_list.count())
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_fig_settings(self) -> FigureSettings:
        return FigureSettings(
            dpi=self.dpi_spin.value(),
            width=self.fig_w.value(),
            height=self.fig_h.value(),
            font_size=self.font_spin.value(),
            title_size=self.font_spin.value() + 1,
            legend_size=self.font_spin.value() - 2,
            log_x=self.logx.isChecked(),
            log_y=self.logy.isChecked(),
            grid=self.grid.isChecked(),
            save_png=self.save_png.isChecked(),
            save_pdf=self.save_pdf.isChecked(),
            show_vs=self.show_vs.isChecked(),
            color_palette=self.palette.currentText(),
            individual_alpha=self.alpha.value(),
            individual_linewidth=self.line_w.value(),
            median_linewidth=self.median_w.value(),
            show_median=self.show_median.isChecked(),
            show_secondary_peaks=self.show_sec.isChecked(),
        )

    # ------------------------------------------------------------------
    # Load output
    # ------------------------------------------------------------------

    def _load_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder with Profile Subfolders"
        )
        if not folder:
            return
        folder_path = Path(folder)
        results, median_result = load_output_folder(folder_path)
        if not results:
            QMessageBox.warning(
                self,
                "No Data Found",
                f"No profile subfolders with hv_curve.csv found in:\n{folder}",
            )
            return

        fig_settings = self.get_fig_settings()
        self.status_label.setText(
            f"Loaded {len(results)} profiles from {folder_path.name}"
        )
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        dialog = OutputViewerDialog(
            results, median_result, fig_settings,
            source_folder=folder, parent=self,
        )
        dialog.exec()

    # ------------------------------------------------------------------
    # Run profiles
    # ------------------------------------------------------------------

    def _run_profiles(self):
        n = self.file_list.count()
        if n == 0:
            QMessageBox.warning(
                self, "No Files",
                "Please add at least one profile file.",
            )
            return

        outdir = self._get_output_dir()
        if not outdir:
            QMessageBox.warning(
                self, "No Output",
                "Please set an output directory first.",
            )
            return

        # Parse all profile files (auto-detect format)
        profiles = []
        errors = []
        for i in range(n):
            fpath = self.file_list.item(i).data(Qt.UserRole)
            try:
                prof = SoilProfile.from_auto(fpath)
                profiles.append((Path(fpath).stem, prof))
            except Exception as e:
                errors.append(f"{Path(fpath).name}: {e}")

        if errors:
            QMessageBox.warning(
                self,
                "Parse Errors",
                f"Could not load {len(errors)} file(s):\n"
                + "\n".join(errors[:10]),
            )
        if not profiles:
            return

        freq_config = self._get_freq_config()
        fig_settings = self.get_fig_settings()

        engine_name = self._get_engine_name()

        dialog = MultiProfilePickerDialog(
            profiles, freq_config, fig_settings, self,
            engine_name=engine_name,
        )

        def _on_report(results, median):
            s = dialog._fig_settings
            self._save_results(results, outdir, s, median)
            out = Path(outdir)
            out.mkdir(parents=True, exist_ok=True)

        dialog.report_requested.connect(_on_report)

        if dialog.exec():
            results = dialog.get_all_results()
            median_result = dialog.get_median_result()
            self._save_results(
                results, outdir, dialog._fig_settings, median_result,
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results: list,
        outdir: str,
        s: FigureSettings,
        median_result=None,
    ):
        """Save per-profile folders + combined overlay plot."""
        import matplotlib.pyplot as plt

        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        extensions = []
        if s.save_png:
            extensions.append("png")
        if s.save_pdf:
            extensions.append("pdf")
        if not extensions:
            extensions = ["png"]

        saved_count = 0

        for r in results:
            if r.freqs is None:
                continue
            folder = out / r.name
            folder.mkdir(parents=True, exist_ok=True)

            # CSV
            csv_path = folder / "hv_curve.csv"
            with open(csv_path, "w") as f:
                f.write("frequency,amplitude\n")
                for freq, amp in zip(r.freqs, r.amps):
                    f.write(f"{freq},{amp}\n")

            # Peak info
            peak_path = folder / "peak_info.txt"
            with open(peak_path, "w") as f:
                if r.f0 is not None:
                    f.write(f"f0_Frequency_Hz,{r.f0[0]:.6f}\n")
                    f.write(f"f0_Amplitude,{r.f0[1]:.6f}\n")
                    f.write(f"f0_Index,{r.f0[2]}\n")
                for i, (sf, sa, si) in enumerate(r.secondary_peaks):
                    f.write(f"Secondary_{i+1}_Frequency_Hz,{sf:.6f}\n")
                    f.write(f"Secondary_{i+1}_Amplitude,{sa:.6f}\n")
                    f.write(f"Secondary_{i+1}_Index,{si}\n")

            # HV-only figure
            fig = plt.figure(figsize=(s.width, s.height))
            ax = fig.add_subplot(111)
            MultiProfilePickerDialog._draw_hv(ax, r, s)
            fig.tight_layout()
            for ext in extensions:
                fig.savefig(
                    folder / f"hv_forward_curve.{ext}",
                    dpi=s.dpi, bbox_inches="tight",
                )
            plt.close(fig)

            # HV + Vs figure
            if s.show_vs and r.profile is not None:
                fig2 = plt.figure(figsize=(s.width * 1.4, s.height))
                gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.25)
                ax_hv = fig2.add_subplot(gs[0])
                ax_vs = fig2.add_subplot(gs[1])
                MultiProfilePickerDialog._draw_hv(ax_hv, r, s)
                MultiProfilePickerDialog._draw_vs(ax_vs, r.profile, s)
                fig2.tight_layout()
                for ext in extensions:
                    fig2.savefig(
                        folder / f"hv_forward_with_vs.{ext}",
                        dpi=s.dpi, bbox_inches="tight",
                    )
                plt.close(fig2)

            saved_count += 1

        # Combined overlay plot
        self._save_combined_plot(results, out, s, extensions, median_result)

        self.status_label.setText(
            f"Done! Saved {saved_count} profiles to {outdir}"
        )
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.status_message.emit(
            f"Multiple Profiles: {saved_count} processed, saved to:\n{outdir}"
        )

    def _save_combined_plot(self, results, out, s, extensions, median_result):
        """Save the combined overlay plot and summary CSV."""
        import matplotlib.pyplot as plt

        computed = [r for r in results if r.freqs is not None]
        if not computed:
            return

        fig_c = plt.figure(figsize=(s.width, s.height))
        ax_c = fig_c.add_subplot(111)

        colors = get_palette_colors(s.color_palette, len(computed))

        for i, r in enumerate(computed):
            c = colors[i]
            ax_c.plot(
                r.freqs, r.amps,
                linewidth=s.individual_linewidth,
                color=c, alpha=s.individual_alpha,
                label=r.name,
            )
            if r.f0 is not None:
                ax_c.scatter(
                    r.f0[0], r.f0[1], color=c, s=100, marker="*",
                    edgecolors="black", linewidth=0.6, zorder=5,
                    alpha=min(s.individual_alpha + 0.3, 1.0),
                )
            if s.show_secondary_peaks:
                for sec_f, sec_a, _ in r.secondary_peaks:
                    ax_c.scatter(
                        sec_f, sec_a, color=c, s=70, marker="d",
                        edgecolors="black", linewidth=0.5, zorder=4,
                        alpha=min(s.individual_alpha + 0.2, 1.0),
                    )

        mr, med_freqs, med_amps = self._resolve_median(
            computed, median_result, s,
        )

        if med_freqs is not None and mr is not None:
            self._draw_median_on_axis(ax_c, mr, med_freqs, med_amps, s)

        ax_c.set_xlabel("Frequency (Hz)", fontsize=s.font_size)
        ax_c.set_ylabel("H/V Amplitude Ratio", fontsize=s.font_size)
        ax_c.set_title(
            "Combined HV Curves — All Profiles",
            fontsize=s.title_size, fontweight="bold",
        )
        if s.log_x:
            ax_c.set_xscale("log")
        if s.log_y:
            ax_c.set_yscale("log")
        if s.grid:
            ax_c.grid(True, alpha=0.3, which="both")
        ax_c.legend(
            loc="upper right",
            fontsize=max(s.legend_size - 1, 6), ncol=2,
        )
        fig_c.tight_layout()
        for ext in extensions:
            fig_c.savefig(
                out / f"combined_hv_curves.{ext}",
                dpi=s.dpi, bbox_inches="tight",
            )
        plt.close(fig_c)

        self._save_summary_csv(computed, mr, med_freqs, med_amps, out, s)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_median(computed, median_result, s):
        """Return (ProfileResult, med_freqs, med_amps) or (None, None, None)."""
        mr = median_result
        med_freqs = med_amps = None

        if mr is not None and mr.freqs is not None:
            med_freqs = mr.freqs
            med_amps = mr.amps
        elif s.show_median and len(computed) >= 2:
            ref_freqs = computed[0].freqs
            all_a = np.column_stack([
                np.interp(ref_freqs, r.freqs, r.amps) for r in computed
            ])
            med_freqs = ref_freqs
            med_amps = np.median(all_a, axis=1)
            peak_idx = int(np.argmax(med_amps))
            mr = ProfileResult(
                name="Median HV",
                profile=computed[0].profile,
                freqs=med_freqs,
                amps=med_amps,
                computed=True,
                f0=(float(med_freqs[peak_idx]),
                    float(med_amps[peak_idx]), peak_idx),
            )

        return mr, med_freqs, med_amps

    @staticmethod
    def _draw_median_on_axis(ax, mr, med_freqs, med_amps, s):
        f0_c = _MARKER_COLORS.get(s.f0_marker_color, "#d62728")
        f0_m = _MARKER_SHAPES.get(s.f0_marker_shape, "*")
        sec_c = _MARKER_COLORS.get(s.secondary_marker_color, "#2ca02c")
        sec_m = _MARKER_SHAPES.get(s.secondary_marker_shape, "*")

        ax.plot(
            med_freqs, med_amps, color="black",
            linewidth=s.median_linewidth, linestyle="-",
            label="Median HV", zorder=10,
        )
        if mr.f0 is not None:
            ax.scatter(
                mr.f0[0], mr.f0[1], color=f0_c,
                s=s.f0_marker_size, marker=f0_m,
                edgecolors=_darken_color(f0_c), linewidth=1.5,
                zorder=11,
                label=f"Median f0 = {mr.f0[0]:.2f} Hz",
            )
        if s.show_secondary_peaks:
            for sec_f, sec_a, _ in mr.secondary_peaks:
                ax.scatter(
                    sec_f, sec_a, color=sec_c,
                    s=s.secondary_marker_size, marker=sec_m,
                    edgecolors=_darken_color(sec_c), linewidth=1.2,
                    zorder=10,
                    label=f"Median Sec. ({sec_f:.2f} Hz, A={sec_a:.2f})",
                )

    @staticmethod
    def _save_summary_csv(computed, mr, med_freqs, med_amps, out, s):
        summary_path = out / "combined_summary.csv"
        with open(summary_path, "w") as f:
            f.write("Profile,f0_Hz,f0_Amplitude")
            if s.show_secondary_peaks:
                f.write(",Secondary_Peaks")
            f.write("\n")
            for r in computed:
                f0_f = f"{r.f0[0]:.6f}" if r.f0 else ""
                f0_a = f"{r.f0[1]:.6f}" if r.f0 else ""
                row = f"{r.name},{f0_f},{f0_a}"
                if s.show_secondary_peaks:
                    sec_str = (
                        "; ".join(f"{sf:.3f} Hz" for sf, _, _ in r.secondary_peaks)
                        if r.secondary_peaks else ""
                    )
                    row += f",{sec_str}"
                f.write(row + "\n")
            if mr is not None and mr.f0 is not None:
                m_f0 = f"{mr.f0[0]:.6f}"
                m_a0 = f"{mr.f0[1]:.6f}"
                row = f"Median,{m_f0},{m_a0}"
                if s.show_secondary_peaks:
                    sec_str = (
                        "; ".join(f"{sf:.3f} Hz" for sf, _, _ in mr.secondary_peaks)
                        if mr.secondary_peaks else ""
                    )
                    row += f",{sec_str}"
                f.write(row + "\n")

        if med_freqs is not None and med_amps is not None:
            med_path = out / "median_hv_curve.csv"
            with open(med_path, "w") as f:
                f.write("frequency,median_amplitude\n")
                for freq, amp in zip(med_freqs, med_amps):
                    f.write(f"{freq},{amp}\n")
            if mr is not None:
                med_peak_path = out / "median_peak_info.txt"
                with open(med_peak_path, "w") as f:
                    if mr.f0 is not None:
                        f.write(f"Median_f0_Frequency_Hz,{mr.f0[0]:.6f}\n")
                        f.write(f"Median_f0_Amplitude,{mr.f0[1]:.6f}\n")
                    for i, (sf, sa, si) in enumerate(mr.secondary_peaks):
                        f.write(f"Median_Secondary_{i+1}_Frequency_Hz,{sf:.6f}\n")
                        f.write(f"Median_Secondary_{i+1}_Amplitude,{sa:.6f}\n")


__all__ = ["MultiProfileTab"]
