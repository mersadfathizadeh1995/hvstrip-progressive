"""All Profiles View — modular rewrite.

Thin orchestrator class that composes the UI from extracted modules:

- ``all_profiles_view_modules.constants``       — palettes, marker shapes, etc.
- ``all_profiles_view_modules.statistics``       — compute_stats, get_colors
- ``all_profiles_view_modules.save_helpers``     — shared save utilities
- ``all_profiles_view_modules.save_profiles``    — per-profile figure saving
- ``all_profiles_view_modules.save_publication`` — publication-quality figures
- ``all_profiles_view_modules.save_combined``    — combined CSV + master save
- ``all_profiles_view_modules.peak_picking``     — PeakPickingMixin
- ``all_profiles_view_modules.drawing``          — redraw_hv, redraw_vs
- ``all_profiles_view_modules.ui_builder``       — build_ui
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QWidget

from .all_profiles_view_modules.constants import FIGURE_SIZES
from .all_profiles_view_modules.drawing import (
    _get_legend_cfg,
    redraw_hv,
    redraw_vs,
)
from .all_profiles_view_modules.peak_picking import PeakPickingMixin
from .all_profiles_view_modules.save_combined import (
    resave_hv_csv,
    resave_hv_figures,
    resave_vs_figures,
    save_canvas_snapshots,
    save_combined_csv,
    update_peak_files,
)
from .all_profiles_view_modules.save_publication import (
    save_f0_histogram,
    save_f0_vs_vs30,
    save_hv_vs_combined,
    save_median_hv,
    save_normalized_hv,
    save_spectral_matrix,
    save_summary_tables,
    save_vs_comparison,
)
from .all_profiles_view_modules.statistics import compute_stats
from .all_profiles_view_modules.ui_builder import build_ui


class AllProfilesView(PeakPickingMixin, QWidget):
    """Canvas view showing all HV profiles overlaid with plot controls."""

    results_loaded = pyqtSignal()

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._results: list = []
        self._peak_data: dict = {}
        self._output_dir: str = ""

        # Peak-picking mixin state
        self._init_peak_picking_state()

        # Build the full UI
        build_ui(self)

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
        has = bool(self._results)
        self._btn_save.setEnabled(has)
        self._btn_save_all.setEnabled(has)
        self._btn_pick_f0.setEnabled(has)
        self._btn_pick_sec.setEnabled(has)
        self._redraw()
        if self._chk_vs.isChecked():
            self._redraw_vs()

    def set_output_dir(self, path: str):
        """Store default output directory (from the Multiple tab panel)."""
        self._output_dir = path

    def update_peak_data(self, peak_data):
        """Update peak selections (e.g. from wizard) and redraw."""
        self._peak_data.update(peak_data)
        self._redraw()

    # ── Drawing delegates ──────────────────────────────────────

    def _redraw(self, *_args):
        redraw_hv(self)

    def _redraw_vs(self, *_args):
        redraw_vs(self)

    def _toggle_vs(self, show):
        self._vs_panel.setVisible(show)
        if show and self._results:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(50, self._redraw_vs)

    # ── Statistics delegate ────────────────────────────────────

    def _compute_stats(self):
        return compute_stats(self._results)

    # ── Save: default (All Profiles outputs only) ──────────────

    def _save_results(self):
        """Save All Profiles outputs (combined figures, median, tables)."""
        if not self._results:
            return

        folder = self._output_dir
        if not folder:
            folder = QFileDialog.getExistingDirectory(
                self, "Save All Profiles Output To")
            if not folder:
                return

        base = Path(folder)
        dpi = self._dpi.value()
        fmt = self._export_fmt.currentText().lower()
        fig_key = self._fig_size.currentText()
        figsize = FIGURE_SIZES.get(fig_key, (12, 8))
        computed = [r for r in self._results if r.computed]
        palette = self._palette.currentText()

        out_dir = base / "all_profile_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        lcfg = _get_legend_cfg(self)

        # Combined CSV + median
        save_combined_csv(out_dir, self._results, self._peak_data,
                          self._median_peaks)
        # Canvas snapshots
        save_canvas_snapshots(
            out_dir, self._hv_plot.figure,
            self._vs_plot.figure if hasattr(self, '_vs_plot') else None,
            self._chk_vs.isChecked(), dpi, fmt)
        # Publication figures
        save_hv_vs_combined(out_dir, computed, self._results,
                            self._median_peaks, palette, dpi, fmt,
                            legend_cfg=lcfg)
        save_median_hv(out_dir, self._results, self._median_peaks, dpi, fmt,
                       legend_cfg=lcfg)
        save_summary_tables(out_dir, computed, self._results,
                            self._peak_data, self._median_peaks)
        save_vs_comparison(out_dir, computed, palette, dpi, fmt,
                           legend_cfg=lcfg)

        # Update per-profile peak_info
        update_peak_files(base, self._results, self._peak_data)

        if self._mw:
            self._mw.log(f"All Profiles output saved to {out_dir}")

    # ── Save: full re-save via dialog ──────────────────────────

    def _save_all_results(self):
        """Open Save Options dialog for full control over what to save."""
        if not self._results:
            return

        from ..dialogs.save_options_dialog import SaveOptionsDialog
        dlg = SaveOptionsDialog(
            default_dir=self._output_dir,
            default_dpi=self._dpi.value(),
            default_fmt=self._export_fmt.currentText(),
            parent=self,
        )
        if dlg.exec_() != dlg.Accepted:
            return

        opts = dlg.get_options()
        base = Path(opts["output_dir"])
        base.mkdir(parents=True, exist_ok=True)
        dpi = opts["dpi"]
        fmt = opts["format"].lower()
        fig_key = self._fig_size.currentText()
        figsize = FIGURE_SIZES.get(fig_key, (12, 8))
        computed = [r for r in self._results if r.computed]
        palette = self._palette.currentText()
        lcfg = _get_legend_cfg(self)

        # Per-profile re-saves
        if opts.get("resave_profiles"):
            resave_hv_csv(base, computed)
            update_peak_files(base, computed, self._peak_data)

        if opts.get("resave_hv_figures"):
            resave_hv_figures(base, computed, self._peak_data,
                              figsize, dpi, fmt, legend_cfg=lcfg)

        if opts.get("resave_vs_figures"):
            resave_vs_figures(base, computed, dpi)

        # All-profile outputs
        out_dir = base / "all_profile_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        if opts.get("combined_overlay"):
            save_combined_csv(out_dir, self._results, self._peak_data,
                              self._median_peaks)
            save_canvas_snapshots(
                out_dir, self._hv_plot.figure,
                self._vs_plot.figure if hasattr(self, '_vs_plot') else None,
                self._chk_vs.isChecked(), dpi, fmt)

        if opts.get("paper_hv_vs"):
            save_hv_vs_combined(out_dir, computed, self._results,
                                self._median_peaks, palette, dpi, fmt,
                                legend_cfg=lcfg)
            save_median_hv(out_dir, self._results, self._median_peaks,
                           dpi, fmt, legend_cfg=lcfg)
            save_summary_tables(out_dir, computed, self._results,
                                self._peak_data, self._median_peaks)
            save_vs_comparison(out_dir, computed, palette, dpi, fmt,
                               legend_cfg=lcfg)

        if opts.get("normalized_hv"):
            save_normalized_hv(out_dir, computed, self._results,
                               palette, figsize, dpi, fmt, legend_cfg=lcfg)

        if opts.get("f0_histogram"):
            save_f0_histogram(out_dir, computed, self._peak_data, dpi, fmt,
                              legend_cfg=lcfg)

        if opts.get("f0_vs_vs30"):
            save_f0_vs_vs30(out_dir, computed, self._peak_data, dpi, fmt)

        if opts.get("spectral_matrix"):
            save_spectral_matrix(out_dir, computed, self._peak_data, dpi, fmt)

        # Update per-profile peak_info
        update_peak_files(base, computed, self._peak_data)

        if self._mw:
            self._mw.log(f"Results saved to {base}")

    # ── Load ───────────────────────────────────────────────────

    def _load_results(self):
        """Load previously saved results folder."""
        folder = QFileDialog.getExistingDirectory(self, "Load Results Folder")
        if not folder:
            return

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
                self._mw.log(
                    f"Loaded {len(results)} profiles from {folder}")
