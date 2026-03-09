"""Forward Multiple Panel — settings and run for multi-profile forward modeling.

Data loading (profile list) is handled by the Multi Input canvas tab.
This panel contains: engine, frequency, output dir, auto-detect options,
Compute All button.  Plot settings moved to All Profiles canvas view.
"""
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QFileDialog, QScrollArea, QProgressBar, QCheckBox,
)

from ..widgets.style_constants import (
    BUTTON_PRIMARY, BUTTON_SUCCESS, GEAR_BUTTON,
    SECONDARY_LABEL, EMOJI,
)
from ..widgets.collapsible_group import CollapsibleGroupBox

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]


class ForwardMultiPanel(QWidget):
    """Left-panel content for Forward → Multiple sub-tab."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._auto_peak_cfg = None  # auto peak detection config
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # ── Settings ───────────────────────────────────────────
        settings_grp = CollapsibleGroupBox(f"{EMOJI['settings']} Settings")
        settings_lay = QVBoxLayout()

        # Engine
        eng_row = QHBoxLayout()
        eng_row.addWidget(QLabel("Engine:"))
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(ENGINES)
        eng_row.addWidget(self._engine_combo, 1)
        self._gear_btn = QPushButton(EMOJI["settings"])
        self._gear_btn.setFixedSize(26, 26)
        self._gear_btn.setStyleSheet(GEAR_BUTTON)
        self._gear_btn.clicked.connect(self._open_engine_settings)
        eng_row.addWidget(self._gear_btn)
        settings_lay.addLayout(eng_row)

        # Frequency
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Fmin:"))
        self._fmin = QDoubleSpinBox()
        self._fmin.setRange(0.01, 10.0)
        self._fmin.setValue(0.5)
        self._fmin.setSingleStep(0.1)
        self._fmin.setDecimals(2)
        freq_row.addWidget(self._fmin)
        freq_row.addWidget(QLabel("Fmax:"))
        self._fmax = QDoubleSpinBox()
        self._fmax.setRange(1.0, 100.0)
        self._fmax.setValue(20.0)
        freq_row.addWidget(self._fmax)
        freq_row.addWidget(QLabel("Pts:"))
        self._nf = QSpinBox()
        self._nf.setRange(50, 2000)
        self._nf.setValue(500)
        freq_row.addWidget(self._nf)
        settings_lay.addLayout(freq_row)

        # Output dir (required)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("(required — results saved here)")
        out_row.addWidget(self._out_dir, 1)
        btn_bro = QPushButton("...")
        btn_bro.setFixedWidth(30)
        btn_bro.clicked.connect(self._browse_output)
        out_row.addWidget(btn_bro)
        settings_lay.addLayout(out_row)

        settings_grp.setContentLayout(settings_lay)
        lay.addWidget(settings_grp)

        # ── Auto-detect Options ────────────────────────────────
        auto_row = QHBoxLayout()
        self._chk_auto = QCheckBox("Auto-detect peaks")
        self._chk_auto.setChecked(True)
        auto_row.addWidget(self._chk_auto)

        self._auto_gear = QPushButton(EMOJI["settings"])
        self._auto_gear.setFixedSize(26, 26)
        self._auto_gear.setStyleSheet(GEAR_BUTTON)
        self._auto_gear.setToolTip("Auto Peak Detection Settings")
        self._auto_gear.clicked.connect(self._open_auto_peak_settings)
        auto_row.addWidget(self._auto_gear)
        auto_row.addStretch()
        lay.addLayout(auto_row)

        self._chk_median = QCheckBox("Include Median Curve")
        self._chk_median.setChecked(True)
        lay.addWidget(self._chk_median)

        # ── Run ────────────────────────────────────────────────
        self._btn_run = QPushButton(f"{EMOJI['run']} Compute All")
        self._btn_run.setStyleSheet(BUTTON_PRIMARY)
        self._btn_run.clicked.connect(self._run_all)
        lay.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._result_label = QLabel("")
        self._result_label.setStyleSheet(SECONDARY_LABEL)
        self._result_label.setWordWrap(True)
        lay.addWidget(self._result_label)

        lay.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _get_data_input(self):
        """Get the MultiInputView from the canvas."""
        if self._mw:
            from ..strip_window import MODE_FWD_MULTI
            return self._mw.get_data_input(MODE_FWD_MULTI)
        return None

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self._out_dir.setText(d)

    def _open_engine_settings(self):
        if self._mw:
            self._mw._on_engine_settings()

    def _open_auto_peak_settings(self):
        """Open the Auto Peak Detection settings dialog."""
        try:
            from ..dialogs.auto_peak_settings_dialog import AutoPeakSettingsDialog
            dlg = AutoPeakSettingsDialog(parent=self)
            if self._auto_peak_cfg:
                dlg._load_config(self._auto_peak_cfg)
            if dlg.exec_() == AutoPeakSettingsDialog.Accepted:
                self._auto_peak_cfg = dlg.get_config()
                if self._mw:
                    self._mw.log("Auto peak settings updated")
        except Exception as e:
            self._result_label.setText(f"Auto peak dialog error: {e}")

    def _run_all(self):
        """Compute forward HV for all loaded profiles."""
        di = self._get_data_input()
        profiles = di.get_profiles() if di else []
        if not profiles:
            self._result_label.setText(
                "Add profiles in the Data Input tab first.")
            self._result_label.setStyleSheet("color: orange; font-size: 11px;")
            return

        # Validate output directory (required)
        out_dir = self._out_dir.text().strip()
        if not out_dir:
            from PyQt5.QtWidgets import QMessageBox
            d = QFileDialog.getExistingDirectory(
                self, "Select Output Directory (required)")
            if not d:
                QMessageBox.warning(
                    self, "Output Required",
                    "An output directory is required to save results.")
                return
            self._out_dir.setText(d)
            out_dir = d

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, len(profiles))
        self._progress.setValue(0)
        self._result_label.setText("Computing...")
        self._result_label.setStyleSheet(SECONDARY_LABEL)

        if self._mw:
            self._mw.log(
                f"Starting multi-profile forward ({len(profiles)} profiles)")

        config = self._build_config()
        engine_name = self._engine_combo.currentText()

        # Pass auto peak config if auto-detect is checked
        auto_cfg = None
        if self._chk_auto.isChecked():
            auto_cfg = self._auto_peak_cfg

        from ..workers.multi_forward_worker import MultiForwardWorker
        self._worker = MultiForwardWorker(
            profiles, config, engine_name, auto_peak_config=auto_cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.profile_done.connect(self._on_profile_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _build_config(self):
        """Build engine config from UI settings."""
        return {
            "fmin": self._fmin.value(),
            "fmax": self._fmax.value(),
            "nf": self._nf.value(),
        }

    def _on_progress(self, msg, current, total):
        self._progress.setValue(current)
        self._result_label.setText(msg)

    def _on_profile_done(self, idx, result):
        self._progress.setValue(idx + 1)
        if self._mw and result.computed:
            self._mw.log(
                f"  {result.name}: f0 = {result.f0[0]:.3f} Hz "
                f"(A = {result.f0[1]:.2f})")

    def _on_all_done(self, results):
        """Handle all profiles computed — route to wizard or All Profiles."""
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        self._results = results

        computed = [r for r in results if r.computed]
        n_ok = len(computed)
        n_total = len(results)
        self._result_label.setText(
            f"Done: {n_ok}/{n_total} profiles computed successfully.")
        self._result_label.setStyleSheet("color: green; font-size: 11px;")

        if self._mw:
            self._mw.log(
                f"Multi-profile done: {n_ok}/{n_total} computed")

        # Update summary dock
        self._update_summary_dock(results)

        # Route based on auto-detect setting
        auto_on = self._chk_auto.isChecked()

        # ALWAYS populate wizard so user can enter it later to modify peaks
        self._populate_wizard(results)

        if auto_on:
            # Auto-detect: populate All Profiles, show tab 2
            self._populate_all_profiles(results)
            if self._mw:
                canvas = self._mw.get_active_canvas()
                if canvas and canvas.count() > 2:
                    canvas.setCurrentIndex(2)
        else:
            # Manual: open wizard (tab 1) for per-profile peak selection
            if self._mw:
                canvas = self._mw.get_active_canvas()
                if canvas and canvas.count() > 1:
                    canvas.setCurrentIndex(1)

        # Save if output dir set
        out_dir = self._out_dir.text().strip()
        if out_dir:
            self._save_results(results, computed, out_dir)

    def _on_error(self, msg):
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        self._result_label.setText(f"Error: {msg}")
        self._result_label.setStyleSheet("color: red; font-size: 11px;")
        if self._mw:
            self._mw.log(f"Multi-profile error: {msg}")

    # ── Canvas population ──────────────────────────────────────

    def _populate_wizard(self, results):
        """Send results to Profile Wizard view (tab index 1)."""
        if not self._mw:
            return
        from ..strip_window import MODE_FWD_MULTI
        wizard = self._mw.get_canvas_view(MODE_FWD_MULTI, 1)
        if wizard and hasattr(wizard, 'set_results'):
            # Build auto peaks dict from results
            auto_peaks = {}
            for r in results:
                if r.computed and r.f0:
                    auto_peaks[r.name] = {
                        "f0": r.f0,
                        "secondary": list(r.secondary_peaks or []),
                    }
            wizard.set_results(results, auto_peaks)

            # Connect wizard finish to All Profiles if not already connected
            if not hasattr(self, '_wizard_connected'):
                wizard.wizard_finished.connect(self._on_wizard_finished)
                self._wizard_connected = True

    def _on_wizard_finished(self, peak_data):
        """Handle wizard completion — populate All Profiles with user peaks."""
        if not hasattr(self, '_results'):
            return
        results = self._results

        # Update results with wizard peak selections
        for r in results:
            pk = peak_data.get(r.name)
            if pk:
                r.f0 = pk.get("f0", r.f0)
                r.secondary_peaks = pk.get("secondary", r.secondary_peaks)

        self._populate_all_profiles(results, peak_data)
        self._update_summary_dock(results, peak_data)

        # Re-save per-profile figures (peaks may have changed)
        out_dir = self._out_dir.text().strip()
        if out_dir:
            from pathlib import Path
            base = Path(out_dir)
            computed = [r for r in results if r.computed]
            for r in computed:
                prof_dir = base / r.name
                if prof_dir.exists():
                    self._save_hv_figure(r, prof_dir, dpi=150)
                    # Re-save peak_info.txt with wizard-selected peaks
                    pk = peak_data.get(r.name, {})
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
            if self._mw:
                self._mw.log("Per-profile figures re-saved with wizard peaks.")

        # Switch to All Profiles tab
        if self._mw:
            canvas = self._mw.get_active_canvas()
            if canvas and canvas.count() > 2:
                canvas.setCurrentIndex(2)

        # Show summary dock
        if self._mw and hasattr(self._mw, '_summary_dock'):
            self._mw._summary_dock.setVisible(True)

    def _populate_all_profiles(self, results, peak_data=None):
        """Send results to the All Profiles view (tab index 2)."""
        if not self._mw:
            return
        from ..strip_window import MODE_FWD_MULTI
        view = self._mw.get_canvas_view(MODE_FWD_MULTI, 2)
        if view and hasattr(view, 'set_results'):
            view.set_results(results, peak_data)
        # Pass the output directory to All Profiles view
        out_dir = self._out_dir.text().strip()
        if view and hasattr(view, 'set_output_dir') and out_dir:
            view.set_output_dir(out_dir)

    def _update_summary_dock(self, results, peak_data=None):
        """Update the summary dock panel."""
        if not self._mw:
            return
        dock = self._mw.get_summary_dock()
        if dock and hasattr(dock, 'set_results'):
            dock.set_results(results, peak_data)
            dock.setVisible(True)

    # ── Save results ───────────────────────────────────────────

    def _save_results(self, results, computed, out_dir):
        """Save per-profile results + combined summary to output directory."""
        import numpy as np
        from pathlib import Path

        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)

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
            if r.f0:
                with open(prof_dir / "peak_info.txt", "w") as f:
                    f.write(f"f0_Frequency_Hz,{r.f0[0]:.6f}\n")
                    f.write(f"f0_Amplitude,{r.f0[1]:.6f}\n")
                    f.write(f"f0_Index,{r.f0[2]}\n")
                    for j, (sf, sa, si) in enumerate(r.secondary_peaks or []):
                        f.write(f"Secondary_{j+1}_Frequency_Hz,{sf:.6f}\n")
                        f.write(f"Secondary_{j+1}_Amplitude,{sa:.6f}\n")
                        f.write(f"Secondary_{j+1}_Index,{si}\n")

            # Vs profile output
            if r.profile:
                self._save_vs_output(r, prof_dir)

            # HV forward curve figure
            self._save_hv_figure(r, prof_dir, dpi=150)

        # Combined / all-profile output
        ap_dir = base / "all_profile_output"
        ap_dir.mkdir(exist_ok=True)

        with open(ap_dir / "combined_summary.csv", "w") as f:
            f.write("Profile,f0_Hz,f0_Amplitude,Secondary_Peaks\n")
            for r in results:
                if r.computed and r.f0:
                    sec = "; ".join(f"{s[0]:.3f} Hz" for s in (r.secondary_peaks or []))
                    f.write(f'{r.name},{r.f0[0]:.6f},{r.f0[1]:.6f},"{sec}"\n')
                else:
                    f.write(f'{r.name},,,\n')

        if len(computed) >= 2:
            med_freqs, med_amps = self._compute_median(computed)
            if med_freqs is not None:
                with open(ap_dir / "median_hv_curve.csv", "w") as f:
                    f.write("frequency,median_amplitude\n")
                    for freq, amp in zip(med_freqs, med_amps):
                        f.write(f"{freq},{amp}\n")

                idx = int(np.argmax(med_amps))
                with open(ap_dir / "median_peak_info.txt", "w") as f:
                    f.write(f"Median_f0_Frequency_Hz,{med_freqs[idx]:.6f}\n")
                    f.write(f"Median_f0_Amplitude,{med_amps[idx]:.6f}\n")

        if self._mw:
            self._mw.log(f"Results saved to {out_dir}")

    @staticmethod
    def _save_vs_output(r, prof_dir):
        """Save Vs profile figure, CSV data, and info for a single profile result."""
        try:
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt

            fig = Figure(figsize=(4, 6))
            ax = fig.add_subplot(111)
            depths, vs = [], []
            z = 0.0
            finite = [L for L in r.profile.layers if not L.is_halfspace]
            for L in finite:
                depths.append(z); vs.append(L.vs)
                z += L.thickness
                depths.append(z); vs.append(L.vs)
            ax.plot(vs, depths, color="teal", lw=1.8)
            ax.fill_betweenx(depths, 0, vs, alpha=0.1, color="teal")
            ax.invert_yaxis()
            ax.set_xlabel("Vs (m/s)"); ax.set_ylabel("Depth (m)")
            ax.set_title(f"Vs Profile — {r.name}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(prof_dir / "vs_profile.png", dpi=150)
            fig.savefig(prof_dir / "vs_profile.pdf")
            plt.close(fig)
        except Exception:
            pass

        # Vs profile CSV data
        try:
            finite = [L for L in r.profile.layers if not L.is_halfspace]
            with open(prof_dir / "vs_profile.csv", "w") as f:
                f.write("depth_top_m,depth_bot_m,thickness_m,vs_m_s,vp_m_s,density_kg_m3\n")
                z = 0.0
                for L in finite:
                    f.write(f"{z:.2f},{z + L.thickness:.2f},{L.thickness:.2f},"
                            f"{L.vs:.2f},{L.vp:.2f},{L.density:.2f}\n")
                    z += L.thickness
            hs = [L for L in r.profile.layers if L.is_halfspace]
            if hs:
                L = hs[0]
                with open(prof_dir / "vs_profile.csv", "a") as f:
                    f.write(f"{z:.2f},inf,inf,{L.vs:.2f},{L.vp:.2f},{L.density:.2f}\n")
        except Exception:
            pass

        try:
            from ..core.vs_average import vs_average_from_profile
            res30 = vs_average_from_profile(r.profile, target_depth=30.0)
            with open(prof_dir / "vs30_info.txt", "w") as f:
                f.write(f"Vs30_m_per_s,{res30.vs_avg:.2f}\n")
                f.write(f"Target_Depth_m,{res30.target_depth:.1f}\n")
                f.write(f"Actual_Depth_m,{res30.actual_depth:.1f}\n")
                f.write(f"Extrapolated,{res30.extrapolated}\n")
        except Exception:
            pass

    @staticmethod
    def _save_hv_figure(r, prof_dir, dpi=150):
        """Save HV forward curve figure for a single profile."""
        try:
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt

            if r.freqs is None:
                return
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.plot(r.freqs, r.amps, color="steelblue", lw=1.5, label="H/V")
            ax.set_xscale("log")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("H/V Amplitude")
            ax.set_title(f"H/V Forward Curve — {r.name}")
            ax.grid(True, which="both", alpha=0.3)

            if r.f0:
                ax.axvline(r.f0[0], color="red", lw=1.2, ls="--", alpha=0.7)
                ax.plot(r.f0[0], r.f0[1], "rv", ms=10, zorder=5)
                ax.annotate(f"f0 = {r.f0[0]:.2f} Hz\nAmp = {r.f0[1]:.2f}",
                            xy=(r.f0[0], r.f0[1]),
                            xytext=(10, 10), textcoords="offset points",
                            fontsize=8, color="red", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                      ec="red", alpha=0.8))
            for j, (sf, sa, _si) in enumerate(r.secondary_peaks or []):
                ax.axvline(sf, color="orange", lw=0.8, ls=":", alpha=0.6)
                ax.plot(sf, sa, "o", color="orange", ms=7, zorder=5)
                ax.annotate(f"{sf:.2f} Hz",
                            xy=(sf, sa),
                            xytext=(8, -12), textcoords="offset points",
                            fontsize=7, color="darkorange")

            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            fig.savefig(prof_dir / "hv_forward_curve.png", dpi=dpi)
            fig.savefig(prof_dir / "hv_forward_curve.pdf")
            plt.close(fig)
        except Exception:
            pass

    # ── Computation helpers ────────────────────────────────────

    @staticmethod
    def _compute_median(computed):
        """Compute median HV curve from list of ProfileResults."""
        import numpy as np
        if len(computed) < 2:
            return None, None
        ref_freqs = computed[0].freqs
        if ref_freqs is None:
            return None, None

        all_interp = []
        for r in computed:
            if r.freqs is not None and r.amps is not None:
                interp = np.interp(ref_freqs, r.freqs, r.amps)
                all_interp.append(interp)

        if len(all_interp) < 2:
            return None, None

        med_amps = np.median(np.array(all_interp), axis=0)
        return ref_freqs, med_amps

    # ── Public API ─────────────────────────────────────────────
    def get_profiles(self):
        di = self._get_data_input()
        return di.get_profiles() if di else []

    def get_engine_name(self):
        return self._engine_combo.currentText()

    def set_batch_folder(self, folder):
        di = self._get_data_input()
        if di and hasattr(di, 'set_batch_folder'):
            di.set_batch_folder(folder)
