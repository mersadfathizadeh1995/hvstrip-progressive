"""Forward Single Panel — settings and run for single profile forward modeling.

Data loading is handled by the Data Input canvas tab.
This panel contains: engine, frequency settings, auto-peak detection,
export settings, output dir, Run/Save buttons.
"""
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QFileDialog, QScrollArea, QProgressBar, QCheckBox,
)

from ..widgets.style_constants import (
    BUTTON_PRIMARY, BUTTON_SUCCESS, GEAR_BUTTON, SECONDARY_LABEL, EMOJI,
)
from ..widgets.collapsible_group import CollapsibleGroupBox

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]


class ForwardSinglePanel(QWidget):
    """Left-panel content for Forward → Single sub-tab."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._worker = None
        self._auto_peak_cfg = None
        self._loaded_profile = None
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # ── Profile Status ─────────────────────────────────────
        prof_grp = CollapsibleGroupBox(
            f"{EMOJI['profile']} Loaded Profile", collapsed=False)
        prof_lay = QVBoxLayout()
        prof_lay.setSpacing(2)
        self._prof_name = QLabel("No profile loaded")
        self._prof_name.setStyleSheet("font-weight: bold; font-size: 11px;")
        prof_lay.addWidget(self._prof_name)
        self._prof_info = QLabel("")
        self._prof_info.setStyleSheet(SECONDARY_LABEL)
        self._prof_info.setWordWrap(True)
        prof_lay.addWidget(self._prof_info)
        prof_grp.setContentLayout(prof_lay)
        lay.addWidget(prof_grp)

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
        self._gear_btn.setToolTip("Engine settings")
        self._gear_btn.clicked.connect(self._open_engine_settings)
        eng_row.addWidget(self._gear_btn)
        settings_lay.addLayout(eng_row)

        # Frequency
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Fmin:"))
        self._fmin = QDoubleSpinBox()
        self._fmin.setRange(0.01, 10.0)
        self._fmin.setValue(0.2)
        self._fmin.setSingleStep(0.1)
        self._fmin.setDecimals(2)
        freq_row.addWidget(self._fmin)

        freq_row.addWidget(QLabel("Fmax:"))
        self._fmax = QDoubleSpinBox()
        self._fmax.setRange(1.0, 100.0)
        self._fmax.setValue(20.0)
        self._fmax.setSingleStep(1.0)
        self._fmax.setDecimals(1)
        freq_row.addWidget(self._fmax)

        freq_row.addWidget(QLabel("Points:"))
        self._nf = QSpinBox()
        self._nf.setRange(50, 2000)
        self._nf.setValue(500)
        freq_row.addWidget(self._nf)
        settings_lay.addLayout(freq_row)

        # Output directory
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("(optional) Save results directory")
        out_row.addWidget(self._out_dir, 1)
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(btn_browse)
        settings_lay.addLayout(out_row)

        settings_grp.setContentLayout(settings_lay)
        lay.addWidget(settings_grp)

        # ── Auto-detect Peaks ──────────────────────────────────
        auto_grp = CollapsibleGroupBox(
            f"{EMOJI['peak']} Peak Detection", collapsed=True)
        auto_lay = QVBoxLayout()
        auto_lay.setSpacing(2)

        auto_row = QHBoxLayout()
        self._chk_auto = QCheckBox("Auto-detect peaks")
        self._chk_auto.setChecked(False)
        self._chk_auto.setToolTip(
            "When checked, uses configurable peak detection\n"
            "(prominence + range) instead of simple max.\n"
            "Also auto-detects secondary peaks.")
        auto_row.addWidget(self._chk_auto)
        self._auto_gear = QPushButton(EMOJI["settings"])
        self._auto_gear.setFixedSize(26, 26)
        self._auto_gear.setStyleSheet(GEAR_BUTTON)
        self._auto_gear.setToolTip("Auto Peak Detection Settings")
        self._auto_gear.clicked.connect(self._open_auto_peak_settings)
        auto_row.addWidget(self._auto_gear)
        auto_row.addStretch()
        auto_lay.addLayout(auto_row)

        self._peak_info_label = QLabel("")
        self._peak_info_label.setStyleSheet(SECONDARY_LABEL)
        self._peak_info_label.setWordWrap(True)
        auto_lay.addWidget(self._peak_info_label)

        auto_grp.setContentLayout(auto_lay)
        lay.addWidget(auto_grp)

        # ── Export Settings ────────────────────────────────────
        # (Moved to HV Curve View's collapsible Plot Settings)

        # ── Run ────────────────────────────────────────────────
        self._btn_run = QPushButton(f"{EMOJI['run']} Compute HV Curve")
        self._btn_run.setStyleSheet(BUTTON_PRIMARY)
        self._btn_run.clicked.connect(self._run_forward)
        lay.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        # Result
        self._result_label = QLabel("")
        self._result_label.setStyleSheet(SECONDARY_LABEL)
        self._result_label.setWordWrap(True)
        lay.addWidget(self._result_label)

        lay.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._apply_config()
        self._connect_data_input()

    def _apply_config(self):
        if not self._mw:
            return
        cfg = self._mw.config
        engine = self._mw.get_engine_name()
        idx = self._engine_combo.findText(engine)
        if idx >= 0:
            self._engine_combo.setCurrentIndex(idx)
        fwd = cfg.get("hv_forward", {})
        self._fmin.setValue(fwd.get("fmin", 0.2))
        self._fmax.setValue(fwd.get("fmax", 20.0))
        self._nf.setValue(fwd.get("nf", 71))

    def _connect_data_input(self):
        """Connect to DataInputView's profile_loaded signal (deferred)."""
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(200, self._try_connect_data_input)

    def _try_connect_data_input(self):
        di = self._get_data_input()
        if di and hasattr(di, "profile_loaded"):
            di.profile_loaded.connect(self._on_profile_loaded)

    def _on_profile_loaded(self, profile, path):
        """Update profile status when user loads a profile."""
        self._loaded_profile = profile
        if profile is None:
            self._prof_name.setText("No profile loaded")
            self._prof_info.setText("")
            return
        name = Path(path).name if path and path != "editor" else "Editor profile"
        self._prof_name.setText(f"📁 {name}")
        finite = [L for L in profile.layers if not L.is_halfspace]
        hs = [L for L in profile.layers if L.is_halfspace]
        total_d = sum(L.thickness for L in finite)
        info_parts = [f"{len(finite)} layers, depth = {total_d:.1f} m"]
        if hs:
            info_parts.append(f"HS: Vs = {hs[0].vs:.0f} m/s")
        vs_range = [L.vs for L in finite]
        if vs_range:
            info_parts.append(f"Vs = {min(vs_range):.0f}–{max(vs_range):.0f} m/s")
        self._prof_info.setText(" · ".join(info_parts))

    def _get_data_input(self):
        """Get the DataInputView from the canvas."""
        if self._mw:
            from ..strip_window import MODE_FWD_SINGLE
            return self._mw.get_data_input(MODE_FWD_SINGLE)
        return None

    # ── Slots ──────────────────────────────────────────────────
    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
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

    def _run_forward(self):
        """Run forward computation on the loaded profile."""
        di = self._get_data_input()
        profile = di.get_profile() if di else None
        if not profile:
            self._result_label.setText(
                "Load a profile in the Data Input tab first.")
            self._result_label.setStyleSheet("color: orange; font-size: 11px;")
            return

        self._loaded_profile = profile
        engine_name = self._engine_combo.currentText()
        config = self._build_config()

        # Write temp model file
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w")
        tmp.write(profile.to_hvf_format())
        tmp.close()

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._result_label.setText("Computing...")

        try:
            from ..workers.forward_worker import ForwardWorker
            self._worker = ForwardWorker(tmp.name, config, engine_name)
            self._worker.finished_signal.connect(
                lambda res: self._on_forward_done(res, tmp.name))
            self._worker.error.connect(
                lambda err: self._on_forward_error(err, tmp.name))
            self._worker.start()
        except ImportError:
            self._result_label.setText("ForwardWorker not available")
            self._btn_run.setEnabled(True)
            self._progress.setVisible(False)
            os.unlink(tmp.name)

    def _on_forward_done(self, result, tmp_path):
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)

        freqs, amps = result
        self._last_freqs = freqs
        self._last_amps = amps

        # Peak detection
        f0, secondary = self._detect_peaks(freqs, amps)
        self._last_f0 = f0
        self._last_secondary = secondary

        # Update result label
        parts = [f"f0 = {f0[0]:.3f} Hz (A = {f0[1]:.2f})"]
        for j, s in enumerate(secondary):
            parts.append(f"Sec.{j+1} = {s[0]:.3f} Hz")
        self._result_label.setText("  |  ".join(parts))
        self._result_label.setStyleSheet("color: green; font-size: 11px;")

        # Update peak info in collapse
        peak_lines = [f"Primary: f0 = {f0[0]:.3f} Hz, A = {f0[1]:.2f}"]
        for j, s in enumerate(secondary):
            peak_lines.append(f"Secondary {j+1}: {s[0]:.3f} Hz, A = {s[1]:.2f}")
        self._peak_info_label.setText("\n".join(peak_lines))

        # Send to HV Curve View
        if self._mw:
            di = self._get_data_input()
            profile = di.get_profile() if di else None
            self._mw.update_hv_curve(freqs, amps, profile)
            if profile:
                self._mw.update_vs_profile(profile)

            # Pass peak data to HV Curve View
            hv_view = self._get_hv_curve_view()
            if hv_view:
                hv_view._f0 = f0
                hv_view._secondary = list(secondary)
                hv_view._redraw()
                hv_view._update_label()

            self._mw.log(
                f"Forward done: f0 = {f0[0]:.3f} Hz, A = {f0[1]:.2f}"
                + (f", {len(secondary)} secondary peak(s)" if secondary else ""))

    def _detect_peaks(self, freqs, amps):
        """Detect peaks using auto-peak config or simple argmax."""
        freqs = np.asarray(freqs)
        amps = np.asarray(amps)

        if not self._chk_auto.isChecked() or self._auto_peak_cfg is None:
            # Simple argmax
            idx = int(np.argmax(amps))
            return (float(freqs[idx]), float(amps[idx]), idx), []

        cfg = self._auto_peak_cfg
        min_prom = cfg.get("min_prominence", 0.3)
        min_amp = cfg.get("min_amplitude", 1.0)

        # Primary peak detection
        try:
            from scipy.signal import find_peaks as _find_peaks
            peaks, props = _find_peaks(amps, prominence=min_prom)
            if len(peaks) > 0:
                best = peaks[np.argmax(amps[peaks])]
                f0 = (float(freqs[best]), float(amps[best]), int(best))
            else:
                idx = int(np.argmax(amps))
                f0 = (float(freqs[idx]), float(amps[idx]), idx)
        except ImportError:
            idx = int(np.argmax(amps))
            f0 = (float(freqs[idx]), float(amps[idx]), idx)

        # Secondary peak detection
        secondary = []
        n_sec = cfg.get("n_secondary", 0)
        ranges = cfg.get("ranges", {})
        claimed = {f0[2]}  # indices already claimed

        for si in range(n_sec):
            rng = ranges.get(si + 1)  # 1-indexed range keys
            if rng and rng[0] is not None and rng[1] is not None:
                # Ranged: direct max search
                mask = (freqs >= rng[0]) & (freqs <= rng[1])
                # Exclude primary peak region (5% tolerance)
                tol = max(0.05 * f0[0], 2 * np.median(np.diff(freqs)))
                mask &= np.abs(freqs - f0[0]) > tol
                for ci in claimed:
                    if ci < len(freqs):
                        mask &= np.abs(freqs - freqs[ci]) > tol
                candidates = np.where(mask)[0]
                if len(candidates) > 0:
                    best = candidates[np.argmax(amps[candidates])]
                    if amps[best] >= min_amp:
                        secondary.append(
                            (float(freqs[best]), float(amps[best]), int(best)))
                        claimed.add(best)
            else:
                # Unranged: use scipy
                try:
                    from scipy.signal import find_peaks as _find_peaks
                    peaks, _ = _find_peaks(amps, prominence=min_prom)
                    valid = [p for p in peaks if p not in claimed
                             and amps[p] >= min_amp]
                    if valid:
                        valid.sort(key=lambda p: amps[p], reverse=True)
                        best = valid[0]
                        secondary.append(
                            (float(freqs[best]), float(amps[best]), int(best)))
                        claimed.add(best)
                except ImportError:
                    pass

        return f0, secondary

    def _get_hv_curve_view(self):
        """Get the HVCurveView from the canvas (tab index 1)."""
        if self._mw:
            from ..strip_window import MODE_FWD_SINGLE
            return self._mw.get_canvas_view(MODE_FWD_SINGLE, 1)
        return None

    def _on_forward_error(self, err_msg, tmp_path):
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        self._result_label.setText(f"Error: {err_msg}")
        self._result_label.setStyleSheet("color: red; font-size: 11px;")
        if self._mw:
            self._mw.log(f"Forward error: {err_msg}")

    def _build_config(self):
        engine_name = self._engine_combo.currentText()
        cfg = {
            "fmin": self._fmin.value(),
            "fmax": self._fmax.value(),
            "nf": self._nf.value(),
        }
        if self._mw:
            es = self._mw.get_engine_settings().get(engine_name, {})
            cfg.update(es)
        cfg["fmin"] = self._fmin.value()
        cfg["fmax"] = self._fmax.value()
        cfg["nf"] = self._nf.value()
        return cfg

    # ── Public API ─────────────────────────────────────────────
    def get_engine_name(self):
        return self._engine_combo.currentText()
