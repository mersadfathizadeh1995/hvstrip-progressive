"""Forward Single Panel — settings and run for single profile forward modeling.

Data loading is handled by the Data Input canvas tab.
This panel contains: engine, frequency settings, output dir, Run button.
"""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QFileDialog, QScrollArea, QProgressBar,
)

from ..widgets.style_constants import (
    BUTTON_PRIMARY, GEAR_BUTTON, SECONDARY_LABEL, EMOJI,
)
from ..widgets.collapsible_group import CollapsibleGroupBox

ENGINES = ["diffuse_field", "sh_wave", "ellipticity"]


class ForwardSinglePanel(QWidget):
    """Left-panel content for Forward → Single sub-tab."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._worker = None
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

        # Save
        self._btn_save = QPushButton(f"{EMOJI['save']} Save Results...")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_results)
        lay.addWidget(self._btn_save)

        lay.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._apply_config()

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

    def _run_forward(self):
        """Run forward computation on the loaded profile."""
        di = self._get_data_input()
        profile = di.get_profile() if di else None
        if not profile:
            self._result_label.setText(
                "Load a profile in the Data Input tab first.")
            self._result_label.setStyleSheet("color: orange; font-size: 11px;")
            return

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
        self._btn_save.setEnabled(True)

        freqs, amps = result
        import numpy as np
        peak_idx = np.argmax(amps)
        peak_freq = freqs[peak_idx]
        peak_amp = amps[peak_idx]

        self._result_label.setText(
            f"f0 = {peak_freq:.3f} Hz  (A = {peak_amp:.2f})")
        self._result_label.setStyleSheet("color: green; font-size: 11px;")

        self._last_freqs = freqs
        self._last_amps = amps

        if self._mw:
            di = self._get_data_input()
            profile = di.get_profile() if di else None
            self._mw.update_hv_curve(freqs, amps, profile)
            if profile:
                self._mw.update_vs_profile(profile)
            self._mw.log(
                f"Forward done: f0 = {peak_freq:.3f} Hz, A = {peak_amp:.2f}")

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

    def _save_results(self):
        if not hasattr(self, '_last_freqs'):
            return
        d = self._out_dir.text().strip()
        if not d:
            d = QFileDialog.getExistingDirectory(self, "Save Results To")
        if not d:
            return
        import numpy as np
        out = Path(d)
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "hv_curve.csv",
                   np.column_stack([self._last_freqs, self._last_amps]),
                   delimiter=",", header="frequency,amplitude", comments="")
        self._result_label.setText(f"Saved to {out}")
        if self._mw:
            self._mw.log(f"Results saved to {out}")

    # ── Public API ─────────────────────────────────────────────
    def get_engine_name(self):
        return self._engine_combo.currentText()
