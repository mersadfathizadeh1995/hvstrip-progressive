"""Engine Settings Dialog — 3-tab configuration for the three forward-modeling engines.

Faithfully ports the original PySide6 EngineSettingsDialog to PyQt5.
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QDoubleSpinBox, QSpinBox, QLineEdit, QPushButton,
    QCheckBox, QComboBox, QLabel, QFileDialog, QGroupBox,
)


class EngineSettingsDialog(QDialog):
    """Configure engine-specific parameters (HVf / Ellipticity / SH Wave)."""

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Engine Settings")
        self.setMinimumWidth(520)
        self._config = config or {}
        self._build_ui()
        self._load_from_config(self._config)

    # ── public API ──────────────────────────────────────────────
    def get_config(self):
        return {
            "diffuse_field": self._get_hvf_config(),
            "ellipticity": self._get_ell_config(),
            "sh_wave": self._get_sh_config(),
        }

    # ── UI build ────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._tabs.addTab(self._build_hvf_tab(), "Diffuse Wave Field")
        self._tabs.addTab(self._build_ell_tab(), "Rayleigh Ellipticity")
        self._tabs.addTab(self._build_sh_tab(), "SH Wave Transfer")

        # Buttons
        btn_row = QHBoxLayout()
        btn_defaults = QPushButton("Restore Defaults")
        btn_defaults.clicked.connect(self._reset_defaults)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_defaults)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    # ── Tab 1: Diffuse Wave Field (HVf) ────────────────────────
    def _build_hvf_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.hvf_exe = QLineEdit()
        self.hvf_exe.setPlaceholderText("Path to HVf executable (auto-detect)")
        btn = QPushButton("...")
        btn.setFixedWidth(30)
        btn.clicked.connect(lambda: self._browse(self.hvf_exe, "Executable (*.exe)"))
        row = QHBoxLayout()
        row.addWidget(self.hvf_exe)
        row.addWidget(btn)
        form.addRow("HVf Executable:", row)

        self.hvf_fmin = QDoubleSpinBox(); self.hvf_fmin.setRange(0.01, 100); self.hvf_fmin.setValue(0.2)
        self.hvf_fmax = QDoubleSpinBox(); self.hvf_fmax.setRange(0.1, 200); self.hvf_fmax.setValue(20.0)
        self.hvf_nf = QSpinBox(); self.hvf_nf.setRange(10, 2000); self.hvf_nf.setValue(71)
        form.addRow("Freq Min (Hz):", self.hvf_fmin)
        form.addRow("Freq Max (Hz):", self.hvf_fmax)
        form.addRow("Freq Points:", self.hvf_nf)

        self.hvf_nmr = QSpinBox(); self.hvf_nmr.setRange(1, 100); self.hvf_nmr.setValue(10)
        self.hvf_nml = QSpinBox(); self.hvf_nml.setRange(1, 100); self.hvf_nml.setValue(10)
        self.hvf_nks = QSpinBox(); self.hvf_nks.setRange(1, 100); self.hvf_nks.setValue(10)
        form.addRow("Rayleigh Modes (nmr):", self.hvf_nmr)
        form.addRow("Love Modes (nml):", self.hvf_nml)
        form.addRow("Wavenumber Steps (nks):", self.hvf_nks)

        return w

    # ── Tab 2: Rayleigh Ellipticity ─────────────────────────────
    def _build_ell_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.ell_gpell = QLineEdit()
        self.ell_gpell.setPlaceholderText("Path to gpell executable")
        btn1 = QPushButton("..."); btn1.setFixedWidth(30)
        btn1.clicked.connect(lambda: self._browse(self.ell_gpell, "Executable (*.exe)"))
        r1 = QHBoxLayout(); r1.addWidget(self.ell_gpell); r1.addWidget(btn1)
        form.addRow("gpell Path:", r1)

        self.ell_bash = QLineEdit()
        self.ell_bash.setPlaceholderText("C:/Program Files/Git/bin/bash.exe")
        btn2 = QPushButton("..."); btn2.setFixedWidth(30)
        btn2.clicked.connect(lambda: self._browse(self.ell_bash, "Executable (*.exe)"))
        r2 = QHBoxLayout(); r2.addWidget(self.ell_bash); r2.addWidget(btn2)
        form.addRow("Git Bash Path:", r2)

        self.ell_fmin = QDoubleSpinBox(); self.ell_fmin.setRange(0.01, 100); self.ell_fmin.setValue(0.5)
        self.ell_fmax = QDoubleSpinBox(); self.ell_fmax.setRange(0.1, 200); self.ell_fmax.setValue(20.0)
        self.ell_nsamples = QSpinBox(); self.ell_nsamples.setRange(50, 5000); self.ell_nsamples.setValue(500)
        self.ell_nmodes = QSpinBox(); self.ell_nmodes.setRange(1, 10); self.ell_nmodes.setValue(1)
        form.addRow("Freq Min (Hz):", self.ell_fmin)
        form.addRow("Freq Max (Hz):", self.ell_fmax)
        form.addRow("Samples:", self.ell_nsamples)
        form.addRow("Modes:", self.ell_nmodes)

        self.ell_sampling = QComboBox(); self.ell_sampling.addItems(["log", "frequency", "period"])
        form.addRow("Sampling:", self.ell_sampling)

        self.ell_absolute = QCheckBox("Output absolute ellipticity")
        self.ell_peak_refine = QCheckBox("Peak-refined curves (-pc)")
        form.addRow(self.ell_absolute)
        form.addRow(self.ell_peak_refine)

        self.ell_love_alpha = QDoubleSpinBox(); self.ell_love_alpha.setRange(0, 0.99); self.ell_love_alpha.setValue(0)
        self.ell_love_alpha.setSingleStep(0.05)
        form.addRow("Love mixing (α):", self.ell_love_alpha)

        self.ell_auto_q = QCheckBox("Auto-compute Qp/Qs")
        self.ell_auto_q.setChecked(True)
        form.addRow(self.ell_auto_q)

        self.ell_q_formula = QComboBox(); self.ell_q_formula.addItems(["default", "brocher", "constant"])
        form.addRow("Q Formula:", self.ell_q_formula)

        self.ell_clip = QDoubleSpinBox(); self.ell_clip.setRange(0, 1000); self.ell_clip.setValue(0)
        form.addRow("Clip Factor:", self.ell_clip)

        return w

    # ── Tab 3: SH Wave Transfer ─────────────────────────────────
    def _build_sh_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.sh_fmin = QDoubleSpinBox(); self.sh_fmin.setRange(0.01, 100); self.sh_fmin.setValue(0.1)
        self.sh_fmax = QDoubleSpinBox(); self.sh_fmax.setRange(0.1, 200); self.sh_fmax.setValue(30.0)
        self.sh_nsamples = QSpinBox(); self.sh_nsamples.setRange(50, 5000); self.sh_nsamples.setValue(512)
        form.addRow("Freq Min (Hz):", self.sh_fmin)
        form.addRow("Freq Max (Hz):", self.sh_fmax)
        form.addRow("Samples:", self.sh_nsamples)

        self.sh_sampling = QComboBox(); self.sh_sampling.addItems(["log", "linear"])
        form.addRow("Sampling:", self.sh_sampling)

        self.sh_dsoil = QDoubleSpinBox(); self.sh_dsoil.setRange(0, 30); self.sh_dsoil.setValue(0)
        self.sh_dsoil.setSpecialValueText("Auto")
        self.sh_drock = QDoubleSpinBox(); self.sh_drock.setRange(0, 30); self.sh_drock.setValue(0.5)
        form.addRow("Soil Damping (%):", self.sh_dsoil)
        form.addRow("Rock Damping (%):", self.sh_drock)

        self.sh_d_tf = QComboBox()
        self.sh_d_tf.addItems(["0 (outcrop)", "within (top of rock)"])
        form.addRow("Transfer Function:", self.sh_d_tf)

        self.sh_darendeli = QComboBox()
        self.sh_darendeli.addItems(["1 (Mean)", "2 (Mean+1σ)", "3 (Mean-1σ)"])
        form.addRow("Darendeli Curve:", self.sh_darendeli)

        self.sh_gamma_max = QDoubleSpinBox(); self.sh_gamma_max.setRange(10, 30); self.sh_gamma_max.setValue(23.0)
        form.addRow("γ_max (kN/m³):", self.sh_gamma_max)

        self.sh_clip = QDoubleSpinBox(); self.sh_clip.setRange(0, 100); self.sh_clip.setValue(0)
        self.sh_clip.setSpecialValueText("Off")
        form.addRow("Clip TF above:", self.sh_clip)

        return w

    # ── config ──────────────────────────────────────────────────
    def _get_hvf_config(self):
        return {
            "exe_path": self.hvf_exe.text(),
            "fmin": self.hvf_fmin.value(),
            "fmax": self.hvf_fmax.value(),
            "nf": self.hvf_nf.value(),
            "nmr": self.hvf_nmr.value(),
            "nml": self.hvf_nml.value(),
            "nks": self.hvf_nks.value(),
        }

    def _get_ell_config(self):
        return {
            "gpell_path": self.ell_gpell.text(),
            "git_bash_path": self.ell_bash.text(),
            "fmin": self.ell_fmin.value(),
            "fmax": self.ell_fmax.value(),
            "n_samples": self.ell_nsamples.value(),
            "n_modes": self.ell_nmodes.value(),
            "sampling": self.ell_sampling.currentText(),
            "absolute": self.ell_absolute.isChecked(),
            "peak_refinement": self.ell_peak_refine.isChecked(),
            "love_alpha": self.ell_love_alpha.value(),
            "auto_q": self.ell_auto_q.isChecked(),
            "q_formula": self.ell_q_formula.currentText(),
            "clip_factor": self.ell_clip.value(),
        }

    def _get_sh_config(self):
        return {
            "fmin": self.sh_fmin.value(),
            "fmax": self.sh_fmax.value(),
            "n_samples": self.sh_nsamples.value(),
            "sampling": self.sh_sampling.currentText(),
            "Dsoil": self.sh_dsoil.value() if self.sh_dsoil.value() > 0 else None,
            "Drock": self.sh_drock.value(),
            "d_tf": self.sh_d_tf.currentIndex(),
            "darendeli_curvetype": self.sh_darendeli.currentIndex() + 1,
            "gamma_max": self.sh_gamma_max.value(),
            "clip_tf": self.sh_clip.value() if self.sh_clip.value() > 0 else None,
        }

    def _load_from_config(self, cfg):
        hvf = cfg.get("diffuse_field", {})
        if hvf.get("exe_path"): self.hvf_exe.setText(hvf["exe_path"])
        if "fmin" in hvf: self.hvf_fmin.setValue(hvf["fmin"])
        if "fmax" in hvf: self.hvf_fmax.setValue(hvf["fmax"])
        if "nf" in hvf: self.hvf_nf.setValue(hvf["nf"])
        if "nmr" in hvf: self.hvf_nmr.setValue(hvf["nmr"])
        if "nml" in hvf: self.hvf_nml.setValue(hvf["nml"])
        if "nks" in hvf: self.hvf_nks.setValue(hvf["nks"])

        ell = cfg.get("ellipticity", {})
        if ell.get("gpell_path"): self.ell_gpell.setText(ell["gpell_path"])
        if ell.get("git_bash_path"): self.ell_bash.setText(ell["git_bash_path"])
        if "fmin" in ell: self.ell_fmin.setValue(ell["fmin"])
        if "fmax" in ell: self.ell_fmax.setValue(ell["fmax"])
        if "n_samples" in ell: self.ell_nsamples.setValue(ell["n_samples"])
        if "n_modes" in ell: self.ell_nmodes.setValue(ell["n_modes"])

        sh = cfg.get("sh_wave", {})
        if "fmin" in sh: self.sh_fmin.setValue(sh["fmin"])
        if "fmax" in sh: self.sh_fmax.setValue(sh["fmax"])
        if "n_samples" in sh: self.sh_nsamples.setValue(sh["n_samples"])
        if "Drock" in sh: self.sh_drock.setValue(sh["Drock"])

    def _reset_defaults(self):
        self.hvf_exe.clear()
        self.hvf_fmin.setValue(0.2); self.hvf_fmax.setValue(20.0); self.hvf_nf.setValue(71)
        self.hvf_nmr.setValue(10); self.hvf_nml.setValue(10); self.hvf_nks.setValue(10)
        self.ell_gpell.clear(); self.ell_bash.clear()
        self.ell_fmin.setValue(0.5); self.ell_fmax.setValue(20.0); self.ell_nsamples.setValue(500)
        self.ell_nmodes.setValue(1)
        self.sh_fmin.setValue(0.1); self.sh_fmax.setValue(30.0); self.sh_nsamples.setValue(512)
        self.sh_dsoil.setValue(0); self.sh_drock.setValue(0.5)

    def _browse(self, line_edit, file_filter):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if path:
            line_edit.setText(path)
