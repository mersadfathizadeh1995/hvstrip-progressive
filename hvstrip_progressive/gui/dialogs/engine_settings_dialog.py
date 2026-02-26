"""
Engine Settings Dialog

Tabbed dialog for configuring forward-modeling engine parameters.
Tabs: Diffuse Wave Field, Rayleigh Ellipticity, SH Wave (placeholder).
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


ENGINE_TAB_INDEX = {
    "diffuse_field": 0,
    "ellipticity": 1,
    "sh_wave": 2,
}


class EngineSettingsDialog(QDialog):
    """Tabbed dialog for engine-specific configuration."""

    def __init__(self, parent=None, config: dict = None, active_engine: str = "diffuse_field"):
        super().__init__(parent)
        self.setWindowTitle("Forward-Modeling Engine Settings")
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)

        self._config = config or {}
        self._build_ui()
        self._load_from_config(self._config)

        idx = ENGINE_TAB_INDEX.get(active_engine, 0)
        self.tabs.setCurrentIndex(idx)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_diffuse_tab(), "Diffuse Wave Field")
        self.tabs.addTab(self._build_ellipticity_tab(), "Rayleigh Ellipticity")
        self.tabs.addTab(self._build_sh_tab(), "SH Wave")
        layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._reset_defaults
        )
        layout.addWidget(buttons)

    # ----------------------- Diffuse Wave Field tab
    def _build_diffuse_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        # HVf executable path
        path_row = QHBoxLayout()
        self.hvf_exe_edit = QLineEdit()
        self.hvf_exe_edit.setPlaceholderText("Leave blank for auto-detect")
        path_row.addWidget(self.hvf_exe_edit)
        btn = QPushButton("Browse...")
        btn.setMaximumWidth(80)
        btn.clicked.connect(self._browse_hvf_exe)
        path_row.addWidget(btn)
        form.addRow("HVf Executable:", path_row)

        self.hvf_fmin = QDoubleSpinBox()
        self.hvf_fmin.setRange(0.01, 10.0)
        self.hvf_fmin.setDecimals(2)
        self.hvf_fmin.setSingleStep(0.1)
        self.hvf_fmin.setSuffix(" Hz")
        form.addRow("Freq Min:", self.hvf_fmin)

        self.hvf_fmax = QDoubleSpinBox()
        self.hvf_fmax.setRange(1.0, 100.0)
        self.hvf_fmax.setDecimals(1)
        self.hvf_fmax.setSingleStep(1.0)
        self.hvf_fmax.setSuffix(" Hz")
        form.addRow("Freq Max:", self.hvf_fmax)

        self.hvf_nf = QSpinBox()
        self.hvf_nf.setRange(10, 2000)
        form.addRow("Freq Points:", self.hvf_nf)

        self.hvf_nmr = QSpinBox()
        self.hvf_nmr.setRange(1, 100)
        form.addRow("nmr (Rayleigh modes):", self.hvf_nmr)

        self.hvf_nml = QSpinBox()
        self.hvf_nml.setRange(1, 100)
        form.addRow("nml (Love modes):", self.hvf_nml)

        self.hvf_nks = QSpinBox()
        self.hvf_nks.setRange(1, 100)
        form.addRow("nks (wavenumber steps):", self.hvf_nks)

        return w

    # ----------------------- Rayleigh Ellipticity tab
    def _build_ellipticity_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        # gpell path
        gpell_row = QHBoxLayout()
        self.gpell_exe_edit = QLineEdit()
        self.gpell_exe_edit.setPlaceholderText("Path to gpell.exe")
        gpell_row.addWidget(self.gpell_exe_edit)
        btn_gpell = QPushButton("Browse...")
        btn_gpell.setMaximumWidth(80)
        btn_gpell.clicked.connect(self._browse_gpell_exe)
        gpell_row.addWidget(btn_gpell)
        form.addRow("gpell Executable:", gpell_row)

        # git bash path
        bash_row = QHBoxLayout()
        self.git_bash_edit = QLineEdit()
        self.git_bash_edit.setPlaceholderText("Path to git-bash.exe")
        bash_row.addWidget(self.git_bash_edit)
        btn_bash = QPushButton("Browse...")
        btn_bash.setMaximumWidth(80)
        btn_bash.clicked.connect(self._browse_git_bash)
        bash_row.addWidget(btn_bash)
        form.addRow("Git Bash:", bash_row)

        self.ell_fmin = QDoubleSpinBox()
        self.ell_fmin.setRange(0.01, 10.0)
        self.ell_fmin.setDecimals(2)
        self.ell_fmin.setSingleStep(0.1)
        self.ell_fmin.setSuffix(" Hz")
        form.addRow("Freq Min:", self.ell_fmin)

        self.ell_fmax = QDoubleSpinBox()
        self.ell_fmax.setRange(1.0, 100.0)
        self.ell_fmax.setDecimals(1)
        self.ell_fmax.setSingleStep(1.0)
        self.ell_fmax.setSuffix(" Hz")
        form.addRow("Freq Max:", self.ell_fmax)

        self.ell_nsamples = QSpinBox()
        self.ell_nsamples.setRange(50, 5000)
        form.addRow("Freq Samples:", self.ell_nsamples)

        self.ell_nmodes = QSpinBox()
        self.ell_nmodes.setRange(1, 10)
        form.addRow("Rayleigh Modes:", self.ell_nmodes)

        self.ell_sampling = QComboBox()
        self.ell_sampling.addItems(["log", "frequency", "period"])
        form.addRow("Sampling:", self.ell_sampling)

        self.ell_absolute = QCheckBox("Output absolute ellipticity")
        form.addRow("", self.ell_absolute)

        self.ell_peak_refine = QCheckBox("Peak-refined curves (-pc)")
        form.addRow("", self.ell_peak_refine)

        self.ell_love_alpha = QDoubleSpinBox()
        self.ell_love_alpha.setRange(0.0, 0.99)
        self.ell_love_alpha.setDecimals(2)
        self.ell_love_alpha.setSingleStep(0.05)
        form.addRow("Love mixing (alpha):", self.ell_love_alpha)

        self.ell_auto_q = QCheckBox("Auto-compute Qp/Qs when missing")
        form.addRow("", self.ell_auto_q)

        self.ell_q_formula = QComboBox()
        self.ell_q_formula.addItems(["default", "brocher", "constant"])
        form.addRow("Q Formula:", self.ell_q_formula)

        self.ell_clip = QDoubleSpinBox()
        self.ell_clip.setRange(0.0, 1000.0)
        self.ell_clip.setDecimals(1)
        self.ell_clip.setSingleStep(5.0)
        form.addRow("Clip Factor:", self.ell_clip)

        return w

    # ----------------------- SH Wave tab
    def _build_sh_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.sh_fmin = QDoubleSpinBox()
        self.sh_fmin.setRange(0.01, 10.0)
        self.sh_fmin.setDecimals(2)
        self.sh_fmin.setSingleStep(0.1)
        self.sh_fmin.setSuffix(" Hz")
        form.addRow("Freq Min:", self.sh_fmin)

        self.sh_fmax = QDoubleSpinBox()
        self.sh_fmax.setRange(1.0, 100.0)
        self.sh_fmax.setDecimals(1)
        self.sh_fmax.setSingleStep(1.0)
        self.sh_fmax.setSuffix(" Hz")
        form.addRow("Freq Max:", self.sh_fmax)

        self.sh_nsamples = QSpinBox()
        self.sh_nsamples.setRange(50, 5000)
        form.addRow("Freq Samples:", self.sh_nsamples)

        self.sh_sampling = QComboBox()
        self.sh_sampling.addItems(["log", "linear"])
        form.addRow("Sampling:", self.sh_sampling)

        self.sh_dsoil = QDoubleSpinBox()
        self.sh_dsoil.setRange(0.0, 20.0)
        self.sh_dsoil.setDecimals(2)
        self.sh_dsoil.setSingleStep(0.1)
        self.sh_dsoil.setSuffix(" %")
        self.sh_dsoil.setSpecialValueText("Auto (Darendeli)")
        form.addRow("Soil Damping:", self.sh_dsoil)

        self.sh_drock = QDoubleSpinBox()
        self.sh_drock.setRange(0.0, 20.0)
        self.sh_drock.setDecimals(2)
        self.sh_drock.setSingleStep(0.1)
        self.sh_drock.setSuffix(" %")
        form.addRow("Rock Damping:", self.sh_drock)

        self.sh_d_tf = QComboBox()
        self.sh_d_tf.addItems(["0 (outcrop)", "within (top of rock)"])
        form.addRow("Reference Depth:", self.sh_d_tf)

        self.sh_darendeli = QComboBox()
        self.sh_darendeli.addItems(["1 — Mean", "2 — Mean + 1\u03c3", "3 — Mean \u2212 1\u03c3"])
        form.addRow("Darendeli Curve:", self.sh_darendeli)

        self.sh_gamma_max = QDoubleSpinBox()
        self.sh_gamma_max.setRange(10.0, 30.0)
        self.sh_gamma_max.setDecimals(1)
        self.sh_gamma_max.setSingleStep(0.5)
        self.sh_gamma_max.setSuffix(" kN/m\u00b3")
        form.addRow("Max Unit Weight:", self.sh_gamma_max)

        self.sh_clip = QDoubleSpinBox()
        self.sh_clip.setRange(0.0, 1000.0)
        self.sh_clip.setDecimals(1)
        self.sh_clip.setSingleStep(5.0)
        self.sh_clip.setSpecialValueText("Off")
        form.addRow("Clip TF Above:", self.sh_clip)

        return w

    # ------------------------------------------------------------------ browse
    def _browse_hvf_exe(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select HVf Executable", "",
            "Executable (*.exe HVf HVf_Serial);;All Files (*)",
        )
        if f:
            self.hvf_exe_edit.setText(f)

    def _browse_gpell_exe(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select gpell Executable", "",
            "Executable (*.exe gpell);;All Files (*)",
        )
        if f:
            self.gpell_exe_edit.setText(f)

    def _browse_git_bash(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select git-bash.exe", "",
            "Executable (*.exe);;All Files (*)",
        )
        if f:
            self.git_bash_edit.setText(f)

    # ------------------------------------------------------------------ config I/O
    def _load_from_config(self, cfg: dict):
        """Populate widgets from a config dict."""
        df = cfg.get("diffuse_field", {})
        self.hvf_exe_edit.setText(df.get("exe_path", ""))
        self.hvf_fmin.setValue(df.get("fmin", 0.2))
        self.hvf_fmax.setValue(df.get("fmax", 20.0))
        self.hvf_nf.setValue(df.get("nf", 71))
        self.hvf_nmr.setValue(df.get("nmr", 10))
        self.hvf_nml.setValue(df.get("nml", 10))
        self.hvf_nks.setValue(df.get("nks", 10))

        el = cfg.get("ellipticity", {})
        self.gpell_exe_edit.setText(el.get("gpell_path", r"C:\Geopsy.org\bin\gpell.exe"))
        self.git_bash_edit.setText(
            el.get("git_bash_path", r"C:\Users\mersadf\AppData\Local\Programs\Git\git-bash.exe")
        )
        self.ell_fmin.setValue(el.get("fmin", 0.5))
        self.ell_fmax.setValue(el.get("fmax", 20.0))
        self.ell_nsamples.setValue(el.get("n_samples", 500))
        self.ell_nmodes.setValue(el.get("n_modes", 1))
        self.ell_sampling.setCurrentText(el.get("sampling", "log"))
        self.ell_absolute.setChecked(el.get("absolute", True))
        self.ell_peak_refine.setChecked(el.get("peak_refinement", False))
        self.ell_love_alpha.setValue(el.get("love_alpha", 0.0))
        self.ell_auto_q.setChecked(el.get("auto_q", False))
        self.ell_q_formula.setCurrentText(el.get("q_formula", "default"))
        self.ell_clip.setValue(el.get("clip_factor", 50.0))

        sh = cfg.get("sh_wave", {})
        self.sh_fmin.setValue(sh.get("fmin", 0.1))
        self.sh_fmax.setValue(sh.get("fmax", 30.0))
        self.sh_nsamples.setValue(sh.get("n_samples", 512))
        self.sh_sampling.setCurrentText(sh.get("sampling", "log"))
        self.sh_dsoil.setValue(sh.get("Dsoil", 0.0) or 0.0)
        self.sh_drock.setValue(sh.get("Drock", 0.5))
        d_tf_val = sh.get("d_tf", 0)
        if d_tf_val == "within":
            self.sh_d_tf.setCurrentIndex(1)
        else:
            self.sh_d_tf.setCurrentIndex(0)
        self.sh_darendeli.setCurrentIndex(sh.get("darendeli_curvetype", 1) - 1)
        self.sh_gamma_max.setValue(sh.get("gamma_max", 23.0))
        self.sh_clip.setValue(sh.get("clip_tf", 0.0))

    def get_config(self) -> dict:
        """Return the full engine settings dict."""
        return {
            "diffuse_field": {
                "exe_path": self.hvf_exe_edit.text(),
                "fmin": self.hvf_fmin.value(),
                "fmax": self.hvf_fmax.value(),
                "nf": self.hvf_nf.value(),
                "nmr": self.hvf_nmr.value(),
                "nml": self.hvf_nml.value(),
                "nks": self.hvf_nks.value(),
            },
            "ellipticity": {
                "gpell_path": self.gpell_exe_edit.text(),
                "git_bash_path": self.git_bash_edit.text(),
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
            },
            "sh_wave": {
                "fmin": self.sh_fmin.value(),
                "fmax": self.sh_fmax.value(),
                "n_samples": self.sh_nsamples.value(),
                "sampling": self.sh_sampling.currentText(),
                "Dsoil": self.sh_dsoil.value() if self.sh_dsoil.value() > 0 else None,
                "Drock": self.sh_drock.value(),
                "d_tf": "within" if self.sh_d_tf.currentIndex() == 1 else 0,
                "darendeli_curvetype": self.sh_darendeli.currentIndex() + 1,
                "gamma_max": self.sh_gamma_max.value(),
                "clip_tf": self.sh_clip.value(),
            },
        }

    def _reset_defaults(self):
        """Reset all widgets to default values."""
        self._load_from_config({})


__all__ = ["EngineSettingsDialog"]
