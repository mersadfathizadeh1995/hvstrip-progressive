"""Settings Window — Global settings dialog opened from File → Settings.

6-tab dialog: General, Engines, Peak Detection, Dual-Resonance,
Plot Defaults, Paths.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QPushButton, QCheckBox, QFileDialog, QDialogButtonBox,
    QFormLayout, QGroupBox,
)


class SettingsWindow(QDialog):
    """Global settings dialog with 6 tabs."""

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(600, 500)
        self._config = config or {}
        self._build_ui()
        self._load_from_config()

    def _build_ui(self):
        lay = QVBoxLayout(self)

        self._tabs = QTabWidget()

        self._tabs.addTab(self._build_general_tab(), "General")
        self._tabs.addTab(self._build_engines_tab(), "Engines")
        self._tabs.addTab(self._build_peak_tab(), "Peak Detection")
        self._tabs.addTab(self._build_dr_tab(), "Dual-Resonance")
        self._tabs.addTab(self._build_plot_tab(), "Plot Defaults")
        self._tabs.addTab(self._build_paths_tab(), "Paths")

        lay.addWidget(self._tabs)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    # ── Tab: General ───────────────────────────────────────────
    def _build_general_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self._default_engine = QComboBox()
        self._default_engine.addItems(
            ["diffuse_field", "sh_wave", "ellipticity"])
        form.addRow("Default Engine:", self._default_engine)

        self._default_fmin = QDoubleSpinBox()
        self._default_fmin.setRange(0.01, 10.0)
        self._default_fmin.setValue(0.2)
        self._default_fmin.setDecimals(2)
        form.addRow("Default Fmin (Hz):", self._default_fmin)

        self._default_fmax = QDoubleSpinBox()
        self._default_fmax.setRange(1.0, 100.0)
        self._default_fmax.setValue(20.0)
        form.addRow("Default Fmax (Hz):", self._default_fmax)

        self._default_nf = QSpinBox()
        self._default_nf.setRange(50, 2000)
        self._default_nf.setValue(500)
        form.addRow("Default Num Points:", self._default_nf)

        self._default_outdir = QLineEdit()
        self._default_outdir.setPlaceholderText("(optional)")
        out_row = QHBoxLayout()
        out_row.addWidget(self._default_outdir)
        btn = QPushButton("...")
        btn.setFixedWidth(30)
        btn.clicked.connect(lambda: self._browse_dir(self._default_outdir))
        out_row.addWidget(btn)
        form.addRow("Default Output Dir:", out_row)

        return w

    # ── Tab: Engines ───────────────────────────────────────────
    def _build_engines_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        # Diffuse Field
        df_grp = QGroupBox("Diffuse Field")
        df_form = QFormLayout(df_grp)
        self._df_nf = QSpinBox()
        self._df_nf.setRange(10, 2000)
        self._df_nf.setValue(71)
        df_form.addRow("nf:", self._df_nf)
        self._df_nmr = QSpinBox()
        self._df_nmr.setRange(1, 100)
        self._df_nmr.setValue(10)
        df_form.addRow("nmr:", self._df_nmr)
        self._df_nml = QSpinBox()
        self._df_nml.setRange(1, 100)
        self._df_nml.setValue(10)
        df_form.addRow("nml:", self._df_nml)
        self._df_nks = QSpinBox()
        self._df_nks.setRange(1, 100)
        self._df_nks.setValue(10)
        df_form.addRow("nks:", self._df_nks)
        lay.addWidget(df_grp)

        # SH Wave
        sh_grp = QGroupBox("SH Wave")
        sh_form = QFormLayout(sh_grp)
        self._sh_nsamples = QSpinBox()
        self._sh_nsamples.setRange(64, 4096)
        self._sh_nsamples.setValue(512)
        sh_form.addRow("n_samples:", self._sh_nsamples)
        self._sh_sampling = QComboBox()
        self._sh_sampling.addItems(["log", "linear"])
        sh_form.addRow("Sampling:", self._sh_sampling)
        self._sh_drock = QDoubleSpinBox()
        self._sh_drock.setRange(0.0, 10.0)
        self._sh_drock.setValue(0.5)
        sh_form.addRow("Drock:", self._sh_drock)
        self._sh_gamma = QDoubleSpinBox()
        self._sh_gamma.setRange(0.0, 100.0)
        self._sh_gamma.setValue(23.0)
        sh_form.addRow("gamma_max:", self._sh_gamma)
        lay.addWidget(sh_grp)

        # Ellipticity
        ell_grp = QGroupBox("Ellipticity")
        ell_form = QFormLayout(ell_grp)
        self._ell_nsamples = QSpinBox()
        self._ell_nsamples.setRange(50, 2000)
        self._ell_nsamples.setValue(500)
        ell_form.addRow("n_samples:", self._ell_nsamples)
        self._ell_nmodes = QSpinBox()
        self._ell_nmodes.setRange(1, 10)
        self._ell_nmodes.setValue(1)
        ell_form.addRow("n_modes:", self._ell_nmodes)
        self._ell_sampling = QComboBox()
        self._ell_sampling.addItems(["log", "linear"])
        ell_form.addRow("Sampling:", self._ell_sampling)
        self._ell_clip = QDoubleSpinBox()
        self._ell_clip.setRange(1.0, 200.0)
        self._ell_clip.setValue(50.0)
        ell_form.addRow("clip_factor:", self._ell_clip)
        lay.addWidget(ell_grp)

        lay.addStretch()
        return w

    # ── Tab: Peak Detection ────────────────────────────────────
    def _build_peak_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self._pk_preset = QComboBox()
        self._pk_preset.addItems(["default", "sesame", "custom"])
        form.addRow("Default Preset:", self._pk_preset)

        self._pk_method = QComboBox()
        self._pk_method.addItems(["find_peaks", "argmax"])
        form.addRow("Method:", self._pk_method)

        self._pk_select = QComboBox()
        self._pk_select.addItems(["leftmost", "highest", "global_max"])
        form.addRow("Selection:", self._pk_select)

        self._pk_min_prom = QDoubleSpinBox()
        self._pk_min_prom.setRange(0.0, 10.0)
        self._pk_min_prom.setValue(0.1)
        self._pk_min_prom.setDecimals(3)
        self._pk_min_prom.setSingleStep(0.01)
        form.addRow("Min Prominence:", self._pk_min_prom)

        return w

    # ── Tab: Dual-Resonance ────────────────────────────────────
    def _build_dr_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self._dr_enable = QCheckBox("Enable by default")
        form.addRow(self._dr_enable)

        self._dr_ratio = QDoubleSpinBox()
        self._dr_ratio.setRange(0.5, 5.0)
        self._dr_ratio.setValue(1.2)
        self._dr_ratio.setSingleStep(0.1)
        form.addRow("Separation Ratio Threshold:", self._dr_ratio)

        self._dr_shift = QDoubleSpinBox()
        self._dr_shift.setRange(0.0, 2.0)
        self._dr_shift.setValue(0.3)
        self._dr_shift.setSingleStep(0.05)
        form.addRow("Separation Shift Threshold:", self._dr_shift)

        return w

    # ── Tab: Plot Defaults ─────────────────────────────────────
    def _build_plot_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self._plot_dpi = QSpinBox()
        self._plot_dpi.setRange(72, 600)
        self._plot_dpi.setValue(200)
        form.addRow("DPI:", self._plot_dpi)

        self._plot_fontsize = QSpinBox()
        self._plot_fontsize.setRange(6, 24)
        self._plot_fontsize.setValue(10)
        form.addRow("Font Size:", self._plot_fontsize)

        self._plot_lw = QDoubleSpinBox()
        self._plot_lw.setRange(0.5, 5.0)
        self._plot_lw.setValue(1.5)
        form.addRow("Line Width:", self._plot_lw)

        self._plot_xscale = QComboBox()
        self._plot_xscale.addItems(["log", "linear"])
        form.addRow("X Scale:", self._plot_xscale)

        self._plot_yscale = QComboBox()
        self._plot_yscale.addItems(["log", "linear"])
        form.addRow("Y Scale:", self._plot_yscale)

        self._plot_grid = QCheckBox("Show Grid")
        self._plot_grid.setChecked(True)
        form.addRow(self._plot_grid)

        return w

    # ── Tab: Paths ─────────────────────────────────────────────
    def _build_paths_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self._path_hvf = QLineEdit()
        row1 = QHBoxLayout()
        row1.addWidget(self._path_hvf)
        btn1 = QPushButton("...")
        btn1.setFixedWidth(30)
        btn1.clicked.connect(
            lambda: self._browse_file(self._path_hvf, "HVf Exe (*.exe)"))
        row1.addWidget(btn1)
        form.addRow("HVf exe path:", row1)

        self._path_gpell = QLineEdit()
        row2 = QHBoxLayout()
        row2.addWidget(self._path_gpell)
        btn2 = QPushButton("...")
        btn2.setFixedWidth(30)
        btn2.clicked.connect(
            lambda: self._browse_file(self._path_gpell, "gpell Exe (*.exe)"))
        row2.addWidget(btn2)
        form.addRow("gpell path:", row2)

        self._path_gitbash = QLineEdit()
        row3 = QHBoxLayout()
        row3.addWidget(self._path_gitbash)
        btn3 = QPushButton("...")
        btn3.setFixedWidth(30)
        btn3.clicked.connect(
            lambda: self._browse_file(self._path_gitbash, "Git Bash (*.exe)"))
        row3.addWidget(btn3)
        form.addRow("Git Bash path:", row3)

        return w

    # ── Load/Get config ────────────────────────────────────────
    def _load_from_config(self):
        cfg = self._config
        # General
        eng = cfg.get("engine", {})
        if isinstance(eng, dict):
            ename = eng.get("name", "diffuse_field")
        else:
            ename = str(eng)
        idx = self._default_engine.findText(ename)
        if idx >= 0:
            self._default_engine.setCurrentIndex(idx)
        fwd = cfg.get("hv_forward", {})
        self._default_fmin.setValue(fwd.get("fmin", 0.2))
        self._default_fmax.setValue(fwd.get("fmax", 20.0))
        self._default_nf.setValue(fwd.get("nf", 71))

        # Engines
        es = cfg.get("engine_settings", {})
        df = es.get("diffuse_field", {})
        self._df_nf.setValue(df.get("nf", 71))
        self._df_nmr.setValue(df.get("nmr", 10))
        self._df_nml.setValue(df.get("nml", 10))
        self._df_nks.setValue(df.get("nks", 10))

        sh = es.get("sh_wave", {})
        self._sh_nsamples.setValue(sh.get("n_samples", 512))
        self._sh_drock.setValue(sh.get("Drock", 0.5))
        self._sh_gamma.setValue(sh.get("gamma_max", 23.0))

        ell = es.get("ellipticity", {})
        self._ell_nsamples.setValue(ell.get("n_samples", 500))
        self._ell_nmodes.setValue(ell.get("n_modes", 1))
        self._ell_clip.setValue(ell.get("clip_factor", 50.0))

        # Peak
        pd = cfg.get("peak_detection", {})
        idx = self._pk_preset.findText(pd.get("preset", "default"))
        if idx >= 0:
            self._pk_preset.setCurrentIndex(idx)
        idx = self._pk_method.findText(pd.get("method", "find_peaks"))
        if idx >= 0:
            self._pk_method.setCurrentIndex(idx)
        idx = self._pk_select.findText(pd.get("select", "leftmost"))
        if idx >= 0:
            self._pk_select.setCurrentIndex(idx)

        # DR
        dr = cfg.get("dual_resonance", {})
        self._dr_enable.setChecked(dr.get("enable", False))
        self._dr_ratio.setValue(dr.get("separation_ratio_threshold", 1.2))
        self._dr_shift.setValue(dr.get("separation_shift_threshold", 0.3))

        # Plot
        plot = cfg.get("plot", {})
        self._plot_dpi.setValue(plot.get("dpi", 200))

        # Paths
        df_exe = es.get("diffuse_field", {}).get("exe_path", "")
        self._path_hvf.setText(df_exe)
        gpell = es.get("ellipticity", {}).get("gpell_path", "")
        self._path_gpell.setText(gpell)
        gitb = es.get("ellipticity", {}).get("git_bash_path", "")
        self._path_gitbash.setText(gitb)

    def get_config(self):
        """Return a config dict reflecting current settings."""
        return {
            "engine": {"name": self._default_engine.currentText()},
            "engine_name": self._default_engine.currentText(),
            "hv_forward": {
                "fmin": self._default_fmin.value(),
                "fmax": self._default_fmax.value(),
                "nf": self._default_nf.value(),
            },
            "engine_settings": {
                "diffuse_field": {
                    "exe_path": self._path_hvf.text(),
                    "nf": self._df_nf.value(),
                    "nmr": self._df_nmr.value(),
                    "nml": self._df_nml.value(),
                    "nks": self._df_nks.value(),
                },
                "sh_wave": {
                    "n_samples": self._sh_nsamples.value(),
                    "sampling": self._sh_sampling.currentText(),
                    "Drock": self._sh_drock.value(),
                    "gamma_max": self._sh_gamma.value(),
                },
                "ellipticity": {
                    "gpell_path": self._path_gpell.text(),
                    "git_bash_path": self._path_gitbash.text(),
                    "n_samples": self._ell_nsamples.value(),
                    "n_modes": self._ell_nmodes.value(),
                    "sampling": self._ell_sampling.currentText(),
                    "clip_factor": self._ell_clip.value(),
                },
            },
            "peak_detection": {
                "preset": self._pk_preset.currentText(),
                "method": self._pk_method.currentText(),
                "select": self._pk_select.currentText(),
                "min_prominence": self._pk_min_prom.value(),
            },
            "dual_resonance": {
                "enable": self._dr_enable.isChecked(),
                "separation_ratio_threshold": self._dr_ratio.value(),
                "separation_shift_threshold": self._dr_shift.value(),
            },
            "plot": {
                "dpi": self._plot_dpi.value(),
                "x_axis_scale": self._plot_xscale.currentText(),
                "y_axis_scale": self._plot_yscale.currentText(),
            },
        }

    # ── Helpers ────────────────────────────────────────────────
    def _browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            line_edit.setText(d)

    def _browse_file(self, line_edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", filter_str)
        if path:
            line_edit.setText(path)
