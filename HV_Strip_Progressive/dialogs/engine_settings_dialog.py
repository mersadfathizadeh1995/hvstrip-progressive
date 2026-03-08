"""
Engine settings dialog — 3-tab configuration for all forward engines.

Tab 0: Diffuse Field (HVf.exe) — path, nmr, nml, nks
Tab 1: Rayleigh Ellipticity (gpell.exe) — path, modes, sampling, alpha, Q
Tab 2: SH Wave Transfer — sampling, damping, Darendeli, clip
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QPushButton, QLabel, QFileDialog, QDialogButtonBox,
)


class EngineSettingsDialog(QDialog):
    """Modal dialog for per-engine configuration."""

    def __init__(self, engine_configs: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Engine Settings')
        self.resize(520, 450)
        self._configs = {k: dict(v) for k, v in engine_configs.items()}

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_diffuse_tab()
        self._build_ellipticity_tab()
        self._build_sh_tab()

        # Buttons
        bbox = QDialogButtonBox()
        ok_btn = bbox.addButton(QDialogButtonBox.Ok)
        cancel_btn = bbox.addButton(QDialogButtonBox.Cancel)
        restore_btn = bbox.addButton('Restore Defaults', QDialogButtonBox.ResetRole)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        restore_btn.clicked.connect(self._restore_defaults)
        layout.addWidget(bbox)

    # ── Tab builders ─────────────────────────────────────────────────

    def _build_diffuse_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        cfg = self._configs.get('diffuse_field', {})

        row = QHBoxLayout()
        self.hvf_path = QLineEdit(cfg.get('exe_path', 'HVf.exe'))
        browse = QPushButton('Browse…')
        browse.clicked.connect(lambda: self._browse(self.hvf_path, 'HVf Executable (*.exe)'))
        row.addWidget(self.hvf_path, 1)
        row.addWidget(browse)
        form.addRow('HVf Executable:', row)

        self.nmr_spin = QSpinBox(); self.nmr_spin.setRange(1, 100); self.nmr_spin.setValue(cfg.get('nmr', 10))
        form.addRow('Rayleigh modes (nmr):', self.nmr_spin)
        self.nml_spin = QSpinBox(); self.nml_spin.setRange(1, 100); self.nml_spin.setValue(cfg.get('nml', 10))
        form.addRow('Love modes (nml):', self.nml_spin)
        self.nks_spin = QSpinBox(); self.nks_spin.setRange(1, 100); self.nks_spin.setValue(cfg.get('nks', 10))
        form.addRow('Wavenumber steps (nks):', self.nks_spin)

        self.tabs.addTab(w, 'Diffuse Field')

    def _build_ellipticity_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        cfg = self._configs.get('ellipticity', {})

        for attr, label, default, filt in [
            ('gpell_path', 'gpell Executable:', '', 'gpell (*.exe)'),
            ('git_bash_path', 'Git Bash:', '', 'bash (*.exe)'),
        ]:
            row = QHBoxLayout()
            edit = QLineEdit(cfg.get(attr, default))
            btn = QPushButton('Browse…')
            btn.clicked.connect(lambda _, e=edit, f=filt: self._browse(e, f))
            row.addWidget(edit, 1); row.addWidget(btn)
            form.addRow(f'{label}', row)
            setattr(self, f'ell_{attr}', edit)

        self.ell_modes = QSpinBox(); self.ell_modes.setRange(1, 10); self.ell_modes.setValue(cfg.get('n_modes', 1))
        form.addRow('Rayleigh Modes:', self.ell_modes)

        self.ell_sampling = QComboBox(); self.ell_sampling.addItems(['log', 'frequency', 'period'])
        self.ell_sampling.setCurrentText(cfg.get('sampling', 'log'))
        form.addRow('Sampling:', self.ell_sampling)

        self.ell_alpha = QDoubleSpinBox(); self.ell_alpha.setRange(0, 0.99); self.ell_alpha.setDecimals(2)
        self.ell_alpha.setValue(cfg.get('alpha', 0.0))
        form.addRow('Love mixing (α):', self.ell_alpha)

        self.ell_auto_q = QCheckBox('Auto-compute Qp/Qs')
        self.ell_auto_q.setChecked(cfg.get('auto_q', True))
        form.addRow(self.ell_auto_q)

        self.ell_q_formula = QComboBox(); self.ell_q_formula.addItems(['default', 'brocher', 'constant'])
        self.ell_q_formula.setCurrentText(cfg.get('q_formula', 'default'))
        form.addRow('Q Formula:', self.ell_q_formula)

        self.ell_clip = QSpinBox(); self.ell_clip.setRange(0, 1000); self.ell_clip.setValue(cfg.get('clip_factor', 0))
        form.addRow('Clip Factor:', self.ell_clip)

        self.ell_abs = QCheckBox('Output absolute ellipticity')
        self.ell_abs.setChecked(cfg.get('absolute', False))
        form.addRow(self.ell_abs)

        self.ell_pc = QCheckBox('Peak-refined curves (-pc)')
        self.ell_pc.setChecked(cfg.get('peak_refined', False))
        form.addRow(self.ell_pc)

        self.tabs.addTab(w, 'Ellipticity')

    def _build_sh_tab(self):
        w = QWidget()
        form = QFormLayout(w)
        cfg = self._configs.get('sh_wave', {})

        self.sh_sampling = QComboBox(); self.sh_sampling.addItems(['log', 'linear'])
        self.sh_sampling.setCurrentText(cfg.get('sampling', 'log'))
        form.addRow('Sampling:', self.sh_sampling)

        self.sh_dsoil = QDoubleSpinBox(); self.sh_dsoil.setRange(0, 20); self.sh_dsoil.setDecimals(1)
        self.sh_dsoil.setValue(cfg.get('soil_damping', 0.0)); self.sh_dsoil.setSuffix('%')
        self.sh_dsoil.setSpecialValueText('Auto (Darendeli)')
        form.addRow('Soil Damping:', self.sh_dsoil)

        self.sh_drock = QDoubleSpinBox(); self.sh_drock.setRange(0, 20); self.sh_drock.setDecimals(1)
        self.sh_drock.setValue(cfg.get('rock_damping', 1.0)); self.sh_drock.setSuffix('%')
        form.addRow('Rock Damping:', self.sh_drock)

        self.sh_ref = QComboBox()
        self.sh_ref.addItems(['0 (outcrop)', 'within (top of rock)'])
        self.sh_ref.setCurrentIndex(cfg.get('reference_depth', 0))
        form.addRow('Reference Depth:', self.sh_ref)

        self.sh_darendeli = QComboBox()
        self.sh_darendeli.addItems(['1 — Mean', '2 — Mean + 1σ', '3 — Mean − 1σ'])
        self.sh_darendeli.setCurrentIndex(cfg.get('darendeli_curve', 1) - 1)
        form.addRow('Darendeli Curve:', self.sh_darendeli)

        self.sh_gamma = QDoubleSpinBox(); self.sh_gamma.setRange(10, 30); self.sh_gamma.setDecimals(1)
        self.sh_gamma.setValue(cfg.get('gamma_max', 25.0)); self.sh_gamma.setSuffix(' kN/m³')
        form.addRow('Max Unit Weight:', self.sh_gamma)

        self.sh_clip = QSpinBox(); self.sh_clip.setRange(0, 1000)
        self.sh_clip.setValue(cfg.get('clip_tf', 0)); self.sh_clip.setSpecialValueText('Off')
        form.addRow('Clip TF Above:', self.sh_clip)

        self.tabs.addTab(w, 'SH Wave')

    # ── Helpers ──────────────────────────────────────────────────────

    def _browse(self, edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(self, 'Select File', '', filt)
        if path:
            edit.setText(path)

    def _restore_defaults(self):
        self.hvf_path.setText('HVf.exe')
        self.nmr_spin.setValue(10); self.nml_spin.setValue(10); self.nks_spin.setValue(10)
        self.ell_modes.setValue(1); self.ell_alpha.setValue(0.0); self.ell_clip.setValue(0)
        self.sh_dsoil.setValue(0.0); self.sh_drock.setValue(1.0); self.sh_clip.setValue(0)

    def get_configs(self) -> dict:
        """Return updated engine configs."""
        return {
            'diffuse_field': {
                'exe_path': self.hvf_path.text(),
                'nmr': self.nmr_spin.value(),
                'nml': self.nml_spin.value(),
                'nks': self.nks_spin.value(),
            },
            'ellipticity': {
                'gpell_path': self.ell_gpell_path.text(),
                'git_bash_path': self.ell_git_bash_path.text(),
                'n_modes': self.ell_modes.value(),
                'sampling': self.ell_sampling.currentText(),
                'alpha': self.ell_alpha.value(),
                'auto_q': self.ell_auto_q.isChecked(),
                'q_formula': self.ell_q_formula.currentText(),
                'clip_factor': self.ell_clip.value(),
                'absolute': self.ell_abs.isChecked(),
                'peak_refined': self.ell_pc.isChecked(),
            },
            'sh_wave': {
                'sampling': self.sh_sampling.currentText(),
                'soil_damping': self.sh_dsoil.value(),
                'rock_damping': self.sh_drock.value(),
                'reference_depth': self.sh_ref.currentIndex(),
                'darendeli_curve': self.sh_darendeli.currentIndex() + 1,
                'gamma_max': self.sh_gamma.value(),
                'clip_tf': self.sh_clip.value(),
            },
        }
