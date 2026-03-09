"""Save Options Dialog — lets users choose which outputs to save.

Opens from the ⚙ button in the All Profiles view.  Provides checkboxes
for every available figure/output type, an editable output directory,
DPI spinner, and format dropdown.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QLabel, QSpinBox, QComboBox, QLineEdit, QPushButton,
    QDialogButtonBox, QFileDialog,
)


class SaveOptionsDialog(QDialog):
    """Dialog for selecting save options and output directory."""

    def __init__(self, default_dir="", default_dpi=300, default_fmt="PNG",
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Options")
        self.setMinimumWidth(480)
        self._build_ui(default_dir, default_dpi, default_fmt)

    def _build_ui(self, default_dir, default_dpi, default_fmt):
        lay = QVBoxLayout(self)

        # ── Output directory ──────────────────────────────────
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output Directory:"))
        self._dir_edit = QLineEdit(default_dir)
        dir_row.addWidget(self._dir_edit, 1)
        btn_browse = QPushButton("...")
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self._browse)
        dir_row.addWidget(btn_browse)
        lay.addLayout(dir_row)

        # ── Format / DPI ──────────────────────────────────────
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self._fmt = QComboBox()
        self._fmt.addItems(["PNG", "PDF", "SVG"])
        idx = self._fmt.findText(default_fmt)
        if idx >= 0:
            self._fmt.setCurrentIndex(idx)
        self._fmt.setMaximumWidth(80)
        fmt_row.addWidget(self._fmt)

        fmt_row.addWidget(QLabel("DPI:"))
        self._dpi = QSpinBox()
        self._dpi.setRange(72, 600)
        self._dpi.setValue(default_dpi)
        self._dpi.setMaximumWidth(80)
        fmt_row.addWidget(self._dpi)
        fmt_row.addStretch()
        lay.addLayout(fmt_row)

        # ── All-Profile Outputs ───────────────────────────────
        grp_all = QGroupBox("All-Profile Outputs")
        g1 = QVBoxLayout()
        self._chk_combined = QCheckBox("Combined HV overlay + median CSV")
        self._chk_combined.setChecked(True)
        g1.addWidget(self._chk_combined)

        self._chk_hv_vs = QCheckBox("Combined HV + Vs side-by-side (publication)")
        self._chk_hv_vs.setChecked(True)
        g1.addWidget(self._chk_hv_vs)

        self._chk_median_pub = QCheckBox("Median-only H/V ± σ (publication)")
        self._chk_median_pub.setChecked(True)
        g1.addWidget(self._chk_median_pub)

        self._chk_vs_comp = QCheckBox("Vs profile comparison (publication)")
        self._chk_vs_comp.setChecked(True)
        g1.addWidget(self._chk_vs_comp)

        self._chk_summary = QCheckBox("Summary table (CSV + LaTeX)")
        self._chk_summary.setChecked(True)
        g1.addWidget(self._chk_summary)

        self._chk_normalized = QCheckBox("Normalized HV comparison")
        self._chk_normalized.setChecked(False)
        g1.addWidget(self._chk_normalized)

        self._chk_histogram = QCheckBox("f0 distribution histogram")
        self._chk_histogram.setChecked(False)
        g1.addWidget(self._chk_histogram)

        self._chk_f0_vs30 = QCheckBox("f0 vs Vs30 scatter plot")
        self._chk_f0_vs30.setChecked(False)
        g1.addWidget(self._chk_f0_vs30)

        self._chk_spectral = QCheckBox("Spectral ratio matrix (small multiples)")
        self._chk_spectral.setChecked(False)
        g1.addWidget(self._chk_spectral)

        grp_all.setLayout(g1)
        lay.addWidget(grp_all)

        # ── Per-Profile Re-save ───────────────────────────────
        grp_per = QGroupBox("Per-Profile Re-save")
        g2 = QVBoxLayout()
        self._chk_profiles = QCheckBox("Re-save per-profile data (CSV + peaks)")
        self._chk_profiles.setChecked(False)
        g2.addWidget(self._chk_profiles)

        self._chk_hv_figs = QCheckBox("Re-save per-profile HV figures")
        self._chk_hv_figs.setChecked(False)
        g2.addWidget(self._chk_hv_figs)

        self._chk_vs_figs = QCheckBox("Re-save per-profile Vs figures + info")
        self._chk_vs_figs.setChecked(False)
        g2.addWidget(self._chk_vs_figs)

        grp_per.setLayout(g2)
        lay.addWidget(grp_per)

        # ── Buttons ───────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Save Selected")
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self._dir_edit.setText(folder)

    def _validate_and_accept(self):
        if not self._dir_edit.text().strip():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Directory",
                                "Please select an output directory.")
            return
        self.accept()

    def get_options(self):
        """Return dict of all save options."""
        return {
            "output_dir": self._dir_edit.text().strip(),
            "format": self._fmt.currentText(),
            "dpi": self._dpi.value(),
            # All-profile outputs
            "combined_overlay": self._chk_combined.isChecked(),
            "paper_hv_vs": self._chk_hv_vs.isChecked(),
            "normalized_hv": self._chk_normalized.isChecked(),
            "f0_histogram": self._chk_histogram.isChecked(),
            "f0_vs_vs30": self._chk_f0_vs30.isChecked(),
            "spectral_matrix": self._chk_spectral.isChecked(),
            # Per-profile
            "resave_profiles": self._chk_profiles.isChecked(),
            "resave_hv_figures": self._chk_hv_figs.isChecked(),
            "resave_vs_figures": self._chk_vs_figs.isChecked(),
        }
