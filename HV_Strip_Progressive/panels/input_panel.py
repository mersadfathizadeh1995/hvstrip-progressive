"""Input Panel — Profile loading, layer editing, output directory.

Replaces the old home_page + forward_modeling_page input sections.
"""
import os
import tempfile
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QRadioButton, QButtonGroup, QFormLayout, QTabWidget,
)

from ..widgets.collapsible_group import CollapsibleGroupBox
from ..widgets.profile_preview_widget import ProfilePreviewWidget
from ..widgets.layer_table_widget import LayerTableWidget
from ..widgets.style_constants import (
    OUTER_MARGINS, SECONDARY_LABEL, BUTTON_PRIMARY, EMOJI,
)


class InputPanel(QWidget):
    """Left-panel tab for all profile input methods."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._active_profile = None
        self._temp_model_path = None
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(*OUTER_MARGINS)

        # ── Profile Source ──────────────────────────────────────
        src = CollapsibleGroupBox(f"{EMOJI['file']} Profile Source")
        src_lay = QVBoxLayout()

        self._input_tabs = QTabWidget()
        self._input_tabs.setMaximumHeight(320)
        self._input_tabs.addTab(self._build_file_tab(), "From File")
        self._input_tabs.addTab(self._build_dinver_tab(), "Dinver")
        self._input_tabs.addTab(self._build_editor_tab(), "Editor")
        src_lay.addWidget(self._input_tabs)
        src.setContentLayout(src_lay)
        lay.addWidget(src)

        # ── Profile Preview ─────────────────────────────────────
        prev = CollapsibleGroupBox(f"{EMOJI['profile']} Vs Preview")
        prev_lay = QVBoxLayout()
        self._preview = ProfilePreviewWidget()
        prev_lay.addWidget(self._preview)
        prev.setContentLayout(prev_lay)
        lay.addWidget(prev)

        # ── Output Directory ────────────────────────────────────
        out = CollapsibleGroupBox(f"{EMOJI['folder']} Output Directory")
        out_lay = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Select output directory for results")
        out_lay.addWidget(self._output_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_output)
        out_lay.addWidget(btn)
        out.setContentLayout(out_lay)
        lay.addWidget(out)

        lay.addStretch()
        scroll.setWidget(inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    # ── File tab ────────────────────────────────────────────────
    def _build_file_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("Model File:"))
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("HVf model file (.txt)")
        row.addWidget(self._file_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_model_file)
        row.addWidget(btn)
        layout.addLayout(row)
        self._file_status = QLabel("")
        self._file_status.setStyleSheet(SECONDARY_LABEL)
        layout.addWidget(self._file_status)
        layout.addStretch()
        return w

    # ── Dinver tab ──────────────────────────────────────────────
    def _build_dinver_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        form = QFormLayout()
        self._din_vs = QLineEdit()
        btn_vs = QPushButton("..."); btn_vs.setFixedWidth(30)
        btn_vs.clicked.connect(lambda: self._browse_dinver("vs"))
        r = QHBoxLayout(); r.addWidget(self._din_vs); r.addWidget(btn_vs)
        form.addRow("Vs File:", r)

        self._din_vp = QLineEdit(); self._din_vp.setPlaceholderText("Optional")
        btn_vp = QPushButton("..."); btn_vp.setFixedWidth(30)
        btn_vp.clicked.connect(lambda: self._browse_dinver("vp"))
        r2 = QHBoxLayout(); r2.addWidget(self._din_vp); r2.addWidget(btn_vp)
        form.addRow("Vp File:", r2)

        self._din_rho = QLineEdit(); self._din_rho.setPlaceholderText("Optional")
        btn_rho = QPushButton("..."); btn_rho.setFixedWidth(30)
        btn_rho.clicked.connect(lambda: self._browse_dinver("rho"))
        r3 = QHBoxLayout(); r3.addWidget(self._din_rho); r3.addWidget(btn_rho)
        form.addRow("Density:", r3)
        layout.addLayout(form)

        btn_load = QPushButton("Load Dinver Profile")
        btn_load.setStyleSheet(BUTTON_PRIMARY)
        btn_load.clicked.connect(self._load_dinver_profile)
        layout.addWidget(btn_load)
        self._din_status = QLabel("")
        self._din_status.setStyleSheet(SECONDARY_LABEL)
        layout.addWidget(self._din_status)
        layout.addStretch()
        return w

    # ── Editor tab ──────────────────────────────────────────────
    def _build_editor_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        btn_row = QHBoxLayout()
        for label, slot in [("New", self._editor_new),
                            ("Open", self._editor_open),
                            ("Save", self._editor_save)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._layer_table = LayerTableWidget()
        self._layer_table.profile_changed.connect(self._on_editor_changed)
        layout.addWidget(self._layer_table)
        return w

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API  (called from strip_window)
    # ══════════════════════════════════════════════════════════════
    def load_profile(self, path):
        """Load a profile from file path."""
        self._file_edit.setText(path)
        self._browse_model_file_at(path)

    def load_dinver(self):
        """Open the Dinver tab and let user browse."""
        self._input_tabs.setCurrentIndex(1)

    def set_profile_data(self, data):
        """Set profile from ProfileLoaderDialog result."""
        path = data.get("path")
        if path:
            self.load_profile(path)

    def get_model_path(self):
        """Return the current model file path (temp or real)."""
        tab = self._input_tabs.currentIndex()
        if tab == 0:
            return self._file_edit.text().strip()
        elif tab == 1:
            return self._temp_model_path
        elif tab == 2:
            return self._get_editor_model_path()
        return None

    def get_output_dir(self):
        return self._output_edit.text().strip()

    def get_profile(self):
        return self._active_profile

    # ══════════════════════════════════════════════════════════════
    #  INTERNALS
    # ══════════════════════════════════════════════════════════════
    def _browse_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "Model Files (*.txt *.csv);;All (*)")
        if path:
            self._browse_model_file_at(path)

    def _browse_model_file_at(self, path):
        self._file_edit.setText(path)
        try:
            from core.soil_profile import SoilProfile
            prof = SoilProfile.from_auto(path)
            self._active_profile = prof
            self._preview.set_profile(prof)
            n = len([L for L in prof.layers if not L.is_halfspace])
            self._file_status.setText(
                f"<span style='color:green;'>Loaded: {n} layers</span>")
            if self._mw:
                self._mw.update_vs_profile(prof)
        except Exception as e:
            self._file_status.setText(f"<span style='color:red;'>Error: {e}</span>")

    def _browse_dinver(self, ftype):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {ftype.upper()} File", "",
            "Text Files (*.txt);;All (*)")
        if not path:
            return
        target = {"vs": self._din_vs, "vp": self._din_vp, "rho": self._din_rho}[ftype]
        target.setText(path)
        if ftype == "vs":
            base = path.replace("_vs.txt", "").replace("_Vs.txt", "")
            for suffix, edit in [("_vp.txt", self._din_vp), ("_rho.txt", self._din_rho)]:
                cand = base + suffix
                if os.path.isfile(cand) and not edit.text():
                    edit.setText(cand)

    def _load_dinver_profile(self):
        vs_path = self._din_vs.text().strip()
        if not vs_path:
            QMessageBox.warning(self, "Error", "Select a Vs file first.")
            return
        try:
            from core.soil_profile import SoilProfile
            vp = self._din_vp.text().strip() or None
            rho = self._din_rho.text().strip() or None
            prof = SoilProfile.from_dinver_files(vs_path, vp, rho)
            self._active_profile = prof
            self._preview.set_profile(prof)
            tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
            tmp.write(prof.to_hvf_format())
            tmp.close()
            self._temp_model_path = tmp.name
            n = len([L for L in prof.layers if not L.is_halfspace])
            self._din_status.setText(
                f"<span style='color:green;'>Loaded: {n} layers</span>")
            if self._mw:
                self._mw.update_vs_profile(prof)
        except Exception as e:
            self._din_status.setText(f"<span style='color:red;'>Error: {e}</span>")

    def _editor_new(self):
        self._layer_table._new_default()
        self._on_editor_changed()

    def _editor_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Profile", "", "Model Files (*.txt *.csv);;All (*)")
        if path:
            try:
                from core.soil_profile import SoilProfile
                prof = SoilProfile.from_auto(path)
                self._layer_table.set_profile(prof)
                self._on_editor_changed()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load: {e}")

    def _editor_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", "", "HVf Files (*.txt);;CSV (*.csv)")
        if path:
            try:
                prof = self._layer_table.get_profile()
                if path.endswith(".csv"):
                    prof.save_csv(path)
                else:
                    prof.save_hvf(path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def _on_editor_changed(self):
        try:
            prof = self._layer_table.get_profile()
            self._active_profile = prof
            self._preview.set_profile(prof)
            if self._mw:
                self._mw.update_vs_profile(prof)
        except Exception:
            pass

    def _get_editor_model_path(self):
        try:
            prof = self._layer_table.get_profile()
            valid, msgs = prof.validate()
            if not valid:
                return None
            tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
            tmp.write(prof.to_hvf_format())
            tmp.close()
            self._temp_model_path = tmp.name
            return tmp.name
        except Exception:
            return None

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_edit.setText(d)
