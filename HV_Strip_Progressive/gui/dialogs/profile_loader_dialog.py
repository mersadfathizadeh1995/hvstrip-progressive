"""Profile Loader Dialog — Separate window for loading soil profiles.

Provides file browser, Dinver import, and layer editor in a dialog.
"""
import os
import tempfile
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QFormLayout, QListWidget,
)

from ..widgets.layer_table_widget import LayerTableWidget
from ..widgets.profile_preview_widget import ProfilePreviewWidget
from ..widgets.style_constants import BUTTON_PRIMARY, BUTTON_SUCCESS


class ProfileLoaderDialog(QDialog):
    """Dialog for loading soil profiles from various sources."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profile Loader")
        self.resize(800, 600)
        self._data = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)

        tabs = QTabWidget()

        # ── Tab 1: From File ────────────────────────────────────
        file_tab = QWidget()
        fl = QVBoxLayout(file_tab)
        row = QHBoxLayout()
        row.addWidget(QLabel("Model File:"))
        self._file_edit = QLineEdit()
        row.addWidget(self._file_edit)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_file)
        row.addWidget(btn)
        fl.addLayout(row)
        self._file_preview = ProfilePreviewWidget()
        fl.addWidget(self._file_preview)
        self._file_status = QLabel("")
        fl.addWidget(self._file_status)
        fl.addStretch()
        tabs.addTab(file_tab, "From File")

        # ── Tab 2: From Dinver ──────────────────────────────────
        din_tab = QWidget()
        dl = QVBoxLayout(din_tab)
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
        dl.addLayout(form)
        btn_load = QPushButton("Load Dinver Profile")
        btn_load.setStyleSheet(BUTTON_PRIMARY)
        btn_load.clicked.connect(self._load_dinver)
        dl.addWidget(btn_load)
        self._din_status = QLabel("")
        dl.addWidget(self._din_status)
        self._din_preview = ProfilePreviewWidget()
        dl.addWidget(self._din_preview)
        dl.addStretch()
        tabs.addTab(din_tab, "Dinver Import")

        # ── Tab 3: Multiple Profiles ────────────────────────────
        multi_tab = QWidget()
        ml = QVBoxLayout(multi_tab)
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Files...")
        btn_add.clicked.connect(self._multi_add)
        btn_dir = QPushButton("Add Dir...")
        btn_dir.clicked.connect(self._multi_add_dir)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self._multi_list.clear())
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_dir)
        btn_row.addWidget(btn_clear)
        ml.addLayout(btn_row)
        self._multi_list = QListWidget()
        ml.addWidget(self._multi_list)
        ml.addStretch()
        tabs.addTab(multi_tab, "Multiple Profiles")

        lay.addWidget(tabs)
        self._tabs = tabs

        # Buttons
        btn_row = QHBoxLayout()
        btn_ok = QPushButton("Load")
        btn_ok.setStyleSheet(BUTTON_SUCCESS)
        btn_ok.clicked.connect(self._on_accept)
        btn_row.addWidget(btn_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        lay.addLayout(btn_row)

    def get_data(self):
        return self._data

    # ── File tab ────────────────────────────────────────────────
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "All Supported (*.txt *.csv *.xlsx);;Text (*.txt);;CSV (*.csv);;Excel (*.xlsx);;All (*)")
        if path:
            self._file_edit.setText(path)
            try:
                from ...core.soil_profile import SoilProfile
                prof = SoilProfile.from_auto(path)
                self._file_preview.set_profile(prof)
                n = len([L for L in prof.layers if not L.is_halfspace])
                self._file_status.setText(
                    f"<span style='color:green;'>Loaded: {n} layers</span>")
            except Exception as e:
                self._file_status.setText(f"<span style='color:red;'>{e}</span>")

    # ── Dinver tab ──────────────────────────────────────────────
    def _browse_dinver(self, ftype):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {ftype.upper()} File", "", "Text (*.txt);;All (*)")
        if not path:
            return
        {"vs": self._din_vs, "vp": self._din_vp, "rho": self._din_rho}[ftype].setText(path)
        if ftype == "vs":
            base = path.replace("_vs.txt", "").replace("_Vs.txt", "")
            for suf, ed in [("_vp.txt", self._din_vp), ("_rho.txt", self._din_rho)]:
                c = base + suf
                if os.path.isfile(c) and not ed.text():
                    ed.setText(c)

    def _load_dinver(self):
        vs = self._din_vs.text().strip()
        if not vs:
            return
        try:
            from ...core.soil_profile import SoilProfile
            prof = SoilProfile.from_dinver_files(
                vs, self._din_vp.text().strip() or None,
                self._din_rho.text().strip() or None)
            self._din_preview.set_profile(prof)
            tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
            tmp.write(prof.to_hvf_format()); tmp.close()
            self._data = {"path": tmp.name, "source": "dinver"}
            self._din_status.setText("<span style='color:green;'>Loaded</span>")
        except Exception as e:
            self._din_status.setText(f"<span style='color:red;'>{e}</span>")

    # ── Multi tab ───────────────────────────────────────────────
    def _multi_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Profiles", "",
            "All Supported (*.txt *.csv *.xlsx);;Text (*.txt);;CSV (*.csv);;Excel (*.xlsx);;All (*)")
        for p in paths:
            self._multi_list.addItem(p)

    def _multi_add_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Add Directory")
        if d:
            for ext in ("*.txt", "*.csv", "*.xlsx"):
                for f in sorted(Path(d).glob(ext)):
                    self._multi_list.addItem(str(f))

    # ── Accept ──────────────────────────────────────────────────
    def _on_accept(self):
        tab = self._tabs.currentIndex()
        if tab == 0:
            path = self._file_edit.text().strip()
            if path and os.path.isfile(path):
                self._data = {"path": path, "source": "file"}
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Select a valid file.")
        elif tab == 1:
            if self._data:
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Load a Dinver profile first.")
        elif tab == 2:
            n = self._multi_list.count()
            if n > 0:
                paths = [self._multi_list.item(i).text() for i in range(n)]
                self._data = {"paths": paths, "source": "multi"}
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Add profile files first.")
