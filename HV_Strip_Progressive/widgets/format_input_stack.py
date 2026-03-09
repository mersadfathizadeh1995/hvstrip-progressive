"""Format Input Stack — dropdown + stacked widget for all input formats.

Supports: HVf File, CSV File, Excel File, Dinver Files, Simple TXT, Profile Editor.
Compact layout — format-specific area adjusts to current page size,
Vs preview fills remaining canvas space.
"""
from pathlib import Path

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QStackedWidget,
    QMessageBox, QSizePolicy, QFrame, QSplitter,
)

from .style_constants import SECONDARY_LABEL, EMOJI
from .profile_preview_widget import ProfilePreviewWidget

FMT_HVF = "HVf File (.txt)"
FMT_CSV = "CSV File (.csv)"
FMT_EXCEL = "Excel File (.xlsx)"
FMT_DINVER = "Dinver Files"
FMT_TXT = "Simple TXT"
FMT_EDITOR = "Profile Editor"

ALL_FORMATS = [FMT_HVF, FMT_CSV, FMT_EXCEL, FMT_DINVER, FMT_TXT, FMT_EDITOR]

# Format descriptions for tooltip / info
_FMT_HINTS = {
    FMT_HVF: "Standard HV-Forward model file (.txt)",
    FMT_CSV: "CSV with columns: thickness, Vs, Vp, density",
    FMT_EXCEL: "Excel file (.xlsx) with layer data",
    FMT_DINVER: "Dinver inversion output files (_vs.txt, _vp.txt, _rho.txt)",
    FMT_TXT: "Simple depth-Vs text file",
    FMT_EDITOR: "Create or edit layers interactively",
}

# ── Compact reusable widgets ──────────────────────────────────────


class _BrowseRow(QWidget):
    """Compact file browse row: label + path + button."""
    path_changed = pyqtSignal(str)

    def __init__(self, label, filter_str="All Files (*)", parent=None):
        super().__init__(parent)
        self._filter = filter_str
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)
        lbl = QLabel(f"<b>{label}</b>")
        lbl.setFixedWidth(55)
        lay.addWidget(lbl)
        self._path = QLineEdit()
        self._path.setReadOnly(True)
        self._path.setPlaceholderText("No file selected")
        self._path.setStyleSheet(
            "QLineEdit { padding: 3px 6px; border: 1px solid #ccc; "
            "border-radius: 3px; background: white; }")
        lay.addWidget(self._path, 1)
        btn = QPushButton("Browse...")
        btn.setFixedWidth(70)
        btn.setStyleSheet(
            "QPushButton { padding: 3px 8px; border: 1px solid #aaa; "
            "border-radius: 3px; background: #f5f5f5; } "
            "QPushButton:hover { background: #e0e0e0; }")
        btn.clicked.connect(self._browse)
        lay.addWidget(btn)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", self._filter)
        if path:
            self._path.setText(path)
            self.path_changed.emit(path)

    def get_path(self):
        return self._path.text().strip()

    def set_path(self, p):
        self._path.setText(p)


class _DinverPage(QWidget):
    """Dinver format: Vs (required) + Vp (opt) + Density (opt). Compact."""
    vs_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self._vs_row = _BrowseRow("Vs:", "Dinver Vs (*_vs.txt);;All (*)")
        self._vp_row = _BrowseRow("Vp:", "Dinver Vp (*_vp.txt);;All (*)")
        self._rho_row = _BrowseRow("Density:", "Dinver Rho (*_rho.txt);;All (*)")

        lay.addWidget(self._vs_row)
        lay.addWidget(self._vp_row)
        lay.addWidget(self._rho_row)

        hint = QLabel("Vp and Density are optional — auto-detected if adjacent.")
        hint.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self._vs_row.path_changed.connect(self._auto_link)

    def _auto_link(self, vs_path):
        p = Path(vs_path)
        base = p.stem.replace("_vs", "")
        parent = p.parent
        for row, suffix in [(self._vp_row, "_vp"), (self._rho_row, "_rho")]:
            candidate = parent / f"{base}{suffix}.txt"
            if candidate.exists():
                row.set_path(str(candidate))
        self.vs_changed.emit(vs_path)

    def get_paths(self):
        return {
            "vs_file": self._vs_row.get_path(),
            "vp_file": self._vp_row.get_path() or None,
            "rho_file": self._rho_row.get_path() or None,
        }


class _EditorPage(QWidget):
    """Profile Editor with layer table. Uses shared external preview."""
    profile_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        btn_row = QHBoxLayout()
        for text, slot_name in [("New", "_new_profile"),
                                ("Open...", "_open_profile"),
                                ("Save...", "_save_profile")]:
            btn = QPushButton(text)
            btn.setStyleSheet(
                "QPushButton { padding: 3px 10px; border: 1px solid #aaa; "
                "border-radius: 3px; background: #f5f5f5; } "
                "QPushButton:hover { background: #e0e0e0; }")
            btn.clicked.connect(getattr(self, slot_name))
            btn_row.addWidget(btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        try:
            from .layer_table_widget import LayerTableWidget
            self._table = LayerTableWidget()
            lay.addWidget(self._table, 1)
        except Exception:
            self._table = None
            lbl = QLabel("Layer table widget not available")
            lbl.setStyleSheet("color: gray; padding: 20px;")
            lbl.setAlignment(Qt.AlignCenter)
            lay.addWidget(lbl, 1)

    def _new_profile(self):
        if self._table and hasattr(self._table, 'set_default_profile'):
            self._table.set_default_profile()
        self.profile_changed.emit(self.get_profile())

    def _open_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Profile", "",
            "Model Files (*.txt *.csv);;All Files (*)")
        if path:
            try:
                from ..core.soil_profile import SoilProfile                profile = SoilProfile.from_auto(path)
                if self._table and hasattr(self._table, 'set_profile'):
                    self._table.set_profile(profile)
                self.profile_changed.emit(profile)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", str(e))

    def _save_profile(self):
        profile = self.get_profile()
        if not profile:
            QMessageBox.information(self, "Save", "No profile to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Profile", "", "HVf TXT (*.txt);;CSV (*.csv)")
        if path:
            try:
                if path.endswith(".csv"):
                    profile.to_csv(path)
                else:
                    profile.to_hvf_file(path)
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))

    def get_profile(self):
        if self._table and hasattr(self._table, 'get_profile'):
            return self._table.get_profile()
        return None


class _CurrentSizeStack(QStackedWidget):
    """QStackedWidget that sizes to current page only."""

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w else super().minimumSizeHint()


# ── Main widget ───────────────────────────────────────────────────


class FormatInputStack(QWidget):
    """Format dropdown + stacked input widgets for all supported formats.

    Compact layout: format selector at top, format-specific area sizes to
    current page, Vs preview fills remaining space, Load button at bottom.
    """

    profile_loaded = pyqtSignal(object, str)  # (SoilProfile, path_or_"editor")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── Format selector ────────────────────────────────────
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(8)
        lbl = QLabel("Format:")
        lbl.setStyleSheet("font-weight: bold;")
        fmt_row.addWidget(lbl)
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(ALL_FORMATS)
        self._fmt_combo.setStyleSheet(
            "QComboBox { padding: 4px 8px; border: 1px solid #999; "
            "border-radius: 3px; background: white; min-width: 160px; }"
            "QComboBox::drop-down { border: none; }")
        self._fmt_combo.currentIndexChanged.connect(self._on_format_changed)
        fmt_row.addWidget(self._fmt_combo, 1)
        lay.addLayout(fmt_row)

        # ── Format hint label ──────────────────────────────────
        self._hint_label = QLabel(_FMT_HINTS[FMT_HVF])
        self._hint_label.setStyleSheet(
            "color: #666; font-size: 10px; font-style: italic; padding: 0 0 2px 0;")
        self._hint_label.setWordWrap(True)
        lay.addWidget(self._hint_label)

        # ── Separator ──────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #ddd;")
        lay.addWidget(sep)

        # ── Format-specific area (current-size stack) ──────────
        self._stack = _CurrentSizeStack()
        self._stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Page 0-4: File browse rows (wrapped for compact sizing)
        self._hvf_browse = _BrowseRow("File:", "HVf Files (*.txt);;All (*)")
        self._csv_browse = _BrowseRow("File:", "CSV Files (*.csv);;All (*)")
        self._xlsx_browse = _BrowseRow("File:", "Excel Files (*.xlsx);;All (*)")
        self._dinver_page = _DinverPage()
        self._txt_browse = _BrowseRow("File:", "Text Files (*.txt);;All (*)")

        for w in [self._hvf_browse, self._csv_browse, self._xlsx_browse,
                  self._dinver_page, self._txt_browse]:
            self._stack.addWidget(w)

        # Page 5: Profile Editor (table only; preview is shared below)
        self._editor_page = _EditorPage()
        self._stack.addWidget(self._editor_page)

        lay.addWidget(self._stack)

        # ── Vs Profile Preview (stretches to fill) ─────────────
        self._preview = ProfilePreviewWidget()
        self._preview.setMinimumHeight(120)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._preview, 1)

        # ── Load button ────────────────────────────────────────
        self._btn_load = QPushButton("Load Profile")
        self._btn_load.setStyleSheet(
            "QPushButton { background-color: #2E86AB; color: white; "
            "padding: 7px 16px; border-radius: 4px; font-weight: bold; "
            "font-size: 12px; } "
            "QPushButton:hover { background-color: #256E8D; } "
            "QPushButton:pressed { background-color: #1F5A73; }")
        self._btn_load.clicked.connect(self._load_profile)
        lay.addWidget(self._btn_load)

        # ── Status ─────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #666; padding: 2px 0;")
        lay.addWidget(self._status)

        # Connect auto-load on browse
        for browse in [self._hvf_browse, self._csv_browse,
                       self._xlsx_browse, self._txt_browse]:
            browse.path_changed.connect(self._on_file_selected)
        self._dinver_page.vs_changed.connect(self._on_file_selected)
        self._editor_page.profile_changed.connect(self._on_editor_changed)

    def _on_format_changed(self, idx):
        self._stack.setCurrentIndex(idx)
        fmt = ALL_FORMATS[idx]
        self._hint_label.setText(_FMT_HINTS.get(fmt, ""))
        # For editor: stack holds the table, preview still shared
        self._btn_load.setText(
            "Apply Profile" if fmt == FMT_EDITOR else "Load Profile")
        # Force stack to recalculate size
        self._stack.adjustSize()

    def _on_file_selected(self, path):
        self._load_profile()

    def _on_editor_changed(self, profile):
        if profile:
            self._profile = profile
            self._preview.set_profile(profile)
            n = len([L for L in profile.layers if not L.is_halfspace])
            self._status.setText(f"Editor: {n} layers")
            self._status.setStyleSheet("font-size: 11px; color: green;")
            self.profile_loaded.emit(profile, "editor")

    def _load_profile(self):
        idx = self._fmt_combo.currentIndex()
        fmt = ALL_FORMATS[idx]

        try:
            from ..core.soil_profile import SoilProfile
        except ImportError:
            try:
                import importlib
                mod = importlib.import_module(
                    "HV_Strip_Progressive.core.soil_profile")
                SoilProfile = mod.SoilProfile
            except ImportError:
                self._status.setText("Cannot import SoilProfile")
                self._status.setStyleSheet("font-size: 11px; color: red;")
                return

        profile = None
        path = ""

        try:
            if fmt == FMT_HVF:
                path = self._hvf_browse.get_path()
                if path:
                    profile = SoilProfile.from_hvf_file(path)
            elif fmt == FMT_CSV:
                path = self._csv_browse.get_path()
                if path:
                    profile = SoilProfile.from_csv_file(path)
            elif fmt == FMT_EXCEL:
                path = self._xlsx_browse.get_path()
                if path:
                    profile = SoilProfile.from_excel_file(path)
            elif fmt == FMT_DINVER:
                paths = self._dinver_page.get_paths()
                path = paths["vs_file"]
                if path:
                    profile = SoilProfile.from_dinver_files(
                        paths["vs_file"], paths["vp_file"], paths["rho_file"])
            elif fmt == FMT_TXT:
                path = self._txt_browse.get_path()
                if path:
                    profile = SoilProfile.from_txt_file(path)
            elif fmt == FMT_EDITOR:
                profile = self._editor_page.get_profile()
                path = "editor"

            if profile:
                self._profile = profile
                self._preview.set_profile(profile)
                n = len([L for L in profile.layers if not L.is_halfspace])
                src = Path(path).name if path != "editor" else "editor"
                self._status.setText(f"Loaded: {n} layers from {src}")
                self._status.setStyleSheet("font-size: 11px; color: green;")
                self.profile_loaded.emit(profile, path)
            elif not path:
                self._status.setText("Select a file first.")
                self._status.setStyleSheet("font-size: 11px; color: #999;")

        except Exception as e:
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet("font-size: 11px; color: red;")

    def get_profile(self):
        return self._profile

    def get_path(self):
        idx = self._fmt_combo.currentIndex()
        fmt = ALL_FORMATS[idx]
        if fmt == FMT_HVF:
            return self._hvf_browse.get_path()
        elif fmt == FMT_CSV:
            return self._csv_browse.get_path()
        elif fmt == FMT_EXCEL:
            return self._xlsx_browse.get_path()
        elif fmt == FMT_DINVER:
            return self._dinver_page.get_paths()["vs_file"]
        elif fmt == FMT_TXT:
            return self._txt_browse.get_path()
        return ""

    def load_profile(self, path):
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            self._fmt_combo.setCurrentIndex(1)
            self._csv_browse.set_path(path)
        elif ext == ".xlsx":
            self._fmt_combo.setCurrentIndex(2)
            self._xlsx_browse.set_path(path)
        elif "_vs" in Path(path).stem:
            self._fmt_combo.setCurrentIndex(3)
            self._dinver_page._vs_row.set_path(path)
            self._dinver_page._auto_link(path)
        else:
            self._fmt_combo.setCurrentIndex(0)
            self._hvf_browse.set_path(path)
        self._load_profile()
