"""Multi Input View — canvas tab for loading multiple soil profiles.

Organized profile list with add/remove, editor support, and
Load Results Folder for re-opening previously analyzed projects.
Used by Forward Multiple mode.
"""
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QGroupBox,
    QSizePolicy, QMessageBox,
)

from ..widgets.style_constants import SECONDARY_LABEL

# File filters
_FILE_FILTER = (
    "Model Files (*.txt *.csv *.xlsx);;HVf (*.txt);;CSV (*.csv);;"
    "Excel (*.xlsx);;All Files (*)")
_DINVER_FILTER = "Dinver Vs (*_vs.txt);;All (*)"

# Format icons for the list
_FORMAT_ICONS = {
    "auto": "📄", "hvf": "📄", "csv": "📊", "xlsx": "📊",
    "dinver": "🔧", "editor": "✏️", "txt": "📄",
}


class MultiInputView(QWidget):
    """Canvas view for loading multiple soil profiles."""

    profiles_changed = pyqtSignal(int)  # emits count

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._profiles = []  # list of (name, path_or_data, format_hint)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        grp_style = (
            "QGroupBox { font-weight: bold; font-size: 11px; "
            "border: 1px solid #ccc; border-radius: 4px; "
            "margin-top: 10px; padding: 8px 6px 6px 6px; } "
            "QGroupBox::title { subcontrol-origin: margin; "
            "left: 8px; padding: 0 4px; color: #333; }")

        btn_style = (
            "QPushButton { padding: 4px 10px; border: 1px solid #aaa; "
            "border-radius: 3px; background: #f8f8f8; font-size: 11px; } "
            "QPushButton:hover { background: #e8e8e8; border-color: #888; }")

        # ── Section 1: Add New Profiles ────────────────────────
        new_grp = QGroupBox("Add New Profiles  —  for forward computation")
        new_grp.setStyleSheet(grp_style)
        new_lay = QVBoxLayout(new_grp)
        new_lay.setSpacing(4)

        add_row = QHBoxLayout()
        add_row.setSpacing(4)

        self._btn_add_file = QPushButton("📄 File...")
        self._btn_add_file.setToolTip(
            "Add one or more profile files (.txt, .csv, .xlsx)\n"
            "Supported formats: HVf, CSV, Excel")
        self._btn_add_dir = QPushButton("📁 Directory...")
        self._btn_add_dir.setToolTip(
            "Add all .txt profiles from a folder at once")
        self._btn_add_dinver = QPushButton("🔧 Dinver...")
        self._btn_add_dinver.setToolTip(
            "Add Dinver inversion output (_vs.txt)")
        self._btn_add_editor = QPushButton("✏️ Editor...")
        self._btn_add_editor.setToolTip(
            "Create a soil profile interactively\nusing the built-in layer editor")

        for btn in [self._btn_add_file, self._btn_add_dir,
                    self._btn_add_dinver, self._btn_add_editor]:
            btn.setStyleSheet(btn_style)
            add_row.addWidget(btn)
        add_row.addStretch()
        new_lay.addLayout(add_row)

        hint = QLabel(
            "Mix any combination of formats. Each profile will be "
            "computed individually, then combined for comparison.")
        hint.setStyleSheet("color: #777; font-size: 10px; font-style: italic;")
        hint.setWordWrap(True)
        new_lay.addWidget(hint)
        lay.addWidget(new_grp)

        # ── Section 2: Open Existing Project ───────────────────
        proj_grp = QGroupBox(
            "Open Existing Project  —  re-analyze previously computed results")
        proj_grp.setStyleSheet(
            grp_style.replace("#ccc", "#2E86AB").replace("#333", "#2E86AB"))
        proj_lay = QVBoxLayout(proj_grp)
        proj_lay.setSpacing(4)

        self._btn_load_results = QPushButton(
            "📂  Load Results Folder...")
        self._btn_load_results.setToolTip(
            "Open a folder of previously computed multi-profile results.\n"
            "Each subfolder should contain hv_curve.csv and peak_info.txt.\n\n"
            "Use this to:\n"
            "  • Re-view and compare H/V curves\n"
            "  • Re-pick peaks on existing curves\n"
            "  • Compute median curves and statistics\n"
            "  • Re-export figures with new settings")
        self._btn_load_results.setStyleSheet(
            "QPushButton { padding: 6px 14px; border: 1px solid #2E86AB; "
            "border-radius: 3px; background: #EBF5FB; color: #2E86AB; "
            "font-weight: bold; font-size: 11px; } "
            "QPushButton:hover { background: #D6EAF8; }")
        proj_lay.addWidget(self._btn_load_results)

        proj_hint = QLabel(
            "Opens a directory that was previously used as output for "
            "multi-profile analysis. Loads all computed H/V curves, peaks, "
            "and settings so you can adjust figures, re-pick peaks, or "
            "export with different themes without re-computing.")
        proj_hint.setStyleSheet(
            "color: #666; font-size: 10px; font-style: italic;")
        proj_hint.setWordWrap(True)
        proj_lay.addWidget(proj_hint)
        lay.addWidget(proj_grp)

        # ── Profile list + side controls ───────────────────────
        list_label = QLabel("Loaded Profiles:")
        list_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        lay.addWidget(list_label)

        list_row = QHBoxLayout()
        list_row.setSpacing(6)

        self._profile_list = QListWidget()
        self._profile_list.setAlternatingRowColors(True)
        self._profile_list.setStyleSheet(
            "QListWidget { border: 1px solid #ccc; border-radius: 3px; "
            "background: white; } "
            "QListWidget::item { padding: 3px 6px; } "
            "QListWidget::item:selected { background: #D6EAF8; color: black; }")
        list_row.addWidget(self._profile_list, 1)

        side = QVBoxLayout()
        side.setSpacing(4)
        side_style = (
            "QPushButton { padding: 3px 8px; border: 1px solid #bbb; "
            "border-radius: 3px; background: #f5f5f5; font-size: 10px; "
            "min-width: 65px; } "
            "QPushButton:hover { background: #e0e0e0; }")

        self._btn_remove = QPushButton("Remove")
        self._btn_clear = QPushButton("Clear All")
        self._btn_move_up = QPushButton("Move Up")
        self._btn_move_down = QPushButton("Move Down")

        for btn in [self._btn_remove, self._btn_clear]:
            btn.setStyleSheet(side_style)
            side.addWidget(btn)
        side.addStretch()
        for btn in [self._btn_move_up, self._btn_move_down]:
            btn.setStyleSheet(side_style)
            side.addWidget(btn)
        list_row.addLayout(side)
        lay.addLayout(list_row, 1)

        # ── Count label ────────────────────────────────────────
        self._count_label = QLabel("0 profiles loaded")
        self._count_label.setStyleSheet(
            "color: #666; font-size: 11px; padding: 2px 0;")
        lay.addWidget(self._count_label)

        # ── Connections ────────────────────────────────────────
        self._btn_add_file.clicked.connect(self._add_file)
        self._btn_add_dir.clicked.connect(self._add_directory)
        self._btn_add_dinver.clicked.connect(self._add_dinver)
        self._btn_add_editor.clicked.connect(self._add_editor)
        self._btn_load_results.clicked.connect(self._load_results_folder)
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_clear.clicked.connect(self._clear_all)
        self._btn_move_up.clicked.connect(self._move_up)
        self._btn_move_down.clicked.connect(self._move_down)

    # ── Add methods ────────────────────────────────────────────

    def _add_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Profile Files", "", _FILE_FILTER)
        for p in paths:
            name = Path(p).stem
            ext = Path(p).suffix.lower()
            hint = {"csv": "csv", ".xlsx": "xlsx"}.get(ext, "auto")
            self._profiles.append((name, p, hint))
            icon = _FORMAT_ICONS.get(hint, "📄")
            self._profile_list.addItem(f"{icon}  {name}  ({ext})")
        self._update_count()

    def _add_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Add Directory of Profiles")
        if d:
            added = 0
            for f in sorted(Path(d).glob("*.txt")):
                name = f.stem
                if not any(n == name for n, _, _ in self._profiles):
                    self._profiles.append((name, str(f), "auto"))
                    self._profile_list.addItem(f"📄  {name}  (.txt)")
                    added += 1
            if added and self._mw:
                self._mw.log(f"Added {added} profiles from {Path(d).name}/")
            self._update_count()

    def _add_dinver(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Dinver Vs File", "", _DINVER_FILTER)
        if path:
            name = Path(path).stem.replace("_vs", "")
            self._profiles.append((name, path, "dinver"))
            self._profile_list.addItem(f"🔧  {name}  (dinver)")
            self._update_count()

    def _add_editor(self):
        """Open profile editor dialog to create a new profile."""
        try:
            from ..dialogs.profile_loader_dialog import ProfileLoaderDialog
            dlg = ProfileLoaderDialog(parent=self)
            if dlg.exec_() == ProfileLoaderDialog.Accepted:
                data = dlg.get_data()
                if data:
                    name = data.get("name", f"editor_{len(self._profiles)}")
                    self._profiles.append((name, data, "editor"))
                    self._profile_list.addItem(f"✏️  {name}  (editor)")
                    self._update_count()
        except ImportError:
            # Fallback: use a simple file dialog instead
            QMessageBox.information(
                self, "Editor",
                "Profile editor dialog not available.\n"
                "Use 'Add File...' to load an existing profile.")
        except Exception as e:
            if self._mw:
                self._mw.log(f"Editor error: {e}")

    def _load_results_folder(self):
        """Load previously computed multi-profile results from a directory.

        Expected structure:
          folder/
            Profile_1/ → hv_curve.csv, peak_info.txt
            Profile_2/ → hv_curve.csv, peak_info.txt
            ...
        """
        folder = QFileDialog.getExistingDirectory(
            self, "Select Results Folder (previously computed profiles)")
        if not folder:
            return

        p = Path(folder)
        loaded = 0
        for sub in sorted(p.iterdir()):
            if sub.is_dir():
                # Look for hv_curve.csv or any .txt profile file
                csv_file = sub / "hv_curve.csv"
                profile_files = list(sub.glob("*.txt"))
                if csv_file.exists() or profile_files:
                    name = sub.name
                    if not any(n == name for n, _, _ in self._profiles):
                        # Store the directory path as data
                        self._profiles.append((name, str(sub), "results"))
                        self._profile_list.addItem(
                            f"📁  {name}  (results)")
                        loaded += 1

        if loaded:
            self._update_count()
            if self._mw:
                self._mw.log(
                    f"Loaded {loaded} profile results from {p.name}/")
        else:
            QMessageBox.information(
                self, "No Results Found",
                f"No profile result subdirectories found in:\n{folder}\n\n"
                "Expected structure: each subfolder containing\n"
                "hv_curve.csv and/or peak_info.txt")

    # ── List management ────────────────────────────────────────

    def _remove_selected(self):
        row = self._profile_list.currentRow()
        if row >= 0:
            self._profile_list.takeItem(row)
            self._profiles.pop(row)
            self._update_count()

    def _clear_all(self):
        self._profile_list.clear()
        self._profiles.clear()
        self._update_count()

    def _move_up(self):
        row = self._profile_list.currentRow()
        if row > 0:
            item = self._profile_list.takeItem(row)
            self._profile_list.insertItem(row - 1, item)
            self._profile_list.setCurrentRow(row - 1)
            self._profiles[row - 1], self._profiles[row] = \
                self._profiles[row], self._profiles[row - 1]

    def _move_down(self):
        row = self._profile_list.currentRow()
        if 0 <= row < self._profile_list.count() - 1:
            item = self._profile_list.takeItem(row)
            self._profile_list.insertItem(row + 1, item)
            self._profile_list.setCurrentRow(row + 1)
            self._profiles[row], self._profiles[row + 1] = \
                self._profiles[row + 1], self._profiles[row]

    def _update_count(self):
        n = len(self._profiles)
        self._count_label.setText(
            f"{n} profile{'s' if n != 1 else ''} loaded")
        self.profiles_changed.emit(n)

    # ── Public API ─────────────────────────────────────────────

    def get_profiles(self):
        return self._profiles

    def set_batch_folder(self, folder):
        for f in sorted(Path(folder).glob("*.txt")):
            name = f.stem
            if not any(n == name for n, _, _ in self._profiles):
                self._profiles.append((name, str(f), "auto"))
                self._profile_list.addItem(f"📄  {name}  (.txt)")
        self._update_count()
