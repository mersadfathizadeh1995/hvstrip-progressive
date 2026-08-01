"""The Data Input tool — the Round-2 FIRST stage.

The unified loader (SKETCH Round 2): every legacy input mode through ONE
format dropdown — HVf .txt · CSV · Excel · **Dinver files** (Vs required +
Vp/ρ optional, sibling auto-link) · Simple TXT · Manual editor — plus
"Add directory…".  Ported from the legacy ``format_input_stack.py``
semantics but routed ENTIRELY through AppState → the api (the legacy stack
called ``core.SoilProfile`` directly).

Canvas views: **Vs Profile** (the ported matplotlib preview, follows the
ProfilesPanel FOCUS) and **Layer Table** (the ported 9-column editable
table; fills on focus; ``Apply changes`` → ``AppState.update_profile``,
which invalidates that profile's downstream results/badges).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.canvas.vs_profile_mpl import VsProfileMpl
from HV_Strip_Progressive.gui.v2.stages.base import PhasePanel, card, status_line
from HV_Strip_Progressive.gui.v2.state import AppState, StripTool
from HV_Strip_Progressive.gui.v2.widgets.house.layer_table import LayerTable

#: (label, api fmt, file filter) — the single-browse formats.
_SINGLE_FORMATS = [
    ("HVf File (.txt)", "hvf", "HVf models (*.txt *.hvf);;All files (*)"),
    ("CSV File (.csv)", "csv", "CSV (*.csv);;All files (*)"),
    ("Excel File (.xlsx)", "excel", "Excel (*.xlsx *.xls);;All files (*)"),
    ("Simple TXT", "simple", "Text (*.txt);;All files (*)"),
]
_FMT_DINVER = "Dinver Files (Vs + Vp + ρ)"
_FMT_EDITOR = "Manual editor"
ALL_FORMATS = [f[0] for f in _SINGLE_FORMATS[:3]] + [_FMT_DINVER] + \
    [_SINGLE_FORMATS[3][0]] + [_FMT_EDITOR]


class _BrowseRow(QWidget):
    """label + path edit + Browse… (one file)."""

    def __init__(self, label: str, file_filter: str, parent=None) -> None:
        super().__init__(parent)
        self._filter = file_filter
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        if label:
            lay.addWidget(QLabel(label))
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Select a file…")
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._on_browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "",
                                              self._filter)
        if path:
            self.edit.setText(path)

    def path(self) -> str:
        return self.edit.text().strip()


class DataToolPanel(PhasePanel):
    """The unified loader (left rail of the Data stage)."""

    tool = StripTool.DATA

    def build(self) -> None:
        self._loading = False
        body = self.scroll_host()
        head = QLabel("Data Input")
        head.setProperty("role", "h2")
        body.addWidget(head)
        body.addWidget(status_line(
            "Load layered models once — every tool (Forward · Strip · "
            "Research) consumes them from the Files panel.", role="caption"))

        # ── Format + per-format input pages ─────────────────────────
        grp, gl = card("Load a profile")
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(ALL_FORMATS)
        frow.addWidget(self.format_combo, 1)
        gl.addLayout(frow)

        self._pages = QStackedWidget()
        self._single_rows = {}
        for label, fmt, ffilter in _SINGLE_FORMATS:
            row = _BrowseRow("", ffilter)
            self._single_rows[label] = (row, fmt)
            holder = QWidget()
            hl = QVBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(row)
            self._pages.addWidget(holder)

        # Dinver page: 3 browse rows, Vs required.
        dinver = QWidget()
        dl = QVBoxLayout(dinver)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(4)
        self.dinver_vs = _BrowseRow("Vs *:", "Dinver Vs (*.txt);;All (*)")
        self.dinver_vp = _BrowseRow("Vp:", "Dinver Vp (*.txt);;All (*)")
        self.dinver_rho = _BrowseRow("ρ:", "Density (*.txt);;All (*)")
        self.dinver_vs.edit.textChanged.connect(self._auto_link_dinver)
        for row in (self.dinver_vs, self.dinver_vp, self.dinver_rho):
            dl.addWidget(row)
        self._pages.addWidget(dinver)

        # Manual editor page: pointer to the table view.
        editor = QWidget()
        el = QVBoxLayout(editor)
        el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(status_line(
            "Edit layers in the Layer Table view, then save it as a new "
            "profile there.", role="muted"))
        self._pages.addWidget(editor)
        gl.addWidget(self._pages)

        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("(optional — file stem)")
        nrow.addWidget(self.name_edit, 1)
        gl.addLayout(nrow)

        lrow = QHBoxLayout()
        self.load_btn = QPushButton("Load profile")
        self.load_btn.setProperty("primary", "true")
        self.load_btn.clicked.connect(self._on_load)
        lrow.addStretch(1)
        lrow.addWidget(self.load_btn)
        gl.addLayout(lrow)
        body.addWidget(grp)

        # Page switching follows the combo (single rows share page order).
        self.format_combo.currentTextChanged.connect(self._on_format_changed)

        # ── Directory ────────────────────────────────────────────────
        grp, gl = card("Load a directory")
        drow = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Folder of model files…")
        dbtn = QPushButton("Browse…")
        dbtn.clicked.connect(self._on_browse_dir)
        drow.addWidget(self.dir_edit, 1)
        drow.addWidget(dbtn)
        gl.addLayout(drow)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Pattern:"))
        self.pattern_edit = QLineEdit("*.txt")
        prow.addWidget(self.pattern_edit, 1)
        self.dir_btn = QPushButton("Load all")
        self.dir_btn.clicked.connect(self._on_load_dir)
        prow.addWidget(self.dir_btn)
        gl.addLayout(prow)
        body.addWidget(grp)

        self._note = status_line("", role="muted")
        body.addWidget(self._note)
        body.addStretch(1)
        self._on_format_changed(self.format_combo.currentText())

    def refresh(self) -> None:
        n = len(self.app_state.profiles() or [])
        self._note.setText(f"{n} profile(s) loaded." if n else
                           "No profiles loaded yet.")

    # ------------------------------------------------------------------
    def _on_format_changed(self, label: str) -> None:
        order = [f[0] for f in _SINGLE_FORMATS[:3]] + [_FMT_DINVER,
                                                       _SINGLE_FORMATS[3][0],
                                                       _FMT_EDITOR]
        # Page order in the stack: 4 single rows, dinver, editor.
        stack_index = {
            _SINGLE_FORMATS[0][0]: 0, _SINGLE_FORMATS[1][0]: 1,
            _SINGLE_FORMATS[2][0]: 2, _SINGLE_FORMATS[3][0]: 3,
            _FMT_DINVER: 4, _FMT_EDITOR: 5,
        }[label]
        self._pages.setCurrentIndex(stack_index)
        self.load_btn.setEnabled(label != _FMT_EDITOR)

    def _auto_link_dinver(self, vs_path: str) -> None:
        """Legacy `_auto_link`: a `*_vs.txt` sibling family auto-fills
        the Vp/ρ rows when those files exist."""
        path = Path(vs_path.strip())
        if not path.is_file():
            return
        m = re.match(r"(.+?)_vs(\.txt)$", path.name, flags=re.IGNORECASE)
        if not m:
            return
        stem, ext = m.group(1), m.group(2)
        for row, suffix in ((self.dinver_vp, "_vp"),
                            (self.dinver_rho, "_rho")):
            if row.path():
                continue
            for cand in (path.parent / f"{stem}{suffix}{ext}",
                         path.parent / f"{stem}{suffix.upper()}{ext}"):
                if cand.is_file():
                    row.edit.setText(str(cand))
                    break

    # ------------------------------------------------------------------
    def _on_load(self) -> None:
        label = self.format_combo.currentText()
        name = self.name_edit.text().strip() or None
        if label == _FMT_DINVER:
            vs = self.dinver_vs.path()
            if not vs:
                self.app_state.error.emit(["Dinver load needs the Vs file."])
                return
            env = self.app_state.load_profile_dinver(
                vs, vp_file=self.dinver_vp.path() or None,
                rho_file=self.dinver_rho.path() or None, name=name)
        else:
            row, fmt = self._single_rows[label]
            path = row.path()
            if not path:
                self.app_state.error.emit(["Select a file to load."])
                return
            env = self.app_state.load_profile(path, name=name, fmt=fmt)
        if env.get("name"):
            self.app_state.set_focus(env["name"])
            self._note.setText(
                f"Loaded '{env['name']}' "
                f"({env['summary']['n_layers']} layers).")

    def _on_browse_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Profiles folder")
        if folder:
            self.dir_edit.setText(folder)

    def _on_load_dir(self) -> None:
        folder = self.dir_edit.text().strip()
        if not folder:
            self.app_state.error.emit(["Select a folder first."])
            return
        env = self.app_state.load_profiles_from_directory(
            folder, self.pattern_edit.text().strip() or "*.txt")
        loaded = env.get("loaded") or []
        errors = env.get("errors") or []
        if loaded:
            self.app_state.set_focus(loaded[-1])
        self._note.setText(
            f"Loaded {len(loaded)} profile(s)"
            + (f", {len(errors)} failed (see Problems)." if errors else "."))


class DataCanvas(QWidget):
    """Vs Profile (mpl) | Layer Table — both follow the panel FOCUS."""

    def __init__(self, app_state: AppState, layer_model=None, parent=None):
        super().__init__(parent)
        self._app = app_state
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs)

        self.vs_view = VsProfileMpl()
        self.tabs.addTab(self.vs_view, "Vs Profile")

        table_view = QWidget()
        tl = QVBoxLayout(table_view)
        tl.setContentsMargins(8, 8, 8, 8)
        tl.setSpacing(6)
        self._table_head = status_line("No profile focused — this is the "
                                       "manual editor.", role="caption")
        tl.addWidget(self._table_head)
        self.layer_table = LayerTable(app_state.suggest_layer_fill)
        tl.addWidget(self.layer_table, 1)
        brow = QHBoxLayout()
        self.apply_btn = QPushButton("Apply changes to profile")
        self.apply_btn.setProperty("primary", "true")
        self.apply_btn.clicked.connect(self._on_apply)
        self.new_btn = QPushButton("New profile from table…")
        self.new_btn.clicked.connect(self._on_new)
        reset_btn = QPushButton("Reset table")
        reset_btn.clicked.connect(self._on_reset)
        brow.addWidget(self.apply_btn)
        brow.addWidget(self.new_btn)
        brow.addWidget(reset_btn)
        brow.addStretch(1)
        tl.addLayout(brow)
        self.tabs.addTab(table_view, "Layer Table")

        app_state.focus_changed.connect(self._on_focus)
        app_state.profiles_changed.connect(self._refresh_focus)
        self._on_focus(app_state.focus)

    # ------------------------------------------------------------------
    def set_palette(self, palette) -> None:
        self.vs_view.apply_theme(palette)

    def _refresh_focus(self) -> None:
        self._on_focus(self._app.focus)

    def _on_focus(self, name) -> None:
        self._loading = True
        try:
            pdict = self._app.profile_dict(name) if name else None
            if pdict:
                info = self._app.profile_info(name) or {}
                self.vs_view.set_profile(pdict, vs30=info.get("vs30"))
                self.layer_table.set_layers(pdict["layers"])
                self._table_head.setText(
                    f"Editing '{name}' — Apply writes back to the session "
                    "and resets its run badges.")
                self.apply_btn.setEnabled(True)
            else:
                self.vs_view.clear()
                self._table_head.setText(
                    "No profile focused — this is the manual editor: build "
                    "layers, then 'New profile from table…'.")
                self.apply_btn.setEnabled(False)
        finally:
            self._loading = False

    # ------------------------------------------------------------------
    def _on_apply(self) -> None:
        name = self._app.focus
        if not name:
            return
        env = self._app.update_profile(name, self.layer_table.get_layers())
        if env.get("success"):
            self._refresh_focus()

    def _on_new(self) -> None:
        layers = self.layer_table.get_layers()
        existing = set(self._app.profile_names())
        base, i = "manual_profile", 1
        name = base
        while name in existing:
            i += 1
            name = f"{base}_{i}"
        env = self._app.add_profile_from_layers(layers, name=name)
        if env.get("name"):
            self._app.set_focus(env["name"])

    def _on_reset(self) -> None:
        if self._app.focus:
            self._refresh_focus()
        else:
            self.layer_table.new_default()


__all__ = ["DataToolPanel", "DataCanvas"]
