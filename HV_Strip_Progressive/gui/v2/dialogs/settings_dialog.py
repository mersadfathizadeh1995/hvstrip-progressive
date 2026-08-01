"""``SettingsDialog`` — the ONE global settings surface (the split design).

Run-relevant knobs live in the tool panels' cards; this dialog holds only
the GLOBAL things: engine binary paths (HVf · gpell · git-bash) and output
defaults, all bound to ``HVStripConfig`` through AppState.  Replaces the
legacy trio (settings_window / config_panel / settings_page).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.state.app_state import AppState


class SettingsDialog(QDialog):
    """Engine binary paths + output defaults (global scope only)."""

    def __init__(self, app_state: AppState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._app = app_state
        self.setWindowTitle("HV Strip — Settings")
        self.setMinimumWidth(560)

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        head = QLabel("Engine binaries")
        head.setProperty("role", "h2")
        outer.addWidget(head)
        outer.addWidget(self._caption(
            "Existence is probed, never executed here. An unavailable engine "
            "shows as a status badge — it never crashes a run setup."))

        form = QGridLayout()
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(6)
        cfg = self._app.config
        self._hvf = self._path_row(
            form, 0, "HVf executable:", cfg.engine.exe_path,
            "Executables (*.exe);;All files (*)")
        self._gpell = self._path_row(
            form, 1, "gpell (Geopsy):", cfg.engine.gpell_path,
            "Executables (*.exe);;All files (*)")
        self._bash = self._path_row(
            form, 2, "Git Bash:", cfg.engine.git_bash_path,
            "Executables (*.exe);;All files (*)")
        outer.addLayout(form)

        head2 = QLabel("Output")
        head2.setProperty("role", "h2")
        outer.addWidget(head2)
        oform = QGridLayout()
        oform.setHorizontalSpacing(6)
        self._out_dir = self._dir_row(
            oform, 0, "Default output folder:", cfg.output.output_dir or "")
        outer.addLayout(oform)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # ------------------------------------------------------------------
    @staticmethod
    def _caption(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("role", "caption")
        lbl.setWordWrap(True)
        return lbl

    def _path_row(self, form, row: int, label: str, value: str,
                  filt: str) -> QLineEdit:
        form.addWidget(QLabel(label), row, 0)
        edit = QLineEdit(value or "")
        form.addWidget(edit, row, 1)
        btn = QPushButton("Browse…")

        def _browse() -> None:
            path, _ = QFileDialog.getOpenFileName(self, label, "", filt)
            if path:
                edit.setText(path)

        btn.clicked.connect(_browse)
        form.addWidget(btn, row, 2)
        form.setColumnStretch(1, 1)
        return edit

    def _dir_row(self, form, row: int, label: str, value: str) -> QLineEdit:
        form.addWidget(QLabel(label), row, 0)
        edit = QLineEdit(value or "")
        form.addWidget(edit, row, 1)
        btn = QPushButton("Browse…")

        def _browse() -> None:
            folder = QFileDialog.getExistingDirectory(self, label)
            if folder:
                edit.setText(folder)

        btn.clicked.connect(_browse)
        form.addWidget(btn, row, 2)
        form.setColumnStretch(1, 1)
        return edit

    def _on_ok(self) -> None:
        self._app.update_config(
            "engine",
            exe_path=self._hvf.text().strip(),
            gpell_path=self._gpell.text().strip(),
            git_bash_path=self._bash.text().strip(),
        )
        self._app.update_config(
            "output", output_dir=self._out_dir.text().strip())
        self.accept()


__all__ = ["SettingsDialog"]
