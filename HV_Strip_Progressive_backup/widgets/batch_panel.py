"""
Batch processing panel.

Provides file list management, batch output directory, batch settings
button, and Run Batch action.
"""

import os
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QAbstractItemView,
)

from ..widgets.collapsible_group import CollapsibleGroup


class BatchPanel(QWidget):
    """Controls for batch processing multiple soil profiles."""

    run_batch_requested = pyqtSignal()
    batch_settings_requested = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── File list ────────────────────────────────────────────────
        grp = CollapsibleGroup('Input Profiles')
        glay = QVBoxLayout()

        info = QLabel('Add model files (.txt, .csv, .xlsx) or directories.')
        info.setStyleSheet('color: gray; font-size: 10px;')
        info.setWordWrap(True)
        glay.addWidget(info)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(150)
        glay.addWidget(self.file_list)

        btn_row = QHBoxLayout()
        add_files = QPushButton('Add Files…')
        add_files.clicked.connect(self._on_add_files)
        add_dir = QPushButton('Add Folder…')
        add_dir.clicked.connect(self._on_add_dir)
        remove_btn = QPushButton('Remove')
        remove_btn.clicked.connect(self._on_remove)
        clear_btn = QPushButton('Clear')
        clear_btn.clicked.connect(self._on_clear)
        for b in (add_files, add_dir, remove_btn, clear_btn):
            btn_row.addWidget(b)
        glay.addLayout(btn_row)

        grp.add_layout(glay)
        layout.addWidget(grp)

        # ── Output directory ──────────────────────────────────────────
        out_grp = CollapsibleGroup('Batch Output')
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText('Batch output directory…')
        out_browse = QPushButton('Browse…')
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_browse)
        out_grp.add_layout(out_row)
        layout.addWidget(out_grp)

        # ── Progress ──────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: gray; font-size: 10px;')
        layout.addWidget(self.status_label)

        # ── Action buttons ────────────────────────────────────────────
        btn_row2 = QHBoxLayout()
        settings_btn = QPushButton('⚙ Batch Settings…')
        settings_btn.clicked.connect(self.batch_settings_requested.emit)
        btn_row2.addWidget(settings_btn)

        self.run_btn = QPushButton('▶ Run Batch')
        self.run_btn.setStyleSheet(
            'background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px;')
        self.run_btn.clicked.connect(self.run_batch_requested.emit)
        btn_row2.addWidget(self.run_btn)

        layout.addLayout(btn_row2)
        layout.addStretch()

        # Wire state signals
        self.state.batch_progress.connect(self._on_progress)
        self.state.batch_done.connect(self._on_batch_done)

    # ── Public API ────────────────────────────────────────────────────

    def get_file_paths(self) -> list:
        """Return list of all file paths in the list."""
        return [self.file_list.item(i).text()
                for i in range(self.file_list.count())]

    # ── Slots ────────────────────────────────────────────────────────

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Add Model Files', '',
            'Model files (*.txt *.csv *.xlsx);;All files (*)')
        for p in paths:
            self.file_list.addItem(p)
        self._update_count()

    def _on_add_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if not path:
            return
        for ext in ('*.txt', '*.csv', '*.xlsx'):
            import glob
            for f in sorted(glob.glob(os.path.join(path, ext))):
                self.file_list.addItem(f)
        self._update_count()

    def _on_remove(self):
        for item in reversed(self.file_list.selectedItems()):
            self.file_list.takeItem(self.file_list.row(item))
        self._update_count()

    def _on_clear(self):
        self.file_list.clear()
        self._update_count()

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, 'Batch Output Directory')
        if path:
            self.out_edit.setText(path)

    def _on_progress(self, pct: int, msg: str):
        self.progress.setVisible(True)
        self.progress.setValue(pct)
        self.status_label.setText(msg)

    def _on_batch_done(self):
        self.progress.setVisible(False)
        self.status_label.setText('Batch processing complete.')

    def _update_count(self):
        n = self.file_list.count()
        self.status_label.setText(f'{n} file(s) queued')
