"""Batch Input View — canvas tab for loading batch profile files.

File list with add/remove buttons. Used by Strip Batch mode.
"""
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QFileDialog,
)

from ..widgets.style_constants import SECONDARY_LABEL


class BatchInputView(QWidget):
    """Canvas view for loading batch profile files for stripping."""

    files_changed = pyqtSignal(int)  # emits count

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._file_list = []  # list of file paths
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        header = QLabel("Batch Profile Files")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        lay.addWidget(header)

        desc = QLabel(
            "Add soil profile files or entire directories for batch "
            "stripping analysis. Each file will be processed independently.")
        desc.setWordWrap(True)
        desc.setStyleSheet(SECONDARY_LABEL)
        lay.addWidget(desc)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add Files...")
        self._btn_add_dir = QPushButton("Add Directory...")
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_add_dir)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # List + side buttons
        list_row = QHBoxLayout()
        self._batch_list = QListWidget()
        self._batch_list.setMinimumHeight(200)
        list_row.addWidget(self._batch_list, 1)

        side_btns = QVBoxLayout()
        self._btn_remove = QPushButton("Remove")
        self._btn_clear = QPushButton("Clear All")
        side_btns.addWidget(self._btn_remove)
        side_btns.addWidget(self._btn_clear)
        side_btns.addStretch()
        list_row.addLayout(side_btns)
        lay.addLayout(list_row)

        # Count
        self._count_label = QLabel("0 files")
        self._count_label.setStyleSheet(SECONDARY_LABEL)
        lay.addWidget(self._count_label)

        lay.addStretch()

        # Connections
        self._btn_add.clicked.connect(self._add_files)
        self._btn_add_dir.clicked.connect(self._add_directory)
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_clear.clicked.connect(self._clear_all)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Profile Files", "",
            "Model Files (*.txt *.csv);;All Files (*)")
        for p in paths:
            if p not in self._file_list:
                self._file_list.append(p)
                self._batch_list.addItem(Path(p).name)
        self._update_count()

    def _add_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Add Directory")
        if d:
            for f in sorted(Path(d).glob("*.txt")):
                fp = str(f)
                if fp not in self._file_list:
                    self._file_list.append(fp)
                    self._batch_list.addItem(f.name)
            self._update_count()

    def _remove_selected(self):
        row = self._batch_list.currentRow()
        if row >= 0:
            self._batch_list.takeItem(row)
            self._file_list.pop(row)
            self._update_count()

    def _clear_all(self):
        self._batch_list.clear()
        self._file_list.clear()
        self._update_count()

    def _update_count(self):
        n = len(self._file_list)
        self._count_label.setText(f"{n} file{'s' if n != 1 else ''}")
        self.files_changed.emit(n)

    # ── Public API ─────────────────────────────────────────────
    def get_files(self):
        return list(self._file_list)

    def set_batch_folder(self, folder):
        for f in sorted(Path(folder).glob("*.txt")):
            fp = str(f)
            if fp not in self._file_list:
                self._file_list.append(fp)
                self._batch_list.addItem(f.name)
        self._update_count()
