"""Multi-Profile Dialog — load and manage multiple profiles for overlay forward modeling."""
import os
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QGroupBox,
    QCheckBox, QComboBox,
)


class MultiProfileDialog(QDialog):
    """Select multiple profiles for batch forward modeling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multiple Profile Selection")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select multiple soil profiles for overlay forward modeling:"))

        # File list
        grp = QGroupBox("Profiles")
        grp_layout = QVBoxLayout(grp)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Files...")
        btn_add.clicked.connect(self._add_files)
        btn_dir = QPushButton("Add Directory...")
        btn_dir.clicked.connect(self._add_directory)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_selected)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_dir)
        btn_row.addWidget(btn_remove)
        btn_row.addWidget(btn_clear)
        grp_layout.addLayout(btn_row)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        grp_layout.addWidget(self._list)

        self._count_label = QLabel("0 profiles loaded")
        grp_layout.addWidget(self._count_label)
        layout.addWidget(grp)

        # Options
        opt_grp = QGroupBox("Options")
        opt_layout = QVBoxLayout(opt_grp)
        self._chk_overlay = QCheckBox("Overlay all curves on single plot")
        self._chk_overlay.setChecked(True)
        opt_layout.addWidget(self._chk_overlay)
        self._chk_save = QCheckBox("Save individual results")
        self._chk_save.setChecked(True)
        opt_layout.addWidget(self._chk_save)
        layout.addWidget(opt_grp)

        # Buttons
        btn_row2 = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row2.addStretch()
        btn_row2.addWidget(btn_ok)
        btn_row2.addWidget(btn_cancel)
        layout.addLayout(btn_row2)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Profile Files", "",
            "Model Files (*.txt *.csv);;All (*)")
        for p in paths:
            self._list.addItem(p)
        self._update_count()

    def _add_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            for f in sorted(Path(d).glob("*.txt")):
                self._list.addItem(str(f))
            self._update_count()

    def _remove_selected(self):
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))
        self._update_count()

    def _clear(self):
        self._list.clear()
        self._update_count()

    def _update_count(self):
        self._count_label.setText(f"{self._list.count()} profiles loaded")

    def get_file_paths(self):
        return [self._list.item(i).text() for i in range(self._list.count())]

    def get_options(self):
        return {
            "overlay": self._chk_overlay.isChecked(),
            "save_individual": self._chk_save.isChecked(),
        }
