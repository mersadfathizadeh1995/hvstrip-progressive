"""Output Viewer Dialog — browse and inspect results folders."""
import os
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QSplitter, QFileDialog,
)


class OutputViewerDialog(QDialog):
    """Browse results directory tree and preview files."""

    def __init__(self, results_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Output Viewer")
        self.resize(800, 600)
        self._results_dir = results_dir
        self._build_ui()
        if results_dir:
            self._load_directory(results_dir)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Top bar
        top = QHBoxLayout()
        top.addWidget(QLabel("Results Directory:"))
        self._dir_label = QLabel(self._results_dir or "Not loaded")
        top.addWidget(self._dir_label, 1)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        top.addWidget(btn_browse)
        layout.addLayout(top)

        # Splitter: tree | preview
        splitter = QSplitter(Qt.Horizontal)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Size"])
        self._tree.itemClicked.connect(self._on_item_clicked)
        splitter.addWidget(self._tree)

        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        splitter.addWidget(self._preview)
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)

        # Close
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select Results Directory")
        if d:
            self._results_dir = d
            self._dir_label.setText(d)
            self._load_directory(d)

    def _load_directory(self, path):
        self._tree.clear()
        root = Path(path)
        if not root.exists():
            return
        root_item = QTreeWidgetItem(self._tree, [root.name, ""])
        root_item.setData(0, Qt.UserRole, str(root))
        self._populate_tree(root_item, root, max_depth=4)
        root_item.setExpanded(True)

    def _populate_tree(self, parent_item, path, depth=0, max_depth=4):
        if depth >= max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                item = QTreeWidgetItem(parent_item, [entry.name, "<dir>"])
                item.setData(0, Qt.UserRole, str(entry))
                self._populate_tree(item, entry, depth + 1, max_depth)
            else:
                size = entry.stat().st_size
                size_str = f"{size:,} B" if size < 1024 else f"{size/1024:.1f} KB"
                item = QTreeWidgetItem(parent_item, [entry.name, size_str])
                item.setData(0, Qt.UserRole, str(entry))

    def _on_item_clicked(self, item):
        path = item.data(0, Qt.UserRole)
        if not path or not os.path.isfile(path):
            self._preview.setPlainText(f"Directory: {path}")
            return

        ext = os.path.splitext(path)[1].lower()
        if ext in (".txt", ".csv", ".yaml", ".yml", ".json", ".log", ".md"):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    content = f.read(50000)
                self._preview.setPlainText(content)
            except Exception as e:
                self._preview.setPlainText(f"Error reading file: {e}")
        elif ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf"):
            self._preview.setPlainText(f"[Image/PDF file: {os.path.basename(path)}]\nSize: {os.path.getsize(path):,} bytes")
        else:
            self._preview.setPlainText(f"Binary file: {os.path.basename(path)}\nSize: {os.path.getsize(path):,} bytes")
