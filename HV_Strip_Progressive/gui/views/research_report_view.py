"""ResearchReportView — generated report file browser and preview."""

import os
import subprocess
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QSplitter,
    QFileDialog,
)


class ResearchReportView(QWidget):
    """Canvas view showing the generated report file tree with preview.

    Displays all files from the report generation phase (CSVs, JSON,
    LaTeX tables, figures) in a tree structure with a text/image preview.
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._files = {}
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>📄 Report Files</b>"))
        header.addStretch()

        self._btn_open_folder = QPushButton("📂 Open in Explorer")
        self._btn_open_folder.clicked.connect(self._open_folder)
        header.addWidget(self._btn_open_folder)

        self._btn_refresh = QPushButton("🔄 Refresh")
        self._btn_refresh.clicked.connect(self._refresh)
        header.addWidget(self._btn_refresh)
        lay.addLayout(header)

        # Splitter: file tree on left, preview on right
        splitter = QSplitter(Qt.Horizontal)

        # File tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Path"])
        self._tree.setColumnWidth(0, 250)
        self._tree.itemClicked.connect(self._on_file_selected)
        splitter.addWidget(self._tree)

        # Preview
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; "
            "font-size: 11px;")
        splitter.addWidget(self._preview)

        splitter.setSizes([300, 500])
        lay.addWidget(splitter, 1)

        self._status = QLabel("Generate a report to see files here.")
        self._status.setStyleSheet("color: #888;")
        lay.addWidget(self._status)

    def set_files(self, files_dict):
        """Populate file tree from report generation result.

        Parameters
        ----------
        files_dict : dict
            Mapping of logical name → absolute file path.
        """
        self._files = files_dict
        self._tree.clear()

        # Group by category
        categories = {}
        for name, path in sorted(files_dict.items()):
            if name.startswith("figure_"):
                cat = "Figures"
            elif name.endswith("_csv"):
                cat = "CSV Data"
            elif name.endswith("_json"):
                cat = "JSON Data"
            elif name.startswith("latex_"):
                cat = "LaTeX Tables"
            else:
                cat = "Other"

            if cat not in categories:
                categories[cat] = QTreeWidgetItem(
                    self._tree, [cat, ""])
                categories[cat].setExpanded(True)

            display_name = name.replace("_", " ").title()
            item = QTreeWidgetItem(
                categories[cat], [display_name, str(path)])

        self._status.setText(
            f"{len(files_dict)} report files generated")

    def _on_file_selected(self, item, column):
        """Preview the selected file."""
        path = item.text(1)
        if not path or not os.path.exists(path):
            return

        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".json", ".tex", ".txt"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read(50000)  # Limit preview size
                self._preview.setPlainText(content)
            except Exception as e:
                self._preview.setPlainText(f"Error reading file: {e}")
        elif ext in (".png", ".jpg", ".svg", ".pdf"):
            self._preview.setHtml(
                f'<p>Image file: {os.path.basename(path)}</p>'
                f'<p><img src="file:///{path}" width="600"></p>')
        else:
            self._preview.setPlainText(
                f"Preview not available for {ext} files.\n"
                f"Path: {path}")

    def _open_folder(self):
        """Open the report directory in the system file explorer."""
        if not self._files:
            return
        # Get directory of first file
        first_path = next(iter(self._files.values()), "")
        if first_path:
            folder = os.path.dirname(first_path)
            if os.path.isdir(folder):
                if sys.platform == "win32":
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    subprocess.run(["open", folder])
                else:
                    subprocess.run(["xdg-open", folder])

    def _refresh(self):
        """Re-populate from stored files."""
        if self._files:
            self.set_files(self._files)

    # Called by strip_window.update_research_results()
    def on_report_complete(self, result):
        """Handle report generation completion."""
        files = result.get("files", {})
        if files:
            self.set_files(files)
