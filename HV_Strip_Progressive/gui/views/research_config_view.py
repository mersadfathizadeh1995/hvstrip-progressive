"""ResearchConfigView — configuration summary and validation display."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QMessageBox,
)


class ResearchConfigView(QWidget):
    """Canvas view showing the current research study configuration.

    Displays a hierarchical tree of all config sections with their values,
    and a validation button that checks config integrity.
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        header.addWidget(QLabel(
            "<b>📋 Study Configuration</b>"))
        header.addStretch()

        self._btn_validate = QPushButton("✔ Validate")
        self._btn_validate.clicked.connect(self._validate)
        header.addWidget(self._btn_validate)

        self._btn_refresh = QPushButton("🔄 Refresh")
        self._btn_refresh.clicked.connect(self._refresh_from_panel)
        header.addWidget(self._btn_refresh)
        lay.addLayout(header)

        self._status = QLabel("Edit configuration in the left panel.")
        self._status.setStyleSheet("color: #888; margin: 4px;")
        lay.addWidget(self._status)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Parameter", "Value"])
        self._tree.setColumnWidth(0, 250)
        self._tree.setAlternatingRowColors(True)
        lay.addWidget(self._tree)

    def set_config(self, config_dict):
        """Populate tree from a ComparisonStudyConfig dict."""
        self._tree.clear()
        self._populate_tree(self._tree.invisibleRootItem(), config_dict)
        self._tree.expandAll()
        self._status.setText("Configuration loaded.")

    def _populate_tree(self, parent, data, prefix=""):
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, (dict, list)):
                    item = QTreeWidgetItem(parent, [str(key), ""])
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self._populate_tree(item, val, f"{prefix}{key}.")
                else:
                    item = QTreeWidgetItem(parent, [str(key), str(val)])
        elif isinstance(data, list):
            for i, val in enumerate(data):
                if isinstance(val, (dict, list)):
                    item = QTreeWidgetItem(parent, [f"[{i}]", ""])
                    self._populate_tree(item, val)
                else:
                    QTreeWidgetItem(parent, [f"[{i}]", str(val)])

    def _refresh_from_panel(self):
        """Pull config from the research panel and display it."""
        if not self._mw:
            return
        panel = self._mw.get_panel()
        if panel and hasattr(panel, '_build_config'):
            try:
                cfg = panel._build_config()
                self.set_config(cfg.to_dict())
            except Exception as e:
                self._status.setText(f"Error: {e}")

    def _validate(self):
        """Validate config and show result."""
        if not self._mw:
            return
        panel = self._mw.get_panel()
        if not panel or not hasattr(panel, '_build_config'):
            return
        try:
            cfg = panel._build_config()
            errors = []
            if not cfg.engines.engines:
                errors.append("No engines selected")
            if cfg.engines.fmin >= cfg.engines.fmax:
                errors.append("fmin must be less than fmax")
            if not cfg.profiles.scenarios and cfg.profiles.n_random == 0:
                errors.append("No profiles configured")
            if not cfg.output.output_dir:
                errors.append("Output directory not set")

            if errors:
                self._status.setText(
                    f"❌ Validation failed: {len(errors)} issue(s)")
                self._status.setStyleSheet("color: red; margin: 4px;")
                QMessageBox.warning(
                    self, "Validation", "\n".join(f"• {e}" for e in errors))
            else:
                self._status.setText("✅ Configuration is valid")
                self._status.setStyleSheet("color: green; margin: 4px;")
        except Exception as e:
            self._status.setText(f"Error: {e}")

    # Called by strip_window.update_research_results()
    def on_profiles_complete(self, result):
        self._refresh_from_panel()
