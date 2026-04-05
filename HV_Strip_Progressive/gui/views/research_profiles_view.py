"""ResearchProfilesView — browse generated/loaded soil profiles."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
)


class ResearchProfilesView(QWidget):
    """Canvas view showing the profile suite as a filterable table.

    Columns: Name, Category, Layers, Depth (m), Vs30 (m/s), f0 est (Hz).
    Supports filtering by geological category.
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._mw = main_window
        self._all_rows = []
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>📐 Profile Suite</b>"))
        header.addStretch()
        header.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItem("All Categories")
        self._filter_combo.currentTextChanged.connect(self._apply_filter)
        header.addWidget(self._filter_combo)
        lay.addLayout(header)

        self._count_label = QLabel("No profiles loaded")
        self._count_label.setStyleSheet("color: #888; margin: 4px;")
        lay.addWidget(self._count_label)

        # Table
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "Name", "Category", "Layers", "Depth (m)",
            "Vs30 (m/s)", "f₀ est (Hz)",
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSortingEnabled(True)
        lay.addWidget(self._table)

        # Summary stats
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #555; font-size: 11px;")
        self._stats_label.setWordWrap(True)
        lay.addWidget(self._stats_label)

    def set_profiles(self, profiles_data):
        """Set profile data from the runner's profile list.

        Parameters
        ----------
        profiles_data : list[dict]
            Each dict has: name, category, n_layers, total_depth, vs30, f0_estimate.
        """
        self._all_rows = profiles_data
        categories = sorted(set(p.get("category", "?") for p in profiles_data))
        self._filter_combo.blockSignals(True)
        self._filter_combo.clear()
        self._filter_combo.addItem("All Categories")
        self._filter_combo.addItems(categories)
        self._filter_combo.blockSignals(False)
        self._populate_table(profiles_data)

    def _populate_table(self, rows):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))
        for i, p in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(
                p.get("name", f"profile_{i}")))
            self._table.setItem(i, 1, QTableWidgetItem(
                p.get("category", "?")))
            self._table.setItem(i, 2, self._num_item(
                p.get("n_layers", 0)))
            self._table.setItem(i, 3, self._num_item(
                p.get("total_depth", 0), fmt=".1f"))
            self._table.setItem(i, 4, self._num_item(
                p.get("vs30", 0), fmt=".0f"))
            self._table.setItem(i, 5, self._num_item(
                p.get("f0_estimate", 0), fmt=".2f"))
        self._table.setSortingEnabled(True)
        self._count_label.setText(f"{len(rows)} profiles")
        self._update_stats(rows)

    def _apply_filter(self, text):
        if text == "All Categories" or not text:
            self._populate_table(self._all_rows)
        else:
            filtered = [p for p in self._all_rows
                        if p.get("category") == text]
            self._populate_table(filtered)

    def _update_stats(self, rows):
        if not rows:
            self._stats_label.setText("")
            return
        vs30_vals = [p.get("vs30", 0) for p in rows if p.get("vs30")]
        f0_vals = [p.get("f0_estimate", 0) for p in rows if p.get("f0_estimate")]
        parts = [f"{len(rows)} profiles"]
        if vs30_vals:
            parts.append(
                f"Vs30: {min(vs30_vals):.0f}–{max(vs30_vals):.0f} m/s")
        if f0_vals:
            parts.append(
                f"f₀: {min(f0_vals):.2f}–{max(f0_vals):.2f} Hz")
        cats = len(set(p.get("category", "?") for p in rows))
        parts.append(f"{cats} categories")
        self._stats_label.setText(" | ".join(parts))

    @staticmethod
    def _num_item(val, fmt=None):
        """Create a right-aligned numeric table item."""
        text = f"{val:{fmt}}" if fmt else str(val)
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        try:
            item.setData(Qt.UserRole, float(val))
        except (TypeError, ValueError):
            pass
        return item

    # Called by strip_window.update_research_results()
    def on_profiles_complete(self, result):
        """Handle profile generation/loading completion."""
        summary = result.get("summary", {})
        # Convert summary into table rows if available
        # The panel pushes full profile list separately
        n = result.get("n_profiles", 0)
        self._count_label.setText(f"{n} profiles ready")

    def on_profile_loading_complete(self, result):
        self.on_profiles_complete(result)
