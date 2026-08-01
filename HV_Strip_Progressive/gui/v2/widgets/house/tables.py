"""Table helpers — content-aware column sizing + numeric cells.

Copied (copy-not-import) from the bedrock reference.  Keeps Qt tables looking
intelligent (ui-tables rule L6): no column hogs the width, numbers are
right-aligned, long text is elided, columns size to their content within sane
bounds.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)


def numeric_item(value, fmt: str = "{:.2f}", dash: str = "—") -> QTableWidgetItem:
    """A right-aligned cell; ``None`` → an em-dash."""
    if value is None:
        item = QTableWidgetItem(dash)
    else:
        try:
            item = QTableWidgetItem(fmt.format(float(value)))
        except (TypeError, ValueError):
            item = QTableWidgetItem(str(value))
        item.setData(Qt.UserRole, float(value) if value is not None else None)
    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def text_item(value, elide_to: int = 0) -> QTableWidgetItem:
    """A left-aligned, non-editable text cell; full value always in the tooltip."""
    s = "" if value is None else str(value)
    shown = s
    if elide_to and len(s) > elide_to:
        shown = "…" + s[-(elide_to - 1):]
    item = QTableWidgetItem(shown)
    if s:
        item.setToolTip(s)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


def configure_table(table: QTableWidget) -> None:
    """Quiet, scannable defaults (ui-tables TB6)."""
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setTextElideMode(Qt.ElideRight)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(24)
    table.setSortingEnabled(True)
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setHighlightSections(False)


def smart_columns(
    table: QTableWidget,
    *,
    min_w: int = 46,
    max_w: int = 280,
    pad: int = 24,
    stretch: Optional[int] = None,
) -> None:
    """Size each column to its content, clamped to ``[min_w, max_w]``."""
    was_sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    table.resizeColumnsToContents()
    fm = table.fontMetrics()
    header = table.horizontalHeader()
    for c in range(table.columnCount()):
        head_item = table.horizontalHeaderItem(c)
        head_w = fm.horizontalAdvance(head_item.text()) + pad if head_item else 0
        w = max(table.columnWidth(c), head_w)
        w = max(min_w, min(max_w, w))
        header.setSectionResizeMode(c, QHeaderView.Interactive)
        table.setColumnWidth(c, w)
    if stretch is not None:
        header.setSectionResizeMode(stretch, QHeaderView.Stretch)
    table.setSortingEnabled(was_sorting)


def fill_table(
    table: QTableWidget,
    headers: Sequence[str],
    rows: Sequence[Sequence[QTableWidgetItem]],
    *,
    stretch: Optional[int] = None,
) -> None:
    """Reset a table to *headers* + *rows* of pre-built items, then size."""
    was_sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    table.clear()
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, item in enumerate(row):
            table.setItem(r, c, item)
    table.setSortingEnabled(was_sorting)
    smart_columns(table, stretch=stretch)


__all__ = [
    "configure_table",
    "fill_table",
    "numeric_item",
    "smart_columns",
    "text_item",
]
