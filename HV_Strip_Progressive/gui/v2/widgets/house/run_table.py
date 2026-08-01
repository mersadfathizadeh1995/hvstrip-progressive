"""``RunTable`` — the batch progress table for the Progress dock.

The agreed batch UX: one row per profile with live **Status · Current step ·
f₀ so far** columns, driven purely by the coalesced progress frames the api
streams (``{"type": "profile", index, total, profile}`` marks a new row
active; ``phase``/``log`` frames narrate into the active row's status; the
op envelope finalises every row).  GUI-only; knows nothing of the session.
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

_COLS = ["Profile", "Status", "Current step", "f₀ (Hz)"]


class RunTable(QTableWidget):
    """Live per-profile batch progress (frame-driven)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, len(_COLS), parent)
        self.setHorizontalHeaderLabels(_COLS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self._rows: Dict[str, int] = {}
        self._active: Optional[str] = None

    # ------------------------------------------------------------------
    def begin_run(self) -> None:
        self.setRowCount(0)
        self._rows.clear()
        self._active = None

    def on_frame(self, frame: dict) -> None:
        kind = frame.get("type")
        if kind == "profile":
            name = str(frame.get("profile", ""))
            self._active = name
            row = self._ensure_row(name)
            self._set(row, 1, "running")
            total = frame.get("total")
            if total:
                self._set(row, 2, f"{frame.get('index', '?')}/{total} queued")
        elif kind == "phase" and self._active is not None:
            row = self._rows.get(self._active)
            if row is not None:
                self._set(row, 2,
                          f"[{frame.get('index')}/{frame.get('total')}] "
                          f"{frame.get('label', '')}")
        elif kind == "step" and self._active is not None:
            row = self._rows.get(self._active)
            if row is not None and frame.get("f0") is not None:
                self._set(row, 3, f"{float(frame['f0']):.3f}")

    def finalize(self, envelope: dict) -> None:
        """Mark every row's final state from the batch envelope."""
        results = (envelope or {}).get("results") or []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("profile_name", ""))
            row = self._ensure_row(name)
            ok = bool(entry.get("success"))
            self._set(row, 1, "done" if ok else "FAILED")
            strip = entry.get("strip_result") or {}
            steps = strip.get("steps") or []
            if steps and steps[0].get("peak_frequency"):
                self._set(row, 3, f"{steps[0]['peak_frequency']:.3f}")
        if self._active and self._active in self._rows and not results:
            self._set(self._rows[self._active], 1,
                      "done" if envelope.get("success") else "FAILED")
        self._active = None

    # ------------------------------------------------------------------
    def _ensure_row(self, name: str) -> int:
        if name in self._rows:
            return self._rows[name]
        row = self.rowCount()
        self.insertRow(row)
        self._set(row, 0, name)
        self._set(row, 1, "queued")
        self._rows[name] = row
        return row

    def _set(self, row: int, col: int, text: str) -> None:
        item = self.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            if col > 0:
                item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, col, item)
        item.setText(text)


__all__ = ["RunTable"]
