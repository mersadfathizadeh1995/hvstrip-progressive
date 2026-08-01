"""``ColorSwatchButton`` — a compact colour swatch → ``QColorDialog``.

Copied (copy-not-import) from ``hv_studio``'s ``color_button.py`` and converted
to **direct PySide6** (this package bans ``qt_compat``; the AST layering guard
enforces PySide6-only).  Clicking opens a colour dialog; the chosen colour is
stored as a hex string and emitted via :pysig:`color_changed`.  Painted in
``paintEvent`` (not a stylesheet) so the fill is always visible in compact form
layouts.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QColorDialog, QPushButton, QWidget


class ColorSwatchButton(QPushButton):
    """Square button rendering its current colour as a fill swatch."""

    color_changed = Signal(str)

    def __init__(
        self,
        color: str = "",
        parent: Optional[QWidget] = None,
        *,
        size: int = 22,
    ) -> None:
        super().__init__(parent)
        self._color: str = (color or "").strip()
        self.setFixedSize(QSize(size, size))
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setText("")
        self.setStyleSheet("")           # paintEvent owns the visual
        self._refresh_tip()
        self.clicked.connect(self._open_dialog)

    # ------------------------------------------------------------------
    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        new = (color or "").strip()
        if new == self._color:
            return
        self._color = new
        self._refresh_tip()
        self.update()

    # ------------------------------------------------------------------
    def _open_dialog(self) -> None:
        initial = QColor(self._color) if self._color else QColor("white")
        if not initial.isValid():
            initial = QColor("white")
        dlg = QColorDialog(initial, self.parent())
        dlg.setOption(QColorDialog.ShowAlphaChannel, False)
        if dlg.exec():
            col = dlg.selectedColor()
            if col.isValid():
                self._color = col.name()
                self._refresh_tip()
                self.update()
                self.color_changed.emit(self._color)

    def _refresh_tip(self) -> None:
        self.setToolTip(self._color or "(pick a colour)")

    # ------------------------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            fill = QColor(self._color) if self._color else QColor("#FFFFFF")
            if not fill.isValid():
                fill = QColor("white")
            painter.setBrush(QBrush(fill))
            border = QPen(QColor("#555"))
            border.setWidth(1)
            painter.setPen(border)
            painter.drawRoundedRect(rect, 4, 4)
            if self.underMouse():
                hover = QPen(QColor("#1976D2"))
                hover.setWidth(1)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(hover)
                painter.drawRoundedRect(rect, 4, 4)
        finally:
            painter.end()


__all__ = ["ColorSwatchButton"]
