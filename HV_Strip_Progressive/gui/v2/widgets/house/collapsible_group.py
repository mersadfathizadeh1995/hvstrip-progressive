"""``CollapsibleGroup`` — a borderless folding group (gui_v2 look).

Copied (copy-not-import) from the bedrock reference into HV Invert's own
house-widget set.  A **borderless** header toggle (arrow + bold text) over a
thin ``HLine`` separator and a tight content area; optionally a small header
checkbox (``header_checkable``) for an "is this on?" toggle that does **not**
expand/collapse the body.

NOTE: this is the *new* house widget.  The legacy ``gui/widgets/collapsible_group.py``
(``CollapsibleGroupBox``) is left in place until Stage 06 retires it — this one
lives under ``gui/widgets/house/`` so the two never clobber.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleGroup(QWidget):
    """A titled, borderless, collapsible group. Add to :attr:`content_layout`."""

    toggled = Signal(bool)               # True = expanded
    header_checked_changed = Signal(bool)

    def __init__(
        self,
        title: str = "",
        collapsed: bool = False,
        parent: Optional[QWidget] = None,
        *,
        header_checkable: bool = False,
        header_checked: bool = True,
    ) -> None:
        super().__init__(parent)
        self._header_checkable = bool(header_checkable)
        self._header_cb: Optional[QCheckBox] = None

        self._header = QToolButton(self)
        self._header.setCheckable(True)
        self._header.setChecked(not collapsed)
        self._header.setText(str(title).replace("&", "&&"))
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(
            Qt.DownArrow if not collapsed else Qt.RightArrow
        )
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setProperty("role", "groupHeader")
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.toggled.connect(self._on_toggled)

        # Optional header checkbox to the left of the toggle.
        if self._header_checkable:
            header_widget: QWidget = QWidget(self)
            hl = QHBoxLayout(header_widget)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            self._header_cb = QCheckBox(header_widget)
            self._header_cb.setChecked(bool(header_checked))
            self._header_cb.toggled.connect(self.header_checked_changed)
            hl.addWidget(self._header_cb)
            hl.addWidget(self._header, 1)
        else:
            header_widget = self._header

        self._line = QFrame(self)
        self._line.setProperty("role", "hline")
        self._line.setFrameShape(QFrame.HLine)

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(10, 4, 4, 8)
        self._body_layout.setSpacing(4)
        self._body.setVisible(not collapsed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(header_widget)
        outer.addWidget(self._line)
        outer.addWidget(self._body)

    # ------------------------------------------------------------------
    @property
    def content_layout(self) -> QVBoxLayout:
        """The body's layout — add your widgets/sub-layouts here."""
        return self._body_layout

    def add_widget(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._body_layout.addLayout(layout)

    # ------------------------------------------------------------------
    def _on_toggled(self, expanded: bool) -> None:
        self._header.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        self._body.setVisible(expanded)
        self._line.setVisible(expanded)
        self.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Preferred if expanded else QSizePolicy.Fixed,
        )
        self.toggled.emit(expanded)

    def set_collapsed(self, collapsed: bool) -> None:
        self._header.setChecked(not collapsed)

    def is_collapsed(self) -> bool:
        return not self._header.isChecked()

    def toggle(self) -> None:
        self._header.toggle()

    def set_title(self, title: str) -> None:
        self._header.setText(str(title).replace("&", "&&"))

    def title(self) -> str:
        return self._header.text().replace("&&", "&")

    # ------------------------------------------------------------------
    # Optional header-checkbox API
    # ------------------------------------------------------------------
    def is_header_checkable(self) -> bool:
        return self._header_checkable

    def set_header_checked(self, checked: bool) -> None:
        """Set the header checkbox (no signal — for syncing from state)."""
        if self._header_cb is None:
            return
        self._header_cb.blockSignals(True)
        try:
            self._header_cb.setChecked(bool(checked))
        finally:
            self._header_cb.blockSignals(False)

    def is_header_checked(self) -> bool:
        return self._header_cb.isChecked() if self._header_cb else True


__all__ = ["CollapsibleGroup"]
