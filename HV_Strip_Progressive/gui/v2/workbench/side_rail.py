"""``CollapsibleSideRail`` — a side pane folded by a small top-corner arrow.

Copied (copy-not-import) from the bedrock reference.  A small, fixed
``autoRaise`` arrow pinned to the rail's **top inner corner** collapses the
content to a narrow sliver (the gui_v2 Process-workspace look — not a
full-height strip).  ``side="left"`` puts the arrow on the right edge;
``side="right"`` mirrors it.  Expanded → drag-resizable inside a
``QSplitter``; collapsed → pinned to the sliver.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

_QWIDGETSIZE_MAX = 16777215  # Qt's QWIDGETSIZE_MAX


class CollapsibleSideRail(QWidget):
    """A left/right rail collapsed by a small arrow at its top inner corner."""

    collapsed_changed = Signal(bool)  # True = collapsed

    _CHEVRON_WIDTH = 18
    _ARROW_HEIGHT = 24

    def __init__(
        self,
        content: QWidget,
        *,
        side: str = "left",
        expanded_width: int = 320,
        sliver_width: int = 18,
        collapsed: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self._side = side
        self._expanded_width = int(expanded_width)
        self._sliver_width = int(sliver_width)
        self._collapsed = bool(collapsed)
        self._content = content

        self._chevron = QToolButton(self)
        self._chevron.setAutoRaise(True)
        self._chevron.setFixedSize(self._CHEVRON_WIDTH, self._ARROW_HEIGHT)
        self._chevron.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._chevron.setCursor(Qt.PointingHandCursor)
        self._chevron.setProperty("role", "railChevron")
        self._chevron.clicked.connect(self.toggle)

        self._strip = QWidget(self)
        self._strip.setFixedWidth(self._CHEVRON_WIDTH)
        self._strip.setProperty("role", "railStrip")
        strip_layout = QVBoxLayout(self._strip)
        strip_layout.setContentsMargins(0, 4, 0, 0)
        strip_layout.setSpacing(0)
        strip_layout.addWidget(
            self._chevron, 0, Qt.AlignHCenter | Qt.AlignTop
        )
        strip_layout.addStretch(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if side == "left":
            layout.addWidget(self._content, 1)
            layout.addWidget(self._strip, 0)
        else:
            layout.addWidget(self._strip, 0)
            layout.addWidget(self._content, 1)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._apply_state()

    # ------------------------------------------------------------------
    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self._apply_state()
        self.collapsed_changed.emit(self._collapsed)

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def content(self) -> QWidget:
        return self._content

    def expanded_width(self) -> int:
        return self._expanded_width

    def set_expanded_width(self, w: int) -> None:
        w = int(w)
        if w <= 0 or w == self._expanded_width:
            return
        self._expanded_width = w
        if not self._collapsed:
            self._apply_state()

    def current_width(self) -> int:
        return self._sliver_width if self._collapsed else self._expanded_width

    # ------------------------------------------------------------------
    def _apply_state(self) -> None:
        if self._collapsed:
            self._content.hide()
            self.setMinimumWidth(self._sliver_width)
            self.setMaximumWidth(self._sliver_width)
            tip = "Expand"
        else:
            self._content.show()
            self.setMinimumWidth(self._expanded_width)
            self.setMaximumWidth(_QWIDGETSIZE_MAX)
            tip = "Collapse"
        self._chevron.setText(self._chevron_text())
        self._chevron.setToolTip(tip)

    def _chevron_text(self) -> str:
        if self._side == "left":
            return "▶" if self._collapsed else "◀"
        return "◀" if self._collapsed else "▶"


__all__ = ["CollapsibleSideRail"]
