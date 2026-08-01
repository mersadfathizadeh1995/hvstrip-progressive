"""``SubBreadcrumb`` — a compact sub-stage tab bar (``role="subTab"`` underline).

Ported (copy-not-import) from bedrock/gui_v2's sub-breadcrumb, trimmed to a
**tabs-only** bar sized for the narrow left dock (Back / Next live in the owning
panel's bottom nav bar).  Emits :pysig:`tab_clicked(int)`; supports per-tab
gating via :meth:`set_tab_enabled`.  GUI-only.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QToolButton, QWidget


class _SubTab(QToolButton):
    """Underline-style sub-phase tab (styled via ``role="subTab"``)."""

    def __init__(
        self,
        index: int,
        name: str,
        parent: Optional[QWidget] = None,
        *,
        numbered: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setAutoRaise(True)
        label = f"{index}.  {name}" if numbered else name
        self.setText(label.replace("&", "&&"))   # escape mnemonic
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFixedHeight(28)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("role", "subTab")


class SubBreadcrumb(QFrame):
    """A row of sub-stage tabs; the active tab gets the violet underline.

    ``numbered=False`` drops the "N." prefix — use it when many short tabs
    must fit the narrow left dock (e.g. the 5 Inversion sub-stages).
    """

    tab_clicked = Signal(int)

    def __init__(
        self,
        phase_names: List[str],
        parent: Optional[QWidget] = None,
        *,
        numbered: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setProperty("role", "subBar")
        self.setAttribute(Qt.WA_StyledBackground, True)   # dark-mode role frame
        self.setAutoFillBackground(True)
        self.setFixedHeight(34)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(2)

        self._tabs: List[_SubTab] = []
        for i, name in enumerate(phase_names):
            tab = _SubTab(i + 1, name, self, numbered=numbered)
            tab.clicked.connect(lambda _c=False, idx=i: self.tab_clicked.emit(idx))
            lay.addWidget(tab)
            self._tabs.append(tab)
        lay.addStretch(1)

        self._active = 0
        self.set_active(0)

    @property
    def active_index(self) -> int:
        return self._active

    @property
    def n_phases(self) -> int:
        return len(self._tabs)

    def set_active(self, index: int) -> None:
        self._active = index
        for i, tab in enumerate(self._tabs):
            tab.setChecked(i == index)

    def set_tab_enabled(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self._tabs):
            self._tabs[index].setEnabled(bool(enabled))


__all__ = ["SubBreadcrumb"]
