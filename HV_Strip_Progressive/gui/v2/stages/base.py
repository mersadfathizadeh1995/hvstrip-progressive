"""Panel building blocks for the HV Strip workbench left dock (Stage 05).

Copied (copy-not-import) from the bedrock reference.  :class:`PhasePanel` is an
interactive controls panel that talks **only** to :class:`AppState` and
refreshes in place on its change signals; :func:`card` is a collapsible
settings group (the house :class:`CollapsibleGroup`); :func:`clear_layout` and
:func:`status_line` are the small shared helpers.

The five real per-stage panels (Data / Bounds / Inversion / Results / Phase 3)
subclass :class:`PhasePanel` in the sibling modules (T029).
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from HV_Strip_Progressive.gui.v2.state import AppState
from HV_Strip_Progressive.gui.v2.widgets.house.collapsible_group import CollapsibleGroup


def status_line(text: str, role: str = "muted", parent=None) -> QLabel:
    """One info line (selectable, role-styled)."""
    label = QLabel(text, parent)
    label.setProperty("role", role)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def clear_layout(layout) -> None:
    """Remove every item from *layout*, detaching widgets synchronously."""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        else:
            child = item.layout()
            if child is not None:
                clear_layout(child)


def card(title: str, parent=None, collapsed: bool = False):
    """A collapsible settings group; returns ``(group, body_layout)``."""
    group = CollapsibleGroup(title, collapsed=collapsed, parent=parent)
    return group, group.content_layout


class PhasePanel(QWidget):
    """Base for an interactive phase panel (talks ONLY to AppState).

    Subclasses implement :meth:`build` (static layout, once) and
    :meth:`refresh` (update dynamic widgets from the session).  The
    constructor builds then refreshes; the panel self-subscribes so it
    refreshes on every AppState change whether or not the shell owns it.
    """

    def __init__(self, app_state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._app_state = app_state
        self.build()
        for signal in (
            app_state.session_opened,
            app_state.profiles_changed,
            app_state.forward_changed,
            app_state.strip_changed,
            app_state.research_changed,
        ):
            signal.connect(self.refresh)
        app_state.config_changed.connect(lambda _s: self.refresh())
        self.refresh()

    @property
    def app_state(self) -> AppState:
        return self._app_state

    @property
    def session(self):
        return self._app_state.session

    def build(self) -> None:  # pragma: no cover - subclass
        raise NotImplementedError

    def refresh(self) -> None:  # default no-op
        pass

    # ------------------------------------------------------------------
    def scroll_host(self):
        """A vertical scroll area filling this panel; returns its body layout."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        body = QVBoxLayout(host)
        body.setContentsMargins(24, 18, 24, 18)
        body.setSpacing(12)
        scroll.setWidget(host)
        outer.addWidget(scroll)
        return body


__all__ = [
    "PhasePanel",
    "card",
    "clear_layout",
    "status_line",
]
