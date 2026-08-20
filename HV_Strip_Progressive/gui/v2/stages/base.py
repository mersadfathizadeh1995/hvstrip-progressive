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


def set_unfocused(widget, value) -> None:
    """Push *value* into an editable widget ONLY when the user is not
    mid-edit in it (spec 002 FR-11 — a refresh must never reformat the
    field being typed in).  Dispatches on the widget's setter."""
    if widget.hasFocus():
        return
    if hasattr(widget, "setValue"):
        widget.setValue(value)
    elif hasattr(widget, "setChecked"):
        widget.setChecked(bool(value))
    elif hasattr(widget, "setCurrentText"):
        widget.setCurrentText(value)
    elif hasattr(widget, "setText"):
        widget.setText(value)


class PhasePanel(QWidget):
    """Base for an interactive phase panel (talks ONLY to AppState).

    Subclasses implement :meth:`build` (static layout, once) and
    :meth:`refresh` (update dynamic widgets from the session).  The
    constructor builds then refreshes; the panel self-subscribes to the
    session/result signals, and to ``config_changed`` **routed by
    section** (spec 002 FR-11): a panel declares the config sections it
    displays in :attr:`sections` and only refreshes for those.
    """

    #: Config sections this panel displays.  ``None`` = refresh on ANY
    #: section (legacy behaviour — avoid); ``()`` = config changes never
    #: refresh this panel; otherwise the exact section names.
    sections: Optional[tuple] = None

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
        app_state.config_changed.connect(self._on_config_changed)
        self.refresh()

    @property
    def app_state(self) -> AppState:
        return self._app_state

    def _on_config_changed(self, section: str) -> None:
        if (self.sections is None or section == "*"
                or section in self.sections):
            self.refresh()

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
    "set_unfocused",
    "status_line",
]
