"""Tool switcher — the tool-collection variant of the family stage ribbon.

Adapted (copy-not-import) from invert_hvsr's ``StageRibbon``: the same flat
buttons + ``[step]`` QSS vocabulary, but the buttons are PEER TOOLS
(Forward Model | HV Strip | Research) with **per-tool status** — no numbers,
no pipeline chain, no ordering semantics.  Status is pulled from
:meth:`AppState.status_for(tool)`; the switcher never decides state itself.
Status flips flow through the :func:`repolish` choke-point.
"""

from __future__ import annotations

from typing import Dict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from HV_Strip_Progressive.gui.v2.state.tool import StripTool, ToolStatus
from HV_Strip_Progressive.gui.v2.widgets.house._qss import repolish

_STATUS_TO_STEP = {
    ToolStatus.IDLE: "ready",
    ToolStatus.RUNNING: "running",
    ToolStatus.DONE: "done",
    ToolStatus.ERROR: "error",
}


class _ToolButton(QPushButton):
    """Tool pill: label + subtitle, left-border accent (no index)."""

    def __init__(self, tool: StripTool, parent=None) -> None:
        super().__init__(parent)
        self._tool = tool
        self._status = ToolStatus.IDLE
        self._active = False

        self.setCheckable(False)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumWidth(150)
        self.setMinimumHeight(46)
        self.setText(f"{tool.label}\n{tool.subtitle}")
        self._refresh_style()

    @property
    def tool(self) -> StripTool:
        return self._tool

    def set_active(self, active: bool) -> None:
        self._active = active
        self._refresh_style()

    def set_status(self, status: ToolStatus) -> None:
        self._status = status
        self.setEnabled(True)   # tools are always reachable
        self._refresh_style()

    def _refresh_style(self) -> None:
        step = "active" if self._active else _STATUS_TO_STEP[self._status]
        repolish(self, "step", step)


class ToolSwitcher(QFrame):
    """Horizontal peer-tool selector + project label; ``tool_activated``."""

    tool_activated = Signal(object)  # StripTool

    def __init__(self, app_state, parent=None) -> None:
        super().__init__(parent)
        self._app_state = app_state
        self._active = StripTool.DATA

        self.setObjectName("ToolSwitcher")
        self.setProperty("role", "header")
        # WA_StyledBackground + autoFill: Windows dark mode otherwise bleeds
        # through the QSS fill.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setFixedHeight(56)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(2)

        self._buttons: Dict[StripTool, _ToolButton] = {}
        for tool in StripTool:
            btn = _ToolButton(tool, self)
            btn.clicked.connect(lambda _=False, t=tool: self._on_clicked(t))
            layout.addWidget(btn)
            self._buttons[tool] = btn

        layout.addStretch(1)

        proj_box = QVBoxLayout()
        proj_box.setContentsMargins(0, 0, 0, 0)
        proj_box.setSpacing(0)
        cap = QLabel("PROJECT")
        cap.setProperty("role", "caption")
        cap.setAlignment(Qt.AlignRight)
        proj_box.addWidget(cap)
        self._proj_label = QLabel("")
        self._proj_label.setAlignment(Qt.AlignRight)
        self._proj_label.setProperty("role", "title")
        proj_box.addWidget(self._proj_label)
        layout.addLayout(proj_box)

        for signal in (
            app_state.session_opened,
            app_state.profiles_changed,
            app_state.forward_changed,
            app_state.strip_changed,
            app_state.research_changed,
        ):
            signal.connect(self.refresh_statuses)
        app_state.op_started.connect(lambda _n: self.refresh_statuses())
        app_state.op_finished.connect(lambda _n, _e: self.refresh_statuses())

        self.refresh_statuses()
        self.set_active(StripTool.DATA)

    # ------------------------------------------------------------------
    def set_active(self, tool: StripTool) -> None:
        self._active = tool
        for t, btn in self._buttons.items():
            btn.set_active(t is tool)

    @property
    def active_tool(self) -> StripTool:
        return self._active

    def button(self, tool: StripTool) -> _ToolButton:
        return self._buttons[tool]

    def set_project_label(self, text: str) -> None:
        self._proj_label.setText(text or "")

    # ------------------------------------------------------------------
    def refresh_statuses(self) -> None:
        for tool, btn in self._buttons.items():
            btn.set_status(self._app_state.status_for(tool))
            btn.set_active(tool is self._active)

    def _on_clicked(self, tool: StripTool) -> None:
        self.set_active(tool)
        self.tool_activated.emit(tool)


__all__ = ["ToolSwitcher"]
